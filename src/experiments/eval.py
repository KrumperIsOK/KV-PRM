import os
import sys
import argparse
import yaml
from typing import Optional
from pathlib import Path
from tqdm import tqdm
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_handler import load_hard_dataset
from mas import build_mas_from_specs
from core import TokenStats, load_generation_model, load_prm_scorer_auto
from tree_builder import TreeBuilder, TreeScorer
from tree_schema import save_tree, load_tree, SearchNode
from offline_strategies import OfflineSearcher
from answer_utils import is_correct

REPO_ROOT = Path(__file__).resolve().parents[2]


def _any_node_has_score(root: SearchNode, key: str) -> bool:
    """True if any node in the tree has ``key`` in ``scores`` (not only the first root child)."""
    stack = [root]
    while stack:
        node = stack.pop()
        if key in node.scores:
            return True
        stack.extend(node.children)
    return False


GEN_ERROR_MARKER = "Error: No generation"


def _tree_has_generation_errors(root: SearchNode) -> bool:
    """True if any non-root node's text contains the generation-failure marker."""
    stack = list(root.children)
    while stack:
        node = stack.pop()
        if isinstance(node.text, str) and GEN_ERROR_MARKER in node.text:
            return True
        stack.extend(node.children)
    return False


def _saved_tree_is_complete(fname: str) -> bool:
    """True if a saved tree file exists and does not contain generation errors.

    Files with missing nodes or generation-failure markers are treated as incomplete
    so the resume logic will regenerate them.
    """
    if not os.path.exists(fname):
        return False
    try:
        root = load_tree(fname)
    except Exception:
        return False
    if not root.children:
        return False
    return not _tree_has_generation_errors(root)


def _latest_checkpoint_dir(run_dir: str) -> Optional[str]:
    """
    Return the path to the highest-numbered checkpoint-* directory under ``run_dir``,
    or None if none exist.
    """
    root = Path(run_dir)
    if not root.is_dir():
        return None
    best: Optional[Path] = None
    best_step = -1
    for p in root.iterdir():
        if not p.is_dir() or not p.name.startswith("checkpoint-"):
            continue
        suffix = p.name[len("checkpoint-") :]
        if suffix.isdigit():
            step = int(suffix)
            if step > best_step:
                best_step = step
                best = p
    return str(best) if best is not None else None


def _resolve_prm_key(root: SearchNode, preferred: str) -> Optional[str]:
    """
    Match PRM scores when ``--prm_name`` differs slightly from stored keys (e.g. bash adds a literal '-?' suffix).
    """
    if _any_node_has_score(root, preferred):
        return preferred
    if preferred.endswith("-?"):
        alt = preferred[:-2]
        if _any_node_has_score(root, alt):
            return alt
    else:
        with_q = preferred + "-?"
        if _any_node_has_score(root, with_q):
            return with_q
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="competition_math")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mas", type=str, default="sequential")
    parser.add_argument("--sample_n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    # parser.add_argument("--tree_dir", default="trees_cache", help="Where to save/load trees")
    parser.add_argument("--width", type=int, default=4)

    parser.add_argument(
        "--build_tree",
        action="store_true",
        help="Whether to build trees (if false, only scoring/eval is done)",
    )

    # OpenAI Generation Config
    parser.add_argument(
        "--use_openai", action="store_true", help="Use OpenAI for generation"
    )
    parser.add_argument(
        "--openai_base_url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"),
    )
    parser.add_argument(
        "--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY")
    )
    parser.add_argument("--openai_model", type=str, default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--no_think",
        default=False,
        action="store_true",
        help="Disable 'thinking' mode in agent output",
    )

    # Local Generation Config (if NOT using OpenAI)
    parser.add_argument("--gen_model_id", default="Qwen/Qwen3-4B")

    # Scoring Config
    parser.add_argument(
        "--prm_model_id",
        default=None,
        help="Base model path for the PRM (may differ from --gen_model_id when using a cross-model PRM). "
        "If omitted, auto-detected from the LoRA adapter's PeftConfig.",
    )
    parser.add_argument(
        "--prm_dir",
        default=None,
        help="Path to a PRM checkpoint directory (checkpoint-N). "
        "If omitted, use --prm_run_dir / $PRM_RUN_DIR to pick the latest checkpoint-*.",
    )
    parser.add_argument(
        "--prm_run_dir",
        default=None,
        help="Training output directory containing checkpoint-* subdirs; "
        "if set and --prm_dir is omitted, the latest checkpoint is used.",
    )
    parser.add_argument(
        "--prm_name",
        default=None,
        help="Name key for this score (default: <parent_dir>_<checkpoint_dir>).",
    )

    # Logprob Config (Policy Scoring)
    parser.add_argument(
        "--score_policy_logprob",
        action="store_true",
        help="Run a scoring pass with local policy model",
    )
    parser.add_argument(
        "--score_prm", action="store_true", help="Run a scoring pass with PRM model"
    )
    parser.add_argument(
        "--prm_encoding",
        type=str,
        default=None,
        choices=["kv", "text"],
        help="PRM encoding type. If not set, auto-detected from prm_config.json in prm_dir.",
    )
    parser.add_argument(
        "--num_verify_tokens",
        type=int,
        default=None,
        help="Number of verify tokens to use at inference (test-time compute knob). "
        "If not set, read from prm_config.json (defaulting to 1 for legacy ckpts).",
    )
    parser.add_argument(
        "--override", action="store_true", help="Override existing scores with new ones"
    )

    parser.add_argument(
        "--output_dir",
        default="artifacts/eval",
        help="Directory in which to save evaluation trees",
    )

    args = parser.parse_args()

    run_dir = args.prm_run_dir or os.environ.get("PRM_RUN_DIR")
    if args.prm_dir is None and run_dir:
        latest = _latest_checkpoint_dir(run_dir)
        if latest is None:
            parser.error(f"No checkpoint-* directory found under {run_dir!r}")
        args.prm_dir = latest
        print(f"Using latest PRM checkpoint: {args.prm_dir}")
    if args.prm_name is None and args.prm_dir:
        p = Path(args.prm_dir).resolve()
        args.prm_name = f"{p.parent.name}_{p.name}"

    # 1. Load Data
    raw_ds, q_fn, gold_fn = load_hard_dataset(
        args.dataset, args.split, n=args.sample_n, seed=args.seed
    )
    cfg_path = REPO_ROOT / "configs" / f"{args.mas}.yaml"
    if not cfg_path.exists():
        parser.error(
            f"Unknown MAS configuration {args.mas!r}: {cfg_path} does not exist"
        )
    cfg = yaml.safe_load(cfg_path.read_text())

    think_flag = "" if not args.no_think else "_no_think"
    model_flag = args.gen_model_id.split("/")[-1] + think_flag
    tree_dir = str(
        Path(args.output_dir)
        / f"{args.dataset}_{args.split}_mas_{args.mas}_{model_flag}_w{args.width}"
    )
    print(f"tree_dir: {tree_dir}")
    os.makedirs(tree_dir, exist_ok=True)

    # --- PHASE 1: GENERATION ---
    # A task is considered done only if its saved tree exists AND contains no
    # generation-failure markers. Files with "Error: No generation" nodes are
    # treated as incomplete and regenerated.
    indices_to_gen = [
        i
        for i in range(len(raw_ds))
        if not _saved_tree_is_complete(
            os.path.join(
                tree_dir, f"{args.dataset}_{args.split}_{i}_w{args.width}.json"
            )
        )
    ]

    if indices_to_gen and args.build_tree:
        print(f"Generating {len(indices_to_gen)} trees...")

        pol_model = None
        pol_tok = None

        # Setup MAS
        if args.use_openai:
            print(f"Using OpenAI Service: {args.openai_model}")
            # For OpenAI, we still need a tokenizer in MAS for template formatting?
            # Usually MAS uses the provided tokenizer. We can load a generic one or the one matching the prompt format.
            from transformers import AutoTokenizer

            try:
                pol_tok = AutoTokenizer.from_pretrained(
                    args.gen_model_id, trust_remote_code=True
                )
            except Exception:
                print(
                    "Warning: Could not load local tokenizer for OpenAI MAS. Using simple fallback if needed."
                )
                pol_tok = None

            mas = build_mas_from_specs(
                model=None,
                tok=pol_tok,
                agent_specs=cfg["agents"],
                edges=cfg["edges"],
                use_openai=True,
                openai_api_key=args.openai_api_key,
                openai_base_url=args.openai_base_url,
                openai_model=args.openai_model,
                enable_thinking=not args.no_think,
            )
        else:
            print(f"Using Local Model: {args.gen_model_id}")
            pol_model, pol_tok = load_generation_model(args.gen_model_id)
            mas = build_mas_from_specs(
                pol_model,
                pol_tok,
                cfg["agents"],
                cfg["edges"],
                enable_thinking=not args.no_think,
            )

        builder = TreeBuilder(mas)

        for idx in tqdm(indices_to_gen, desc="Building Trees"):
            ex = raw_ds[idx]
            root = builder.build(q_fn(ex), width=args.width)
            if _tree_has_generation_errors(root):
                # Generation failed for at least one agent in this tree; do not
                # persist a partial/errored tree so that subsequent runs will
                # retry the task instead of treating it as done.
                print(
                    f"[warn] Skipping save for idx={idx}: tree contains "
                    f"'{GEN_ERROR_MARKER}' nodes (generation failure)."
                )
                continue
            save_tree(
                root,
                os.path.join(
                    tree_dir, f"{args.dataset}_{args.split}_{idx}_w{args.width}.json"
                ),
            )

        # Cleanup
        del pol_model
        torch.cuda.empty_cache()

    # --- PHASE 2: SCORING ---

    # 2a. Policy Logprobs (Optional)
    if args.score_policy_logprob:
        print("Enriching with Policy Logprobs (using local model)...")
        # Ensure model is loaded
        pol_model, pol_tok = load_generation_model(args.gen_model_id)
        mas = build_mas_from_specs(
            pol_model,
            pol_tok,
            cfg["agents"],
            cfg["edges"],
            enable_thinking=not args.no_think,
        )
        scorer = TreeScorer(mas)

        for idx in tqdm(range(len(raw_ds)), desc="Scoring Logprobs"):
            fname = os.path.join(
                tree_dir, f"{args.dataset}_{args.split}_{idx}_w{args.width}.json"
            )
            if not os.path.exists(fname):
                continue

            root = load_tree(fname)
            if _tree_has_generation_errors(root):
                # Stale pre-fix file with generation errors; skip scoring.
                continue
            # Skip if already scored (check first child)
            done = (
                root.children
                and "sum_logprob" in root.children[0].scores
                and "avg_logprob" in root.children[0].scores
                and "perplexity" in root.children[0].scores
            )
            if done and not args.override:
                continue

            scorer.score_tree(
                root,
                q_fn(raw_ds[idx]),
                "policy_logprob",
                score_type="policy_logprob",
                enable_thinking=not args.no_think,
            )
            save_tree(root, fname)

        del pol_model
        torch.cuda.empty_cache()

    # 2b. PRM Scoring
    if args.score_prm and args.prm_dir:
        print(f"Enriching trees with scorer: {args.prm_name} from {args.prm_dir}")
        prm_fn, prm_tok, _ = load_prm_scorer_auto(
            args.prm_dir,
            base_model_id=args.prm_model_id,  # None → auto-detected from LoRA PeftConfig
            prm_encoding=args.prm_encoding,
            num_verify_tokens=args.num_verify_tokens,
        )

        # For scoring text construction, we need a MAS with a tokenizer
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.gen_model_id, trust_remote_code=True)
        mas = build_mas_from_specs(
            model=None,
            tok=tok,
            agent_specs=cfg["agents"],
            edges=cfg["edges"],
            # use_openai=True,
            # openai_api_key=args.openai_api_key,
            # openai_base_url=args.openai_base_url,
            # openai_model=args.openai_model,
            enable_thinking=not args.no_think,
        )

        scorer = TreeScorer(mas)

        for idx in tqdm(range(len(raw_ds)), desc="Enriching PRM"):
            fname = os.path.join(
                tree_dir, f"{args.dataset}_{args.split}_{idx}_w{args.width}.json"
            )
            if not os.path.exists(fname):
                continue

            root = load_tree(fname)
            if _tree_has_generation_errors(root):
                # Stale pre-fix file with generation errors; skip scoring.
                continue
            if (
                root.children
                and args.prm_name in root.children[0].scores
                and not args.override
            ):
                continue

            scorer.score_tree(
                root,
                q_fn(raw_ds[idx]),
                args.prm_name,
                prm_fn,
                prm_tok,
                score_type="kvprm",  # Enforce kvprm logic
                enable_thinking=not args.no_think,
            )
            save_tree(root, fname)

        del prm_fn, prm_tok
        torch.cuda.empty_cache()

    # --- PHASE 3: EVALUATION ---
    conditions = []

    # 1. Logprob Strategies (if available)
    # if args.score_policy_logprob:
    # Greedy-per-step (Reference Style)
    # conditions.append({"name": "Logprob (Greedy)", "algo": "sbs", "kwargs": {"width": 1, "score_key": "policy_logprob"}})
    # conditions.append({"name": "perplexity-sbs-1-4 (Local)", "algo": "sbs", "kwargs": {"width": 1, "n_candidates": 4, "score_key": "perplexity", "accumulate_scores": False}})
    # conditions.append({"name": "perplexity-sbs-2-4 (Local)", "algo": "sbs", "kwargs": {"width": 2, "n_candidates": 4, "score_key": "perplexity", "accumulate_scores": False}})
    # conditions.append({"name": "perplexity-sbs-4-4 (Local)", "algo": "sbs", "kwargs": {"width": 4, "n_candidates": 4, "score_key": "perplexity", "accumulate_scores": False}})
    #     # Cumulative Beam Search
    # conditions.append({"name": "perplexity-sbs-1-4 (CumuSum)", "algo": "sbs", "kwargs": {"width": 1, "n_candidates": 4, "score_key": "perplexity", "accumulate_scores": True, "logprob_agg": "sum"}})
    # conditions.append({"name": "perplexity-sbs-2-4 (CumuSum)", "algo": "sbs", "kwargs": {"width": 2, "n_candidates": 4, "score_key": "perplexity", "accumulate_scores": True, "logprob_agg": "sum"}})
    # conditions.append({"name": "perplexity-sbs-4-4 (CumuSum)", "algo": "sbs", "kwargs": {"width": 4, "n_candidates": 4, "score_key": "perplexity", "accumulate_scores": True, "logprob_agg": "sum"}})

    # conditions.append({"name": "sum_logprob-sbs-1-4 (Local)", "algo": "sbs", "kwargs": {"width": 1, "n_candidates": 4, "score_key": "sum_logprob", "accumulate_scores": False}})
    conditions.append(
        {
            "name": "avg_logprob-sbs-1-4 (Local)",
            "algo": "sbs",
            "kwargs": {
                "width": 1,
                "n_candidates": 4,
                "score_key": "avg_logprob",
                "accumulate_scores": False,
            },
        }
    )
    conditions.append(
        {
            "name": "avg_logprob-sbs-2-4 (Local)",
            "algo": "sbs",
            "kwargs": {
                "width": 2,
                "n_candidates": 4,
                "score_key": "avg_logprob",
                "accumulate_scores": False,
            },
        }
    )
    conditions.append(
        {
            "name": "avg_logprob-sbs-4-4 (Local)",
            "algo": "sbs",
            "kwargs": {
                "width": 4,
                "n_candidates": 4,
                "score_key": "avg_logprob",
                "accumulate_scores": False,
            },
        }
    )

    # MCTS
    conditions.append(
        {
            "name": "MCTS-10 (avg_logprob)",
            "algo": "mcts",
            "kwargs": {"n_simulations": 10, "score_key": "avg_logprob"},
        }
    )
    conditions.append(
        {
            "name": "MCTS-20 (avg_logprob)",
            "algo": "mcts",
            "kwargs": {"n_simulations": 20, "score_key": "avg_logprob"},
        }
    )
    conditions.append(
        {
            "name": "MCTS-30 (avg_logprob)",
            "algo": "mcts",
            "kwargs": {"n_simulations": 30, "score_key": "avg_logprob"},
        }
    )
    conditions.append(
        {
            "name": "MCTS-40 (avg_logprob)",
            "algo": "mcts",
            "kwargs": {"n_simulations": 40, "score_key": "avg_logprob"},
        }
    )
    conditions.append(
        {
            "name": "MCTS-80 (avg_logprob)",
            "algo": "mcts",
            "kwargs": {"n_simulations": 80, "score_key": "avg_logprob"},
        }
    )
    conditions.append(
        {
            "name": "MCTS-160 (avg_logprob)",
            "algo": "mcts",
            "kwargs": {"n_simulations": 160, "score_key": "avg_logprob"},
        }
    )

    # 2. PRM Strategies (if available)
    if args.prm_dir:
        key = args.prm_name
        # Greedy-per-step
        # conditions.append({"name": f"{key} (Greedy)", "algo": "sbs", "kwargs": {"width": 1, "score_key": key}})
        print(f"args.prm_dir: {args.prm_dir}")
        print(f"args.prm_name: {args.prm_name}")
        conditions.append(
            {
                "name": f"{key}-sbs-1-4 (Local)",
                "algo": "sbs",
                "kwargs": {
                    "width": 1,
                    "n_candidates": 4,
                    "score_key": key,
                    "accumulate_scores": False,
                },
            }
        )
        conditions.append(
            {
                "name": f"{key}-sbs-2-4 (Local)",
                "algo": "sbs",
                "kwargs": {
                    "width": 2,
                    "n_candidates": 4,
                    "score_key": key,
                    "accumulate_scores": False,
                },
            }
        )
        conditions.append(
            {
                "name": f"{key}-sbs-4-4 (Local)",
                "algo": "sbs",
                "kwargs": {
                    "width": 4,
                    "n_candidates": 4,
                    "score_key": key,
                    "accumulate_scores": False,
                },
            }
        )
        # Cumulative Beam Search
        conditions.append(
            {
                "name": f"{key}-sbs-1-4 (CumuSum)",
                "algo": "sbs",
                "kwargs": {
                    "width": 1,
                    "n_candidates": 4,
                    "score_key": key,
                    "accumulate_scores": True,
                    "logprob_agg": "sum",
                },
            }
        )
        conditions.append(
            {
                "name": f"{key}-sbs-2-4 (CumuSum)",
                "algo": "sbs",
                "kwargs": {
                    "width": 2,
                    "n_candidates": 4,
                    "score_key": key,
                    "accumulate_scores": True,
                    "logprob_agg": "sum",
                },
            }
        )
        conditions.append(
            {
                "name": f"{key}-sbs-4-4 (CumuSum)",
                "algo": "sbs",
                "kwargs": {
                    "width": 4,
                    "n_candidates": 4,
                    "score_key": key,
                    "accumulate_scores": True,
                    "logprob_agg": "sum",
                },
            }
        )

        # MCTS
        conditions.append(
            {
                "name": f"MCTS-10 ({key})",
                "algo": "mcts",
                "kwargs": {"n_simulations": 10, "score_key": key},
            }
        )
        conditions.append(
            {
                "name": f"MCTS-20 ({key})",
                "algo": "mcts",
                "kwargs": {"n_simulations": 20, "score_key": key},
            }
        )
        conditions.append(
            {
                "name": f"MCTS-40 ({key})",
                "algo": "mcts",
                "kwargs": {"n_simulations": 40, "score_key": key},
            }
        )
        conditions.append(
            {
                "name": f"MCTS-80 ({key})",
                "algo": "mcts",
                "kwargs": {"n_simulations": 80, "score_key": key},
            }
        )
        conditions.append(
            {
                "name": f"MCTS-160 ({key})",
                "algo": "mcts",
                "kwargs": {"n_simulations": 160, "score_key": key},
            }
        )

        # Weighted Voting
        conditions.append(
            {
                "name": f"Weighted-Vote-10 ({key}, Sum)",
                "algo": "majority_vote",
                "kwargs": {"n_samples": 10, "score_key": key, "agg_method": "sum"},
            }
        )
        conditions.append(
            {
                "name": f"Weighted-Vote-20 ({key}, Sum)",
                "algo": "majority_vote",
                "kwargs": {"n_samples": 20, "score_key": key, "agg_method": "sum"},
            }
        )
        conditions.append(
            {
                "name": f"Weighted-Vote-30 ({key}, Sum)",
                "algo": "majority_vote",
                "kwargs": {"n_samples": 30, "score_key": key, "agg_method": "sum"},
            }
        )
        conditions.append(
            {
                "name": f"Weighted-Vote-40 ({key}, Sum)",
                "algo": "majority_vote",
                "kwargs": {"n_samples": 40, "score_key": key, "agg_method": "sum"},
            }
        )

        conditions.append(
            {
                "name": f"Weighted-Vote-10 ({key}, Prod)",
                "algo": "majority_vote",
                "kwargs": {
                    "n_samples": 10,
                    "score_key": key,
                    "agg_method": "product",
                    "use_kvprm_logic": True,
                },
            }
        )
        conditions.append(
            {
                "name": f"Weighted-Vote-20 ({key}, Prod)",
                "algo": "majority_vote",
                "kwargs": {
                    "n_samples": 20,
                    "score_key": key,
                    "agg_method": "product",
                    "use_kvprm_logic": True,
                },
            }
        )
        conditions.append(
            {
                "name": f"Weighted-Vote-30 ({key}, Prod)",
                "algo": "majority_vote",
                "kwargs": {
                    "n_samples": 30,
                    "score_key": key,
                    "agg_method": "product",
                    "use_kvprm_logic": True,
                },
            }
        )
        conditions.append(
            {
                "name": f"Weighted-Vote-40 ({key}, Prod)",
                "algo": "majority_vote",
                "kwargs": {
                    "n_samples": 40,
                    "score_key": key,
                    "agg_method": "product",
                    "use_kvprm_logic": True,
                },
            }
        )

    # 3. Baselines (Score Agnostic)
    conditions.append(
        {"name": "Random-1", "algo": "majority_vote", "kwargs": {"n_samples": 1}}
    )

    conditions.append(
        {
            "name": "Majority-Vote-10",
            "algo": "majority_vote",
            "kwargs": {"n_samples": 10},
        }
    )
    conditions.append(
        {
            "name": "Majority-Vote-20",
            "algo": "majority_vote",
            "kwargs": {"n_samples": 20},
        }
    )
    conditions.append(
        {
            "name": "Majority-Vote-40",
            "algo": "majority_vote",
            "kwargs": {"n_samples": 40},
        }
    )
    conditions.append(
        {
            "name": "Majority-Vote-80",
            "algo": "majority_vote",
            "kwargs": {"n_samples": 80},
        }
    )
    conditions.append(
        {
            "name": "Majority-Vote-160",
            "algo": "majority_vote",
            "kwargs": {"n_samples": 160},
        }
    )

    conditions.append(
        {"name": "Average-Accuracy", "algo": "get_all_answers", "kwargs": {}}
    )

    # Dummy MAS for replay
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.gen_model_id, trust_remote_code=True)
    eval_mas = build_mas_from_specs(
        model=None,
        tok=tok,
        agent_specs=cfg["agents"],
        edges=cfg["edges"],
        # use_openai=True,
        # openai_api_key=args.openai_api_key,
        # openai_base_url=args.openai_base_url,
        # openai_model=args.openai_model,
        enable_thinking=not args.no_think,
    )

    results = {c["name"]: {"correct": 0, "total": 0} for c in conditions}

    print("\nEvaluating Strategies...")
    for idx, ex in enumerate(tqdm(raw_ds)):
        fname = os.path.join(
            tree_dir, f"{args.dataset}_{args.split}_{idx}_w{args.width}.json"
        )
        if not os.path.exists(fname):
            continue

        root = load_tree(fname)
        if _tree_has_generation_errors(root):
            # Stale pre-fix file with generation errors; skip in evaluation.
            continue
        q = q_fn(ex)
        gold = gold_fn(ex)

        searcher = OfflineSearcher(root, eval_mas, q)

        for cond in conditions:
            # Skip if this strategy needs scores that are not present anywhere in the tree.
            s_key = cond["kwargs"].get("score_key")
            kwargs = dict(cond["kwargs"])
            if s_key:
                if args.prm_dir and s_key == args.prm_name:
                    resolved = _resolve_prm_key(root, args.prm_name)
                    if resolved is None:
                        continue
                    kwargs["score_key"] = resolved
                elif not _any_node_has_score(root, s_key):
                    continue

            algo = cond["algo"]

            # Execute Algo
            if algo == "sbs":
                pred, usage = searcher.sbs_search(**kwargs)
                if is_correct(pred, str(gold)):
                    results[cond["name"]]["correct"] += 1.0

            elif algo == "mcts":
                pred, usage = searcher.mcts_search(**kwargs)
                if is_correct(pred, str(gold)):
                    results[cond["name"]]["correct"] += 1.0

            elif algo == "majority_vote":
                pred, usage = searcher.majority_vote(**kwargs)
                if is_correct(pred, str(gold)):
                    results[cond["name"]]["correct"] += 1.0

            elif algo == "get_all_answers":
                # Average Accuracy over all leaves
                preds, usage = searcher.get_all_answers()
                if preds:
                    correct_count = sum(1 for p in preds if is_correct(p, str(gold)))
                    results[cond["name"]]["correct"] += correct_count / len(preds)

            results[cond["name"]]["total"] += 1
            if "usage" not in results[cond["name"]]:
                results[cond["name"]]["usage"] = TokenStats()
            results[cond["name"]]["usage"].add(usage)

    print("\n=== Final Results ===")
    for name, res in results.items():
        if res["total"] > 0:
            print(
                f"{name:<30} Acc: {res['correct'] / res['total']:.4f} ({int(res['correct'])}/{res['total']}). Token Usage: {res['usage']}"
            )


if __name__ == "__main__":
    main()
