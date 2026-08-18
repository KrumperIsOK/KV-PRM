#!/usr/bin/env python3
# Limit each process (and any spawned workers) to 1 thread for linear algebra
# libraries. Must be set before numpy/scipy/transformers are imported.
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

"""
Offline evaluation: reads existing tree files (no building / no scoring).
Evaluates all combinations of:
  - scorers : avg_logprob  +  every PRM key found in the trees
  - SBS     : width ∈ {1, 2, 4},   n_candidates = tree_width
  - MCTS    : n_simulations ∈ {10, 40, 160}
  - Weighted majority vote : n_samples ∈ {10, 40, 160}
  - Plain majority vote (no scorer) : n_samples ∈ {10, 40, 160}

Results are printed as a table and saved to <output_dir>/eval_results_{tag}.csv

For each scored baseline we also report the scorer's total input sequence
length (avg and max across trees/questions). L per scored node is the
token length of the full conversation (input + assistant response at the
node), computed with the base-model tokenizer.
"""

import sys
import argparse
import csv
import yaml
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset_handler import load_hard_dataset
from mas import build_mas_from_specs
from tree_schema import load_tree, SearchNode
from offline_strategies import OfflineSearcher
from answer_utils import is_correct
from mcts import _replay_trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_score_keys(root: SearchNode) -> Set[str]:
    """Return every score key present anywhere in the tree."""
    keys: Set[str] = set()
    stack = [root]
    while stack:
        node = stack.pop()
        keys.update(node.scores.keys())
        stack.extend(node.children)
    return keys


_LOGPROB_KEYS = {"sum_logprob", "avg_logprob", "perplexity", "policy_logprob"}


def _discover_prm_keys(
    tree_dir: str, dataset: str, split: str, width: int, n_probe: int = 10
) -> List[str]:
    """Scan the first *n_probe* tree files and return score keys ending with '-?'."""
    prm_keys: Set[str] = set()
    pattern = f"{dataset}_{split}_*_w{width}.json"
    probed = 0
    for fpath in sorted(Path(tree_dir).glob(pattern)):
        if probed >= n_probe:
            break
        root = load_tree(str(fpath))
        all_keys = _collect_score_keys(root) - _LOGPROB_KEYS
        prm_keys.update(k for k in all_keys if k.endswith("-?"))
        probed += 1
    return sorted(prm_keys)


def _shorten_prm_key(key: str) -> str:
    """Create a human-readable short label for a (potentially very long) PRM key."""
    # Strip trailing '-?' if present
    label = key.rstrip("-?").rstrip("?").rstrip("-")
    # Keep only the last two path-like components (separated by '_')
    parts = label.split("_")
    if len(parts) > 4:
        label = "_".join(parts[-4:])
    return label


# ---------------------------------------------------------------------------
# Per-node L computation
# ---------------------------------------------------------------------------


def _chat_template_len(tok, msgs, enable_thinking: bool) -> int:
    """Return the tokenized length of `msgs` under the base-model chat template."""
    try:
        enc = tok.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        # Some tokenizers don't support enable_thinking kwarg.
        enc = tok.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=False,
        )
    if isinstance(enc, list):
        return len(enc)
    try:
        return int(enc.shape[-1])
    except Exception:
        return int(len(enc))


def _compute_L_for_tree(
    root: SearchNode,
    tok,
    mas,
    question: str,
    enable_thinking: bool,
) -> Dict[str, int]:
    """Walk the tree; for every non-root node, compute L = token length of the
    full conversation up to (and including) the assistant turn at that node."""
    L_map: Dict[str, int] = {}
    required_parents = {i: set(mas.parents.get(i, [])) for i in range(mas.n)}
    sys_prompts = [
        getattr(mas.agents[i], "system_prompt", "You are a helpful assistant.")
        for i in range(mas.n)
    ]

    def dfs(node: SearchNode, traj_before: Dict[int, List[str]]):
        if node.agent_idx == -1:
            # Root: descend.
            for c in node.children:
                dfs(c, traj_before)
            return

        agent_idx = node.agent_idx
        text = node.text or ""

        inbox, _, _ = _replay_trajectory(mas, question, traj_before)
        have = inbox.get(agent_idx, {})
        ordered_parents = sorted(required_parents[agent_idx])
        inputs = [have[p] for p in ordered_parents if p in have]
        msgs_full = [
            {"role": "system", "content": sys_prompts[agent_idx]},
            {"role": "user", "content": "\n\n".join(inputs).strip()},
            {"role": "assistant", "content": text},
        ]

        L_map[node.id] = _chat_template_len(tok, msgs_full, enable_thinking)

        if node.children:
            new_traj = dict(traj_before)
            new_traj[agent_idx] = [text]
            for c in node.children:
                dfs(c, new_traj)

    dfs(root, {})
    return L_map


# ---------------------------------------------------------------------------
# Condition builder
# ---------------------------------------------------------------------------


def _build_conditions(prm_keys: List[str], tree_width: int) -> List[Dict]:
    """
    Return a flat list of condition dicts.  Each dict has:
      scorer  – display label ("none" / "avg_logprob" / short PRM label)
      search  – display label ("sbs" / "mcts" / "weighted_mv" / "majority_vote")
      hyper   – short hyper-param string
      algo    – internal algo name understood by OfflineSearcher
      kwargs  – passed directly to the OfflineSearcher method
      _score_key – the actual score key to check availability (or None)
    """
    conditions: List[Dict] = []
    SBS_WIDTHS = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
    ]
    MCTS_SIMS = [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        210,
        220,
        230,
        240,
        250,
        260,
        270,
        280,
        290,
        300,
    ]
    MV_SAMPLES = [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        170,
        180,
        190,
        200,
        210,
        220,
        230,
        240,
        250,
        260,
        270,
        280,
        290,
        300,
    ]

    scored_keys = ["avg_logprob"] + prm_keys

    for raw_key in scored_keys:
        label = "avg_logprob" if raw_key == "avg_logprob" else _shorten_prm_key(raw_key)

        # SBS
        for w in SBS_WIDTHS:
            conditions.append(
                dict(
                    scorer=label,
                    search="sbs",
                    hyper=f"{w}-{tree_width}",
                    algo="sbs",
                    kwargs=dict(
                        width=w,
                        n_candidates=tree_width,
                        score_key=raw_key,
                        accumulate_scores=False,
                    ),
                    _score_key=raw_key,
                )
            )

        # MCTS
        for n in MCTS_SIMS:
            conditions.append(
                dict(
                    scorer=label,
                    search="mcts",
                    hyper=f"n{n}",
                    algo="mcts",
                    kwargs=dict(n_simulations=n, score_key=raw_key),
                    _score_key=raw_key,
                )
            )

        # Weighted majority vote
        for n in MV_SAMPLES:
            conditions.append(
                dict(
                    scorer=label,
                    search="weighted_mv",
                    hyper=f"n{n}",
                    algo="majority_vote",
                    kwargs=dict(n_samples=n, score_key=raw_key, agg_method="sum"),
                    _score_key=raw_key,
                )
            )

    # Plain majority vote (no scorer)
    for n in MV_SAMPLES:
        conditions.append(
            dict(
                scorer="none",
                search="majority_vote",
                hyper=f"n{n}",
                algo="majority_vote",
                kwargs=dict(n_samples=n),
                _score_key=None,
            )
        )

    # Random-1 baseline
    conditions.append(
        dict(
            scorer="none",
            search="majority_vote",
            hyper="n1",
            algo="majority_vote",
            kwargs=dict(n_samples=1),
            _score_key=None,
        )
    )

    # Average accuracy (all leaves)
    conditions.append(
        dict(
            scorer="none",
            search="avg_accuracy",
            hyper="all",
            algo="get_all_answers",
            kwargs={},
            _score_key=None,
        )
    )

    return conditions


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

# Globals populated by the worker initializer (one set per process).
_WORKER_TOK = None
_WORKER_MAS = None
_WORKER_ENABLE_THINKING = True


def _init_worker(gen_model_id: str, cfg_path_str: str, enable_thinking: bool):
    global _WORKER_TOK, _WORKER_MAS, _WORKER_ENABLE_THINKING
    from transformers import AutoTokenizer

    _WORKER_TOK = AutoTokenizer.from_pretrained(
        gen_model_id,
        trust_remote_code=True,
    )
    cfg = yaml.safe_load(Path(cfg_path_str).read_text())
    _WORKER_MAS = build_mas_from_specs(
        model=None,
        tok=_WORKER_TOK,
        agent_specs=cfg["agents"],
        edges=cfg["edges"],
        enable_thinking=enable_thinking,
    )
    _WORKER_ENABLE_THINKING = enable_thinking


def _run_one_example(args):
    """
    Evaluate every condition on a single tree.

    Returns a tuple:
      (idx, condition_results)
    where condition_results is a dict
      key = (scorer, search, hyper) -> dict with fields:
          correct : float       – 1.0 / fractional (for avg_accuracy)
          total   : int         – 1 if this condition was applicable else 0
          L       : int         – total scorer input length (0 if no scorer)
    """
    idx, tree_fname, question, gold, conditions = args

    root = load_tree(tree_fname)
    avail = _collect_score_keys(root)

    # Precompute per-node L once per tree.
    node_L_map = _compute_L_for_tree(
        root,
        _WORKER_TOK,
        _WORKER_MAS,
        question,
        _WORKER_ENABLE_THINKING,
    )

    searcher = OfflineSearcher(root, _WORKER_MAS, question, node_L_map=node_L_map)

    out: Dict[Tuple[str, str, str], Dict[str, float]] = {}
    for cond in conditions:
        key = (cond["scorer"], cond["search"], cond["hyper"])
        s_key = cond["_score_key"]
        if s_key and s_key not in avail:
            out[key] = {"correct": 0.0, "total": 0, "L": 0}
            continue

        algo = cond["algo"]
        kwargs = cond["kwargs"]

        searcher.reset_scorer_L()
        correct = 0.0

        if algo == "sbs":
            pred, _ = searcher.sbs_search(**kwargs)
            if is_correct(pred, str(gold)):
                correct = 1.0

        elif algo == "mcts":
            pred, _ = searcher.mcts_search(**kwargs)
            if is_correct(pred, str(gold)):
                correct = 1.0

        elif algo == "majority_vote":
            pred, _ = searcher.majority_vote(**kwargs)
            if is_correct(pred, str(gold)):
                correct = 1.0

        elif algo == "get_all_answers":
            preds, _ = searcher.get_all_answers()
            if preds:
                n_correct = sum(1 for p in preds if is_correct(p, str(gold)))
                correct = n_correct / len(preds)

        sum_L = int(searcher.scorer_L_total)
        n_scored = searcher.scorer_L_count
        mean_L = (sum_L / n_scored) if n_scored > 0 else 0.0
        max_L = int(searcher.scorer_L_max)
        out[key] = {
            "correct": correct,
            "total": 1,
            "sum_L": sum_L,
            "mean_L": mean_L,
            "max_L": max_L,
        }

    return idx, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Offline eval from pre-built and pre-scored tree files."
    )
    parser.add_argument(
        "--tree_dir",
        type=str,
        required=True,
        help="Directory containing <dataset>_<split>_<idx>_w<width>.json files",
    )
    parser.add_argument("--dataset", default="competition_math")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mas", type=str, default="hierarchical")
    parser.add_argument(
        "--width",
        type=int,
        default=4,
        help="Tree width used when the files were generated (part of filename)",
    )
    parser.add_argument("--sample_n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gen_model_id", default="Qwen/Qwen3-4B")
    parser.add_argument("--no_think", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="artifacts/tables",
        help="Directory to write the CSV results table",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel worker processes (default: 16)",
    )
    args = parser.parse_args()

    enable_thinking = not args.no_think

    # -----------------------------------------------------------------------
    # 1. Dataset (main process only — strings are picklable and passed to workers)
    # -----------------------------------------------------------------------
    print("Loading dataset…")
    raw_ds, q_fn, gold_fn = load_hard_dataset(
        args.dataset, args.split, n=args.sample_n, seed=args.seed
    )

    # -----------------------------------------------------------------------
    # 2. Discover PRM scorer keys (cheap — scans a few trees only)
    # -----------------------------------------------------------------------
    print("Scanning tree files for available scorer keys…")
    prm_keys = _discover_prm_keys(args.tree_dir, args.dataset, args.split, args.width)
    if prm_keys:
        print(f"  Found {len(prm_keys)} PRM key(s):")
        for k in prm_keys:
            print(f"    {k}")
    else:
        print("  No PRM keys found – only avg_logprob will be used as a scorer.")

    # -----------------------------------------------------------------------
    # 3. Build conditions + accumulator
    # -----------------------------------------------------------------------
    conditions = _build_conditions(prm_keys, tree_width=args.width)

    # key -> {correct, total, sum_Ls, mean_Ls, max_Ls}
    results: Dict[Tuple[str, str, str], Dict[str, Any]] = {
        (c["scorer"], c["search"], c["hyper"]): {
            "correct": 0.0,
            "total": 0,
            "sum_Ls": [],
            "mean_Ls": [],
            "max_Ls": [],
        }
        for c in conditions
    }

    # -----------------------------------------------------------------------
    # 4. Build per-example tasks
    # -----------------------------------------------------------------------
    tasks = []
    for idx, ex in enumerate(raw_ds):
        fname = os.path.join(
            args.tree_dir,
            f"{args.dataset}_{args.split}_{idx}_w{args.width}.json",
        )
        if not os.path.exists(fname):
            continue
        q = q_fn(ex)
        gold = str(gold_fn(ex))
        tasks.append((idx, fname, q, gold, conditions))

    print(
        f"\nEvaluating {len(conditions)} conditions over {len(tasks)} examples "
        f"with {args.num_workers} worker(s)…"
    )

    # -----------------------------------------------------------------------
    # 5. Dispatch in parallel
    # -----------------------------------------------------------------------
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path_str = str(repo_root / "configs" / f"{args.mas}.yaml")

    if args.num_workers <= 1:
        _init_worker(args.gen_model_id, cfg_path_str, enable_thinking)
        for task in tqdm(tasks, desc="Examples"):
            _, per_cond = _run_one_example(task)
            _merge_results(results, per_cond)
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(
                args.gen_model_id,
                cfg_path_str,
                enable_thinking,
            ),
        ) as pool:
            futures = [pool.submit(_run_one_example, task) for task in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Examples"):
                _, per_cond = fut.result()
                _merge_results(results, per_cond)

    # -----------------------------------------------------------------------
    # 6. Print table
    # -----------------------------------------------------------------------
    SCORER_W = 52
    SEARCH_W = 16
    HYPER_W = 10
    ACC_W = 10
    COUNT_W = 14
    L_W = 12

    sep = (
        "-" * SCORER_W
        + "+-"
        + "-" * SEARCH_W
        + "+-"
        + "-" * HYPER_W
        + "+-"
        + "-" * ACC_W
        + "+-"
        + "-" * COUNT_W
        + "+-"
        + "-" * L_W
        + "+-"
        + "-" * L_W
        + "+-"
        + "-" * L_W
        + "+-"
        + "-" * L_W
        + "+-"
        + "-" * L_W
        + "+-"
        + "-" * L_W
    )
    header = (
        f"{'Scorer':<{SCORER_W}} | "
        f"{'Search':<{SEARCH_W}} | "
        f"{'Hyper':<{HYPER_W}} | "
        f"{'Accuracy':<{ACC_W}} | "
        f"{'Correct/Total':<{COUNT_W}} | "
        f"{'Avg sumL':<{L_W}} | {'Max sumL':<{L_W}} | "
        f"{'Avg meanL':<{L_W}} | {'Max meanL':<{L_W}} | "
        f"{'Avg maxL':<{L_W}} | {'Max maxL':<{L_W}}"
    )

    print("\n" + "=" * len(sep))
    print("OFFLINE EVALUATION RESULTS")
    print("=" * len(sep))
    print(header)
    print(sep)

    rows = []
    for cond in conditions:
        key = (cond["scorer"], cond["search"], cond["hyper"])
        r = results[key]
        if r["total"] == 0:
            continue
        acc = r["correct"] / r["total"]
        sum_Ls = r["sum_Ls"]
        mean_Ls = r["mean_Ls"]
        max_Ls = r["max_Ls"]
        avg_sumL = (sum(sum_Ls) / len(sum_Ls)) if sum_Ls else 0.0
        max_sumL = max(sum_Ls) if sum_Ls else 0
        avg_meanL = (sum(mean_Ls) / len(mean_Ls)) if mean_Ls else 0.0
        max_meanL = max(mean_Ls) if mean_Ls else 0.0
        avg_maxL = (sum(max_Ls) / len(max_Ls)) if max_Ls else 0.0
        max_maxL = max(max_Ls) if max_Ls else 0
        scorer_disp = (
            (cond["scorer"][: SCORER_W - 1] + "…")
            if len(cond["scorer"]) > SCORER_W
            else cond["scorer"]
        )
        line = (
            f"{scorer_disp:<{SCORER_W}} | "
            f"{cond['search']:<{SEARCH_W}} | "
            f"{cond['hyper']:<{HYPER_W}} | "
            f"{acc:<{ACC_W}.4f} | "
            f"{int(r['correct'])}/{r['total']:<{COUNT_W - len(str(int(r['correct']))) - 1}} | "
            f"{avg_sumL:<{L_W}.1f} | {max_sumL:<{L_W}} | "
            f"{avg_meanL:<{L_W}.1f} | {max_meanL:<{L_W}.1f} | "
            f"{avg_maxL:<{L_W}.1f} | {max_maxL:<{L_W}}"
        )
        print(line)
        rows.append(
            [
                cond["scorer"],
                cond["search"],
                cond["hyper"],
                f"{acc:.4f}",
                int(r["correct"]),
                r["total"],
                f"{avg_sumL:.2f}",
                int(max_sumL),
                f"{avg_meanL:.2f}",
                f"{max_meanL:.2f}",
                f"{avg_maxL:.2f}",
                int(max_maxL),
            ]
        )

    print("=" * len(sep))

    # -----------------------------------------------------------------------
    # 7. Save CSV
    # -----------------------------------------------------------------------
    tree_tag = Path(args.tree_dir).name
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"eval_results_{tree_tag}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "scorer",
                "search",
                "hyper",
                "accuracy",
                "correct",
                "total",
                "avg_sumL",
                "max_sumL",
                "avg_meanL",
                "max_meanL",
                "avg_maxL",
                "max_maxL",
            ]
        )
        writer.writerows(rows)
    print(f"\nResults saved → {csv_path}")


def _merge_results(
    agg: Dict[Tuple[str, str, str], Dict[str, Any]],
    per_cond: Dict[Tuple[str, str, str], Dict[str, float]],
) -> None:
    for key, r in per_cond.items():
        if r["total"] == 0:
            continue
        slot = agg[key]
        slot["correct"] += r["correct"]
        slot["total"] += r["total"]
        slot["sum_Ls"].append(int(r["sum_L"]))
        slot["mean_Ls"].append(float(r["mean_L"]))
        slot["max_Ls"].append(int(r["max_L"]))


if __name__ == "__main__":
    main()
