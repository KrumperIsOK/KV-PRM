import uuid
import torch
from typing import Dict, Any, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from mas import MAS
from core import (
    _replay_trajectory,
    compute_logits_metrics,
    _text_token_len_tok,
    render_state_text,
    _aggregate_final,
)
from mcts import propose_agent_candidates
from tree_schema import SearchNode, NodeStats


class TreeBuilder:
    """
    Responsibility: Generate the search space (nodes and text) using the MAS (which may use OpenAI).
    Does NOT handle external scoring or policy logprobs (deferred to TreeScorer).
    Supports parallel generation via multithreading.
    """

    def __init__(
        self,
        mas: MAS,
        step_separator: str = "</step>",
    ):
        self.mas = mas
        self.sep = step_separator
        self.required_parents = {i: set(mas.parents.get(i, [])) for i in range(mas.n)}

    def build(
        self,
        question: str,
        width: int = 10,
        gen_kwargs: Dict[str, Any] = None,
        max_workers: int = 16,
    ) -> SearchNode:
        gen_kwargs = gen_kwargs or {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_new_tokens": 1024,
        }

        root = SearchNode(
            id=str(uuid.uuid4()),
            agent_idx=-1,
            text=question,
            stats=NodeStats(),
            is_terminal=False,
        )

        # Tuple: (Node, trajectory_dict)
        current_layer = [(root, {})]

        for agent_idx in range(self.mas.n):
            next_layer = []

            def expand_node(args):
                parent_node, traj_so_far = args
                local_results = []

                # 1. Prepare Inputs
                replay_state, _, _ = _replay_trajectory(self.mas, question, traj_so_far)
                inbox = replay_state

                # Calculate prompt tokens if tokenizer exists
                prompt_tokens = 0
                agent = self.mas.agents[agent_idx]
                if hasattr(agent, "tok") and agent.tok is not None:
                    try:
                        have = inbox.get(agent_idx, {})
                        ordered_parents = sorted(self.required_parents[agent_idx])
                        inputs = [have[p] for p in ordered_parents if p in have]

                        sys_prompt = getattr(
                            agent, "system_prompt", "You are a helpful assistant."
                        )
                        msgs = [
                            {"role": "system", "content": sys_prompt},
                            {
                                "role": "user",
                                "content": "\n\n".join(inputs).strip(),
                            },
                        ]

                        enc = agent.tok.apply_chat_template(
                            msgs, tokenize=True, add_generation_prompt=True
                        )
                        if isinstance(enc, list):
                            prompt_tokens = len(enc)
                        elif isinstance(enc, torch.Tensor):
                            prompt_tokens = enc.numel()
                        elif hasattr(enc, "shape"):  # numpy
                            prompt_tokens = enc.shape[-1]
                    except Exception:
                        pass

                # 2. Generate Candidates
                # propose_agent_candidates abstracts the model call (OpenAI/Local)
                try:
                    candidates_text = propose_agent_candidates(
                        self.mas,
                        agent_idx,
                        inbox,
                        self.required_parents,
                        n_candidates=width,
                        **gen_kwargs,
                    )
                except Exception as e:
                    # In high concurrency, print might interleave, but it's acceptable for debug
                    print(f"Gen Error (Agent {agent_idx}): {e}")
                    candidates_text = []

                # Dedup
                candidates_text = list(set([c[0] for c in candidates_text if c]))
                if not candidates_text:
                    candidates_text = ["Error: No generation"]

                # 3. Create Children
                for text in candidates_text:
                    stats = NodeStats()
                    stats.prompt_tokens = prompt_tokens

                    # Estimate tokens (cheap)
                    agent = self.mas.agents[agent_idx]
                    if hasattr(agent, "tok") and agent.tok is not None:
                        stats.gen_tokens = _text_token_len_tok(agent.tok, text)

                    stats.agent_runs += 1
                    is_last_agent = agent_idx == self.mas.n - 1
                    new_traj = {**traj_so_far, agent_idx: [text]}

                    child = SearchNode(
                        id=str(uuid.uuid4()),
                        agent_idx=agent_idx,
                        text=text,
                        scores={},
                        stats=stats,
                        is_terminal=is_last_agent,
                    )

                    # Safely append to parent
                    parent_node.children.append(child)
                    local_results.append((child, new_traj))

                return local_results

            # Execute Layer Parallelism
            if len(current_layer) == 1:
                # Single thread for root expansion
                next_layer = expand_node(current_layer[0])
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(expand_node, item) for item in current_layer
                    ]
                    for future in as_completed(futures):
                        try:
                            res = future.result()
                            next_layer.extend(res)
                        except Exception as e:
                            print(f"TreeBuild Worker Error: {e}")

            current_layer = next_layer
            if not current_layer:
                break

        return root


class TreeScorer:
    """
    Responsibility: Traverse an existing tree and add scores using a provided scorer function.
    Supports KVPrm logic via the `score_fn` passed in.
    """

    def __init__(self, mas: MAS, step_separator: str = "</step>"):
        self.mas = mas
        self.sep = step_separator
        self.required_parents = {i: set(mas.parents.get(i, [])) for i in range(mas.n)}

    def score_tree(
        self,
        root: SearchNode,
        question: str,
        scorer_name: str,
        score_fn: Callable[[str], float] = None,  # Optional if only doing logprobs
        tokenizer=None,
        score_type: str = "prm",  # "prm", "kvprm", "orm", "policy_logprob"
        enable_thinking: bool = True,
    ):
        """
        In-place update of tree nodes with new scores.
        """
        # DFS traversal
        stack = [(root, {})]  # Node, traj_so_far (which is parent's traj)

        pbar = tqdm(desc=f"Scoring {scorer_name}", leave=False)

        while stack:
            node, traj = stack.pop()

            # If not root, score this node
            if node.agent_idx != -1:
                # Update trajectory for current node (this node's output)
                curr_traj = {**traj, node.agent_idx: [node.text]}

                # Determine if we should score this node
                should_score = True
                if score_type == "orm" and not node.is_terminal:
                    should_score = False

                if should_score:
                    val = 0.0

                    # --- Logic Branching ---

                    if score_type == "policy_logprob":
                        # Requires MAS to have a local model loaded
                        # We use the prompt + current text
                        inbox, _, _ = _replay_trajectory(self.mas, question, traj)
                        have = inbox.get(node.agent_idx, {})
                        ordered_parents = sorted(self.required_parents[node.agent_idx])
                        inputs = [have[p] for p in ordered_parents if p in have]

                        sys_prompt = getattr(
                            self.mas.agents[node.agent_idx],
                            "system_prompt",
                            "You are a helpful assistant.",
                        )
                        msgs = [
                            {"role": "system", "content": sys_prompt},
                            {
                                "role": "user",
                                "content": "\n\n".join(inputs).strip(),
                            },
                        ]

                        # Use core utility
                        agent = self.mas.agents[node.agent_idx]
                        if getattr(agent, "model", None) is not None:
                            val, p_len, g_len = compute_logits_metrics(
                                agent, msgs, node.text, enable_thinking=enable_thinking
                            )
                            node.stats.prompt_tokens = p_len
                            node.stats.gen_tokens = g_len
                            node.stats.scorer_tokens += p_len + g_len
                        else:
                            val = -999.0  # Sentinel for missing model

                    elif score_type == "kvprm":
                        # Specific Logic: Reconstruct chat template including Assistant response
                        inbox, _, _ = _replay_trajectory(self.mas, question, traj)
                        have = inbox.get(node.agent_idx, {})
                        ordered_parents = sorted(self.required_parents[node.agent_idx])
                        inputs = [have[p] for p in ordered_parents if p in have]

                        sys_prompt = getattr(
                            self.mas.agents[node.agent_idx],
                            "system_prompt",
                            "You are a helpful assistant.",
                        )
                        msgs = [
                            {"role": "system", "content": sys_prompt},
                            {
                                "role": "user",
                                "content": "\n\n".join(inputs).strip(),
                            },
                        ]

                        # Add current assistant output for KVPRM context
                        new_msgs = msgs + [{"role": "assistant", "content": node.text}]

                        # Apply template without tokenization
                        agent = self.mas.agents[node.agent_idx]
                        s_text = agent.tok.apply_chat_template(
                            new_msgs, tokenize=False, add_generation_prompt=False
                        )

                        try:
                            # score_fn should handle s_text -> float
                            val = 0.0 if score_fn is None else float(score_fn(s_text))
                            node.stats.prm_calls += 1
                            if tokenizer:
                                node.stats.scorer_tokens += _text_token_len_tok(
                                    tokenizer, s_text
                                )
                        except Exception:
                            # print(f"Scoring error (kvprm): {e}")
                            val = 0.0

                    elif score_type == "orm":
                        inbox_f, primary_out_f, last_f = _replay_trajectory(
                            self.mas, question, curr_traj
                        )
                        text_to_score = _aggregate_final(
                            self.mas, primary_out_f, last_f
                        )
                        try:
                            val = float(score_fn(text_to_score))
                            if tokenizer:
                                node.stats.scorer_tokens += _text_token_len_tok(
                                    tokenizer, text_to_score
                                )
                        except Exception:
                            val = 0.0

                    else:  # Standard PRM (State History)
                        text_to_score = render_state_text(
                            self.mas, question, curr_traj, self.sep
                        )
                        try:
                            val = float(score_fn(text_to_score))
                            node.stats.prm_calls += 1
                            if tokenizer:
                                node.stats.scorer_tokens += _text_token_len_tok(
                                    tokenizer, text_to_score
                                )
                        except Exception:
                            val = 0.0

                    if isinstance(val, dict):
                        node.scores.update(val)
                    else:
                        node.scores[scorer_name] = val

                # Prepare children
                for child in node.children:
                    stack.append((child, curr_traj))
            else:
                # Root
                for child in node.children:
                    stack.append((child, {}))

            pbar.update(1)
        pbar.close()
