import math
import random
from typing import List, Tuple, Dict
from tree_schema import SearchNode, NodeStats
from core import TokenStats, _aggregate_final, _replay_trajectory
from answer_utils import extract_pred_number


def node_to_token_stats(ns: NodeStats) -> TokenStats:
    return TokenStats(
        prompt=ns.prompt_tokens,
        generated=ns.gen_tokens,
        scorer=ns.scorer_tokens,
        prm_calls=ns.prm_calls,
        agent_runs=ns.agent_runs,
    )


class OfflineSearcher:
    def __init__(
        self, root: SearchNode, mas, question: str, node_L_map: Dict[str, int] = None
    ):
        self.root = root
        self.mas = mas
        self.question = question
        # node.id -> L (full-conversation token length for this node's output)
        self.node_L_map: Dict[str, int] = node_L_map or {}
        # Accumulators for scoring cost during the current search invocation.
        # Call reset_scorer_L() before each method call and read these after:
        #   scorer_L_total  – sum of input-conversation lengths across all scoring events
        #   scorer_L_count  – number of scoring events (for computing mean L per event)
        #   scorer_L_max    – maximum input-conversation length over all scoring events
        self.scorer_L_total: int = 0
        self.scorer_L_count: int = 0
        self.scorer_L_max: int = 0

    def reset_scorer_L(self) -> None:
        self.scorer_L_total = 0
        self.scorer_L_count = 0
        self.scorer_L_max = 0

    def sbs_search(
        self,
        score_key: str = "prm",  # Key in node.scores
        width: int = 1,
        n_candidates: int = 4,
        logprob_agg: str = "sum",  # Only used if score_key points to logprob-like score
        use_kvprm_logic: bool = False,  # Used when accumulate_scores=True to treat score as probability
        accumulate_scores: bool = False,  # False = Greedy-per-step (Reference), True = Standard Beam Search
    ) -> Tuple[str, TokenStats]:
        usage = TokenStats()

        # (Node, SCORE_FOR_SORTING, TrajectoryDict)
        beam = []

        # Init with root children
        for child in self.root.children[:n_candidates]:
            s = self._get_score(child, score_key, logprob_agg)
            traj = {child.agent_idx: [child.text]}
            # For the first step, cumulative score is just the step score
            beam.append((child, s, traj))
            usage.add(node_to_token_stats(child.stats))

        # Sort
        beam.sort(key=lambda x: x[1], reverse=True)
        beam = beam[:width]

        while beam and not beam[0][0].is_terminal:
            next_beam = []
            # Gather all candidates from current beam
            for node, cum_score, traj in beam:
                if not node.children:
                    continue
                for child in node.children[:n_candidates]:
                    step_score = self._get_score(child, score_key, logprob_agg)

                    if not accumulate_scores:
                        # Greedy-per-step: Sort candidates solely by their current step score
                        new_score = step_score
                    else:
                        # Cumulative: Accumulate score based on type
                        if use_kvprm_logic or "kv" in score_key:
                            # Probability Product [0,1]
                            w = max(1e-9, step_score)
                            new_score = cum_score * w
                        elif "logprob" in score_key:
                            # Sum Logprobs
                            new_score = cum_score + step_score
                        elif "prm" in score_key:
                            # Standard PRM [-1, 1] mapped to [0, 1] Product
                            w = (step_score + 1.0) / 2.0
                            new_score = cum_score * max(1e-6, w)
                        else:
                            # Default/ORM fallback
                            new_score = step_score

                    new_traj = traj.copy()
                    new_traj[child.agent_idx] = [child.text]

                    next_beam.append((child, new_score, new_traj))
                    usage.add(node_to_token_stats(child.stats))

            if not next_beam:
                break
            # Global sort across all candidates from all beams
            next_beam.sort(key=lambda x: x[1], reverse=True)
            beam = next_beam[:width]

        if not beam:
            return "Error", usage

        best_traj = beam[0][2]
        inbox, primary, last = _replay_trajectory(self.mas, self.question, best_traj)
        return _aggregate_final(self.mas, primary, last), usage

    def mcts_search(
        self,
        n_simulations: int = 10,
        c_uct: float = 2.0,
        score_key: str = "prm",
        leaf_score_key: str = None,
        logprob_agg: str = "sum",
    ) -> Tuple[str, TokenStats]:
        usage = TokenStats()
        leaf_score_key = leaf_score_key or score_key

        class MCTSNode:
            def __init__(self, search_node: SearchNode, parent=None):
                self.s_node = search_node
                self.parent = parent
                self.visits = 0
                self.value_sum = 0.0
                self.children: List[MCTSNode] = []
                self.is_expanded = False  # "Expanded" means we have incorporated static children into MCTS tree
                self.q_init = 0.0

            @property
            def q_mean(self):
                # Value Prior: incorporating q_init as a 'virtual' first visit (PUCT-like)
                return (self.value_sum + self.q_init) / (self.visits + 1)

        root_wrapper = MCTSNode(self.root)

        def _uct(parent, child):
            # UCT calculation similar to core.py logic
            # Np = parent.visits + 1 to account for current visit or avoid log(0)
            Np = parent.visits + 1
            Nc = child.visits + 1
            expl = c_uct * math.sqrt(math.log(Np) / Nc)
            return child.q_mean + expl

        for _ in range(n_simulations):
            node = root_wrapper
            path = [node]

            # 1. Selection & Expansion & Rollout (Unified Loop)
            # We traverse until we hit a terminal state in the underlying search space.
            while not node.s_node.is_terminal:
                if not node.children:
                    # -- EXPANSION Phase --
                    # Incorporate pre-generated candidates from Search Space
                    if not node.is_expanded:
                        for child_s in node.s_node.children:
                            child_w = MCTSNode(child_s, parent=node)
                            # Use score_key as heuristic init value
                            child_w.q_init = self._get_score(
                                child_s, score_key, logprob_agg
                            )
                            node.children.append(child_w)

                            # Account for "generating" these candidates (simulated cost)
                            usage.add(node_to_token_stats(child_s.stats))

                        node.is_expanded = True

                    if not node.children:
                        # Dead end in search space (no candidates generated)
                        break

                    # Select immediately after expansion (continue the rollout)
                    node = max(node.children, key=lambda ch: _uct(path[-1], ch))
                    path.append(node)
                else:
                    # -- SELECTION Phase --
                    # Already expanded, select based on UCT
                    node = max(node.children, key=lambda ch: _uct(path[-1], ch))
                    path.append(node)

            # 2. Leaf Evaluation
            leaf_node = path[-1]
            val = self._get_score(leaf_node.s_node, leaf_score_key, logprob_agg)

            # If the leaf wasn't expanded in the loop (e.g. terminal), add its usage cost for access
            # (If it was intermediate, its cost was added during expansion)
            # To be safe and simple: usage is added during expansion.
            # If we reached a terminal node that was already expanded in a previous sim, no new cost.
            # If we reached a terminal node for the first time, it was just expanded/added above.

            # 3. Backpropagation
            for n in path:
                n.visits += 1
                n.value_sum += val

        # 4. Final Selection
        if not root_wrapper.children:
            return "", usage

        # Select robust child (most visits)
        best_child = max(root_wrapper.children, key=lambda x: x.q_mean)

        # Traverse best path greedy to leaf for final answer reconstruction
        curr = best_child
        traj = {curr.s_node.agent_idx: [curr.s_node.text]}

        while not curr.s_node.is_terminal:
            if curr.children:
                # Prefer staying within the MCTS statistics
                curr = max(curr.children, key=lambda x: x.q_mean)
            #  elif curr.s_node.children:
            #      # Fallback: if MCTS tree ends but Static tree continues (rare if simulations suffice)
            #      # Greedy on static scores
            #      best_s = max(curr.s_node.children, key=lambda x: self._get_score(x, score_key))
            #      curr = MCTSNode(best_s)
            else:
                break
            traj[curr.s_node.agent_idx] = [curr.s_node.text]

        inbox, primary, last = _replay_trajectory(self.mas, self.question, traj)
        return _aggregate_final(self.mas, primary, last), usage

    def majority_vote(
        self,
        n_samples: int = 1,
        score_key: str = None,
        agg_method: str = "leaf",  # "leaf", "sum", "product"
        logprob_agg: str = "sum",
        use_kvprm_logic: bool = False,
    ) -> Tuple[str, TokenStats]:
        """
        Randomly samples `n_samples` trajectories from the static search space.
        If score_key is provided, results are weighted by the trajectory score calculated via agg_method.
        Otherwise, performs naive majority voting (weight=1).
        """
        usage = TokenStats()
        vote_tallies = {}  # answer -> total weight

        for _ in range(n_samples):
            node = self.root
            curr_traj = {}  # agent_idx -> list of text
            path_scores = []

            # Traverse randomly from root to leaf
            while not node.is_terminal and node.children:
                # Random selection (Uniform)
                node = random.choice(node.children)

                # Account for usage of the visited node
                usage.add(node_to_token_stats(node.stats))

                # Collect score if key provided
                if score_key:
                    s = self._get_score(node, score_key, logprob_agg)
                    path_scores.append(s)

                # Build trajectory for this path
                curr_traj[node.agent_idx] = [node.text]

            # Reconstruct answer for this path
            inbox, primary, last = _replay_trajectory(
                self.mas, self.question, curr_traj
            )
            raw_ans = _aggregate_final(self.mas, primary, last)

            # Normalize answer for voting (to group equivalent answers)
            ans = str(extract_pred_number(raw_ans))

            # Calculate Weight
            weight = 1.0
            if score_key:
                if agg_method == "leaf":
                    weight = path_scores[-1] if path_scores else 0.0
                elif agg_method == "sum":
                    weight = sum(path_scores)
                elif agg_method == "product":
                    weight = 1.0
                    for s in path_scores:
                        # Normalize/Map if needed
                        if use_kvprm_logic or "prm" in score_key:
                            # Heuristic mapping [-1, 1] -> [0, 1] if not already prob
                            # If KV-PRM, likely already prob [0, 1]
                            if (
                                "prm" in score_key
                                and not use_kvprm_logic
                                and "kv" not in score_key
                            ):
                                w_step = (s + 1.0) / 2.0
                            else:
                                w_step = max(1e-9, s)
                            weight *= w_step
                        else:
                            # Raw product
                            weight *= s

            vote_tallies[ans] = vote_tallies.get(ans, 0.0) + weight

        if not vote_tallies:
            return "Error: No samples", usage

        most_voted = max(vote_tallies.items(), key=lambda x: x[1])[0]
        return most_voted, usage

    def get_all_answers(self) -> Tuple[List[str], TokenStats]:
        """
        Traverses the entire static search tree to collect all leaf answers.
        Useful for calculating average accuracy over the search space.
        """
        usage = TokenStats()
        all_answers = []

        # DFS Stack: (Node, TrajectoryDict)
        stack = [(self.root, {})]

        while stack:
            node, traj = stack.pop()

            # Account for usage (every node in the tree is visited exactly once)
            if node.agent_idx != -1:
                usage.add(node_to_token_stats(node.stats))

            # Update trajectory
            if node.agent_idx != -1:
                new_traj = traj.copy()
                new_traj[node.agent_idx] = [node.text]
            else:
                new_traj = traj

            if not node.children:
                # Leaf node - verify it's not the root itself
                if node.agent_idx != -1:
                    inbox, primary, last = _replay_trajectory(
                        self.mas, self.question, new_traj
                    )
                    ans = _aggregate_final(self.mas, primary, last)
                    all_answers.append(ans)
            else:
                for child in node.children:
                    stack.append((child, new_traj))

        return all_answers, usage

    def _get_score(self, node: SearchNode, key: str, lp_agg="sum") -> float:
        val = node.scores.get(key, 0.0)

        # Account the scorer's input length (full conversation up to & including
        # this node's response) for each scoring event.
        if key:
            node_L = int(self.node_L_map.get(node.id, 0))
            self.scorer_L_total += node_L
            self.scorer_L_count += 1
            if node_L > self.scorer_L_max:
                self.scorer_L_max = node_L

        if "logprob" in key and lp_agg == "avg_token":
            return val / max(1, node.stats.gen_tokens)

        return val
