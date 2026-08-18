from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json


@dataclass
class NodeStats:
    prompt_tokens: int = 0
    gen_tokens: int = 0
    scorer_tokens: int = 0
    prm_calls: int = 0
    agent_runs: int = 0

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        return NodeStats(**d)


@dataclass
class SearchNode:
    id: str
    agent_idx: int
    text: str  # The output text at this step

    # Flexible Dictionary for multiple scores (e.g., "policy_logprob", "prm_1", "prm_2", "orm")
    # This allows decoupling: scores can be added incrementally.
    scores: Dict[str, float] = field(default_factory=dict)

    # Metadata for reconstruction (optional)
    full_output_ids: Optional[List[int]] = None

    # Cost to generate/score THIS specific node
    stats: NodeStats = field(default_factory=NodeStats)

    # Tree structure
    children: List["SearchNode"] = field(default_factory=list)
    is_terminal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_idx": self.agent_idx,
            "text": self.text,
            "scores": self.scores,
            "stats": self.stats.to_dict(),
            "children": [c.to_dict() for c in self.children],
            "is_terminal": self.is_terminal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchNode":
        node = cls(
            id=data["id"],
            agent_idx=data["agent_idx"],
            text=data["text"],
            scores=data.get("scores", {}),
            stats=NodeStats.from_dict(data.get("stats", {})),
            is_terminal=data.get("is_terminal", False),
        )
        node.children = [cls.from_dict(c) for c in data.get("children", [])]
        return node


def save_tree(root: SearchNode, filepath: str):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(root.to_dict(), f, indent=2)


def load_tree(filepath: str) -> SearchNode:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return SearchNode.from_dict(data)
