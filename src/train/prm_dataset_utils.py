"""Graph prompt construction for PRM training JSONL records."""

from typing import Any, Dict, List


def build_messages_graph(
    item: Dict[str, Any],
    agent_idx: int,
    edges: List[List[int]],
    agent_specs: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    parents = sorted(src for src, dst in edges if dst == agent_idx)
    input_texts = []
    for p in parents:
        if p == -1:
            input_texts.append(item.get("prompt", ""))
        else:
            comps = item.get("completions", [])
            if p < len(comps):
                input_texts.append(comps[p])
            else:
                input_texts.append("")
    user_content = "\n\n".join(input_texts).strip()
    system_prompt = agent_specs[agent_idx].get("system_prompt", "")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": item["completions"][agent_idx]},
    ]
