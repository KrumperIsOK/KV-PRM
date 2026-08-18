"""
JSONL-backed dataset for KV-PRM / text-PRM training.
"""

import json
import os
from typing import Any, Dict, List, Optional

import torch

from prm_dataset_utils import build_messages_graph


class MASDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: Any,
        mas_config: Dict,
        data_path: Optional[str] = None,
        data: Optional[List] = None,
        max_length: int = 1024,
        enable_thinking: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.edges = mas_config["edges"]
        self.agent_specs = mas_config["agents"]
        self.num_agents = len(self.agent_specs)
        self.enable_thinking = enable_thinking

        self.data: List = []
        if data is not None:
            self.data = data
        elif data_path and os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))

        if not self.data:
            raise ValueError(f"No training records were loaded from {data_path!r}")

    def __len__(self):
        return len(self.data) * self.num_agents

    def __getitem__(self, idx):
        row_idx = idx // self.num_agents
        agent_idx = idx % self.num_agents
        item = self.data[row_idx]

        messages = build_messages_graph(item, agent_idx, self.edges, self.agent_specs)

        try:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=self.max_length,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=self.max_length,
            )
        except Exception:
            # Fallback single-string encoding
            text = "\n".join(
                f"{m.get('role', '')}: {m.get('content', '')}" for m in messages
            )
            input_ids = self.tokenizer.encode(
                text, truncation=True, max_length=self.max_length
            )

        if len(input_ids) == 0:
            input_ids = [self.tokenizer.eos_token_id]

        label_val = item["labels"][agent_idx]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(label_val, dtype=torch.float),
        }
