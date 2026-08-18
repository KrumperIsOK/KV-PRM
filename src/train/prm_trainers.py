"""
PRM trainers: KV-transfer (ref KV + verify step) vs full-sequence (encode context+verify in one forward).

Both trainers support a configurable number of verify tokens (`num_verify_tokens`):
- During training, N verify tokens are appended; loss is averaged over all N positions
  so each position learns to be a valid score readout.
- At inference, K verify tokens are appended (K can differ from training N) and the
  score is read from the last position. K is loaded from prm_config.json by default.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import Trainer


class FullSequencePRMTrainer(Trainer):
    """
    Single forward over [padded context | N * verify_token]; LoRA applies to the full sequence.
    Loss averaged over all N verify positions.
    """

    def __init__(
        self,
        verify_token_id: int,
        positive_token_id: int,
        negative_token_id: int,
        num_verify_tokens: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.verify_token_id = verify_token_id
        self.positive_token_id = positive_token_id
        self.negative_token_id = negative_token_id
        if num_verify_tokens < 1:
            raise ValueError("num_verify_tokens must be >= 1")
        self.num_verify_tokens = num_verify_tokens

    def _forward_score_logits(self, model, input_ids, attention_mask):
        """Return score_logits of shape [B, N, 2] (neg, pos) over N verify positions."""
        device = input_ids.device
        batch_size = input_ids.shape[0]
        N = self.num_verify_tokens
        verify_input_ids = torch.full(
            (batch_size, N),
            self.verify_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        full_ids = torch.cat([input_ids, verify_input_ids], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        verify_mask = torch.ones(
            (batch_size, N), dtype=attention_mask.dtype, device=device
        )
        full_mask = torch.cat([attention_mask, verify_mask], dim=1)
        outputs = model(
            input_ids=full_ids,
            attention_mask=full_mask,
            use_cache=False,
            logits_to_keep=N,
        )
        verify_logits = outputs.logits[:, -N:, :]  # [B, N, V]
        target_token_ids = torch.tensor(
            [self.negative_token_id, self.positive_token_id], device=device
        )
        score_logits = verify_logits[:, :, target_token_ids]  # [B, N, 2]
        return score_logits, outputs

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        score_logits, outputs = self._forward_score_logits(
            model, input_ids, attention_mask
        )
        loss = _multi_pos_mse(score_logits, labels)
        # Store last-position logits for eval metric computation.
        outputs.logits = score_logits[:, -1, :]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        with torch.no_grad():
            score_logits, _ = self._forward_score_logits(
                model, input_ids, attention_mask
            )
        loss = None
        if labels is not None:
            loss = _multi_pos_mse(score_logits, labels)
        last_logits = score_logits[:, -1, :]
        if prediction_loss_only:
            return (loss, None, None)
        return (loss, last_logits, labels)


def _multi_pos_mse(score_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    score_logits: [B, N, 2] (neg, pos) -- per-position scoring logits.
    labels:       [B]       -- targets in [-1, 1].
    Returns MSE between sigmoid-prob-of-pos and mapped label, averaged over B*N.
    """
    probs = torch.softmax(score_logits, dim=-1)[..., 1]  # [B, N]
    mapped_labels = ((labels + 1) / 2).unsqueeze(-1).expand_as(probs)
    return nn.functional.mse_loss(probs, mapped_labels)


def save_prm_config(
    output_dir: str,
    prm_encoding: str,
    num_verify_tokens: int = 1,
) -> None:
    import json
    from pathlib import Path

    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / "prm_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "prm_encoding": prm_encoding,
                "num_verify_tokens": int(num_verify_tokens),
            },
            f,
            indent=2,
        )
