import sys
from pathlib import Path

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))

import torch
import torch.nn as nn
import numpy as np
import random
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    EvalPrediction,
)
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Any, Tuple
import json
import os
import yaml
from contextlib import nullcontext
from sklearn.metrics import r2_score, explained_variance_score

from prm_mas_dataset import MASDataset
from prm_trainers import FullSequencePRMTrainer, save_prm_config, _multi_pos_mse

REPO_ROOT = Path(__file__).resolve().parents[2]

# ==========================================
# 0. Arguments Definition
# ==========================================


@dataclass
class ScriptArguments:
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to the JSONL training data."}
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the jsonl evaluation data. If None, no eval is performed."
        },
    )
    test_size: float = field(
        default=0.0,
        metadata={"help": "Proportion of training records reserved for evaluation."},
    )
    dataset: str = field(
        default="competition_math",
        metadata={
            "help": "Logical dataset / task name (logging only). MAS graph comes from --mas → configs/<mas>.yaml, "
            "same as --dataset for data in run_mcts.py and --mas in eval.py."
        },
    )
    mas: str = field(
        default="sequential",
        metadata={
            "help": "MAS graph: loads configs/<mas>.yaml (agents + edges), same as --mas in run_mcts / experiments/eval."
        },
    )
    verify_token: str = field(
        default="?", metadata={"help": "The string representation of the verify token."}
    )
    positive_token: str = field(
        default="+", metadata={"help": "Token representing positive class/high score."}
    )
    negative_token: str = field(
        default="-", metadata={"help": "Token representing negative class/low score."}
    )
    model_name_or_path: str = field(
        default="Qwen/Qwen3-4B",
        metadata={"help": "Hugging Face model ID or local path."},
    )
    max_length: int = field(default=8192, metadata={"help": "Maximum sequence length."})
    lora_r: int = field(default=256, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})

    # Specifics for the user's requested config style
    grad_accum: int = field(
        default=16, metadata={"help": "Gradient accumulation steps."}
    )
    log_freq: int = field(default=10, metadata={"help": "Logging frequency."})
    eval_freq: int = field(default=200, metadata={"help": "Evaluation frequency."})
    save_freq: int = field(default=200, metadata={"help": "Save frequency."})
    prm_encoding: str = field(
        default="kv",
        metadata={
            "help": "kv: ref KV + verify step; text: full-sequence forward with LoRA on all tokens."
        },
    )
    enable_thinking: bool = field(
        default=True,
        metadata={"help": "Passed to tokenizer.apply_chat_template when supported."},
    )
    num_verify_tokens: int = field(
        default=1,
        metadata={
            "help": "Number of verify tokens appended for scoring. Loss is averaged over all N positions during training."
        },
    )


# ==========================================
# 1. Custom Trainer Definition
# ==========================================


class KVTransferTrainer(Trainer):
    """
    Custom Trainer adapted for DDP/Multi-GPU:
    1. Uses ref_model (unwrapped) to generate KV Cache.
    2. Passes KV Cache to the training model.
    """

    def __init__(
        self,
        ref_model=None,
        verify_token_id=None,
        positive_token_id=None,
        negative_token_id=None,
        num_verify_tokens: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # In DDP, ref_model might just be the initial model passed in.
        # We don't manually .to(device) here; Accelerator handles it.
        self.ref_model = ref_model if ref_model is not None else self.model

        self.verify_token_id = verify_token_id
        self.positive_token_id = positive_token_id
        self.negative_token_id = negative_token_id
        if num_verify_tokens < 1:
            raise ValueError("num_verify_tokens must be >= 1")
        self.num_verify_tokens = num_verify_tokens

        if any(
            id is None for id in [verify_token_id, positive_token_id, negative_token_id]
        ):
            raise ValueError(
                "Must provide verify_token_id, positive_token_id, and negative_token_id"
            )

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")

        # DDP Fix: Use input_ids.device as the source of truth for the current rank's device
        device = input_ids.device

        # DDP Fix: Unwrap models to access underlying methods (like disable_adapter)
        # and to prevent DDP sync issues during the "reference" forward pass.
        # If self.ref_model IS self.model, we need the unwrapped version to avoid
        # recursive DDP calls or gradient issues during the no_grad pass.

        # Note: self.accelerator is available in Trainer
        unwrapped_model = self.accelerator.unwrap_model(model)

        # Determine if we are sharing the same object
        # We compare the unwrap of ref with unwrap of model to be safe
        unwrapped_ref = self.accelerator.unwrap_model(self.ref_model)
        is_sharing_model = unwrapped_ref is unwrapped_model

        # 1. Reference Model Forward (Get KV Cache)
        # We use the unwrapped model for the reference pass to avoid DDP hooks firing
        # during what is effectively an inference step.
        adapter_ctx = (
            unwrapped_ref.disable_adapter()
            if (is_sharing_model and hasattr(unwrapped_ref, "disable_adapter"))
            else nullcontext()
        )

        previous_training_state = unwrapped_ref.training
        if is_sharing_model:
            unwrapped_ref.eval()

        with torch.no_grad():
            with adapter_ctx:
                # We use unwrapped_ref here.
                # Inputs are already on the correct 'device' for this rank.
                ref_outputs = unwrapped_ref(
                    input_ids=input_ids,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                past_key_values = ref_outputs.past_key_values

        if is_sharing_model and previous_training_state:
            unwrapped_ref.train()

        # 2. Construct Train Model Input -- N verify tokens
        batch_size = input_ids.shape[0]
        N = self.num_verify_tokens
        verify_input_ids = torch.full(
            (batch_size, N), self.verify_token_id, dtype=torch.long, device=device
        )

        ref_attention_mask = inputs.get("attention_mask")
        if ref_attention_mask is None:
            ref_attention_mask = torch.ones_like(input_ids)

        verify_mask = torch.ones(
            (batch_size, N), dtype=ref_attention_mask.dtype, device=device
        )
        extended_attention_mask = torch.cat([ref_attention_mask, verify_mask], dim=1)

        # 3. Training Model Forward (DDP-wrapped `model`)
        outputs = model(
            input_ids=verify_input_ids,
            past_key_values=past_key_values,
            attention_mask=extended_attention_mask,
        )

        # outputs.logits: [B, N, V] -- one row per verify position
        target_token_ids = torch.tensor(
            [self.negative_token_id, self.positive_token_id], device=device
        )
        score_logits = outputs.logits[:, :, target_token_ids]  # [B, N, 2]

        # Loss averaged over all N verify positions; each is trained as a valid scorer.
        loss = _multi_pos_mse(score_logits, labels)

        # Store last-position logits for metrics (matches inference readout).
        outputs.logits = score_logits[:, -1, :]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        CRITICAL OVERRIDE:
        Ensure Evaluation follows the same Ref->Verify logic as Training.
        Without this, Trainer.evaluate() runs standard LM forward pass which yields garbage metrics.
        """
        input_ids = inputs.get("input_ids")
        labels = inputs.get("labels")
        device = input_ids.device

        # 1. Prepare Reference Models
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_ref = self.accelerator.unwrap_model(self.ref_model)
        is_sharing_model = unwrapped_ref is unwrapped_model

        adapter_ctx = nullcontext()
        if is_sharing_model and hasattr(unwrapped_model, "disable_adapter"):
            adapter_ctx = unwrapped_model.disable_adapter()

        # 2. Reference Pass (KV Cache)
        # Usually prediction_step is called in eval mode already, but we ensure consistency
        with torch.no_grad():
            with adapter_ctx:
                forward_model = model if is_sharing_model else self.ref_model
                ref_outputs = forward_model(
                    input_ids=input_ids,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                past_key_values = ref_outputs.past_key_values

        # 3. Verify Pass -- N verify tokens
        batch_size = input_ids.shape[0]
        N = self.num_verify_tokens
        verify_input_ids = torch.full(
            (batch_size, N), self.verify_token_id, dtype=torch.long, device=device
        )

        ref_attention_mask = inputs.get("attention_mask")
        if ref_attention_mask is None:
            ref_attention_mask = torch.ones_like(input_ids)
        verify_mask = torch.ones(
            (batch_size, N), dtype=ref_attention_mask.dtype, device=device
        )
        extended_attention_mask = torch.cat([ref_attention_mask, verify_mask], dim=1)

        with torch.no_grad():
            outputs = model(
                input_ids=verify_input_ids,
                past_key_values=past_key_values,
                attention_mask=extended_attention_mask,
            )

        target_token_ids = torch.tensor(
            [self.negative_token_id, self.positive_token_id], device=device
        )
        score_logits = outputs.logits[:, :, target_token_ids]  # [B, N, 2]

        loss = None
        if labels is not None:
            loss = _multi_pos_mse(score_logits, labels)

        last_logits = score_logits[:, -1, :]  # [B, 2] for metrics
        if prediction_loss_only:
            return (loss, None, None)

        return (loss, last_logits, labels)


# ==========================================
# 2. Metrics & Preprocessing
# ==========================================


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits


def compute_metrics(eval_pred: EvalPrediction):
    predictions, labels = eval_pred

    # Ensure numpy conversion works regardless of shape/device history
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # predictions: [batch_size, 2]
    probs = torch.tensor(predictions)
    probs = torch.softmax(probs, dim=-1)[:, 1].numpy()

    y = (labels + 1) / 2
    yhat = probs

    mask = ~np.isnan(y) & ~np.isnan(yhat)
    y = y[mask]
    yhat = yhat[mask]

    if len(y) == 0:
        return {"mse": 0.0}

    mse = np.mean((y - yhat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - yhat))
    bias = np.mean(yhat - y)

    if len(y) > 1 and np.std(yhat) > 1e-9 and np.std(y) > 1e-9:
        pearson = float(np.corrcoef(yhat, y)[0, 1])
    else:
        pearson = 0.0

    try:
        r2 = r2_score(y, yhat)
        explained_var = explained_variance_score(y, yhat)
    except (ValueError, TypeError):
        r2 = 0.0
        explained_var = 0.0
    return {
        "rmse": rmse,
        "pearson": pearson,
        "r2": r2,
        "explained_variance": explained_var,
        "mae": mae,
        "mse": mse,
        "bias": bias,
    }


# MASDataset uses the same graph message layout as inference.

# ==========================================
# 3. Main Execution
# ==========================================


class SavePRMConfigCallback(TrainerCallback):
    """Write prm_config.json into every checkpoint-N directory so evaluators can auto-detect the encoding."""

    def __init__(self, prm_encoding: str, num_verify_tokens: int = 1):
        self.prm_encoding = prm_encoding
        self.num_verify_tokens = num_verify_tokens

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_prm_config(ckpt_dir, self.prm_encoding, self.num_verify_tokens)


def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.prm_encoding not in ("kv", "text"):
        raise ValueError("prm_encoding must be 'kv' or 'text'")
    if not script_args.train_file:
        parser.error("--train_file is required")
    if not Path(script_args.train_file).is_file():
        parser.error(f"Training file does not exist: {script_args.train_file}")
    if script_args.eval_file and not Path(script_args.eval_file).is_file():
        parser.error(f"Evaluation file does not exist: {script_args.eval_file}")
    if not 0.0 <= script_args.test_size < 1.0:
        parser.error("--test_size must be in [0, 1)")
    if script_args.num_verify_tokens < 1:
        parser.error("--num_verify_tokens must be at least 1")

    # ---------------------------------------------------------
    # DDP Best Practices & Overrides
    # ---------------------------------------------------------

    # DDP often requires disabling find_unused_parameters=True if your graph is fully connected
    # or enabling it if some parameters (like in PEFT) might not be used in every pass.
    # However, since we are doing custom forward passes, False is usually safer to avoid errors
    # unless you get a specific DDP error asking for it.
    if training_args.ddp_find_unused_parameters is None:
        training_args.ddp_find_unused_parameters = False

    if script_args.prm_encoding == "text" and not training_args.gradient_checkpointing:
        training_args.gradient_checkpointing = True
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        if training_args.local_rank in [-1, 0]:
            print(
                "Auto-enabling gradient_checkpointing for text-mode PRM "
                "(full-sequence forward requires it to fit in memory)."
            )

    has_eval_data = (script_args.eval_file is not None) or (script_args.test_size > 0)
    training_args.do_train = True
    training_args.do_eval = has_eval_data
    training_args.eval_strategy = "steps" if has_eval_data else "no"

    training_args.eval_steps = script_args.eval_freq
    training_args.save_strategy = "steps"
    training_args.save_steps = script_args.save_freq
    training_args.logging_steps = script_args.log_freq
    training_args.gradient_accumulation_steps = script_args.grad_accum
    training_args.remove_unused_columns = False

    training_args.label_names = ["labels"]

    # 1. Load Tokenizer & Model
    # Only print on the main process
    if training_args.local_rank in [-1, 0]:
        print(f"Loading model: {script_args.model_name_or_path}")
        print(
            f"dataset (log)={script_args.dataset}, mas={script_args.mas}, "
            f"prm_encoding={script_args.prm_encoding}"
        )

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Get Token IDs
    def get_token_id(token_str):
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Scoring token {token_str!r} must map to exactly one token; got IDs {ids}"
            )
        return ids[0]

    verify_token_id = get_token_id(script_args.verify_token)
    positive_token_id = get_token_id(script_args.positive_token)
    negative_token_id = get_token_id(script_args.negative_token)

    # 3. Initialize Models
    # DDP CRITICAL: Do NOT use device_map="auto" when using Trainer with multiple GPUs.
    # The Trainer handles device placement. Using "auto" forces model to specific GPUs
    # prematurely and breaks DDP.
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map=None,  # Let Trainer handle placement
        trust_remote_code=True,
        # torch_dtype is handled automatically by Trainer's bf16/fp16 args usually,
        # but you can explicit set torch_dtype=torch.bfloat16 if needed.
        torch_dtype=torch.bfloat16 if training_args.bf16 else None,
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
    )
    model = get_peft_model(base_model, peft_config)

    if training_args.local_rank in [-1, 0]:
        model.print_trainable_parameters()

    # 4. Prepare Data — configs/<mas>.yaml matches --mas in run_mcts.py / eval.py
    cfg_path = REPO_ROOT / "configs" / f"{script_args.mas}.yaml"
    if not cfg_path.exists():
        parser.error(
            f"Unknown MAS configuration {script_args.mas!r}: {cfg_path} does not exist"
        )
    cfg = yaml.safe_load(cfg_path.read_text())
    mas_config = {"agents": cfg["agents"], "edges": cfg["edges"]}

    # Load Data (Only on rank 0 usually to save IO, but for simplicity here we load on all)
    # Ideally, load on rank 0 and broadcast, or use datasets library with caching.
    raw_train_data = []
    with open(script_args.train_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    raw_train_data.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {script_args.train_file}:{line_number}: {exc}"
                    ) from exc
    if not raw_train_data:
        parser.error(f"Training file contains no records: {script_args.train_file}")

    train_data_list = raw_train_data
    eval_data_list = None

    if not script_args.eval_file and script_args.test_size > 0:
        if len(raw_train_data) < 2:
            parser.error(
                "At least two records are required when --test_size is nonzero"
            )
        # Fixed seed for consistent splitting across processes
        rng = random.Random(42)
        rng.shuffle(raw_train_data)
        split_idx = max(
            1,
            min(
                len(raw_train_data) - 1,
                int(len(raw_train_data) * (1 - script_args.test_size)),
            ),
        )
        train_data_list = raw_train_data[:split_idx]
        eval_data_list = raw_train_data[split_idx:]

    # Instantiate datasets with the same graph message layout used for inference.
    ds_kw = dict(
        tokenizer=tokenizer,
        mas_config=mas_config,
        max_length=script_args.max_length,
        enable_thinking=script_args.enable_thinking,
    )
    train_dataset = MASDataset(data=train_data_list, **ds_kw)

    eval_dataset = None
    if training_args.do_eval:
        if eval_data_list is not None:
            eval_dataset = MASDataset(data=eval_data_list, **ds_kw)
        elif script_args.eval_file:
            eval_dataset = MASDataset(data_path=script_args.eval_file, **ds_kw)

    def regression_data_collator(features):
        # 1. Extract inputs
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]

        # 2. Extract labels (and ensure they are tensors)
        labels = [f["labels"].clone().detach() for f in features]

        # 3. Pad inputs using the tokenizer
        batch = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        # 4. Stack labels manually (Trainer expects this key)
        batch["labels"] = torch.stack(labels)

        return batch

    # KV mode reuses reference KV; text mode performs full-sequence re-encoding.
    trainer_cls = (
        KVTransferTrainer
        if script_args.prm_encoding == "kv"
        else FullSequencePRMTrainer
    )
    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=regression_data_collator,
        verify_token_id=verify_token_id,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
        num_verify_tokens=script_args.num_verify_tokens,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    if script_args.prm_encoding == "kv":
        trainer_kw["ref_model"] = model
    trainer_kw["callbacks"] = [
        SavePRMConfigCallback(
            script_args.prm_encoding,
            script_args.num_verify_tokens,
        ),
    ]
    trainer = trainer_cls(**trainer_kw)

    # 6. Run
    if training_args.local_rank in [-1, 0]:
        effective_batch = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size
        )
        print(f"Training examples: {len(train_dataset)}")
        print(f"Evaluation examples: {len(eval_dataset) if eval_dataset else 0}")
        print(f"Evaluation strategy: {training_args.eval_strategy}")
        print(f"Effective global batch size: {effective_batch}")

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        if training_args.local_rank in [-1, 0]:
            save_prm_config(
                training_args.output_dir,
                script_args.prm_encoding,
                script_args.num_verify_tokens,
            )

    if training_args.do_eval:
        if training_args.local_rank in [-1, 0]:
            print("Running final evaluation...")
        metrics = trainer.evaluate()
        if training_args.local_rank in [-1, 0]:
            print(metrics)


if __name__ == "__main__":
    main()
