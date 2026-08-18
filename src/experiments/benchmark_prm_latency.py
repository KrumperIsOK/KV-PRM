"""
Benchmark text-PRM vs kv-PRM scoring latency and GPU memory increase.

For each (model_size, prm_kind, mode, cache_impl, seq_len, batch_size):

  text-PRM:
    Measured operation = single forward over [input_ids | K verify_tokens],
    re-encoding the entire context (no kv cache reuse).

  kv-PRM (cache_impl=dynamic):
    Pre-step (NOT measured): produce past_key_values for input_ids by running
    the base model with the adapter disabled — emulates the cache the agent
    would already have resident on the GPU. Uses transformers' DynamicCache.
    Measured operation = verify-token forward only (K tokens, with the
    resident past_key_values and the LoRA adapter active). The DynamicCache
    `update` performs a torch.cat(old_kv, new_kv) per layer, allocating a new
    tensor of size ~B*L per layer; this transient dominates the peak.

  kv-PRM (cache_impl=static):
    Same protocol but with `transformers.StaticCache`, pre-allocated to size
    L+K. Cache updates become in-place writes (no realloc), so the per-layer
    transient drops to ~O(B*K).

We report, per cell:
  * mean time per sequence (ms) over `n_iters` timed runs
  * peak GPU memory increase during the measured op (MiB)  -- max_memory_allocated
  * time-average GPU memory increase during the op (MiB)   -- background sampler
    (both excluding any state already resident before reset_peak, e.g. the kv cache)

Modes:
  auto         — auto-found largest power-of-2 batch size that fits.
  match_kv     — text-PRM rerun at the same B that kv-PRM (dynamic) used.

Resume:
  Each row is flushed+fsynced after measurement. Re-running picks up where
  it stopped, keyed on (model_size, prm_kind, mode, cache_impl, seq_len).
  CSVs from earlier runs are auto-migrated to the current schema.
"""

import argparse
import csv
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

import torch
from transformers import StaticCache

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from experiments.core import load_text_prm_scorer, load_kv_prm_scorer  # noqa: E402


MIB = 1024 * 1024


def _resolve_ckpt_dir(prm_dir: Path) -> Path:
    if (prm_dir / "adapter_config.json").exists():
        return prm_dir
    ckpts = []
    for p in prm_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("checkpoint-"):
            continue
        try:
            ckpts.append((int(p.name.split("-", 1)[1]), p))
        except ValueError:
            continue
    if not ckpts:
        raise FileNotFoundError(
            f"No adapter_config.json or checkpoint-*/ found under {prm_dir}"
        )
    ckpts.sort()
    return ckpts[-1][1]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_scorer(prm_kind: str, prm_dir: Path, base_model_id: str, device: str):
    if prm_kind == "text":
        score_fn, tok, model = load_text_prm_scorer(
            str(prm_dir),
            base_model_id=base_model_id,
            device=device,
        )
    elif prm_kind == "kv":
        score_fn, tok, model = load_kv_prm_scorer(
            str(prm_dir),
            base_model_id=base_model_id,
            device=device,
        )
    else:
        raise ValueError(prm_kind)

    def get_id(s):
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if ids else tok.eos_token_id

    K = _read_k(prm_dir)
    return (
        model,
        tok,
        {"verify": get_id("?"), "pos": get_id("+"), "neg": get_id("-"), "K": K},
    )


def _read_k(prm_dir: Path) -> int:
    cfg = prm_dir / "prm_config.json"
    if cfg.exists():
        try:
            return max(1, int(json.loads(cfg.read_text()).get("num_verify_tokens", 1)))
        except Exception:
            pass
    return 1


# ---------------------------------------------------------------------------
# Memory sampler — runs on a background thread, samples memory_allocated().
# Used for time-averaged memory; peak comes from torch's built-in stats.
# ---------------------------------------------------------------------------
class CudaMemSampler:
    def __init__(self, interval: float = 0.0005):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.samples = []

    def start(self):
        self.samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(torch.cuda.memory_allocated())
            except Exception:
                pass
            time.sleep(self.interval)

    @property
    def mean(self) -> float:
        return sum(self.samples) / max(1, len(self.samples))


# ---------------------------------------------------------------------------
# Op kernels — each takes a context-bag dict and runs one timed step.
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _text_step(model, ctx):
    out = model(
        input_ids=ctx["full_ids"],
        attention_mask=ctx["full_mask"],
        use_cache=False,
        logits_to_keep=ctx["K"],
    )
    return out.logits


@torch.inference_mode()
def _kv_step_dynamic(model, ctx):
    out = model(
        input_ids=ctx["verify_ids"],
        attention_mask=ctx["full_mask"],
        past_key_values=ctx["pkv"],
    )
    return out.logits


@torch.inference_mode()
def _kv_step_static(model, ctx):
    out = model(
        input_ids=ctx["verify_ids"],
        attention_mask=ctx["full_mask"],
        past_key_values=ctx["pkv"],
        cache_position=ctx["verify_positions"],
    )
    return out.logits


@torch.inference_mode()
def _kv_prepare_dynamic(model, input_ids, attention_mask):
    with model.disable_adapter():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
    return out.past_key_values


@torch.inference_mode()
def _kv_prepare_static(model, input_ids, attention_mask, K):
    """StaticCache prepare: lazy-allocates full cache to L+K on first forward."""
    B, L = input_ids.shape
    base = model.base_model.model if hasattr(model, "base_model") else model
    config = base.config
    cache = StaticCache(config=config, max_cache_len=L + K)
    cache_position = torch.arange(L, device=input_ids.device)
    with model.disable_adapter():
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
    return cache


# ---------------------------------------------------------------------------
# Build a context bag for one (kind, cache_impl, B, L) configuration.
# Caller is responsible for releasing whatever the bag holds (esp. cache).
# ---------------------------------------------------------------------------
def _make_inputs(B, L, vocab_size, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(0, vocab_size, (B, L), generator=g, dtype=torch.long)
    return ids.to(device, non_blocking=True)


def _build_ctx(model, prm_kind, cache_impl, B, L, vocab_size, K, verify_id, device):
    """Returns (ctx, op). ctx holds tensors; op(model, ctx) runs one step."""
    ids = _make_inputs(B, L, vocab_size, device)
    attn = torch.ones_like(ids)
    verify_ids = torch.full((B, K), verify_id, dtype=ids.dtype, device=device)
    verify_mask = torch.ones((B, K), dtype=attn.dtype, device=device)
    full_mask = torch.cat([attn, verify_mask], dim=1)
    ctx = {"K": K, "verify_ids": verify_ids, "full_mask": full_mask}

    if prm_kind == "text":
        ctx["full_ids"] = torch.cat([ids, verify_ids], dim=1)
        return ctx, _text_step

    # kv-prm: untimed prepare, then return the verify-step op.
    if cache_impl == "static":
        ctx["pkv"] = _kv_prepare_static(model, ids, attn, K)
        ctx["verify_positions"] = torch.arange(L, L + K, device=device)
        op = _kv_step_static
    else:
        ctx["pkv"] = _kv_prepare_dynamic(model, ids, attn)
        op = _kv_step_dynamic
    return ctx, op


# ---------------------------------------------------------------------------
# Time + measure peak/avg memory of one configuration.
# ---------------------------------------------------------------------------
def _time_and_measure(model, ctx, op, n_warmup, n_iters, sampler_interval):
    for _ in range(n_warmup):
        op(model, ctx)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    sampler = CudaMemSampler(interval=sampler_interval)
    sampler.start()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        op(model, ctx)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    sampler.stop()

    peak_inc = torch.cuda.max_memory_allocated() - mem_before
    avg_inc = sampler.mean - mem_before
    return {
        "total_time_s": elapsed,
        "peak_mem_inc_mib": peak_inc / MIB,
        "avg_mem_inc_mib": avg_inc / MIB,
        "n_samples": len(sampler.samples),
    }


def _try_run(
    model,
    prm_kind,
    cache_impl,
    B,
    L,
    vocab_size,
    K,
    verify_id,
    device,
    n_warmup,
    n_iters,
    sampler_interval,
):
    """Run one (B,L) config; return result dict or raise on OOM."""
    ctx, op = _build_ctx(
        model, prm_kind, cache_impl, B, L, vocab_size, K, verify_id, device
    )
    try:
        res = _time_and_measure(model, ctx, op, n_warmup, n_iters, sampler_interval)
    finally:
        # Drop any cache held by ctx so the next config starts clean.
        ctx.clear()
    res.update(
        {
            "time_per_seq_ms": (res["total_time_s"] / n_iters / B) * 1000.0,
            "time_per_batch_ms": (res["total_time_s"] / n_iters) * 1000.0,
        }
    )
    return res


def _is_oom(e: Exception) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def auto_find_batch_size(
    model, prm_kind, cache_impl, L, vocab_size, K, verify_id, device, max_bs, n_warmup
):
    """Double B from 1 until OOM. Returns the largest B that didn't OOM."""
    best = 0
    bs = 1
    while bs <= max_bs:
        try:
            ctx, op = _build_ctx(
                model,
                prm_kind,
                cache_impl,
                bs,
                L,
                vocab_size,
                K,
                verify_id,
                device,
            )
            for _ in range(n_warmup):
                op(model, ctx)
            torch.cuda.synchronize()
            best = bs
            ctx.clear()
            torch.cuda.empty_cache()
            bs *= 2
        except Exception as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            gc.collect()
            break
    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
FIELDS = [
    "model_size",
    "prm_kind",
    "mode",
    "cache_impl",
    "seq_len",
    "batch_size",
    "time_per_seq_ms",
    "time_per_batch_ms",
    "peak_mem_inc_mib",
    "avg_mem_inc_mib",
    "total_time_s",
    "n_iters",
    "K",
    "gpu_name",
    "gpu_total_mib",
]


def _migrate_csv(out_path: Path):
    """Bring older CSVs up to FIELDS (adds cache_impl, avg_mem_inc_mib).
    Pre-existing rows are tagged cache_impl='dynamic' for kv (the old code
    path) and 'n/a' for text (no cache). avg_mem is left blank."""
    if not out_path.exists():
        return
    with out_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        old = list(reader)
        old_header = reader.fieldnames or []
    if not old:
        return
    needs_migration = any(
        c not in old_header for c in ("cache_impl", "avg_mem_inc_mib")
    )
    if not needs_migration:
        return
    print(f"Migrating {out_path}: adding cache_impl, avg_mem_inc_mib")
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for row in old:
            row.setdefault("mode", "auto")
            if "cache_impl" not in row:
                row["cache_impl"] = (
                    "n/a" if row.get("prm_kind") == "text" else "dynamic"
                )
            row.setdefault("avg_mem_inc_mib", "")
            w.writerow({k: row.get(k, "") for k in FIELDS})


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint_map",
        type=Path,
        required=True,
        help="JSON mapping model sizes to base_model, kv, and text checkpoint paths.",
    )
    p.add_argument("--model_sizes", nargs="+", default=["0.6B", "4B", "8B"])
    p.add_argument("--prm_kinds", nargs="+", default=["text", "kv"])
    p.add_argument("--seq_lens", nargs="+", type=int, default=[1024, 2048, 3072, 4096])
    p.add_argument(
        "--batch_size",
        type=int,
        default=0,
        help="Fixed batch size; 0 = auto-find largest power of 2 that fits.",
    )
    p.add_argument("--max_batch_size", type=int, default=512)
    p.add_argument("--n_warmup", type=int, default=2)
    p.add_argument("--n_iters", type=int, default=5)
    p.add_argument(
        "--sampler_interval_ms",
        type=float,
        default=0.5,
        help="Memory-sampler period (ms). Lower = more samples, more overhead.",
    )
    p.add_argument(
        "--cache_impls",
        nargs="+",
        default=["dynamic", "static"],
        help="Which cache implementations to benchmark for kv-PRM.",
    )
    p.add_argument("--output", default="artifacts/benchmarks/prm_latency.csv")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if not args.checkpoint_map.is_file():
        p.error(f"Checkpoint map does not exist: {args.checkpoint_map}")
    with args.checkpoint_map.open(encoding="utf-8") as fh:
        checkpoint_map = json.load(fh)
    for size in args.model_sizes:
        if size not in checkpoint_map:
            p.error(f"Model size {size!r} is missing from {args.checkpoint_map}")
        missing = {"base_model", *args.prm_kinds} - set(checkpoint_map[size])
        if missing:
            p.error(f"Model size {size!r} is missing keys: {sorted(missing)}")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    sampler_interval = args.sampler_interval_ms / 1000.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _migrate_csv(out_path)

    # Resume index. Key = (model_size, prm_kind, mode, cache_impl, seq_len).
    done = set()
    b_kv_dynamic = {}  # (size, L) -> B for kv/dynamic auto, used for match_kv text rows
    if out_path.exists():
        with out_path.open("r", newline="") as fh:
            for row in csv.DictReader(fh):
                mode = row.get("mode") or "auto"
                cimpl = row.get("cache_impl") or (
                    "n/a" if row["prm_kind"] == "text" else "dynamic"
                )
                key = (
                    row["model_size"],
                    row["prm_kind"],
                    mode,
                    cimpl,
                    int(row["seq_len"]),
                )
                done.add(key)
                if row["prm_kind"] == "kv" and mode == "auto" and cimpl == "dynamic":
                    b_kv_dynamic[(row["model_size"], int(row["seq_len"]))] = int(
                        row["batch_size"]
                    )
        print(f"Resume: loaded {len(done)} completed cells from {out_path}")

    new_file = not out_path.exists()
    f = out_path.open("a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()
        f.flush()

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total = int(torch.cuda.get_device_properties(0).total_memory / MIB)

    def write_row(size, kind, mode, cimpl, L, B, K, res):
        row = {
            "model_size": size,
            "prm_kind": kind,
            "mode": mode,
            "cache_impl": cimpl,
            "seq_len": L,
            "batch_size": B,
            "time_per_seq_ms": round(res["time_per_seq_ms"], 3),
            "time_per_batch_ms": round(res["time_per_batch_ms"], 3),
            "peak_mem_inc_mib": round(res["peak_mem_inc_mib"], 1),
            "avg_mem_inc_mib": round(res["avg_mem_inc_mib"], 1),
            "total_time_s": round(res["total_time_s"], 4),
            "n_iters": args.n_iters,
            "K": K,
            "gpu_name": gpu_name,
            "gpu_total_mib": gpu_total,
        }
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        done.add((size, kind, mode, cimpl, L))
        print(f"  -> {row}")

    def measure(model, kind, cimpl, B, L, vocab, K, verify_id):
        try:
            return B, _try_run(
                model,
                kind,
                cimpl,
                B,
                L,
                vocab,
                K,
                verify_id,
                args.device,
                args.n_warmup,
                args.n_iters,
                sampler_interval,
            )
        except Exception as e:
            if not _is_oom(e):
                raise
            torch.cuda.empty_cache()
            gc.collect()
            B2 = max(1, B // 2)
            print(f"  [oom] retry at B={B2}")
            return B2, _try_run(
                model,
                kind,
                cimpl,
                B2,
                L,
                vocab,
                K,
                verify_id,
                args.device,
                args.n_warmup,
                args.n_iters,
                sampler_interval,
            )

    def load(size, kind):
        prm_root = Path(checkpoint_map[size][kind]).expanduser()
        base_id = checkpoint_map[size]["base_model"]
        print(f"\n==== loading {size} / {kind} ====")
        print(f"     prm_root  = {prm_root}")
        print(f"     base_id   = {base_id}")
        if not prm_root.exists():
            print("     [skip] prm_root does not exist")
            return None
        try:
            prm_dir = _resolve_ckpt_dir(prm_root)
        except FileNotFoundError as e:
            print(f"     [skip] {e}")
            return None
        if prm_dir != prm_root:
            print(f"     prm_dir   = {prm_dir}  (latest checkpoint)")
        return load_scorer(kind, prm_dir, base_id, args.device)

    kv_impls = [c for c in args.cache_impls if c in ("dynamic", "static")]

    for size in args.model_sizes:
        # ---------------------------- kv ----------------------------
        if "kv" in args.prm_kinds:
            kv_pending = [
                (L, c)
                for L in args.seq_lens
                for c in kv_impls
                if (size, "kv", "auto", c, L) not in done
            ]
            if not kv_pending:
                print(f"\n[skip] {size}/kv: all cells already in CSV")
            else:
                loaded = load(size, "kv")
                if loaded is not None:
                    model, tok, ids = loaded
                    K = ids["K"]
                    verify_id = ids["verify"]
                    vocab = tok.vocab_size
                    # Outer loop: cache_impl. Saves us from re-loading the model.
                    for cimpl in kv_impls:
                        for L in args.seq_lens:
                            if (size, "kv", "auto", cimpl, L) in done:
                                print(f"  [done] {size}/kv L={L} {cimpl} auto -- skip")
                                continue
                            if args.batch_size > 0:
                                B = args.batch_size
                            else:
                                print(f"  >>> auto-finding kv/{cimpl} B for L={L}...")
                                try:
                                    B = auto_find_batch_size(
                                        model,
                                        "kv",
                                        cimpl,
                                        L,
                                        vocab,
                                        K,
                                        verify_id,
                                        args.device,
                                        args.max_batch_size,
                                        args.n_warmup,
                                    )
                                except Exception as e:
                                    print(
                                        f"  [error] kv/{cimpl} L={L} auto-find failed: {e}"
                                    )
                                    continue
                                if B == 0:
                                    print(f"  >>> L={L}: even B=1 OOMed -- skipping")
                                    continue
                                print(f"  >>> kv/{cimpl} L={L}: largest fitting B={B}")
                            try:
                                B, res = measure(
                                    model, "kv", cimpl, B, L, vocab, K, verify_id
                                )
                            except Exception as e:
                                print(f"  [error] kv/{cimpl} L={L} measure failed: {e}")
                                torch.cuda.empty_cache()
                                gc.collect()
                                continue
                            if cimpl == "dynamic":
                                b_kv_dynamic[(size, L)] = B
                            write_row(size, "kv", "auto", cimpl, L, B, K, res)
                            torch.cuda.empty_cache()
                    del model, tok
                    gc.collect()
                    torch.cuda.empty_cache()

        # ---------------------------- text ----------------------------
        if "text" in args.prm_kinds:
            text_pending = []
            for L in args.seq_lens:
                if (size, "text", "auto", "n/a", L) not in done:
                    text_pending.append((L, "auto"))
                if (size, L) in b_kv_dynamic and (
                    size,
                    "text",
                    "match_kv",
                    "n/a",
                    L,
                ) not in done:
                    text_pending.append((L, "match_kv"))
            if not text_pending:
                print(f"\n[skip] {size}/text: all cells already in CSV")
                continue
            loaded = load(size, "text")
            if loaded is None:
                continue
            model, tok, ids = loaded
            K = ids["K"]
            verify_id = ids["verify"]
            vocab = tok.vocab_size
            for L in args.seq_lens:
                # ---- auto ----
                if (size, "text", "auto", "n/a", L) in done:
                    print(f"  [done] {size}/text L={L} auto -- skip")
                else:
                    if args.batch_size > 0:
                        B = args.batch_size
                    else:
                        print(f"  >>> auto-finding text B for L={L}...")
                        B = auto_find_batch_size(
                            model,
                            "text",
                            "n/a",
                            L,
                            vocab,
                            K,
                            verify_id,
                            args.device,
                            args.max_batch_size,
                            args.n_warmup,
                        )
                        if B == 0:
                            print(f"  >>> L={L}: text even B=1 OOMed")
                            B = None
                    if B is not None:
                        B, res = measure(
                            model, "text", "n/a", B, L, vocab, K, verify_id
                        )
                        write_row(size, "text", "auto", "n/a", L, B, K, res)
                        torch.cuda.empty_cache()

                # ---- match_kv (matches dynamic kv batch size) ----
                if (size, "text", "match_kv", "n/a", L) in done:
                    print(f"  [done] {size}/text L={L} match_kv -- skip")
                    continue
                B_kv = b_kv_dynamic.get((size, L))
                if B_kv is None:
                    print(f"  [info] no dynamic kv B for {size} L={L} -- skip match_kv")
                    continue
                print(f"  >>> text L={L}: matching dynamic-kv B={B_kv}")
                try:
                    B_used, res = measure(
                        model, "text", "n/a", B_kv, L, vocab, K, verify_id
                    )
                except Exception as e:
                    if not _is_oom(e):
                        raise
                    print(f"  [oom] match_kv {size} L={L} B={B_kv} after retry -- skip")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                write_row(size, "text", "match_kv", "n/a", L, B_used, K, res)
                torch.cuda.empty_cache()
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()

    f.close()
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
