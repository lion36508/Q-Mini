from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .config import QMiniConfig
from .model import QMiniMoE


def _sha256_file(path: str | Path) -> dict[str, int | str]:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": target.stat().st_size}


def _load_tokens(path: str | Path, name: str) -> np.memmap:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{name} token stream not found: {target}")
    if target.stat().st_size % np.dtype(np.uint16).itemsize:
        raise ValueError(f"{name} token stream is not aligned to uint16: {target}")
    values = np.memmap(target, mode="r", dtype=np.uint16)
    if values.size < 4:
        raise ValueError(f"{name} token stream is too short: {target}")
    return values


def _validate_token_ids(tokens: np.memmap, vocab_size: int, name: str) -> None:
    """Scan a memmap in bounded chunks; fail before training if IDs exceed vocab."""
    chunk_size = 1 << 20
    observed_max = 0
    for start in range(0, len(tokens), chunk_size):
        chunk = np.asarray(tokens[start:start + chunk_size])
        if chunk.size:
            observed_max = max(observed_max, int(chunk.max()))
    if observed_max >= vocab_size:
        raise ValueError(
            f"{name} stream contains token ID {observed_max}, which is outside the configured vocabulary "
            f"[0, {vocab_size - 1}]. "
            "Rebuild token streams with the tokenizer paired to this model config."
        )


def _load_masks(path: str | Path | None, tokens: np.memmap, name: str) -> np.memmap | None:
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{name} loss-mask file not found: {target}")
    if target.stat().st_size != len(tokens):
        raise ValueError(
            f"{name} loss mask has {target.stat().st_size} bytes, but token stream has {len(tokens)} tokens."
        )
    masks = np.memmap(target, mode="r", dtype=np.uint8)
    chunk_size = 1 << 20
    target_count = 0
    for start in range(0, len(masks), chunk_size):
        chunk = np.asarray(masks[start:start + chunk_size])
        if not np.isin(chunk, (0, 1)).all():
            raise ValueError(f"{name} loss mask must contain only 0 and 1.")
        target_count += int(chunk.sum())
    if target_count == 0:
        raise ValueError(f"{name} loss mask contains no supervised target tokens.")
    return masks


def _sample_batch(
    tokens: np.memmap,
    masks: np.memmap | None,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_start = len(tokens) - seq_len - 1
    if max_start < 0:
        raise ValueError(
            f"Token stream needs at least {seq_len + 2} tokens; found {len(tokens)}."
        )

    samples = []
    target_masks = []
    for _row in range(batch_size):
        selected = None
        for _attempt in range(128):
            start = int(rng.integers(0, max_start + 1))
            label_mask = (
                np.ones(seq_len, dtype=np.uint8)
                if masks is None
                else np.asarray(masks[start + 1:start + seq_len + 1], dtype=np.uint8)
            )
            if int(label_mask.sum()) > 0:
                selected = start, label_mask
                break
        if selected is None:
            raise ValueError(
                "Could not sample a sequence containing supervised targets. "
                "Increase seq_len or provide denser assistant-response labels."
            )
        start, label_mask = selected
        samples.append(np.asarray(tokens[start:start + seq_len + 1], dtype=np.int64))
        target_masks.append(label_mask)

    batch = np.stack(samples)
    x = torch.as_tensor(batch[:, :-1], dtype=torch.long, device=device)
    y = torch.as_tensor(batch[:, 1:], dtype=torch.long, device=device)
    loss_mask = torch.as_tensor(np.stack(target_masks), dtype=torch.float32, device=device)
    return x, y, loss_mask


def _masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    per_token = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape_as(targets)
    weights = target_mask.to(dtype=per_token.dtype)
    denominator = weights.sum()
    if not bool(denominator > 0):
        raise ValueError("A sampled batch contains no supervised target tokens.")
    return (per_token * weights).sum() / denominator


def _evaluate(
    model: QMiniMoE,
    tokens: np.memmap,
    masks: np.memmap | None,
    batch_size: int,
    seq_len: int,
    batches: int,
    device: torch.device,
    seed: int,
) -> dict[str, float | None]:
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    try:
        with torch.inference_mode():
            for _ in range(batches):
                x, y, target_mask = _sample_batch(
                    tokens, masks, batch_size, seq_len, rng, device
                )
                result = model(x)
                loss = _masked_cross_entropy(
                    result["logits"], y, target_mask
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Validation loss became non-finite.")
                losses.append(float(loss.item()))
    finally:
        model.train(was_training)

    mean_loss = float(np.mean(losses))
    perplexity = math.exp(mean_loss) if mean_loss < 709.0 else None
    return {"eval_loss": mean_loss, "eval_perplexity": perplexity}


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _resume_contract(
    train_path: str | Path,
    valid_path: str | Path,
    tokenizer_path: str | Path | None,
    train_mask_path: str | Path | None,
    valid_mask_path: str | Path | None,
    batch_size: int,
    seq_len: int,
    grad_accum: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    train_file = Path(train_path).resolve()
    valid_file = Path(valid_path).resolve()
    train_fingerprint = _sha256_file(train_file)
    valid_fingerprint = _sha256_file(valid_file)
    if train_file == valid_file or train_fingerprint == valid_fingerprint:
        raise ValueError("Training and validation streams must be different files with different content.")
    tokenizer_fingerprint = (
        _sha256_file(tokenizer_path) if tokenizer_path is not None and Path(tokenizer_path).is_file()
        else None
    )
    return {
        "train": train_fingerprint,
        "validation": valid_fingerprint,
        "train_loss_mask": _sha256_file(train_mask_path) if train_mask_path else None,
        "validation_loss_mask": _sha256_file(valid_mask_path) if valid_mask_path else None,
        "tokenizer": tokenizer_fingerprint,
        "batch_size": int(batch_size),
        "sequence_length": int(seq_len),
        "gradient_accumulation": int(grad_accum),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
    }


def _reconcile_metrics(path: Path, checkpoint_step: int) -> None:
    """Discard log rows newer than the last durable checkpoint after an interrupted run."""
    if not path.is_file():
        return
    kept: list[str] = []
    changed = False
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                step = int(record["step"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                changed = True
                continue
            if step <= checkpoint_step:
                kept.append(json.dumps(record, allow_nan=False))
            else:
                changed = True
    if changed:
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for line in kept:
                    handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def train_model(
    config: QMiniConfig,
    train_bin: str | Path,
    valid_bin: str | Path,
    output_dir: str | Path,
    steps: int,
    batch_size: int = 4,
    seq_len: int = 512,
    grad_accum: int = 8,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    eval_every: int = 250,
    save_every: int = 250,
    eval_batches: int = 20,
    resume: bool = False,
    tokenizer_path: str | Path | None = None,
    seed: int | None = None,
    extend_schedule: bool = False,
    allow_legacy_resume: bool = False,
    train_mask_bin: str | Path | None = None,
    valid_mask_bin: str | Path | None = None,
) -> Path:
    """Train from scratch or resume only when data and optimization settings match."""
    config.validate()
    positive = {
        "steps": steps,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "grad_accum": grad_accum,
        "eval_every": eval_every,
        "save_every": save_every,
        "eval_batches": eval_batches,
    }
    invalid = [name for name, value in positive.items() if value < 1]
    if invalid:
        raise ValueError("These training settings must be positive: " + ", ".join(invalid))
    if not 2 <= seq_len <= config.max_context:
        raise ValueError(f"seq_len must be between 2 and {config.max_context}.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative.")

    if (train_mask_bin is None) != (valid_mask_bin is None):
        raise ValueError("Provide both --train-mask-bin and --valid-mask-bin, or neither.")
    train_tokens = _load_tokens(train_bin, "Training")
    valid_tokens = _load_tokens(valid_bin, "Validation")
    train_masks = _load_masks(train_mask_bin, train_tokens, "Training")
    valid_masks = _load_masks(valid_mask_bin, valid_tokens, "Validation")
    _validate_token_ids(train_tokens, config.vocab_size, "Training")
    _validate_token_ids(valid_tokens, config.vocab_size, "Validation")
    # Fail early before model allocation when a batch cannot be sampled.
    if len(train_tokens) < seq_len + 2 or len(valid_tokens) < seq_len + 2:
        raise ValueError(
            f"Both streams need at least {seq_len + 2} tokens for sequence length {seq_len}."
        )

    actual_seed = config.seed if seed is None else int(seed)
    contract = _resume_contract(
        train_bin, valid_bin, tokenizer_path, train_mask_bin, valid_mask_bin,
        batch_size, seq_len, grad_accum, learning_rate, weight_decay, actual_seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "latest.pt"
    metrics_path = run_dir / "metrics.jsonl"
    run_manifest_path = run_dir / "run.json"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError(
            f"Checkpoint exists: {checkpoint_path}; pass --resume or choose a new run directory."
        )
    if metrics_path.exists() and not resume and not checkpoint_path.exists():
        raise FileExistsError(
            f"Metrics log already exists: {metrics_path}; refusing to mix unrelated training runs."
        )

    random.seed(actual_seed)
    rng = np.random.default_rng(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)

    model = QMiniMoE(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )
    use_amp = device.type == "cuda"
    bf16_supported = (
        bool(torch.cuda.is_bf16_supported())
        if use_amp and hasattr(torch.cuda, "is_bf16_supported")
        else False
    )
    use_bf16 = bool(use_amp and bf16_supported)
    use_fp16 = bool(use_amp and not use_bf16)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    start_step = 0
    tokens_seen = 0
    supervised_tokens_seen = 0
    schedule_total_steps = steps
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"No checkpoint to resume: {checkpoint_path}")
        try:
            state = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
        except TypeError:
            state = torch.load(checkpoint_path, map_location=device)

        if state.get("format") not in {"qmini-checkpoint-v1", "qmini-checkpoint-v2"}:
            raise ValueError("Unsupported or unrecognized Q-Mini checkpoint format.")
        saved_config = QMiniConfig.from_dict(state["config"])
        if saved_config.to_dict() != config.to_dict():
            raise ValueError("Checkpoint model config differs; refusing an unsafe resume.")

        saved_contract = state.get("resume_contract")
        if saved_contract is None:
            if not allow_legacy_resume:
                raise ValueError(
                    "This checkpoint has no dataset/training fingerprint. "
                    "Use --allow-legacy-resume only after personally verifying its source data."
                )
            warnings.warn(
                "Legacy checkpoint accepted without dataset and optimizer-contract verification.",
                RuntimeWarning,
            )
        elif saved_contract != contract:
            raise ValueError(
                "Resume contract mismatch (dataset/tokenizer fingerprint or optimizer settings). "
                "Use a new run directory for changed data or settings."
            )

        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        tokens_seen = int(state.get(
            "tokens_seen", start_step * batch_size * grad_accum * seq_len
        ))
        supervised_tokens_seen = int(state.get("supervised_tokens_seen", tokens_seen))
        schedule_total_steps = int(state.get("schedule_total_steps", steps))
        if "numpy_rng" in state:
            rng.bit_generator.state = state["numpy_rng"]
        if "python_rng" in state:
            random.setstate(state["python_rng"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        saved_cuda_rng = state.get("cuda_rng")
        if torch.cuda.is_available() and saved_cuda_rng is not None:
            if len(saved_cuda_rng) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all([item.cpu() for item in saved_cuda_rng])
        if "scaler" in state:
            scaler.load_state_dict(state["scaler"])

        if start_step >= steps:
            raise ValueError(
                f"Checkpoint is already at step {start_step}; --steps must be a larger total target."
            )
        if steps > schedule_total_steps:
            if not extend_schedule:
                raise ValueError(
                    f"Checkpoint scheduler horizon is {schedule_total_steps} steps. "
                    "Pass --extend-schedule to deliberately extend the learning-rate schedule."
                )
            schedule_total_steps = steps
        print(
            f"Resuming from step {start_step:,}; schedule horizon={schedule_total_steps:,}."
        )

    run_manifest = {
        "format": "qmini-run-manifest-v1",
        "model_config": config.to_dict(),
        "resume_contract": contract,
        "schedule_total_steps": schedule_total_steps,
        "precision": "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
        "device": str(device),
    }
    _atomic_json_save(run_manifest, run_manifest_path)
    if resume:
        _reconcile_metrics(metrics_path, start_step)

    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    print(
        f"Device={device}; AMP={'BF16' if use_bf16 else 'FP16' if use_fp16 else 'off'}; "
        f"parameters={model.parameter_report()['total']:,}; "
        f"global_batch={batch_size * grad_accum}; "
        f"tokens/optimizer_step={batch_size * grad_accum * seq_len:,}"
    )

    def learning_rate_at(step: int) -> float:
        warmup = max(1, int(schedule_total_steps * 0.03))
        if step < warmup:
            return learning_rate * (step + 1) / warmup
        progress = min(
            1.0,
            max(0.0, (step - warmup) / max(1, schedule_total_steps - warmup)),
        )
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    model.train()
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True)
        accum = {"loss": 0.0, "ce": 0.0, "aux": 0.0, "targets": 0.0}
        sampled_thinking_steps = random.randint(1, config.max_thinking_steps)
        router_counts_total: list[torch.Tensor] | None = None

        for _micro_step in range(grad_accum):
            x, y, target_mask = _sample_batch(
                train_tokens, train_masks, batch_size, seq_len, rng, device
            )
            amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
            amp_context = (
                torch.autocast(device_type="cuda", dtype=amp_dtype)
                if use_amp else nullcontext()
            )
            with amp_context:
                result = model(x, thinking_steps=sampled_thinking_steps)
                cross_entropy = _masked_cross_entropy(
                    result["logits"], y, target_mask
                )
                auxiliary = result["aux_loss"].float()
                loss = cross_entropy + config.router_aux_loss_coef * auxiliary

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at optimizer step {step + 1}; checkpoint is unchanged."
                )
            scaled_loss = loss / grad_accum
            if use_fp16:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum["loss"] += float(loss.detach()) / grad_accum
            accum["ce"] += float(cross_entropy.detach()) / grad_accum
            accum["aux"] += float(auxiliary.detach()) / grad_accum
            accum["targets"] += float(target_mask.sum().detach())
            batch_counts = [
                item.detach().to(device="cpu", dtype=torch.int64)
                for item in result["router_counts"]
            ]
            if router_counts_total is None:
                router_counts_total = batch_counts
            else:
                for index, count in enumerate(batch_counts):
                    router_counts_total[index] += count

        if use_fp16:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        lr = learning_rate_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        completed = step + 1
        tokens_seen += batch_size * grad_accum * seq_len
        supervised_tokens_seen += int(accum["targets"])
        record: dict[str, Any] = {
            "step": completed,
            "tokens_seen": tokens_seen,
            "supervised_tokens_seen": supervised_tokens_seen,
            "train_loss": accum["loss"],
            "train_cross_entropy": accum["ce"],
            "router_aux_loss": accum["aux"],
            "gradient_norm": float(grad_norm.detach()),
            "learning_rate": lr,
            "thinking_steps": sampled_thinking_steps,
            "router_token_counts": (
                [counts.tolist() for counts in router_counts_total]
                if router_counts_total is not None else []
            ),
        }

        if completed % eval_every == 0 or completed == steps:
            record.update(_evaluate(
                model, valid_tokens, valid_masks, batch_size, seq_len, eval_batches,
                device, actual_seed + completed,
            ))
        if use_amp:
            record["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())

        if completed % save_every == 0 or completed == steps:
            checkpoint_payload = {
                "format": "qmini-checkpoint-v2",
                "step": completed,
                "tokens_seen": tokens_seen,
                "supervised_tokens_seen": supervised_tokens_seen,
                "schedule_total_steps": schedule_total_steps,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config.to_dict(),
                "resume_contract": contract,
                "numpy_rng": rng.bit_generator.state,
                "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "scaler": scaler.state_dict(),
            }
            _atomic_torch_save(checkpoint_payload, checkpoint_path)

        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        if completed == 1 or completed % 10 == 0 or "eval_loss" in record:
            message = (
                f"step={completed:,}/{steps:,} tokens={tokens_seen:,} "
                f"loss={accum['loss']:.4f} ce={accum['ce']:.4f} "
                f"aux={accum['aux']:.4f} grad={float(grad_norm):.3f} lr={lr:.2e}"
            )
            if "eval_loss" in record:
                perplexity_value = record["eval_perplexity"]
                perplexity_text = (
                    f"{perplexity_value:.2f}"
                    if perplexity_value is not None else "overflow"
                )
                message += (
                    f" val_loss={record['eval_loss']:.4f}"
                    f" ppl={perplexity_text}"
                )
            print(message, flush=True)

    final_dir = run_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(model.state_dict(), final_dir / "model.pt")
    config.save(final_dir / "config.json")
    if tokenizer_path is not None and Path(tokenizer_path).is_file():
        shutil.copy2(tokenizer_path, final_dir / "tokenizer.json")

    from .export import export_packed_state
    packed_path = export_packed_state(
        model.state_dict(), config, final_dir / "model.qmini.npz"
    )
    final_report = {
        "format": "qmini-final-model-v1",
        "step": steps,
        "tokens_seen": tokens_seen,
        "supervised_tokens_seen": supervised_tokens_seen,
        "loss_masked_training": train_masks is not None,
        "parameters": model.parameter_report()["total"],
        "packed_export": packed_path.name,
        "packed_export_bytes": packed_path.stat().st_size,
        "config": config.to_dict(),
    }
    _atomic_json_save(final_report, final_dir / "model_manifest.json")
    print(
        f"Training complete: {final_dir} | packed export={packed_path.stat().st_size:,} bytes"
    )
    return final_dir
