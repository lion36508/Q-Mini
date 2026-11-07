from __future__ import annotations
import json, math, os, random, shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
from .config import QMiniConfig
from .model import QMiniMoE

def _load_tokens(path):
    target = Path(path)
    if not target.is_file(): raise FileNotFoundError(f"Token stream not found: {target}")
    if target.stat().st_size % 2: raise ValueError(f"uint16 token file has invalid byte length: {target}")
    values = np.memmap(target, mode="r", dtype=np.uint16)
    if values.size < 4: raise ValueError(f"Token stream is too short: {target}")
    return values

def _sample_batch(tokens, batch_size, seq_len, rng, device):
    max_start = len(tokens) - seq_len - 1
    if max_start <= 0: raise ValueError(f"Need > {seq_len + 1} tokens, found {len(tokens)}.")
    starts = rng.integers(0, max_start + 1, size=batch_size)
    samples = np.stack([np.asarray(tokens[s:s + seq_len + 1], dtype=np.int64) for s in starts])
    return (torch.as_tensor(samples[:, :-1], dtype=torch.long, device=device),
            torch.as_tensor(samples[:, 1:], dtype=torch.long, device=device))

def _evaluate(model, tokens, batch_size, seq_len, batches, device, seed):
    model.eval(); rng = np.random.default_rng(seed); losses = []
    with torch.inference_mode():
        for _ in range(batches):
            x, y = _sample_batch(tokens, batch_size, seq_len, rng, device)
            result = model(x)
            losses.append(float(F.cross_entropy(result["logits"].float().reshape(-1, model.config.vocab_size), y.reshape(-1)).item()))
    mean_loss = float(np.mean(losses))
    try: perplexity = math.exp(mean_loss)
    except OverflowError: perplexity = float("inf")
    model.train(); return {"eval_loss": mean_loss, "eval_perplexity": perplexity}

def _atomic_save(payload: dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    try: torch.save(payload, tmp); os.replace(tmp, path)
    finally: tmp.unlink(missing_ok=True)

def train_model(config, train_bin, valid_bin, output_dir, steps, batch_size=4, seq_len=512,
                grad_accum=8, learning_rate=3e-4, weight_decay=0.1, eval_every=250,
                save_every=250, eval_batches=20, resume=False, tokenizer_path=None, seed=None):
    config.validate()
    if min(steps, batch_size, grad_accum, eval_every, save_every, eval_batches) < 1: raise ValueError("Step, batch and interval values must be positive.")
    if not 2 <= seq_len <= config.max_context: raise ValueError("seq_len must be 2..max_context.")
    if learning_rate <= 0 or weight_decay < 0: raise ValueError("Invalid learning rate or weight decay.")
    train_tokens, valid_tokens = _load_tokens(train_bin), _load_tokens(valid_bin)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(output_dir); run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, metrics = run_dir / "latest.pt", run_dir / "metrics.jsonl"
    if checkpoint.exists() and not resume: raise FileExistsError(f"Checkpoint exists: {checkpoint}; pass --resume or choose a new run directory.")
    actual_seed = config.seed if seed is None else seed
    random.seed(actual_seed); rng = np.random.default_rng(actual_seed); torch.manual_seed(actual_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(actual_seed)
    model = QMiniMoE(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=weight_decay)
    start_step = 0
    if resume:
        if not checkpoint.is_file(): raise FileNotFoundError(f"No checkpoint to resume: {checkpoint}")
        try: state = torch.load(checkpoint, map_location=device, weights_only=False)
        except TypeError: state = torch.load(checkpoint, map_location=device)
        if QMiniConfig.from_dict(state["config"]).to_dict() != config.to_dict(): raise ValueError("Checkpoint config mismatch; refusing unsafe resume.")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); start_step = int(state["step"])
        if "numpy_rng" in state: rng.bit_generator.state = state["numpy_rng"]
        if "python_rng" in state: random.setstate(state["python_rng"])
        print(f"Resuming from optimizer step {start_step:,}.")
    amp = device.type == "cuda"
    bf16 = bool(amp and torch.cuda.is_bf16_supported()); fp16 = bool(amp and not bf16)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    print(f"Device={device}; AMP={'BF16' if bf16 else 'FP16' if fp16 else 'off'}; parameters={model.parameter_report()['total']:,}; global_batch={batch_size * grad_accum}")
    def lr_at(step):
        warmup = max(1, int(steps * 0.03))
        if step < warmup: return learning_rate * (step + 1) / warmup
        progress = min(1.0, max(0.0, (step - warmup) / max(1, steps - warmup)))
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    model.train()
    for step in range(start_step, steps):
        optimizer.zero_grad(set_to_none=True); accum = {"loss": 0.0, "ce": 0.0, "aux": 0.0}
        sampled_steps = random.randint(1, config.max_thinking_steps)
        for _ in range(grad_accum):
            x, y = _sample_batch(train_tokens, batch_size, seq_len, rng, device)
            dtype = torch.bfloat16 if bf16 else torch.float16
            ctx = torch.autocast(device_type="cuda", dtype=dtype) if amp else nullcontext()
            with ctx:
                result = model(x, thinking_steps=sampled_steps)
                ce = F.cross_entropy(result["logits"].float().reshape(-1, config.vocab_size), y.reshape(-1))
                aux = result["aux_loss"].float(); loss = ce + config.router_aux_loss_coef * aux
            scaled = loss / grad_accum
            if fp16: scaler.scale(scaled).backward()
            else: scaled.backward()
            accum["loss"] += float(loss.detach()) / grad_accum
            accum["ce"] += float(ce.detach()) / grad_accum
            accum["aux"] += float(aux.detach()) / grad_accum
        if fp16: scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = lr_at(step)
        for group in optimizer.param_groups: group["lr"] = lr
        if fp16: scaler.step(optimizer); scaler.update()
        else: optimizer.step()
        complete = step + 1
        record = {"step": complete, "train_loss": accum["loss"], "train_ce": accum["ce"],
                  "router_aux_loss": accum["aux"], "learning_rate": lr, "thinking_steps": sampled_steps}
        if complete % eval_every == 0 or complete == steps:
            record.update(_evaluate(model, valid_tokens, batch_size, seq_len, eval_batches, device, actual_seed + complete))
        if complete % save_every == 0 or complete == steps:
            _atomic_save({"format": "qmini-checkpoint-v1", "step": complete, "model": model.state_dict(),
                          "optimizer": optimizer.state_dict(), "config": config.to_dict(),
                          "numpy_rng": rng.bit_generator.state, "python_rng": random.getstate()}, checkpoint)
        with metrics.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record, allow_nan=False) + "\n")
        if complete == 1 or complete % 10 == 0 or "eval_loss" in record:
            msg = f"step {complete}/{steps} loss={accum['loss']:.4f} ce={accum['ce']:.4f} aux={accum['aux']:.4f} lr={lr:.2e}"
            if "eval_loss" in record: msg += f" eval={record['eval_loss']:.4f} ppl={record['eval_perplexity']:.2f}"
            print(msg, flush=True)
    final_dir = run_dir / "final_model"; final_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), final_dir / "model.pt"); config.save(final_dir / "config.json")
    if tokenizer_path and Path(tokenizer_path).is_file(): shutil.copy2(tokenizer_path, final_dir / "tokenizer.json")
    print(f"Training complete: {final_dir}"); return final_dir
