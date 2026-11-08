from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from .generate import generate
from .train import _load_masks, _load_tokens, _masked_cross_entropy, _sample_batch


def normalize_answer(text: str) -> str:
    """Normalize Unicode word content for deterministic exact-match scoring."""
    if not isinstance(text, str):
        raise TypeError("Evaluation answers must be strings.")
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def token_f1(prediction: str, reference: str) -> float:
    """Multiset word-level F1; a lexical diagnostic, not a semantic score."""
    predicted = re.findall(r"\w+", prediction.casefold(), flags=re.UNICODE)
    expected = re.findall(r"\w+", reference.casefold(), flags=re.UNICODE)
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)


def _task_records(path: str | Path) -> Iterator[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Task evaluation JSONL not found: {source}")
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid task JSON in {source} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"Task record at {source}:{line_number} must be a JSON object.")
            prompt = record.get("prompt")
            reference = record.get("reference", record.get("expected", record.get("answer")))
            category = record.get("category", record.get("task", "uncategorized"))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Task record at {source}:{line_number} needs a non-empty prompt.")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f"Task record at {source}:{line_number} needs reference/expected/answer text."
                )
            yield {
                "prompt": prompt,
                "reference": reference,
                "category": str(category).strip() or "uncategorized",
            }


def evaluate_language_model(
    model,
    token_path: str | Path,
    mask_path: str | Path | None = None,
    batch_size: int = 2,
    seq_len: int = 512,
    batches: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate held-out next-token loss, throughput and per-layer expert load."""
    if min(batch_size, seq_len, batches) < 1:
        raise ValueError("batch_size, seq_len and batches must be positive.")
    if seq_len > model.config.max_context:
        raise ValueError("seq_len exceeds the model context window.")

    tokens = _load_tokens(token_path, "Evaluation")
    masks = _load_masks(mask_path, tokens, "Evaluation")
    if len(tokens) < seq_len + 1:
        raise ValueError(
            f"Evaluation stream needs at least {seq_len + 1} tokens; found {len(tokens)}."
        )

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(seed)
    total_nll = 0.0
    target_tokens = 0
    input_tokens = 0
    router_totals: list[torch.Tensor] | None = None
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            for _ in range(batches):
                inputs, targets, mask = _sample_batch(
                    tokens, masks, batch_size, seq_len, rng, device
                )
                output = model(inputs)
                loss = _masked_cross_entropy(output["logits"], targets, mask)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Evaluation produced a non-finite loss.")
                count = int(mask.sum().item())
                total_nll += float(loss.item()) * count
                target_tokens += count
                input_tokens += int(inputs.numel())
                layer_counts = [
                    item.detach().to(device="cpu", dtype=torch.int64)
                    for item in output["router_counts"]
                ]
                if router_totals is None:
                    router_totals = layer_counts
                else:
                    for index, counts in enumerate(layer_counts):
                        router_totals[index] += counts
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    finally:
        model.train(was_training)

    elapsed = max(time.perf_counter() - started, 1e-9)
    if target_tokens == 0:
        raise ValueError("Evaluation did not include supervised target tokens.")
    mean_loss = total_nll / target_tokens
    perplexity = math.exp(mean_loss) if mean_loss < 709.0 else None

    expert_report: list[dict[str, Any]] = []
    for layer_index, counts in enumerate(router_totals or []):
        values = [int(value) for value in counts.tolist()]
        total = sum(values)
        expert_report.append({
            "block": layer_index,
            "tokens_per_expert": values,
            "fraction_per_expert": [value / total for value in values] if total else [],
            "active_experts": sum(value > 0 for value in values),
        })

    return {
        "format": "qmini-language-evaluation-v1",
        "model_parameters": model.parameter_report()["total"],
        "evaluation_stream": str(Path(token_path)),
        "loss_mask": str(Path(mask_path)) if mask_path is not None else None,
        "seed": int(seed),
        "batches": int(batches),
        "batch_size": int(batch_size),
        "sequence_length": int(seq_len),
        "input_tokens": int(input_tokens),
        "supervised_target_tokens": int(target_tokens),
        "cross_entropy": float(mean_loss),
        "perplexity": float(perplexity) if perplexity is not None else None,
        "elapsed_seconds": float(elapsed),
        "input_tokens_per_second": float(input_tokens / elapsed),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        "router_load": expert_report,
    }


def evaluate_task_file(
    model,
    tokenizer,
    dataset_path: str | Path,
    max_new_tokens: int = 128,
    limit: int | None = None,
    thinking_steps: int | None = None,
) -> dict[str, Any]:
    """Evaluate a user-supplied JSONL prompt/reference set with greedy decoding."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when supplied.")

    was_training = model.training
    model.eval()
    total = 0
    exact = 0
    f1_total = 0.0
    category_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "exact_match": 0.0, "token_f1": 0.0}
    )
    cases: list[dict[str, Any]] = []
    try:
        for task in _task_records(dataset_path):
            if limit is not None and total >= limit:
                break
            prediction = generate(
                model, tokenizer, task["prompt"],
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                thinking_steps=thinking_steps,
            )
            exact_match = normalize_answer(prediction) == normalize_answer(task["reference"])
            score = token_f1(prediction, task["reference"])
            total += 1
            exact += int(exact_match)
            f1_total += score
            category = category_stats[task["category"]]
            category["count"] += 1
            category["exact_match"] += int(exact_match)
            category["token_f1"] += score
            cases.append({
                "category": task["category"],
                "prompt": task["prompt"],
                "reference": task["reference"],
                "prediction": prediction,
                "exact_match": bool(exact_match),
                "token_f1": float(score),
            })
    finally:
        model.train(was_training)

    if total == 0:
        raise ValueError("Task evaluation file contained no usable records.")
    categories = {}
    for name, values in sorted(category_stats.items()):
        count = int(values["count"])
        categories[name] = {
            "examples": count,
            "exact_match": values["exact_match"] / count,
            "mean_token_f1": values["token_f1"] / count,
        }
    return {
        "format": "qmini-task-evaluation-v1",
        "model_parameters": model.parameter_report()["total"],
        "dataset": str(Path(dataset_path)),
        "examples": total,
        "exact_match": exact / total,
        "mean_token_f1": f1_total / total,
        "categories": categories,
        "cases": cases,
        "scoring_note": (
            "Exact match and lexical token F1 are diagnostic string metrics, not semantic "
            "correctness, factuality or reasoning guarantees. Use independently checked "
            "answers and task-specific executable tests for capability claims."
        ),
    }


def emit_report(report: dict[str, Any], output_path: str | Path | None = None) -> None:
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output_path is not None:
        target = Path(output_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.write("\n")
                handle.flush()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Evaluation report saved: {target}")
    print(serialized)
