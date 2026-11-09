from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from qmini.config import QMiniConfig
from qmini.quantization import (
    pack_int4_groupwise,
    pack_quaternary,
    pack_ternary,
    quantization_error_metrics,
    unpack_int4_groupwise,
    unpack_quaternary,
    unpack_ternary,
    quantize_int8_rowwise,
    dequantize_int8_rowwise,
)


def _metric_accumulator() -> dict[str, float]:
    return {"reference_sq": 0.0, "error_sq": 0.0, "dot": 0.0, "approx_sq": 0.0, "max_abs": 0.0}


def _add_metrics(total: dict[str, float], reference: torch.Tensor, approximation: torch.Tensor) -> None:
    ref = reference.detach().float().reshape(-1)
    approx = approximation.detach().float().reshape(-1)
    difference = approx - ref
    total["reference_sq"] += float(torch.sum(ref * ref))
    total["error_sq"] += float(torch.sum(difference * difference))
    total["dot"] += float(torch.sum(ref * approx))
    total["approx_sq"] += float(torch.sum(approx * approx))
    total["max_abs"] = max(total["max_abs"], float(difference.abs().max()))


def _finish_metrics(total: dict[str, float]) -> dict[str, float]:
    relative_l2 = math.sqrt(total["error_sq"]) / max(math.sqrt(total["reference_sq"]), 1e-12)
    cosine = total["dot"] / max(
        math.sqrt(total["reference_sq"] * total["approx_sq"]), 1e-12
    )
    return {
        "relative_l2_error": relative_l2,
        "cosine_similarity": max(-1.0, min(1.0, cosine)),
        "max_absolute_error": total["max_abs"],
    }


def run_ablation(
    model_dir: Path,
    output_path: Path,
    group_size: int,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Compare formats using a Q-Mini final snapshot or own training checkpoint."""
    source_path = Path(model_dir).expanduser()
    if checkpoint_path is not None:
        weight_path = Path(checkpoint_path).expanduser()
    elif source_path.is_file():
        weight_path = source_path
    elif (source_path / "model.pt").is_file():
        weight_path = source_path / "model.pt"
    elif (source_path / "latest.pt").is_file():
        weight_path = source_path / "latest.pt"
    else:
        raise FileNotFoundError(
            "Provide a Q-Mini final_model directory with model.pt, a training run "
            "directory with latest.pt, or an explicit --checkpoint. External/pretrained "
            "weights are not accepted."
        )

    loaded = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError("Selected Q-Mini weight source is not a non-empty dictionary.")

    if loaded.get("format") in {"qmini-checkpoint-v1", "qmini-checkpoint-v2"}:
        if not isinstance(loaded.get("model"), dict) or not isinstance(loaded.get("config"), dict):
            raise ValueError("Q-Mini training checkpoint is missing its model/config state.")
        state = loaded["model"]
        config = QMiniConfig.from_dict(loaded["config"])
        source_kind = "qmini-training-checkpoint"
    else:
        config_path = (
            source_path / "config.json"
            if source_path.is_dir()
            else weight_path.parent / "config.json"
        )
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Config file not found beside model weights: {config_path}"
            )
        config = QMiniConfig.from_json(config_path)
        state = loaded
        source_kind = "qmini-final-model-state-dict"

    if not isinstance(state, dict) or not state:
        raise ValueError("Q-Mini model state is not a non-empty tensor dictionary.")

    formats = ("fp16", "int8", "q2", "ternary", "int4")
    totals = {name: _metric_accumulator() for name in formats}
    byte_counts = {name: 0 for name in formats}
    per_tensor: dict[str, Any] = {}
    matrix_count = 0
    vector_count = 0

    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.is_floating_point(value):
            continue
        reference = value.detach().float().cpu()
        if reference.ndim < 2:
            vector_count += 1
            approximation = reference.half().float()
            _add_metrics(totals["fp16"], reference, approximation)
            per_tensor[name] = {
                "shape": list(reference.shape),
                "kind": "fp16-vector",
                "fp16": quantization_error_metrics(reference, approximation),
            }
            for format_name in ("int8", "q2", "ternary", "int4"):
                _add_metrics(totals[format_name], reference, approximation)
            vector_bytes = reference.numel() * 2
            for format_name in formats:
                byte_counts[format_name] += vector_bytes
            continue

        matrix_count += 1
        flat = reference.reshape(reference.shape[0], -1)
        row_report: dict[str, Any] = {
            "shape": list(reference.shape),
            "flattened_matrix_shape": list(flat.shape),
            "kind": "quantized-matrix",
            "formats": {},
        }

        fp16 = flat.half().float()
        _add_metrics(totals["fp16"], flat, fp16)
        byte_counts["fp16"] += flat.numel() * 2
        row_report["formats"]["fp16"] = {
            **quantization_error_metrics(flat, fp16),
            "bytes": flat.numel() * 2,
        }

        int8_values, int8_scales = quantize_int8_rowwise(flat)
        int8 = dequantize_int8_rowwise(int8_values, int8_scales)
        int8_bytes = int(int8_values.numel() + int8_scales.numel() * 4)
        _add_metrics(totals["int8"], flat, int8)
        byte_counts["int8"] += int8_bytes
        row_report["formats"]["int8"] = {
            **quantization_error_metrics(flat, int8),
            "bytes": int8_bytes,
        }

        q2_packed, q2_scales = pack_quaternary(flat)
        q2 = torch.from_numpy(
            unpack_quaternary(q2_packed, q2_scales, flat.shape[1])
        )
        q2_bytes = int(q2_packed.nbytes + q2_scales.nbytes)
        _add_metrics(totals["q2"], flat, q2)
        byte_counts["q2"] += q2_bytes
        row_report["formats"]["q2"] = {
            **quantization_error_metrics(flat, q2),
            "bytes": q2_bytes,
        }

        ternary_packed, ternary_scales = pack_ternary(flat)
        ternary = torch.from_numpy(
            unpack_ternary(ternary_packed, ternary_scales, flat.shape[1])
        )
        ternary_bytes = int(ternary_packed.nbytes + ternary_scales.nbytes)
        _add_metrics(totals["ternary"], flat, ternary)
        byte_counts["ternary"] += ternary_bytes
        row_report["formats"]["ternary"] = {
            **quantization_error_metrics(flat, ternary),
            "bytes": ternary_bytes,
        }

        int4_packed, int4_scales = pack_int4_groupwise(flat, group_size=group_size)
        int4 = torch.from_numpy(
            unpack_int4_groupwise(
                int4_packed, int4_scales, flat.shape[1], group_size=group_size
            )
        )
        int4_bytes = int(int4_packed.nbytes + int4_scales.nbytes)
        _add_metrics(totals["int4"], flat, int4)
        byte_counts["int4"] += int4_bytes
        row_report["formats"]["int4"] = {
            **quantization_error_metrics(flat, int4),
            "bytes": int4_bytes,
            "group_size": group_size,
        }
        per_tensor[name] = row_report

    if matrix_count == 0:
        raise ValueError("No matrix weights were found in this model state dictionary.")

    report = {
        "format": "qmini-weight-quantization-ablation-v1",
        "source_model_dir": str(source_path.resolve()),
        "source_weight_path": str(weight_path.resolve()),
        "source_kind": source_kind,
        "source_config": config.to_dict(),
        "source_parameter_count": config.estimated_parameter_count(),
        "matrix_tensor_count": matrix_count,
        "vector_tensor_count": vector_count,
        "group_size": group_size,
        "formats": {
            name: {
                **_finish_metrics(totals[name]),
                "estimated_weight_payload_bytes": byte_counts[name],
            }
            for name in formats
        },
        "tensor_results": per_tensor,
        "limitations": [
            "This report measures weight reconstruction error only.",
            "It does not measure validation NLL, assistant quality, runtime speed or peak RSS.",
            "All formats are experimental; this script does not change the model's default quantizer.",
            "Only Q-Mini-owned final_model/model.pt or its own latest.pt training checkpoint is accepted; no external pretrained weights are loaded.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare FP16, rowwise INT8, Q2, zero-preserving ternary and groupwise INT4 weight error."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", help="Q-Mini final_model directory or training run directory")
    source.add_argument("--checkpoint", help="Q-Mini-owned latest.pt training checkpoint")
    parser.add_argument("--output", default="quantization-ablation.json")
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    if args.group_size < 1:
        parser.error("--group-size must be positive.")
    model_source = (
        Path(args.model_dir).expanduser()
        if args.model_dir is not None
        else Path(args.checkpoint).expanduser()
    )
    report = run_ablation(
        model_source,
        Path(args.output).expanduser(),
        args.group_size,
    )
    for name, result in report["formats"].items():
        print(
            f"{name}: rel_l2={result['relative_l2_error']:.6f} "
            f"cosine={result['cosine_similarity']:.6f} "
            f"bytes={result['estimated_weight_payload_bytes']:,}"
        )
    print(f"Report saved: {Path(args.output).expanduser()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
