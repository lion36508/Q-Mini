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


def run_ablation(model_dir: Path, output_path: Path, group_size: int) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "model.pt"
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "This weight-only ablation requires a Q-Mini final_model directory "
            "containing config.json and model.pt. Do not use external/pretrained weights."
        )

    config = QMiniConfig.from_json(config_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("model.pt is not a non-empty tensor state dictionary.")

    formats = ("fp16", "q2", "ternary", "int4")
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
            for format_name in ("q2", "ternary", "int4"):
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
        "source_model_dir": str(model_dir.resolve()),
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
            "Only Q-Mini's own model.pt checkpoint is accepted; no external pretrained weights are loaded.",
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
        description="Compare FP16, Q2, zero-preserving ternary and groupwise INT4 weight error."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="quantization-ablation.json")
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    if args.group_size < 1:
        parser.error("--group-size must be positive.")
    report = run_ablation(
        Path(args.model_dir).expanduser(),
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
