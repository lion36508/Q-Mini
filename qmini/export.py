from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import QMiniConfig
from .quantization import pack_quaternary, unpack_quaternary


def export_packed_state(
    state_dict: dict[str, torch.Tensor],
    config: QMiniConfig,
    output_path: str | Path,
) -> Path:
    """Write a portable NPZ snapshot with packed 2-bit matrices and FP16 metadata tensors.

    Every tensor with rank >= 2 is row-wise quaternary packed. Biases and normalization
    vectors are kept as FP16. This is a storage/export format, not a native runtime.
    """
    config.validate()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    specs: list[dict[str, Any]] = []

    for index, (name, tensor) in enumerate(state_dict.items()):
        value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        if value.ndim >= 2:
            shape = list(value.shape)
            rows = shape[0]
            flat = torch.from_numpy(value.reshape(rows, -1))
            packed, scales = pack_quaternary(flat)
            packed_key, scale_key = f"p_{index}", f"s_{index}"
            arrays[packed_key] = packed
            arrays[scale_key] = scales
            specs.append({
                "name": name,
                "kind": "q2",
                "shape": shape,
                "packed_key": packed_key,
                "scale_key": scale_key,
                "in_features": int(flat.shape[1]),
            })
        else:
            value_key = f"f_{index}"
            arrays[value_key] = value.astype(np.float16)
            specs.append({"name": name, "kind": "fp16", "shape": list(value.shape), "value_key": value_key})

    metadata = {
        "format": "qmini-packed-v2",
        "q2_codebook_version": 2,
        "config": config.to_dict(),
        "tensor_count": len(specs),
        "quaternary_tensor_count": sum(item["kind"] == "q2" for item in specs),
        "fp16_tensor_count": sum(item["kind"] == "fp16" for item in specs),
        "tensors": specs,
        "quaternary_codes": {"00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0},
        "q2_scale_convention": "row_scale=2*mean(abs(original_row)); decoded_weight=codebook_value*row_scale",
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, separators=(",", ":")))
    temporary = target.with_name(target.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def reconstruct_packed_state(path: str | Path) -> tuple[QMiniConfig, dict[str, np.ndarray]]:
    """Reconstruct a Q2-v1 legacy or Q2-v2 packed snapshot."""
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        fmt = metadata.get("format")
        if fmt == "qmini-packed-v1":
            version = 1
        elif fmt == "qmini-packed-v2":
            if metadata.get("quaternary_codes") != {
                "00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0
            }:
                raise ValueError("Q2-v2 archive has an unexpected codebook.")
            version = 2
        else:
            raise ValueError("Unsupported Q-Mini packed export format.")
        config = QMiniConfig.from_dict(metadata["config"])
        state: dict[str, np.ndarray] = {}
        for spec in metadata["tensors"]:
            shape = tuple(int(x) for x in spec["shape"])
            if spec["kind"] == "q2":
                value = unpack_quaternary(
                    archive[spec["packed_key"]],
                    archive[spec["scale_key"]],
                    int(spec["in_features"]),
                    format_version=version,
                ).reshape(shape)
            elif spec["kind"] == "fp16":
                value = np.asarray(archive[spec["value_key"]], dtype=np.float32).reshape(shape)
            else:
                raise ValueError(f"Unknown tensor encoding: {spec['kind']!r}")
            if not np.isfinite(value).all():
                raise ValueError(f"Non-finite tensor found in packed archive: {spec['name']}")
            state[spec["name"]] = value
        if len(state) != int(metadata["tensor_count"]):
            raise ValueError("Packed archive tensor count does not match its metadata.")
    return config, state


def verify_packed_export(path: str | Path) -> dict[str, int | str | float]:
    """Validate packed arrays one tensor at a time without expanding a 5B model."""
    target = Path(path)
    packed_weight_bytes = 0
    scale_bytes = 0
    vector_bytes = 0
    parameter_count = 0
    q2_count = 0
    fp16_count = 0

    with np.load(target, allow_pickle=False) as archive:
        if "metadata_json" not in archive:
            raise ValueError("Packed archive is missing metadata_json.")
        metadata = json.loads(str(archive["metadata_json"].item()))
        fmt = metadata.get("format")
        if fmt not in {"qmini-packed-v1", "qmini-packed-v2"}:
            raise ValueError("Unsupported Q-Mini packed export format.")
        if fmt == "qmini-packed-v2" and metadata.get("quaternary_codes") != {
            "00": -1.0, "01": -0.5, "10": 0.5, "11": 1.0
        }:
            raise ValueError("Q2-v2 archive has an unexpected codebook.")
        config = QMiniConfig.from_dict(metadata["config"])
        specs = metadata.get("tensors")
        if not isinstance(specs, list) or len(specs) != int(metadata.get("tensor_count", -1)):
            raise ValueError("Packed archive tensor manifest count is invalid.")

        seen: set[str] = set()
        for spec in specs:
            name = str(spec.get("name", ""))
            if not name or name in seen:
                raise ValueError(f"Missing or duplicate tensor name in packed archive: {name!r}")
            seen.add(name)
            shape = tuple(int(value) for value in spec.get("shape", []))
            if not shape or any(value < 1 for value in shape):
                raise ValueError(f"Invalid shape in packed archive for {name}.")
            parameter_count += math.prod(shape)

            if spec.get("kind") == "q2":
                if len(shape) < 2:
                    raise ValueError(f"Q2 tensor {name} has rank less than two.")
                in_features = math.prod(shape[1:])
                if int(spec.get("in_features", -1)) != in_features:
                    raise ValueError(f"Q2 input dimension mismatch for {name}.")
                packed = np.asarray(archive[spec["packed_key"]])
                scales = np.asarray(archive[spec["scale_key"]])
                if packed.dtype != np.uint8 or packed.shape != (
                    shape[0], (in_features + 3) // 4
                ):
                    raise ValueError(f"Invalid packed Q2 payload for {name}.")
                if scales.dtype != np.float32 or scales.shape != (shape[0],):
                    raise ValueError(f"Invalid Q2 row scales for {name}.")
                if not np.isfinite(scales).all() or np.any(scales < 0):
                    raise ValueError(f"Invalid or non-finite Q2 scales for {name}.")
                packed_weight_bytes += int(packed.nbytes)
                scale_bytes += int(scales.nbytes)
                q2_count += 1
            elif spec.get("kind") == "fp16":
                value = np.asarray(archive[spec["value_key"]])
                if len(shape) >= 2 or value.dtype != np.float16 or value.shape != shape:
                    raise ValueError(f"Invalid FP16 vector payload for {name}.")
                if not np.isfinite(value).all():
                    raise ValueError(f"Non-finite FP16 parameter found in {name}.")
                vector_bytes += int(value.nbytes)
                fp16_count += 1
            else:
                raise ValueError(f"Unknown tensor encoding in packed archive: {name}.")

    if parameter_count != config.estimated_parameter_count():
        raise ValueError(
            f"Packed archive has {parameter_count:,} parameters; config expects "
            f"{config.estimated_parameter_count():,}."
        )
    return {
        "format": fmt,
        "tensor_count": len(seen),
        "parameter_count": parameter_count,
        "q2_tensor_count": q2_count,
        "fp16_tensor_count": fp16_count,
        "packed_weight_bytes": packed_weight_bytes,
        "row_scale_bytes": scale_bytes,
        "fp16_vector_bytes": vector_bytes,
        "decoded_payload_bytes": packed_weight_bytes + scale_bytes + vector_bytes,
        "artifact_bytes": target.stat().st_size,
        "estimated_raw_layout_bytes": config.estimated_packed_export_bytes(),
        "context": config.max_context,
    }
