from __future__ import annotations

import json
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
        "format": "qmini-packed-v1",
        "config": config.to_dict(),
        "tensor_count": len(specs),
        "quaternary_tensor_count": sum(item["kind"] == "q2" for item in specs),
        "fp16_tensor_count": sum(item["kind"] == "fp16" for item in specs),
        "tensors": specs,
        "quaternary_codes": {"00": -2, "01": -1, "10": 1, "11": 2},
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
    """Reconstruct dequantized FP32 arrays from a packed Q-Mini archive."""
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("format") != "qmini-packed-v1":
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


def verify_packed_export(path: str | Path) -> dict[str, int | str]:
    config, state = reconstruct_packed_state(path)
    return {
        "format": "qmini-packed-v1",
        "tensor_count": len(state),
        "parameter_count": sum(int(value.size) for value in state.values()),
        "context": config.max_context,
    }
