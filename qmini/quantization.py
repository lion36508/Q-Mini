from __future__ import annotations

import numpy as np
import torch


def quantize_quaternary(weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise {-2,-1,+1,+2} fake quantization with straight-through gradients."""
    if weight.ndim < 2:
        raise ValueError("Weight tensor must have at least two dimensions.")
    flat = weight.reshape(weight.shape[0], -1)
    scale = flat.detach().abs().mean(dim=-1, keepdim=True).clamp_min(eps)
    magnitude = torch.where(flat.detach().abs() / scale >= 1.5, 2.0, 1.0)
    sign = torch.where(flat.detach() < 0, -1.0, 1.0)
    q = (sign * magnitude * scale).reshape_as(weight)
    return weight + (q - weight).detach()


def quantize_int8_per_token(
    activation: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed INT8 values and FP32 per-token AbsMax scales for native inference."""
    if activation.ndim < 1:
        raise ValueError("Activation tensor must have a feature dimension.")
    flat = activation.detach().reshape(-1, activation.shape[-1]).float()
    scale = flat.abs().amax(dim=-1).clamp_min(eps) / 127.0
    quantized = torch.round(flat / scale.unsqueeze(1)).clamp(-127, 127)
    return quantized.to(torch.int8), scale.to(torch.float32)


def fake_quantize_int8(activation: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-token AbsMax signed INT8 fake quantization with STE."""
    if activation.ndim < 1:
        raise ValueError("Activation tensor must have a feature dimension.")
    scale = activation.detach().abs().amax(dim=-1, keepdim=True).clamp_min(eps) / 127.0
    q = torch.round(activation / scale).clamp(-127, 127) * scale
    return activation + (q - activation).detach()


def pack_quaternary(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Pack four 2-bit signed quaternary values per byte; return row scales."""
    if weight.ndim != 2:
        raise ValueError("Expected a [out_features, in_features] matrix.")
    values = weight.detach().float().cpu().numpy()
    scales = np.maximum(np.mean(np.abs(values), axis=1), 1e-8).astype(np.float32)
    magnitude = np.where(np.abs(values) / scales[:, None] >= 1.5, 2, 1)
    quantized = np.where(values < 0, -magnitude, magnitude).astype(np.int8)
    codes = np.where(
        quantized == -2, 0,
        np.where(quantized == -1, 1, np.where(quantized == 1, 2, 3)),
    ).astype(np.uint8)
    rows, columns = codes.shape
    padded = np.zeros((rows, ((columns + 3) // 4) * 4), dtype=np.uint8)
    padded[:, :columns] = codes
    packed = (
        padded[:, 0::4]
        | (padded[:, 1::4] << 2)
        | (padded[:, 2::4] << 4)
        | (padded[:, 3::4] << 6)
    )
    return packed, scales


def unpack_quaternary(
    packed: np.ndarray,
    scales: np.ndarray,
    in_features: int,
) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float32)
    if packed.ndim != 2 or scales.ndim != 1 or packed.shape[0] != scales.shape[0]:
        raise ValueError("Packed matrix and scale dimensions do not match.")
    if in_features < 1 or packed.shape[1] * 4 < in_features:
        raise ValueError("Invalid original input dimension for packed weights.")
    codes = np.empty((packed.shape[0], packed.shape[1] * 4), dtype=np.uint8)
    for offset in range(4):
        codes[:, offset::4] = (packed >> (2 * offset)) & 3
    values = np.select(
        [codes == 0, codes == 1, codes == 2, codes == 3],
        [-2.0, -1.0, 1.0, 2.0],
    ).astype(np.float32)
    return values[:, :in_features] * scales[:, None]
