import pytest
import numpy as np
import torch
from qmini.quantization import (
    fake_quantize_int8, pack_quaternary, pack_ternary,
    pack_int4_groupwise, quantization_error_metrics,
    quantize_quaternary, quantize_ternary_rowwise,
    unpack_quaternary, unpack_ternary, unpack_int4_groupwise,
)

def test_packed_quaternary_roundtrip():
    w = torch.tensor([[-4., -1., 1., 4., 2.], [4., 1., -1., -4., -2.]])
    packed, scales = pack_quaternary(w); restored = unpack_quaternary(packed, scales, 5)
    expected = quantize_quaternary(w).detach().numpy()
    assert np.allclose(restored, expected, atol=1e-6)

def test_quaternary_ste_gradient():
    w = torch.randn(3, 7, requires_grad=True); quantize_quaternary(w).square().mean().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0

def test_int8_ste_gradient():
    x = torch.randn(2, 5, 16, requires_grad=True); y = fake_quantize_int8(x)
    assert torch.isfinite(y).all(); y.sum().backward()
    assert x.grad is not None and torch.all(x.grad == 1)


def test_ternary_quantization_preserves_exact_zeros_and_round_trips_odd_width():
    weight = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -0.01, 0.0, 0.01, 0.0, -0.01, 0.0],
        [0.0, -2.0, 0.0, 1.0, 0.0, -0.5, 0.0],
    ])
    quantized, scales = quantize_ternary_rowwise(weight)
    assert set(quantized.unique().tolist()).issubset({-1, 0, 1})
    assert torch.all(quantized[weight == 0] == 0)
    assert torch.all(quantized[0] == 0)
    packed, packed_scales = pack_ternary(weight)
    assert packed.dtype == np.uint8
    assert packed_scales.dtype == np.float32
    restored = unpack_ternary(packed, packed_scales, in_features=7)
    expected = quantized.numpy().astype(np.float32) * scales.numpy()[:, None]
    assert np.array_equal(restored, expected)
    assert np.all(restored[weight.numpy() == 0] == 0)


def test_groupwise_int4_quantization_round_trips_odd_width_and_zero_rows():
    weight = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-3.0, -2.0, -1.0, 0.0, 0.5, 1.5, 2.5],
        [7.0, -6.0, 5.0, -4.0, 3.0, -2.0, 1.0],
    ])
    packed, scales = pack_int4_groupwise(weight, group_size=4)
    assert packed.dtype == np.uint8
    assert scales.shape == (3, 2)
    restored = unpack_int4_groupwise(
        packed, scales, in_features=7, group_size=4
    )
    assert restored.shape == tuple(weight.shape)
    assert np.array_equal(restored[0], np.zeros(7, dtype=np.float32))
    assert np.max(np.abs(restored - weight.numpy())) <= float(scales.max()) / 2 + 1e-6


def test_quantization_error_metrics_report_weight_error_not_quality():
    reference = torch.tensor([[1.0, -2.0, 0.0]])
    approximation = torch.tensor([[1.0, -1.5, 0.0]])
    report = quantization_error_metrics(reference, approximation)
    assert report["relative_l2_error"] == pytest.approx(0.5 / np.sqrt(5))
    assert 0.0 < report["cosine_similarity"] <= 1.0
    assert report["max_absolute_error"] == pytest.approx(0.5)


def test_experimental_quantizers_reject_invalid_shapes_and_nonfinite_values():
    from qmini.quantization import pack_ternary, pack_int4_groupwise

    with pytest.raises(ValueError, match="at least two dimensions"):
        quantize_ternary_rowwise(torch.ones(5))
    with pytest.raises(ValueError, match="group_size"):
        pack_int4_groupwise(torch.ones(2, 5), group_size=0)
    with pytest.raises(ValueError, match="non-finite"):
        pack_ternary(torch.tensor([[float("nan"), 1.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        pack_int4_groupwise(torch.tensor([[float("inf"), 1.0]]))


def test_q2_codebook_has_only_four_requested_values_and_zero_rows_round_trip():
    weight = torch.tensor([
        [-4.0, -1.0, 0.0, 1.0, 4.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    quantized = quantize_quaternary(weight)
    scales = 2.0 * weight.abs().mean(dim=1, keepdim=True)
    assert torch.equal(quantized[1], torch.zeros_like(quantized[1]))
    normalized = quantized[0] / scales[0]
    assert set(normalized.tolist()).issubset({-1.0, -0.5, 0.5, 1.0})

    packed, packed_scales = pack_quaternary(weight)
    assert packed_scales[1] == 0.0
    restored = unpack_quaternary(packed, packed_scales, in_features=5)
    assert np.allclose(restored, quantized.numpy(), atol=1e-6)


def test_legacy_q2_v1_unpack_remains_compatible():
    packed = np.array([[0b11100100]], dtype=np.uint8)
    scales = np.array([2.0], dtype=np.float32)
    legacy = unpack_quaternary(packed, scales, in_features=4, format_version=1)
    current = unpack_quaternary(packed, scales, in_features=4, format_version=2)
    assert np.array_equal(legacy, np.array([[-4.0, -2.0, 2.0, 4.0]], dtype=np.float32))
    assert np.array_equal(current, np.array([[-2.0, -1.0, 1.0, 2.0]], dtype=np.float32))
