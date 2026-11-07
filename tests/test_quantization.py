import numpy as np
import torch
from qmini.quantization import fake_quantize_int8, pack_quaternary, quantize_quaternary, unpack_quaternary

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
