import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from qmini.native import NativeKernel
from qmini.quantization import pack_quaternary, unpack_quaternary


@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("clang++")), reason="C++ compiler missing")
def test_native_projection_matches_reference_on_odd_dimensions(tmp_path):
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    library = tmp_path / "libqmini_test.so"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )

    weight = torch.tensor([
        [-4., -1., 1., 4., 4., 1., -1.],
        [1., 4., -4., -1., -1., -4., 4.],
        [2., -3., 1., -2., 4., -1., 3.],
    ])
    packed, scales = pack_quaternary(weight)
    dequantized = unpack_quaternary(packed, scales, in_features=7)
    x = np.array([
        [.2, -.4, .7, 1., -.1, .3, .5],
        [-.9, .1, .4, -.2, .7, -.5, .3],
    ], dtype=np.float32)
    x_scale = np.maximum(np.max(np.abs(x), axis=1), 1e-8) / 127.
    x_q = np.rint(x / x_scale[:, None]).clip(-127, 127).astype(np.int8)
    bias = np.array([.25, -.5, .1], dtype=np.float32)

    output = NativeKernel(library).linear(x_q, packed, scales, x_scale, bias)
    expected = x_q.astype(np.float32) @ dequantized.T
    expected = expected * x_scale[:, None] + bias
    assert output.shape == (2, 3)
    assert np.allclose(output, expected, atol=1e-5, rtol=1e-5)


def test_native_wrapper_rejects_incompatible_shapes(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    library = tmp_path / "libqmini_test.so"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    native = NativeKernel(library)
    with pytest.raises(ValueError, match="one scale per input row"):
        native.linear(
            np.ones((2, 4), dtype=np.int8),
            np.zeros((3, 1), dtype=np.uint8),
            np.ones(3, dtype=np.float32),
            np.ones(1, dtype=np.float32),
        )


@pytest.fixture(scope="module")
def compiled_native_library(tmp_path_factory):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    library = tmp_path_factory.mktemp("native-runtime") / "libqmini_test.so"
    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    return library


def test_qlinear_uses_native_kernel_and_invalidates_packed_cache(
    compiled_native_library, monkeypatch
):
    import torch.nn.functional as F
    import qmini.native as native_module
    from qmini.model import QLinear
    from qmini.quantization import fake_quantize_int8, quantize_quaternary

    kernel = NativeKernel(compiled_native_library)
    calls = {"count": 0}
    original = kernel.linear

    def counted_linear(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    kernel.linear = counted_linear
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: kernel)

    layer = QLinear(7, 5).eval()
    values = torch.randn(2, 3, 7, dtype=torch.float32)
    with torch.inference_mode():
        actual = layer(values)
        expected = F.linear(
            fake_quantize_int8(values),
            quantize_quaternary(layer.weight),
            layer.bias,
        )
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
        assert calls["count"] == 1
        first_key = layer._qmini_native_packed_key

        # A changed master weight must regenerate the packed row data.
        layer.weight.add_(0.05)
        actual_after_update = layer(values)
        expected_after_update = F.linear(
            fake_quantize_int8(values),
            quantize_quaternary(layer.weight),
            layer.bias,
        )
        assert torch.allclose(
            actual_after_update, expected_after_update, atol=2e-5, rtol=2e-5
        )
        assert calls["count"] == 2
        assert layer._qmini_native_packed_key != first_key


def test_tied_embedding_output_uses_native_projection(
    compiled_native_library, monkeypatch
):
    import torch.nn.functional as F
    import qmini.native as native_module
    from qmini.model import QEmbedding
    from qmini.quantization import fake_quantize_int8, quantize_quaternary

    kernel = NativeKernel(compiled_native_library)
    monkeypatch.setattr(native_module, "get_native_kernel", lambda: kernel)
    embedding = QEmbedding(37, 9).eval()
    hidden = torch.randn(2, 4, 9)
    with torch.inference_mode():
        actual = embedding.project(hidden)
        expected = F.linear(
            fake_quantize_int8(hidden),
            quantize_quaternary(embedding.weight),
        )
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert hasattr(embedding, "_qmini_native_packed_weight")


def test_pytorch_fallback_is_used_without_native_library(monkeypatch):
    import torch.nn.functional as F
    import qmini.native as native_module
    from qmini.model import QLinear
    from qmini.quantization import fake_quantize_int8, quantize_quaternary

    monkeypatch.setattr(native_module, "get_native_kernel", lambda: None)
    layer = QLinear(4, 3).eval()
    values = torch.randn(2, 4)
    with torch.inference_mode():
        actual = layer(values)
        expected = F.linear(
            fake_quantize_int8(values), quantize_quaternary(layer.weight), layer.bias
        )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert not hasattr(layer, "_qmini_native_packed_weight")


def test_native_loader_retries_after_library_is_built(
    tmp_path, monkeypatch
):
    import qmini.native as native_module

    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler missing")
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    library = tmp_path / "late-built.so"

    monkeypatch.setenv("QMINI_NATIVE_LIBRARY", str(library))
    monkeypatch.setattr(native_module, "_NATIVE_KERNEL", None)
    monkeypatch.setattr(native_module, "_NATIVE_CACHE_KEY", None)
    assert native_module.get_native_kernel() is None

    subprocess.run(
        [compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
    )
    assert native_module.get_native_kernel() is not None
    assert native_module.native_kernel_status()["available"] is True
