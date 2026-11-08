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
