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
def test_native_projection_matches_reference(tmp_path):
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    library = tmp_path / "libqmini_test.so"
    subprocess.run([compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    native = ctypes.CDLL(str(library)).qmini_linear_i8
    p_i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
    p_u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    p_f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    native.argtypes = [p_i8, p_u8, p_f32, p_f32, p_f32, p_f32, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    native.restype = ctypes.c_int
    weight = torch.tensor([[-4., -1., 1., 4., 4., 1., -1., -4.], [1., 4., -4., -1., -1., -4., 4., 1.]])
    packed, scales = pack_quaternary(weight)
    dequantized = unpack_quaternary(packed, scales, 8)
    x = np.array([[.2, -.4, .7, 1., -.1, .3, .5, -.8]], dtype=np.float32)
    x_scale = np.maximum(np.max(np.abs(x), axis=1), 1e-8) / 127.
    x_q = np.rint(x / x_scale[:, None]).clip(-127, 127).astype(np.int8)
    bias = np.array([.25, -.5], dtype=np.float32)
    output = np.empty((1, 2), dtype=np.float32)
    status = native(x_q, packed, scales, x_scale.astype(np.float32), bias, output, 1, 8, 2)
    expected = x_q.astype(np.float32) @ dequantized.T * x_scale[:, None] + bias
    assert status == 0
    assert np.allclose(output, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("clang++")), reason="C++ compiler missing")
def test_native_wrapper_rejects_mismatched_shapes(tmp_path):
    source = Path(__file__).resolve().parents[1] / "qmini" / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    library = tmp_path / "libqmini_test.so"
    subprocess.run([compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    wrapper = NativeKernel(library)
    with pytest.raises(ValueError, match="Packed weight shape"):
        wrapper.linear(
            np.zeros((2, 8), dtype=np.int8),
            np.zeros((1, 1), dtype=np.uint8),
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )
