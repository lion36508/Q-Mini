import ctypes, shutil, subprocess
from pathlib import Path
import numpy as np
import pytest, torch
from qmini.quantization import pack_quaternary, unpack_quaternary

@pytest.mark.skipif(not (shutil.which("g++") or shutil.which("clang++")), reason="C++ compiler unavailable")
def test_native_kernel_matches_reference(tmp_path):
    root = Path(__file__).resolve().parents[1]; source = root / "qmini" / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++"); library = tmp_path / "libqmini_test.so"
    subprocess.run([compiler, "-O2", "-std=c++17", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
    fn = ctypes.CDLL(str(library)).qmini_linear_i8
    i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS"); u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    fn.argtypes = [i8, u8, f32, f32, f32, f32, ctypes.c_int, ctypes.c_int, ctypes.c_int]; fn.restype = ctypes.c_int
    weight = torch.tensor([[-4., -1., 1., 4., 4., 1., -1., -4.], [1., 4., -4., -1., -1., -4., 4., 1.]])
    packed, scales = pack_quaternary(weight); deq = unpack_quaternary(packed, scales, 8)
    x = np.array([[.2, -.4, .7, 1., -.1, .3, .5, -.8]], np.float32)
    xs = np.maximum(np.max(np.abs(x), axis=1), 1e-8) / 127.; xq = np.rint(x / xs[:, None]).clip(-127, 127).astype(np.int8)
    bias = np.array([.25, -.5], np.float32); out = np.empty((1, 2), np.float32)
    status = fn(xq, packed, scales, xs.astype(np.float32), bias, out, 1, 8, 2)
    expected = xq.astype(np.float32) @ deq.T * xs[:, None] + bias
    assert status == 0 and np.allclose(out, expected, atol=1e-5, rtol=1e-5)
