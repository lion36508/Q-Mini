from __future__ import annotations
import argparse, ctypes, shutil, subprocess
from pathlib import Path
import numpy as np

def build_kernel(output=None):
    source = Path(__file__).resolve().parent / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler: raise RuntimeError("Install clang++ or g++ to build the native kernel.")
    target = Path(output) if output else source.with_name("libqmini_kernel.so")
    target.parent.mkdir(parents=True, exist_ok=True); temp = target.with_name(target.name + ".tmp")
    subprocess.run([compiler, "-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-shared", str(source), "-o", str(temp)], check=True)
    temp.replace(target); return target

class NativeKernel:
    def __init__(self, library_path):
        self.library = ctypes.CDLL(str(library_path)); self.function = self.library.qmini_linear_i8
        i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.function.argtypes = [i8, u8, f32, f32, f32, f32, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.function.restype = ctypes.c_int
    def linear(self, activations_i8, packed_weights, weight_scales, activation_scales, bias=None):
        x = np.ascontiguousarray(activations_i8, dtype=np.int8); w = np.ascontiguousarray(packed_weights, dtype=np.uint8)
        ws = np.ascontiguousarray(weight_scales, dtype=np.float32); xs = np.ascontiguousarray(activation_scales, dtype=np.float32)
        b = np.ascontiguousarray(bias, dtype=np.float32) if bias is not None else np.zeros(ws.size, np.float32)
        out = np.empty((x.shape[0], ws.size), np.float32)
        status = self.function(x, w, ws, xs, b, out, x.shape[0], x.shape[1], ws.size)
        if status: raise RuntimeError(f"Native kernel returned error {status}.")
        return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--build", action="store_true"); p.add_argument("--output")
    a = p.parse_args()
    if a.build: print(build_kernel(a.output))
    else: p.print_help()
if __name__ == "__main__": main()
