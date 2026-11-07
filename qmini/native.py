from __future__ import annotations

import argparse
import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np


def build_kernel(output=None):
    source = Path(__file__).resolve().parent / "native_kernel.cpp"
    compiler = shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("Install clang++ or g++ to build the native kernel.")
    target = Path(output) if output else source.with_name("libqmini_kernel.so")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    subprocess.run(
        [compiler, "-O3", "-DNDEBUG", "-std=c++17", "-fPIC", "-shared",
         str(source), "-o", str(temporary)],
        check=True,
    )
    temporary.replace(target)
    return target


class NativeKernel:
    def __init__(self, library_path):
        self.library = ctypes.CDLL(str(library_path))
        self.function = self.library.qmini_linear_i8
        i8 = np.ctypeslib.ndpointer(dtype=np.int8, flags="C_CONTIGUOUS")
        u8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
        self.function.argtypes = [
            i8, u8, f32, f32, f32, f32,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.function.restype = ctypes.c_int

    def linear(self, activations_i8, packed_weights, weight_scales, activation_scales, bias=None):
        x = np.ascontiguousarray(activations_i8, dtype=np.int8)
        packed = np.ascontiguousarray(packed_weights, dtype=np.uint8)
        ws = np.ascontiguousarray(weight_scales, dtype=np.float32)
        xs = np.ascontiguousarray(activation_scales, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("activations_i8 must have shape [rows, in_features].")
        if packed.ndim != 2 or ws.ndim != 1 or xs.ndim != 1:
            raise ValueError("Packed weights and scales have invalid ranks.")
        if packed.shape[0] != ws.size or packed.shape[1] * 4 < x.shape[1]:
            raise ValueError("Packed weight shape does not match activation dimensions.")
        if xs.shape != (x.shape[0],):
            raise ValueError("activation_scales must contain one scale per input row.")
        if not np.isfinite(ws).all() or np.any(ws <= 0):
            raise ValueError("weight_scales must be finite and positive.")
        if not np.isfinite(xs).all() or np.any(xs <= 0):
            raise ValueError("activation_scales must be finite and positive.")
        if bias is None:
            b = np.zeros(ws.size, dtype=np.float32)
        else:
            b = np.ascontiguousarray(bias, dtype=np.float32)
            if b.shape != (ws.size,):
                raise ValueError("bias must contain one value per output feature.")
            if not np.isfinite(b).all():
                raise ValueError("bias values must be finite.")
        output = np.empty((x.shape[0], ws.size), dtype=np.float32)
        status = self.function(
            x, packed, ws, xs, b, output, x.shape[0], x.shape[1], ws.size
        )
        if status:
            raise RuntimeError(f"Native kernel returned error {status}.")
        return output


def main():
    parser = argparse.ArgumentParser(description="Build the optional Q-Mini C++ projection primitive.")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.build:
        print(build_kernel(args.output))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
