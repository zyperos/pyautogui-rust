"""Compile TinyLocate CUDA kernels to distributable PTX using NVRTC."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path


def check(result: int, operation: str) -> None:
    if result != 0:
        raise RuntimeError(f"{operation} failed with NVRTC status {result}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("src/cuda/tinylocate.cu"))
    parser.add_argument("--output", type=Path, default=Path("src/cuda/tinylocate.ptx"))
    args = parser.parse_args()

    import torch

    library_dir = Path(torch.__file__).resolve().parent / "lib"
    os.add_dll_directory(str(library_dir))
    nvrtc = ctypes.WinDLL(str(library_dir / "nvrtc64_130_0.dll"))
    program = ctypes.c_void_p()
    source = args.source.read_bytes()
    check(nvrtc.nvrtcCreateProgram(ctypes.byref(program), source, b"tinylocate.cu", 0, None, None), "create")
    options = (ctypes.c_char_p * 3)(b"--std=c++14", b"--gpu-architecture=compute_86", b"--use_fast_math")
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    log_size = ctypes.c_size_t()
    check(nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(log_size)), "log size")
    if log_size.value > 1:
        log = ctypes.create_string_buffer(log_size.value)
        check(nvrtc.nvrtcGetProgramLog(program, log), "log")
        print(log.value.decode("utf-8", errors="replace"))
    check(result, "compile")
    ptx_size = ctypes.c_size_t()
    check(nvrtc.nvrtcGetPTXSize(program, ctypes.byref(ptx_size)), "PTX size")
    ptx = ctypes.create_string_buffer(ptx_size.value)
    check(nvrtc.nvrtcGetPTX(program, ptx), "PTX")
    check(nvrtc.nvrtcDestroyProgram(ctypes.byref(program)), "destroy")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(ptx.raw.rstrip(b"\0"))
    print(f"compiled={args.output} bytes={args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
