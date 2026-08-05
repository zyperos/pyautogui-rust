"""Cross-check the framework-free CUDA runtime against a PyTorch fixture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from pyautogui import _rust_core

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--fixtures", type=Path, default=ROOT / "tests" / "fixtures" / "tinylocate")
    args = parser.parse_args()

    reference = Image.open(args.fixtures / "reference.png").convert("RGB")
    search = Image.open(args.fixtures / "search.png").convert("RGB")
    expected = json.loads((args.fixtures / "expected.json").read_text())
    call = (
        str(args.model.resolve()),
        reference.tobytes(),
        reference.width,
        reference.height,
        search.tobytes(),
        search.width,
        search.height,
    )
    started = time.perf_counter()
    first = _rust_core.tinylocate_infer(*call)
    warmup_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    result = _rust_core.tinylocate_infer(*call)
    steady_ms = (time.perf_counter() - started) * 1000

    expected_values = [*expected["box"], expected["score"]]
    maximum_error = max(abs(actual - wanted) for actual, wanted in zip(result, expected_values))
    if maximum_error > 0.002:
        raise AssertionError(f"CUDA/PyTorch parity error {maximum_error:.6f}: {result} != {expected_values}")
    if not _rust_core.tinylocate_gpu_self_test():
        raise AssertionError("CUDA kernel self-test failed")
    print(f"warmup_ms={warmup_ms:.3f} steady_ms={steady_ms:.3f} max_error={maximum_error:.6f}")
    print(f"result={result}")
    print(f"first={first}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
