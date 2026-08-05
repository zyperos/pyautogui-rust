"""Measure framework-free TinyLocate warm-up and steady-state latency."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from pyautogui import _rust_core

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    fixtures = ROOT / "tests" / "fixtures" / "tinylocate"
    reference = Image.open(fixtures / "reference.png").convert("RGB")
    search = Image.open(fixtures / "search.png").convert("RGB")
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
    _rust_core.tinylocate_infer(*call)
    warmup = (time.perf_counter() - started) * 1000
    timings = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        _rust_core.tinylocate_infer(*call)
        timings.append((time.perf_counter() - started) * 1000)
    report = {
        "gpu": _rust_core.tinylocate_gpu_info(),
        "model_bytes": args.model.stat().st_size,
        "iterations": args.iterations,
        "warmup_ms": warmup,
        "mean_ms": statistics.fmean(timings),
        "p50_ms": percentile(timings, 0.50),
        "p95_ms": percentile(timings, 0.95),
        "maximum_ms": max(timings),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

