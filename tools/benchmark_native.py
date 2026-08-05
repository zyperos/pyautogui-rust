"""Benchmark read-only native screenshot and template-matching paths."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import pyscreeze

import pyautogui


def timings(operation, iterations: int) -> list[float]:
    operation()
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        values.append((time.perf_counter() - started) * 1_000)
    return values


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * 0.95)))
    return {
        "median_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[p95_index], 4),
        "minimum_ms": round(ordered[0], 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.iterations < 3 or args.width < 40 or args.height < 40:
        parser.error("iterations must be >= 3 and capture dimensions must be >= 40")

    screen_width, screen_height = pyautogui.size()
    width = min(args.width, screen_width)
    height = min(args.height, screen_height)
    region = (0, 0, width, height)

    accelerated = timings(lambda: pyautogui.screenshot(region=region), args.iterations)
    baseline = timings(lambda: pyscreeze.screenshot(region=region), max(3, args.iterations // 2))

    frame = pyautogui.screenshot(region=region)
    template_size = min(32, width // 2, height // 2)
    template_left = max(0, (width - template_size) // 3)
    template_top = max(0, (height - template_size) // 3)
    template = frame.crop(
        (
            template_left,
            template_top,
            template_left + template_size,
            template_top + template_size,
        )
    )

    def locate():
        result = pyautogui.locateOnScreen(template, confidence=0.99, region=region)
        if result is None:
            raise RuntimeError("captured template was not found in the source region")

    matching = timings(locate, max(3, args.iterations // 3))
    accelerated_summary = summary(accelerated)
    baseline_summary = summary(baseline)
    report = {
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pyautogui": pyautogui.__version__,
            "screen": [screen_width, screen_height],
            "region": list(region),
        },
        "screenshot_accelerated": accelerated_summary,
        "screenshot_pyscreeze": baseline_summary,
        "screenshot_speedup": round(
            baseline_summary["median_ms"] / accelerated_summary["median_ms"], 3
        ),
        "template_match": summary(matching),
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
