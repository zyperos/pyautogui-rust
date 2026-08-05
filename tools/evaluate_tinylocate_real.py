"""Evaluate TinyLocate on transformed real images through the shipped CUDA runtime."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from pyautogui import _rust_core


def reference_canvas(image: Image.Image) -> Image.Image:
    value = image.copy()
    value.thumbnail((120, 120), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (128, 128), "black")
    canvas.paste(value, ((128 - value.width) // 2, (128 - value.height) // 2))
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("images", type=Path)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--search-size", type=int, default=384)
    parser.add_argument("--negative-samples", type=int, default=50)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    extensions = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    paths = [path for path in args.images.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    if len(paths) < 2:
        raise ValueError("real-image evaluation requires at least two images")
    rng = random.Random(71_377)
    errors = []
    scores = []
    for _ in range(args.samples):
        source_path, background_path = rng.sample(paths, 2)
        with Image.open(source_path) as opened:
            source = ImageOps.fit(opened.convert("RGB"), (128, 128), method=Image.Resampling.LANCZOS)
        with Image.open(background_path) as opened:
            search = ImageOps.fit(
                opened.convert("RGB"),
                (args.search_size, args.search_size),
                method=Image.Resampling.LANCZOS,
            )
        reference = reference_canvas(source)
        target = source.rotate(rng.uniform(-22, 22), resample=Image.Resampling.BICUBIC, expand=True)
        target = ImageEnhance.Brightness(target).enhance(rng.uniform(0.65, 1.35))
        target = ImageEnhance.Color(target).enhance(rng.uniform(0.6, 1.4))
        if rng.random() < 0.25:
            target = target.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.0)))
        size = rng.randint(36, min(150, args.search_size // 2))
        target.thumbnail((size, size), Image.Resampling.LANCZOS)
        left = rng.randrange(args.search_size - target.width + 1)
        top = rng.randrange(args.search_size - target.height + 1)
        search.paste(target, (left, top))
        result = _rust_core.tinylocate_infer(
            str(args.model.resolve()),
            reference.tobytes(),
            reference.width,
            reference.height,
            search.tobytes(),
            search.width,
            search.height,
        )
        predicted = ((result[0] + result[2]) * search.width / 2, (result[1] + result[3]) * search.height / 2)
        expected = (left + target.width / 2, top + target.height / 2)
        errors.append(math.dist(predicted, expected))
        scores.append(result[4])

    negative_scores = []
    for _ in range(args.negative_samples):
        source_path, background_path = rng.sample(paths, 2)
        with Image.open(source_path) as opened:
            source = ImageOps.fit(opened.convert("RGB"), (128, 128), method=Image.Resampling.LANCZOS)
        with Image.open(background_path) as opened:
            search = ImageOps.fit(
                opened.convert("RGB"),
                (args.search_size, args.search_size),
                method=Image.Resampling.LANCZOS,
            )
        reference = reference_canvas(source)
        result = _rust_core.tinylocate_infer(
            str(args.model.resolve()),
            reference.tobytes(),
            reference.width,
            reference.height,
            search.tobytes(),
            search.width,
            search.height,
        )
        negative_scores.append(result[4])

    metrics = {
        "samples": len(errors),
        "negative_samples": len(negative_scores),
        "search_size": args.search_size,
        "mean_error_px": statistics.fmean(errors),
        "recall_16px": sum(value <= 16 for value in errors) / len(errors),
        "recall_32px": sum(value <= 32 for value in errors) / len(errors),
        "recall_64px": sum(value <= 64 for value in errors) / len(errors),
        "raw_score_mean": statistics.fmean(scores),
    }
    print(f"samples={metrics['samples']} mean_error_px={metrics['mean_error_px']:.2f}")
    print(f"recall_16px={metrics['recall_16px']:.4f}")
    print(f"recall_32px={metrics['recall_32px']:.4f}")
    print(f"recall_64px={metrics['recall_64px']:.4f}")
    print(f"raw_score_mean={metrics['raw_score_mean']:.4f}")
    positive_ordered = sorted(scores)
    metrics["positive_score_p05"] = positive_ordered[round((len(positive_ordered) - 1) * 0.05)]
    print(f"positive_score_p05={metrics['positive_score_p05']:.4f}")
    if negative_scores:
        ordered = sorted(negative_scores)
        p95 = ordered[round((len(ordered) - 1) * 0.95)]
        metrics["negative_score_p95"] = p95
        metrics["negative_score_p99"] = ordered[round((len(ordered) - 1) * 0.99)]
        metrics["false_positive_public_0.8"] = sum(value >= 0.508 for value in negative_scores) / len(
            negative_scores
        )
        print(f"negative_score_p95={p95:.4f}")
        print(f"negative_score_p99={metrics['negative_score_p99']:.4f}")
        print(f"false_positive_public_0.8={metrics['false_positive_public_0.8']:.4f}")
        threshold_metrics = {}
        for threshold in (0.6, 0.7, 0.8, 0.9):
            recall = sum(value >= threshold for value in scores) / len(scores)
            false_positive = sum(value >= threshold for value in negative_scores) / len(negative_scores)
            threshold_metrics[str(threshold)] = {"recall": recall, "false_positive": false_positive}
            print(f"raw_threshold_{threshold:.1f}=recall:{recall:.4f},false_positive:{false_positive:.4f}")
        metrics["raw_thresholds"] = threshold_metrics
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"json={args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
