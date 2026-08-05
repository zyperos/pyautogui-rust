"""Synthetic accuracy and throughput gate for TinyLocate checkpoints."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import torch  # noqa: E402
from tinylocate import TinyLocateNet  # noqa: E402
from tinylocate.gpu_data import make_batch  # noqa: E402


def box_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    upper_left = torch.maximum(first[:, :2], second[:, :2])
    lower_right = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (lower_right - upper_left).clamp_min(0).prod(dim=1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=1)
    return intersection / (first_area + second_area - intersection).clamp_min(1e-6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--search-size", type=int, default=384)
    parser.add_argument("--negative-batches", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = TinyLocateNet().to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    generator = torch.Generator(device=device).manual_seed(91_337)
    errors = []
    overlaps = []
    positive_scores = []
    negative_scores = []
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(args.batches):
            sample = make_batch(args.batch_size, args.search_size, 128, device, generator, negative_rate=0.0)
            output = model(sample["reference"], sample["search"])
            positive_scores.append(output["objectness"].flatten(1).amax(dim=1).sigmoid())
            height, width = output["objectness"].shape[-2:]
            index = output["objectness"].flatten(2).argmax(dim=2)
            predicted_y = torch.div(index, width, rounding_mode="floor").squeeze(1)
            predicted_x = (index % width).squeeze(1)
            predicted_center = torch.stack(((predicted_x + 0.5) / width, (predicted_y + 0.5) / height), dim=1)
            errors.append(torch.linalg.vector_norm(predicted_center - sample["center"], dim=1) * args.search_size)
            box_map = output["box"].sigmoid().flatten(2)
            distances = box_map.gather(2, index.unsqueeze(1).expand(-1, 4, -1)).squeeze(2)
            predicted_box = torch.cat(
                (predicted_center - distances[:, :2], predicted_center + distances[:, 2:]), dim=1
            )
            overlaps.append(box_iou(predicted_box, sample["box"]))
        for _ in range(args.negative_batches):
            sample = make_batch(args.batch_size, args.search_size, 128, device, generator, negative_rate=1.0)
            output = model(sample["reference"], sample["search"])
            negative_scores.append(output["objectness"].flatten(1).amax(dim=1).sigmoid())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    error = torch.cat(errors)
    overlap = torch.cat(overlaps)
    positives = torch.cat(positive_scores).float()
    negatives = torch.cat(negative_scores).float() if negative_scores else torch.empty(0, device=device)
    print(f"samples={error.numel()} samples_per_second={error.numel() / elapsed:.1f}")
    print(f"center_error_mean_px={error.mean():.2f}")
    print(f"recall_16px={(error <= 16).float().mean():.4f}")
    print(f"recall_32px={(error <= 32).float().mean():.4f}")
    print(f"mean_iou={overlap.mean():.4f}")
    print(f"positive_score_p05={positives.quantile(0.05):.4f}")
    print(f"score_recall_at_0.8={(positives >= 0.8).float().mean():.4f}")
    if negatives.numel():
        print(f"negative_score_p95={negatives.quantile(0.95):.4f}")
        print(f"false_positive_at_0.8={(negatives >= 0.8).float().mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
