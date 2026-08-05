"""Evaluate multi-peak recall for a TinyLocate training checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import torch  # noqa: E402
from tinylocate import TinyLocateNet  # noqa: E402
from tinylocate.gpu_data import make_batch  # noqa: E402
from torch.nn import functional as F  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--image-pool", type=Path)
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--search-size", type=int, default=384)
    parser.add_argument("--max-instances", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda")
    model = TinyLocateNet().to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    image_pool = partner_pool = None
    if args.image_pool:
        packed = torch.load(args.image_pool, map_location=device, weights_only=True)
        image_pool = packed["images"]
        partner_pool = packed.get("partners")
    generator = torch.Generator(device=device).manual_seed(731_991)
    matched_32 = matched_64 = predictions = targets = 0
    errors = []

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(args.batches):
            sample = make_batch(
                args.batch_size,
                args.search_size,
                128,
                device,
                generator,
                negative_rate=0.0,
                image_pool=image_pool,
                partner_pool=partner_pool,
                semantic_rate=0.0,
                max_instances=args.max_instances,
            )
            output = model(sample["reference"], sample["search"])
            scores = output["objectness"].sigmoid()
            maxima = scores == F.max_pool2d(scores, 3, stride=1, padding=1)
            scores = scores * maxima
            height, width = scores.shape[-2:]
            values, indices = scores.flatten(1).topk(args.max_instances * 2, dim=1)
            for batch in range(args.batch_size):
                predicted = []
                for value, index in zip(values[batch], indices[batch]):
                    if float(value) < args.threshold:
                        continue
                    y = int(index) // width
                    x = int(index) % width
                    predicted.append(((x + 0.5) * args.search_size / width, (y + 0.5) * args.search_size / height))
                actual = (
                    sample["instance_centers"][batch][sample["instance_mask"][batch]] * args.search_size
                ).tolist()
                predictions += len(predicted)
                targets += len(actual)
                pairs = sorted(
                    (torch.dist(torch.tensor(first), torch.tensor(second)).item(), i, j)
                    for i, first in enumerate(predicted)
                    for j, second in enumerate(actual)
                )
                used_predictions = set()
                used_targets = set()
                for error, prediction_index, target_index in pairs:
                    if prediction_index in used_predictions or target_index in used_targets:
                        continue
                    if error > 64:
                        break
                    used_predictions.add(prediction_index)
                    used_targets.add(target_index)
                    errors.append(error)
                    matched_64 += 1
                    if error <= 32:
                        matched_32 += 1

    precision = matched_64 / max(1, predictions)
    recall = matched_64 / max(1, targets)
    print(f"targets={targets} predictions={predictions}")
    print(f"precision_64px={precision:.4f} recall_64px={recall:.4f}")
    print(f"recall_32px={matched_32 / max(1, targets):.4f}")
    if errors:
        print(f"matched_center_error_mean_px={sum(errors) / len(errors):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
