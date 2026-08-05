"""Train the compact reference-conditioned locator on generated pairs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import torch  # noqa: E402
from tinylocate import TinyLocateNet  # noqa: E402
from tinylocate.data import SyntheticPairDataset, gaussian_targets  # noqa: E402
from tinylocate.format import save_tln  # noqa: E402
from tinylocate.gpu_data import make_batch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid()
    positive = targets * torch.pow(1 - probabilities, 2) * F.softplus(-logits)
    negative = (1 - targets) * torch.pow(probabilities, 2) * F.softplus(logits)
    return (positive + negative).mean()


def box_iou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    upper_left = torch.maximum(predicted[:, :2], target[:, :2])
    lower_right = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (lower_right - upper_left).clamp_min(0).prod(dim=1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).clamp_min(0).prod(dim=1)
    target_area = (target[:, 2:] - target[:, :2]).clamp_min(0).prod(dim=1)
    overlap = intersection / (predicted_area + target_area - intersection).clamp_min(1e-6)
    return (1 - overlap).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--search-size", type=int, default=384)
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--generator", choices=("gpu", "pil"), default="gpu")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "models" / "tinylocate-v1.tln")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "tinylocate-v1.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--image-pool", type=Path)
    parser.add_argument("--negative-rate", type=float, default=0.35)
    parser.add_argument("--semantic-rate", type=float, default=0.15)
    parser.add_argument("--presence-weight", type=float, default=0.1)
    parser.add_argument("--max-instances", type=int, default=1)
    args = parser.parse_args()

    if not 0 <= args.negative_rate <= 1:
        parser.error("--negative-rate must be between 0 and 1")
    if not 0 <= args.semantic_rate <= 1:
        parser.error("--semantic-rate must be between 0 and 1")
    if args.presence_weight < 0:
        parser.error("--presence-weight must be non-negative")
    if args.max_instances < 1:
        parser.error("--max-instances must be positive")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training device was not detected")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = TinyLocateNet().to(device)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loader = None
    if args.generator == "pil":
        dataset = SyntheticPairDataset(search_size=args.search_size)
        loader = iter(
            DataLoader(
                dataset,
                batch_size=args.batch_size,
                num_workers=args.workers,
                pin_memory=True,
                persistent_workers=args.workers > 0,
            )
        )
    generator = torch.Generator(device=device).manual_seed(17)
    image_pool = None
    partner_pool = None
    if args.image_pool is not None:
        packed = torch.load(args.image_pool, map_location=device, weights_only=True)
        if isinstance(packed, dict):
            image_pool = packed["images"]
            partner_pool = packed.get("partners")
        else:
            image_pool = packed
        if image_pool.ndim != 4 or image_pool.shape[1:] != (3, 128, 128) or image_pool.dtype != torch.uint8:
            raise ValueError("image pool must be a uint8 Nx3x128x128 tensor")
        print(f"image_pool={args.image_pool} samples={image_pool.shape[0]}")
    started = time.perf_counter()
    running = 0.0
    window_steps = 0
    model.train()
    for step in range(1, args.steps + 1):
        if loader is None:
            search_size = (256, 320, 384)[(step - 1) % 3] if args.multi_scale else args.search_size
            sample = make_batch(
                args.batch_size,
                search_size,
                128,
                device,
                generator,
                negative_rate=args.negative_rate,
                image_pool=image_pool,
                partner_pool=partner_pool,
                semantic_rate=args.semantic_rate,
                max_instances=args.max_instances,
            )
        else:
            sample = next(loader)
        reference = sample["reference"].to(device, non_blocking=True)
        search = sample["search"].to(device, non_blocking=True)
        centers = sample.get("instance_centers", sample["center"].unsqueeze(1)).to(device, non_blocking=True)
        target_boxes = sample.get("instance_boxes", sample["box"].unsqueeze(1)).to(device, non_blocking=True)
        instance_mask = sample.get(
            "instance_mask", torch.ones(centers.shape[:2], dtype=torch.bool)
        ).to(device, non_blocking=True)
        present = instance_mask.any(dim=1)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(reference, search)
            output_height, output_width = output["objectness"].shape[-2:]
            heatmap = torch.zeros_like(output["objectness"])
            instance_heatmaps = []
            for instance in range(centers.shape[1]):
                instance_heatmap = gaussian_targets(
                    centers[:, instance], output_height, output_width
                ) * instance_mask[:, instance].view(-1, 1, 1, 1)
                instance_heatmaps.append(instance_heatmap)
                heatmap = torch.maximum(heatmap, instance_heatmap)
            location_loss = focal_loss(output["objectness"], heatmap)
            # Dense focal loss is deliberately dominated by easy background
            # cells.  An explicit image-level term teaches a stable confidence
            # boundary for searches where the reference is absent.
            box_map = output["box"].sigmoid().flatten(2)
            objectness_flat = output["objectness"].flatten(2)
            predicted_distance_items = []
            target_distance_items = []
            predicted_box_items = []
            target_box_items = []
            positive_logit_items = []
            for instance, instance_heatmap in enumerate(instance_heatmaps):
                valid = instance_mask[:, instance]
                flat_index = instance_heatmap.flatten(2).argmax(dim=2)
                predicted_distances = box_map.gather(
                    2, flat_index.unsqueeze(1).expand(-1, 4, -1)
                ).squeeze(2)
                positive_logits = objectness_flat.gather(2, flat_index.unsqueeze(1)).squeeze(2)
                target_box = target_boxes[:, instance]
                center = centers[:, instance]
                target_distances = torch.cat(
                    (center - target_box[:, :2], target_box[:, 2:] - center), dim=1
                )
                grid_y = torch.div(flat_index.squeeze(1), output_width, rounding_mode="floor")
                grid_x = flat_index.squeeze(1) % output_width
                predicted_center = torch.stack(
                    ((grid_x + 0.5) / output_width, (grid_y + 0.5) / output_height), dim=1
                )
                predicted_box = torch.cat(
                    (
                        predicted_center - predicted_distances[:, :2],
                        predicted_center + predicted_distances[:, 2:],
                    ),
                    dim=1,
                )
                predicted_distance_items.append(predicted_distances[valid])
                target_distance_items.append(target_distances[valid])
                predicted_box_items.append(predicted_box[valid])
                target_box_items.append(target_box[valid])
                positive_logit_items.append(positive_logits[valid])

            predicted_distance_values = torch.cat(predicted_distance_items)
            target_distance_values = torch.cat(target_distance_items)
            predicted_box_values = torch.cat(predicted_box_items)
            target_box_values = torch.cat(target_box_items)
            positive_logits = torch.cat(positive_logit_items)
            negative_logits = output["objectness"].flatten(1).amax(dim=1)[~present]
            positive_presence = (
                F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
                if positive_logits.numel()
                else output["objectness"].sum() * 0
            )
            negative_presence = (
                F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
                if negative_logits.numel()
                else positive_presence * 0
            )
            presence_loss = positive_presence + negative_presence
            box_loss = (
                F.smooth_l1_loss(predicted_distance_values, target_distance_values)
                + box_iou_loss(predicted_box_values, target_box_values) * 0.5
                if predicted_distance_values.numel()
                else output["box"].sum() * 0
            )
            loss = location_loss + box_loss * 4 + presence_loss * args.presence_weight
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.detach())
        window_steps += 1

        if step % 100 == 0 or step == args.steps:
            elapsed = time.perf_counter() - started
            print(f"step={step} loss={running / window_steps:.5f} samples/s={step * args.batch_size / elapsed:.1f}")
            running = 0.0
            window_steps = 0

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    save_tln(model.state_dict(), args.output)
    print(f"checkpoint={args.checkpoint}")
    print(f"runtime_weights={args.output} bytes={args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
