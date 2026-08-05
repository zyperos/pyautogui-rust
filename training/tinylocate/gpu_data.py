"""High-throughput GPU synthesis for reference/search training pairs."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def _affine(value: Tensor, generator: torch.Generator) -> Tensor:
    batch = value.shape[0]
    angle = (torch.rand(batch, device=value.device, generator=generator) - 0.5) * math.radians(40)
    scale = torch.rand(batch, device=value.device, generator=generator) * 0.35 + 0.82
    cosine = torch.cos(angle) / scale
    sine = torch.sin(angle) / scale
    theta = torch.zeros(batch, 2, 3, device=value.device, dtype=value.dtype)
    theta[:, 0, 0] = cosine
    theta[:, 0, 1] = -sine
    theta[:, 1, 0] = sine
    theta[:, 1, 1] = cosine
    grid = F.affine_grid(theta, value.shape, align_corners=False)
    return F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def make_batch(
    batch_size: int,
    search_size: int,
    reference_size: int,
    device: torch.device,
    generator: torch.Generator,
    negative_rate: float = 0.2,
    image_pool: Tensor | None = None,
    partner_pool: Tensor | None = None,
    semantic_rate: float = 0.25,
    max_instances: int = 1,
) -> dict[str, Tensor]:
    if max_instances < 1:
        raise ValueError("max_instances must be positive")
    if image_pool is None:
        low_resolution = max(8, reference_size // 8)
        texture = torch.rand(
            batch_size,
            3,
            low_resolution,
            low_resolution,
            device=device,
            generator=generator,
        )
        texture = F.interpolate(texture, (reference_size, reference_size), mode="bicubic", align_corners=False)
        detail = torch.rand(
            batch_size,
            3,
            reference_size,
            reference_size,
            device=device,
            generator=generator,
        )
        texture = (texture * 0.78 + detail * 0.22).clamp(0, 1)
    else:
        indices = torch.randint(0, image_pool.shape[0], (batch_size,), device=device, generator=generator)
        texture = image_pool[indices].float().div_(255)
    target_texture = texture
    if image_pool is not None and partner_pool is not None:
        semantic = torch.rand(batch_size, device=device, generator=generator) < semantic_rate
        target_indices = torch.where(semantic, partner_pool[indices], indices)
        target_texture = image_pool[target_indices].float().div_(255)

    axis = torch.linspace(-1, 1, reference_size, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    x = x.view(1, 1, reference_size, reference_size)
    y = y.view(1, 1, reference_size, reference_size)
    radius_x = torch.rand(batch_size, 1, 1, 1, device=device, generator=generator) * 0.35 + 0.55
    radius_y = torch.rand(batch_size, 1, 1, 1, device=device, generator=generator) * 0.35 + 0.55
    ellipse = ((x / radius_x).square() + (y / radius_y).square() < 1).float()
    rectangle = ((x.abs() < radius_x) & (y.abs() < radius_y)).float()
    choose_ellipse = (torch.rand(batch_size, 1, 1, 1, device=device, generator=generator) > 0.5).float()
    mask = ellipse * choose_ellipse + rectangle * (1 - choose_ellipse)
    canonical = texture * mask
    target_canonical = target_texture * mask

    reference = _affine(canonical, generator)
    target = _affine(target_canonical, generator)
    for value in (reference, target):
        gain = torch.rand(batch_size, 3, 1, 1, device=device, generator=generator) * 0.7 + 0.65
        bias = (torch.rand(batch_size, 3, 1, 1, device=device, generator=generator) - 0.5) * 0.2
        value.mul_(gain).add_(bias).clamp_(0, 1)

    present = torch.rand(batch_size, device=device, generator=generator) >= negative_rate
    if image_pool is None:
        background = torch.rand(
            batch_size,
            3,
            max(8, search_size // 16),
            max(8, search_size // 16),
            device=device,
            generator=generator,
        )
    else:
        background_indices = torch.randint(
            0, image_pool.shape[0], (batch_size,), device=device, generator=generator
        )
        if partner_pool is not None:
            hard_negative = (~present) & (
                torch.rand(batch_size, device=device, generator=generator) < 0.5
            )
            background_indices = torch.where(hard_negative, partner_pool[indices], background_indices)
        background = image_pool[background_indices].float().div_(255)
    search = F.interpolate(background, (search_size, search_size), mode="bicubic", align_corners=False).clamp_(0, 1)
    maximum_target = min(160, search_size // 2)
    instance_boxes = torch.zeros(batch_size, max_instances, 4, device=device)
    instance_centers = torch.zeros(batch_size, max_instances, 2, device=device)
    instance_mask = torch.zeros(batch_size, max_instances, device=device, dtype=torch.bool)
    instance_counts = torch.where(
        present,
        torch.randint(1, max_instances + 1, (batch_size,), device=device, generator=generator),
        torch.zeros(batch_size, device=device, dtype=torch.int64),
    )
    counts = instance_counts.tolist()
    sizes = torch.randint(
        32,
        maximum_target + 1,
        (batch_size, max_instances),
        device=device,
        generator=generator,
    ).tolist()
    candidate_offsets = torch.rand(
        batch_size, max_instances, 16, 2, device=device, generator=generator
    ).tolist()
    flip_values = (
        torch.rand(batch_size, max_instances, device=device, generator=generator) < 0.35
    ).tolist()
    patch_gains = (
        torch.rand(batch_size, max_instances, 3, 1, 1, device=device, generator=generator) * 0.5
        + 0.75
    )
    patch_biases = (
        torch.rand(batch_size, max_instances, 3, 1, 1, device=device, generator=generator) - 0.5
    ) * 0.16
    for index in range(batch_size):
        occupied: list[tuple[int, int, int, int]] = []
        for instance in range(counts[index]):
            size = sizes[index][instance]
            maximum_offset = search_size - size
            x1 = y1 = 0
            for attempt in range(16):
                offset = candidate_offsets[index][instance][attempt]
                x1 = min(maximum_offset, int(offset[0] * (maximum_offset + 1)))
                y1 = min(maximum_offset, int(offset[1] * (maximum_offset + 1)))
                candidate = (x1, y1, x1 + size, y1 + size)
                if all(
                    max(0, min(candidate[2], old[2]) - max(candidate[0], old[0]))
                    * max(0, min(candidate[3], old[3]) - max(candidate[1], old[1]))
                    < size * size * 0.12
                    for old in occupied
                ):
                    break
            occupied.append((x1, y1, x1 + size, y1 + size))
            patch = F.interpolate(
                target[index : index + 1], (size, size), mode="bilinear", align_corners=False
            )[0]
            gain = patch_gains[index, instance]
            bias = patch_biases[index, instance]
            patch = (patch * gain + bias).clamp_(0, 1)
            if flip_values[index][instance]:
                patch = patch.flip(-1)
            patch_mask = (patch.abs().sum(dim=0, keepdim=True) > 0.01).float()
            existing = search[index, :, y1 : y1 + size, x1 : x1 + size]
            search[index, :, y1 : y1 + size, x1 : x1 + size] = patch * patch_mask + existing * (1 - patch_mask)
            inverse_search_size = 1.0 / float(search_size)
            instance_boxes[index, instance] = torch.tensor(
                (x1, y1, x1 + size, y1 + size), device=device
            ) * inverse_search_size
            instance_centers[index, instance] = torch.tensor(
                (x1 + size / 2, y1 + size / 2), device=device
            ) * inverse_search_size
            instance_mask[index, instance] = True

    return {
        "reference": reference,
        "search": search,
        "box": instance_boxes[:, 0],
        "center": instance_centers[:, 0],
        "present": present,
        "instance_boxes": instance_boxes,
        "instance_centers": instance_centers,
        "instance_mask": instance_mask,
    }
