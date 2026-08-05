"""Procedural pair generation for category-agnostic visual matching."""

from __future__ import annotations

import math
import random
from typing import Iterator

import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import functional as TF


def _procedural_image(size: int, rng: random.Random) -> Image.Image:
    base = Image.new(
        "RGB",
        (size, size),
        tuple(rng.randrange(20, 236) for _ in range(3)),
    )
    draw = ImageDraw.Draw(base)
    for _ in range(rng.randint(8, 24)):
        x1, y1 = rng.randrange(size), rng.randrange(size)
        x2, y2 = rng.randrange(x1, size), rng.randrange(y1, size)
        color = tuple(rng.randrange(256) for _ in range(3))
        shape = rng.randrange(3)
        if shape == 0:
            draw.rectangle((x1, y1, x2, y2), fill=color, outline="white", width=1)
        elif shape == 1:
            draw.ellipse((x1, y1, x2, y2), fill=color, outline="black", width=1)
        else:
            draw.line((x1, y1, x2, y2), fill=color, width=rng.randint(1, 5))
    return base


def _alter(image: Image.Image, rng: random.Random, output_size: int) -> Image.Image:
    value = image.rotate(rng.uniform(-22, 22), resample=Image.Resampling.BILINEAR, expand=True)
    value = ImageEnhance.Brightness(value).enhance(rng.uniform(0.65, 1.35))
    value = ImageEnhance.Contrast(value).enhance(rng.uniform(0.7, 1.35))
    value = ImageEnhance.Color(value).enhance(rng.uniform(0.55, 1.45))
    if rng.random() < 0.3:
        value = value.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 1.2)))
    value.thumbnail((output_size, output_size), Image.Resampling.LANCZOS)
    return value


class SyntheticPairDataset(IterableDataset):
    def __init__(self, search_size: int = 384, reference_size: int = 128, seed: int = 17) -> None:
        super().__init__()
        self.search_size = search_size
        self.reference_size = reference_size
        self.seed = seed

    def _samples(self, rng: random.Random) -> Iterator[dict[str, Tensor]]:
        while True:
            target_source = _procedural_image(rng.randint(56, 144), rng)
            reference = _alter(target_source, rng, self.reference_size)
            target_size = rng.randint(32, min(150, self.search_size // 2))
            target = _alter(target_source, rng, target_size)
            if target.width < 8 or target.height < 8:
                continue

            search = _procedural_image(self.search_size, rng).filter(
                ImageFilter.GaussianBlur(rng.uniform(0.0, 0.8))
            )
            left = rng.randrange(0, self.search_size - target.width + 1)
            top = rng.randrange(0, self.search_size - target.height + 1)
            search.paste(target, (left, top))

            if rng.random() < 0.4:
                draw = ImageDraw.Draw(search)
                occlusion_width = rng.randint(1, max(1, target.width // 3))
                occlusion_height = rng.randint(1, max(1, target.height // 3))
                ox = left + rng.randrange(target.width)
                oy = top + rng.randrange(target.height)
                draw.rectangle(
                    (ox, oy, min(left + target.width, ox + occlusion_width), min(top + target.height, oy + occlusion_height)),
                    fill=tuple(rng.randrange(256) for _ in range(3)),
                )

            reference_canvas = Image.new("RGB", (self.reference_size, self.reference_size), "black")
            reference.thumbnail((self.reference_size, self.reference_size), Image.Resampling.LANCZOS)
            reference_canvas.paste(
                reference,
                ((self.reference_size - reference.width) // 2, (self.reference_size - reference.height) // 2),
            )
            box = torch.tensor(
                (
                    left / self.search_size,
                    top / self.search_size,
                    (left + target.width) / self.search_size,
                    (top + target.height) / self.search_size,
                ),
                dtype=torch.float32,
            )
            center = torch.tensor(
                (
                    (left + target.width / 2) / self.search_size,
                    (top + target.height / 2) / self.search_size,
                ),
                dtype=torch.float32,
            )
            yield {
                "reference": TF.pil_to_tensor(reference_canvas).float().div_(255),
                "search": TF.pil_to_tensor(search).float().div_(255),
                "box": box,
                "center": center,
            }

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        seed = self.seed + worker_id * 1_000_003
        return self._samples(random.Random(seed))


def gaussian_targets(center: Tensor, height: int, width: int, sigma: float = 1.5) -> Tensor:
    y = torch.arange(height, device=center.device, dtype=center.dtype).view(1, height, 1)
    x = torch.arange(width, device=center.device, dtype=center.dtype).view(1, 1, width)
    center_x = center[:, 0].view(-1, 1, 1) * width
    center_y = center[:, 1].view(-1, 1, 1) * height
    distance = (x - center_x).square() + (y - center_y).square()
    return torch.exp(-distance / (2 * math.pow(sigma, 2))).unsqueeze(1)

