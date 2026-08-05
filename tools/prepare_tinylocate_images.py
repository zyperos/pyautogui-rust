"""Pack a real-image directory into a GPU-friendly TinyLocate tensor pool."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    extensions = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    paths = [path for path in args.source.rglob("*") if path.is_file() and path.suffix.lower() in extensions]
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(f"no images found under {args.source}")
    packed = torch.empty((len(paths), 3, 128, 128), dtype=torch.uint8)
    accepted = 0
    labels = []
    label_ids = {name: index for index, name in enumerate(sorted({path.parent.name for path in paths}))}
    for path in paths:
        try:
            with Image.open(path) as opened:
                image = ImageOps.fit(opened.convert("RGB"), (128, 128), method=Image.Resampling.LANCZOS)
                packed[accepted].copy_(pil_to_tensor(image))
                labels.append(label_ids[path.parent.name])
                accepted += 1
        except (OSError, ValueError):
            continue
        if accepted % 500 == 0:
            print(f"packed={accepted}/{len(paths)}")
    packed = packed[:accepted].contiguous()
    labels_tensor = torch.tensor(labels, dtype=torch.int64)
    partners = torch.empty(accepted, dtype=torch.int64)
    for label in labels_tensor.unique():
        members = torch.nonzero(labels_tensor == label, as_tuple=False).flatten()
        partners[members] = members.roll(-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": packed, "labels": labels_tensor, "partners": partners}, args.output)
    print(f"output={args.output} samples={accepted} bytes={args.output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
