"""Regenerate the framework/runtime parity fixture from a training checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tinylocate import TinyLocateNet  # noqa: E402
from torchvision.transforms.functional import pil_to_tensor  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "tinylocate",
    )
    args = parser.parse_args()

    model = TinyLocateNet().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    images = []
    for name in ("reference.png", "search.png"):
        with Image.open(args.fixtures / name) as opened:
            images.append(pil_to_tensor(opened.convert("RGB")).float().div(255).unsqueeze(0))

    with torch.inference_mode():
        output = model(*images)
        objectness = output["objectness"]
        height, width = objectness.shape[-2:]
        flat_index = int(objectness.flatten().argmax())
        y, x = divmod(flat_index, width)
        center_x = (x + 0.5) / width
        center_y = (y + 0.5) / height
        distances = output["box"].sigmoid()[0, :, y, x]
        box = [
            center_x - float(distances[0]),
            center_y - float(distances[1]),
            center_x + float(distances[2]),
            center_y + float(distances[3]),
        ]
        expected = {"box": box, "score": float(objectness[0, 0, y, x].sigmoid())}

    destination = args.fixtures / "expected.json"
    destination.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"updated={destination}")
    print(json.dumps(expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
