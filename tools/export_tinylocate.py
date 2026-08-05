"""Export a TinyLocate checkpoint to the runtime-specific TLN1 format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "training"))

import torch  # noqa: E402
from tinylocate import TinyLocateNet  # noqa: E402
from tinylocate.format import save_tln  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = TinyLocateNet()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    output = save_tln(model.state_dict(), args.output)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"exported {parameter_count:,} parameters to {output} ({output.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

