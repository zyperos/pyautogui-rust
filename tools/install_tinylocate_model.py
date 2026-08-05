"""Validate and install an external TinyLocate model for the current user."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from pyautogui import _rust_core


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()

    source = args.model.expanduser().resolve(strict=True)
    tensors, values, byte_count = _rust_core.tinylocate_model_info(str(source))
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    destination = args.destination or local_app_data / "PyAutoGUI" / "models" / "tinylocate-v1.tln"
    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    _rust_core.tinylocate_model_info(str(temporary))
    os.replace(temporary, destination)
    print(f"installed={destination}")
    print(f"tensors={tensors} values={values} bytes={byte_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

