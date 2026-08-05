"""Read and write the compact TLN1 tensor container."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO, Mapping

import torch
from torch import Tensor

MAGIC = b"TLN1"
VERSION = 1
DTYPE_FLOAT16 = 1


def _write_tensor(stream: BinaryIO, name: str, tensor: Tensor) -> None:
    encoded_name = name.encode("utf-8")
    value = tensor.detach().to(device="cpu", dtype=torch.float16).contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    if len(encoded_name) > 65535 or value.ndim > 255:
        raise ValueError(f"tensor metadata is too large: {name}")
    stream.write(struct.pack("<HBB", len(encoded_name), value.ndim, DTYPE_FLOAT16))
    stream.write(encoded_name)
    stream.write(struct.pack(f"<{value.ndim}I", *value.shape))
    stream.write(struct.pack("<Q", len(raw)))
    stream.write(raw)


def save_tln(state: Mapping[str, Tensor], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    # BatchNorm's training-only counter has no role in inference and is a
    # scalar integer, while TLN1 deliberately stores only FP16 tensors.
    ordered = sorted((name, tensor) for name, tensor in state.items() if not name.endswith("num_batches_tracked"))
    with path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<HI", VERSION, len(ordered)))
        for name, tensor in ordered:
            _write_tensor(stream, name, tensor)
    return path
