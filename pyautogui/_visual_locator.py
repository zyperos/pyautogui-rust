"""Reference preparation and temporal state for adaptive visual location.

The module intentionally contains no screen-capture or platform code.  It
turns still images and animated image files into a compact set of grayscale
reference variants, and maintains conservative motion predictions between
calls.  Native, GPU, and compatibility backends can consume the same data.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

from PIL import Image, ImageChops, ImageStat

_PROFILE_CACHE_LIMIT = 64
_MAX_ANIMATION_FRAMES = 12
_SCALE_LEVELS = (1.0, 0.875, 1.125, 0.75, 1.25)


@dataclass(frozen=True)
class ReferenceVariant:
    pixels: bytes
    width: int
    height: int
    frame_index: int
    scale: float

    def native_tuple(self) -> Tuple[bytes, int, int]:
        return self.pixels, self.width, self.height


@dataclass(frozen=True)
class ReferenceProfile:
    key: Optional[Tuple[str, int, int]]
    primary: Tuple[ReferenceVariant, ...]
    adaptive: Tuple[ReferenceVariant, ...]
    neural: Tuple[Tuple[bytes, int, int], ...]
    animated: bool


@dataclass(frozen=True)
class RuntimeModelInfo:
    path: Path
    tensors: int
    values: int
    bytes: int


@dataclass
class TrackState:
    current: Tuple[int, int, int, int]
    previous: Optional[Tuple[int, int, int, int]]
    misses: int
    updated_at: float


def path_cache_key(image: Any) -> Optional[Tuple[str, int, int]]:
    if not isinstance(image, (str, bytes, os.PathLike)):
        return None
    try:
        path = Path(os.fsdecode(os.fspath(image))).resolve()
        stat = path.stat()
    except (OSError, TypeError, ValueError):
        return None
    return str(path), stat.st_mtime_ns, stat.st_size


def _representative_frames(opened: Image.Image) -> Iterable[Image.Image]:
    frame_count = int(getattr(opened, "n_frames", 1) or 1)
    if frame_count <= _MAX_ANIMATION_FRAMES:
        indices: Sequence[int] = tuple(range(frame_count))
    else:
        indices = tuple(
            sorted(
                {
                    round(index * (frame_count - 1) / (_MAX_ANIMATION_FRAMES - 1))
                    for index in range(_MAX_ANIMATION_FRAMES)
                }
            )
        )

    previous_signature = None
    for index in indices:
        opened.seek(index)
        frame = opened.convert("RGBA")
        # Deduplicate animation frames using a tiny luminance signature. This
        # keeps matching cost bounded for GIFs with repeated delay frames.
        signature = frame.convert("L").resize((16, 16), Image.Resampling.BILINEAR)
        if previous_signature is not None:
            difference = ImageStat.Stat(ImageChops.difference(signature, previous_signature)).mean[0]
            if difference < 1.0:
                continue
        previous_signature = signature
        yield frame


def _variant(frame: Image.Image, frame_index: int, scale: float) -> Optional[ReferenceVariant]:
    gray = frame.convert("L")
    if scale != 1.0:
        width = max(1, round(gray.width * scale))
        height = max(1, round(gray.height * scale))
        if width < 2 or height < 2:
            return None
        gray = gray.resize((width, height), Image.Resampling.BILINEAR)
    return ReferenceVariant(gray.tobytes(), gray.width, gray.height, frame_index, scale)


def build_profile(image: Any) -> ReferenceProfile:
    key = path_cache_key(image)
    if key is not None:
        with Image.open(key[0]) as opened:
            frames = tuple(_representative_frames(opened))
            animated = int(getattr(opened, "n_frames", 1) or 1) > 1
    elif isinstance(image, Image.Image):
        frames = (image.convert("RGBA"),)
        animated = False
    else:
        raise TypeError("reference must be a path or PIL image")

    if not frames:
        raise ValueError("reference image contains no decodable frames")

    primary = tuple(_variant(frame, index, 1.0) for index, frame in enumerate(frames))
    primary = tuple(item for item in primary if item is not None)
    adaptive = []
    for scale in _SCALE_LEVELS:
        for index, frame in enumerate(frames):
            item = _variant(frame, index, scale)
            if item is not None:
                adaptive.append(item)
    neural = []
    for frame in frames:
        rgb = frame.convert("RGB")
        rgb.thumbnail((128, 128), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (128, 128), "black")
        canvas.paste(rgb, ((128 - rgb.width) // 2, (128 - rgb.height) // 2))
        neural.append((canvas.tobytes(), canvas.width, canvas.height))
    return ReferenceProfile(key, primary, tuple(adaptive), tuple(neural), animated)


class ProfileCache:
    def __init__(self, limit: int = _PROFILE_CACHE_LIMIT) -> None:
        self._limit = limit
        self._items: "OrderedDict[Any, ReferenceProfile]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, image: Any) -> ReferenceProfile:
        key = path_cache_key(image)
        if key is None:
            return build_profile(image)
        cache_key = key
        with self._lock:
            cached = self._items.get(cache_key)
            if cached is not None:
                self._items.move_to_end(cache_key)
                return cached
        profile = build_profile(image)
        with self._lock:
            self._items[cache_key] = profile
            self._items.move_to_end(cache_key)
            while len(self._items) > self._limit:
                self._items.popitem(last=False)
        return profile

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class TemporalTracker:
    """Constant-velocity ROI prediction with miss-driven expansion."""

    def __init__(self, limit: int = 128) -> None:
        self._limit = limit
        self._states: "OrderedDict[Any, TrackState]" = OrderedDict()
        self._lock = threading.RLock()

    def update(self, key: Any, box: Sequence[int]) -> None:
        if key is None:
            return
        value = tuple(int(part) for part in box)
        if len(value) != 4 or value[2] <= 0 or value[3] <= 0:
            return
        with self._lock:
            old = self._states.get(key)
            self._states[key] = TrackState(value, old.current if old else None, 0, time.monotonic())
            self._states.move_to_end(key)
            while len(self._states) > self._limit:
                self._states.popitem(last=False)

    def miss(self, key: Any) -> None:
        if key is None:
            return
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return
            state.misses += 1
            if state.misses >= 4:
                self._states.pop(key, None)

    def predict_region(
        self, key: Any, bounds: Sequence[int]
    ) -> Optional[Tuple[int, int, int, int]]:
        if key is None:
            return None
        screen_width, screen_height = (int(value) for value in bounds)
        with self._lock:
            state = self._states.get(key)
            if state is None:
                return None
            left, top, width, height = state.current
            if state.previous is not None:
                left += left - state.previous[0]
                top += top - state.previous[1]
            padding = 64 * (state.misses + 1)

        region_left = max(0, left - padding)
        region_top = max(0, top - padding)
        region_right = min(screen_width, left + width + padding)
        region_bottom = min(screen_height, top + height + padding)
        if region_right <= region_left or region_bottom <= region_top:
            return None
        return region_left, region_top, region_right - region_left, region_bottom - region_top

    def clear(self) -> None:
        with self._lock:
            self._states.clear()


class OnlineReferenceBank:
    """Conservative per-target appearance memory for high-confidence hits."""

    def __init__(self, targets: int = 64, variants: int = 4) -> None:
        self._target_limit = targets
        self._variant_limit = variants
        self._items: "OrderedDict[Any, list[Tuple[bytes, int, int]]]" = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: Any) -> Tuple[Tuple[bytes, int, int], ...]:
        if key is None:
            return ()
        with self._lock:
            values = self._items.get(key)
            if values is None:
                return ()
            self._items.move_to_end(key)
            return tuple(values)

    def update(self, key: Any, image: Image.Image) -> None:
        if key is None or image.width < 4 or image.height < 4:
            return
        rgb = image.convert("RGB")
        rgb.thumbnail((120, 120), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (128, 128), "black")
        canvas.paste(rgb, ((128 - rgb.width) // 2, (128 - rgb.height) // 2))
        variant = canvas.tobytes(), 128, 128
        with self._lock:
            values = self._items.setdefault(key, [])
            if any(existing[0] == variant[0] for existing in values):
                return
            values.append(variant)
            del values[:-self._variant_limit]
            self._items.move_to_end(key)
            while len(self._items) > self._target_limit:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def external_model_candidates() -> Tuple[Path, ...]:
    candidates = []
    configured = os.environ.get("PYAUTOGUI_TINYLOCATE_MODEL")
    if configured:
        candidates.append(Path(configured).expanduser())
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "PyAutoGUI" / "models" / "tinylocate-v1.tln")
    candidates.append(Path.home() / ".cache" / "pyautogui" / "models" / "tinylocate-v1.tln")
    return tuple(candidates)


def discover_external_model(native_core: Any) -> Optional[RuntimeModelInfo]:
    inspect_model = getattr(native_core, "tinylocate_model_info", None)
    if not callable(inspect_model):
        return None
    for candidate in external_model_candidates():
        try:
            path = candidate.resolve()
            if not path.is_file():
                continue
            tensors, values, byte_count = (int(value) for value in inspect_model(str(path)))
            if tensors <= 0 or values <= 0 or byte_count <= 0:
                continue
            return RuntimeModelInfo(path, tensors, values, byte_count)
        except (OSError, TypeError, ValueError):
            continue
    return None
