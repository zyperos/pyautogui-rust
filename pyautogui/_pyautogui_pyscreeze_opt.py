"""Compatibility-first screenshot and locate acceleration.

The public functions in this module mirror PyScreeze.  Rust and MSS are
strictly optional implementation details: a missing extension, optional
dependency, or one failing fast path leaves the established PyAutoGUI call
shape and fallback behavior intact.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import OrderedDict, abc
from pathlib import Path
from typing import Any

import pyscreeze
from PIL import Image

from ._visual_locator import (
    OnlineReferenceBank,
    ProfileCache,
    ReferenceProfile,
    TemporalTracker,
    discover_external_model,
)

Box = pyscreeze.Box
Point = pyscreeze.Point
RGB = pyscreeze.RGB
center = pyscreeze.center
ImageNotFoundException = pyscreeze.ImageNotFoundException

try:
    from . import _rust_core
except Exception:
    _rust_core = None

try:
    import mss as _mss_module
except ImportError:
    _mss_module = None


_UNAVAILABLE = object()
_INVALID_REGION = object()
_disabled_rust_functions: set[str] = set()
_template_cache: "OrderedDict[tuple[str, int, int], tuple[bytes, int, int]]" = OrderedDict()
_last_match_cache: "OrderedDict[tuple[str, int, int], Box]" = OrderedDict()
_profile_cache = ProfileCache()
_temporal_tracker = TemporalTracker()
_online_reference_bank = OnlineReferenceBank()
_CACHE_LIMIT = 128
_NEURAL_MAX_SIDE = 512


def _try_rust(function_name: str, *args: Any) -> Any:
    """Run one native fast path, disabling only that operation after failure."""

    if _rust_core is None or function_name in _disabled_rust_functions:
        return _UNAVAILABLE
    function = getattr(_rust_core, function_name, None)
    if not callable(function):
        _disabled_rust_functions.add(function_name)
        return _UNAVAILABLE
    try:
        return function(*args)
    except Exception:
        _disabled_rust_functions.add(function_name)
        return _UNAVAILABLE


def _bounded_put(cache: OrderedDict, key: Any, value: Any) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def _path_cache_key(image: Any) -> tuple[str, int, int] | None:
    if not isinstance(image, (str, bytes, os.PathLike)):
        return None
    try:
        path = Path(os.fsdecode(os.fspath(image))).resolve()
        stat = path.stat()
    except (OSError, TypeError, ValueError):
        return None
    return str(path), stat.st_mtime_ns, stat.st_size


def _needle_gray_data(image: Any) -> tuple[bytes, int, int]:
    """Return tightly packed 8-bit luminance data without requiring NumPy."""

    key = _path_cache_key(image)
    if key is not None:
        cached = _template_cache.get(key)
        if cached is not None:
            _template_cache.move_to_end(key)
            return cached
        with Image.open(key[0]) as opened:
            gray = opened.convert("L")
            value = gray.tobytes(), gray.width, gray.height
        _bounded_put(_template_cache, key, value)
        return value

    if isinstance(image, Image.Image):
        gray = image.convert("L")
        return gray.tobytes(), gray.width, gray.height

    # NumPy is deliberately imported lazily. It remains an optional vision
    # dependency and is not needed for Rust screenshots or PIL templates.
    try:
        import numpy as np
    except ImportError as exc:
        raise TypeError("template must be a path or PIL image") from exc

    if not isinstance(image, np.ndarray):
        raise TypeError("template must be a path, PIL image, or NumPy array")
    array = image
    if array.ndim == 2:
        gray = np.ascontiguousarray(array, dtype=np.uint8)
    elif array.ndim == 3 and array.shape[2] in (3, 4):
        # PyScreeze/OpenCV arrays use BGR/BGRA ordering.
        blue = array[..., 0].astype(np.uint32)
        green = array[..., 1].astype(np.uint32)
        red = array[..., 2].astype(np.uint32)
        gray = ((red * 77 + green * 150 + blue * 29) >> 8).astype(np.uint8)
        gray = np.ascontiguousarray(gray)
    else:
        raise ValueError("template array must be HxW, HxWx3, or HxWx4")
    height, width = gray.shape[:2]
    return gray.tobytes(), int(width), int(height)


def _coerce_region(region: Any) -> tuple[int, int, int, int] | object | None:
    if region is None:
        return None
    try:
        left, top, width, height = region
        rect = int(left), int(top), int(width), int(height)
    except (TypeError, ValueError):
        return _INVALID_REGION
    if rect[2] <= 0 or rect[3] <= 0:
        return _INVALID_REGION
    return rect


def _primary_capture_size() -> tuple[int, int] | None:
    width = _try_rust("get_system_metrics", 0)
    height = _try_rust("get_system_metrics", 1)
    if (
        width is _UNAVAILABLE
        or height is _UNAVAILABLE
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        return None
    return width, height


def _rust_screenshot(region: Any) -> Image.Image | None:
    if sys.platform != "win32":
        return None
    rect = _coerce_region(region)
    if rect is _INVALID_REGION:
        return None
    if rect is None:
        size = _primary_capture_size()
        if size is None:
            return None
        width, height = size
    else:
        _, _, width, height = rect

    raw = _try_rust("capture_screen_gdi", rect)
    if raw is _UNAVAILABLE:
        return None
    expected_length = width * height * 4
    if not isinstance(raw, (bytes, bytearray, memoryview)) or len(raw) != expected_length:
        _disabled_rust_functions.add("capture_screen_gdi")
        return None
    return Image.frombytes("RGB", (width, height), raw, "raw", "BGRX")


def _mss_screenshot(region: Any) -> Image.Image | None:
    if _mss_module is None:
        return None
    rect = _coerce_region(region)
    if rect is _INVALID_REGION:
        return None
    try:
        factory = getattr(_mss_module, "MSS", None) or _mss_module.mss
        with factory() as capture:
            if rect is None:
                monitor = capture.monitors[1]
            else:
                left, top, width, height = rect
                monitor = {"left": left, "top": top, "width": width, "height": height}
            frame = capture.grab(monitor)
            return Image.frombytes("RGB", frame.size, frame.bgra, "raw", "BGRX")
    except Exception:
        return None


def screenshot(imageFilename=None, region=None):
    """Return a PIL screenshot using Rust, MSS, then PyScreeze in that order."""

    image = _rust_screenshot(region) or _mss_screenshot(region)
    if image is None:
        return pyscreeze.screenshot(imageFilename=imageFilename, region=region)
    if imageFilename is not None:
        image.save(imageFilename)
    return image


def _can_use_rust_locate(kwargs: dict[str, Any]) -> bool:
    # PyScreeze's confidence path is grayscale OpenCV matching by default.
    # Calls that request exact/color matching or extra PyScreeze controls stay
    # on PyScreeze so their historical semantics remain unchanged.
    if sys.platform != "win32" or "confidence" not in kwargs:
        return False
    if set(kwargs) - {"confidence", "region", "grayscale"}:
        return False
    if kwargs.get("grayscale", True) is False:
        return False
    try:
        confidence = float(kwargs["confidence"])
    except (TypeError, ValueError):
        return False
    return 0.0 <= confidence <= 1.0


def _search_region_from_cache(key: tuple[str, int, int] | None) -> tuple[int, int, int, int] | None:
    if key is None:
        return None
    box = _last_match_cache.get(key)
    size = _primary_capture_size()
    if box is None or size is None:
        return None
    width, height = size
    predicted = _temporal_tracker.predict_region(key, size)
    if predicted is not None:
        return predicted
    padding = 64
    left = max(0, box.left - padding)
    top = max(0, box.top - padding)
    right = min(width, box.left + box.width + padding)
    bottom = min(height, box.top + box.height + padding)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def _profile_tracking_key(profile: ReferenceProfile, image: Any) -> Any:
    return profile.key if profile.key is not None else ("object", id(image))


def _rust_locate_once(
    needle: tuple[bytes, int, int], confidence: float, region: tuple[int, int, int, int] | None
) -> Box | object | None:
    data, width, height = needle
    result = _try_rust("locate_on_screen_rust", data, width, height, confidence, region)
    if result is _UNAVAILABLE or result is None:
        return result
    try:
        left, top, result_width, result_height = (int(value) for value in result)
    except (TypeError, ValueError):
        _disabled_rust_functions.add("locate_on_screen_rust")
        return _UNAVAILABLE
    if result_width <= 0 or result_height <= 0:
        _disabled_rust_functions.add("locate_on_screen_rust")
        return _UNAVAILABLE
    return Box(left, top, result_width, result_height)


def _rust_locate_variants_once(profile: ReferenceProfile, confidence: float, region, adaptive: bool = False):
    variants = profile.adaptive if adaptive else profile.primary
    if len(variants) == 1 and not adaptive:
        return _rust_locate_once(variants[0].native_tuple(), confidence, region)
    native_variants = [item.native_tuple() for item in variants]
    result = _try_rust("locate_variants_on_screen_rust", native_variants, confidence, region)
    if result is _UNAVAILABLE or result is None:
        return result
    try:
        left, top, width, height, _score, _variant_index = result
        box = Box(int(left), int(top), int(width), int(height))
    except (TypeError, ValueError):
        _disabled_rust_functions.add("locate_variants_on_screen_rust")
        return _UNAVAILABLE
    if box.width <= 0 or box.height <= 0:
        _disabled_rust_functions.add("locate_variants_on_screen_rust")
        return _UNAVAILABLE
    return box


def _neural_locate_once(profile: ReferenceProfile, confidence: float, region):
    model = discover_external_model(_rust_core)
    if model is None or not profile.neural:
        return None
    image = _rust_screenshot(region)
    if image is None:
        return None
    capture_width, capture_height = image.size
    search = image.convert("RGB")
    if max(search.size) > _NEURAL_MAX_SIDE:
        scale = _NEURAL_MAX_SIDE / max(search.size)
        search = search.resize(
            (max(1, round(search.width * scale)), max(1, round(search.height * scale))),
            Image.Resampling.BILINEAR,
        )
    best = None
    references = profile.neural + _online_reference_bank.get(profile.key)
    for reference, reference_width, reference_height in references:
        result = _try_rust(
            "tinylocate_infer",
            str(model.path),
            reference,
            reference_width,
            reference_height,
            search.tobytes(),
            search.width,
            search.height,
        )
        if result is _UNAVAILABLE:
            return None
        try:
            left, top, right, bottom, score = (float(value) for value in result)
        except (TypeError, ValueError):
            _disabled_rust_functions.add("tinylocate_infer")
            return None
        bounded_score = min(max(score, 1e-6), 1 - 1e-6)
        raw_logit = math.log(bounded_score / (1 - bounded_score))
        score = 1 / (1 + math.exp(-((raw_logit + 0.8) / 0.6)))
        if score < confidence or not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
            continue
        if best is None or score > best[0]:
            best = score, left, top, right, bottom
    if best is None:
        return None
    best_score, left, top, right, bottom = best
    if best_score >= 0.97 and profile.key is not None:
        learned = image.crop(
            (
                round(left * capture_width),
                round(top * capture_height),
                round(right * capture_width),
                round(bottom * capture_height),
            )
        )
        _online_reference_bank.update(profile.key, learned)
    origin_x, origin_y = (region[:2] if region is not None else (0, 0))
    box_left = origin_x + round(left * capture_width)
    box_top = origin_y + round(top * capture_height)
    box_right = origin_x + round(right * capture_width)
    box_bottom = origin_y + round(bottom * capture_height)
    return Box(box_left, box_top, max(1, box_right - box_left), max(1, box_bottom - box_top))


def _neural_locate_all(profile: ReferenceProfile, confidence: float, region):
    model = discover_external_model(_rust_core)
    if model is None or not profile.neural:
        return []
    image = _rust_screenshot(region)
    if image is None:
        return []
    capture_width, capture_height = image.size
    search = image.convert("RGB")
    if max(search.size) > _NEURAL_MAX_SIDE:
        scale = _NEURAL_MAX_SIDE / max(search.size)
        search = search.resize(
            (max(1, round(search.width * scale)), max(1, round(search.height * scale))),
            Image.Resampling.BILINEAR,
        )
    bounded = min(max(confidence, 1e-6), 1 - 1e-6)
    calibrated_logit = math.log(bounded / (1 - bounded))
    raw_confidence = 1 / (1 + math.exp(-((0.6 * calibrated_logit) - 0.8)))
    origin_x, origin_y = (region[:2] if region is not None else (0, 0))
    boxes = []
    references = profile.neural[:4] + _online_reference_bank.get(profile.key)
    for reference, reference_width, reference_height in references:
        results = _try_rust(
            "tinylocate_infer_all",
            str(model.path),
            reference,
            reference_width,
            reference_height,
            search.tobytes(),
            search.width,
            search.height,
            raw_confidence,
            128,
        )
        if results is _UNAVAILABLE:
            return []
        try:
            for left, top, right, bottom, _score in results:
                box_left = origin_x + round(float(left) * capture_width)
                box_top = origin_y + round(float(top) * capture_height)
                box_right = origin_x + round(float(right) * capture_width)
                box_bottom = origin_y + round(float(bottom) * capture_height)
                candidate = Box(
                    box_left,
                    box_top,
                    max(1, box_right - box_left),
                    max(1, box_bottom - box_top),
                )
                if candidate not in boxes:
                    boxes.append(candidate)
        except (TypeError, ValueError):
            _disabled_rust_functions.add("tinylocate_infer_all")
            return []
    return boxes


def locateOnScreen(needleImage, minSearchTime=0, **kwargs):
    """Locate one template, using Rust only for compatible confidence calls."""

    if not _can_use_rust_locate(kwargs):
        return pyscreeze.locateOnScreen(needleImage, minSearchTime=minSearchTime, **kwargs)

    try:
        profile = _profile_cache.get(needleImage)
    except (OSError, TypeError, ValueError):
        try:
            needle = _needle_gray_data(needleImage)
        except (OSError, TypeError, ValueError):
            return pyscreeze.locateOnScreen(needleImage, minSearchTime=minSearchTime, **kwargs)
        profile = None

    confidence = float(kwargs["confidence"])
    requested_region = _coerce_region(kwargs.get("region"))
    if requested_region is _INVALID_REGION:
        return pyscreeze.locateOnScreen(needleImage, minSearchTime=minSearchTime, **kwargs)

    key = _profile_tracking_key(profile, needleImage) if profile is not None and requested_region is None else None
    if key is not None:
        local_region = _search_region_from_cache(key)
        if local_region is not None:
            local_neural_first = profile is not None and profile.primary[0].width * profile.primary[0].height >= 4096
            local_result = _neural_locate_once(profile, confidence, local_region) if local_neural_first else None
            if local_result is None:
                local_result = (
                    _rust_locate_variants_once(profile, confidence, local_region)
                    if profile is not None
                    else _rust_locate_once(needle, confidence, local_region)
                )
            if isinstance(local_result, Box):
                _bounded_put(_last_match_cache, key, local_result)
                _temporal_tracker.update(key, local_result)
                return local_result
            if local_result is _UNAVAILABLE:
                return pyscreeze.locateOnScreen(needleImage, minSearchTime=minSearchTime, **kwargs)
            _temporal_tracker.miss(key)

    started = time.perf_counter()
    while True:
        neural_first = profile is not None and profile.primary[0].width * profile.primary[0].height >= 4096
        result = _neural_locate_once(profile, confidence, requested_region) if neural_first else None
        if result is None:
            result = (
                _rust_locate_variants_once(profile, confidence, requested_region)
                if profile is not None
                else _rust_locate_once(needle, confidence, requested_region)
            )
        # A canonical miss is routed to the compact neural matcher before the
        # more expensive exhaustive scale pyramid. This keeps large-variation
        # recovery bounded while retaining a deterministic non-neural path.
        if result is None and profile is not None and not neural_first:
            result = _neural_locate_once(profile, confidence, requested_region)
        if result is None and profile is not None:
            adaptive_result = _rust_locate_variants_once(profile, confidence, requested_region, adaptive=True)
            if adaptive_result is not _UNAVAILABLE:
                result = adaptive_result
        if isinstance(result, Box):
            if key is not None:
                _bounded_put(_last_match_cache, key, result)
                _temporal_tracker.update(key, result)
            return result
        if result is _UNAVAILABLE:
            return pyscreeze.locateOnScreen(needleImage, minSearchTime=minSearchTime, **kwargs)
        if time.perf_counter() - started >= minSearchTime:
            break
        time.sleep(0.02)

    if key is not None:
        _temporal_tracker.miss(key)
        _last_match_cache.pop(key, None)

    if getattr(pyscreeze, "USE_IMAGE_NOT_FOUND_EXCEPTION", False):
        raise pyscreeze.ImageNotFoundException("Image not found")
    return None


def locateAllOnScreen(needleImage, minSearchTime=0, **kwargs):
    if _can_use_rust_locate(kwargs):
        try:
            profile = _profile_cache.get(needleImage)
            area = profile.primary[0].width * profile.primary[0].height
            if profile.animated or area >= 4096:
                matches = _neural_locate_all(profile, float(kwargs["confidence"]), _coerce_region(kwargs.get("region")))
                if matches:
                    return iter(matches)
        except (OSError, TypeError, ValueError):
            pass
    if minSearchTime <= 0:
        return pyscreeze.locateAllOnScreen(needleImage, **kwargs)

    def retrying_generator():
        started = time.perf_counter()
        while True:
            try:
                matches = list(pyscreeze.locateAllOnScreen(needleImage, **kwargs))
            except pyscreeze.ImageNotFoundException:
                matches = []
            if matches:
                yield from matches
                return
            if time.perf_counter() - started >= minSearchTime:
                if getattr(pyscreeze, "USE_IMAGE_NOT_FOUND_EXCEPTION", False):
                    raise pyscreeze.ImageNotFoundException("Image not found")
                return
            time.sleep(0.02)

    return retrying_generator()


def locateCenterOnScreen(needleImage, **kwargs):
    result = locateOnScreen(needleImage, **kwargs)
    return center(result) if result is not None else None


def locate(needleImage, haystackImage, *args, **kwargs):
    return pyscreeze.locate(needleImage, haystackImage, *args, **kwargs)


def locateAll(needleImage, haystackImage, *args, **kwargs):
    return pyscreeze.locateAll(needleImage, haystackImage, *args, **kwargs)


def locateOnWindow(needleImage, windowTitle, **kwargs):
    function = getattr(pyscreeze, "locateOnWindow", None)
    if callable(function):
        return function(needleImage, windowTitle, **kwargs)

    import pygetwindow

    windows = pygetwindow.getWindowsWithTitle(windowTitle)
    if not windows:
        raise pyscreeze.PyScreezeException(f"Could not find a window titled {windowTitle!r}")
    window = windows[0]
    kwargs["region"] = window.left, window.top, window.width, window.height
    return locateOnScreen(needleImage, **kwargs)


def pixel(x, y):
    return screenshot(region=(x, y, 1, 1)).getpixel((0, 0))


def pixelMatchesColor(x, y, expectedColor, tolerance=0):
    if isinstance(x, abc.Sequence) and len(x) == 2:
        raise TypeError(
            "pixelMatchesColor() no longer accepts (x, y) as its first argument; "
            "pass x and y separately"
        )
    actual = pixel(x, y)
    return all(abs(actual[channel] - expectedColor[channel]) <= tolerance for channel in range(3))
