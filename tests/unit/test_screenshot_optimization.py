"""Side-effect-free tests for the optional screenshot and locate fast paths."""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

from PIL import Image

from pyautogui import _pyautogui_pyscreeze_opt as opt


def _install_core(monkeypatch, **functions):
    core = SimpleNamespace(**functions)
    monkeypatch.setattr(opt, "_rust_core", core)
    monkeypatch.setattr(opt.sys, "platform", "win32")
    opt._disabled_rust_functions.clear()
    return core


def test_pil_template_conversion_needs_no_numpy():
    template = Image.new("RGB", (3, 2), (255, 0, 0))

    data, width, height = opt._needle_gray_data(template)

    assert (width, height) == (3, 2)
    assert data == bytes([76]) * 6


def test_bounded_cache_evicts_oldest_entry():
    cache = OrderedDict()
    for index in range(opt._CACHE_LIMIT + 1):
        opt._bounded_put(cache, index, index)

    assert len(cache) == opt._CACHE_LIMIT
    assert 0 not in cache
    assert cache[opt._CACHE_LIMIT] == opt._CACHE_LIMIT


def test_rust_screenshot_works_without_optional_vision_packages(monkeypatch):
    def metric(index):
        return {0: 2, 1: 1}[index]

    _install_core(
        monkeypatch,
        get_system_metrics=metric,
        capture_screen_gdi=lambda region: bytes([0, 0, 255, 0, 0, 255, 0, 0]),
    )

    image = opt.screenshot()

    assert image.size == (2, 1)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert image.getpixel((1, 0)) == (0, 255, 0)


def test_bad_native_capture_disables_only_capture_and_falls_back(monkeypatch):
    calls = []
    _install_core(
        monkeypatch,
        get_system_metrics=lambda index: 2 if index == 0 else 1,
        capture_screen_gdi=lambda region: b"short",
    )
    monkeypatch.setattr(opt, "_mss_module", None)
    monkeypatch.setattr(
        opt.pyscreeze,
        "screenshot",
        lambda imageFilename=None, region=None: calls.append((imageFilename, region)) or Image.new("RGB", (1, 1)),
    )

    image = opt.screenshot(region=(0, 0, 2, 1))

    assert image.size == (1, 1)
    assert calls == [(None, (0, 0, 2, 1))]
    assert "capture_screen_gdi" in opt._disabled_rust_functions
    assert "get_system_metrics" not in opt._disabled_rust_functions


def test_locate_with_confidence_uses_pil_to_rust_path(monkeypatch):
    observed = []

    def locate(data, width, height, confidence, region):
        observed.append((data, width, height, confidence, region))
        return 10, 20, width, height

    _install_core(monkeypatch, locate_on_screen_rust=locate)
    template = Image.new("L", (4, 3), 17)

    result = opt.locateOnScreen(template, confidence=0.9, region=(1, 2, 30, 40))

    assert result == opt.Box(10, 20, 4, 3)
    assert observed == [(bytes([17]) * 12, 4, 3, 0.9, (1, 2, 30, 40))]


def test_exact_locate_stays_on_pyscreeze(monkeypatch):
    marker = object()
    _install_core(
        monkeypatch,
        locate_on_screen_rust=lambda *args: (_ for _ in ()).throw(AssertionError("native path used")),
    )
    monkeypatch.setattr(opt.pyscreeze, "locateOnScreen", lambda *args, **kwargs: marker)

    assert opt.locateOnScreen(Image.new("RGB", (2, 2))) is marker


def test_native_locate_failure_silently_uses_pyscreeze(monkeypatch):
    marker = object()
    _install_core(
        monkeypatch,
        locate_on_screen_rust=lambda *args: (_ for _ in ()).throw(OSError("native failure")),
    )
    monkeypatch.setattr(opt.pyscreeze, "locateOnScreen", lambda *args, **kwargs: marker)

    result = opt.locateOnScreen(Image.new("RGB", (2, 2)), confidence=0.8)

    assert result is marker
    assert "locate_on_screen_rust" in opt._disabled_rust_functions


def test_animated_reference_uses_shared_capture_variant_path(monkeypatch, tmp_path):
    path = tmp_path / "target.gif"
    red = Image.new("RGB", (4, 3), "red")
    blue = Image.new("RGB", (4, 3), "blue")
    red.save(path, save_all=True, append_images=[blue], duration=20, loop=0)
    observed = []

    def locate(variants, confidence, region):
        observed.append((variants, confidence, region))
        return 8, 9, 4, 3, 0.97, 1

    _install_core(monkeypatch, locate_variants_on_screen_rust=locate)

    result = opt.locateOnScreen(path, confidence=0.85, region=(0, 0, 100, 100))

    assert result == opt.Box(8, 9, 4, 3)
    assert len(observed) == 1
    assert len(observed[0][0]) == 2


def test_static_reference_expands_scale_after_canonical_miss(monkeypatch):
    calls = []

    def locate_one(*_args):
        return None

    def locate_variants(variants, confidence, region):
        calls.append((variants, confidence, region))
        selected = next(item for item in variants if item[1:] == (5, 5))
        return 20, 30, selected[1], selected[2], 0.91, variants.index(selected)

    _install_core(
        monkeypatch,
        locate_on_screen_rust=locate_one,
        locate_variants_on_screen_rust=locate_variants,
    )

    result = opt.locateOnScreen(Image.new("RGB", (4, 4), "green"), confidence=0.8, region=(0, 0, 100, 100))

    assert result == opt.Box(20, 30, 5, 5)
    assert len(calls[0][0]) == 5


def test_neural_reference_path_runs_after_native_misses(monkeypatch, tmp_path):
    model = tmp_path / "model.tln"
    model.write_bytes(b"model")
    monkeypatch.setenv("PYAUTOGUI_TINYLOCATE_MODEL", str(model))
    observed = []

    def infer(model_path, reference, reference_width, reference_height, search, search_width, search_height):
        observed.append((model_path, len(reference), len(search), search_width, search_height))
        return 0.25, 0.2, 0.75, 0.8, 0.93

    _install_core(
        monkeypatch,
        locate_on_screen_rust=lambda *_args: None,
        locate_variants_on_screen_rust=lambda *_args: None,
        tinylocate_model_info=lambda _path: (2, 20, 5),
        tinylocate_infer=infer,
    )
    monkeypatch.setattr(opt, "_rust_screenshot", lambda region: Image.new("RGB", (200, 100), "gray"))

    result = opt.locateOnScreen(Image.new("RGB", (12, 10), "red"), confidence=0.9, region=(10, 20, 200, 100))

    assert result == opt.Box(60, 40, 100, 60)
    assert observed == [(str(model.resolve()), 128 * 128 * 3, 200 * 100 * 3, 200, 100)]


def test_locate_all_uses_neural_multi_candidate_path(monkeypatch, tmp_path):
    model = tmp_path / "model.tln"
    model.write_bytes(b"model")
    monkeypatch.setenv("PYAUTOGUI_TINYLOCATE_MODEL", str(model))

    _install_core(
        monkeypatch,
        tinylocate_model_info=lambda _path: (2, 20, 5),
        tinylocate_infer_all=lambda *_args: [
            (0.1, 0.2, 0.3, 0.5, 0.9),
            (0.6, 0.4, 0.9, 0.8, 0.85),
        ],
    )
    monkeypatch.setattr(opt, "_rust_screenshot", lambda region: Image.new("RGB", (200, 100), "gray"))

    matches = list(opt.locateAllOnScreen(Image.new("RGB", (80, 80), "red"), confidence=0.8, region=(10, 20, 200, 100)))

    assert matches == [opt.Box(30, 40, 40, 30), opt.Box(130, 60, 60, 40)]


def test_high_confidence_neural_hit_becomes_bounded_online_reference(monkeypatch, tmp_path):
    model = tmp_path / "model.tln"
    model.write_bytes(b"model")
    reference_path = tmp_path / "reference.png"
    Image.new("RGB", (24, 20), "red").save(reference_path)
    monkeypatch.setenv("PYAUTOGUI_TINYLOCATE_MODEL", str(model))
    calls = []

    def infer(*args):
        calls.append(args[1])
        return 0.2, 0.2, 0.6, 0.7, 0.99

    _install_core(
        monkeypatch,
        tinylocate_model_info=lambda _path: (2, 20, 5),
        tinylocate_infer=infer,
    )
    monkeypatch.setattr(opt, "_rust_screenshot", lambda region: Image.new("RGB", (200, 100), "gray"))
    opt._online_reference_bank.clear()
    profile = opt._profile_cache.get(reference_path)

    assert opt._neural_locate_once(profile, 0.8, None) == opt.Box(40, 20, 80, 50)
    assert len(opt._online_reference_bank.get(profile.key)) == 1
    assert opt._neural_locate_once(profile, 0.8, None) == opt.Box(40, 20, 80, 50)
    assert len(calls) == 3
    opt._online_reference_bank.clear()
