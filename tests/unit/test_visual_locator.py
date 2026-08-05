from pathlib import Path

from PIL import Image

from pyautogui import _visual_locator as locator


def test_still_profile_has_primary_and_multiscale_variants():
    image = Image.new("RGB", (40, 24), "navy")
    profile = locator.build_profile(image)

    assert not profile.animated
    assert [(item.width, item.height) for item in profile.primary] == [(40, 24)]
    assert (35, 21) in {(item.width, item.height) for item in profile.adaptive}
    assert (50, 30) in {(item.width, item.height) for item in profile.adaptive}


def test_gif_profile_deduplicates_frames(tmp_path: Path):
    path = tmp_path / "animated.gif"
    red = Image.new("RGB", (20, 12), "red")
    blue = Image.new("RGB", (20, 12), "blue")
    red.save(path, save_all=True, append_images=[red, blue], duration=20, loop=0)

    profile = locator.build_profile(path)

    assert profile.animated
    assert len(profile.primary) == 2
    assert {item.frame_index for item in profile.primary} == {0, 1}


def test_profile_cache_invalidates_changed_path(tmp_path: Path):
    path = tmp_path / "target.png"
    Image.new("RGB", (10, 10), "red").save(path)
    cache = locator.ProfileCache()
    first = cache.get(path)
    Image.new("RGB", (14, 9), "blue").save(path)
    second = cache.get(path)

    assert first.primary[0].width == 10
    assert (second.primary[0].width, second.primary[0].height) == (14, 9)
    assert first is not second


def test_tracker_predicts_velocity_and_expands_after_miss():
    tracker = locator.TemporalTracker()
    key = ("target", 1)
    tracker.update(key, (100, 80, 20, 30))
    tracker.update(key, (110, 85, 20, 30))

    first = tracker.predict_region(key, (500, 400))
    tracker.miss(key)
    expanded = tracker.predict_region(key, (500, 400))

    assert first == (56, 26, 148, 158)
    assert expanded == (0, 0, 268, 248)


def test_tracker_evicts_state_after_repeated_misses():
    tracker = locator.TemporalTracker()
    tracker.update("target", (5, 5, 10, 10))
    for _ in range(4):
        tracker.miss("target")
    assert tracker.predict_region("target", (100, 100)) is None


def test_external_model_is_discovered_and_validated(monkeypatch, tmp_path: Path):
    model = tmp_path / "model.tln"
    model.write_bytes(b"model")
    monkeypatch.setenv("PYAUTOGUI_TINYLOCATE_MODEL", str(model))

    class Core:
        @staticmethod
        def tinylocate_model_info(path):
            assert Path(path) == model.resolve()
            return 10, 500, 5

    info = locator.discover_external_model(Core())

    assert info == locator.RuntimeModelInfo(model.resolve(), 10, 500, 5)


def test_corrupt_external_model_is_skipped(monkeypatch, tmp_path: Path):
    model = tmp_path / "bad.tln"
    model.write_bytes(b"bad")
    monkeypatch.setenv("PYAUTOGUI_TINYLOCATE_MODEL", str(model))

    class Core:
        @staticmethod
        def tinylocate_model_info(_path):
            raise ValueError("bad model")

    assert locator.discover_external_model(Core()) is None


def test_online_reference_bank_is_bounded_and_deduplicated():
    bank = locator.OnlineReferenceBank(targets=2, variants=2)
    bank.update("a", Image.new("RGB", (20, 10), "red"))
    bank.update("a", Image.new("RGB", (20, 10), "red"))
    bank.update("a", Image.new("RGB", (20, 10), "green"))
    bank.update("a", Image.new("RGB", (20, 10), "blue"))
    bank.update("b", Image.new("RGB", (20, 10), "white"))
    bank.update("c", Image.new("RGB", (20, 10), "black"))

    assert len(bank.get("a")) == 0
    assert len(bank.get("b")) == 1
    assert len(bank.get("c")) == 1
