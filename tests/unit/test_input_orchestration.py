"""High-level behavior tests backed by an in-memory platform adapter."""

from __future__ import annotations

import pytest

import pyautogui


def event_args(fake_platform, name):
    return [call.args for call in fake_platform.calls_named(name)]


def test_position_size_and_screen_bounds_use_backend_state(fake_platform):
    fake_platform.position[:] = [123, 456]
    fake_platform.screen_size = (800, 600)

    assert pyautogui.position() == pyautogui.Point(123, 456)
    assert pyautogui.position(x=7) == pyautogui.Point(7, 456)
    assert pyautogui.size() == pyautogui.Size(800, 600)
    assert pyautogui.onScreen(0, 0)
    assert pyautogui.onScreen((799, 599))
    assert not pyautogui.onScreen(800, 599)
    assert not pyautogui.onScreen(-1, 0)


def test_move_to_and_relative_move_preserve_coordinates(fake_platform):
    pyautogui.moveTo(400.9, 300.1, _pause=False)
    pyautogui.moveRel(25, -10, _pause=False)

    assert fake_platform.position == [425, 290]
    assert event_args(fake_platform, "move_to") == [(400, 300), (425, 290)]


def test_linear_duration_uses_backend_smooth_path(fake_platform):
    pyautogui.moveTo(500, 400, duration=0.2, tween=pyautogui.linear, _pause=False)

    calls = event_args(fake_platform, "move_to_smooth")
    assert len(calls) == 1
    x, y, duration, steps = calls[0]
    assert (x, y, duration) == (500, 400, 0.2)
    assert steps > 0


def test_click_count_button_and_destination(fake_platform, monkeypatch):
    sleeps = []
    monkeypatch.setattr(pyautogui.time, "sleep", sleeps.append)

    pyautogui.click(100, 200, clicks=3, interval=0.125, button="right", _pause=False)

    assert event_args(fake_platform, "move_to") == [(100, 200)]
    assert event_args(fake_platform, "click") == [(100, 200, "right")] * 3
    assert sleeps == [0.125, 0.125, 0.125]


@pytest.mark.parametrize(
    ("function_name", "backend_name"),
    [("scroll", "scroll"), ("hscroll", "hscroll"), ("vscroll", "vscroll")],
)
def test_scroll_variants_forward_coordinates(fake_platform, function_name, backend_name):
    getattr(pyautogui, function_name)(-4, x=50, y=60, _pause=False)
    assert event_args(fake_platform, backend_name) == [(-4, 50, 60)]


def test_drag_brackets_movement_with_button_events(fake_platform):
    pyautogui.dragTo(700, 500, button="left", mouseDownUp=True, _pause=False)

    assert event_args(fake_platform, "mouse_down") == [(320, 240, "left")]
    assert event_args(fake_platform, "move_to")[-1] == (700, 500)
    assert event_args(fake_platform, "mouse_up") == [(700, 500, "left")]


def test_press_normalizes_named_keys_and_repeats(fake_platform, monkeypatch):
    monkeypatch.setattr(pyautogui.time, "sleep", lambda _: None)

    pyautogui.press(["CTRL", "a"], presses=2, interval=0, _pause=False)

    assert event_args(fake_platform, "key_down") == [("ctrl",), ("a",), ("ctrl",), ("a",)]
    assert event_args(fake_platform, "key_up") == [("ctrl",), ("a",), ("ctrl",), ("a",)]


def test_hotkey_releases_keys_in_reverse_order(fake_platform, monkeypatch):
    monkeypatch.setattr(pyautogui.time, "sleep", lambda _: None)

    pyautogui.hotkey("CTRL", "shift", "x", interval=0, _pause=False)

    assert event_args(fake_platform, "key_down") == [("ctrl",), ("shift",), ("x",)]
    assert event_args(fake_platform, "key_up") == [("x",), ("shift",), ("ctrl",)]


def test_hold_releases_every_key_when_body_raises(fake_platform):
    with pytest.raises(RuntimeError, match="body failed"):
        with pyautogui.hold(["CTRL", "x"], _pause=False):
            raise RuntimeError("body failed")

    assert event_args(fake_platform, "key_down") == [("ctrl",), ("x",)]
    assert event_args(fake_platform, "key_up") == [("ctrl",), ("x",)]


def test_raw_relative_alias_never_touches_real_input(fake_platform):
    pyautogui.moveRelRaw(7, -3, _pause=False)
    assert event_args(fake_platform, "move_rel") == [(7, -3)]


def test_primary_and_secondary_follow_swapped_button_setting(fake_platform):
    assert pyautogui._normalizeButton(pyautogui.PRIMARY) == pyautogui.LEFT
    assert pyautogui._normalizeButton(pyautogui.SECONDARY) == pyautogui.RIGHT

    fake_platform.swapped = True
    assert pyautogui._normalizeButton(pyautogui.PRIMARY) == pyautogui.RIGHT
    assert pyautogui._normalizeButton(pyautogui.SECONDARY) == pyautogui.LEFT
