"""Fast tests for argument normalization, geometry, and safety controls."""

from __future__ import annotations

import sys

import pytest

import pyautogui


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, 0.0),
        (0.25, 0.25),
        (1.0, 1.0),
    ],
)
def test_linear_tween(value, expected):
    assert pyautogui.linear(value) == expected


@pytest.mark.parametrize("value", [-0.01, 1.01, 2.0])
def test_linear_tween_rejects_out_of_range_values(value):
    with pytest.raises(pyautogui.PyAutoGUIException):
        pyautogui.linear(value)


def test_get_point_on_line_interpolates_both_axes():
    assert pyautogui.getPointOnLine(0, 10, 100, 30, 0.25) == (25.0, 15.0)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((10, 20), None, (10, 20)),
        ([10, 20], None, (10, 20)),
        (pyautogui.Point(10, 20), None, (10, 20)),
        (10, 20, (10, 20)),
        (None, None, pyautogui.Point(320, 240)),
    ],
)
def test_normalize_xy_args(first, second, expected):
    assert pyautogui._normalizeXYArgs(first, second) == expected


def test_normalize_xy_rejects_sequence_plus_y():
    with pytest.raises(pyautogui.PyAutoGUIException):
        pyautogui._normalizeXYArgs((10, 20), 30)


@pytest.mark.parametrize("character", ["A", "Z", "!", "?", "_"])
def test_shift_characters(character):
    assert pyautogui.isShiftCharacter(character)


@pytest.mark.parametrize("character", ["a", "z", "1", "-", " "])
def test_non_shift_characters(character):
    assert not pyautogui.isShiftCharacter(character)


@pytest.mark.skipif(sys.platform != "win32", reason="native failsafe hook is Windows-only")
def test_failsafe_reads_native_monitor_and_resets_it(fake_platform, monkeypatch):
    monkeypatch.setattr(pyautogui, "FAILSAFE", True)
    fake_platform.failsafe_triggered = True

    with pytest.raises(pyautogui.FailSafeException):
        pyautogui.failSafeCheck()

    assert fake_platform.failsafe_triggered is False
    assert fake_platform.calls_named("consume_failsafe_trigger")


def test_disabled_failsafe_has_no_native_call(fake_platform):
    pyautogui.FAILSAFE = False
    pyautogui.failSafeCheck()
    assert fake_platform.calls == []


def test_context_pause_is_optional(monkeypatch, fake_platform):
    sleeps = []
    monkeypatch.setattr(pyautogui.time, "sleep", sleeps.append)
    pyautogui.PAUSE = 0.25

    pyautogui.moveTo(10, 20, _pause=False)
    assert sleeps == []

    pyautogui.moveTo(20, 30)
    assert sleeps == [0.25]
    assert fake_platform.position == [20, 30]
