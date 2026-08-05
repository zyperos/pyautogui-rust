"""Microbenchmarks for Python compatibility-layer overhead.

These benchmarks intentionally use an inert backend. They measure argument
normalization and orchestration, not operating-system scheduling latency.
"""

import pytest

import pyautogui

pytestmark = pytest.mark.benchmark


def test_position_overhead(benchmark):
    result = benchmark(pyautogui.position)
    assert result == pyautogui.Point(320, 240)


def test_screen_bounds_overhead(benchmark):
    assert benchmark(pyautogui.onScreen, 1919, 1079)


def test_coordinate_normalization_overhead(benchmark):
    result = benchmark(pyautogui._normalizeXYArgs, (120, 240), None)
    assert result == (120, 240)


def test_instant_move_orchestration_overhead(benchmark):
    benchmark(pyautogui.moveTo, 640, 480, _pause=False)
    assert pyautogui.position() == pyautogui.Point(640, 480)


def test_click_orchestration_overhead(benchmark):
    benchmark(pyautogui.click, 640, 480, button="left", _pause=False)


def test_short_hotkey_orchestration_overhead(benchmark):
    benchmark(pyautogui.hotkey, "ctrl", "c", _pause=False)
