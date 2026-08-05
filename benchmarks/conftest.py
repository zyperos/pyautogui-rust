"""Side-effect-free backend used by the opt-in microbenchmarks."""

from __future__ import annotations

import sys
import types

import pytest


class InertRust(types.ModuleType):
    def set_process_dpi_aware(self):
        return None

    def start_failsafe_hook(self):
        return None

    def vk_key_scan_a(self, character_code):
        return character_code

    def get_system_metrics(self, index):
        return {0: 1920, 1: 1080}.get(index, 0)

    def check_failsafe_triggered(self):
        return False

    def reset_failsafe_triggered(self):
        return None


sys.modules["pyautogui._rust_core"] = InertRust("pyautogui._rust_core")

import pyautogui  # noqa: E402


class BenchmarkPlatform:
    keyboardMapping = dict.fromkeys(pyautogui.KEY_NAMES, 1)

    def __init__(self):
        self.position = [320, 240]

    def _position(self):
        return tuple(self.position)

    def _size(self):
        return (1920, 1080)

    def _moveTo(self, x, y):
        self.position[:] = [x, y]

    def _moveToSmooth(self, x, y, duration, steps):
        self.position[:] = [x, y]
        return True

    def _mouse_is_swapped(self):
        return False

    def _click(self, x, y, button):
        return None

    def _keyDown(self, key):
        return None

    def _keyUp(self, key):
        return None


@pytest.fixture(autouse=True)
def inert_backend(monkeypatch):
    backend = BenchmarkPlatform()
    monkeypatch.setattr(pyautogui, "platformModule", backend)
    monkeypatch.setattr(pyautogui, "FAILSAFE", False)
    monkeypatch.setattr(pyautogui, "PAUSE", 0.0)
    monkeypatch.setattr(pyautogui.time, "sleep", lambda _: None)
    return backend
