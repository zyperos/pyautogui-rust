"""Hermetic test doubles for PyAutoGUI's operating-system boundary.

The native module is replaced *before* importing :mod:`pyautogui`. Unit tests
therefore exercise Python orchestration without moving the host mouse, typing,
scrolling, changing timer resolution, or installing a Windows hook.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class BackendCall:
    name: str
    args: tuple[Any, ...]


class RustStub(types.ModuleType):
    """Import-compatible, recording replacement for ``pyautogui._rust_core``."""

    def __init__(self) -> None:
        super().__init__("pyautogui._rust_core")
        self.__file__ = "<hermetic-test-double>"
        self.calls: list[BackendCall] = []
        self.failsafe_triggered = False

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append(BackendCall(name, args))

    def set_process_dpi_aware(self) -> None:
        self._record("set_process_dpi_aware")

    def start_failsafe_hook(self) -> None:
        self._record("start_failsafe_hook")

    def check_failsafe_triggered(self) -> bool:
        self._record("check_failsafe_triggered")
        return self.failsafe_triggered

    def reset_failsafe_triggered(self) -> None:
        self._record("reset_failsafe_triggered")
        self.failsafe_triggered = False

    def vk_key_scan_a(self, character_code: int) -> int:
        self._record("vk_key_scan_a", character_code)
        return character_code

    def get_cursor_pos(self) -> tuple[int, int]:
        self._record("get_cursor_pos")
        return (320, 240)

    def get_system_metrics(self, index: int) -> int:
        self._record("get_system_metrics", index)
        return {0: 1920, 1: 1080, 76: 0, 77: 0, 78: 1920, 79: 1080}.get(index, 0)

    def send_inputs(self, events: Any) -> int:
        frozen_events = tuple(tuple(event) for event in events)
        self._record("send_inputs", frozen_events)
        return len(frozen_events)

    def set_cursor_pos(self, x: int, y: int) -> None:
        self._record("set_cursor_pos", x, y)

    def move_to_smooth(self, x: int, y: int, duration: float, steps: int) -> None:
        self._record("move_to_smooth", x, y, duration, steps)

    def send_mouse_event(self, *args: Any) -> None:
        self._record("send_mouse_event", *args)

    def move_rel(self, x: int, y: int) -> None:
        self._record("move_rel", x, y)

    def mouse_is_swapped(self) -> bool:
        self._record("mouse_is_swapped")
        return False

    def __getattr__(self, name: str):
        # New native helpers can be imported by the compatibility layer before
        # a dedicated test assertion is added. They remain inert and recorded.
        def no_op(*args: Any, **kwargs: Any) -> None:
            self._record(name, *args, kwargs)

        return no_op


RUST_STUB = RustStub()
sys.modules["pyautogui._rust_core"] = RUST_STUB

import pyautogui  # noqa: E402  (must follow native-module injection)


class FakePlatform:
    """Stateful implementation of the private platform adapter protocol."""

    def __init__(self) -> None:
        self.position = [320, 240]
        self.screen_size = (1920, 1080)
        self.calls: list[BackendCall] = []
        self.swapped = False
        self.failsafe_triggered = False
        self.keyboardMapping = {key: index + 1 for index, key in enumerate(pyautogui.KEY_NAMES)}

    def _record(self, name: str, *args: Any) -> None:
        self.calls.append(BackendCall(name, args))

    def calls_named(self, name: str) -> list[BackendCall]:
        return [call for call in self.calls if call.name == name]

    def _position(self) -> tuple[int, int]:
        self._record("position")
        return tuple(self.position)

    def _size(self) -> tuple[int, int]:
        self._record("size")
        return self.screen_size

    def _mouse_is_swapped(self) -> bool:
        self._record("mouse_is_swapped")
        return self.swapped

    def _moveTo(self, x: int, y: int) -> None:
        self._record("move_to", x, y)
        self.position[:] = [x, y]

    def _moveRel(self, x: int, y: int) -> None:
        self._record("move_rel", x, y)
        self.position[0] += x
        self.position[1] += y

    def _moveToSmooth(self, x: int, y: int, duration: float, steps: int) -> bool:
        self._record("move_to_smooth", x, y, duration, steps)
        self.position[:] = [x, y]
        return True

    def _consume_failsafe_trigger(self) -> bool:
        self._record("consume_failsafe_trigger")
        triggered = self.failsafe_triggered
        self.failsafe_triggered = False
        return triggered

    def _dragTo(self, x: int, y: int, button: str) -> None:
        self._record("drag_to", x, y, button)
        self.position[:] = [x, y]

    def _mouseDown(self, x: int, y: int, button: str) -> None:
        self._record("mouse_down", x, y, button)

    def _mouseUp(self, x: int, y: int, button: str) -> None:
        self._record("mouse_up", x, y, button)

    def _click(self, x: int, y: int, button: str) -> None:
        self._record("click", x, y, button)

    def _multiClick(self, x: int, y: int, button: str, clicks: int, interval: float = 0.0) -> None:
        self._record("multi_click", x, y, button, clicks, interval)

    def _scroll(self, clicks: int, x: int, y: int) -> None:
        self._record("scroll", clicks, x, y)

    def _hscroll(self, clicks: int, x: int, y: int) -> None:
        self._record("hscroll", clicks, x, y)

    def _vscroll(self, clicks: int, x: int, y: int) -> None:
        self._record("vscroll", clicks, x, y)

    def _keyDown(self, key: str) -> None:
        self._record("key_down", key)

    def _keyUp(self, key: str) -> None:
        self._record("key_up", key)


@pytest.fixture
def rust_stub() -> RustStub:
    RUST_STUB.calls.clear()
    RUST_STUB.failsafe_triggered = False
    return RUST_STUB


@pytest.fixture
def fake_platform() -> FakePlatform:
    return FakePlatform()


@pytest.fixture(autouse=True)
def hermetic_backend(monkeypatch: pytest.MonkeyPatch, fake_platform: FakePlatform, rust_stub: RustStub):
    """Make every unit test side-effect-free by default."""

    monkeypatch.setattr(pyautogui, "platformModule", fake_platform)
    monkeypatch.setattr(pyautogui, "FAILSAFE", False)
    monkeypatch.setattr(pyautogui, "PAUSE", 0.0)
    monkeypatch.setattr(pyautogui, "LOG_SCREENSHOTS", False)
    yield fake_platform
