"""Focused tests for the optional Windows Rust acceleration layer.

These tests mock user32 and never generate real keyboard or mouse input.
"""

import importlib
import inspect
import sys
import unittest
from unittest import mock

import pyautogui


@unittest.skipUnless(sys.platform == "win32", "Windows backend test")
class WindowsFallbackTests(unittest.TestCase):
    def setUp(self):
        # The hermetic unit-test fixture replaces pyautogui.platformModule;
        # import the real adapter directly so these tests exercise its
        # optional-native/fallback boundary without generating real input.
        self.backend = importlib.import_module("pyautogui._pyautogui_win")
        self.original_core = self.backend._rust_core
        self.original_disabled = set(self.backend._disabled_rust_functions)

    def tearDown(self):
        self.backend._rust_core = self.original_core
        self.backend._disabled_rust_functions.clear()
        self.backend._disabled_rust_functions.update(self.original_disabled)

    def test_public_signatures_remain_pyautogui_0954_compatible(self):
        self.assertEqual(
            list(inspect.signature(pyautogui.moveTo).parameters),
            ["x", "y", "duration", "tween", "logScreenshot", "_pause"],
        )
        self.assertEqual(
            list(inspect.signature(pyautogui.click).parameters),
            [
                "x",
                "y",
                "clicks",
                "interval",
                "button",
                "duration",
                "tween",
                "logScreenshot",
                "_pause",
            ],
        )

    def test_native_move_failure_executes_ctypes_fallback(self):
        class FailingCore:
            @staticmethod
            def set_cursor_pos(x, y):
                raise OSError("native path failed")

        class User32:
            calls = []

            @classmethod
            def SetCursorPos(cls, x, y):
                cls.calls.append((x, y))
                return 1

        self.backend._rust_core = FailingCore()
        self.backend._disabled_rust_functions.clear()
        with mock.patch.object(self.backend.ctypes.windll, "user32", User32):
            self.backend._moveTo(123, 456)

        self.assertEqual(User32.calls, [(123, 456)])
        self.assertIn("set_cursor_pos", self.backend._disabled_rust_functions)

    def test_ctypes_move_failure_is_reported(self):
        class User32:
            @staticmethod
            def SetCursorPos(x, y):
                return 0

        self.backend._rust_core = None
        self.backend._disabled_rust_functions.clear()
        with mock.patch.object(self.backend.ctypes.windll, "user32", User32), mock.patch.object(
            self.backend.ctypes, "WinError", return_value=OSError("ctypes path failed")
        ):
            with self.assertRaisesRegex(OSError, "ctypes path failed"):
                self.backend._moveTo(1, 2)

    def test_short_native_keyboard_submission_retries_with_ctypes(self):
        class ShortCore:
            @staticmethod
            def send_inputs(events):
                return 0

        class User32:
            calls = []

            @classmethod
            def keybd_event(cls, vk, scan, flags, extra_info):
                cls.calls.append((vk, scan, flags, extra_info))

        self.backend._rust_core = ShortCore()
        self.backend._disabled_rust_functions.clear()
        with mock.patch.object(self.backend.ctypes.windll, "user32", User32):
            self.backend._keyDown("a")

        self.assertEqual(len(User32.calls), 1)
        self.assertEqual(User32.calls[0][2], self.backend.KEYEVENTF_KEYDOWN)
        self.assertIn("send_inputs", self.backend._disabled_rust_functions)

    def test_press_batches_key_down_and_up_in_one_native_call(self):
        class RecordingCore:
            calls = []

            @classmethod
            def send_inputs(cls, events):
                cls.calls.append(tuple(events))
                return len(events)

        self.backend._rust_core = RecordingCore()
        self.backend._disabled_rust_functions.clear()

        self.backend._press("a")

        self.assertEqual(len(RecordingCore.calls), 1)
        events = RecordingCore.calls[0]
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0][3], self.backend.KEYEVENTF_KEYDOWN)
        self.assertEqual(events[1][3], self.backend.KEYEVENTF_KEYUP)

    def test_mouse_error_is_not_swallowed_after_native_fallback(self):
        class FailingCore:
            @staticmethod
            def send_mouse_event(ev, x, y, data):
                raise OSError("native mouse failure")

        class User32:
            @staticmethod
            def GetSystemMetrics(index):
                return 100

            @staticmethod
            def mouse_event(ev, x, y, data, extra_info):
                raise OSError("ctypes mouse failure")

        self.backend._rust_core = FailingCore()
        self.backend._disabled_rust_functions.clear()
        with mock.patch.object(self.backend.ctypes.windll, "user32", User32):
            with self.assertRaisesRegex(OSError, "ctypes mouse failure"):
                self.backend._click(10, 20, pyautogui.LEFT)

    def test_cursor_query_uses_ctypes_when_native_query_fails(self):
        class FailingCore:
            @staticmethod
            def get_cursor_pos():
                raise RuntimeError("native query failure")

        class User32:
            @staticmethod
            def GetCursorPos(point_pointer):
                point_pointer._obj.x = 17
                point_pointer._obj.y = 29
                return 1

        self.backend._rust_core = FailingCore()
        self.backend._disabled_rust_functions.clear()
        with mock.patch.object(self.backend.ctypes.windll, "user32", User32):
            self.assertEqual(self.backend._position(), (17, 29))

    def test_missing_native_core_leaves_original_tween_path_available(self):
        # The smooth function is defined conditionally when the module loads.
        # Its absence is what makes _mouseMoveDrag use the 0.9.54 Python loop.
        if self.original_core is None:
            self.assertFalse(hasattr(self.backend, "_moveToSmooth"))

    def test_native_failsafe_hook_augments_traditional_custom_points(self):
        class InactiveCore:
            @staticmethod
            def check_failsafe_triggered():
                return False

        old_failsafe = pyautogui.FAILSAFE
        old_points = pyautogui.FAILSAFE_POINTS
        self.backend._rust_core = InactiveCore()
        self.backend._disabled_rust_functions.clear()
        pyautogui.FAILSAFE = True
        pyautogui.FAILSAFE_POINTS = [(12, 34)]
        try:
            with mock.patch.object(pyautogui, "platformModule", self.backend), mock.patch.object(
                pyautogui, "position", return_value=(12, 34)
            ):
                with self.assertRaises(pyautogui.FailSafeException):
                    pyautogui.failSafeCheck()
        finally:
            pyautogui.FAILSAFE = old_failsafe
            pyautogui.FAILSAFE_POINTS = old_points

    def test_failed_smooth_fast_path_continues_original_tween_loop(self):
        class Platform:
            smooth_calls = []
            move_calls = []

            @staticmethod
            def _position():
                return (0, 0)

            @staticmethod
            def _size():
                return (100, 100)

            @classmethod
            def _moveToSmooth(cls, x, y, duration, steps):
                cls.smooth_calls.append((x, y, duration, steps))
                return False

            @classmethod
            def _moveTo(cls, x, y):
                cls.move_calls.append((x, y))

        with mock.patch.object(pyautogui, "platformModule", Platform), mock.patch.object(
            pyautogui, "FAILSAFE", False
        ), mock.patch.object(pyautogui.time, "sleep", return_value=None):
            pyautogui.moveTo(50, 40, duration=0.2, _pause=False)

        self.assertEqual(len(Platform.smooth_calls), 1)
        self.assertGreater(len(Platform.move_calls), 1)
        self.assertEqual(Platform.move_calls[-1], (50, 40))


if __name__ == "__main__":
    unittest.main()
