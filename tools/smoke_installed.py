"""Read-only smoke test for an installed wheel.

Run this script outside the source checkout so Python resolves the installed
package. It queries display/cursor state but never emits an input event.
"""

from __future__ import annotations

import pyautogui
from pyautogui import _rust_core

assert pyautogui.__version__ == "0.9.54.1"
assert callable(pyautogui.moveTo)
assert callable(pyautogui.click)
assert callable(pyautogui.hotkey)
backend_info = pyautogui.getBackendInfo()
assert backend_info["native_extension"]
assert backend_info["windows_acceleration"]
assert {"input", "screenshot", "template_matching"}.issubset(backend_info["native_features"])
width, height = pyautogui.size()
assert width > 0 and height > 0
x, y = pyautogui.position()
assert isinstance(x, int) and isinstance(y, int)
assert callable(_rust_core.get_system_metrics)
print(f"PyAutoGUI {pyautogui.__version__}: native wheel loaded; display={width}x{height}, cursor=({x}, {y})")
