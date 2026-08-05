# Windows implementation of PyAutoGUI functions.
# BSD license
# Al Sweigart al@inventwithpython.com

import ctypes
import ctypes.wintypes
import sys

if sys.platform != "win32":
    raise Exception("The pyautogui_win module should only be loaded on a Windows system.")

import pyautogui
from pyautogui import LEFT, MIDDLE, RIGHT

# The native extension is an optional acceleration layer. Importing PyAutoGUI
# from a source checkout, or on a Python version for which no wheel was built,
# must continue to use the original ctypes implementation.
try:
    from . import _rust_core
except Exception:
    _rust_core = None


_RUST_UNAVAILABLE = object()
_disabled_rust_functions = set()


def _try_rust(function_name, *args):
    """Call one optional Rust function and report whether it succeeded.

    A native load/runtime failure disables only that fast path. Callers always
    execute their ctypes implementation after a failed attempt, so a native
    error never turns an input operation into a false success.
    """
    if _rust_core is None or function_name in _disabled_rust_functions:
        return _RUST_UNAVAILABLE

    function = getattr(_rust_core, function_name, None)
    if function is None:
        _disabled_rust_functions.add(function_name)
        return _RUST_UNAVAILABLE

    try:
        return function(*args)
    except Exception:
        _disabled_rust_functions.add(function_name)
        return _RUST_UNAVAILABLE


# Fixes scaling issues where PyAutoGUI reports the wrong resolution. Failure
# of the optional native path falls through to the original Win32 call.
if _try_rust("set_process_dpi_aware") is _RUST_UNAVAILABLE:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except AttributeError:
        pass  # Windows XP does not provide SetProcessDPIAware().

# The hook is an optional enhancement; the traditional corner check in
# pyautogui.failSafeCheck() remains active when it is absent.
_try_rust("start_failsafe_hook")


# Event codes passed to the Win32 mouse_event() function.
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_LEFTCLICK = MOUSEEVENTF_LEFTDOWN + MOUSEEVENTF_LEFTUP
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_RIGHTCLICK = MOUSEEVENTF_RIGHTDOWN + MOUSEEVENTF_RIGHTUP
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_MIDDLECLICK = MOUSEEVENTF_MIDDLEDOWN + MOUSEEVENTF_MIDDLEUP

MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_HWHEEL = 0x01000

# Event codes passed to the Win32 keybd_event() function.
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1


# These structures are kept for compatibility with code that imports the
# platform module directly, and match PyAutoGUI 0.9.54's Windows backend.
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.wintypes.ULONG)),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.wintypes.DWORD),
        ("wParamL", ctypes.wintypes.WORD),
        ("wParamH", ctypes.wintypes.DWORD),
    ]


class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("hi", HARDWAREINPUT)]

    _anonymous_ = ("i",)
    _fields_ = [("type", ctypes.wintypes.DWORD), ("i", _I)]


"""Keyboard key mapping for PyAutoGUI.

The dictionaries in the platform backends map a public key name to the value
used by the operating system. Keys remain lowercase and match other backends.
"""
keyboardMapping = dict((key, None) for key in pyautogui.KEY_NAMES)
keyboardMapping.update(
    {
        "backspace": 0x08,
        "\b": 0x08,
        "super": 0x5B,
        "tab": 0x09,
        "\t": 0x09,
        "clear": 0x0C,
        "enter": 0x0D,
        "\n": 0x0D,
        "return": 0x0D,
        "shift": 0x10,
        "ctrl": 0x11,
        "alt": 0x12,
        "pause": 0x13,
        "capslock": 0x14,
        "kana": 0x15,
        "hanguel": 0x15,
        "hangul": 0x15,
        "junja": 0x17,
        "final": 0x18,
        "hanja": 0x19,
        "kanji": 0x19,
        "esc": 0x1B,
        "escape": 0x1B,
        "convert": 0x1C,
        "nonconvert": 0x1D,
        "accept": 0x1E,
        "modechange": 0x1F,
        " ": 0x20,
        "space": 0x20,
        "pgup": 0x21,
        "pgdn": 0x22,
        "pageup": 0x21,
        "pagedown": 0x22,
        "end": 0x23,
        "home": 0x24,
        "left": 0x25,
        "up": 0x26,
        "right": 0x27,
        "down": 0x28,
        "select": 0x29,
        "print": 0x2A,
        "execute": 0x2B,
        "prtsc": 0x2C,
        "prtscr": 0x2C,
        "prntscrn": 0x2C,
        "printscreen": 0x2C,
        "insert": 0x2D,
        "del": 0x2E,
        "delete": 0x2E,
        "help": 0x2F,
        "win": 0x5B,
        "winleft": 0x5B,
        "winright": 0x5C,
        "apps": 0x5D,
        "sleep": 0x5F,
        "num0": 0x60,
        "num1": 0x61,
        "num2": 0x62,
        "num3": 0x63,
        "num4": 0x64,
        "num5": 0x65,
        "num6": 0x66,
        "num7": 0x67,
        "num8": 0x68,
        "num9": 0x69,
        "multiply": 0x6A,
        "add": 0x6B,
        "separator": 0x6C,
        "subtract": 0x6D,
        "decimal": 0x6E,
        "divide": 0x6F,
        "f1": 0x70,
        "f2": 0x71,
        "f3": 0x72,
        "f4": 0x73,
        "f5": 0x74,
        "f6": 0x75,
        "f7": 0x76,
        "f8": 0x77,
        "f9": 0x78,
        "f10": 0x79,
        "f11": 0x7A,
        "f12": 0x7B,
        "f13": 0x7C,
        "f14": 0x7D,
        "f15": 0x7E,
        "f16": 0x7F,
        "f17": 0x80,
        "f18": 0x81,
        "f19": 0x82,
        "f20": 0x83,
        "f21": 0x84,
        "f22": 0x85,
        "f23": 0x86,
        "f24": 0x87,
        "numlock": 0x90,
        "scrolllock": 0x91,
        "shiftleft": 0xA0,
        "shiftright": 0xA1,
        "ctrlleft": 0xA2,
        "ctrlright": 0xA3,
        "altleft": 0xA4,
        "altright": 0xA5,
        "browserback": 0xA6,
        "browserforward": 0xA7,
        "browserrefresh": 0xA8,
        "browserstop": 0xA9,
        "browsersearch": 0xAA,
        "browserfavorites": 0xAB,
        "browserhome": 0xAC,
        "volumemute": 0xAD,
        "volumedown": 0xAE,
        "volumeup": 0xAF,
        "nexttrack": 0xB0,
        "prevtrack": 0xB1,
        "stop": 0xB2,
        "playpause": 0xB3,
        "launchmail": 0xB4,
        "launchmediaselect": 0xB5,
        "launchapp1": 0xB6,
        "launchapp2": 0xB7,
    }
)


def _vk_key_scan(character_code):
    value = _try_rust("vk_key_scan_a", character_code)
    if value is not _RUST_UNAVAILABLE:
        return value
    character = chr(character_code)
    return ctypes.windll.user32.VkKeyScanA(ctypes.wintypes.WCHAR(character))


# Populate the printable ASCII keys exactly as PyAutoGUI 0.9.54 does.
for c in range(32, 128):
    keyboardMapping[chr(c)] = _vk_key_scan(c)


def _build_key_events(key, release):
    if key not in keyboardMapping or keyboardMapping[key] is None:
        return []

    needsShift = pyautogui.isShiftCharacter(key)
    mods, vkCode = divmod(keyboardMapping[key], 0x100)

    events = []
    for apply_mod, vk_mod in (
        (mods & 4, 0x12),
        (mods & 2, 0x11),
        (mods & 1 or needsShift, 0x10),
    ):
        if apply_mod:
            events.append((INPUT_KEYBOARD, vk_mod, 0, KEYEVENTF_KEYDOWN))
    events.append((INPUT_KEYBOARD, vkCode, 0, KEYEVENTF_KEYUP if release else KEYEVENTF_KEYDOWN))
    for apply_mod, vk_mod in (
        (mods & 1 or needsShift, 0x10),
        (mods & 2, 0x11),
        (mods & 4, 0x12),
    ):
        if apply_mod:
            events.append((INPUT_KEYBOARD, vk_mod, 0, KEYEVENTF_KEYUP))
    return events


def _send_key_events(events):
    if not events:
        return

    sent = _try_rust("send_inputs", events)
    if sent is not _RUST_UNAVAILABLE:
        # Older cores return the number submitted; newer cores may validate
        # the count internally and return None. A short submission falls back
        # rather than reporting a successful key operation.
        if sent is None or sent == len(events):
            return
        _disabled_rust_functions.add("send_inputs")

    for _, vk_mod, _, flags in events:
        ctypes.windll.user32.keybd_event(vk_mod, 0, flags, 0)


def _keyDown(key):
    """Performs a keyboard key press without the release."""
    _send_key_events(_build_key_events(key, release=False))


def _keyUp(key):
    """Performs a keyboard key release without pressing it first."""
    _send_key_events(_build_key_events(key, release=True))


def _press(key):
    """Press and release one key in a single optional native batch."""
    events = _build_key_events(key, release=False)
    events.extend(_build_key_events(key, release=True))
    _send_key_events(events)


def _position():
    """Returns the current cursor position as an ``(x, y)`` tuple."""
    result = _try_rust("get_cursor_pos")
    if result is not _RUST_UNAVAILABLE:
        return result

    cursor = ctypes.wintypes.POINT()
    if not ctypes.windll.user32.GetCursorPos(ctypes.byref(cursor)):
        raise ctypes.WinError()
    return cursor.x, cursor.y


def _size():
    """Returns the primary screen size as a ``(width, height)`` tuple."""
    width = _try_rust("get_system_metrics", 0)
    if width is not _RUST_UNAVAILABLE:
        height = _try_rust("get_system_metrics", 1)
        if height is not _RUST_UNAVAILABLE:
            return width, height
    return (
        ctypes.windll.user32.GetSystemMetrics(0),
        ctypes.windll.user32.GetSystemMetrics(1),
    )


def _moveTo(x, y):
    """Moves the mouse cursor to ``(x, y)``."""
    if _try_rust("set_cursor_pos", x, y) is not _RUST_UNAVAILABLE:
        return
    if not ctypes.windll.user32.SetCursorPos(x, y):
        raise ctypes.WinError()


# Expose the smooth fast path only when it was present at import time. This
# keeps the original Python tween loop active in source-only installations.
if _rust_core is not None and callable(getattr(_rust_core, "move_to_smooth", None)):

    def _moveToSmooth(x, y, duration, steps):
        """Runs the optional native linear movement path.

        The bool return is internal: False tells ``_mouseMoveDrag`` to execute
        the original Python tween loop after a native runtime failure.
        """
        return _try_rust("move_to_smooth", x, y, duration, steps) is not _RUST_UNAVAILABLE


def _mouseDown(x, y, button):
    """Sends a mouse-button-down event."""
    if button not in (LEFT, MIDDLE, RIGHT):
        raise ValueError(
            'button arg to _click() must be one of "left", "middle", or "right", not %s'
            % button
        )

    if button == LEFT:
        event = MOUSEEVENTF_LEFTDOWN
    elif button == MIDDLE:
        event = MOUSEEVENTF_MIDDLEDOWN
    else:
        event = MOUSEEVENTF_RIGHTDOWN
    _sendMouseEvent(event, x, y)


def _mouseUp(x, y, button):
    """Sends a mouse-button-up event."""
    if button not in (LEFT, MIDDLE, RIGHT):
        raise ValueError(
            'button arg to _click() must be one of "left", "middle", or "right", not %s'
            % button
        )

    if button == LEFT:
        event = MOUSEEVENTF_LEFTUP
    elif button == MIDDLE:
        event = MOUSEEVENTF_MIDDLEUP
    else:
        event = MOUSEEVENTF_RIGHTUP
    _sendMouseEvent(event, x, y)


def _click(x, y, button):
    """Sends a complete mouse click."""
    if button not in (LEFT, MIDDLE, RIGHT):
        raise ValueError(
            'button arg to _click() must be one of "left", "middle", or "right", not %s'
            % button
        )

    if button == LEFT:
        event = MOUSEEVENTF_LEFTCLICK
    elif button == MIDDLE:
        event = MOUSEEVENTF_MIDDLECLICK
    else:
        event = MOUSEEVENTF_RIGHTCLICK
    _sendMouseEvent(event, x, y)


def _mouse_is_swapped():
    """Returns True when Windows has swapped the primary mouse buttons."""
    result = _try_rust("mouse_is_swapped")
    if result is not _RUST_UNAVAILABLE:
        return result
    return ctypes.windll.user32.GetSystemMetrics(23) != 0


def _consume_failsafe_trigger():
    """Consumes an optional native fail-safe notification.

    The public fail-safe always performs the traditional position check too;
    this hook merely remembers a brief visit to a corner between API calls.
    """
    triggered = _try_rust("check_failsafe_triggered")
    if triggered is _RUST_UNAVAILABLE:
        return False
    if triggered:
        reset = _try_rust("reset_failsafe_triggered")
        if reset is _RUST_UNAVAILABLE:
            _disabled_rust_functions.add("check_failsafe_triggered")
    return bool(triggered)


def _sendMouseEvent(ev, x, y, dwData=0):
    """Sends one mouse event through Rust or the original ctypes backend."""
    assert x is not None and y is not None, "x and y cannot be set to None"
    if _try_rust("send_mouse_event", ev, x, y, dwData) is not _RUST_UNAVAILABLE:
        return

    width, height = _size()
    convertedX = 65536 * x // width + 1
    convertedY = 65536 * y // height + 1
    ctypes.windll.user32.mouse_event(
        ev, ctypes.c_long(convertedX), ctypes.c_long(convertedY), dwData, 0
    )


def _scroll(clicks, x=None, y=None):
    """Sends a vertical mouse-wheel event."""
    startx, starty = _position()
    width, height = _size()

    if x is None:
        x = startx
    elif x < 0:
        x = 0
    elif x >= width:
        x = width - 1

    if y is None:
        y = starty
    elif y < 0:
        y = 0
    elif y >= height:
        y = height - 1

    _sendMouseEvent(MOUSEEVENTF_WHEEL, x, y, dwData=clicks)


def _hscroll(clicks, x, y):
    """Compatibility wrapper for PyAutoGUI 0.9.54 horizontal scrolling."""
    return _scroll(clicks, x, y)


def _vscroll(clicks, x, y):
    """Compatibility wrapper for vertical scrolling."""
    return _scroll(clicks, x, y)


def _moveRel(x, y):
    """Sends one raw relative mouse movement packet."""
    if _try_rust("move_rel", x, y) is not _RUST_UNAVAILABLE:
        return
    ctypes.windll.user32.mouse_event(
        MOUSEEVENTF_MOVE, ctypes.c_long(x), ctypes.c_long(y), 0, 0
    )
