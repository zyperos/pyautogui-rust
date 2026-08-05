"""Compatibility contract for the established PyAutoGUI public surface."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import pyautogui

LEGACY_PUBLIC_NAMES = {
    "FAILSAFE",
    "FAILSAFE_POINTS",
    "KEYBOARD_KEYS",
    "PAUSE",
    "Point",
    "PyAutoGUIException",
    "Size",
    "alert",
    "center",
    "click",
    "confirm",
    "countdown",
    "displayMousePosition",
    "doubleClick",
    "drag",
    "dragRel",
    "dragTo",
    "failSafeCheck",
    "getInfo",
    "hold",
    "hotkey",
    "hscroll",
    "keyDown",
    "keyUp",
    "leftClick",
    "locate",
    "locateAll",
    "locateAllOnScreen",
    "locateCenterOnScreen",
    "locateOnScreen",
    "middleClick",
    "move",
    "moveRel",
    "moveTo",
    "mouseDown",
    "mouseUp",
    "onScreen",
    "password",
    "pixel",
    "pixelMatchesColor",
    "position",
    "press",
    "prompt",
    "resolution",
    "rightClick",
    "run",
    "screenshot",
    "scroll",
    "shortcut",
    "size",
    "sleep",
    "tripleClick",
    "typewrite",
    "useImageNotFoundException",
    "vscroll",
    "write",
}

PYAutoGUI_0954_FUNCTION_SIGNATURES = {
    "click": "x=None, y=None, clicks=1, interval=0.0, button=PRIMARY, duration=0.0, tween=linear, logScreenshot=None, _pause=True",
    "countdown": "seconds",
    "displayMousePosition": "xOffset=0, yOffset=0",
    "doubleClick": "x=None, y=None, interval=0.0, button=LEFT, duration=0.0, tween=linear, logScreenshot=None, _pause=True",
    "dragRel": "xOffset=0, yOffset=0, duration=0.0, tween=linear, button=PRIMARY, logScreenshot=None, _pause=True, mouseDownUp=True",
    "dragTo": "x=None, y=None, duration=0.0, tween=linear, button=PRIMARY, logScreenshot=None, _pause=True, mouseDownUp=True",
    "failSafeCheck": "",
    "getInfo": "",
    "getPointOnLine": "x1, y1, x2, y2, n",
    "hold": "keys, logScreenshot=None, _pause=True",
    "hotkey": "*args, **kwargs",
    "hscroll": "clicks, x=None, y=None, logScreenshot=None, _pause=True",
    "isShiftCharacter": "character",
    "isValidKey": "key",
    "keyDown": "key, logScreenshot=None, _pause=True",
    "keyUp": "key, logScreenshot=None, _pause=True",
    "leftClick": "x=None, y=None, interval=0.0, duration=0.0, tween=linear, logScreenshot=None, _pause=True",
    "linear": "n",
    "middleClick": "x=None, y=None, interval=0.0, duration=0.0, tween=linear, logScreenshot=None, _pause=True",
    "mouseDown": "x=None, y=None, button=PRIMARY, duration=0.0, tween=linear, logScreenshot=None, _pause=True",
    "mouseUp": "x=None, y=None, button=PRIMARY, duration=0.0, tween=linear, logScreenshot=None, _pause=True",
    "moveRel": "xOffset=None, yOffset=None, duration=0.0, tween=linear, logScreenshot=False, _pause=True",
    "moveTo": "x=None, y=None, duration=0.0, tween=linear, logScreenshot=False, _pause=True",
    "onScreen": "x, y=None",
    "position": "x=None, y=None",
    "press": "keys, presses=1, interval=0.0, logScreenshot=None, _pause=True",
    "printInfo": "dontPrint=False",
    "raisePyAutoGUIImageNotFoundException": "wrappedFunction",
    "rightClick": "x=None, y=None, interval=0.0, duration=0.0, tween=linear, logScreenshot=None, _pause=True",
    "run": "commandStr, _ssCount=None",
    "scroll": "clicks, x=None, y=None, logScreenshot=None, _pause=True",
    "size": "",
    "sleep": "seconds",
    "tripleClick": "x=None, y=None, interval=0.0, button=LEFT, duration=0.0, tween=linear, logScreenshot=None, _pause=True",
    "typewrite": "message, interval=0.0, logScreenshot=None, _pause=True",
    "useImageNotFoundException": "value=None",
    "vscroll": "clicks, x=None, y=None, logScreenshot=None, _pause=True",
}


def test_established_public_names_are_preserved():
    missing = sorted(name for name in LEGACY_PUBLIC_NAMES if not hasattr(pyautogui, name))
    assert missing == []


def test_every_0954_top_level_function_keeps_its_source_signature():
    tree = ast.parse(Path(pyautogui.__file__).read_text(encoding="utf-8"))
    actual = {
        node.name: ast.unparse(node.args)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in PYAutoGUI_0954_FUNCTION_SIGNATURES
    }
    assert actual == PYAutoGUI_0954_FUNCTION_SIGNATURES


@pytest.mark.parametrize(
    ("function_name", "args", "kwargs"),
    [
        ("moveTo", (100, 200), {"duration": 0.2, "tween": pyautogui.linear}),
        ("moveRel", (10, -20), {"duration": 0.2, "tween": pyautogui.linear}),
        ("dragTo", (100, 200), {"duration": 0.2, "button": "left"}),
        ("dragRel", (10, -20), {"duration": 0.2, "button": "right"}),
        ("click", (100, 200), {"clicks": 2, "interval": 0.01, "button": "left"}),
        ("mouseDown", (), {"x": 100, "y": 200, "button": "left"}),
        ("mouseUp", (), {"x": 100, "y": 200, "button": "left"}),
        ("press", (["ctrl", "c"],), {"presses": 2, "interval": 0.01}),
        ("typewrite", ("hello",), {"interval": 0.01}),
        ("hotkey", ("ctrl", "shift", "s"), {"interval": 0.01}),
        ("screenshot", (), {"imageFilename": "capture.png", "region": (0, 0, 10, 10)}),
        ("locateOnScreen", ("button.png",), {"confidence": 0.9}),
    ],
)
def test_established_call_shapes_still_bind(function_name, args, kwargs):
    """Binding checks compatibility without executing any GUI operation."""

    signature = inspect.signature(getattr(pyautogui, function_name))
    signature.bind(*args, **kwargs)


def test_historical_aliases_keep_their_identity():
    assert pyautogui.move is pyautogui.moveRel
    assert pyautogui.drag is pyautogui.dragRel
    assert pyautogui.write is pyautogui.typewrite
    assert pyautogui.shortcut is pyautogui.hotkey
    assert pyautogui.resolution is pyautogui.size


def test_keyboard_and_button_constants_remain_compatible():
    assert pyautogui.KEYBOARD_KEYS is pyautogui.KEY_NAMES
    assert {pyautogui.LEFT, pyautogui.MIDDLE, pyautogui.RIGHT} == {"left", "middle", "right"}
    for key in ("a", "ctrl", "shift", "enter", "f12", "browserback"):
        assert key in pyautogui.KEYBOARD_KEYS


def test_version_remains_a_pep_440_string():
    from packaging.version import Version

    assert str(Version(pyautogui.__version__)) == pyautogui.__version__


def test_backend_info_reports_loaded_native_capabilities_without_side_effects(rust_stub):
    before = list(rust_stub.calls)

    info = pyautogui.getBackendInfo()

    assert info == {
        "package": "pyautogui-rust",
        "version": pyautogui.__version__,
        "platform": "win32",
        "native_extension": True,
        "windows_acceleration": True,
        "native_features": ("input", "screenshot", "template_matching", "visual_location"),
    }
    assert rust_stub.calls == before
