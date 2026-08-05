"""Regression tests for the legacy ``pyautogui.run()`` mini-language."""

from __future__ import annotations

import pytest

import pyautogui


def args_for(fake_platform, name):
    return [call.args for call in fake_platform.calls_named(name)]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("42rest", "42"),
        ("  -3.5rest", "  -3.5"),
        ("+12rest", "+12"),
    ],
)
def test_number_token(source, expected):
    assert pyautogui._getNumberToken(source) == expected


@pytest.mark.parametrize("source", ["", "word", "   ", ".25"])
def test_number_token_rejects_invalid_source(source):
    with pytest.raises(pyautogui.PyAutoGUIException, match="number was expected"):
        pyautogui._getNumberToken(source)


def test_nested_command_tokenization():
    source = " g 10,20  k'ENTER' h'ctrl, c' f2(su g+1,-2) "
    assert pyautogui._tokenizeCommandStr(source) == [
        "g",
        "10",
        "20",
        "k",
        "ENTER",
        "h",
        "ctrl, c",
        "f",
        "2",
        ["su", "g", "+1", "-2"],
    ]


@pytest.mark.parametrize("source", ["x", "g10,+20", "d-1,2", "f2(c"])
def test_invalid_commands_report_the_source_index(source):
    with pytest.raises(pyautogui.PyAutoGUIException, match=r"Invalid command at index \d+"):
        pyautogui._tokenizeCommandStr(source)


def test_run_executes_nested_commands_against_inert_backend(fake_platform, monkeypatch):
    monkeypatch.setattr(pyautogui.time, "sleep", lambda _: None)

    pyautogui.run("g10,20 c k'ENTER' h'ctrl,c' f2(su g+1,-2)")

    assert fake_platform.position == [12, 16]
    assert args_for(fake_platform, "click") == [(10, 20, "left")]
    assert args_for(fake_platform, "scroll") == [(1, 10, 20), (1, 11, 18)]
    assert args_for(fake_platform, "key_down") == [("enter",), ("ctrl",), ("c",)]
    assert args_for(fake_platform, "key_up") == [("enter",), ("c",), ("ctrl",)]


def test_run_restores_global_pause(monkeypatch):
    sleeps = []
    monkeypatch.setattr(pyautogui.time, "sleep", sleeps.append)
    pyautogui.PAUSE = 0.125

    pyautogui.run("p0.5c")

    assert pyautogui.PAUSE == 0.125
    assert 0.5 in sleeps
