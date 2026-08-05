import sys
import os

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "-gui":
        from ._gui import show_window
        show_window()
    else:
        print("Usage: pyautogui -gui")
