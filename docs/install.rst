.. default-role:: code

============
Installation
============

Install ``pyautogui-rust`` from PyPI. The import name remains ``pyautogui``.

OS-specific instructions are below.

Windows
-------

On Windows, you can use the ``py.exe`` program to run the latest version of Python:

    ``py -m pip install pyautogui-rust``

If you have multiple Python versions installed, select the intended Python 3.9+ interpreter with ``py``. For example:

    ``py -3.12 -m pip install pyautogui-rust``

(This is equivalent to ``python -m pip install pyautogui-rust``.)

macOS
-----

On macOS and Linux, you need to run ``python3``:

    ``python3 -m pip install pyautogui-rust``

If you are running El Capitan and have problems installing pyobjc try:

    ``MACOSX_DEPLOYMENT_TARGET=10.11 pip install pyobjc``

Linux
-----

On macOS and Linux, you need to run ``python3``:

    ``python3 -m pip install pyautogui-rust``

On Linux, additionally you need to install the ``scrot`` application, as well as Tkinter:

    ``sudo apt-get install scrot``

    ``sudo apt-get install python3-tk``

    ``sudo apt-get install python3-dev``

PyAutoGUI install the modules it depends on, including PyTweening, PyScreeze, PyGetWindow, PymsgBox, and MouseInfo.
