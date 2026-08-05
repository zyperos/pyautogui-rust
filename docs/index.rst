pyautogui-rust documentation
==============================

``pyautogui-rust`` preserves the PyAutoGUI 0.9.54 public API and adds an
optional Rust acceleration layer on Windows. The import name is always
``pyautogui``. When a native operation is unavailable or fails, that operation
returns to the established Python, Win32 ctypes, or PyScreeze implementation.

Install from PyPI with ``python -m pip install pyautogui-rust``. Python 3.9 or
newer is required. Windows wheels use CPython ABI3.

The project supports standard mouse, keyboard, screenshot, and image-location
workflows on Windows, macOS, and Linux. Windows can additionally use batched
native input, virtual-desktop coordinates, DPI-aware GDI capture, and Rust
template matching. Use ``pyautogui.getBackendInfo()`` to inspect which optional
backend features are available in the current process.

Contents
--------

.. toctree::
   :maxdepth: 2

   install.rst
   quickstart.rst
   mouse.rst
   keyboard.rst
   msgbox.rst
   screenshot.rst
   tests.rst
   roadmap.rst
   source/modules.rst
