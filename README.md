# pyautogui-rust

`pyautogui-rust` is a compatibility-focused PyAutoGUI fork with an optional
Rust acceleration layer for Windows. Existing `import pyautogui` programs keep
their public API and use the established Python backend whenever a native path
is unavailable or fails.

## Why this project

- Drop-in PyAutoGUI 0.9.54 API compatibility
- Windows-native `SendInput`, DPI awareness, virtual-desktop coordinates, and GDI capture
- Rust template matching with per-operation Python/PyScreeze fallback
- Optional TinyLocate CUDA path for large, varied, or animated references
- No driver, service, elevation, or boot-time component
- Python 3.9+ and CPython ABI3 Windows wheels

## Install

```powershell
python -m pip install pyautogui-rust
```

The import name remains `pyautogui`:

```python
import pyautogui

pyautogui.moveTo(500, 300, duration=0.2)
pyautogui.click()
pyautogui.write("hello", interval=0.03)
image = pyautogui.screenshot(region=(0, 0, 800, 600))
box = pyautogui.locateOnScreen("button.png", confidence=0.9)

print(pyautogui.getBackendInfo())
```

`getBackendInfo()` reports the package version, host platform, and the native
features available in the active installation. It performs no mouse or keyboard
input.

## Visual location

Normal `locate*` calls preserve PyScreeze semantics. On compatible Windows
confidence calls, the project can route work through the Rust matcher and falls
back automatically on a miss or backend error. For large, transformed, or
animated references, TinyLocate can use an external model without adding
PyTorch, ONNX, or CUDA packages to the Python runtime dependencies.

```powershell
python tools/install_tinylocate_model.py path\to\tinylocate-v1.tln
python tools/smoke_tinylocate_runtime.py path\to\tinylocate-v1.tln
```

## Reliability contract

- The default tests are hermetic and never emit operating-system input.
- Native errors disable only the failed operation and retain the original
  ctypes/PyScreeze path.
- The release gate runs Python tests, Rust formatting/tests/Clippy, wheel/sdist
  validation, and an installed-wheel smoke test.

See [DEVELOPING.md](DEVELOPING.md) for architecture, local checks, benchmarks,
and release validation. See [docs](docs/index.rst) for API guidance and
[PRODUCT_VISION.md](PRODUCT_VISION.md) for the visual-location routing model.

## Contributing and support

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Security
reports follow [SECURITY.md](SECURITY.md). This project retains the BSD-3-Clause
license and upstream PyAutoGUI attribution in [LICENSE.txt](LICENSE.txt).
