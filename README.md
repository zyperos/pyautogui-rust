# pyautogui-rust

`pyautogui-rust` is a PyAutoGUI-compatible fork with a Rust-accelerated
Windows backend. Existing `import pyautogui` programs retain their public API.

## Compared with PyAutoGUI

| Area | PyAutoGUI | pyautogui-rust |
| --- | --- | --- |
| Public API | Original API | Compatible with PyAutoGUI 0.9.54 |
| Mouse and keyboard | Platform Python backend | Windows `SendInput`, with the same Python calls |
| Screenshot | PyScreeze/Pillow path | Windows GDI capture through Rust, with fallback |
| Image lookup | PyScreeze path | Rust matcher where supported, then PyScreeze fallback |
| Failure behavior | Backend exception | Failed native operation falls back to the established Python path |

The project does not require a driver, background service, elevation, or a
different import name. The native acceleration is Windows-specific; macOS and
Linux retain the compatible Python behavior.

## Performance

The acceleration target is screen capture and image location, rather than every
mouse or keyboard call. A local read-only benchmark on Windows 10 22H2,
CPython 3.12.7, and a 1920x1080 display captured a 320x240 region with these
median times:

| Operation | pyautogui-rust | PyScreeze baseline | Result |
| --- | ---: | ---: | ---: |
| Screenshot | 4.15 ms | 16.84 ms | 4.06x faster |

The complete result is in [`.benchmarks/native-win64.json`](.benchmarks/native-win64.json).
Hardware, Windows configuration, display size, capture region, and Pillow
version affect results. Reproduce it with:

```powershell
python -m pip install -e ".[benchmark]"
python tools/benchmark_native.py --iterations 30 --json .benchmarks/native-win64.json
```

## Usage

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

`getBackendInfo()` is only for diagnostics. Normal automation code does not
need to call it.

## Image location

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
