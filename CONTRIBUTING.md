# Contributing

## Before changing code

Keep the established PyAutoGUI API compatible. New behavior should be optional,
observable, and have a clear fallback when Windows-native acceleration is not
available.

## Local checks

```powershell
python -m pip install -e ".[dev,benchmark]"
python tools/check.py --build
```

The default test suite is hermetic and must not emit desktop input. Add unit
tests for new Python behavior and Rust tests for native algorithms or boundary
conditions. Update documentation whenever a public capability, platform limit,
or release requirement changes.

## Pull requests

Describe the user-visible outcome, compatibility impact, fallback behavior, and
the commands used to verify the change. Keep unrelated formatting and generated
artifacts out of a pull request.
