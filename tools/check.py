"""Run the repository's repeatable local quality gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(label: str, command: list[str]) -> None:
    print(f"\n==> {label}\n    {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--python-only", action="store_true", help="skip Rust checks")
    group.add_argument("--rust-only", action="store_true", help="skip Python checks")
    parser.add_argument("--build", action="store_true", help="also build and validate wheel/sdist artifacts")
    args = parser.parse_args()

    if not args.rust_only:
        run(
            "Python bytecode",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "pyautogui",
                "tests/unit",
                "benchmarks",
                "tools",
                "training",
            ],
        )
        run(
            "Ruff",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "pyautogui/_pyautogui_pyscreeze_opt.py",
                "pyautogui/_pyautogui_win.py",
                "tests/unit",
                "benchmarks",
                "tools",
                "training",
                "setup.py",
            ],
        )
        run(
            "Hermetic unit tests",
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=pyautogui",
                "--cov-report=term-missing:skip-covered",
                "--cov-fail-under=60",
            ],
        )
    if not args.python_only:
        run("Rust formatting", ["cargo", "fmt", "--all", "--", "--check"])
        run("Rust tests", ["cargo", "test", "--locked", "--all-targets"])
        run("Rust lint", ["cargo", "clippy", "--locked", "--all-targets", "--", "-D", "warnings"])

    if args.build:
        if args.rust_only:
            parser.error("--build requires Python checks")
        run("Build wheel and source distribution", [sys.executable, "-m", "build"])
        run("Validate distributions", [sys.executable, "tools/verify_dist.py", "dist"])

    print("\nAll requested checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
