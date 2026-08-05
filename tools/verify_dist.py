"""Validate wheel/sdist structure and release metadata without installing it."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

EXPECTED_NAME = "pyautogui-rust"
EXPECTED_VERSION = "0.9.54.1"


def only(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise AssertionError(f"expected exactly one {description}, found: {[path.name for path in paths]}")
    return paths[0]


def verify_wheel(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_name = only(
            [PurePosixPath(name) for name in names if name.endswith(".dist-info/METADATA")],
            "wheel METADATA file",
        )
        metadata = BytesParser().parsebytes(archive.read(str(metadata_name)))

    assert metadata["Name"] == EXPECTED_NAME
    assert metadata["Version"] == EXPECTED_VERSION
    assert metadata["Requires-Python"] == ">=3.9"
    assert metadata["License-Expression"] == "BSD-3-Clause"
    requirements = [requirement.lower() for requirement in metadata.get_all("Requires-Dist", [])]
    for dependency in ("mouseinfo", "pillow", "pygetwindow", "pymsgbox", "pyscreeze", "pytweening"):
        assert any(requirement.startswith(dependency) for requirement in requirements), f"missing dependency: {dependency}"
    assert set(metadata.get_all("Provides-Extra", [])) == {"benchmark", "dev", "vision"}
    assert "pyautogui/__init__.py" in names

    extension_suffix = ".pyd" if sys.platform == "win32" else ".so"
    native_extensions = [
        name for name in names if name.startswith("pyautogui/_rust_core") and name.endswith(extension_suffix)
    ]
    assert len(native_extensions) == 1, f"native extension entries: {native_extensions}"
    assert not any(name.endswith(".rs") for name in names), "Rust sources belong in the sdist, not the wheel"
    assert not any(name.endswith((".tln", ".onnx", ".pt")) for name in names), (
        "training checkpoints and external visual models must not be embedded in the wheel"
    )

    if sys.platform == "win32":
        assert "-cp37-abi3-win_amd64.whl" in path.name, path.name


def verify_sdist(path: Path) -> None:
    required_suffixes = {
        "Cargo.lock",
        "Cargo.toml",
        "LICENSE.txt",
        "MANIFEST.in",
        "pyproject.toml",
        "setup.py",
        "src/lib.rs",
    }
    with tarfile.open(path, mode="r:gz") as archive:
        names = archive.getnames()

    for suffix in required_suffixes:
        assert any(name == suffix or name.endswith("/" + suffix) for name in names), f"sdist is missing {suffix}"

    forbidden_parts = {"target", "__pycache__", ".git", ".venv"}
    leaked = [name for name in names if forbidden_parts.intersection(PurePosixPath(name).parts)]
    assert leaked == [], f"sdist contains generated/private paths: {leaked[:10]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()

    wheel = only(sorted(args.dist.glob("*.whl")), "wheel")
    sdist = only(sorted(args.dist.glob("*.tar.gz")), "source distribution")
    verify_wheel(wheel)
    verify_sdist(sdist)
    print(f"validated {wheel.name}")
    print(f"validated {sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
