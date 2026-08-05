"""Setuptools bridge for the PyO3 extension.

Project metadata lives in ``pyproject.toml``. Keeping this file deliberately
small makes legacy editable installs work while setuptools-rust builds the
same ABI3 extension for wheels and source distributions.
"""

from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension(
            "pyautogui._rust_core",
            path="Cargo.toml",
            binding=Binding.PyO3,
            py_limited_api=True,
        )
    ],
    zip_safe=False,
)
