import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

root = Path(__file__).resolve().parent.parent

ext = Pybind11Extension(
    "classicml_py",
    [str(Path(__file__).parent / "bindings.cpp")],
    include_dirs=[str(root)],
    cxx_std=17,
    language="c++",
)

setup(
    name="classicml",
    version="0.1.0",
    description="Python bindings for ClassicML C++ library",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
