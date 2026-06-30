# Execute with python pyat/pyat/tools/build_convolve.py build_ext --inplace
import os
import sys

from setuptools import setup, Extension

# Need pyat_dev environment
# pylint: disable=import-error
from Cython.Build import cythonize
import numpy

C_CONVOLVE_PKGDIR = "src/pyat/common/convolve"

SRC_FILES = [os.path.join(C_CONVOLVE_PKGDIR, filename) for filename in ["src/convolve.c"]]

extra_compile_args = ["-UNDEBUG"]
if not sys.platform.startswith("win"):
    extra_compile_args.append("-fPIC")


# Add '-Rpass-missed=.*' to ``extra_compile_args`` when compiling with clang
# to report missed optimizations
_convolve_ext = Extension(
    name="pyat.common.convolve._convolve",
    sources=SRC_FILES,
    extra_compile_args=extra_compile_args,
    include_dirs=[numpy.get_include()],
    language="c",
)

setup(
    name="My hello app",
    ext_modules=cythonize([_convolve_ext]),
)
