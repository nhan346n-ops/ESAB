# Execute with python pyat/pyat/tools/build_dtm2ascii.py build_ext --inplace
import sys

from setuptools import setup, Extension

# Need pyat_dev environment
# pylint: disable=import-error
from Cython.Build import cythonize
import numpy


extra_compile_args = ["-UNDEBUG"]
if not sys.platform.startswith("win"):
    extra_compile_args.append("-fPIC")


# Add '-Rpass-missed=.*' to ``extra_compile_args`` when compiling with clang
# to report missed optimizations
_dtm2ascii_ext = Extension(
    name="pyat.dtm.export.cython_dtm2ascii_export",
    sources=["pyat/dtm/export/cython_dtm2ascii_export.pyx"],
    extra_compile_args=extra_compile_args,
    include_dirs=[numpy.get_include()],
    language="c",
)

setup(
    name="cython_dtm2ascii_export",
    ext_modules=cythonize([_dtm2ascii_ext]),
)
