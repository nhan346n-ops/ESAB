from setuptools import setup, Extension, Command

from Cython.Build import cythonize
import numpy
import sys

from setuptools import find_packages
extra_compile_args=['-UNDEBUG']
if not sys.platform.startswith('win'):
    extra_compile_args.append('-fPIC')

_convolve_ext = Extension(name='pyat.common.convolve._convolve', sources=["src/pyat/common/convolve/src/convolve.c"],
                          extra_compile_args=extra_compile_args,
                          include_dirs=[numpy.get_include()],
                          language='c')
_ascii_export=Extension(name="pyat.dtm.export.cython_dtm2ascii_export",sources=["src/pyat/dtm/export/cython_dtm2ascii_export.pyx"],  include_dirs=[numpy.get_include()])


setup(ext_modules=cythonize([_convolve_ext,_ascii_export],language_level="3",force=True, annotate = True))
