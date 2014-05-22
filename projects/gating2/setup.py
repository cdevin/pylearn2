from distutils.core import setup
#from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup, find_packages, Extension
import numpy


cmdclass = {'build_ext': build_ext}
ext_modules = [Extension("parse_text", ["parse_text.pyx"], include_dirs=[numpy.get_include()])]

setup(
    cmdclass = cmdclass,
    ext_modules = ext_modules,
    name = 'Parse  text',
)
