from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np
import sys

if sys.platform == "win32":
    compile_args = ["/O2", "/openmp"]
    link_args = []
else:
    compile_args = ["-O3", "-fopenmp"]
    link_args = ["-fopenmp"]

ext = Extension(
    "parallel_kmeans",
    ["parallel_kmeans.pyx"],
    include_dirs=[np.get_include()],
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(ext_modules=cythonize([ext], compiler_directives={"language_level": "3"}))
