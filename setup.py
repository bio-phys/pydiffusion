from setuptools import setup, find_packages

from Cython.Build import cythonize

import numpy as np

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

setup(
    name="pydiffusion",
    author="Max Linke",
    version="1.0dev",
    license="GPLv3",
    install_requires=[
        "numpy",
        "numba",
        "MDAnalysis>=2.3",
        "MDAnalysisTests>=2.3",
        "joblib",
        "scipy",
        "pandas",
        "pytest",
    ],
    packages=find_packages(),
    include_dirs=[numpy_include],
    ext_modules=cythonize(
        "pydiffusion/quaternionsimulation.pyx",
        language_level=3,
    ),
)
