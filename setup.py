import sys
from setuptools import setup, find_packages

try:
    import numpy as np
except ImportError:
    print("Need numpy for installation")
    sys.exit(1)

try:
    from Cython.Build import cythonize
except ImportError:
    print("Need cython for installation")
    sys.exit(1)

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

setup(
    name='pydiffusion',
    author='Max Linke',
    version='1.0dev',
    license='GPLv3',
    install_requires=['numpy', 'numba', 'MDAnalysis>=0.17.0', 'joblib', 'scipy>=0.19', 'six'],
    packages=find_packages(),
    include_dirs=[numpy_include],
    ext_modules=cythonize('pydiffusion/quaternionsimulation.pyx')
)
