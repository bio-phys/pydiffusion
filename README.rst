=============
 PyDiffusion
=============

PyDiffusion is a python library to analyze the rotational and translational
diffusion of molecules in molecular dynamics simulation or rigid body simulations.

INSTALL
=======
**Note**: You need cython and numpy to install pydiffusion

```
python setpy.py install
```

If you want to install the library local for your user then append the `--user`
flag. This is recommended.


DEVELOPMENT
===========

To install the library in development mode use

```
python setup.py develop --user
```

This will create a python-package-symlink to this folder and every change you
make is directly applied to your installed package.
