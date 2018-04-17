=============
 PyDiffusion
=============

.. image:: https://travis-ci.org/bio-phys/pydiffusion.svg?branch=master
   :target: https://travis-ci.org/bio-phys/pydiffusion

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/bio-phys/pydiffusion/master?filepath=example%2FAnalysis.ipynb

PyDiffusion is a python library to analyze the rotational and translational
diffusion of molecules in molecular dynamics simulation or rigid body simulations.

INSTALL
=======
**Note**: You need cython and numpy to install pydiffusion

.. code::

   python setup.py install

If you want to install the library locally for your user then append the ``--user``
flag.

Usage
=====

Please refer to the `example notebook <https://github.com/bio-phys/pydiffusion/blob/master/example/Analysis.ipynb>`_.

References
==========

 | M. Linke, J. Köfinger, G. Hummer: Fully Anisotropic Rotational Diffusion Tensor from Molecular Dynamics Simulations. The Journal of Physical Chemistry Part B `(2018, in print)  <https://pubs.acs.org/doi/abs/10.1021/acs.jpcb.7b11988>`_


DEVELOPMENT
===========

To install the library in development mode use

.. code::
   python setup.py develop --user

This will create a python-package-symlink to this folder and every change you
make is directly applied to your installed package.
