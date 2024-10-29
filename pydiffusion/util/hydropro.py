# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# pydiffusion
# Copyright (c) 2017 Max Linke and contributors
# (see the file AUTHORS for the full list of names)
#
# pydiffusion is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pydiffusion is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pydiffusion.  If not, see <http://www.gnu.org/licenses/>.
from __future__ import absolute_import
from os.path import basename, splitext
import os
import subprocess
import numpy as np


def config(pdb, name=None, radius=None, temperatur=298, viscosity=0.01, model="shell"):
    """Build up particle configuration for hydropro. The important missing part is
    the EOF declaration of the HYDROPRO format. This makes it possible to build
    up a configuration file for several input structures. Use ``write_config``
    to save final configuration.

    Parameter
    ---------
    pdb : str
        pdb-file to be used
    name : str (optional)
    radius : float (optional)
        Radius of spheres used for modeling in Angstrom. If None use suggested default for model
    temperatur : float (optional)
        Temperatur for the diffusion coefficient in Kelvin
    viscosity : float (optional)
        viscosity of solvent in poises
    model : str (optional)
        model used for hydrodynamic calculations. It can be one of 'atom',
        'shell', 'bead'. Refer to HYDROPRO manual for difference between models.

    Returns
    -------
    config : str
        configuration string. Can be saved in a file

    Examples
    --------
    >>> from hummer.rot import hydropro
    >>> c = hydropro.config('6LYZ.pdb')
    >>> c += hydropro.config('1WQP.pdb')
    >>> c += hydropro.config('HEWL.pdb')
    >>> hydropro.write_config(c)

    See Also
    --------
    hummer.hydropro.write_config

    """
    avail_models = {"atom": 1, "shell": 2, "bead": 4}
    radius_suggested = {"atom": 2.9, "shell": 4.8, "bead": 6.1}
    default_conf = """{name}                               !name of molecule
{name}                               !name for output file
{pdb}                                !strucutural (pbd) file
{model}                                    !type of calculation
{radius},                            !aer, radius of primary elements
-1,                                  !nsig
{temperatur:.3f},                        !t (temperature, centigrade)
{viscosity},                         !eta (viscosity of the solvent in poises)
788.,                                !rm (molecular weight)
0.702,                               !partial specific volume, cm3/g
1.0,                                 !solvent density, g/cm3
-1                                   !n_values of q
-1                                   !n_intervals
0,                                   !n_trials for mc calculation of covolume
1                                    !idif=1 (yes) for full diffusion tensors
"""
    if name is None:
        name = splitext(basename(pdb))[0]
    if radius is None:
        radius = radius_suggested[model]
    return default_conf.format(
        name=name,
        pdb=pdb,
        radius=radius,
        temperatur=temperatur - 273,
        viscosity=viscosity,
        model=avail_models[model],
    )


def write_config(conf, folder="."):
    """write HYDROPRO config file. Always write configs using this method because it
    ensure that the config has a valid EOF. It also saves the file with the
    correct filename, hydropro.dat.

    Parameters
    ----------
    conf : str
        configuration str
    folder : str (optional)
        folder where to write config file. Be default save in current folder

    """
    eof = "*                                    !EOF"
    fname = folder + "/hydropro.dat"
    with open(fname, "w") as f:
        f.write(conf + eof)


def run(folder=".", hp_name="hydropro"):
    """Run HYDROPRO in folder

    Parameters
    ----------
    folder : str (optional)
        folder where HYDROPRO should be run in
    hp_name : str (optional)
        name of HYDROPRO executable

    Returns
    -------
    return_code : int
        HYDROPRO return code
    """
    cwd = os.getcwd()
    os.chdir(folder)
    return_code = subprocess.call(hp_name)
    os.chdir(cwd)
    return return_code


def read_diffusion_tensor(fname):
    """parse full 6x6 diffusion tensor from HYDROPRO result file

       Dtt Dtr
       Drt Drr

    Parameters
    ----------
    fname : str
        filename for a HYDROPRO result. Usually ending in *-res.txt

    Returns
    -------
    tensor : ndarray
        6x6 diffusion tensor
    """
    with open(fname) as f:
        tensor = f.readlines()[48:56]
    return np.array([np.float32(t.split()) for t in tensor if len(t) > 4])


def read_center_of_diffusion(fname):
    """parse center of diffusion from HYDROPRO result file

    Parameters
    ----------
    fname : str
        filename for a HYDROPRO result. Usually ending in *-res.txt

    Returns
    -------
    cd : ndarray
        center of diffusion
    """
    with open(fname) as f:
        lines = f.readlines()[39:42]
    return np.float32([l[40:51] for l in lines])
