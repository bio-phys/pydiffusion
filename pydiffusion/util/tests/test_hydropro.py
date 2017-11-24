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
import numpy as np
from six.moves import zip
from os.path import join as pjoin

from numpy.testing import assert_array_almost_equal, assert_equal

from hummer.util import hydropro


def test_config():
    res = """foo                               !name of molecule
foo                               !name for output file
6LYZ.pdb                                !strucutural (pbd) file
2                                    !type of calculation
2,                            !aer, radius of primary elements
-1,                                  !nsig
30.000,                        !t (temperature, centigrade)
1,                         !eta (viscosity of the solvent in poises)
788.,                                !rm (molecular weight)
1.702,                               !partial specific volume, cm3/g
1.0,                                 !solvent density, g/cm3
-1                                   !n_values of q
-1                                   !n_intervals
0,                                   !n_trials for mc calculation of covolume
1                                    !idif=1 (yes) for full diffusion tensors
"""
    c = hydropro.config('6LYZ.pdb', name='foo', radius=2, temperatur=303,
                        viscosity=1)
    assert_equal(res, c)


def test_write_config(tmpdir):
    res = """6LYZ                               !name of molecule
6LYZ                               !name for output file
6LYZ.pdb                                !strucutural (pbd) file
2                                    !type of calculation
4.8,                            !aer, radius of primary elements
-1,                                  !nsig
25.000,                        !t (temperature, centigrade)
0.01,                         !eta (viscosity of the solvent in poises)
788.,                                !rm (molecular weight)
1.702,                               !partial specific volume, cm3/g
1.0,                                 !solvent density, g/cm3
-1                                   !n_values of q
-1                                   !n_intervals
0,                                   !n_trials for mc calculation of covolume
1                                    !idif=1 (yes) for full diffusion tensors
*                                    !EOF
""".split('\n')
    c = hydropro.config('6LYZ.pdb')
    hydropro.write_config(c, tmpdir.dirname)
    with open(pjoin(tmpdir.dirname, 'hydropro.dat')) as f:
        lines = f.readlines()

    for r, l in zip(res, lines):
        assert_equal(r, l.strip())


def test_read_diffusion_tensor():
    res_tensor = np.array(
        [[ 1.165E-06,-1.519E-08, 5.281E-08, 1.044E-02, 2.228E-02, 1.652E-02],
         [-1.520E-08, 1.146E-06,-1.588E-08, 2.228E-02,-1.056E-03,-8.636E-03],
         [ 5.281E-08,-1.588E-08, 1.180E-06, 1.651E-02,-8.644E-03,-4.427E-03],
         [ 1.044E-02, 2.228E-02, 1.651E-02, 2.466E+07,-1.917E+06, 4.468E+06],
         [ 2.228E-02,-1.056E-03,-8.644E-03,-1.917E+06, 2.163E+07,-1.584E+06],
         [ 1.652E-02,-8.636E-03,-4.427E-03, 4.468E+06,-1.584E+06, 2.503E+07]])
    fname = 'hummer/simulation/tests/data/hydropro-res.txt'
    tensor = hydropro.read_diffusion_tensor(fname)
    assert_array_almost_equal(res_tensor, tensor)


def test_read_center_of_diffusion():

    res_cd = np.array([1.994E-07, 1.728E-07, 2.292E-07])
    fname = 'hummer/simulation/tests/data/hydropro-res.txt'
    cd = hydropro.read_center_of_diffusion(fname)
    assert_array_almost_equal(res_cd, cd)
