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

from numpy.testing import assert_almost_equal

from pydiffusion.util import hydropro


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
0.702,                               !partial specific volume, cm3/g
1.0,                                 !solvent density, g/cm3
-1                                   !n_values of q
-1                                   !n_intervals
0,                                   !n_trials for mc calculation of covolume
1                                    !idif=1 (yes) for full diffusion tensors
"""
    c = hydropro.config("6LYZ.pdb", name="foo", radius=2, temperatur=303, viscosity=1)
    assert res == c


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
0.702,                               !partial specific volume, cm3/g
1.0,                                 !solvent density, g/cm3
-1                                   !n_values of q
-1                                   !n_intervals
0,                                   !n_trials for mc calculation of covolume
1                                    !idif=1 (yes) for full diffusion tensors
*                                    !EOF
""".split("\n")
    c = hydropro.config("6LYZ.pdb")
    hydropro.write_config(c, tmpdir.dirname)
    with open(pjoin(tmpdir.dirname, "hydropro.dat")) as f:
        lines = f.readlines()

    for r, l in zip(res, lines):
        assert r == l.strip()


def test_read_diffusion_tensor(data):
    res_tensor = np.array(
        [
            [1.165e-06, -1.519e-08, 5.281e-08, 1.044e-02, 2.228e-02, 1.652e-02],
            [-1.520e-08, 1.146e-06, -1.588e-08, 2.228e-02, -1.056e-03, -8.636e-03],
            [5.281e-08, -1.588e-08, 1.180e-06, 1.651e-02, -8.644e-03, -4.427e-03],
            [1.044e-02, 2.228e-02, 1.651e-02, 2.466e07, -1.917e06, 4.468e06],
            [2.228e-02, -1.056e-03, -8.644e-03, -1.917e06, 2.163e07, -1.584e06],
            [1.652e-02, -8.636e-03, -4.427e-03, 4.468e06, -1.584e06, 2.503e07],
        ]
    )
    tensor = hydropro.read_diffusion_tensor(data["hydropro-res.txt"])
    assert_almost_equal(res_tensor, tensor)


def test_read_center_of_diffusion(data):
    res_cd = np.array([1.994e-07, 1.728e-07, 2.292e-07])
    cd = hydropro.read_center_of_diffusion(data["hydropro-res.txt"])
    assert_almost_equal(res_cd, cd)
