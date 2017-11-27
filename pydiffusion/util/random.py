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
from MDAnalysis.lib import transformations
from scipy._lib._util import check_random_state
import numpy as np


def rotation(angle, random_state=None):
    """random rotation with given angle about a random axis

    Parameters
    ----------
    angle : float
        angle in radian

    Returns
    -------
    M : ndarray (3, 3)
        rotation matrix
    """
    random_state = check_random_state(random_state)
    axis = random_state.normal(size=3)
    r = transformations.rotation_matrix(angle, axis)
    return r[:3, :3]


def random_walk(n_dim, n_frames, random_state=None):
    """
    generate randomwalk without barriers in n_dim dimensions

    Parameters
    ----------
    n_frames : int
        length of trajectory
    n_dim : int
        number of dimensions
    random_state : RandomState (optional)
        numpy RandomState to use. If ``None`` use default RandomState

    Returns
    -------
    ndarray : (n_frames, n_dim, order='F')
        random walk
    """
    random_state = check_random_state(random_state)
    trj = random_state.uniform(low=-1, high=1, size=(n_frames, n_dim))
    trj = np.array(trj, order='F', dtype=np.double)
    return trj.cumsum(0)
