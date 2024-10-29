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
from __future__ import division, absolute_import
import numpy as np
from MDAnalysis.analysis import align
from MDAnalysis.analysis.base import AnalysisBase

from .util.timeseries import _acf
from .util.mda import parse_common_selection
from .rotation import pcs
from .util import hydropro

__all__ = ["msd"]


def _msd(x, nlags):
    """
    calculate msd with the normal sum approach
    """
    MSD = np.zeros(nlags)
    for i in range(1, nlags):
        MSD[i] = np.mean(np.sum((x[i:] - x[:-i]) ** 2, 1))
    return MSD


def _msd_fft(x, nlags):
    """
    calculate msd with an fft

    References
    ----------
    Calandrini, V., Pellegrini, E., Calligari, P., Hinsen, K., & Kneller, G. R.
    (2011). nMoldyn-Interfacing spectroscopic experiments, molecular dynamics
    simulations and models for time correlation functions. Collection SFN, 12,
    201–232. http://doi.org/10.1051/sfn/201112010
    """
    N, ndim = x.shape
    x2 = x**2

    zeros = np.zeros(ndim)
    x2 = np.vstack((zeros, x2, zeros))
    x2 = x2.sum(axis=1)

    s_ab = np.sum([_acf(_x, N, demean=False, norm=False) for _x in x.T], axis=0)

    MSD = 2 * x2.sum() + np.cumsum(-x2[:N] - x2[:1:-1])
    return (MSD / (N - np.arange(N)) - 2 * s_ab)[:nlags]


def msd(x, nlags=None, fft=True):
    r"""Calculate the MSD for a 1D trajectory.

    .. math:
    MSD(m) = \frac{1}{N - m} \sum_{k=0}^{N-m-1} [r(k+m) - r(k)]^2

    By default a fast FFT algorithm is used to calculate the MSD. This is an
    approximation that is normally the same as the standard algorithm upto
    numerical round-off errors. It might deviate for short trajectories. This
    algorithm will always calculate the MSD for all values. If you are only
    interested in the first few values the standard algorithm might be faster.

    Parameters
    ----------
    x : array-like
        trajectory. Should either be 1D or 2D with (nframes, ndim)
    nlags : int (optional)
        return MSD for nlags frames
    fft : bool (optional)
        use FFT to calclate MSD

    Returns
    -------
    msd : np.ndarray
        mean square displacement

    References
    ----------
    Calandrini, V., Pellegrini, E., Calligari, P., Hinsen, K., & Kneller, G. R.
    (2011). nMoldyn-Interfacing spectroscopic experiments, molecular dynamics
    simulations and models for time correlation functions. Collection SFN, 12,
    201–232. http://doi.org/10.1051/sfn/201112010

    """
    x = np.asarray(x)
    if nlags is None:
        nlags = len(x) - 1

    if x.ndim == 1:
        x = x.reshape(len(x), 1)
    elif x.ndim > 2:
        raise ValueError("MSD can only be calculated for 1D or 2D arrays")

    if fft:
        return _msd_fft(x, nlags)
    else:
        return _msd(x, nlags)


class TranslationTensor(object):
    def __init__(self, D, R):
        D = np.array(D)
        if D.shape != (3,):
            raise ValueError("D shape should be (3,)")
        self._D = D

        if R is not None:
            R = np.asarray(R)
            if R.shape != (3, 3):
                raise ValueError("R shape should be (3, 3)")
        self._R = R

    @property
    def D(self):
        return self._D

    @property
    def R(self):
        return self._R

    @classmethod
    def from_hydropro(cls, res_file):
        # convert from cm^2/s to AA^2/ns
        transD = hydropro.read_diffusion_tensor(res_file)[:3, :3] * 1e7
        D, R = pcs(transD)[:2]
        return cls(D, R.T)

    @property
    def tensor(self):
        return np.dot(self.R, np.dot(np.eye(3) * self.D, self.R.T))


class ParticleCS(AnalysisBase):
    """Transform a trajectory of a rigid body in the laboratory coordinate system
    into the particle coordinate system (PCS). This removes all rotations and
    recovers the pure translational movement in the PCS.

    Attributes
    ----------
    pcs : ndarray
        trajectory in the PCS

    """

    def __init__(self, mobile, reference=None, rotation=None, **kwargs):
        """Parameter
        ---------
        mobile : mda.Atomgroup
            Atomgroup defining a rigid-body
        reference : mda.Atomgroup, optional
            Reference Atomgroup. If none is given use mobile at time 0
        rotation : array-like, optional
            rotation matrix to transform the particle at time 0 from the LCS to
            the PCS. If none is given the identiy matrix is used (this assume
            the particle is in orientated in the PCS at the beginning)
        start/stop/step : int
            trajectory indexing
        """
        super(ParticleCS, self).__init__(mobile.universe.trajectory, **kwargs)

        self._mobile, self._ref = parse_common_selection(
            mobile.universe, mobile, reference
        )
        if rotation is None:
            self._rotation = np.eye(3)
        else:
            self._rotation = np.asarray(rotation)

    def _prepare(self):
        # The run() method should start one frame after start since I need the
        # first frame for setting up the analysis. So do some manual conversion
        # of the start and step
        # TODO: still correct if we use step argument?
        self.start = self.step
        self._dx = [[0, 0, 0]]
        self._trajectory[self.start - self.step]
        self._pos_prev = self._mobile.positions.copy()

        # now I have to recreate the trajectory slice as we modified start
        slicer = slice(self.start, self.stop, self.step)
        self._sliced_trajectory = self._trajectory[slicer]

        # I'm analysing one frame more then `_setup_frames` thinks
        # FXME: this probably messes with the frame info tooling in mda
        self.n_frames = len(self._sliced_trajectory) + 1
        self.frames = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)

    def _single_frame(self):
        pos = self._mobile.positions.copy()
        com = pos.mean(0)
        com_prev = self._pos_prev.mean(0)
        diff = com - com_prev

        R = align.rotation_matrix(self._pos_prev - com_prev, self._ref.positions)[0]
        self._dx.append(np.dot(diff, np.asarray(R).T))
        self._pos_prev = pos

    def _conclude(self):
        self.pcs = np.dot(np.asarray(self._dx), self._rotation.T).cumsum(0)
