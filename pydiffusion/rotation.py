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
from __future__ import absolute_import, division, print_function
from six.moves import range

from six import string_types
import itertools
import warnings
from MDAnalysis.analysis import align
from MDAnalysis.analysis.base import AnalysisBase
from joblib import Parallel, delayed
from numba import jit
import numpy as np
from scipy import integrate, linalg
from scipy.special import legendre
from scipy._lib._util import check_random_state

from .util import random
from .util import hydropro
from .util.mda import parse_common_selection


def delta(D):
    """
    hydrodynamic anisotropy for a given rotation diffusion tensor ``D``.

    Parameters
    ----------
    D : array_like
        Diagonalized rotation diffusion tensor of shape (3,)

    Returns
    -------
    delta : float
        anisotropy
    """
    D = np.asarray(D)
    if D.shape != (3,):
        raise ValueError("D must be of shape (3,)")
    return np.sqrt(np.sum(D**2) - D[2] * D[1] - D[2] * D[0] - D[1] * D[0])


def _F(mu):
    # See Garcia Paper
    return -1.0 / 3.0 + np.sum(np.asarray(mu) ** 4)


def _G(mu, d):
    # See Garcia Paper
    D = _D(d)
    dd = delta(d)
    mu = np.asarray(mu)
    s = d[0] * (mu[0] ** 4 + 2 * mu[1] ** 2 * mu[2] ** 2)
    s += d[1] * (mu[1] ** 4 + 2 * mu[2] ** 2 * mu[0] ** 2)
    s += d[2] * (mu[2] ** 4 + 2 * mu[0] ** 2 * mu[1] ** 2)
    return (-D + s) / dd


def _aa(mu, D):
    # See Garcia Paper
    aa = np.empty(5)
    mu = np.asarray(mu)
    f = _F(mu)
    g = _G(mu, D)
    aa[0] = 0.75 * (f + g)
    aa[1] = 3 * np.prod(mu[1:] ** 2)
    aa[2] = 3 * mu[0] ** 2 * mu[2] ** 2
    aa[3] = 3 * np.prod(mu[:-1] ** 2)
    aa[4] = 0.75 * (f - g)
    return aa


def _taus(D):
    # See Garcia Paper
    d = _D(D)
    _delta = delta(D)
    tau = np.empty(5)
    tau[0] = 6 * d - 2 * _delta
    tau[1] = 3 * (d + D[2])
    tau[2] = 3 * (d + D[1])
    tau[3] = 3 * (d + D[0])
    tau[4] = 6 * d + 2 * _delta
    return 1.0 / tau


def _D(D):
    # See Garcia Paper
    return np.sum(D) / 3.0


def rcf(t, D, v_body, rank=2):
    """calculate the analytical rotational correlation function

    Parameters
    ----------
    t: ndarray
       1D-array times to use in seconds
    D: array_like
       1D-array containing the eigenvalues of the diffusion tensor
    v_body : array_like
        1D-array, vector of interest in the body-fixed coordinate system at
        time 0.
    rank : int (optional)
        rank of the rcf. Only first and second are supported

    Returns
    -------
    rcf : ndarray
         1D-array with the rotational correlation function

    References
    ----------
    Favro, L. D. (1960). Theory of the rotational Brownian motion of a free
    rigid body. Physical Review, 119(1), 53. http://doi.org/10.1007/BF00815660

    Garcia De La Torre, J., Harding, S. E., & Carrasco, B. (1999). Calculation
    of NMR relaxation, covolume, and scattering-related properties of bead
    models using the SOLPRO computer program. European Biophysics Journal, 28,
    119–132. https://doi.org/10.1007/s002490050191

    """
    t = np.asarray(t)
    D = np.asarray(D)
    v_body = np.asarray(v_body)
    v_body /= np.linalg.norm(v_body)

    if rank == 1:
        a = v_body**2
        # D_y + D_z and all the other combinations of 2 components of D
        r = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        tau = 1.0 / np.dot(r, D)
        return np.sum(a * np.exp(-t[:, np.newaxis] / tau), axis=1)
    elif rank == 2:
        aa = _aa(v_body, D)
        tau = _taus(D)
        return np.sum(aa * np.exp(-t[:, np.newaxis] / tau), axis=1)
    else:
        raise NotImplementedError("only 1. and 2. order correlation are available")


def tau_1(D, v_body="average"):
    """Calculate analytical rotational correlation time of P_1

    Parameters
    ----------
    D : array_like (3, )
        Diffusion tensor
    v_body : array_like (3, ), optional
        body fixed reference vector. If its 'average' use equation in Linke et al.

    Returns
    -------
    correlation time : float


    See Also
    --------
    tau_2

    References
    ----------
    Dlugosz et al. 2014

    Max Linke, Jürgen Köfinger, Gerhard Hummer (2017) in preparation
    """
    if isinstance(v_body, string_types) and v_body == "average":
        el1 = 1 / (3 * D[1] * (1 + D[2] / D[1]))
        el2 = 1 / (3 * D[0] * (1 + D[2] / D[0]))
        el3 = 1 / (3 * D[0] * (1 + D[1] / D[0]))
        return el1 + el2 + el3
    else:
        v_body = np.asarray(v_body, dtype=float)
        v_body /= np.linalg.norm(v_body)
        a = v_body**2
        taus = 1.0 / np.array([D[1] + D[2], D[0] + D[2], D[0] + D[1]])
        return np.sum(a * taus)


def tau_2(D, v_body="average"):
    """Calculate analytical rotational correlation time of P_2

    Parameters
    ----------
    D : array_like (3, )
        Diffusion tensor
    v_body : array_like (3, ), optional
        body fixed reference vector. If its 'average' use equation in Linke et al.

    Returns
    -------
    correlation time : float

    See Also
    --------
    tau_1

    References
    ----------
    Garcia De La Torre, J., Harding, S. E., & Carrasco, B. (1999). Calculation
    of NMR relaxation, covolume, and scattering-related properties of bead
    models using the SOLPRO computer program. European Biophysics Journal, 28,
    119–132. https://doi.org/10.1007/s002490050191

    Max Linke, Jürgen Finger, Gerhard Hummer (2017) in preparation
    """
    # catch isotropic case here because equation doesn't divide by 0 in this case
    if (isinstance(v_body, string_types) and v_body == "average") or D[0] == D[1] == D[
        2
    ]:
        DD = _D(D)
        tau = np.sum([1 / (DD + d) for d in D]) + np.sum(D) / (
            D[0] * D[1] + D[1] * D[2] + D[2] * D[0]
        )
        return tau / 15
    else:
        v_body = np.asarray(v_body, dtype=float)
        v_body /= np.linalg.norm(v_body)
        return np.sum(_aa(v_body, D) * _taus(D))


def hmc_time(D):
    """
    harmonic mean relaxation time

    See Also
    --------
    tau_1
    tau_2

    References
    ----------
    Garcia De La Torre, J., Harding, S. E., & Carrasco, B. (1999). Calculation
    of NMR relaxation, covolume, and scattering-related properties of bead
    models using the SOLPRO computer program. European Biophysics Journal, 28,
    119–132. https://doi.org/10.1007/s002490050191
    """
    return 1 / (2 * D.sum())


def p_l(x, l=2):
    r"""
    Calculate the l-th legendre polynominal of a timeseries

    Parameters
    ----------
    x : array_like
        values to evaluate at
    l : int, optional
        give the order of the legendre polynominal

    Returns
    -------
    P : ndarray
        Legendre polynominal for timeseries

    See Also
    --------
    cos_t
    """
    return legendre(l)(x)


def cos_t(R, x0):
    r"""
    calculate Cosine \Theta for any given vector and rotation matrices

    Parameters
    ----------
    R : ndarray
        3D array of shape (T, 3, 3) containing T rotation matrices.
    x : array_like
        1D array of unit vector in coordinate system of interest

    Returns
    -------
    cos_t: ndarray
        1D-Array with cos(\theta_t) for all t
    """
    x = np.dot(x0, R)
    return np.sum(x0 * x, axis=1)


def rotation_matrix(mobile, ref, weights=None):
    """
    Calculate rotation matrix between two atom groups

    Parameters
    ----------
    mobile : mda.AtomGroup
        mobile atomgroup
    ref : mda.AtomGroup
        reference atomgroup

    Returns
    -------
    R : ndarray (3, 3)
        matrix to turn mobile coordinates to ref
    """
    xyz = mobile.positions - mobile.center(weights)
    ref_xyz = ref.positions - ref.center(weights)
    return np.array(align.rotation_matrix(ref_xyz, xyz, weights=weights)[0])


class RotationMatrix(AnalysisBase):
    """Calculate the rotation matrices required to turn from the initial structure
    to the trajectory at any time t. This is useful to look at the time
    development of the rotations in the body fixed coordinate system of the
    diffusion operator at time 0.

    Attributes
    ----------
    R : ndarray
        rotation matrix for every frame

    """

    def __init__(
        self, mobile, ref=None, weights=None, align_first_to_ref=True, **kwargs
    ):
        """Parameters
        ----------
        mobile : mda.AtomGroup
            calculate rotation matrix for this selection
        ref : mda.AtomGroup
            reference structure to align against
        weights : str, array_like (optional)
            weights used for RMSD fit. If 'mass' use massse, if `None` use equal
            weights or specify array_like with arbitrary weights.
        align_first_to_ref : bool (optional)
            Align first frame in mobile to ref. This ensure that all other
            rotations are recorded in the coordinate system of the reference.
        **kwargs : dict
            arguments to pass to AnalysisBase

        """
        super(RotationMatrix, self).__init__(mobile.universe.trajectory, **kwargs)
        self._mobile, self._ref = parse_common_selection(
            mobile.universe, mobile, ref=ref
        )
        if isinstance(weights, string_types) and weights == "mass":
            weights = self._ref.masses
        self._weights = weights
        self._align_first_to_ref = align_first_to_ref

    def _prepare(self):
        self.R = []
        self._frames = []
        # store the rotation matrix to the reference. This way it is later
        # possible to move all other frames into the coordinate system of the
        # reference
        if self._align_first_to_ref:
            # reset trajectory frame to start
            self._trajectory[self.start]
            self._first_rot = rotation_matrix(self._ref, self._mobile, self._weights)
        else:
            self._first_rot = np.eye(3)

    def _single_frame(self):
        if self._align_first_to_ref:
            # self._mobile.translate(-self._mobile.center_of_geometry())
            self._mobile.rotate(self._first_rot, self._mobile.center_of_geometry())
        self.R.append(rotation_matrix(self._ref, self._mobile, self._weights))
        self._frames.append(self._mobile.universe.trajectory.frame)

    def _conclude(self):
        self.R = np.asarray(self.R)
        self.frames = np.asarray(self._frames)


def is_right_handed(M):
    """Is matrix right handed"""
    det = np.linalg.det(M)
    return np.allclose(det / np.abs(det), 1)


def make_right_handed(M):
    """
    excepts vectors in M are column vectors
    """
    if is_right_handed(M):
        return M
    # don't modify original
    M = M.copy()
    for i in range(3):
        M[:, i] *= -1
        if is_right_handed(M):
            return M
    raise RuntimeError("Couldn't find simple conversion to right-hand system")


def pcs(D):
    """Take an arbitrary Diffusion 3x3 diffusion tensor and convert it into the
    diagional coefficient in the particle coordinate system (PCS). As well
    matrices to convert between the PCS and the Lab system, the original
    coordinate system for D. This function ensures that ``toPCS`` is right
    handed.

    Parameters
    ----------
    D : array-like
        3x3 diffusion tensor

    Returns
    -------
    D : ndarray (3)
        diagional elements of D in PCS
    toPCS : ndarray (3, 3)
        matrix to transform coordinates into the PCS

    """
    Dr, V = np.linalg.eig(D)
    idx_sort = np.argsort(Dr)[::-1]
    Dr = Dr[idx_sort]
    V = make_right_handed(V[:, idx_sort])
    return Dr, V


def rotations_at_t(R, t, step=None):
    """Given rotation matrices from a simulation with T steps calculate the
    rotation for all pairs ``t`` steps apart compared at time 0. This uses the
    common 'window' technique to emulate independent runs from a single
    trajectory.


    Parameters
    ----------
    R : ndarray (T, 3, 3)
        T 3x3 rotation matrices
    t : int
        timestep
    step : int (optional)
        steps between windows

    Returns
    -------
    rots : ndarray (T-t, 3, 3)
        rotations takes in R at a t step interval

    """
    if t < 0 or not isinstance(t, (int, np.uint)):
        raise ValueError("t must be positive integer")
    if t >= len(R):
        raise ValueError("t must smaller then len(R)")

    R = np.asarray(R)

    if len(R.shape) != 3 or (R.shape[1:] != (3, 3)):
        raise ValueError("shape of R has to be (T, 3, 3)")

    if len(R) == 1:
        raise ValueError("len(R) > 1, need more then one rotation matrix")

    # This results in R * R.T for an array of 3x3 matrices
    einsum = "kil,kjl->kji"
    if t == 0:
        return np.einsum(einsum, R[::step], R[::step])
    else:
        return np.einsum(einsum, R[t::step], R[:-t:step])


def _quaternion_covariance(R, t, step=None):
    """calculate rotational covariance in quaternion space for a
    trajectory of rotations or a single rotation

    Parameters
    ----------
    R : ndarray (T, 3, 3)
        T 3x3 rotation matrices
    t : int
        timestep
    step : int (optional)
        steps between windows

    Returns
    -------
    u : ndarray (4, 4)
        quaternion covariance
    """
    if not len(R) >= 2:
        raise ValueError(
            "len(R) needs to be greater then one to " " calculate a variance"
        )

    # calculate mean rotation matrix t steps appart
    R = rotations_at_t(R, t, step).mean(0)

    u = np.zeros((4, 4))

    u[0, 0] = 1 + np.trace(R)
    u[0, 1] = R[1, 2] - R[2, 1]
    u[0, 2] = R[2, 0] - R[0, 2]
    u[0, 3] = R[0, 1] - R[1, 0]

    u[1, 1] = 1 + R[0, 0] - R[1, 1] - R[2, 2]
    u[1, 2] = R[0, 1] + R[1, 0]
    u[1, 3] = R[0, 2] + R[2, 0]

    u[2, 2] = 1 - R[0, 0] + R[1, 1] - R[2, 2]
    u[2, 3] = R[1, 2] + R[2, 1]

    u[3, 3] = 1 - R[0, 0] - R[1, 1] + R[2, 2]

    # copy diagonal elements
    u += np.triu(u, 1).T

    return u / 4.0


def quaternion_covariance(R, t, step=None, n_jobs=1, **kwargs):
    """calculate rotation covariance in quaternion space  upto step t

    Parameters
    ----------
    R : ndarray (T, 3, 3)
        T 3x3 rotation matrices
    t : int
        timestep
    step : int (optional)
        steps between windows
    n_jobs : int (optional)
        number of processes used for calculation. -1 uses all available cores
    **kwargs : (optional)
        joblib.Parallel kwargs besides n_jobs

    Returns
    -------
    u : ndarray(t, 4, 4)
        quaternion covariance upto time t

    """
    R = np.asarray(R)
    # split this to avoid joblib warnings
    if n_jobs == 1:
        u = [_quaternion_covariance(R, i, step) for i in range(t)]
    else:
        u = Parallel(n_jobs=n_jobs, **kwargs)(
            delayed(_quaternion_covariance)(R, i, step) for i in range(t)
        )
    return np.asarray(u)


@jit(forceobj=True)
def chi2(obs, model, time=None):
    """chi2 function for comparing two quaternion correlation functions

    Parameters
    ----------
    obs : array_like (3, 3, T)
        experimental correlations of length T
    model : class:`RotationTensor`
    time : array_like (T, ) (optional)
        measurement time points. If ``None`` assume dt=1

    Returns
    -------
    chi2 value

    """
    if time is None:
        time = np.arange(obs.shape[2])
    ref = moment_2(time, model)
    m4 = moment_4(time, model)
    var = m4 - ref**2

    # if t == 0 then the var_ij == 0 and chi2 will be NaN
    if np.isclose(time[0], 0):
        ref = ref[:, :, 1:]
        obs = obs[:, :, 1:]
        var = var[:, :, 1:]

    T = ref.shape[-1]
    s = 0
    for t in range(T):
        for i in range(3):
            for j in range(i, 3):
                s += (ref[i, j, t] - obs[i, j, t]) ** 2 / var[i, j, t]

    return s


class RotationTensor(object):
    def __init__(self, D, R):
        D = np.array(D)
        if D.shape != (3,):
            raise ValueError("D shape should be (3,)")
        self._D = D

        if R is not None:
            R = np.asarray(R)
            if R.shape != (3, 3):
                raise ValueError("V shape should be (3, 3)")
        self._R = R

    @property
    def D(self):
        return self._D

    @property
    def R(self):
        return self._R

    @classmethod
    def from_hydropro(cls, res_file):
        # convert from 1/s to 1/ns
        rotD = hydropro.read_diffusion_tensor(res_file)[3:, 3:] * 1e-9
        D, R = pcs(rotD)[:2]
        return cls(D, R.T)

    @property
    def tensor(self):
        return np.dot(self.R, np.dot(np.eye(3) * self.D, self.R.T))

    def __getstate__(self):
        return (self.D.tolist(), self.R.tolist())

    def __setstate__(self, state):
        self._D = np.array(state[0])
        self._R = np.array(state[1])


def sigma_rotation_tensor(a, b):
    """compare 2 models"""

    def _D_in_lab_frame(D, R):
        return np.dot(R, np.dot(np.eye(3) * D, R.T))

    D_a = _D_in_lab_frame(a.D, a.R)
    D_b = _D_in_lab_frame(b.D, b.R)
    return np.linalg.norm(D_a - D_b, ord="fro")


def moment_2(time, model):
    """analytic correlations of quaternion components for rotations in arbitrary
    coordinate systems

    Parameters
    ----------
    time : array_like (T, )
        timepoints to evaluate of length T
    model : class:`RotationTensor`
        RotationTensor

    Returns
    -------
    correlation : ndarray (3, 3, T)

    """

    def _msd_q(D, t):
        """analytical correlation of rotations in quaternion space.
        Quaternions w+ix+jy+kz are represented as [w, x, y, z].
        The rotations are calculated in the principle axes coordinate system

        Parameters
        ----------
        D : array_like
            Diagonalized rotation diffusion tensor of shape (3,)
        t : array_like
            times at which to evaluate the msd

        Returns
        -------
        msd : ndarray (3, len(t))

        """
        D = np.asarray(D)
        t = np.asarray(t)
        b1 = [
            np.exp(D[0] * t) + np.exp(D[1] * t) + np.exp(D[2] * t),
            np.exp(D[0] * t) - np.exp(D[2] * t) - np.exp(D[1] * t),
            np.exp(D[1] * t) - np.exp(D[0] * t) - np.exp(D[2] * t),
            np.exp(D[2] * t) - np.exp(D[1] * t) - np.exp(D[0] * t),
        ]
        return 0.25 * (1 + np.exp(-D.sum() * t)[np.newaxis, :] * b1)

    time = np.asarray(time)
    msd = _msd_q(model.D, time)
    res = np.zeros((3, 3, len(time)))
    for i in range(3):
        res[i, i] = msd[i + 1]

    if model.R is not None:
        # res = R^T res R, for all 3x3 matrices in res at the same time
        res = np.dot(model.R.T, np.dot(model.R.T, res))

    return res


# TODO: Implement in cython for inclusion in MDAnalysis
@jit()
def _change_frame_m4(m4, R):
    """rotate the 4th moment into the right coordinate system. this gets super fast
    thanks to numba"""
    time = m4.shape[2]
    u4 = np.zeros((3, 3, time))
    for t in range(time):
        for i in range(3):
            for j in range(3):
                # sum_k q_k^4
                for k in range(3):
                    u4[i, j, t] += m4[k, k, t] * R[k, i] ** 2 * R[k, j] ** 2
                # sum_{m<n}
                for m in range(3):
                    for n in range(3):
                        if m < n:
                            u4[i, j, t] += m4[m, n, t] * (
                                R[m, i] ** 2 * R[n, j] ** 2
                                + R[n, i] ** 2 * R[m, j] ** 2
                                + 4 * R[m, i] * R[n, i] * R[m, j] * R[n, j]
                            )

    return u4


def moment_4(time, model):
    """4th moment quaternion term"""

    def qiiii(D, t, ave_D, cosh, perm):
        """<q_i^4>"""
        mut_D = D[perm]
        expD = np.exp(-3 * ave_D * t) * (
            np.exp(mut_D[0] * t) - np.exp(mut_D[1] * t) - np.exp(mut_D[2] * t)
        )
        exp3D = np.exp(-3 * ave_D * t) * (
            np.exp(-3 * mut_D[0] * t)
            - np.exp(-3 * mut_D[1] * t)
            - np.exp(-3 * mut_D[2] * t)
        )
        cosh = np.exp(-6 * ave_D * t) * cosh
        return 1 / 8 * (1 + 3 / 2 * expD + 1 / 2 * exp3D + cosh)

    def qiijj(D, t, aveD, cosh, sinh, _delta, i, j):
        """<q_i^2 q_j^2>"""
        index = [k for k in range(3) if k not in (i, j)][0]
        d = D[index]

        # Catch numerical error in case of isotropic tensor
        if np.isclose(_delta, 1e-12):
            sinhcosh = 2 * t * (d - aveD) - 1 / 3 * cosh
        else:
            sinhcosh = (d - aveD) * sinh / _delta - 1 / 3 * cosh
        exp = np.exp(-3 * aveD * t) * (np.exp(d * t) - np.exp(-3 * d * t))
        return 1 / 8 * (1 / 3 - 1 / 2 * exp + np.exp(-6 * aveD * t) * sinhcosh)

    aveD = np.sum(model.D) / 3

    _delta = delta(model.D)
    cosh = np.cosh(2 * time * _delta)
    sinh = np.sinh(2 * time * _delta)

    time = np.asarray(time)
    m4 = np.empty((3, 3, len(time)))

    PERMUTATIONS = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    for i, j in itertools.combinations_with_replacement(range(3), 2):
        if i == j:
            m4[i, j] = qiiii(model.D, time, aveD, cosh, PERMUTATIONS[i])
        else:
            m4[i, j] = qiijj(model.D, time, aveD, cosh, sinh, _delta, i, j)
            m4[j, i] = m4[i, j]

    if model.R is not None:
        m4 = _change_frame_m4(m4, model.R)

    return m4


def tau_laplace(uu, time, s):
    """calculate tau matrix with Laplace transformation

    Parameters
    ----------
    uu : ndarray (4, 4, t)
         observed quaternion correlations
    time : ndarray (t, )
         time points of observation
    s : float
         laplace constant

    Returns
    -------
    ndarray (4, 4)
         tau matrix


    See Also
    --------
    fit_laplace
    """
    mat = (4 * uu - np.eye(3)[:, :, np.newaxis]) * np.exp(-time * s)

    t = np.zeros((3, 3))
    t[0, 0] = integrate.simpson(mat[0, 0], x=time)
    for i, j in itertools.combinations_with_replacement(range(3), 2):
        t[i, j] = integrate.simpson(mat[i, j], x=time)

    return t + np.triu(t, 1).T


def fit_laplace(corr, time, s):
    """
    estimate decay time from u_00 and use that as filter value

    Parameters
    ----------
    corr : ndarray (4, 4, t)
         correlations
    dt : float
         time step
    s : float (optional)
         s value in transform. If ``None`` guess based on inverse decay time of <u_0 u_0>
    mult : int (optional)
         how many correlation lengths should be fitted if ``s=None``
    full : bool (optional)
         return s value if ``True``

    Returns
    -------
    RotationTensor
        fitted rotation tensor
    float
        value of s if ``full`` is ``True``

    See Also
    --------
    tau_laplace

    """

    tau = tau_laplace(corr, time, s)

    val, vec = linalg.eig(tau)

    idx_sort = np.argsort(val)[::-1]
    val = np.real(val[idx_sort])
    vec = np.real(vec[:, idx_sort])

    D = np.empty(3)
    D[0] = (
        -1 / (val[0] + val[1]) + 1 / (val[1] + val[2]) - 1 / (val[2] + val[0]) - s / 2.0
    )
    D[1] = (
        -1 / (val[0] + val[1]) - 1 / (val[1] + val[2]) + 1 / (val[2] + val[0]) - s / 2.0
    )
    D[2] = (
        +1 / (val[0] + val[1]) - 1 / (val[1] + val[2]) - 1 / (val[2] + val[0]) - s / 2.0
    )

    R = make_right_handed(vec)
    return RotationTensor(D, R.T)


def metropolis(deltaE, beta):
    """metropolis acceptance criteria"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return min(1, np.exp(-beta * deltaE))


def inf_generator(start, stop, step):
    cond = min if step > 0 else max
    start -= step
    while True:
        start += step
        yield cond(start, stop)


def anneal(
    obs,
    time,
    D=None,
    R=None,
    maxiter=500,
    niter_success=5,
    verbose=False,
    eps=1,
    beta_mima=(0.1, 20, 0.05),
    D_mima=(0.5, 0.05, -0.005),
    angle_mima=(90, 1, -0.5),
    switch=20,
    random_state=None,
    cylinder_fit=False,
):
    """Fit a rotation diffusion tensor to a quaterion covariance with simulated
    annealing.

    The annealing work by alternating between suggesting moves for the
    diffusion coefficients and the orientation. The frequency how often this
    switch happens can be adjusted with the ``switch`` variable. The suggested
    stepsize for D is given as a ratio of the current value of D. This ensure
    we can get good estimates indenpendt of the chosen time scale.

    It is recommened to run this function a couple of times to find the global
    minimum.

    Parameters
    ----------
    obs : array_like (3, 3, T)
        T observations
    time : array_like (T,)
        time points of observations
    D : array_like (3, ) (optional)
        starting value of D. Default (1e8, 1e8, 1e7).
    R : array_like (3, 3) (optional)
        starting orientation. Default np.eye(3)
    maxiter : int (optional)
        maximal number of iterations
    niter_success : int (optional)
        stop iteration if we made ``niter_success`` failed attempts in a row
        with a delta energy below eps.
    eps : float (optional)
        energy delta to stop iterations with ``niter_success``
    switch : int (optional)
        how many iterations until switching between steps in D or R
    random_state : int or np.random.RandomState (optional)
        set random state. Good for reproducable results
    verbose : bool (optional)
        print detailed statistics after every iteration. See Notes
    beta_mima : (start, stop, step) (optional)
        inverse temperature cooling
    D_mima : (start, stop, step) (optional)
        scaled suggestions for D
    angle_mima : (start, stop, step) (optional)
        maximal angle of rotation for R

    Returns
    -------
    RotationTensor
    chi2

    Notes
    -----
    The verbose log is using a shorthand to fit the statistics on a single
    line. `p` is the current probability to accept the step. `b` is the current
    inverse temperature. `ta` is the total acceptance rate, `ma` acceptance
    rate in the current mode, this variable is reset after switching mode. `c`
    current count if we stop the iteration early. `m` current mode. `dE`
    current delta energy.

    """
    random_state = check_random_state(random_state)

    if D is None:
        D = (1e8, 1e8, 1e7)
    if R is None:
        R = np.eye(3)
    model = RotationTensor(D, R)

    beta = inf_generator(*beta_mima)
    D_scale = inf_generator(*D_mima)
    angle_scale = inf_generator(*angle_mima)

    min_val = chi2(obs, model, time)
    global_min = min_val
    global_model = model

    mode = "D"

    # variables to track progress of annealing
    count = 0
    tacc = 0
    macc = 0
    for i in range(maxiter):
        if mode == "D":
            size = next(D_scale) * model.D
            Dnew = np.sort(model.D + size * random_state.uniform(-0.5, 0.5, size=3))[
                ::-1
            ]
            new_model = RotationTensor(Dnew, model.R)

            if cylinder_fit:
                Dnew = np.asarray([Dnew[0], Dnew[1], Dnew[1]])
                new_model = RotationTensor(Dnew, model.R)

            val = chi2(obs, new_model, time)

            # D_cylinder = np.array([Dnew[0], np.mean(Dnew[1:]), np.mean(Dnew[1:])])
            # new_model_cylinder = RotationTensor(D_cylinder, model.R)
            # val_cy = chi2(obs, new_model_cylinder, time)
            # if val_cy < val:
            #     new_model = new_model_cylinder
            #     val = val_cy

            # do a little shake to help float math along
            # D_iso = random_state.uniform(-1e-5, 1e5, size=3) + np.ones(3) * np.mean(Dnew)
            # new_model_iso = RotationTensor(D_iso, model.R)
            # val_iso = chi2(obs, new_model_iso, time)
            # if val_iso < val:
            #     new_model = new_model_iso
            #     val = val_iso
        else:
            max_angle = np.deg2rad(next(angle_scale))
            dR = random.rotation(
                angle=np.random.uniform(0, max_angle), random_state=random_state
            )
            R = make_right_handed(np.dot(model.R, dR).T).T
            new_model = RotationTensor(model.D, R)
            val = chi2(obs, new_model, time)

        dE = val - min_val
        b = next(beta)
        if random_state.uniform(0, 1) < metropolis(dE, b):
            model = new_model
            min_val = val
            count = 0
            tacc += 1
            macc += 1
        else:
            # At the beginning when large steps are allowed we often throw away
            # a lot of attempts with a very large delta energy. This is
            # especially true when we only update R. But we only want to stop
            # prematurely when we actually reduced the stepsize so much that we
            # anyway only explore a nearby region.
            if np.abs(dE) < eps * val:
                count += 1
            else:
                count = 0

        # We seem to have found a local minima and don't attempt any more large
        # steps
        if count > niter_success:
            break

        if (i + 1) % switch == 0:
            mode = "R" if mode == "D" else "D"
            macc = 0

        if verbose:
            total_acc_prop = tacc / (i + 1)
            mode_acc_prop = macc / ((i + 1) % switch + 1)
            print(
                "{:3d}: chi2={:>7.2f}, p={:.2f}, b={:^5.2f}, ta={:.2f}, ma={:.2f}, c={}, m={}, "
                "dE={:>7.2f}".format(
                    i,
                    global_min,
                    metropolis(dE, b),
                    b,
                    total_acc_prop,
                    mode_acc_prop,
                    count,
                    mode,
                    dE,
                )
            )
        if min_val < global_min:
            global_model = model
            global_min = min_val

    return global_model, min_val
