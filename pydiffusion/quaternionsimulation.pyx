# cython: linetrace=True
# cython: embedsignature=True
# distutils: define_macros=CYTHON_TRACE=1
# -*- Mode: cython; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
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
cimport numpy as np
cimport cython
from scipy._lib._util import check_random_state

from libc.math cimport sqrt

ctypedef np.double_t DTYPE_t
DTYPE = np.double


##################################################
# Quaternion implementation with python wrappers #
##################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_t c_norm(DTYPE_t[::1] q):
    cdef DTYPE_t s = 0
    cdef int i
    for i in range(4):
        s += q[i] * q[i]
    return sqrt(s)


def norm(q):
    """norm of a quaternion (s, v1, v2, v3)

    Parameters
    ----------
    q : array-like
        4 element array, initial quaternion

    Returns
    -------
    norm : float
    """
    return c_norm(np.asarray(q, dtype=DTYPE, order='C'))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_conjugate(DTYPE_t[::1] q, DTYPE_t[::1] res):
    res[0] = q[0]
    res[1] = -q[1]
    res[2] = -q[2]
    res[3] = -q[3]


def conjugate(q):
    """conjugate of a quaternion (s, v1, v2, v3)

    conjugate(q) = (s, -v1, -v2, -v3)

    Parameters
    ----------
    q : array-like
        4 element array, initial quaternion

    Returns
    -------
    np.ndarray (4)
        conjugate quaternion
    """
    res = np.empty(4, dtype=DTYPE)
    c_conjugate(np.asarray(q, dtype=DTYPE, order='C'), res)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_inv(DTYPE_t[::1] q, DTYPE_t[::1] res):
    cdef DTYPE_t norm2 = 0
    cdef int i
    for i in range(4):
        norm2 += q[i] * q[i]
    c_conjugate(q, res)
    for i in range(4):
        res[i] /= norm2


def inv(q):
    """ inverse quaternions

    Parameters
    ----------
    q : array-like
        4 element array, initial quaternion

    Returns
    -------
    np.ndarray (4)
        inverse quaternion

    """
    res = np.empty(4, dtype=DTYPE)
    q = np.asarray(q, dtype=DTYPE)
    c_inv(q, res)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_mul(DTYPE_t[::1] q0, DTYPE_t[::1] q1, DTYPE_t[::1] qq):
    qq[0] = q1[0]*q0[0] - q1[1]*q0[1] - q1[2]*q0[2] - q1[3]*q0[3]
    qq[1] = q1[0]*q0[1] + q1[1]*q0[0] + q1[2]*q0[3] - q1[3]*q0[2]
    qq[2] = q1[0]*q0[2] - q1[1]*q0[3] + q1[2]*q0[0] + q1[3]*q0[1]
    qq[3] = q1[0]*q0[3] + q1[1]*q0[2] - q1[2]*q0[1] + q1[3]*q0[0]


def mul(q1, q2):
    """ multiply two quaternions

    Parameters
    ----------
    q1 : array-like
        4 element array, quaternion
    q2 : array-like
        4 element array, quaternion

    Returns
    -------
    q : np.ndarray (4)
        result of multiplication
    """
    res = np.empty(4, dtype=DTYPE)
    c_mul(np.asarray(q1, dtype=DTYPE, order='C'),
          np.asarray(q2, dtype=DTYPE, order='C'),
          res)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void c_rotate_by(DTYPE_t[::1] q, DTYPE_t[::1] vq, DTYPE_t[::1] res,
                      DTYPE_t[::1] tmp0, DTYPE_t[::1] tmp1):
    """
    vq = vector transformed to quaternion

    allocate temporaties outside of this function
    """
    # tmp = q * vq
    c_mul(q, vq, tmp0)
    # tmp1 = conjugate(q)
    c_conjugate(q, tmp1)
    # res = (q * vq) * conjugate(q)
    c_mul(tmp0, tmp1, res)


def rotate_by(q, vec):
    """rotation vector ``vec`` using quaternion ``q``

    Parameters
    ----------
    q : array-like
        quaternion
    vec : array-like
        vector

    Returns
    -------
    vec : array-like
        rotated vector
    """
    q = np.asarray(q, dtype=DTYPE)
    vec = np.asarray(vec, dtype=DTYPE)

    qvec = np.zeros(4, dtype=DTYPE)
    res = np.zeros(4, dtype=DTYPE)
    tmp0 = np.zeros(4, dtype=DTYPE)
    tmp1 = np.zeros(4, dtype=DTYPE)

    qvec[1:] = vec

    c_rotate_by(q, qvec, res, tmp0, tmp1)

    return res[1:]


def quat(axis, angle):
    """ quaternion from axis and angle

    Parameters
    ----------
    axis : array-like
        vector along to rotate
    angle : float
        angle in degree

    Returns
    -------
    q : ndarray (4)
        quaternion to describe rotation
    """
    axis /= np.linalg.norm(axis)
    quat = np.empty(4, dtype=DTYPE)
    angle = np.radians(.5 * angle)
    quat[0] = np.cos(angle)
    quat[1:] = np.sin(angle) * axis
    return quat

def quaternion_matrix(q):
    cdef DTYPE_t[::1] u = np.asarray(q, dtype=DTYPE)
    cdef DTYPE_t[:, ::1] m = np.empty((3, 3))

    m[0, 0] = 1 - 2 * (u[2]*u[2] + u[3]*u[3])
    m[0, 1] = 2 * (u[0]*u[3] + u[1]*u[2])

    return m

###########################
# Rotation diffusion code #
###########################

@cython.boundscheck(False)
@cython.wraparound(False)
def run(D, niter, dt, random_state=None):
    """Brownian dynamics simulation of the rotation of a point on the sphere.
    Results are recorded as rotations relative to any starting point.

    Parameters
    ----------
    D : array-like (3)
        diagonalized elements of diffusion tensor
    niter : int
        number of iterations
    dt : float
        timestep in seconds
    random_state : int (optional)
        seed for rng. If ``None`` a random seed is chosen. Note for
        multiproccesing using explicit seeds is recommended.

    Returns:
    q : ndarray (niter, 4)
        rotation trajectory in quaternions
    """
    D = np.asarray(D, dtype=DTYPE)

    if dt <= 0:
        raise ValueError("dt must be larger then 0")

    if niter <= 0:
        raise ValueError("niter must be larger then 0")

    rng = check_random_state(random_state)
    a = np.sqrt(.5 * D * dt)

    q = np.empty((niter, 4), dtype=DTYPE)
    q[:, 1:] = a * rng.normal(size=(niter, 3))
    q[:, 0] = np.sqrt(1 - .5 * D.sum() * dt)
    q /= np.sqrt((q**2).sum(1))[:, np.newaxis]
    q[0] = np.array([1, 0, 0, 0], dtype=DTYPE)

    cdef DTYPE_t[:, ::1] view_q = q
    cdef DTYPE_t[::1] q_new = np.empty(4, dtype=DTYPE)

    cdef int i
    for i in range(1, niter):
        c_mul(view_q[i], view_q[i-1], q_new)
        view_q[i] = q_new

    return q
