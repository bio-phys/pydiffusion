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
from __future__ import absolute_import, division
from six.moves import range, zip

import numpy as np
from os.path import join as pjoin, dirname
import MDAnalysis as mda
from MDAnalysis.lib import transformations
from collections import namedtuple
from itertools import combinations_with_replacement

import pytest
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_almost_equal)

from scipy.special import legendre

import pydiffusion.rotation as rot

DATA_DIR = pjoin(dirname(__file__), 'data')


def test__D():
    D = np.arange(1, 4)
    assert_equal(2., rot._D(D))


def test_delta():
    D = np.zeros(3)
    delta = rot.delta(D)
    assert_equal(delta, 0)

    D = np.arange(1, 4)
    delta = rot.delta(D)
    assert_equal(delta, np.sqrt(3))

    with pytest.raises(ValueError):
        rot.delta([0, 1])


def test__F():
    v = np.eye(3)[0]
    assert_almost_equal(2. / 3., rot._F(v))


def test__G():
    v = np.eye(3)[0]
    D = np.arange(1, 4)
    assert_almost_equal(-1. / np.sqrt(3), rot._G(v, D))


def test__aa():
    v = np.eye(3)[0]
    D = np.arange(1, 4)
    aa = rot._aa(v, D)
    assert_almost_equal(aa, [
        3 / 4 * (2 / 3 + (-1 / np.sqrt(3))), 0, 0, 0,
        3 / 4 * (2 / 3 - (-1 / np.sqrt(3)))
    ])


def test_p_l():
    x = np.linspace(0, 10)
    assert_array_equal(rot.p_l(x, l=2), legendre(2)(x))
    assert_array_equal(rot.p_l(x, l=1), legendre(1)(x))


def test_correlation_time():
    D = 1, 0, 1
    v = 1, 0, 0
    assert_almost_equal(rot.correlation_time(D, v), 0.416666666666)

    D = 1, 1, 1
    assert_almost_equal(rot.correlation_time(D, v), 0.166666666666666)


def test_rcf_rank_1():
    D = np.arange(1, 4)
    axes = np.eye(3)
    t = np.linspace(0, 1e-3, 1000)

    # test along axis in the particle coordinate system of rotation
    for i, ax in enumerate(axes):
        rcf = rot.rcf(t, D, ax, rank=1)
        assert_equal(len(rcf), 1000)
        assert_array_almost_equal(rcf, np.exp(-(D.sum() - D[i]) * t))


def test_rcf_rank_2():
    D = np.arange(1, 4)
    Dr = rot._D(D)
    delta = rot.delta(D)
    axes = np.eye(3)
    t = np.linspace(0, 1e-3, 1000)

    for i, ax in enumerate(axes):
        rcf = rot.rcf(t, D, ax, rank=2)
        theo_rcf = (3. / 2. * np.exp(-6 * Dr * t) *
                    (2. / 3. * np.cosh(2 * delta * t) +
                     (D[i] - Dr) / delta * np.sinh(2 * delta * t)))
        assert_equal(len(rcf), 1000)
        assert_array_almost_equal(rcf, theo_rcf)


def test_rcf_rank_bigger_2():
    D = np.arange(1, 4)
    axes = np.eye(3)
    t = np.linspace(0, 1e-3, 1000)
    with pytest.raises(NotImplementedError):
        rot.rcf(t, D, axes[0], rank=3)


@pytest.fixture
def curve():
    # Code to generate trajectory from any atomgroup `s`
    # def rotate(ag, R):
    #     """This is the default rotation in MDAnalysis >0.16.0"""
    #     cg = ag.center_of_geometry()
    #     ag.translate(-cg)
    #     ag.rotate(R)
    #     ag.translate(cg)
    #     return ag
    #
    # s.translate(-s.center_of_geometry())
    # pa = s.principal_axes()
    # s.rotate(pa)
    # s.write('curve.pdb')
    # n_frames = 27
    # r = 20
    # trans = np.array([2 * np.pi * r / n_frames, 0, 10])
    # rot=mda.lib.transformations.rotation_matrix(
    #     2 * np.pi / n_frames,(0, 0, 1))[:3, :3]
    # rot[np.abs(rot) < 1e-12] = 0
    # with mda.Writer('curve.xtc', s.n_atoms) as w:
    #     s.translate([r, 0, 0])
    #     w.write(s)
    #     for i in range(n_frames):
    #         s = rotate(s, rot.T)
    #         s.translate(trans)
    #         trans = np.dot(trans, rot.T)
    #         w.write(s)
    u = mda.Universe(
        pjoin(DATA_DIR, 'curve.pdb'), pjoin(DATA_DIR, 'curve.xtc'))
    n_frames = u.trajectory.n_frames - 1
    return namedtuple('TT', 'atoms, rot')(
        u.atoms, mda.lib.transformations.rotation_matrix(
            2 * np.pi / n_frames, (0, 0, 1))[:3, :3])


@pytest.fixture
def curve_rotated():
    # added random rotation to a frame in curve
    return mda.Universe(pjoin(DATA_DIR, 'curve-rotated.pdb'))


def test_RotationMatrix(curve):
    rm = rot.RotationMatrix(curve.atoms, align_first_to_ref=False).run()
    for i, R in enumerate(rm.R):
        refR = np.linalg.matrix_power(curve.rot.T, i)
        assert_array_almost_equal(R, refR.T, decimal=3)

    u2 = mda.Universe(curve.atoms.universe.filename,
                      curve.atoms.universe.trajectory.filename)

    for R, ts in zip(rm.R, curve.atoms.universe.trajectory):
        curve.atoms.translate(-curve.atoms.center_of_geometry())
        curve.atoms.rotate(R)
        est_R = rot.rotation_matrix(curve.atoms, u2.atoms)
        assert_array_almost_equal(est_R, np.eye(3))


def test_RotationMatrix_with_weights(curve):
    n_atoms = curve.atoms.n_atoms
    weights = np.zeros(n_atoms)
    weights[:10] = 1 / 10.0
    rm = rot.RotationMatrix(
        curve.atoms, weights=weights, align_first_to_ref=False).run()
    for i, R in enumerate(rm.R):
        refR = np.linalg.matrix_power(curve.rot, i)
        assert_array_almost_equal(R, refR, decimal=3)


def test_RotationMatrix_with_alignment(curve, curve_rotated):
    # test more alignments
    rm_ref = rot.RotationMatrix(curve.atoms, align_first_to_ref=False).run()
    rm = rot.RotationMatrix(
        curve.atoms, start=2, align_first_to_ref=True).run()
    # this checks for sure if my rotation to first is correct
    assert_array_almost_equal(rm._first_rot.T, rm_ref.R[2].T, decimal=4)

    # check if we can reconstruct the rotations from a different starting point
    for i, (R, R_ref) in enumerate(zip(rm.R, rm_ref.R[rm.start:])):
        assert_array_almost_equal(np.dot(rm._first_rot, R), R_ref, decimal=4)

    # test more specific
    rm = rot.RotationMatrix(
        curve.atoms, ref=curve_rotated.atoms, align_first_to_ref=True).run()
    assert_array_almost_equal(rm.R[0], np.eye(3))
    R_first = rot.rotation_matrix(curve.atoms, curve_rotated.atoms)
    assert_array_almost_equal(R_first.T, rm._first_rot)

    for R, ts in zip(rm.R, curve.atoms.universe.trajectory):
        curve.atoms.translate(-curve.atoms.center_of_geometry())
        curve.atoms.rotate(R_first.T)
        curve.atoms.rotate(R)
        est_R = rot.rotation_matrix(curve.atoms, curve_rotated.atoms)
        assert_array_almost_equal(est_R, np.eye(3))


def test_RotationMatrix_with_reference(curve, curve_rotated):
    rm = rot.RotationMatrix(
        curve.atoms, ref=curve_rotated.atoms, align_first_to_ref=False).run()
    curve.atoms.universe.trajectory[0]
    r = rot.rotation_matrix(curve_rotated.atoms, curve.atoms)
    assert_array_almost_equal(rm.R[0], r)


def test_RotationMatrix_iterations(curve):
    # This is to test against MDAnalysis bug #1031
    rm = rot.RotationMatrix(
        curve.atoms, stop=5, align_first_to_ref=False).run()
    rm = rot.RotationMatrix(curve.atoms, align_first_to_ref=True).run()
    assert_array_almost_equal(rm._first_rot, np.eye(3))


def test_cos_t():
    axis = [0, 0, 1]
    R = np.array([
        mda.lib.transformations.rotation_matrix(i * 5, axis)[:3, :3]
        for i in range(10)
    ])
    cost = rot.cos_t(R, [1, 0, 0])
    assert_equal(cost.shape, (10, ))
    assert_almost_equal(cost, np.cos(np.arange(10) * 5))


@pytest.mark.filterwarnings('ignore:invalid value')
def test_make_right_handed():
    M = np.eye(3)
    M[:, 0] *= -1
    M = rot._make_right_handed(M)
    assert_array_equal(np.eye(3), M)

    with pytest.raises(RuntimeError):
        M = np.eye(3)
        M[:, 0] = M[:, 2]
        rot._make_right_handed(M)


def test_pcs():
    D = [[24660000., -1917000., 4468000.], [-1917000., 21630000., -1584000.],
         [4468000., -1584000., 25030000.]]
    Dpcs, toPCS, toLab = rot.pcs(D)
    # more checks don't make sense because I would only check that np.linalg.eig
    # works correct, that is something I want to assume already
    assert len(Dpcs) == 3
    assert_array_almost_equal(np.linalg.inv(toPCS), toLab)


@pytest.fixture
def rotations(curve):
    return rot.RotationMatrix(curve.atoms, align_first_to_ref=False).run().R


@pytest.mark.parametrize('t', [-1, 1.4, 1001])
def test_rotations_at_t_exceptions_t(t):
    R = transformations.random_rotation_matrix()[:3, :3][np.newaxis]
    with pytest.raises(ValueError):
        rot.rotations_at_t(R, t)


@pytest.mark.parametrize('shape', [(3, 3), (2, 3, 4), (2, 4, 3), (1, 3, 3)])
def test_rotations_at_t_exceptions_R_shape(shape):
    R = np.arange(np.prod(shape)).reshape(shape)
    with pytest.raises(ValueError):
        rot.rotations_at_t(R, 0)


def test_rotations_at_t_0(rotations):
    R = rot.rotations_at_t(rotations, 0)
    for r in R:
        assert_array_almost_equal(r, np.eye(3))


@pytest.mark.parametrize('t', range(5))
def test_rotations_at_t(rotations, t):
    R = rot.rotations_at_t(rotations, t)
    assert R.shape == (len(rotations) - t, 3, 3)
    for i in range(len(rotations) - t):
        assert_array_almost_equal(R[i],
                                  np.dot(rotations[i], rotations[i + t].T))


def test_quaternion_covariance_exceptions():
    shape = (1, 3, 3)
    R = np.arange(np.prod(shape)).reshape(shape)
    with pytest.raises(ValueError):
        rot.quaternion_covariance(R, 1)


def test_rotations_covariance(rotations):
    t = 1
    R = rot.rotations_at_t(rotations, t).mean(0)

    ######################
    # Reference implementation, so this test only serves as a regression test
    ######################
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
    u += np.triu(u, 1).T
    u /= 4

    cov = rot.quaternion_covariance(rotations, t)
    assert_array_almost_equal(u, cov)


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_rotation_correlations(rotations, n_jobs):
    u = rot.rotation_correlations(rotations, 20, n_jobs=n_jobs)
    assert u.shape == (20, 4, 4)


@pytest.fixture
def params():
    dt = 10e-9  # 1 nano second
    msdq = np.array([[
        0., 0.11424535, 0.17804407, 0.21315337, 0.23210453, 0.24206588,
        0.24710393, 0.24950183, 0.25052477, 0.25086202
    ], [
        0., 0.08339403, 0.13911204, 0.17627704, 0.20103169, 0.21750017,
        0.2284448, 0.23571197, 0.2405337, 0.24373084
    ], [
        0., 0.08569921, 0.1418894, 0.17878675, 0.20304756, 0.21901818,
        0.22954219, 0.23648325, 0.24106472, 0.24409074
    ]])

    Params = namedtuple("Params", "model, dt, t, msdq")
    return Params(
        model=rot.RotationTensor(
            np.asarray([30042854.0, 20255914.0, 21021232.0]), np.eye(3)),
        dt=dt,
        t=np.arange(10) * dt,
        msdq=msdq)


def test_quaternion_correlations(params):
    cor = rot.moment_2(params.t, params.model)
    for i, j in combinations_with_replacement(range(3), 2):
        if i == j:
            assert_array_almost_equal(cor[i, i], params.msdq[i])
        else:
            assert_array_almost_equal(cor[i, j], np.zeros(params.t.size))


def test_metropolis():
    assert_equal(rot.metropolis(-1, 1), 1)
    assert_equal(rot.metropolis(1, 1), np.exp(-1))


def test_RotationTensor_wrong_type():
    with pytest.raises(ValueError):
        rot.RotationTensor([1, 2], np.eye(3))
    with pytest.raises(ValueError):
        rot.RotationTensor(np.ones(3), np.eye(4))


def test_RotationTensor_tensor():
    D = np.arange(3)
    R = np.eye(3)
    T = rot.RotationTensor(D, R)
    # TODO use more complicated R
    assert_almost_equal(R * D, T.tensor)


@pytest.mark.parametrize('start, stop, step', ((1, 10, 2), (10, 1, -2)))
def test_inf_generator(start, stop, step):
    gen = rot.inf_generator(start, stop, step)
    assert next(gen) == start
    assert next(gen) == start + step
    for i in range(10):
        next(gen)
    assert next(gen) == stop


@pytest.mark.parametrize('D', [np.arange(1, 4), np.ones(3)])
def test_chi2(D):
    D = D * 1e-3
    dt = 1
    t = np.arange(100) * dt
    tensor = rot.RotationTensor(D, np.eye(3))
    moment2 = rot.moment_2(t, tensor)
    assert_almost_equal(0, rot.chi2(moment2, tensor, t))
    assert_almost_equal(0, rot.chi2(moment2, tensor))


@pytest.mark.parametrize('D', [np.arange(3, 0, -1), np.ones(3)])
def test_laplace_fit(D):
    D = D * 1e-3
    dt = 1
    time = np.arange(100) * dt
    tensor = rot.RotationTensor(D, np.eye(3))
    moment2 = rot.moment_2(time, tensor)
    result = rot.fit_laplace(moment2, time, s=1 / 10)
    assert_almost_equal(result.D, D, decimal=4)


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('D', [np.arange(3, 0, -1), np.ones(3)])
def test_anneal(D):
    D = D * 1e-3
    dt = 1
    t = np.arange(100) * dt
    tensor = rot.RotationTensor(D, np.eye(3))
    moment2 = rot.moment_2(t, tensor)
    result = rot.anneal(moment2, t, D=D * 2, eps=0.01)[0]
    assert_almost_equal(result.D, D, decimal=4)
