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
from MDAnalysis.lib import transformations

from numpy.testing import assert_equal, assert_almost_equal
import pytest

from pydiffusion import quaternionsimulation as rotation

##################################
# Test Quaternion implementation #
##################################


@pytest.fixture
def zero_quat():
    return [1, 0, 0, 0]


@pytest.fixture
def right_angle():
    return rotation.quat([1, -1, 1], 90)


@pytest.fixture
def right_angle_mat():
    return transformations.rotation_matrix(np.radians(90), [1, -1, 1])[:3, :3]


@pytest.fixture
def vec():
    return np.ones(3) / np.sqrt(3)


def test_norm(zero_quat):
    assert 1 == rotation.norm(zero_quat)


def test_conjugate():
    q = [2, 1, 1, 1]
    assert_equal([2, -1, -1, -1], rotation.conjugate(q))


@pytest.mark.parametrize('axis, angle', (([1, 1, 1], 90), ([0, 1, 1], 45)))
def test_quat(axis, angle):
    q = rotation.quat(axis, angle)
    axis = np.array(axis)
    assert 1 == rotation.norm(q)
    assert np.cos(np.radians(.5 * angle)) == q[0]
    assert_almost_equal(
        np.sin(np.radians(.5 * angle)) * axis / np.linalg.norm(axis), q[1:])
    assert_almost_equal(
        transformations.quaternion_about_axis(np.radians(angle), axis), q)


def test_mul(zero_quat, right_angle):
    assert_almost_equal(right_angle, rotation.mul(zero_quat, right_angle))
    assert_almost_equal(right_angle, rotation.mul(right_angle, zero_quat))
    assert_almost_equal(
        transformations.quaternion_multiply(right_angle, right_angle),
        rotation.mul(right_angle, right_angle))


def test_rotate_by(right_angle, zero_quat, vec, right_angle_mat):
    assert_almost_equal(vec, rotation.rotate_by(zero_quat, vec))
    assert_almost_equal(
        rotation.rotate_by(right_angle, vec),
        rotation.rotate_by(-right_angle, vec))
    assert_almost_equal(
        np.dot(right_angle_mat, vec), rotation.rotate_by(right_angle, vec))


def test_inverse(right_angle):
    assert_almost_equal([1, 0, 0, 0],
                        rotation.mul(right_angle, rotation.inv(right_angle)))


###################
# Test Simulation #
###################


@pytest.fixture
def D():
    return np.asarray([30042854.0, 20255914.0, 21021232.0])


@pytest.mark.parametrize('dt', (-1, 0))
def test_run_exception(dt):
    with pytest.raises(ValueError):
        rotation.run(D=np.ones(3), niter=10, dt=dt, random_state=42)
    with pytest.raises(ValueError):
        rotation.run(D=np.ones(3), niter=dt, dt=1, random_state=42)


@pytest.mark.parametrize('niter', (42, 1238))
def test_run_return_size(D, niter):
    r1 = rotation.run(D=D, niter=niter, dt=1e-9, random_state=42)
    assert r1.shape == (niter, 4)


@pytest.mark.parametrize('seed', (42, 1238230))
def test_run_reproducibility(D, seed):
    r1 = rotation.run(D=D, niter=10, dt=1e-9, random_state=seed)
    r2 = rotation.run(D=D, niter=10, dt=1e-9, random_state=seed)
    assert_equal(r1, r2)

    r1 = rotation.run(D=D, niter=10, dt=1e-9)
    r2 = rotation.run(D=D, niter=10, dt=1e-9)
    assert not np.allclose(r1, r2)


def test_run(D):
    r1 = rotation.run(D=D, niter=10, dt=1e-9)
    assert_almost_equal(r1[0], [1, 0, 0, 0])
    for r in r1[1:]:
        assert not np.allclose(r1[0], r)
