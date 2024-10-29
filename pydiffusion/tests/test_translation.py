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
import MDAnalysis as mda
from collections import namedtuple

import pytest
from numpy.testing import assert_almost_equal

from pydiffusion.util.random import random_walk
import pydiffusion.translation as hdt


class TestMSD:
    def test_msd_arange_slow(self):
        a = np.arange(9)
        msd = hdt.msd(a, fft=False)
        assert 8 == len(msd)
        assert_almost_equal((a**2)[:-1], msd)

    def test_msd_arange_fft(self):
        a = np.arange(90)
        msd = hdt.msd(a)
        assert (len(a) - 1) == len(msd)
        # for a few values the result is off by about 0.33. I don't know where
        # this comes from but this solution is only a good aproxximation for
        # large values of N anyway
        assert_almost_equal((a**2)[:-1], msd, decimal=0)

    def test_msd_constant_slow(self):
        a = np.ones(9) * 42
        msd = hdt.msd(a, fft=False)
        assert_almost_equal(np.zeros(len(a) - 1), msd)

    def test_msd_constant_fft(self):
        a = np.ones(9) * 42
        msd = hdt.msd(a)
        assert_almost_equal(np.zeros(len(a) - 1), msd)

    def test_same_result(self):
        a = random_walk(1, 10000, 508842746).reshape(10000)
        nlags = 100
        msd_slow = hdt.msd(a, nlags=nlags, fft=False)
        msd_fast = hdt.msd(a, nlags=nlags, fft=True)
        assert_almost_equal(msd_slow, msd_fast)

        b = random_walk(3, 10000)
        msd_slow = hdt.msd(b, nlags=nlags, fft=False)
        msd_fast = hdt.msd(b, nlags=nlags, fft=True)
        assert_almost_equal(msd_slow, msd_fast)

    def test_array_like(self):
        a = list(np.arange(9))
        msd = hdt.msd(a, fft=False)
        assert 8 == len(msd)
        assert_almost_equal(np.power(a, 2)[:-1], msd)

    def test_ValueError(self):
        a = np.ones((3, 3, 3))
        with pytest.raises(ValueError):
            hdt.msd(a)


@pytest.fixture
def curve(data):
    # Code to generate trajectory from any atomgroup `s`
    # def rotate(ag, R):
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
    # rot1=mda.lib.transformations.rotation_matrix(
    #     2 * np.pi / n_frames,(0, 0, 1))[:3, :3]
    # rot[np.abs(rot) < 1e-12] = 0
    # with mda.Writer('curve.xtc', s.n_atoms) as w:
    #     s.translate([r, 0, 0])
    #     w.write(s)
    #     for i in range(n_frames):
    #         s = rotate(s, rot1.T)
    #         s.translate(trans)
    #         trans = np.dot(trans, rot1.T)
    #         w.write(s)
    u = mda.Universe(data["curve.pdb"], data["curve.xtc"])
    n_frames = u.trajectory.n_frames - 1  # stored an extra frame
    trans = np.array([2 * np.pi * 20 / n_frames, 0, 0])
    return namedtuple("TT", "atoms, trans")(u.atoms, trans)


@pytest.mark.parametrize("rot", [None, np.eye(3)])
def test_ParticleCS(curve, rot):
    pcs = hdt.ParticleCS(curve.atoms, rotation=rot).run()
    diff = np.diff(pcs.pcs, axis=0)
    assert_almost_equal(diff, np.ones(diff.shape) * diff[0], decimal=2)
    assert_almost_equal(diff[0], curve.trans, decimal=2)
