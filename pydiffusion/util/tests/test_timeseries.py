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
import pandas as pd
from collections import namedtuple

import pytest
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_almost_equal)

from hummer.util import timeseries


@pytest.fixture
def trigonometric():
    Trig = namedtuple('Trig', 't, sin, cos')
    t = np.linspace(0, 10 * np.pi, 10000)
    sin = np.sin(t)
    cos = np.cos(t)
    return Trig(t, sin, cos)


def test_norm(trigonometric):
    acf = timeseries.acf(trigonometric.sin, 10, norm=True)
    assert acf[0] == 1.0

    acf = timeseries.acf(trigonometric.sin, 10, norm=False)
    assert_almost_equal(acf[0], 0.49995)


def test_demean(trigonometric):
    acf = timeseries.acf(trigonometric.sin + 10, 10, demean=False, norm=False)
    acf2 = timeseries.acf(trigonometric.sin, 10, norm=False)
    assert_array_almost_equal(acf - 10**2, acf2)


def test_for_pandas(trigonometric):
    acf = timeseries._acf(trigonometric.sin, 10, for_pandas=True)
    acf_normal = timeseries._acf(trigonometric.sin, 10)
    assert len(acf) == len(trigonometric.sin)
    assert_array_equal(acf[10:], np.zeros(len(trigonometric.sin) - 10))
    assert_array_equal(acf[:10], acf_normal)


def test_nlags(trigonometric):
    acf = timeseries.acf(trigonometric.sin, 10)
    assert len(acf) == 10


@pytest.mark.parametrize('nlags', [100, None])
def test_acf_1d(trigonometric, nlags):
    _acf = timeseries.acf(trigonometric.sin, nlags=nlags)
    # calculate default nlags
    if nlags is None:
        nlags = int(0.1 * (len(trigonometric.t) / 2 - 1))

    assert len(_acf) == nlags
    assert_almost_equal(_acf[0], 1)
    assert_array_almost_equal(trigonometric.cos[:nlags], _acf, decimal=1)


@pytest.mark.parametrize('nlags', [100, None])
def test_acf_2d(trigonometric, nlags):
    sin = np.vstack((trigonometric.sin for _ in range(5)))
    _acf = timeseries.acf(sin, row_var=True, nlags=nlags)

    # calculate default nlags
    if nlags is None:
        nlags = int(0.1 * (len(trigonometric.t) / 2 - 1))

    cos = np.vstack((trigonometric.cos[:nlags] for _ in range(5)))

    assert_array_equal(_acf.shape, (5, nlags))
    assert_array_almost_equal(_acf[:, 0], np.ones(5))
    assert_array_almost_equal(cos, _acf, decimal=1)


def test_acf_col_var(trigonometric):
    nlags = 100
    sin = np.vstack((trigonometric.sin for _ in range(5)))
    _acf = timeseries.acf(sin.T, row_var=False, nlags=nlags)

    cos = np.vstack((trigonometric.cos[:nlags] for _ in range(5)))

    assert_array_equal(_acf.shape, (nlags, 5))
    assert_array_almost_equal(_acf[0, :], np.ones(5))
    assert_array_almost_equal(cos, _acf.T, decimal=1)


@pytest.mark.parametrize('nlags', [100, None])
def test_acf_DataFrame(trigonometric, nlags):
    sin = pd.DataFrame(np.vstack((trigonometric.sin for _ in range(5))).T)
    _acf = timeseries.acf(sin, nlags=nlags)

    # calculate default nlags
    if nlags is None:
        nlags = int(0.1 * (len(trigonometric.t) / 2 - 1))

    cos = np.vstack((trigonometric.cos[:nlags] for _ in range(5))).T

    assert_array_equal(_acf.shape, (nlags, 5))
    assert_array_almost_equal(_acf.iloc[0], np.ones(5))
    assert_array_almost_equal(cos, _acf.values, decimal=1)
