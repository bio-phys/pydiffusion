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
import pandas as pd
from scipy import fftpack


def _acf(x, nlags, demean=True, norm=True, for_pandas=False):
    """Calculate the unbiased autocorrelation of `x` using the FFT"""
    x = np.squeeze(np.asarray(x))
    if demean:
        x = x - x.mean()

    nobs = len(x)
    # We are using the unbiased estimator only!
    d = nobs - np.arange(nobs)
    # ensure that we always use a power of 2 or 3 for zero-padding,
    # this way we'll ensure O(n log n) runtime of the fft.
    n = fftpack.helper.next_fast_len(2 * nobs + 1)

    Frf = fftpack.fft(x, n=n)
    ac = fftpack.ifft(Frf * np.conjugate(Frf))[:nobs] / d
    ac = np.real(ac[:nlags])

    if norm:
        ac /= ac[0]

    # fill with zeros to allow using this on pandas df with `.apply`
    if for_pandas:
        _len = len(x)
        ac = np.hstack((ac, np.zeros(_len - nlags)))

    return ac


def acf(x, nlags=None, row_var=False, demean=True, norm=True):
    """Wrapper for statsmodels acf function for different data types. Currently
    pandas dataframes and numpy arrays in 1d and 2d are supported.

    Parameters
    ----------
    x : array-like / pd.DataFrame
        1-D or 2-D array containing multiple variables and observations.
        Alternatively pandas DataFrames are also accepted.
    nlags : int optional
        Number of lags to return autocorrelation for. Default is to chose 10%
        all possible values ``0.1 * ((len(x) / 2) - 1)``. The default is a good
        rule of thumb more value might be unreliable
    row_var : boolean optional
        If row_var is True, then each row represents a variable, with
        observations in the columns. Otherwise the relationship is transposed:
        each column represents a variable, while rows contain observations. For
        pandas DataFrames row_var is always exptected to be false.
    demean : bool optional
        remove the mean of data before calculating acf
    norm : bool optional
        normalize acf(0) to 1

    Returns
    -------
    acf : array-like
        Depending on the input either a 1-D/2-D numpy array or pd.DataFrame with
        the autocorrelation values.

    """
    if nlags is None:
        nlags = int(0.1 * (len(x) / 2 - 1))
        if row_var:
            nlags = int(0.1 * (x.shape[1] / 2 - 1))
    if isinstance(x, pd.DataFrame):
        return x.apply(
            _acf, raw=True, nlags=nlags, for_pandas=True, demean=demean, norm=norm
        )[:nlags]
    elif x.ndim == 2:
        if not row_var:
            x = x.T
        ac = np.array([_acf(var, nlags, demean=demean, norm=norm) for var in x])
        return ac if row_var else ac.T
    else:
        return _acf(x, nlags, demean=demean, norm=norm)
