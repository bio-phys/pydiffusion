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
# along with pydiffusion. If not, see <http://www.gnu.org/licenses/>.
from __future__ import division, absolute_import

import itertools
import matplotlib.pyplot as plt
import numpy as np


class CovFig(object):
    """Have a figure like object that makes plotting of a quaternion covariance
    easier. The axes already get appropriate titles.

    """
    def __init__(self, figsize=None):
        if figsize is None:
            figsize = [10.8, 7.2]

        self.fig, self.axes = plt.subplots(2, 3, figsize=figsize)

        arow = self.axes[0]
        for i, ax in enumerate(arow):
            ax.set_title('{}-{}'.format(i + 1, i + 1))
        arow = self.axes[1]
        for ax, (i, j) in zip(arow, itertools.combinations(range(3), 2)):
            ax.set_title('{}-{}'.format(i + 1, j + 1))

    def set_all_axes(self, **kwargs):
        """call `set` with same arguments on all axes"""
        for ax in self.axes.ravel():
            ax.set(**kwargs)

    def tight_layout(self):
        self.fig.tight_layout()


def plot_covariance(covar, time=None, covfig=None, show_limit=False, **kwargs):
    """ given a covariance plot it in a Covariance Figure
    """
    if covfig is None:
        covfig = CovFig()

    if time is None:
        time = np.arange(covar.shape[-1])

    axes = covfig.axes

    arow = axes[0]
    for i, ax in enumerate(arow):
        ax.plot(time, covar[i, i], **kwargs)
        if show_limit:
            ax.axhline(0.25, linestyle='--', color='#808080')
    arow = axes[1]
    for ax, (i, j) in zip(arow, itertools.combinations(range(3), 2)):
        ax.plot(time, covar[i, j], **kwargs)

    return covfig
