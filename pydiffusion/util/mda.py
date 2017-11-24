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
import MDAnalysis as mda
import numpy as np


def select_atoms_wrapper(ag, sel):
    """wrapper for `select_atoms` that expands sel

    Parameters
    ----------
    ag : mda.AtomGroup, mda.Universe
        AtomGroup from which new atoms are selected
    sel : str, list, tuple, mda.AtomGroup
        a selection.

    Returns
    -------
    sel : mda.AtomGroup
        Atomgroup defined by selection
    """
    if isinstance(sel, str):
        return ag.select_atoms(sel)
    elif isinstance(sel, (list, tuple)):
        return ag.select_atoms(*sel)
    else:
        return sel


def parse_common_selection(universe, mobile, ref=None, strict=False):
    """Helper function to unpack mobile and ref selection used in common
    analysis algorithms of MDAnalysis. This also ensures that mobile and
    ref use different universe to ensure that iterating over the universe
    doesn't change the reference.

    Parameters
    ----------
    universe : mda.Universe
        universe from which to select the mobile atomgroup
    mobile : str, list, tuple, mda.AtomGroup
        selection for mobile atomgroup
    ref : str, list, tuple, mda.AtomGroup (optional)
        selection for reference atomgroup. If `None` ref will use the same
        selection as `mobile` from a new universe.
    strict : bool (optional)
        check if `mobile` and `ref` are refering to exactly the same atoms

    Returns
    -------
    mobile : mda.AtomGroup
        Atomgroup that refers to the given universe
    ref : mda.AtomGroup
        reference atomgroup that does not refer to universe

    Raises
    ------
    RuntimeError

    """
    mobile = select_atoms_wrapper(universe, mobile)

    if mobile.universe != universe:
        raise RuntimeError("mobile selection doesn't refer to given universe")

    if ref is None:
        u2 = mda.Universe(universe.filename, universe.trajectory.filename)
        ref = u2.select_atoms(*('bynum {}'.format(i + 1)
                                for i in mobile.indices))
    else:
        ref = select_atoms_wrapper(universe, ref)

    if mobile.n_atoms != ref.n_atoms:
        raise RuntimeError("mobile and ref have different number of atoms")

    if strict:
        resids_ok = np.all(mobile.resids == ref.resids)
        segids_ok = np.all(mobile.segids == ref.segids)
        ids_ok = np.all(mobile.ids == ref.ids)
        names_ok = np.all(mobile.names == ref.names)
        resnames_ok = np.all(mobile.resnames == ref.resnames)
        if not (resids_ok and segids_ok and ids_ok and names_ok and
                resnames_ok):
            raise RuntimeError("mobile and ref aren't strictly the same")

    if universe == ref.universe:
        u2 = mda.Universe(universe.filename, universe.trajectory.filename)
        ref = u2.select_atoms(*('bynum {}'.format(i + 1) for i in ref.indices))

    return mobile, ref
