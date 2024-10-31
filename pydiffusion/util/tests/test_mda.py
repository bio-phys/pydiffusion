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
import MDAnalysis as mda

from MDAnalysisTests.datafiles import PSF, DCD
from numpy.testing import assert_equal
import pytest

from pydiffusion import util


@pytest.fixture(scope="module")
def ag():
    return mda.Universe(PSF, DCD).atoms


@pytest.fixture(scope="module")
def u():
    return mda.Universe(PSF, DCD)


class TestSelectAtomsWrapper:
    @pytest.mark.parametrize(
        "sel", ["name CA", ("bynum 1", "bynum 2"), ["bynum 1", "bynum 2"]]
    )
    def test_non_atomgroup_input(self, ag, sel):
        util.mda.select_atoms_wrapper(ag, sel)

    def test_atomgroup(self, ag, u):
        sel = util.mda.select_atoms_wrapper(ag, ag)
        assert sel == ag
        # don't touch selection if select_atoms can't be called!
        sel = util.mda.select_atoms_wrapper(ag, u.atoms)
        assert_equal(sel.indices, ag.indices)


class TestParseCommonSelection:
    @pytest.mark.parametrize(
        "mobile, ref",
        [
            ("name CA", None),
            (("bynum 1", "bynum 2"), None),
            (["bynum 1", "bynum 2"], None),
            ("name CA", "name CA"),
            ("all", None),
            ("bynum 1-3", "bynum 2-4"),
        ],
    )
    def test_input_combination(self, u, mobile, ref):
        m, r = util.mda.parse_common_selection(u, mobile, ref)
        assert m.universe != r.universe
        assert m.n_atoms == r.n_atoms

    @pytest.mark.parametrize(
        "mobile, ref", [("all", None), ("name CA", None), ("name CA", "name CA")]
    )
    def test_input_combination_strict(self, u, mobile, ref):
        m, r = util.mda.parse_common_selection(u, mobile, ref, strict=True)
        assert m.universe != r.universe
        assert m.n_atoms == r.n_atoms

    def test_with_atomgroups(self, u):
        ca = u.select_atoms("name CA")
        util.mda.parse_common_selection(u, ca)
        util.mda.parse_common_selection(u, ca, ca)
        u2 = mda.Universe(PSF, DCD)
        util.mda.parse_common_selection(u, ca, u2.select_atoms("name CA"))

    def test_exceptions(self, u):
        with pytest.raises(RuntimeError):
            u2 = mda.Universe(PSF, DCD)
            util.mda.parse_common_selection(u, u2.select_atoms("name CA"))

        with pytest.raises(RuntimeError):
            util.mda.parse_common_selection(u, "all", "name CA")

        with pytest.raises(RuntimeError):
            util.mda.parse_common_selection(u, "bynum 1-3", "bynum 2-4", strict=True)
