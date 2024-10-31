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

from os.path import join as pjoin, isfile


class TestDataDir(object):
    """
    Simple class to access a directory with test data
    """

    def __init__(self, folder, data_folder):
        self.folder = pjoin(folder, data_folder)

    def __getitem__(self, file):
        data_filename = pjoin(self.folder, file)
        if isfile(data_filename):
            return data_filename
        else:
            raise RuntimeError(
                "no file '{}' found in folder '{}'".format(file, self.folder)
            )
