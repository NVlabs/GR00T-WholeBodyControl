#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehumancommunity.org/

**Code Home Page:**    https://github.com/makehumancommunity/makehuman

**Authors:**           Aranuvir

**Copyright(c):**      MakeHuman Team 2001-2020

**Licensing:**         AGPL3

    This file is part of MakeHuman (www.makehumancommunity.org).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.


Abstract
--------
This module provides the "well known" user directories from a Linux Desktop. Common
directories are 'DESKTOP', 'DOCUMENTS', 'DOWNLOAD', 'MUSIC', 'PICTURES', '
PUBLICSHARE', 'TEMPLATES' and 'VIDEOS'. The predefined directories and their
paths are stored in the dictionary XDG_PATHS, which will be None on platforms other
than Linux."""


import os
import sys

isLinux = sys.platform.startswith("linux")

if isLinux:

    # Default path to xdg configuration file
    CONFIG_PATH = os.path.expanduser("~/.config/user-dirs.dirs")

    def get_userdirs(path=CONFIG_PATH):
        """This function parses a configuration file to retrieve user directories and their configured paths.
        Environment variables are replaced by their values"""

        dirs = dict()

        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    if line and line.startswith("XDG_"):
                        line = line.strip()
                        key, value = line.split("=")
                        key = key.split("_")[1]
                        value = os.path.expandvars(value.strip('"'))
                        if os.path.isdir(value):
                            dirs[key] = value

        return dirs

else:

    def get_userdirs(path=""):
        """Dummy function for use on other Platform than Linux"""
        return {}


XDG_PATHS = get_userdirs()

if XDG_PATHS:
    XDG_DIRS = sorted(XDG_PATHS.keys())
else:
    XDG_DIRS = []


# Main Function #############################################################
if __name__ == "__main__":

    if isLinux and XDG_DIRS:

        print("Found XDG_DIRS:", ", ".join(XDG_DIRS), "\n")

        max_len = max([len(d) for d in XDG_DIRS])
        f_str = "{0:" + str(max_len) + "s}  --->   {1}"

        for d in XDG_DIRS:
            print(f_str.format(d, XDG_PATHS.get(d)))

    else:

        print("This module should be used on Linux platforms only!")
