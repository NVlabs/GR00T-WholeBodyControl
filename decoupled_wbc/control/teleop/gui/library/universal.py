#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehumancommunity.org/

**Github Code Home Page:**    https://github.com/makehumancommunity/

**Authors:**           Glynn Clements

**Copyright(c):**      MakeHuman Team 2001-2020

**Licensing:**         AGPL3

    This file is part of MakeHuman Community (www.makehumancommunity.org).

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

TODO
"""

import json
import os
from typing import Any

from library.getpath import getPath

_PRE_STARTUP_KEYS = ["useHDPI", "noShaders", "noSampleBuffers"]


class Globals(object):
    def __init__(self):
        self.app: Any = None
        self.args = {}
        self.world = []
        self.cameras = []
        self.canvas = None
        self.windowHeight = 600
        self.windowWidth = 800
        self.clearColor = (0.0, 0.0, 0.0, 0.0)
        self.preStartupSettings = dict()
        self._preStartupConfigScan()

    def _preStartupConfigScan(self):
        """Run a very primitive scan in order to pick up settings which has
        to be known before we launch the QtApplication object."""
        iniPath = getPath("settings.ini")
        data = None
        if os.path.exists(iniPath):
            with open(iniPath, encoding="utf-8") as f:
                data = json.load(f)
        if data is None:
            for key in _PRE_STARTUP_KEYS:
                self.preStartupSettings[key] = None
        else:
            for key in _PRE_STARTUP_KEYS:
                if key in data:
                    self.preStartupSettings[key] = data[key]
                    # Would be nice to log this, but log has not been initialized yet
                    print("PRE STARTUP SETTING: " + key + " = " + str(data[key]))
                else:
                    self.preStartupSettings[key] = None


G = Globals()
