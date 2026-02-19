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

import atexit
from cProfile import Profile

_sort = "cumulative"
_profiler = None
_show = None


def run(cmd, globals, locals):
    prof = Profile()
    try:
        prof.runctx(cmd, globals, locals)
    finally:
        show(prof)


def active():
    return _profiler is not None


def start():
    global _profiler
    if active():
        return
    _profiler = Profile()


def stop():
    global _profiler
    if not active():
        return
    show(_profiler)
    _profiler = None


def accum(cmd, globals, locals):
    if not active():
        return
    _profiler.runctx(cmd, globals, locals)


def show(prof):
    try:
        if _show is not None:
            _show(prof)
        else:
            prof.print_stats(_sort)
    except TypeError:
        pass


def set_sort(sort):
    global _sort
    _sort = sort


def set_show(show):
    global _show
    _show = show


atexit.register(stop)
