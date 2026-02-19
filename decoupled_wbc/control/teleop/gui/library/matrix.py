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

Numpy powered matrix transformations.
"""

import math

import numpy as np


def transform(m, v):
    return np.asarray(m * np.asmatrix(v).T)[:, 0]


def transform3(m, v):
    x, y, z = v
    v = np.asmatrix((x, y, z, 1)).T
    v = np.asarray(m * v)[:, 0]
    return v[:3] / v[3]


def magnitude(v):
    return math.sqrt(np.sum(v**2))


def normalize(v):
    m = magnitude(v)
    if m == 0:
        return v
    return v / m


def ortho(l1, r1, b1, t1, n1, f1):
    dx = r1 - l1
    dy = t1 - b1
    dz = f1 - n1
    rx = -(r1 + l1) / (r1 - l1)
    ry = -(t1 + b1) / (t1 - b1)
    rz = -(f1 + n1) / (f1 - n1)
    return np.matrix(
        [[2.0 / dx, 0, 0, rx], [0, 2.0 / dy, 0, ry], [0, 0, -2.0 / dz, rz], [0, 0, 0, 1]]
    )


def perspective(fovy, aspect, n, f):
    s = 1.0 / math.tan(math.radians(fovy) / 2.0)
    sx, sy = s / aspect, s
    zz = (f + n) / (n - f)
    zw = 2 * f * n / (n - f)
    return np.matrix([[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, zz, zw], [0, 0, -1, 0]])


def frustum(x0, x1, y0, y1, z0, z1):
    a = (x1 + x0) / (x1 - x0)
    b = (y1 + y0) / (y1 - y0)
    c = -(z1 + z0) / (z1 - z0)
    d = -2 * z1 * z0 / (z1 - z0)
    sx = 2 * z0 / (x1 - x0)
    sy = 2 * z0 / (y1 - y0)
    return np.matrix([[sx, 0, a, 0], [0, sy, b, 0], [0, 0, c, d], [0, 0, -1, 0]])


def translate(xyz):
    x, y, z = xyz
    return np.matrix([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])


def scale(xyz):
    x, y, z = xyz
    return np.matrix([[x, 0, 0, 0], [0, y, 0, 0], [0, 0, z, 0], [0, 0, 0, 1]])


def _sincos(a):
    a = math.radians(a)
    return math.sin(a), math.cos(a)


def rotate(a, xyz):
    x, y, z = normalize(xyz)
    s, c = _sincos(a)
    nc = 1 - c
    return np.matrix(
        [
            [x * x * nc + c, x * y * nc - z * s, x * z * nc + y * s, 0],
            [y * x * nc + z * s, y * y * nc + c, y * z * nc - x * s, 0],
            [x * z * nc - y * s, y * z * nc + x * s, z * z * nc + c, 0],
            [0, 0, 0, 1],
        ]
    )


def rotx(a):
    s, c = _sincos(a)
    return np.matrix([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def roty(a):
    s, c = _sincos(a)
    return np.matrix([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def rotz(a):
    s, c = _sincos(a)
    return np.matrix([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def lookat(eye, target, up):
    F = target[:3] - eye[:3]
    f = normalize(F)
    U = normalize(up[:3])
    s = np.cross(f, U)
    u = np.cross(s, f)
    M = np.matrix(np.identity(4))
    M[:3, :3] = np.vstack([s, u, -f])
    T = translate(-eye)
    return M * T


def viewport(x, y, w, h):
    x, y, w, h = list(map(float, (x, y, w, h)))
    return np.matrix(
        [[w / 2, 0, 0, x + w / 2], [0, h / 2, 0, y + h / 2], [0, 0, 0.5, 0.5], [0, 0, 0, 1]]
    )
