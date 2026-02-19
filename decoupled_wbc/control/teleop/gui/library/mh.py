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

Python compatibility layer replacing the old C functions of MakeHuman.
"""

from library.universal import G

cameras = G.cameras


def setClearColor(r, g, b, a):
    G.clearColor = (r, g, b, a)


def setCaption(caption):
    G.app.mainwin.setWindowTitle(caption)


def changeCategory(category):
    G.app.mainwin.tabs.changeTab(category)


def changeTask(category, task):
    if not G.app.mainwin.tabs.findTab(category):
        return
    changeCategory(category)
    G.app.mainwin.tabs.findTab(category).child.changeTab(task)


def refreshLayout():
    G.app.mainwin.refreshLayout()


def addPanels():
    return G.app.mainwin.addPanels()


def showPanels(left, right):
    return G.app.mainwin.showPanels(left, right)


def addTopWidget(widget, *args, **kwargs):
    return G.app.mainwin.addTopWidget(widget, *args, **kwargs)


def removeTopWidget(widget):
    return G.app.mainwin.removeTopWidget(widget)


def addToolBar(name):
    return G.app.mainwin.addToolBar(name)


def redraw():
    G.app.redraw()


def getKeyModifiers():
    return int(G.app.keyboardModifiers())


def addTimer(milliseconds, callback):
    return G.app.addTimer(milliseconds, callback)


def removeTimer(id):
    G.app.removeTimer(id)


def callAsync(func, *args, **kwargs):
    G.app.callAsync(func, *args, **kwargs)


def getSetting(setting_name):
    return G.app.getSetting(setting_name)


def setSetting(setting_name, value):
    G.app.setSetting(setting_name, value)
