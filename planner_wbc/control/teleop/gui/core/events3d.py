#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehumancommunity.org/

**Github Code Home Page:**    https://github.com/makehumancommunity/

**Authors:**           Marc Flerackers

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

This module contains classes to allow an object to handle events.
"""
import library.log as log
import library.profiler as profiler
from library.universal import G


class Event:
    """
    Base class for all events, does not contain information.
    """

    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "event:"


class MouseEvent(Event):
    """
    Contains information about a mouse event.

    :param button: the button that is pressed in case of a mousedown or mouseup event,
    or button flags in case of a mousemove event.
    :type button: int
    :param x: the x position of the mouse in window coordinates.
    :type x: int
    :param y: the y position of the mouse in window coordinates.
    :type y: int
    :param dx: the difference in x position in case of a mousemove event.
    :type dx: int
    :param dy: the difference in y position in case of a mousemove event.
    :type dy: int
    """

    def __init__(self, button, x, y, dx=0, dy=0):
        self.button = button
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy

    def __repr__(self):
        return "MouseEvent(%d, %d, %d, %d, %d)" % (self.button, self.x, self.y, self.dx, self.dy)


class MouseWheelEvent(Event):
    """
    Contains information about a mouse wheel event.

    :param wheelDelta: the amount and direction that the wheel was scrolled.
    :type wheelDelta: int
    """

    def __init__(self, wheelDelta, x, y):
        self.wheelDelta = wheelDelta
        self.x = x
        self.y = y

    def __repr__(self):
        return "MouseWheelEvent(%d)" % self.wheelDelta


class KeyEvent(Event):
    """
    Contains information about a keyboard event.

    :param key: the key code of the key that was pressed or released.
    :type key: int
    :param character: the unicode character if the key represents a character.
    :type character: unicode
    :param modifiers: the modifier keys that were down at the time of pressing the key.
    :type modifiers: int
    """

    def __init__(self, key, character, modifiers):
        self.key = key
        self.character = character
        self.modifiers = modifiers

    def __repr__(self):
        return "KeyEvent(%d, %04x %s, %d)" % (
            self.key,
            ord(self.character),
            self.character,
            self.modifiers,
        )


class FocusEvent(Event):
    """
    Contains information about a view focus/blur event

    :param blurred: the view that lost the focus.
    :type blurred: guid3d.View
    :param focused: the view that gained the focus.
    :type focused: guid3d.View
    """

    def __init__(self, blurred, focused):
        self.blurred = blurred
        self.focused = focused

    def __repr__(self):
        return "FocusEvent(%s, %s)" % (self.blurred, self.focused)


class ResizeEvent(Event):
    """
    Contains information about a resize event

    :param width: the new width of the window in pixels.
    :type width: int
    :param height: the new height of the window in pixels.
    :type height: int
    :param fullscreen: the new fullscreen state of the window.
    :type fullscreen: Boolean
    :param dx: the change in width of the window in pixels.
    :type dx: int
    :param dy: the change in height of the window in pixels.
    :type dy: int
    """

    def __init__(self, width, height, fullscreen):
        self.width = width
        self.height = height
        self.fullscreen = fullscreen

    def __repr__(self):
        return "ResizeEvent(%d, %d, %s)" % (self.width, self.height, self.fullscreen)


class HumanEvent(Event):
    def __init__(self, human, change):
        self.human = human
        self.change = change

    def __repr__(self):
        return "event: %s, %s" % (self.human, self.change)


class EventHandler(object):
    """
    Base event handler class. Derive from this class if an object needs to be able to have events attached to it.
    Currently only one event per event name can be attached. This is because we either allow a class method or
    a custom method to be attached as event handling method. Since the custom method replaces the class method,
    it is needed in some case to call the base class's method from the event handling method.

    There are 2 ways to attach handlers:

    1. Override the method. This is the most appropriate way when you want to add distinctive behaviour to many
    EventHandlers.

    ::

        class Widget(View):

            def onMouseDown(self, event):
                #Handle event

    2. Use the event decorator. This is the most appropriate way when you want
    to attach distinctive behaviour to one EventHandler.

    ::

        widget = Widget()

        @widget.mhEvent:
        def onMouseDown(event):
            #Handle event

    Note that self is not passed to the handler in this case, which should not be a problem
    as you can just use the variable since you are creating a closure.
    """

    def __init__(self):
        self.sortOrder = None

    _logger = log.getLogger("mh.callEvent")
    _depth = 0

    def callEvent(self, eventType, event):
        if not hasattr(self, eventType):
            return False
        topLevel = EventHandler._depth == 0
        EventHandler._depth += 1
        try:
            self._logger.debug("callEvent[%d]: %s.%s(%s)", self._depth, self, eventType, event)
            method = getattr(self, eventType)
            if topLevel and profiler.active():
                profiler.accum("method(event)", globals(), locals())
            else:
                method(event)
        except Exception as _:
            log.warning("Exception during event %s", eventType, exc_info=True)
            self.eventFailed(EventHandler._depth)
        EventHandler._depth -= 1
        if topLevel:
            self._logger.debug("callEvent: done")
            if G.app:
                G.app.redraw()
            return True
        return False

    def eventFailed(self, level):
        # Reset progress
        if G.app:
            G.app.progress(1)

    def attachEvent(self, eventName, eventMethod):
        setattr(self, eventName, eventMethod)

    def detachEvent(self, eventName):
        delattr(self, eventName)

    def mhEvent(self, eventMethod):
        self.attachEvent(eventMethod.__name__, eventMethod)
