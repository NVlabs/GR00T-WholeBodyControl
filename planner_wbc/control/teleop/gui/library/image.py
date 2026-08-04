#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Image class definition
======================

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

The image module contains the definition of the Image class, the container
that MakeHuman uses to handle images.

Image only depends on the numpy library, except when image have to be loaded
or saved to disk, in which case one of the back-ends (Qt or PIL) will have to
be imported (import happens only when needed).
"""

import time

import numpy as np

FILTER_NEAREST = 0  # Nearest neighbour resize filter (no real filtering)
FILTER_BILINEAR = 1  # Bi-linear filter
FILTER_BICUBIC = 2  # Bi-cubic filter (not supported with Qt, only PIL)


class Image(object):
    """Container for handling images.

    It is equipped with the necessary methods that one needs for loading
    and saving images and fetching their data, as well as many properties
    providing information about the loaded image.
    """

    def __init__(self, path=None, width=0, height=0, bitsPerPixel=32, components=None, data=None):
        """Image constructor.

        The Image can be constructed by an existing Image, a QPixmap, a
        QImage, loaded from a file path, or created empty.

        To construct the Image by copying the data from the source,
        pass the source as the first argument of the constructor.
        ``dest = Image(source)``

        To create the Image by sharing the data of another Image, pass
        the source Image as the data argument.
        ``dest = Image(data=sharedsource)``

        The data argument can be a numpy array of 3 dimensions (w, h, c) which
        will be used directly as the image's data, where w is the width, h is
        the height, and c is the number of channels.
        The data argument can also be a path to load.

        To create an empty Image, leave path=None, and specify the width and
        height. You can then optionally adjust the new Image's channels by
        setting bitsPerPixel = 8, 16, 24, 32, or components = 1, 2, 3, 4,
        which are equivalent to W (Grayscale), WA (Grayscale with Alpha),
        RGB, and RGBA respectively.
        """
        import image_qt as image_lib

        if path is not None:
            self._is_empty = False
            if isinstance(path, Image):
                # Create a copy of the image.
                self._data = path.data.copy()
            elif _isQPixmap(path):
                qimg = path.toImage()
                self._data = image_lib.load(qimg)
            else:  # Path string / QImage.
                self._data = image_lib.load(path)
                self.sourcePath = path
        elif data is not None:
            self._is_empty = False
            if isinstance(data, Image):
                # Share data between images.
                self._data = data.data
            elif isinstance(data, str):
                self._data = image_lib.load(data)
            else:  # Data array.
                self._data = data
        else:
            self._is_empty = True
            if components is None:
                if bitsPerPixel == 32:
                    components = 4
                elif bitsPerPixel == 24:
                    components = 3
                else:
                    raise NotImplementedError("bitsPerPixel must be 24 or 32")
            self._data = np.empty((height, width, components), dtype=np.uint8)
        self._data = np.ascontiguousarray(self._data)

        self.modified = time.time()

    @property
    def size(self):
        """Return the size of the Image as a (width, height) tuple."""
        h, w, c = self._data.shape
        return (w, h)

    @property
    def width(self):
        """Return the width of the Image in pixels."""
        h, w, c = self._data.shape
        return w

    @property
    def height(self):
        """Return the height of the Image in pixels."""
        h, w, c = self._data.shape
        return h

    @property
    def components(self):
        """Return the number of the Image channels."""
        h, w, c = self._data.shape
        return c

    @property
    def bitsPerPixel(self):
        """Return the number of bits per pixel used for the Image."""
        h, w, c = self._data.shape
        return c * 8

    @property
    def data(self):
        """Return the numpy ndarray that contains the Image data."""
        return self._data

    def save(self, path):
        """Save the Image to a file."""
        import image_qt as image_lib

        image_lib.save(path, self._data)

    def toQImage(self):
        """
        Get a QImage copy of this Image.
        Useful when the image should be shown in a Qt GUI
        """
        import image_qt

        # return image_qt.toQImage(self.data)
        # ^ For some reason caused problems
        if self.components == 1:
            fmt = image_qt.QtGui.QImage.Format_RGB888
            h, w, c = self.data.shape
            data = np.repeat(self.data[:, :, 0], 3).reshape((h, w, 3))
        elif self.components == 2:
            fmt = image_qt.QtGui.QImage.Format_ARGB32
            h, w, c = self.data.shape
            data = np.repeat(self.data[:, :, 0], 3).reshape((h, w, 3))
            data = np.insert(data, 3, values=self.data[:, :, 1], axis=2)
        elif self.components == 3:
            """
            fmt = image_qt.QtGui.QImage.Format_RGB888
            data = self.data
            """
            # The above causes a crash or misaligned image raster.
            # Quickhack solution:
            fmt = image_qt.QtGui.QImage.Format_ARGB32
            _data = self.convert(components=4).data
            # There appear to be channel mis-alignments, another hack:
            data = np.zeros(_data.shape, dtype=_data.dtype)
            data[:, :, :] = _data[:, :, [2, 1, 0, 3]]
        else:
            # components == 4
            fmt = image_qt.QtGui.QImage.Format_ARGB32
            data = self.data
        return image_qt.QtGui.QImage(data.tostring(), data.shape[1], data.shape[0], fmt)

    def resized_(self, width, height, filter=FILTER_NEAREST):
        if filter == FILTER_NEAREST:
            dw, dh = width, height
            sw, sh = self.size
            xmap = np.floor((np.arange(dw) + 0.5) * sw / dw).astype(int)
            ymap = np.floor((np.arange(dh) + 0.5) * sh / dh).astype(int)
            return self._data[ymap, :][:, xmap]
        else:
            # NOTE: bi-cubic filtering is not supported by Qt, use bi-linear
            import image_qt

            return image_qt.resized(self, width, height, filter=filter)

    def resized(self, width, height, filter=FILTER_NEAREST):
        """Get a resized copy of the Image."""
        return Image(data=self.resized_(width, height, filter))

    def resize(self, width, height, filter=FILTER_NEAREST):
        """Resize the Image to a specified size."""
        self._data = self.resized_(width, height, filter)
        self.modified = time.time()

    def blit(self, other, x, y):
        """Copy the contents of an Image to another.
        The target image may have a different size."""
        dh, dw, dc = self._data.shape
        sh, sw, sc = other._data.shape
        if sc != dc:
            raise ValueError("source image has incorrect format")
        sw = min(sw, dw - x)
        sh = min(sh, dh - y)
        self._data[y : y + sh, x : x + sw, :] = other._data

        self.modified = time.time()

    def flip_vertical(self):
        """Turn the Image upside down."""
        return Image(data=self._data[::-1, :, :])

    def flip_horizontal(self):
        """Flip the Image in the left-right direction."""
        return Image(data=self._data[:, ::-1, :])

    def __getitem__(self, xy):
        """Get the color of a specified pixel by using the
        square brackets operator.

        Example: my_color = my_image[(17, 42)]"""
        if not isinstance(xy, tuple) or len(xy) != 2:
            raise TypeError("tuple of length 2 expected")

        x, y = xy

        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("tuple of 2 ints expected")

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise IndexError("element index out of range")

        pix = self._data[y, x, :]
        if self.components == 4:
            return (pix[0], pix[1], pix[2], pix[3])
        elif self.components == 3:
            return (pix[0], pix[1], pix[2], 255)
        elif self.components == 2:
            return (pix[0], pix[0], pix[0], pix[1])
        elif self.components == 1:
            return (pix[0], pix[0], pix[0], 255)
        else:
            return None

    def __setitem__(self, xy, color):
        """Set the color of a pixel using the square brackets
        operator.

        Example: my_image[(17, 42)] = (0, 255, 64, 255)"""
        if not isinstance(xy, tuple) or len(xy) != 2:
            raise TypeError("tuple of length 2 expected")

        x, y = xy

        if not isinstance(x, int) or not isinstance(y, int):
            raise TypeError("tuple of 2 ints expected")

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise IndexError("element index out of range")

        if not isinstance(color, tuple):
            raise TypeError("tuple expected")

        self._data[y, x, :] = color
        self.modified = time.time()

    def convert(self, components):
        """Convert the Image to a different color format.

        'components': The number of color channels the Image
        will have after the conversion."""
        if self.components == components:
            return self

        hasAlpha = self.components in (2, 4)
        needAlpha = components in (2, 4)

        if hasAlpha:
            alpha = self._data[..., -1]
            color = self._data[..., :-1]
        else:
            alpha = None
            color = self._data

        isMono = self.components in (1, 2)
        toMono = components in (1, 2)

        if isMono and not toMono:
            color = np.dstack((color, color, color))
        elif toMono and not isMono:
            color = np.sum(color.astype(np.uint16), axis=-1) / 3
            color = color.astype(np.uint8)[..., None]

        if needAlpha and alpha is None:
            alpha = np.zeros_like(color[..., :1]) + 255

        if needAlpha:
            data = np.dstack((color, alpha))
        else:
            data = color

        return type(self)(data=data)

    def markModified(self):
        """Mark the Image as modified."""
        self.modified = time.time()
        self._is_empty = False

    @property
    def isEmpty(self):
        """
        Returns True if the Image is empty or new.
        Returns False if the Image contains data or has been modified.
        """
        return self._is_empty


def _isQPixmap(img):
    """
    Test an image object for being a QPixmap instance if Qt libraries were
    loaded in the application.
    """
    import sys

    if "PyQt5" in list(sys.modules.keys()):
        import image_qt

        return isinstance(img, image_qt.QtGui.QPixmap)
    else:
        return False
