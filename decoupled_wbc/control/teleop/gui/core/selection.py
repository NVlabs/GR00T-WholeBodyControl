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


class SelectionColorMap:
    """
    The objects support the use of a technique called *Selection Using Unique
    Color IDs*, that internally uses color-coding of components within the
    scene to support the selection of objects by the user using the mouse.

    This technique generates a sequence of colors (color IDs), assigning a
    unique color to each uniquely selectable object or component in the scene.
    These colors are not displayed, but are used by MakeHuman to generates an
    unseen image of the various selectable elements. This image uses the same
    camera settings currently being used for the actual, on-screen image.
    When the mouse is clicked, the position of the mouse is used with the
    unseen image to retrieve a color. MakeHuman uses this color as an ID to
    identify which object or component the user clicked with the mouse.

    This technique uses glReadPixels() to read the single pixel at the
    current mouse location, using the unseen, color-coded image.

    For further information on this technique, see:

      - http://www.opengl.org/resources/faq/technical/selection.htm and
      - http://wiki.gamedev.net/index.php/OpenGL_Selection_Using_Unique_Color_IDs

    **Note.** Because the 3D engine uses glDrawElements in a highly opimized
    way and each vertex can have only one color ID, there there is a known
    problem with selecting individual faces with very small FaceGroups using
    this technique. However, this is not a major problem for MakeHuman, which
    doesn't use such low polygon groupings.

    - **self.colorIDToFaceGroup**: *Dictionary of colors IDs* A dictionary of the color IDs used for
      selection (see MakeHuman Selectors, above).
    - **self.colorID**: *float list* A progressive color ID.

    The attributes *self.colorID* and *self.colorIDToFaceGroup*
    support a technique called *Selection Using Unique Color IDs* to make each
    FaceGroup independently clickable.

    The attribute *self.colorID* stores a progressive color that is incremented for each successive
    FaceGroup added to the scene.
    The *self.colorIDToFaceGroup* attribute contains a list that serves as a directory to map
    each color back to the corresponding FaceGroup by using its color ID.
    """

    def __init__(self):

        self.colorIDToFaceGroup = {}
        self.colorID = 0

    def assignSelectionID(self, obj):
        """
        This method generates a new, unique color ID for each FaceGroup,
        within a particular Object3D object, that forms a part of this scene3D
        object. This color ID can subsequently be used in a non-displayed
        image map to determine the FaceGroup that a mouse click was made in.

        This method loops through the FaceGroups, assigning the next color
        in the sequence to each subsequent FaceGroup. The color value is
        written into a 'dictionary' to serve as a color ID that can be
        translated back into the corresponding FaceGroup name when a mouse
        click is detected.
        This is part of a technique called *Selection Using Unique Color IDs*
        to make each FaceGroup independently clickable.

        :param obj: The object3D object for which color dictionary entries need to be generated.
        :type obj: module3d.Object 3D
        """
        for g in obj.faceGroups:
            self.colorID += 1

            # 555 to 24-bit rgb

            idR = (self.colorID % 32) * 8
            idG = ((self.colorID >> 5) % 32) * 8
            idB = ((self.colorID >> 10) % 32) * 8

            g.colorID = (idR, idG, idB)

            self.colorIDToFaceGroup[self.colorID] = g

    def getSelectedFaceGroup(self, picked):
        """
        This method uses a non-displayed image containing color-coded faces
        to return the index of the FaceGroup selected by the user with the mouse.
        This is part of a technique called *Selection Using Unique Color IDs* to make each
        FaceGroup independently clickable.

        :return: The selected face group.
        :rtype: :py:class:`module3d.FaceGroup`
        """
        # Force integer divide
        IDkey = picked[0] // 8 | picked[1] // 8 << 5 | picked[2] // 8 << 10  # 555

        # print "DEBUG COLOR PICKED: %s,%s,%s %s"%(picked[0], picked[1], picked[2], IDkey)

        try:
            groupSelected = self.colorIDToFaceGroup[IDkey]
        except Exception:
            print("Color %s (%s) not found" % (IDkey, picked))
            # print (groupSelected.name)
            # this print should only come on while debugging color picking
            # print ('Color %s (%s) not found' % (IDkey, picked))
            groupSelected = None
        return groupSelected

    def getSelectedFaceGroupAndObject(self, picked):
        """
        This method determines whether a FaceGroup or a non-selectable zone has been
        clicked with the mouse. It returns a tuple, showing the FaceGroup and the parent
        Object3D object, or None.
        If no object is picked, this method will simply print \"no clickable zone.\"

        :return: The selected face group and object.
        :rtype: (:py:class:`module3d.FaceGroup`, :py:class:`module3d.Object3d`)
        """

        facegroupPicked = self.getSelectedFaceGroup(picked)
        if facegroupPicked:
            objPicked = facegroupPicked.parent
            return (facegroupPicked, objPicked)
        else:
            # this print should only be made while debugging picking
            # print ('not a clickable zone')
            return None


selectionColorMap = SelectionColorMap()
