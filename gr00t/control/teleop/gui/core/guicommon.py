#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehumancommunity.org/

**Github Code Home Page:**    https://github.com/makehumancommunity/

**Authors:**           Joel Palmius, Marc Flerackers

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

Common GUI elements extracted from gui3d to minimize coupling with gui backend.
"""

import collections

import events3d
import library.log as log
import library.matrix as matrix
import numpy as np
import shared.material as material


class Action(object):
    def __init__(self, name):
        self.name = name

    def do(self):
        return True

    def undo(self):
        return True


# Wrapper around Object3D
class Object(events3d.EventHandler):
    """
    An object on the screen.

    :param position: The position in 3d space.
    :type position: list or tuple
    :param mesh: The mesh object.
    :param visible: Wether the object should be initially visible.
    :type visible: Boolean
    """

    def __init__(self, mesh, position=[0.0, 0.0, 0.0], visible=True):

        if mesh.object:
            raise RuntimeError("This mesh is already attached to an object")

        self.mesh = mesh
        self.mesh.object = self
        self.mesh.setVisibility(visible)

        self._material = material.Material(self.name + "_Material")  # Render material

        self._loc = np.zeros(3, dtype=np.float32)
        self._rot = np.zeros(3, dtype=np.float32)
        self._scale = np.ones(3, dtype=np.float32)

        self.setLoc(*position)
        # Set to true to make the rotation of this object independent of the camera rotation
        self.lockRotation = False
        # Set to true to automatically set Y loc (position) to height of ground helper joint of the human
        self.placeAtFeet = False

        self._view = None

        self.visible = visible
        self.excludeFromProduction = False  # Set to true to exclude from production renders

        self.proxy = None

        self.__seedMesh = self.mesh
        self.__proxyMesh = None
        self.__subdivisionMesh = None
        self.__proxySubdivisionMesh = None

        self.setUVMap(mesh.material.uvMap)

    def __str__(self):
        return "<guicommon.Object %s>" % self.name

    # TODO
    def clone(self):
        raise NotImplementedError("You probably want to do object.mesh.clone()")

    def _attach(self):
        if self.view.isVisible() and self.visible:
            self.mesh.setVisibility(1)
        else:
            self.mesh.setVisibility(0)

        for mesh in self._meshes():
            self.attachMesh(mesh)

    def _detach(self):
        for mesh in self._meshes():
            self.detachMesh(mesh)

    @staticmethod
    def attachMesh(mesh):
        import object3d
        import selection

        selection.selectionColorMap.assignSelectionID(mesh)
        object3d.Object3D.attach(mesh)

    @staticmethod
    def detachMesh(mesh):
        import object3d

        object3d.Object3D.detach(mesh)

    def _meshes(self):
        for mesh in (
            self.__seedMesh,
            self.__proxyMesh,
            self.__subdivisionMesh,
            self.__proxySubdivisionMesh,
        ):
            if mesh is not None:
                yield mesh

    @property
    def view(self):
        if not self._view or not isinstance(self._view, collections.abc.Callable):
            return None
        return self._view()

    def show(self):
        self.visible = True
        self.setVisibility(True)

    def hide(self):
        self.visible = False
        self.setVisibility(False)

    def isVisible(self):
        return self.visible

    @property
    def name(self):
        return self.mesh.name

    def setVisibility(self, visibility):
        if not self.view:
            self.mesh.setVisibility(0)
        if self.view.isVisible() and self.visible and visibility:
            self.mesh.setVisibility(1)
        else:
            self.mesh.setVisibility(0)

    def getPriority(self):
        return self.mesh.priority

    def setPriority(self, priority):
        self.mesh.priority = priority

    priority = property(getPriority, setPriority)

    ##
    # Orientation properties
    ##

    def getLoc(self):
        result = np.zeros(3, dtype=np.float32)
        result[:] = self._loc[:]
        if self.placeAtFeet:
            from core import G

            human = G.app.selectedHuman
            result[1] = human.getJointPosition("ground")[1]
        return result

    def setLoc(self, locx, locy, locz):
        """
        This method is used to set the location of the object in the 3D coordinate space of the scene.

        :param locx: The x coordinate of the object.
        :type locx: float
        :param locy: The y coordinate of the object.
        :type locy: float
        :param locz: The z coordinate of the object.
        :type locz: float
        """
        self._loc[...] = (locx, locy, locz)

    loc = property(getLoc, setLoc)

    def get_x(self):
        return self.loc[0]

    def set_x(self, x):
        self._loc[0] = x

    x = property(get_x, set_x)

    def get_y(self):
        return self.loc[1]

    def set_y(self, y):
        self._loc[1] = y

    y = property(get_y, set_y)

    def get_z(self):
        return self.loc[2]

    def set_z(self, z):
        self._loc[2] = z

    z = property(get_z, set_z)

    def get_rx(self):
        return self._rot[0]

    def set_rx(self, rx):
        self._rot[0] = rx

    rx = property(get_rx, set_rx)

    def get_ry(self):
        return self._rot[1]

    def set_ry(self, ry):
        self._rot[1] = ry

    ry = property(get_ry, set_ry)

    def get_rz(self):
        return self._rot[2]

    def set_rz(self, rz):
        self._rot[2] = rz

    rz = property(get_rz, set_rz)

    def get_sx(self):
        return self._scale[0]

    def set_sx(self, sx):
        self._scale[0] = sx

    sx = property(get_sx, set_sx)

    def get_sy(self):
        return self._scale[1]

    def set_sy(self, sy):
        self._scale[1] = sy

    sy = property(get_sy, set_sy)

    def get_sz(self):
        return self._scale[2]

    def set_sz(self, sz):
        self._scale[2] = sz

    sz = property(get_sz, set_sz)

    def getPosition(self):
        return [self.x, self.y, self.z]

    def setPosition(self, position):
        self.setLoc(position[0], position[1], position[2])

    def getRotation(self):
        return [self.rx, self.ry, self.rz]

    def setRotation(self, rotation):
        rotation[0] = rotation[0] % 360
        rotation[1] = rotation[1] % 360

        if rotation[2] != 0.0:
            log.warning("Setting a non-zero rotation around Z axis is not supported!")
            rotation[2] = 0.0

        self.setRot(rotation[0], rotation[1], rotation[2])

    def setRot(self, rx, ry, rz):
        """
        This method sets the orientation of the object in the 3D coordinate space of the scene.

        :param rx: Rotation around the x-axis.
        :type rx: float
        :param ry: Rotation around the y-axis.
        :type ry: float
        :param rz: Rotation around the z-axis.
        :type rz: float
        """
        self._rot[...] = (rx, ry, rz)

    rot = property(getRotation, setRotation)

    def setScale(self, scale, scaleY=None, scaleZ=1):
        """
        This method sets the scale of the object in the 3D coordinate space of
        the scene, relative to the initially defined size of the object.

        :param scale: Scale along the x-axis, uniform scale if other params not
                      specified.
        :type scale: float
        :param scaleY: Scale along the x-axis.
        :type scaleY: float
        :param scaleZ: Scale along the x-axis.
        :type scaleZ: float
        """
        if scaleY is None:
            scaleY = scale
            scaleZ = scale
        self._scale[...] = (scale, scaleY, scaleZ)

    def getScale(self):
        return list(self._scale)

    scale = property(getScale, setScale)

    @property
    def transform(self):
        m = matrix.translate(self.loc)
        if any(x != 0 for x in self.rot):
            m = m * matrix.rotx(self.rx)
            m = m * matrix.roty(self.ry)
            m = m * matrix.rotz(self.rz)
        if any(x != 1 for x in self.scale):
            m = m * matrix.scale(self.scale)
        return m

    def getSeedMesh(self):
        return self.__seedMesh

    def getProxyMesh(self):
        return self.__proxyMesh

    def updateProxyMesh(self, fit_to_posed=False):
        if self.proxy and self.__proxyMesh:
            self.proxy.update(self.__proxyMesh, fit_to_posed)
            self.__proxyMesh.update()

    def isProxied(self):
        return self.mesh == self.__proxyMesh or self.mesh == self.__proxySubdivisionMesh

    def setProxy(self, proxy):
        isSubdivided = self.isSubdivided()

        if self.proxy:
            self.proxy = None
            self.detachMesh(self.__proxyMesh)
            self.__proxyMesh.clear()
            self.__proxyMesh = None
            if self.__proxySubdivisionMesh:
                self.detachMesh(self.__proxySubdivisionMesh)
                self.__proxySubdivisionMesh.clear()
                self.__proxySubdivisionMesh = None
            self.mesh = self.__seedMesh
            self.mesh.setVisibility(1)

        if proxy:

            self.proxy = proxy

            self.__proxyMesh = proxy.object.mesh.clone()
            self.__proxyMesh.object = self

            # Copy attributes from human mesh to proxy mesh
            for attr in ("visibility", "pickable", "cameraMode"):
                setattr(self.__proxyMesh, attr, getattr(self.mesh, attr))

            self.updateProxyMesh()

            # Attach to GL object if this object is attached to viewport
            if self.__seedMesh.object3d:
                self.attachMesh(self.__proxyMesh)

            self.mesh.setVisibility(0)
            self.mesh = self.__proxyMesh
            self.mesh.setVisibility(1)

        self.setSubdivided(isSubdivided)

    def getProxy(self):
        return self.proxy

    def getSubdivisionMesh(self, update=True):
        """
        Create or update the Catmull-Clark subdivided (or smoothed) mesh for
        this mesh.
        This does not change the status of isSubdivided(), use setSubdivided()
        for that.

        If this mesh is doubled by a proxy, when isProxied() is true, a
        subdivision mesh for the proxy is used.

        Returns the subdivided mesh data.

        """
        import catmull_clark_subdivision as cks

        if self.isProxied():
            if not self.__proxySubdivisionMesh:
                self.__proxySubdivisionMesh = cks.createSubdivisionObject(self.__proxyMesh, None)
                if self.__seedMesh.object3d:
                    self.attachMesh(self.__proxySubdivisionMesh)
            elif update:
                cks.updateSubdivisionObject(self.__proxySubdivisionMesh)

            return self.__proxySubdivisionMesh
        else:
            if not self.__subdivisionMesh:
                self.__subdivisionMesh = cks.createSubdivisionObject(
                    self.__seedMesh, self.staticFaceMask
                )
                if self.__seedMesh.object3d:
                    self.attachMesh(self.__subdivisionMesh)
            elif update:
                cks.updateSubdivisionObject(self.__subdivisionMesh)

            return self.__subdivisionMesh

    def isSubdivided(self):
        """
        Returns whether this mesh is currently set to be subdivided
        (or smoothed).

        """
        return self.mesh == self.__subdivisionMesh or self.mesh == self.__proxySubdivisionMesh

    def setSubdivided(self, flag, update=True):
        """
        Set whether this mesh is to be subdivided (or smoothed).
        When set to true, the subdivision mesh is automatically created or
        updated.

        """
        if flag == self.isSubdivided():
            return False

        if flag:
            self.mesh.setVisibility(0)
            originalMesh = self.mesh
            self.mesh = self.getSubdivisionMesh(update)
            self.mesh.setVisibility(1)
        else:
            originalMesh = (
                self.__seedMesh if self.mesh == self.__subdivisionMesh else self.__proxyMesh
            )

            self.mesh.setVisibility(0)
            self.mesh = originalMesh
            if update:
                self.mesh.calcNormals()
                self.mesh.update()
            self.mesh.setVisibility(1)

        return True

    def updateSubdivisionMesh(self, rebuildIndexBuffer=False):
        if rebuildIndexBuffer:
            # Purge old subdivision mesh and recalculate entirely
            self.setSubdivided(False)
            self.__subdivisionMesh = self.__proxySubdivisionMesh = None
            self.setSubdivided(True)
        else:
            self.getSubdivisionMesh(True)

    def _setMeshUVMap(self, filename, mesh):
        if filename == self.material.uvMap:
            # No change, do nothing
            return

        if not hasattr(mesh, "_originalUVMap") or not mesh._originalUVMap:
            # Backup original mesh UVs
            mesh._originalUVMap = dict()
            mesh._originalUVMap["texco"] = mesh.texco
            mesh._originalUVMap["fuvs"] = mesh.fuvs
            # self._originalUVMap['fvert'] = mesh.fvert
            # self._originalUVMap['group'] = mesh.group

        faceMask = mesh.getFaceMask()
        faceGroups = mesh.group

        self.material.uvMap = filename

        if not filename:
            # Restore original UVs
            mesh.setUVs(mesh._originalUVMap["texco"])
            mesh.setFaces(mesh.fvert, mesh._originalUVMap["fuvs"], faceGroups)
            mesh._originalUVMap = None
        else:
            uvset = material.UVMap(filename)
            uvset.read(mesh, filename)

            if len(uvset.fuvs) != len(mesh.fuvs):
                raise NameError(
                    "The UV file %s is not valid for mesh %s. \
                Number of faces %d != %d"
                    % (filename, mesh.name, len(uvset.fuvs), len(mesh.fuvs))
                )

            mesh.setUVs(uvset.uvs)
            mesh.setFaces(mesh.fvert, uvset.fuvs, faceGroups)

        mesh.changeFaceMask(faceMask)
        mesh.updateIndexBuffer()

    @property
    def staticFaceMask(self):
        if not hasattr(self, "_staticFaceMask") or self._staticFaceMask is None:
            # If not already set, consider the current face mask state of the
            # mesh to be the static face mask
            self._staticFaceMask = self.__seedMesh.face_mask.copy()
        return self._staticFaceMask

    def changeVertexMask(self, vertsMask):
        """
        Apply a face mask to the meshes (original seed mesh, subdivided mesh
        and proxied meshes) specified by a vertex mask.
        The vertex mask is a list of booleans, one for each vertex, where True
        means not masked (visible), and False means masked (hidden).
        A face is masked if all of the vertices that define it are masked.
        """
        if vertsMask is None:
            # Undo face mask set by vertex mask
            self.__seedMesh.changeFaceMask(self.staticFaceMask)
            self.__seedMesh.updateIndexBufferFaces()
            if self.__subdivisionMesh:
                self.__subdivisionMesh.changeFaceMask(self.staticFaceMask)
                self.__subdivisionMesh.updateIndexBufferFaces()
            if self.__proxyMesh:
                self.__proxyMesh.changeFaceMask(
                    np.ones(self.__proxyMesh.getFaceCount(), dtype=bool)
                )
                self.__proxyMesh.updateIndexBufferFaces()
            if self.__proxySubdivisionMesh:
                self.__proxySubdivisionMesh.changeFaceMask(
                    np.ones(self.__proxySubdivisionMesh.getFaceCount(), dtype=bool)
                )
                self.__proxySubdivisionMesh.updateIndexBufferFaces()
            return

        # Mask seed mesh
        faceMask = self.__seedMesh.getFaceMaskForVertices(np.argwhere(vertsMask)[..., 0])
        self.__seedMesh.changeFaceMask(np.logical_and(faceMask, self.staticFaceMask))
        self.__seedMesh.updateIndexBufferFaces()

        import log

        log.debug("%s faces masked for %s", np.count_nonzero(~faceMask), self.__seedMesh.name)

        # Mask smoothed seed mesh
        if self.__subdivisionMesh:
            # Remap faceMask to subdivision mesh base faces, accounting for the
            # excluded faces of the static facemask (staticFaceMask).

            # Statically masked faces (staticFaceMask) are excluded from
            # subdivision mesh geometry, for performance.
            # Dynamically masked faces (eg. using this method) are not excluded
            # from the subdivision mesh and simply masked, allowing faster
            # changes to the face mask without requiring a rebuild of the
            # subdivision mesh.
            self.__subdivisionMesh.changeFaceMask(faceMask)
            self.__subdivisionMesh.updateIndexBufferFaces()

        # Mask proxy and subdivided proxy mesh
        if self.__proxyMesh:
            import proxy

            # Transfer face mask to proxy
            proxyVertMask = proxy.transferVertexMaskToProxy(vertsMask, self.proxy)
            proxyFaceMask = self.__proxyMesh.getFaceMaskForVertices(
                np.argwhere(proxyVertMask)[..., 0]
            )

            self.__proxyMesh.changeFaceMask(proxyFaceMask)
            self.__proxyMesh.updateIndexBufferFaces()

            if self.__proxySubdivisionMesh:
                self.__proxySubdivisionMesh.changeFaceMask(proxyFaceMask)
                self.__proxySubdivisionMesh.updateIndexBufferFaces()

    ##
    # Material properties
    ##

    def setTexture(self, path):
        """
        This method is used to specify the path of a file on disk containing the object texture.

        :param path: The path of a texture file.
        :type path: str
        :param cache: The texture cache to use.
        :type cache: dict
        """
        self.material.diffuseTexture = path

    def getTexture(self):
        return self.material.diffuseTexture

    texture = property(getTexture, setTexture)

    def clearTexture(self):
        """
        This method is used to clear an object's texture.
        """
        self.material.diffuseTexture = None

    def hasTexture(self):
        return self.texture is not None

    def setShader(self, shader):
        """
        This method is used to specify the shader.

        :param shader: The path to a pair of shader files.
        :type shader: string
        """
        self.material.setShader(shader)

    def getShader(self):
        return self.material.shader

    shader = property(getShader, setShader)

    @property
    def shaderObj(self):
        return self.material.shaderObj

    def configureShading(
        self, diffuse=None, bump=None, normal=None, displacement=None, spec=None, vertexColors=None
    ):
        """
        Configure shader options and set the necessary properties based on
        the material configuration of this object.
        This can be done without an actual shader being set for this object.
        Call this method when changes are made to the material property.
        """
        self.material.configureShading(diffuse, bump, normal, displacement, spec, vertexColors)

    def getMaterial(self):
        return self._material

    def setMaterial(self, material):
        if self.material.uvMap != material.uvMap:
            # UV map has changed
            self.setUVMap(material.uvMap)
        self._material.copyFrom(material)

    material = property(getMaterial, setMaterial)

    def setShaderParameter(self, name, value):
        """
        Updates the shader parameters.
        """
        self.material.setShaderParameter(name, value)

    @property
    def shaderParameters(self):
        return self.material.shaderParameters

    @property
    def shaderConfig(self):
        return self.material.shaderConfig

    @property
    def shaderDefines(self):
        return self.material.shaderDefines

    def addShaderDefine(self, defineStr):
        self.material.addShaderDefine(defineStr)

    def removeShaderDefine(self, defineStr):
        self.material.removeShaderDefine(defineStr)

    def clearShaderDefines(self):
        self.material.clearShaderDefines()

    def setShadeless(self, shadeless):
        """
        This method is used to specify whether or not the object is affected by lights.
        This is used for certain GUI controls to give them a more 2D type
        appearance (predominantly the top bar of GUI controls).

        NOTE enabling this option disables the use of the shader configured in the material.

        :param shadeless: Whether or not the object is unaffected by lights.
        :type shadeless: Boolean
        """
        self.material.shadeless = shadeless

    def getShadeless(self):
        return self.material.shadeless

    shadeless = property(getShadeless, setShadeless)

    def setDepthless(self, depthless):
        """
        This method is used to specify whether or not the object occludes or is occluded
        by other objects

        :param depthless: Whether or not the object is occluded or occludes.
        :type depthless: Boolean
        """
        self.material.depthless = depthless

    def getDepthless(self):
        return self.material.depthless

    depthless = property(getDepthless, setDepthless)

    def setSolid(self, solid):
        """
        This method is used to specify whether or not the object is drawn solid or wireframe.

        :param solid: Whether or not the object is drawn solid or wireframe.
        :type solid: Boolean
        """
        self.material.wireframe = not solid

    def isSolid(self):
        return self.solid

    def getSolid(self):
        return not self.material.wireframe

    solid = property(getSolid, setSolid)

    def setCull(self, cull):
        """
        This method is used to specify whether or not the object is back-face culled.

        :param cull: Whether and how to cull
        :type cull: 0 => no culling, >0 => draw front faces, <0 => draw back faces
        """

        # Because we don't really need frontface culling, we simplify to only backface culling
        if (isinstance(cull, bool) and cull) or cull > 0:
            self.material.backfaceCull = True
        else:
            self.material.backfaceCull = False

    def getCull(self):
        # Because we don't really need frontface culling, we simplify to only backface culling
        if self.material.backfaceCull:
            return 1
        else:
            return 0

    cull = property(getCull, setCull)

    def getAlphaToCoverage(self):
        return self.material.alphaToCoverage

    def setAlphaToCoverage(self, a2cEnabled):
        self.material.alphaToCoverage = a2cEnabled

    alphaToCoverage = property(getAlphaToCoverage, setAlphaToCoverage)

    def setUVMap(self, filename):
        subdivided = self.isSubdivided()

        if subdivided:
            # Re-generate the subdivided mesh
            self.setSubdivided(False)

        # Remove stale subdivision cache if present
        self.__subdivisionMesh = None

        # Set uv map on original, unsubdivided, unproxied mesh
        self._setMeshUVMap(filename, self.__seedMesh)

        if self.isProxied():
            # TODO transfer UV coordinates to proxy mesh
            pass

        if subdivided:
            # Re-generate the subdivided mesh with new UV coordinates
            self.setSubdivided(True)

    def onMouseDown(self, event):
        if self.view:
            self.view.callEvent("onMouseDown", event)
        else:
            import log

            log.debug("FAILED: mouseDown")

    def onMouseMoved(self, event):
        if self.view:
            self.view.callEvent("onMouseMoved", event)
        else:
            import log

            log.debug("FAILED: mouseMoved")

    def onMouseDragged(self, event):
        if self.view:
            self.view.callEvent("onMouseDragged", event)
        else:
            import log

            log.debug("FAILED: mouseDragged")

    def onMouseUp(self, event):
        if self.view:
            self.view.callEvent("onMouseUp", event)

    def onMouseEntered(self, event):
        if self.view:
            self.view.callEvent("onMouseEntered", event)
        else:
            import log

            log.debug("FAILED: mouseEntered")

    def onMouseExited(self, event):
        if self.view:
            self.view.callEvent("onMouseExited", event)
        else:
            import log

            log.debug("FAILED: mouseExited")

    def onClicked(self, event):
        if self.view:
            self.view.callEvent("onClicked", event)
        else:
            import log

            log.debug("FAILED: clicked")

    def onMouseWheel(self, event):
        if self.view:
            self.view.callEvent("onMouseWheel", event)
