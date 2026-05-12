#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
**Project Name:**      MakeHuman

**Product Home Page:** http://www.makehumancommunity.org/

**Github Code Home Page:**    https://github.com/makehumancommunity/

**Authors:**           Marc Flerackers, Glynn Clements, Jonas Hauquier

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

import weakref

# import material
import library.log as log
import numpy as np

# import unique # Bugfix for numpy.unique on older numpy versions


class FaceGroup(object):
    """
    A FaceGroup (a group of faces with a unique name).

    Each Face object can be part of one FaceGroup. Each face object has an
    attribute, *group*, storing the FaceGroup it is a member of.

    The FaceGroup object contains a list of the faces in the group and must be
    kept in sync with the FaceGroup references stored by the individual faces.

    .. py:attribute:: name

        The name. str

    .. py:attribute:: parent

        The parent. :py:class:`module3d.Object3D`

    :param name: The name of the group.
    :type name: str
    """

    black = np.zeros(3, dtype=np.uint8)

    def __init__(self, object, name, idx):
        self.object = object
        self.name = name
        self.idx = idx
        self.color = None
        self.colorID = self.black.copy()

    def __str__(self):
        """
        This method returns a string containing the name of the FaceGroup. This
        method is called when the object is passed to the 'print' function.

        **Parameters:** This method has no parameters.

        """
        return "facegroup %s" % self.name

    def setColor(self, rgba):
        self.color = np.asarray(rgba, dtype=np.uint8)

    def getObject(self):
        if self.__object:
            return self.__object()
        else:
            return None

    def setObject(self, value):
        if value is None:
            self.__object = None
        else:
            # Ensure link to gui3d.object is weak to avoid circular references (which break garbage collection)
            self.__object = weakref.ref(value)

    object = property(getObject, setObject)

    @property
    def parent(self):
        return self.object


class Object3D(object):
    def __init__(self, objName, vertsPerPrimitive=4):
        self.clear()

        self.name = objName
        self.vertsPerPrimitive = vertsPerPrimitive
        self._faceGroups = []
        self._groups_rev = {}
        self.cameraMode = 0
        self._visibility = True
        self.pickable = False
        self.calculateTangents = True  # TODO disable when not needed by shader
        self.object3d = None
        self._priority = 0
        self.MAX_FACES = 8

        # Cache used for retrieving vertex colors multiplied with material diffuse color
        self._old_diff = None
        self._r_color_diff = None

        self.__object = None

    def clone(
        self, scale=1.0, filterMaskedVerts=False
    ):  # TODO it's also possible to add offset to the parameters
        """
        Create a clone of this mesh, with adapted scale.
        If filterVerts is True, all vertices that are not required (do not
        belong to any visible face) are removed and vertex mapping is added to
        cloned object (see filterMaskedVerts()). For a face mapping, the
        facemask of the original mesh can be used.

        By default cloning a mesh links it to the object of its parent, remove
        the link by setting other.object to None before attaching it to a new
        object.
        """
        if self.getFaceCount(excludeMaskedFaces=filterMaskedVerts) == 0:
            raise RuntimeError(
                "Error cloning mesh %s. Cannot clone a mesh with 0 (unmasked) faces!", self.name
            )

        other = type(self)(self.name, self.vertsPerPrimitive)

        for prop in [
            "cameraMode",
            "visibility",
            "pickable",
            "calculateTangents",
            "priority",
            "MAX_FACES",
            "object",
        ]:
            setattr(other, prop, getattr(self, prop))

        for fg in self.faceGroups:
            ofg = other.createFaceGroup(fg.name)
            if fg.color is not None:
                ofg.color = fg.color.copy()
            else:
                ofg.color = fg.color

        if filterMaskedVerts:
            self.filterMaskedVerts(other, update=False)
            if scale != 1:
                other.coord = scale * other.coord
        else:
            other.setCoords(scale * self.coord)
            other.setColor(self.color.copy())
            other.setUVs(self.texco.copy())
            other.setFaces(self.fvert.copy(), self.fuvs.copy(), self.group.copy())
            other.changeFaceMask(self.face_mask.copy())

        other.calcNormals()
        other.updateIndexBuffer()

        return other

    def transformed(self, transform_mat, filterMaskedVerts=False):
        """Create a clone of this mesh with its coordinates transformed
        with the specified transformation matrix. filterMaskedVerts works the
        same is for clone()
        """
        if transform_mat.shape == (4, 4):
            translation = transform_mat[:3, 3]
            transform_mat = transform_mat[:3, :3]
        else:
            translation = np.zeros(3, dtype=np.float32)

        result = self.clone(filterMaskedVerts=filterMaskedVerts)

        coords = np.dot(transform_mat, result.getCoords().T).T
        coords += translation
        result.changeCoords(coords)
        result.calcNormals()
        result.update()
        return result

    @property
    def parent_map(self):
        """
        Maps vertex indices from this mesh to its original parent mesh
        (self.parent). This happens recursively to the topmost parent.

        Forward vertex mapping (self -> self.parent):
        parent_map[idx] = mIdx: self.coord[idx] -> self.parent.coord[mIdx]

        Will return a (n, self.MAX_FACES) array if this chain of meshes
        contains a subdivided mesh.

        Note that vertex maps only support one subdivision in a chain of mesh
        to parent meshes.
        """
        if not hasattr(self, "parent") or not self.parent:
            return None

        if not hasattr(self, "_parent_map"):
            mapping = np.arange(self.getVertexCount(), dtype=np.uint32)
        else:
            mapping = self._parent_map

        # Traverse to parent
        pmap = self.parent.parent_map
        if pmap is None:
            return mapping
        else:
            # Combine mappings to topmost parent
            return pmap[mapping]

    @property
    def parent_map_weights(self):
        return np.ones(self.getVertexCount(), dtype=np.float32)

    @property
    def inverse_parent_map(self):
        """
        Maps vertex indices from original parent mesh (self.parent) to
        this mesh (-1 if vertex is removed in this mesh).
        This happens recursively to the topmost parent.

        Reverse vertex mapping:
        inverse_parent_map[idx] = mIdx: self.parent.coord[idx] -> self.coord[mIdx]

        Will return a (n, 1+self.MAX_FACES*2) array if this chain of meshes
        contains a subdivided mesh.

        Note that vertex maps only support one subdivision in a chain of mesh
        to parent meshes.
        """
        # TODO will require nxn matrix if subdivided (catmull-clark module)

        if not hasattr(self, "parent") or not self.parent:
            return None

        if not hasattr(self, "_inverse_parent_map"):
            mapping = np.arange(self.parent.getVertexCount(), dtype=np.int32)
        else:
            mapping = self._inverse_parent_map

        # Traverse to parent
        pmap = self.parent.inverse_parent_map
        if pmap is None:
            return mapping
        else:
            # Combine mappings to topmost parent
            if len(mapping.shape) > 1:
                shape = (len(pmap), mapping.shape[1])
            else:
                shape = len(pmap)
            result = -np.ones(shape, dtype=np.int32)
            idx = np.where(pmap > -1)
            result[idx] = mapping[pmap[idx]]
            return result

    def filterMaskedVerts(self, other, update=True):
        """
        Set the vertices, faces and vertex attributes of other object to the
        vertices and faces of this mesh object, with the hidden faces and
        vertices filtered out.

        The other mesh contains a parent_map which maps vertex indices from
        the other to its original mesh and inverse_parent_map which maps vertex
        indexes from original to other (-1 if removed).

        other.parent is set to the original mesh.
        """
        # TODO or build a chain of parents?
        if hasattr(self, "parent") and self.parent:
            other.parent = self.parent
        else:
            other.parent = self

        # Forward vertex mapping:
        # _parent_map[idx] = mIdx: other.coord[idx] -> self.coord[mIdx]
        other._parent_map = np.unique(self.getVerticesForFaceMask(self.face_mask))

        # Reverse vertex mapping:
        # _inverse_parent_map[idx] = mIdx: self.coord[idx] -> other.coord[mIdx]
        other._inverse_parent_map = -np.ones(self.getVertexCount(), dtype=np.int32)
        other._inverse_parent_map[other.parent_map] = np.arange(
            len(other._parent_map), dtype=np.int32
        )
        # other._inverse_parent_map = np.ma.masked_less(other._inverse_parent_map, 0)  # TODO might be useful

        other.setCoords(self.coord[other.parent_map])
        other.setColor(self.color[other.parent_map])

        # Filter out and remap masked faces
        fvert = self.fvert[self.face_mask]
        for i in range(self.vertsPerPrimitive):
            fvert[:, i] = other._inverse_parent_map[fvert[:, i]]

        # Filter out and remap unused UVs
        fuvs = self.fuvs[self.face_mask]
        uv_idx = np.unique(fuvs.reshape(-1))
        inverse_uv_idx = -np.ones(self.texco.shape[0], dtype=np.int32)
        inverse_uv_idx[uv_idx] = np.arange(uv_idx.shape[0], dtype=np.int32)
        for i in range(self.vertsPerPrimitive):
            fuvs[:, i] = inverse_uv_idx[fuvs[:, i]]

        other.setUVs(self.texco[uv_idx])

        other.setFaces(fvert, fuvs, self.group[self.face_mask])

        if update:
            other.calcNormals()
            other.updateIndexBuffer()

    def getCenter(self):
        """
        Get center position of mesh using center of its bounding box.
        """
        bBox = self.calcBBox()
        bmin = np.asarray(bBox[0], dtype=np.float32)
        bmax = np.asarray(bBox[1], dtype=np.float32)
        return -(bmin + ((bmax - bmin) / 2))

    def calcFaceNormals(self, ix=None):
        """
        Calculate the face normals. A right-handed coordinate system is assumed,
        which requires mesh faces to be defined with counter clock-wise vertex order.
        Face normals are not normalized.

        Faces are treated as if they were triangles (using only the 3 first verts),
        so for quads with non co-planar points, inaccuracies may occur (even though
        those have a high change of being corrected by neighbouring faces).
        """
        # We assume counter clock-wise winding order
        # (if your normals are inversed, you're using clock-wise winding order)
        if ix is None:
            ix = np.s_[:]
        fvert = self.coord[self.fvert[ix]]
        v1 = fvert[:, 0, :]
        v2 = fvert[:, 1, :]
        v3 = fvert[:, 2, :]
        va = v1 - v2
        vb = v2 - v3
        self.fnorm[ix] = np.cross(va, vb)

    def calcVertexNormals(self, ix=None):
        """
        Calculate per-vertex normals from the face normals for smooth shading
        the model. Requires face normals to be calculated first.
        """
        self.markCoords(ix, norm=True)
        if ix is None:
            ix = np.s_[:]

        vface = self.vface[ix]
        norms = self.fnorm[vface]
        norms *= np.arange(self.MAX_FACES)[None, :, None] < self.nfaces[ix][:, None, None]
        norms = np.sum(norms, axis=1)
        norms /= np.sqrt(np.sum(norms**2, axis=-1))[:, None]
        self.vnorm[ix] = norms

    def calcVertexTangents(self, ix=None):
        """
        Calculate vertex tangents using Lengyel’s Method.
        """
        if not self.has_uv:
            return
        self.markCoords(ix, norm=True)
        if ix is None:
            ix = np.s_[:]
            xLen = self.getVertexCount()
            f_ix = np.s_[:]
        else:
            xLen = len(ix)
            f_ix = np.unique(self.vface[ix])

        # This implementation is based on
        # http://www.terathon.com/code/tangent.html

        tan = np.zeros((xLen, 2, 3), dtype=np.float32)

        fvert = self.coord[self.fvert[f_ix]]
        v1 = fvert[:, 0, :]
        v2 = fvert[:, 1, :]
        v3 = fvert[:, 2, :]

        x1 = v2[:, 0] - v1[:, 0]
        x2 = v3[:, 0] - v1[:, 0]
        y1 = v2[:, 1] - v1[:, 1]
        y2 = v3[:, 1] - v1[:, 1]
        z1 = v2[:, 2] - v1[:, 2]
        z2 = v3[:, 2] - v1[:, 2]

        fuv = self.texco[self.fuvs[f_ix]]
        w1 = fuv[:, 0, :]
        w2 = fuv[:, 1, :]
        w3 = fuv[:, 2, :]

        s1 = w2[:, 0] - w1[:, 0]
        s2 = w3[:, 0] - w1[:, 0]
        t1 = w2[:, 1] - w1[:, 1]
        t2 = w3[:, 1] = w1[:, 1]

        # Prevent NANs because of borked up UV coordinates  # TODO perhaps remove this
        s1[np.argwhere(np.equal(s1, 0.0))] = 0.0000001
        s2[np.argwhere(np.equal(s2, 0.0))] = 0.0000001
        t1[np.argwhere(np.equal(t1, 0.0))] = 0.0000001
        t2[np.argwhere(np.equal(t2, 0.0))] = 0.0000001

        r = np.repeat(1.0, len(s1)) / ((s1 * t2) - (s2 * t1))
        sdir = np.zeros((self.getFaceCount(), 3), dtype=np.float32)
        tdir = np.zeros((self.getFaceCount(), 3), dtype=np.float32)
        sdir[f_ix] = np.column_stack(
            [((t2 * x1) - (t1 * x2)) * r, ((t2 * y1) - (t1 * y2)) * r, ((t2 * z1) - (t1 * z2)) * r]
        )
        tdir[f_ix] = np.column_stack(
            [((s1 * x2) - (s2 * x1)) * r, ((s1 * y2) - (s2 * y1)) * r, ((s1 * z2) - (s2 * z1)) * r]
        )

        tan[:, 0] = np.sum(sdir[self.vface[ix]])
        tan[:, 1] = np.sum(tdir[self.vface[ix]])

        # Gramm-Schmidt orthogonalize
        dotP = dot_v3(self.vnorm[ix], tan[:, 0])
        # Duplicate dot product value in 3 columns because scalar * (n x 3)
        # does not work
        dotP = np.tile(dotP, (3, 1)).transpose().reshape(len(tan), 3)
        self.vtang[ix, :3] = tan[:, 0] - dotP * self.vnorm[ix]
        # Normalize
        self.vtang[ix, :3] /= np.sqrt(np.sum(self.vtang[ix, :3] ** 2, axis=-1))[:, None]

        # Determine Handedness as w parameter
        self.vtang[ix, 3] = 1.0
        indx = np.argwhere(np.less(dot_v3(np.cross(self.vnorm[ix], tan[:, 0]), tan[:, 1]), 0.0))
        self.vtang[ix, 3][indx] = -1.0

    def getObject(self):
        if self.__object:
            return self.__object()
        else:
            return None

    def setObject(self, value):
        if value is None:
            self.__object = None
        else:
            # Ensure link to gui3d.object is weak to avoid circular references (which break garbage collection)
            self.__object = weakref.ref(value)

    object = property(getObject, setObject)

    @property
    def faceGroups(self):
        return iter(self._faceGroups)

    @property
    def faceGroupCount(self):
        return len(self._faceGroups)

    def clear(self):
        """
        Clears both local and remote data to repurpose this object
        """

        # Clear remote data
        self._faceGroups = []

        self._transparentPrimitives = 0

        # Reference to vertex attributes (coordinate, normal, color, tang) that form the faces (idx = face idx)
        self.fvert = []
        # Stores the face normal of the faces (idx = face idx)
        self.fnorm = []
        # References to UVs at the verts of the faces (idx = face idx)
        # (NOTE: UVs are not tied to vertex IDs, and are not necessarily
        # uniform per vertex, like the other attributes!)
        self.fuvs = []
        self.group = []  # Determines facegroup per face (idx = face idx)
        self.face_mask = []  # Determines visibility per face (idx = face idx)

        self.texco = []  # UV coordinates (idx = uv idx)

        self.coord = []  # Vertex coordinates (positions) (idx = vertex idx)
        self.vnorm = []  # Vertex normals (idx = vertex idx)
        self.vtang = []  # Vertex tangents (idx = vertex idx)
        self.color = []  # Vertex colors (idx = vertex idx)
        self.vface = (
            []
        )  # References the faces that a vertex belongs to (limited to MAX_FACES) (idx = vertex idx)
        self.nfaces = 0  # Polycount

        self.ucoor = False  # Update flags for updating to OpenGL renderbuffers
        self.unorm = False
        self.utang = False
        self.ucolr = False
        self.utexc = False

        self.has_uv = False

        if hasattr(self, "index"):
            del self.index
        if hasattr(self, "grpix"):
            del self.grpix

        # Maps unwelded vertices back to original welded ones (idx = unwelded vertex idx)
        self.vmap = None
        # Maps unwelded vertex texture (UV) coordinates back to original ones (idx = unwelded vertex idx)
        self.tmap = None

        # Cached inverse of vmap: maps original welded vert idx (coord) to one or multiple unwelded
        # vert idxs (r_coord)
        self._inverse_vmap = None

        # Unwelded vertex buffers used by OpenGL
        if hasattr(self, "r_coord"):
            del self.r_coord
        if hasattr(self, "r_texco"):
            del self.r_texco
        if hasattr(self, "r_vnorm"):
            del self.r_vnorm
        if hasattr(self, "r_vtang"):
            del self.r_vtang
        if hasattr(self, "r_color"):
            del self.r_color
        if hasattr(self, "r_faces"):
            del self.r_faces

    def setCoords(self, coords):
        nverts = len(coords)
        self.coord = np.asarray(coords, dtype=np.float32)
        self.vnorm = np.zeros((nverts, 3), dtype=np.float32)
        self.vtang = np.zeros((nverts, 4), dtype=np.float32)
        self.color = np.zeros((nverts, 4), dtype=np.uint8) + 255
        self.vface = np.zeros((nverts, self.MAX_FACES), dtype=np.uint32)
        self.nfaces = np.zeros(nverts, dtype=np.uint8)

        self.orig_coord = self.coord.copy()  # Keep a copy of the original coordinates

        self.ucoor = True
        self.unorm = True
        self.utang = True
        self.ucolr = True

        self.markCoords(None, True, True, True)

    def getVertexCount(self, excludeMaskedVerts=False):
        # return len(self.vface)
        if excludeMaskedVerts:
            return np.count_nonzero(self.getVertexMaskForFaceMask(self.getFaceMask()))
        return len(self.coord)

    def getCoords(self, indices=None):
        if indices is None:
            indices = np.s_[...]
        return self.coord[indices]

    def getNormals(self, indices=None):
        if indices is None:
            indices = np.s_[...]
        return self.vnorm[indices]

    def markCoords(self, indices=None, coor=False, norm=False, colr=False):
        if isinstance(indices, tuple):
            indices = indices[0]

        nverts = len(self.coord)

        if coor:
            if indices is None:
                self.ucoor = True
            else:
                if self.ucoor is False:
                    self.ucoor = np.zeros(nverts, dtype=bool)
                if self.ucoor is not True:
                    self.ucoor[indices] = True

        if norm:
            if indices is None:
                self.unorm = self.utang = True
            else:
                if self.unorm is False:
                    self.unorm = self.utang = np.zeros(nverts, dtype=bool)
                if self.unorm is not True:
                    self.unorm[indices] = True
                    self.utang[indices] = True

        if colr:
            if indices is None:
                self.ucolr = True
            else:
                if self.ucolr is False:
                    self.ucolr = np.zeros(nverts, dtype=bool)
                if self.ucolr is not True:
                    self.ucolr[indices] = True

    def changeCoords(self, coords, indices=None):
        self.markCoords(indices, coor=True)

        if indices is None:
            indices = np.s_[...]
        self.coord[indices] = coords

    def setUVs(self, uvs):
        self.texco = np.asarray(uvs, dtype=np.float32)
        self.utexc = True

    def getUVCount(self):
        return len(self.texco)

    def getUVs(self, indices=None):
        if indices is None:
            indices = np.s_[...]
        return self.texco[indices]

    def markUVs(self, indices=None):
        if isinstance(indices, tuple):
            indices = indices[0]

        ntexco = len(self.texco)

        if indices is None:
            self.utexc = True
        else:
            if self.utexc is False:
                self.utexc = np.zeros(ntexco, dtype=bool)
            if self.utexc is not True:
                self.utexc[indices] = True

    def setFaces(self, verts, uvs=None, groups=None, skipUpdate=False):
        nfaces = len(verts)

        self.fvert = np.empty((nfaces, self.vertsPerPrimitive), dtype=np.uint32)
        self.fnorm = np.zeros((nfaces, 3), dtype=np.float32)
        self.fuvs = np.zeros(self.fvert.shape, dtype=np.uint32)
        self.group = np.zeros(nfaces, dtype=np.uint16)
        self.face_mask = np.ones(nfaces, dtype=bool)

        # preset number of vertices for all export formats to 4
        self.vertsPerFaceForExport = 4
        if nfaces != 0:

            # evaluate if we have 3 or 4 vertices per face, do it here so it works for npz or obj file
            if len(verts[0]) == 4 and verts[0][0] == verts[0][3]:
                self.vertsPerFaceForExport = 3

            self.fvert[...] = verts
            if uvs is not None:
                self.fuvs[...] = uvs
            if groups is not None:
                self.group[...] = groups

        self.has_uv = uvs is not None

        if not skipUpdate:
            self._update_faces(resize=True)

    def changeFaceMask(self, mask, indices=None):
        if indices is None:
            indices = np.s_[...]
        self.face_mask[indices] = mask

    def getFaceMask(self, indices=None):
        if indices is None:
            indices = np.s_[...]
        return self.face_mask[indices]

    def hasUVs(self):
        return self.has_uv

    def getFaceCount(self, excludeMaskedFaces=False):
        if excludeMaskedFaces:
            return np.count_nonzero(self.getFaceMask())
        return len(self.fvert)

    def getFaceVerts(self, indices=None):
        if indices is None:
            indices = np.s_[...]
        return self.fvert[indices]

    def getFaceUVs(self, indices=None):
        if indices is None:
            indices = np.s_[...]
        return self.fuvs[indices]

    @property
    def inverse_vmap(self):
        """
        The inverse of vmap: a mapping of original welded (relating to UVs)
        vertex (coord indices) to a set of unwelded vertices that represent the
        same coordinate (r_coord indices).
        """
        if self._inverse_vmap is None:
            # TODO this loop is quite slow and could benefit from numpy optimization
            originalToUnweldedMap = {}
            for unweldedIdx, originalIdx in enumerate(self.vmap):
                if originalIdx not in originalToUnweldedMap:
                    originalToUnweldedMap[originalIdx] = []
                originalToUnweldedMap[originalIdx].append(unweldedIdx)
            self._inverse_vmap = originalToUnweldedMap
        return self._inverse_vmap

    def _update_faces(self, resize=False):
        #
        # this procedure is only called, when geometry is not taken from npz-file
        #
        # create an 2dim array self.vface for each vertex
        # which contains an array of faces, where the vertex belongs to, maximum of
        # the array inside is self.MAX_FACES
        #
        # use self.nfaces as counter for the inner array, it is needed afterwards

        nverts = len(self.coord)
        self.nfaces = np.zeros(nverts, dtype=np.uint8)  # reset the counters to zero

        for idx, vert in enumerate(self.fvert):  # contains faces with vertex number
            for i in range(
                0, min(len(vert), self.vertsPerFaceForExport)
            ):  # use minimum of attached vertices, works for less than 3 also
                vn = vert[i]  # vertex.number
                if self.nfaces[vn] >= self.MAX_FACES:
                    log.error(
                        "Failed to index faces of mesh %s, you are probably loading a mesh "
                        "with mixed nb of verts per face (do not mix tris and quads). "
                        "Or your mesh has too many faces attached to one vertex "
                        "(the maximum is %s-poles). In the second case, either increase "
                        "MAX_FACES for this mesh, or improve the mesh topology.",
                        self.name,
                        self.MAX_FACES,
                    )
                    raise RuntimeError("Incompatible mesh topology.")
                    return
                self.vface[vn, self.nfaces[vn]] = (
                    idx  # add the index of the face to the array, row given by vn, column by counter nfaces[vn]
                )
                self.nfaces[vn] += 1  # now increment the counter

        imax = i + 1  # just needed later not to be recalculated.

        # in case this function is not called from catmull-clark function resize the self.vface to a minimum
        if resize is True:
            # now lets recalculate the maximum number of faces belonging to a plane
            newmax = np.max(self.nfaces)
            if (
                newmax > 2
            ):  # Avoid the information when function calculates internal objects not to confuse users
                log.debug("Calculated maximum number of faces for one vertex: %d", newmax)

            # unfortunately catmull-clark expects maxpoles and not maxfaces, so we need
            # also to calculate max-poles
            #
            # for each vertex we have to check all attached faces, these faces are already collected in self.vface
            # when we check all the faces we will collect the neighbors using a modulo function, the vertices are
            # entered clockwise ... modulo allows us to use neighbor of 1 and 3 when index is 0
            # if not already existent, the neighbors are added to an array and the maximum size of this array will
            # be maxpole

            maxpole = 0
            for vn, row in enumerate(self.vface):
                noticed = np.zeros(self.MAX_FACES * 4, dtype=int)
                m = 0
                for j in range(0, self.nfaces[vn]):
                    # collect neighbors
                    face = row[j]
                    for ix, v2 in enumerate(self.fvert[face]):
                        if v2 == vn:
                            ln = self.fvert[face][(ix - 1) % imax]
                            rn = self.fvert[face][(ix + 1) % imax]
                            if ln not in noticed:
                                noticed[m] = ln
                                m += 1
                            if rn not in noticed:
                                noticed[m] = rn
                                m += 1
                            break
                if m > maxpole:
                    maxpole = m

            if (
                maxpole > 2
            ):  # Avoid the information when function calculates internal objects not to confuse users
                log.debug("Calculated maximum number of poles for one vertex: %d", maxpole)

            if newmax < maxpole:
                newmax = maxpole

            # keep in mind that a maximum number of neighboring faces lower than 4 will crash subdiv
            # (try to load a cube)
            if newmax < 4:
                newmax = 4

            if newmax != self.MAX_FACES:
                # it is different so resize vface
                self.vface = np.delete(self.vface, np.s_[newmax::], 1)
                self.MAX_FACES = newmax

    def getVertexWeights(self, parentWeights):
        """
        Map armature weights mapped to the root parent (original mesh) to this
        child mesh. Returns parentWeights unaltered if this mesh has no parent.
        If this is a proxy mesh, parentWeights should be the weights mapped
        through the proxy.getVertexWeights() method first.

        This particular vertex weights mapping is only used for exporting rigged
        meshes, as in MakeHuman only unsubdivided meshes are posed, and then
        smoothed in their posed state if required. Vertices are not removed
        within MH when faces are hidden, either.
        """
        if not hasattr(self, "parent") or not self.parent:
            return parentWeights

        vmap = self.inverse_parent_map
        vwmap = self.parent_map_weights

        from collections import OrderedDict

        weights = OrderedDict()
        for bname, (verts, wghts) in list(parentWeights.data.items()):
            vgroup = []
            empty = True
            for v, wt in zip(verts, wghts):
                mvs = vmap[v]
                if isinstance(mvs, (int, np.int32)):
                    mvs = [mvs]
                for mv in mvs:
                    w = vwmap[mv]
                    if mv > -1:
                        vgroup.append((mv, w * wt))
                        empty = False
            if not empty:
                weights[bname] = vgroup

        return parentWeights.create(weights, self.getVertexCount())

    def updateIndexBuffer(self):
        self.updateIndexBufferVerts()
        self.updateIndexBufferFaces()

    def updateIndexBufferVerts(self):
        packed = self.fvert.astype(np.uint64) << 32
        packed |= self.fuvs
        packed = packed.reshape(-1)

        u, rev = np.unique(packed, return_inverse=True)
        del packed

        unwelded = u[:, None] >> np.array([[32, 0]], dtype=np.uint64)
        unwelded = unwelded.astype(np.uint32)
        nverts = len(unwelded)
        iverts = rev.reshape(self.fvert.shape)
        del rev, u

        self.vmap = unwelded[:, 0]
        self.tmap = unwelded[:, 1]
        self._inverse_vmap = None
        del unwelded

        self.r_coord = np.empty((nverts, 3), dtype=np.float32)
        self.r_texco = np.empty((nverts, 2), dtype=np.float32)
        self.r_vnorm = np.zeros((nverts, 3), dtype=np.float32)
        self.r_vtang = np.zeros((nverts, 4), dtype=np.float32)
        self.r_color = np.zeros((nverts, 4), dtype=np.uint8) + 255

        self.r_faces = np.array(iverts, dtype=np.uint32)

    def updateIndexBufferFaces(self):
        index = self.r_faces[self.face_mask]
        group = self.group[self.face_mask]

        if len(group) > 0:
            order = np.argsort(group)
            group = group[order]
            index = index[order]

            group, start = np.unique(group, return_index=True)
            count = np.empty(len(start), dtype=np.uint32)
            count[:-1] = start[1:] - start[:-1]
            count[-1] = len(index) - start[-1]

            grpix = np.zeros((max(self.group) + 1, 2), dtype=np.uint32)
            grpix[group, 0] = start
            grpix[group, 1] = count
        else:
            grpix = np.zeros((0, 2), dtype=np.uint32)

        self.index = index
        self.grpix = grpix

        self.ucoor = True
        self.unorm = True
        self.utang = True
        self.ucolr = True
        self.utexc = True
        self.sync_all()

    def sync_coord(self):
        if self.ucoor is False:
            return
        if self.vmap is None or len(self.vmap) == 0:
            return
        if self.ucoor is True:
            self.r_coord[...] = self.coord[self.vmap]
        else:
            self.r_coord[self.ucoor[self.vmap]] = self.coord[self.vmap][self.ucoor[self.vmap]]
        self.ucoor = False

    def sync_norms(self):
        if self.unorm is False:
            return
        if self.vmap is None or len(self.vmap) == 0:
            return
        if self.unorm is True:
            self.r_vnorm[...] = self.vnorm[self.vmap]
        else:
            self.r_vnorm[self.unorm[self.vmap]] = self.vnorm[self.vmap][self.unorm[self.vmap]]
        self.unorm = False

    def sync_tangents(self):
        if not self.has_uv:
            return
        if self.utang is False:
            return
        if self.vmap is None or len(self.vmap) == 0:
            return
        if self.utang is True:
            self.r_vtang[...] = self.vtang[self.vmap]
        else:
            self.r_vtang[self.utang[self.vmap]] = self.vtang[self.vmap][self.utang[self.vmap]]
        self.utang = False

    def sync_color(self):
        if self.ucolr is False:
            return
        if self.vmap is None or len(self.vmap) == 0:
            return
        if self.ucolr is True:
            self.r_color[...] = self.color[self.vmap]
        else:
            self.r_color[self.ucolr[self.vmap]] = self.color[self.vmap][self.ucolr[self.vmap]]
        self.ucolr = False
        self._r_color_diff = None

    def sync_texco(self):
        if not self.has_uv:
            return
        if self.utexc is False:
            return
        if self.tmap is None or len(self.tmap) == 0:
            return
        if self.utexc is True:
            self.r_texco[...] = self.texco[self.tmap]
        else:
            self.r_texco[self.utexc[self.tmap]] = self.texco[self.tmap][self.utexc[self.tmap]]
        self.utexc = False

    def sync_all(self):
        self.sync_coord()
        self.sync_norms()
        if self.calculateTangents:
            self.sync_tangents()
        self.sync_color()
        self.sync_texco()

    def createFaceGroup(self, name):
        """
        Creates a new module3d.FaceGroup with the given name.

        :param name: The name for the face group.
        :type name: [float, float, float]
        :return: The new face group.
        :rtype: :py:class:`module3d.FaceGroup`
        """
        idx = len(self._faceGroups)
        fg = FaceGroup(self, name, idx)
        self._groups_rev[name] = fg
        self._faceGroups.append(fg)
        return fg

    def setColor(self, color):
        """
        Sets the vertex colors for the entire object.

        :param color: The color in rgba.
        :type color: [byte, byte, byte, byte]
        """
        color = np.asarray(color, dtype=np.uint8)
        if len(color.shape) == 1:  # One color for all vertices
            if len(color) == 3:
                # Add alpha component to simple RGB color
                color = list(color) + [255]
            self.color[:] = color
        else:
            self.color[...] = color[None, :]
        self.markCoords(colr=True)
        self.sync_color()

    @property
    def r_color_diff(self):
        """
        Vertex colors multiplied with the diffuse material color.
        """
        if self._r_color_diff is None:
            self._r_color_diff = np.zeros(self.r_color.shape, dtype=np.uint8)
            self._old_diff = None

        diff = self.material.diffuseColor.values + [self.material.opacity]
        if diff != self._old_diff:
            # Update diffuse * vertex colors
            self._r_color_diff[:] = diff * self.r_color
            self._old_diff = diff
        return self._r_color_diff

    @property
    def visibility(self):
        return self._visibility

    def setVisibility(self, visible):
        self.visibility = visible

    @visibility.setter
    def visibility(self, visible):
        """
        This method sets the visibility of the object.

        :param visible: Whether or not the object is visible.
        :type visible: Boolean
        """
        self._visibility = visible

    def setPickable(self, pickable):
        """
        This method sets the pickable flag of the object.

        :param pickable: Whether or not the object is pickable.
        :type pickable: Boolean
        """

        self.pickable = pickable

    def getPriority(self):
        """
        The rendering priority of this object.
        Objects with higher priorities will be drawn later.

        Common priorities used:
        file                                 description       2D/3D priority

        core/mhmain.py                       background-gradient 2D   -200
        0_modeling_background.py             background          2D    -90
        core/mhmain.py                       human               3D      0
        3_libraries_clothes_chooser.py       clothing            3D     10
        3_libraries_polygon_hair_chooser.py  hair                3D     20
        2_posing_armature.py                 armature            3D     30
        0_modeling_a_measurement.py          measureMesh         2D     50
        5_settings_censor.py                 censor rectangles   2D     80
        apps/gui/guifiles.py                 black frame         2D     90
        """
        return self._priority

    def setPriority(self, priority):
        """
        Set the rendering priority of this object.
        Objects with higher priorities will be drawn later.

        Common priorities used:
        file                                 description       2D/3D priority

        0_modeling_background.py             background          2D    -90
        core/mhmain.py                       human               3D      0
        3_libraries_clothes_chooser.py       clothing            3D     10
        3_libraries_polygon_hair_chooser.py  hair                3D     20
        2_posing_armature.py                 armature            3D     30
        0_modeling_a_measurement.py          measureMesh         2D     50
        5_settings_censor.py                 censor rectangles   2D     80
        apps/gui/guifiles.py                 black frame         2D     90
        """
        self._priority = priority

    priority = property(getPriority, setPriority)

    @property
    def material(self):
        if not self.object and hasattr(self, "parent"):
            return self.parent.material
        return self.object.material

    def setTransparentPrimitives(self, transparentPrimitives):
        """
        This method is used to specify the amount of transparent faces.
        This property is overridden if self.material.transparent is set to True.

        :param transparentPrimitives: The amount of transparent faces.
        :type transparentPrimitives: int
        """
        self._transparentPrimitives = transparentPrimitives

    def getTransparentPrimitives(self):
        # Object allows transparency rendering of only a subset (starting from
        # the first face) of faces of the mesh, but material property transparent
        # set to True overrides this and makes all faces render with transparency technique
        if self.material.transparent:
            return len(self.fvert)
        else:
            return self._transparentPrimitives

    transparentPrimitives = property(getTransparentPrimitives, setTransparentPrimitives)

    def getFaceGroup(self, name):
        """
        This method searches the list of FaceGroups held for this object, and
        returns the FaceGroup with the specified name. If no FaceGroup is found
        for that name, this method returns None.

        :param name: The name of the FaceGroup to retrieve.
        :type name: str
        :return: The FaceGroup if found, None otherwise.
        :rtype: :py:class:`module3d.FaceGroup`
        """
        return self._groups_rev.get(name, None)

    def getFaceGroups(self):
        """
        The names of the facegroups available on this mesh.
        """
        return list(self._groups_rev.keys())

    def getGroupMaskForGroups(self, groupNames):
        groups = np.zeros(len(self._faceGroups), dtype=bool)
        for name in groupNames:
            groups[self._groups_rev[name].idx] = True
        return groups

    def getFaceMaskForGroups(self, groupNames):
        groups = self.getGroupMaskForGroups(groupNames)
        face_mask = groups[self.group]
        return face_mask

    def getFacesForGroups(self, groupNames):
        face_mask = self.getFaceMaskForGroups(groupNames)
        faces = np.argwhere(face_mask)[..., 0]
        return faces

    def getVertexMaskForGroups(self, groupNames):
        face_mask = self.getFaceMaskForGroups(groupNames)
        verts = self.fvert[face_mask]
        vert_mask = np.zeros(len(self.coord), dtype=bool)
        vert_mask[verts] = True
        return vert_mask

    def getVerticesForGroups(self, groupNames):
        vert_mask = self.getVertexMaskForGroups(groupNames)
        verts = np.argwhere(vert_mask)[..., 0]
        return verts

    def getVerticesForFaceMask(self, face_mask):
        verts = self.fvert[face_mask]
        verts = verts.reshape(-1)
        return verts

    def getVertexMaskForFaceMask(self, face_mask):
        verts = self.getVerticesForFaceMask(face_mask)
        vert_mask = np.zeros(len(self.coord), dtype=bool)
        vert_mask[verts] = True
        return vert_mask

    def getVertexAndFaceMasksForGroups(self, groupNames):
        face_mask = self.getFaceMaskForGroups(groupNames)
        verts = self.fvert[face_mask]
        vert_mask = np.zeros(len(self.coord), dtype=bool)
        vert_mask[verts] = True
        return vert_mask, face_mask

    def getFaceMaskForVertices(self, verts):
        """
        Get mask that selects all faces that are connected to the specified
        vertices.
        """
        mask = np.zeros(len(self.fvert), dtype=bool)
        valid = (
            np.arange(self.MAX_FACES)[None, :] < self.nfaces[verts][:, None]
        )  # Mask that filters out unused slots for faces connected to a vert
        vface = self.vface[verts]
        faces = vface[valid]
        mask[faces] = True
        return mask

    def getFacesForVertices(self, verts):
        return np.argwhere(self.getFaceMaskForVertices(verts))[..., 0]

    def setCameraProjection(self, cameraMode):
        """
        This method sets the camera mode used to visualize this object (fixed or movable).
        The 3D engine has two camera modes (both perspective modes).
        The first is moved by the mouse, while the second is fixed.
        The first is generally used to model 3D objects (a human, clothes,
        etc.), while the second is used for 3D GUI controls.

        :param cameraMode: The camera mode. 0 = movable camera (modelCamera);
        1 = static camera (guiCamera).
        :type cameraMode: int
        """

        self.cameraMode = cameraMode

    def getCamera(self):
        """
        The camera with which this mesh is rendered.
        """
        from core import G

        return G.cameras[self.cameraMode]

    def update(self):
        """
        This method is used to call the update methods on each of a list of vertices
        or all vertices that form part of this object. Passes the current mesh state
        to the OpenGL buffers.
        """
        self.sync_all()

    def calcNormals(
        self, recalcVertexNormals=1, recalcFaceNormals=1, verticesToUpdate=None, facesToUpdate=None
    ):
        """
        Updates the given vertex and/or face normals.

        :param recalcVertexNormals: A flag to indicate whether or not the vertex normals should be recalculated.
        :type recalcVertexNormals: Boolean
        :param recalcFaceNormals: A flag to indicate whether or not the face normals should be recalculated.
        :type recalcFaceNormals: Boolean
        :param verticesToUpdate: The list of vertices to be updated, if None all vertices are updated.
        :type verticesToUpdate: list of :py:class:`module3d.Vert`
        :param facesToUpdate: The list of faces to be updated, if None all faces are updated.
        :type facesToUpdate: list of :py:class:`module3d.Face`
        """

        if recalcFaceNormals:
            self.calcFaceNormals(facesToUpdate)

        if recalcVertexNormals:
            self.calcVertexNormals(verticesToUpdate)

        if recalcFaceNormals or recalcVertexNormals and self.calculateTangents:
            self.calcVertexTangents(verticesToUpdate)

    def calcBBox(self, ix=None, onlyVisible=True, fixedFaceMask=None):
        """
        Calculates the axis aligned bounding box of this object in the object's coordinate system.
        """
        # TODO maybe cache bounding box
        if ix is None:
            ix = np.s_[:]
        if fixedFaceMask is not None:
            verts = self.getVerticesForFaceMask(fixedFaceMask)
            coord = self.coord[verts]
        elif onlyVisible:
            verts = self.getVerticesForFaceMask(self.getFaceMask())
            coord = self.coord[verts]
        else:
            coord = self.coord[ix]
        if len(coord) == 0:
            return np.zeros((2, 3), dtype=np.float32)
        v0 = np.amin(coord, axis=0)
        v1 = np.amax(coord, axis=0)
        return np.vstack((v0, v1))

    def __str__(self):
        return "object3D Mesh named: %s, nverts: %s, nfaces: %s" % (
            self.name,
            self.getVertexCount(),
            self.getFaceCount(),
        )


def dot_v3(v3_arr1, v3_arr2):
    """
    Numpy Ufunc'ed implementation of a series of dot products of two vector3
    objects.
    """
    return (
        (v3_arr1[:, 0] * v3_arr2[:, 0])
        + (v3_arr1[:, 1] * v3_arr2[:, 1])
        + (v3_arr2[:, 2] * v3_arr1[:, 2])
    )
