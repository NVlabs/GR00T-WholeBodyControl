import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import CustomMaterial, add_to_dict


class Lid(CompositeObject):
    """
    Generates a square lid with a simple handle.
    Args:
        name (str): Name of this Lid object
        lid_size (3-array): (length, width, thickness) of lid
        handle_size (3-array): (thickness, length, height) of handle
        transparent (bool): If True, lid will be semi-translucent
        friction (3-array or None): If specified, sets friction values for this lid. None results in default values
        density (float): Density value to use for all geoms. Defaults to 1000
        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored
        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
    """

    def __init__(
        self,
        name,
        lid_size=(0.3, 0.3, 0.01),
        handle_size=(0.02, 0.08, 0.03),
        transparent=True,
        friction=None,
        density=250.0,
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.lid_size = np.array(lid_size)
        self.handle_size = np.array(handle_size)
        self.transparent = transparent
        self.friction = friction if friction is None else np.array(friction)
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.lid_mat_name = "dark_wood_mat"

        # Element references
        self._handle_geom = "handle"

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "3 3",
            "specular": "0.4",
            "shininess": "0.1",
        }
        lid_mat = CustomMaterial(
            texture="WoodDark",
            tex_name="dark_wood",
            mat_name=self.lid_mat_name,
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(lid_mat)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor
        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        full_height = self.lid_size[2] + self.handle_size[2]
        full_size = np.array([self.lid_size[0], self.lid_size[1], full_height])
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": full_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        obj_args = {}

        # Top
        if self.transparent:
            top_rgba = (1.0, 1.0, 1.0, 0.3)
            top_mat = None
        else:
            top_rgba = None if self.use_texture else self.rgba
            top_mat = self.lid_mat_name if self.use_texture else None
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, (-full_size[2] + self.lid_size[2]) / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array((full_size[0], full_size[1], self.lid_size[2])) / 2,
            geom_names="top",
            geom_rgbas=top_rgba,
            geom_materials=top_mat,
            geom_frictions=self.friction,
            density=self.density,
        )

        # Handle
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, (full_size[2] - self.handle_size[2]) / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=self.handle_size / 2,
            geom_names=self._handle_geom,
            geom_rgbas=None if self.use_texture else self.rgba,
            geom_materials=self.lid_mat_name if self.use_texture else None,
            geom_frictions=self.friction,
            density=self.density * 2,
        )

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args

    @property
    def handle_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to lid handle
        """
        return [self.correct_naming(self._handle_geom)]
