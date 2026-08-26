import numpy as np

from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import add_to_dict, CustomMaterial
import robosuite.utils.transform_utils as T


class NeedleObject(CompositeObject):
    """
    Generates a needle with a handle (used in Threading task)

    Args:
        name (str): Name of this Needle object
    """

    def __init__(
        self,
        name,
    ):

        ### TODO: make this object more general (with more args and configuration options) later ###

        # Set object attributes
        self._name = name
        self.needle_mat_name = "darkwood_mat"

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }
        needle_mat = CustomMaterial(
            texture="WoodDark",
            tex_name="darkwood",
            mat_name="darkwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(needle_mat)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": [0.02, 0.08, 0.02],
            "name": self.name,
            "locations_relative_to_center": False,
            "obj_types": "all",
            "density": 100.0,
        }
        obj_args = {}

        # make a skinny needle object with a large handle
        needle_size = [0.005, 0.06, 0.005]
        handle_size = [0.02, 0.02, 0.02]

        # Needle
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            # needle geom needs to be offset from boundary in (x, z)
            geom_locations=(
                (handle_size[0] - needle_size[0]),
                0.0,
                (handle_size[2] - needle_size[2]),
            ),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=tuple(needle_size),
            geom_names="needle",
            geom_rgbas=None,
            geom_materials=self.needle_mat_name,
            # make the needle low friction to ensure easy insertion
            geom_frictions=(0.3, 5e-3, 1e-4),
        )

        # Handle
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            # handle geom needs to be offset in y
            geom_locations=(0.0, 2.0 * needle_size[1], 0.0),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=tuple(handle_size),
            geom_names="handle",
            geom_rgbas=None,
            geom_materials=self.needle_mat_name,
            geom_frictions=None,
        )

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args
