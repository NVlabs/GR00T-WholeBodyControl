import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import CustomMaterial, add_to_dict


class Bin(CompositeObject):
    """
    Generates a four-walled bin container with an open top.
    Args:
        name (str): Name of this Bin object
        bin_size (3-array): (x,y,z) full size of bin
        wall_thickness (float): How thick to make walls of bin
        transparent_walls (bool): If True, walls will be semi-translucent
        friction (3-array or None): If specified, sets friction values for this bin. None results in default values
        density (float): Density value to use for all geoms. Defaults to 1000
        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored
        rgba (4-array or None): If specified, sets rgba values for all geoms. None results in default values
        material: If specified, use this material
        upside_down (bool): if True, construct and initialize the Bin so the bottom geom is at the top
    """

    def __init__(
        self,
        name,
        bin_size=(0.3, 0.3, 0.15),
        wall_thickness=0.01,
        transparent_walls=True,
        friction=None,
        density=1000.0,
        use_texture=True,
        rgba=(0.2, 0.1, 0.0, 1.0),
        material=None,
        upside_down=False,
        add_second_base=False,
        transparent_base=False,
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.bin_size = np.array(bin_size)
        self.wall_thickness = wall_thickness
        self.transparent_walls = transparent_walls
        self.friction = friction if friction is None else np.array(friction)
        self.density = density
        self.use_texture = use_texture
        self.rgba = rgba
        self.bin_mat_name = "dark_wood_mat"

        # if box should be constructed and initialized upside down
        self.upside_down = upside_down

        # if box should have a second base (so it will be a closed box)
        self.add_second_base = add_second_base
        if self.add_second_base:
            assert not self.upside_down

        # if base(s) should be transparent
        self.transparent_base = transparent_base

        self.has_material = material is not None
        if self.has_material:
            assert isinstance(material, CustomMaterial)
            self.material = material
            self.bin_mat_name = self.material.mat_attrib["name"]
        else:
            # default material
            tex_attrib = {
                "type": "cube",
            }
            mat_attrib = {
                "texrepeat": "3 3",
                "specular": "0.4",
                "shininess": "0.1",
            }
            bin_mat = CustomMaterial(
                texture="WoodDark",
                tex_name="dark_wood",
                mat_name=self.bin_mat_name,
                tex_attrib=tex_attrib,
                mat_attrib=mat_attrib,
            )
            self.material = bin_mat

        # Element references
        self._base_geom = "base"
        if self.add_second_base:
            self._second_base_geom = "base2"

        # Other private attributes
        self._important_sites = {}

        # Create dictionary of values to create geoms for composite object and run super init
        super().__init__(**self._get_geom_attrs())

        # Define materials we want to use for this object
        self.append_material(self.material)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor
        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": self.bin_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
            "density": self.density,
        }
        obj_args = {}

        # Base(s)
        base_geom_loc = (0, 0, -(self.bin_size[2] - self.wall_thickness) / 2)
        if self.upside_down:
            base_geom_loc = (
                base_geom_loc[0],
                base_geom_loc[1],
                -1.0 * base_geom_loc[2],
            )
        if self.transparent_base:
            base_rgba = (1.0, 1.0, 1.0, 0.3)
            base_mat = None
        else:
            base_rgba = None if self.use_texture else self.rgba
            base_mat = self.bin_mat_name if self.use_texture else None

        base_geom_names = [self._base_geom]
        base_geom_locs = [base_geom_loc]
        if self.add_second_base:
            base_geom_names.append(self._second_base_geom)
            base_geom_locs.append((base_geom_loc[0], base_geom_loc[1], -1.0 * base_geom_loc[2]))

        for base_g_name, base_g_loc in zip(base_geom_names, base_geom_locs):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=base_g_loc,
                geom_quats=(1, 0, 0, 0),
                geom_sizes=(
                    np.array((self.bin_size[0], self.bin_size[1], self.wall_thickness))
                    - np.array((self.wall_thickness, self.wall_thickness, 0))
                )
                / 2,
                geom_names=base_g_name,
                geom_rgbas=base_rgba,
                geom_materials=base_mat,
                geom_frictions=self.friction,
            )

        # Walls
        x_vals = np.array(
            [
                0,
                -(self.bin_size[0] - self.wall_thickness) / 2,
                0,
                (self.bin_size[0] - self.wall_thickness) / 2,
            ]
        )
        y_vals = np.array(
            [
                -(self.bin_size[1] - self.wall_thickness) / 2,
                0,
                (self.bin_size[1] - self.wall_thickness) / 2,
                0,
            ]
        )
        w_vals = np.array([self.bin_size[0], self.bin_size[1], self.bin_size[0], self.bin_size[1]])
        r_vals = np.array([np.pi / 2, 0, -np.pi / 2, np.pi])
        if self.transparent_walls:
            wall_rgba = (1.0, 1.0, 1.0, 0.3)
            wall_mat = None
        else:
            wall_rgba = None if self.use_texture else self.rgba
            wall_mat = self.bin_mat_name if self.use_texture else None
        for i, (x, y, w, r) in enumerate(zip(x_vals, y_vals, w_vals, r_vals)):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(x, y, 0),
                geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, r])), to="wxyz"),
                geom_sizes=(self.wall_thickness / 2, w / 2, self.bin_size[2] / 2),
                geom_names=f"wall{i}",
                geom_rgbas=wall_rgba,
                geom_materials=wall_mat,
                geom_frictions=self.friction,
            )

        # Add back in base args and site args
        obj_args.update(base_args)

        # Return this dict
        return obj_args

    @property
    def base_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to bin base
        """
        return [self.correct_naming(self._base_geom)]
