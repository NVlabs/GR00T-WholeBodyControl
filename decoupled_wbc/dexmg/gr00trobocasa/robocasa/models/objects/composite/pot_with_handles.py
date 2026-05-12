import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.models.objects import CompositeObject
from robosuite.utils.mjcf_utils import (
    BLUE,
    GREEN,
    RED,
    CustomMaterial,
    add_to_dict,
    array_to_string,
)


class PotWithHandlesObject(CompositeObject):
    """
    Generates the Pot object with side handles (used in TwoArmLift)

    Args:
        name (str): Name of this Pot object

        body_half_size (3-array of float): If specified, defines the (x,y,z) half-dimensions of the main pot
            body. Otherwise, defaults to [0.07, 0.07, 0.07]

        handle_radius (float): Determines the pot handle radius

        handle_length (float): Determines the pot handle length

        handle_width (float): Determines the pot handle width

        handle_friction (float): Friction value to use for pot handles. Defauls to 1.0

        density (float): Density value to use for all geoms. Defaults to 1000

        use_texture (bool): If true, geoms will be defined by realistic textures and rgba values will be ignored

        rgba_body (4-array or None): If specified, sets pot body rgba values

        rgba_handle_0 (4-array or None): If specified, sets handle 0 rgba values

        rgba_handle_1 (4-array or None): If specified, sets handle 1 rgba values

        solid_handle (bool): If true, uses a single geom to represent the handle

        thickness (float): How thick to make the pot body walls
    """

    def __init__(
        self,
        name,
        body_half_size=(0.07, 0.07, 0.07),
        handle_radius=0.01,
        handle_length=0.09,
        handle_width=0.09,
        handle_friction=1.0,
        density=1000,
        use_texture=True,
        rgba_body=None,
        rgba_handle_0=None,
        rgba_handle_1=None,
        solid_handle=False,
        thickness=0.01,  # For body
    ):
        # Set name
        self._name = name

        # Set object attributes
        self.body_half_size = np.array(body_half_size)
        self.thickness = thickness
        self.handle_radius = handle_radius
        self.handle_length = handle_length
        self.handle_width = handle_width
        self.handle_friction = handle_friction
        self.density = density
        self.use_texture = use_texture
        self.rgba_body = np.array(rgba_body) if rgba_body else RED
        self.rgba_handle_0 = np.array(rgba_handle_0) if rgba_handle_0 else GREEN
        self.rgba_handle_1 = np.array(rgba_handle_1) if rgba_handle_1 else BLUE
        self.solid_handle = solid_handle

        # Element references to be filled when generated
        self._handle0_geoms = None
        self._handle1_geoms = None
        self.pot_base = None

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
        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="pot_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        greenwood = CustomMaterial(
            texture="WoodGreen",
            tex_name="greenwood",
            mat_name="handle0_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        bluewood = CustomMaterial(
            texture="WoodBlue",
            tex_name="bluewood",
            mat_name="handle1_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )
        self.append_material(redwood)
        self.append_material(greenwood)
        self.append_material(bluewood)

    def _get_geom_attrs(self):
        """
        Creates geom elements that will be passed to superclass CompositeObject constructor

        Returns:
            dict: args to be used by CompositeObject to generate geoms
        """
        full_size = np.array(
            (
                self.body_half_size,
                self.body_half_size + self.handle_length * 2,
                self.body_half_size,
            )
        )
        # Initialize dict of obj args that we'll pass to the CompositeObject constructor
        base_args = {
            "total_size": full_size / 2.0,
            "name": self.name,
            "locations_relative_to_center": True,
            "obj_types": "all",
        }
        site_attrs = []
        obj_args = {}

        # Initialize geom lists
        self._handle0_geoms = []
        self._handle1_geoms = []

        # Add main pot body
        # Base geom
        name = f"base"
        self.pot_base = [name]
        add_to_dict(
            dic=obj_args,
            geom_types="box",
            geom_locations=(0, 0, -self.body_half_size[2] + self.thickness / 2),
            geom_quats=(1, 0, 0, 0),
            geom_sizes=np.array(
                [self.body_half_size[0], self.body_half_size[1], self.thickness / 2]
            ),
            geom_names=name,
            geom_rgbas=None if self.use_texture else self.rgba_body,
            geom_materials="pot_mat" if self.use_texture else None,
            geom_frictions=None,
            density=self.density,
        )

        # Walls
        x_off = np.array(
            [
                0,
                -(self.body_half_size[0] - self.thickness / 2),
                0,
                self.body_half_size[0] - self.thickness / 2,
            ]
        )
        y_off = np.array(
            [
                -(self.body_half_size[1] - self.thickness / 2),
                0,
                self.body_half_size[1] - self.thickness / 2,
                0,
            ]
        )
        w_vals = np.array(
            [
                self.body_half_size[0],
                self.body_half_size[1],
                self.body_half_size[0],
                self.body_half_size[1],
            ]
        )
        r_vals = np.array([np.pi / 2, 0, -np.pi / 2, np.pi])
        for i, (x, y, w, r) in enumerate(zip(x_off, y_off, w_vals, r_vals)):
            add_to_dict(
                dic=obj_args,
                geom_types="box",
                geom_locations=(x, y, 0),
                geom_quats=T.convert_quat(T.axisangle2quat(np.array([0, 0, r])), to="wxyz"),
                geom_sizes=np.array([self.thickness / 2, w, self.body_half_size[2]]),
                geom_names=f"body{i}",
                geom_rgbas=None if self.use_texture else self.rgba_body,
                geom_materials="pot_mat" if self.use_texture else None,
                geom_frictions=None,
                density=self.density,
            )

        # Add handles
        main_bar_size = np.array(
            [
                self.handle_width / 2 + self.handle_radius,
                self.handle_radius,
                self.handle_radius,
            ]
        )
        side_bar_size = np.array([self.handle_radius, self.handle_length / 2, self.handle_radius])
        handle_z = self.body_half_size[2] - self.handle_radius
        for i, (g_list, handle_side, rgba) in enumerate(
            zip(
                [self._handle0_geoms, self._handle1_geoms],
                [1.0, -1.0],
                [self.rgba_handle_0, self.rgba_handle_1],
            )
        ):
            handle_center = np.array(
                (
                    0,
                    handle_side * (self.body_half_size[1] + self.handle_length),
                    handle_z,
                )
            )
            # Solid handle case
            if self.solid_handle:
                handle_center = np.array(
                    (
                        0,
                        handle_side * (self.body_half_size[1] + self.handle_length / 2),
                        handle_z,
                    )
                )
                name = f"handle{i}"
                g_list.append(name)
                add_to_dict(
                    dic=obj_args,
                    geom_types="box",
                    geom_locations=handle_center,
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=np.array(
                        [
                            self.handle_width / 2,
                            self.handle_length / 2,
                            self.handle_radius,
                        ]
                    ),
                    geom_names=name,
                    geom_rgbas=None if self.use_texture else rgba,
                    geom_materials=f"handle{i}_mat" if self.use_texture else None,
                    geom_frictions=(self.handle_friction, 0.005, 0.0001),
                    density=self.density,
                )
            # Hollow handle case
            else:
                # Center bar
                name = f"handle{i}_c"
                g_list.append(name)
                add_to_dict(
                    dic=obj_args,
                    geom_types="box",
                    geom_locations=handle_center,
                    geom_quats=(1, 0, 0, 0),
                    geom_sizes=main_bar_size,
                    geom_names=name,
                    geom_rgbas=None if self.use_texture else rgba,
                    geom_materials=f"handle{i}_mat" if self.use_texture else None,
                    geom_frictions=(self.handle_friction, 0.005, 0.0001),
                    density=self.density,
                )
                # Side bars
                for bar_side, suffix in zip([-1.0, 1.0], ["-", "+"]):
                    name = f"handle{i}_{suffix}"
                    g_list.append(name)
                    add_to_dict(
                        dic=obj_args,
                        geom_types="box",
                        geom_locations=(
                            bar_side * self.handle_width / 2,
                            handle_side * (self.body_half_size[1] + self.handle_length / 2),
                            handle_z,
                        ),
                        geom_quats=(1, 0, 0, 0),
                        geom_sizes=side_bar_size,
                        geom_names=name,
                        geom_rgbas=None if self.use_texture else rgba,
                        geom_materials=f"handle{i}_mat" if self.use_texture else None,
                        geom_frictions=(self.handle_friction, 0.005, 0.0001),
                        density=self.density,
                    )
            # Add relevant site
            handle_site = self.get_site_attrib_template()
            handle_name = f"handle{i}"
            handle_site.update(
                {
                    "name": handle_name,
                    "pos": array_to_string(handle_center - handle_side * np.array([0, 0.005, 0])),
                    "size": "0.005",
                    "rgba": rgba,
                }
            )
            site_attrs.append(handle_site)
            # Add to important sites
            self._important_sites[f"handle{i}"] = self.naming_prefix + handle_name

        # Add pot body site
        pot_site = self.get_site_attrib_template()
        center_name = "center"
        pot_site.update(
            {
                "name": center_name,
                "size": "0.005",
            }
        )
        site_attrs.append(pot_site)
        # Add to important sites
        self._important_sites["center"] = self.naming_prefix + center_name

        # Add back in base args and site args
        obj_args.update(base_args)
        obj_args["sites"] = site_attrs  # All sites are part of main (top) body

        # Return this dict
        return obj_args

    @property
    def handle_distance(self):
        """
        Calculates how far apart the handles are

        Returns:
            float: handle distance
        """
        return self.body_half_size[1] * 2 + self.handle_length * 2

    @property
    def handle0_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to handle0 (green handle)
        """
        return self.correct_naming(self._handle0_geoms)

    @property
    def handle1_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to handle1 (blue handle)
        """
        return self.correct_naming(self._handle1_geoms)

    @property
    def handle_geoms(self):
        """
        Returns:
            list of str: geom names corresponding to both handles
        """
        return self.handle0_geoms + self.handle1_geoms

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle0'`: Name of handle0 location site
                :`'handle1'`: Name of handle1 location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update(self._important_sites)
        return dic

    @property
    def bottom_offset(self):
        return np.array([0, 0, -1 * self.body_half_size[2]])

    @property
    def top_offset(self):
        return np.array([0, 0, self.body_half_size[2]])

    @property
    def horizontal_radius(self):
        return np.sqrt(2) * (max(self.body_half_size) + self.handle_length)
