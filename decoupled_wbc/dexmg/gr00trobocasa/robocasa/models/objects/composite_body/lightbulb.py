from robosuite.models.objects import (
    CompositeBodyObject,
    BoxObject,
    CylinderObject,
    BallObject,
)
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string, new_site
from robosuite.utils.mjcf_utils import RED, BLUE, CustomMaterial


class BallObjectWithSite(BallObject):
    """
    A ball object with a inner site (used for the bulb).
    """

    def _get_object_subtree(self):
        # tree = super()._get_object_subtree()
        tree = self._get_object_subtree_(ob_type="sphere")
        site_element_attr = self.get_site_attrib_template()

        site_element_attr["pos"] = "0 0 0.1"
        site_element_attr["name"] = "center_site"
        site_element_attr["size"] = "{} {} {}".format(
            2.0 * self.size[0], 2.0 * self.size[0], 2.0 * self.size[0]
        )
        site_element_attr["rgba"] = "1.0 0.0 0.0 1.0"
        site_element_attr["group"] = "1"
        # site_element_attr["rgba"] = "1.0 0.976 0.839 0.9"
        # site_element_attr["rgba"] = "1.0 0.976 0.839 0.0"
        tree.append(new_site(**site_element_attr))
        return tree


class LightbulbObject(CompositeBodyObject):
    """
    A simple lightbulb constructed out of a base of alternating radius cylinders
    and a sphere on top.
    """

    def __init__(
        self,
        name,
        radius_low,
        radius_high,
        cylinder_height,
        num_cylinders,
        sphere_radius,
        joints="default",
        density=100.0,
        friction=None,
    ):

        # Object properties

        # radii of alternating cylinders for base
        self.radius_low = radius_low
        self.radius_high = radius_high

        # half-height of each cylinder
        self.cylinder_height = cylinder_height

        # number of cylinders for base
        self.num_cylinders = num_cylinders

        # radius of sphere at top
        self.sphere_radius = sphere_radius

        # toggle between translucent and yellow
        self.translucent_rgba = (1.0, 1.0, 1.0, 0.3)
        self.yellow_rgba = (1.0, 0.976, 0.839, 0.7)
        # self.yellow_rgba = (0.0, 0.0, 0.0, 0.0)

        # Create objects
        objects = []
        object_locations = []
        object_quats = []
        object_parents = []

        # NOTE: we will place the object frame at the vertical center of all the stacked objects
        self.z_center = (
            (2.0 * self.cylinder_height) * self.num_cylinders + 2.0 * self.sphere_radius
        ) / 2.0

        metal = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"},
        )

        # we will define all objects relative to the bottom of the object, and then subtract the z_center value
        for cylinder_ind in range(self.num_cylinders):
            r = self.radius_low if ((cylinder_ind % 2) == 0) else self.radius_high
            cyl_obj = CylinderObject(
                name="cylinder_{}".format(cylinder_ind),
                size=[r, self.cylinder_height],
                rgba=None,
                material=metal,
                density=density,
                friction=friction,
                solref=[0.02, 1.0],
                solimp=[0.9, 0.95, 0.001],
                joints=None,
            )
            objects.append(cyl_obj)
            z_cyl = (2.0 * cylinder_ind + 1) * self.cylinder_height
            object_locations.append([0.0, 0.0, z_cyl])
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(None)

        # then add translucent sphere at top
        self.bulb = BallObject(
            name="bulb",
            size=[self.sphere_radius],
            density=density,
            friction=friction,
            rgba=self.translucent_rgba,
            material=None,
            joints=None,
        )
        objects.append(self.bulb)
        z_bulb = object_locations[-1][2] + self.cylinder_height + self.sphere_radius
        object_locations.append([0.0, 0.0, z_bulb])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # do frame conversion from bottom of object to z_center
        object_locations = [[loc[0], loc[1], loc[2] - self.z_center] for loc in object_locations]

        # add site that can be toggled to turn lightbulb on
        sites = [
            dict(
                name="bulb_on",
                pos=array_to_string(object_locations[-1]),
                size="{}".format(0.95 * self.sphere_radius),
                rgba=array_to_string(self.yellow_rgba),
            )
        ]

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=object_locations,
            object_quats=object_quats,
            object_parents=object_parents,
            joints=joints,
            sites=sites,
        )
