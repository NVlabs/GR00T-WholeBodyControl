import numpy as np

from robosuite.models.objects import (
    BoxObject,
    CompositeBodyObject,
    CylinderObject,
    ConeObject,
)
from robosuite.utils.mjcf_utils import BLUE, RED, CustomMaterial, array_to_string


class SprayBottleObject(CompositeBodyObject):
    """
    A spray bottle object where the trigger is modeled via a slide joint.
    """

    def __init__(
        self,
        name,
        base_cylinder_radius,
        base_cylinder_height,
        cone_outer_radius,
        cone_inner_radius,
        cone_height,
        cone_ngeoms,
        neck_height,
        top_length,
        top_thickness,
        trigger_height,
        trigger_width,
        trigger_length,
        joints="default",
        rgba=None,
        material=None,
        density=100.0,
        friction=None,
        square_base_width=None,
        square_base_height=None,
    ):
        # Sizes
        self.base_cylinder_radius = base_cylinder_radius
        self.base_cylinder_height = base_cylinder_height
        self.cone_outer_radius = cone_outer_radius
        self.cone_inner_radius = cone_inner_radius
        self.cone_height = cone_height
        self.cone_ngeoms = cone_ngeoms
        self.neck_height = neck_height
        self.top_length = top_length
        self.top_thickness = top_thickness
        self.trigger_height = trigger_height
        self.trigger_width = trigger_width
        self.trigger_length = trigger_length

        # whether to add square base
        self.add_square_base = (square_base_width is not None) and (square_base_height is not None)

        # half-width and half-height for square base
        self.square_base_width = square_base_width
        self.square_base_height = square_base_height

        assert self.top_length > self.cone_inner_radius, "top needs to stick out past neck"

        # Create objects
        objects = []
        object_locations = []
        object_quats = []
        object_parents = []

        # base cylinder
        self.base_cylinder = CylinderObject(
            name="base_cylinder",
            size=[self.base_cylinder_radius, self.base_cylinder_height],
            rgba=rgba,
            material=material,
            density=density,
            friction=friction,
            solref=[0.02, 1.0],
            # solimp=[0.998, 0.998, 0.001],
            solimp=[0.9, 0.95, 0.001],
            joints=None,
        )
        objects.append(self.base_cylinder)
        object_locations.append([0.0, 0.0, 0.0])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # cone that is part of neck
        # z_cone = self.base_cylinder_height # cone frame is bottom center of cone
        z_cone = self.base_cylinder_height + (self.cone_height / 2.0)
        self.neck_cone = ConeObject(
            name="neck_cone",
            outer_radius=self.cone_outer_radius,
            inner_radius=self.cone_inner_radius,
            height=self.cone_height,  # full height, not half-height
            ngeoms=self.cone_ngeoms,
            use_box=False,
            rgba=rgba,
            material=material,
            density=density,
            solref=(0.02, 1.0),
            # solimp=[0.998, 0.998, 0.001],
            solimp=(0.9, 0.95, 0.001),
            friction=friction,
        )
        objects.append(self.neck_cone)
        object_locations.append([0.0, 0.0, z_cone])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # cylinder that is part of neck
        z_cyl = self.base_cylinder_height + self.cone_height + self.neck_height
        self.neck_cylinder = CylinderObject(
            name="neck_cylinder",
            size=[self.cone_inner_radius, self.neck_height],  # match cone radius
            rgba=rgba,
            material=material,
            density=density,
            friction=friction,
            solref=[0.02, 1.0],
            # solimp=[0.998, 0.998, 0.001],
            solimp=[0.9, 0.95, 0.001],
            joints=None,
        )
        objects.append(self.neck_cylinder)
        object_locations.append([0.0, 0.0, z_cyl])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # top box object
        box_x_off = -(self.top_length - self.cone_inner_radius)
        box_z_off = z_cyl + self.neck_height + self.top_thickness
        self.top_box = BoxObject(
            name="top_box",
            size=[self.top_length, self.cone_inner_radius, self.top_thickness],
            density=density,
            friction=friction,
            rgba=rgba,
            material=material,
            joints=None,
        )
        objects.append(self.top_box)
        object_locations.append([box_x_off, 0.0, box_z_off])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # trigger object, which is defined relative to top box object
        trigger_x_off = -(0.5 * self.top_length)  # place halfway along length of top object
        trigger_z_off = -(self.top_thickness + self.trigger_height)
        self.trigger = BoxObject(
            name="trigger",
            size=[self.trigger_length, self.trigger_width, self.trigger_height],
            density=density,
            friction=friction,
            rgba=rgba,
            material=material,
            joints=None,
        )
        objects.append(self.trigger)
        object_locations.append([trigger_x_off, 0.0, trigger_z_off])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(objects[-2].root_body)  # NOTE: relative to top box object

        # define slide joint for trigger
        rel_joint_pos = [0, 0, 0]  # at trigger
        joint_lim_min = 0.0
        joint_lim_max = (
            self.top_length
        )  # trigger was shifted to left by 0.5 * top_length, so allow it to go to right by top_length
        slide_joint = {
            "name": "trigger_slide",
            "type": "slide",
            "axis": "1 0 0",  # x-axis slide
            "pos": array_to_string(rel_joint_pos),
            "springref": "0",
            "springdamper": "0.1 1.0",  # mass-spring system with 0.1 time constant, 1.0 damping ratio
            "limited": "true",
            "range": "{} {}".format(joint_lim_min, joint_lim_max),
        }

        if self.add_square_base:
            # add square base underneath bottom cylinder
            s1_offset = -(self.square_base_height + self.base_cylinder_height)
            self.square_base = BoxObject(
                name="square_base",
                size=[
                    self.square_base_width,
                    self.square_base_width,
                    self.square_base_height,
                ],
                rgba=rgba,
                material=material,
            )
            objects.append(self.square_base)
            object_locations.append([0.0, 0.0, s1_offset])
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(None)

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=object_locations,
            object_quats=object_quats,
            object_parents=object_parents,
            joints=joints,
            body_joints={objects[-1].root_body: [slide_joint]},
        )
