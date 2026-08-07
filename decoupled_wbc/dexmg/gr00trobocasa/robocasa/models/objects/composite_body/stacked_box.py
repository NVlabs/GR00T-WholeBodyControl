from robosuite.models.objects import CompositeBodyObject, BoxObject, CylinderObject, Bin
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.mjcf_utils import RED, BLUE, CustomMaterial


class StackedBoxObject(CompositeBodyObject):
    """
    Two boxes - one stacked on top of the other.
    """

    def __init__(
        self,
        name,
        box_1_size,
        box_2_size,
        joints="default",
        box_1_rgba=None,
        box_2_rgba=None,
        box_1_material=None,
        box_2_material=None,
        density=100.0,
        friction=None,
        make_box_2_transparent=False,
    ):

        # Object properties

        # half-sizes of first (bottom) box
        self.box_1_size = list(box_1_size)

        # half-sizes of second (top) box
        self.box_2_size = list(box_2_size)

        # maybe make box 2 have transparent top and bottom walls
        self.make_box_2_transparent = make_box_2_transparent

        # Create objects
        objects = []
        object_locations = []
        object_quats = []
        object_parents = []

        # NOTE: we will place the object frame at the vertical center of the two stacked boxes
        z_center = (self.box_1_size[2] + self.box_2_size[2]) / 2.0
        b1_offset = self.box_1_size[2] - z_center
        b2_offset = 2.0 * self.box_1_size[2] + self.box_2_size[2] - z_center

        # first (bottom) box
        self.box_1 = BoxObject(
            name="box_1",
            size=self.box_1_size,
            rgba=box_1_rgba,
            material=box_1_material,
            density=density,
            friction=friction,
            joints=None,
        )
        objects.append(self.box_1)
        object_locations.append([0.0, 0.0, b1_offset])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # second (top) box
        if self.make_box_2_transparent:
            self.box_2 = Bin(
                name="box_2",
                bin_size=(
                    2.0 * self.box_2_size[0],
                    2.0 * self.box_2_size[1],
                    2.0 * self.box_2_size[2],
                ),
                wall_thickness=0.01,
                transparent_walls=False,
                friction=friction,
                density=density,
                use_texture=True,
                rgba=box_2_rgba,
                material=box_2_material,
                upside_down=False,
                add_second_base=True,
                transparent_base=True,
            )
        else:
            self.box_2 = BoxObject(
                name="box_2",
                size=self.box_2_size,
                rgba=box_2_rgba,
                material=box_2_material,
                density=density,
                friction=friction,
                joints=None,
            )
        objects.append(self.box_2)
        object_locations.append([0.0, 0.0, b2_offset])
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
            # total_size=body_total_size,
            # locations_relative_to_corner=True,
        )
