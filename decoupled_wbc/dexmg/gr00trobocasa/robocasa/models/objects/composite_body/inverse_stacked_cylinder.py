from robosuite.models.objects import (
    CompositeBodyObject,
    BoxObject,
    CylinderObject,
    HollowCylinderObject,
)
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.mjcf_utils import RED, BLUE, CustomMaterial


class InverseStackedCylinderObject(CompositeBodyObject):
    """
    Inverse of stacked cylinder object, where the top piece is a hollow cylinder object
    and the bottom piece is a cylinder object. Optionally add a square base for stability.
    """

    def __init__(
        self,
        name,
        radius_1,
        radius_2,
        height_1,
        height_2,
        ngeoms=64,
        joints="default",
        rgba=None,
        material=None,
        density=100.0,
        friction=None,
        square_base_width=None,
        square_base_height=None,
    ):

        # Object properties

        # radius of first (bottom) cylinder and inner radius of second (top) hollow cylinder
        self.r1 = radius_1
        self.r2 = radius_2

        # half-height of first (bottom) cylinder and second (top) hollow cylinder
        self.h1 = height_1
        self.h2 = height_2

        # num geoms to approximate the hollow cylinder
        self.ngeoms = ngeoms

        # whether to add square base
        self.add_square_base = (square_base_width is not None) and (square_base_height is not None)

        # half-width and half-height for square base
        self.square_base_width = square_base_width
        self.square_base_height = square_base_height

        # Create objects
        objects = []
        object_locations = []
        object_quats = []
        object_parents = []

        # NOTE: we will place the object frame at the vertical center of the two stacked cylinders
        z_center = (self.h1 + self.h2) / 2.0
        c1_offset = self.h1 - z_center
        c2_offset = 2.0 * self.h1 + self.h2 - z_center

        # first (bottom) cylinder
        self.cylinder_1 = CylinderObject(
            name="cylinder_1",
            size=[self.r1, self.h1],
            rgba=rgba,
            material=material,
            density=density,
            friction=friction,
            solref=[0.02, 1.0],
            # solimp=[0.998, 0.998, 0.001],
            solimp=[0.9, 0.95, 0.001],
            joints=None,
        )
        objects.append(self.cylinder_1)
        object_locations.append([0.0, 0.0, c1_offset])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # second (top) hollow cylinder
        self.cylinder_2 = HollowCylinderObject(
            name="cylinder_2",
            outer_radius=self.r1,  # match radius of first cylinder
            inner_radius=self.r2,
            height=self.h2,
            ngeoms=self.ngeoms,
            rgba=rgba,
            material=material,
            density=density,
            friction=friction,
            # TODO: maybe tune solimp and try (0.998, 0.998, 0.001)
            solref=[0.02, 1.0],
            solimp=[0.9, 0.95, 0.001],
            # solimp=(0.998, 0.998, 0.001),
        )
        objects.append(self.cylinder_2)
        object_locations.append([0.0, 0.0, c2_offset])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # # total size of object
        # max_r = max(self.r1, self.r2)
        # body_total_size = [max_r, max_r, self.h1 + self.h2]

        if self.add_square_base:
            # add square base underneath bottom cylinder
            s1_offset = c1_offset - (self.square_base_height + self.h1)
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
            # total_size=body_total_size,
            # locations_relative_to_corner=True,
        )
