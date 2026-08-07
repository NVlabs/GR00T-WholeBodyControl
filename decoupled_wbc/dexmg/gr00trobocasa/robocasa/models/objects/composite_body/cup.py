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


class CupObject(CompositeBodyObject):
    """
    Cup object with optional handle.
    """

    def __init__(
        self,
        name,
        outer_cup_radius=0.0425,
        inner_cup_radius=0.03,
        cup_height=0.05,
        cup_ngeoms=8,
        cup_base_height=0.01,
        cup_base_offset=0.005,
        add_handle=False,
        handle_outer_radius=0.03,
        handle_inner_radius=0.015,
        handle_thickness=0.005,
        handle_ngeoms=8,
        joints="default",
        rgba=None,
        material=None,
        density=100.0,
        friction=None,
    ):

        # Object properties

        # radius of the inner cup hole and entire cup
        self.r1 = inner_cup_radius
        self.r2 = outer_cup_radius

        # number of geoms used to approximate the cylindrical shell
        self.n = cup_ngeoms

        # cup half-height
        self.cup_height = cup_height

        # cup base args
        self.cup_base_height = cup_base_height
        self.cup_base_offset = cup_base_offset

        # handle args
        self.add_handle = add_handle
        self.handle_outer_radius = handle_outer_radius
        self.handle_inner_radius = handle_inner_radius
        self.handle_thickness = handle_thickness
        self.handle_ngeoms = handle_ngeoms

        # Create objects
        objects = []
        object_locations = []
        object_quats = []
        object_parents = []

        # cup body
        self.cup_body = HollowCylinderObject(
            name="cup_body",
            outer_radius=self.r2,
            inner_radius=self.r1,
            height=self.cup_height,
            ngeoms=self.n,
            rgba=rgba,
            material=material,
            density=density,
            friction=friction,
        )
        objects.append(self.cup_body)
        object_locations.append([0.0, 0.0, 0.0])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # cup base
        self.cup_base = CylinderObject(
            name="cup_base",
            size=[self.cup_body.int_r, self.cup_base_height],
            rgba=rgba,
            material=material,
            density=density,
            solref=[0.02, 1.0],
            solimp=[0.998, 0.998, 0.001],
            joints=None,
        )
        objects.append(self.cup_base)
        object_locations.append(
            [0.0, 0.0, -self.cup_height + self.cup_base_height + self.cup_base_offset]
        )
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        if self.add_handle:
            # cup handle is a hollow half-cylinder
            self.cup_handle = HollowCylinderObject(
                name="cup_handle",
                outer_radius=self.handle_outer_radius,
                inner_radius=self.handle_inner_radius,
                height=self.handle_thickness,
                ngeoms=self.handle_ngeoms,
                rgba=rgba,
                material=material,
                density=density,
                make_half=True,
                friction=friction,
            )
            # translate handle to right side of cup body, and rotate by +90 degrees about y-axis
            # to orient the handle geoms on the cup body
            objects.append(self.cup_handle)
            object_locations.append([0.0, (self.cup_body.r2 + self.cup_handle.unit_box_width), 0.0])
            object_quats.append(
                T.convert_quat(
                    T.mat2quat(
                        T.rotation_matrix(angle=np.pi / 2.0, direction=[0.0, 1.0, 0.0])[:3, :3]
                    ),
                    to="wxyz",
                )
            )
            object_parents.append(None)

        # total size of cup
        body_total_size = [self.r2, self.r2, self.cup_height]
        if self.add_handle:
            body_total_size[1] += self.handle_outer_radius

        """
        To match on cup 1:

        objects
        [<robosuite.models.objects.generated_objects.HollowCylinderObject at 0x112881370>,
        <robosuite.models.objects.generated_objects.CylinderObject at 0x112881c40>]

        body_total_size
        [0.0295, 0.0295, 0.028]

        object_locations
        [[0.0, 0.0, 0.0], [0.0, 0.0, -0.020999999999999998]]

        object_quats
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]


        To match on cup 2:

        objects
        [<robosuite.models.objects.generated_objects.HollowCylinderObject at 0x113e2b0d0>,
         <robosuite.models.objects.generated_objects.CylinderObject at 0x113e2bbe0>,
         <robosuite.models.objects.generated_objects.HollowCylinderObject at 0x113e2b040>]

         body_total_size
         [0.03, 0.045, 0.025]

         object_locations
         [[0.0, 0.0, 0.0], [0.0, 0.0, -0.015], [0.0, 0.03073601511491127, 0.0]]

         object_quats
         [[1.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         array([ 0.70710677, -0.        ,  0.70710677, -0.        ], dtype=float32)]
        """

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=object_locations,
            object_quats=object_quats,
            object_parents=object_parents,
            joints=joints,
            total_size=body_total_size,
            # locations_relative_to_corner=True,
        )
