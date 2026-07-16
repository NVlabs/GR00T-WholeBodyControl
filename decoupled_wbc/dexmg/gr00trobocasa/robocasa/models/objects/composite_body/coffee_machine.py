from robosuite.models.objects import CompositeBodyObject, BoxObject
from robocasa.models.objects.composite_body.cup import (
    CupObject,
)
from robocasa.models.objects.xml_objects import (
    CoffeeMachineBodyObject,
    CoffeeMachineLidObject,
    CoffeeMachineBaseObject,
)
import numpy as np

from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.mjcf_utils import RED, BLUE, CustomMaterial


class CoffeeMachineObject(CompositeBodyObject):
    """
    Coffee machine object with a lid fixed on a hinge joint.
    """

    def __init__(
        self,
        name,
        add_cup=True,
        pod_holder_friction=None,
        joints="default",
        density=1000.0,
    ):

        # pieces of the coffee machine
        body = CoffeeMachineBodyObject(name="body")
        body_size = body.get_bounding_box_half_size()
        body_location = [0.0, 0.0, 0.0]

        lid = CoffeeMachineLidObject(name="lid")
        lid_size = self.lid_size = lid.get_bounding_box_half_size()
        # add tolerance to allow lid to open fully
        lid_location = [
            body_size[0] - lid_size[0],
            2.0 * body_size[1] + 0.01,
            2.0 * (body_size[2] - lid_size[2]) + 0.005,
        ]

        # add in hinge joint to lid
        hinge_pos = [0.0, -lid_size[1], 0.0]
        hinge_joint = dict(
            type="hinge",
            axis="1 0 0",
            pos=array_to_string(hinge_pos),
            limited="true",
            range="{} {}".format(0, 2.0 * np.pi / 3.0),
            damping="0.005",
        )
        body_joints = dict(lid_main=[hinge_joint])  # note: "main" gets appended to body name
        lid = CoffeeMachineLidObject(name="lid")

        base = CoffeeMachineBaseObject(name="base")
        base_size = base.get_bounding_box_half_size()
        base_location = [body_size[0] - base_size[0], 2.0 * body_size[1], 0.0]

        pod_holder_holder = BoxObject(
            name="pod_holder_holder",
            size=[
                0.01,
                # tolerance for having the lid stick out a little from the holder
                0.9 * (lid_size[1] - lid_size[0]),
                0.005,
            ],
            rgba=[0.839, 0.839, 0.839, 1],  # silver
            joints=None,
        )
        pod_holder_holder_size = pod_holder_holder.get_bounding_box_half_size()
        pod_holder_holder_location = [
            body_size[0] - pod_holder_holder_size[0],
            2.0 * body_size[1],
            # put right underneath lid
            2.0 * (body_size[2] - lid_size[2] - pod_holder_holder_size[2]),
        ]

        pod_holder = CupObject(
            name="pod_holder",
            outer_cup_radius=lid_size[0],
            inner_cup_radius=0.028,
            cup_height=0.028,
            cup_ngeoms=64,  # 8,
            cup_base_height=0.005,
            cup_base_offset=0.002,
            add_handle=False,
            rgba=[1, 0, 0, 1],
            density=density,
            joints=None,
            friction=pod_holder_friction,
        )
        pod_holder_size = self.pod_holder_size = pod_holder.get_bounding_box_half_size()
        # pod_holder_size = self.pod_holder_size = np.array([0.0295, 0.0295, 0.028 ])
        pod_holder_location = [
            body_size[0] - pod_holder_size[0],
            2.0 * (body_size[1] + pod_holder_holder_size[1]),
            # put right underneath lid
            2.0 * (body_size[2] - lid_size[2] - pod_holder_size[2]),
        ]

        total_size = [
            body_size[0],
            body_size[1] + base_size[1],
            body_size[2],
        ]

        objects = [
            body,
            lid,
            base,
            pod_holder_holder,
            pod_holder,
        ]

        object_locations = [
            body_location,
            lid_location,
            base_location,
            pod_holder_holder_location,
            pod_holder_location,
        ]

        object_quats = [
            [0.0, 0.0, 0.0, 1.0],  # z-rotate body and base by 180
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ]

        # add a rigidly mounted cup to the base
        self.add_cup = add_cup
        if self.add_cup:
            cup = CupObject(
                name="cupppp",
                outer_cup_radius=0.03,
                inner_cup_radius=0.025,
                cup_height=0.025,
                cup_ngeoms=64,  # 8,
                cup_base_height=0.005,
                cup_base_offset=0.005,
                add_handle=True,
                handle_outer_radius=0.015,
                handle_inner_radius=0.010,
                handle_thickness=0.003,
                handle_ngeoms=64,
                rgba=[0.839, 0.839, 0.839, 1],
                density=1000.0,
                joints=None,
            )
            cup_total_size = cup.get_bounding_box_half_size()
            # cup_total_size = np.array([0.03 , 0.045, 0.025])
            objects.append(cup)
            object_locations.append(
                [
                    body_size[0] - cup_total_size[0],
                    2.0 * (body_size[1] + pod_holder_holder_size[1])
                    + pod_holder_size[1]
                    - cup_total_size[1],
                    2.0 * base_size[2],
                ]
            )
            rot_angle = -np.pi / 2.0
            object_quats.append([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)])

        object_parents = [None] * len(objects)

        """
        Variables to compare:

        objects
        [<robosuite.models.objects.xml_objects.CoffeeMachineBodyObject at 0x10f0f7640>,
         <robosuite.models.objects.xml_objects.CoffeeMachineLidObject at 0x10f0f7c10>,
         <robosuite.models.objects.xml_objects.CoffeeMachineBaseObject at 0x10f0f7190>,
         <robosuite.models.objects.generated_objects.BoxObject at 0x10f0f7d90>,
         <robosuite.models.objects.generated_objects.CupObject at 0x10f0f71f0>,
         <robosuite.models.objects.generated_objects.CupObject at 0x10f0f7070>]

         object_locations
         [[0.0, 0.0, 0.0],
         [0.056999999999999995, 0.21100000000000002, 0.20700000000000002],
         [0.04449999999999999, 0.201, 0.0],
         [0.0765, 0.201, 0.192],
         [0.056999999999999995, 0.22710000000000002, 0.14600000000000002],
         [0.056499999999999995, 0.21160000000000007, 0.01]]

        object_quats
         [[0.0, 0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0],
         [1.0, 0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0],
         [0.7071067811865476, 0, 0, -0.7071067811865475]]

         body_size
         array([0.0865, 0.1005, 0.1105])

         lid_size
         array([0.0295, 0.044 , 0.0095])

         lid_location
         [0.056999999999999995, 0.21100000000000002, 0.20700000000000002]

         base_size
         array([0.042, 0.05 , 0.005])

         base_location
         [0.04449999999999999, 0.201, 0.0]

         pod_holder_holder_size
         array([0.01   , 0.01305, 0.005  ])

         pod_holder_holder_location
         [0.0765, 0.201, 0.192]

         pod_holder_size
         array([0.0295, 0.0295, 0.028 ])

         pod_holder_location
         [0.056999999999999995, 0.22710000000000002, 0.14600000000000002]

         total_size
         [0.0865, 0.15050000000000002, 0.1105]

         cup.total_size
         array([0.03 , 0.045, 0.025])
        """

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=object_locations,
            object_quats=object_quats,
            object_parents=object_parents,
            body_joints=body_joints,  # make sure hinge joint is added
            joints=joints,
            # joints="default", # coffee machine can move
            # joints=None, # coffee machine does not move
            total_size=total_size,
            locations_relative_to_corner=True,
        )
