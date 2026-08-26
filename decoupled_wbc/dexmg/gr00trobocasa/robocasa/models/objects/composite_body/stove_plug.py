from robosuite.models.objects import CompositeBodyObject, BoxObject, CylinderObject
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.mjcf_utils import RED, BLUE, CustomMaterial
from .inverse_stacked_cylinder import InverseStackedCylinderObject
from .lightbulb import LightbulbObject

import robosuite_task_zoo
from robosuite_task_zoo.models.kitchen import StoveObject


class StoveObjectNew(StoveObject):
    """
    Override some offsets for placement sampler.
    """

    @property
    def bottom_offset(self):
        # unused since we directly hardcode z
        return np.array([0, 0, -0.02])

    @property
    def top_offset(self):
        # unused since we directly hardcode z
        return np.array([0, 0, 0.02])

    @property
    def horizontal_radius(self):
        return 0.1


class StovePlugObject(CompositeBodyObject):
    """
    Stove on wooden block with chain (plug) connected. Optionally replace the stove
    with a lightbulb if @lightbulb_args is provided.
    """

    def __init__(
        self,
        name,
        joints="default",
        rgba=None,
        material=None,
        density=100.0,
        friction=None,
        stove_base_size=(0.12, 0.12, 0.01),
        stove_z_half_size=0.025,
        wire_box_geom_size=(0.005, 0.02, 0.005),
        wire_box_geom_rgba=(0.0, 0.0, 0.0, 1.0),
        num_box_geoms_left=5,
        num_box_geoms_vert=8,
        num_box_geoms_right=8,
        merge_box_geoms=False,
        merge_size=1,
        cylinder_args=None,
        lightbulb_args=None,
    ):

        # Object properties

        # half sizes for stove base box object
        # self.stove_base_size = (0.1, 0.1, 0.02)
        self.stove_base_size = stove_base_size
        self.stove_z_half_size = stove_z_half_size  # note: estimated approximately

        # box geoms used for wire
        self.wire_box_geom_size = wire_box_geom_size
        self.wire_box_geom_rgba = wire_box_geom_rgba

        # wire parameters - number of geoms to use for left, down, and right portions
        self.num_box_geoms_left = num_box_geoms_left
        self.num_box_geoms_vert = num_box_geoms_vert
        self.num_box_geoms_right = num_box_geoms_right

        # if true, merge the box geoms along each direction of the wire into a single box geom
        self.merge_box_geoms = merge_box_geoms

        # number of box geoms to use for each merged size (set to higher than 1 to merge the geoms into more
        # than one box geom)
        self.merge_size = merge_size

        if cylinder_args is None:
            # default cylinder args
            cylinder_args = dict(
                # bottom cylinder radius and half-height
                radius_1=0.03,
                height_1=0.01,
                # top hollow cylinder inner radius and half-height
                radius_2=0.025,
                height_2=0.025,
                # NOTE: reduce to 8 geoms if desired
                ngeoms=64,
                rgba=[0.839, 0.839, 0.839, 1],
                density=1000.0,
                # add square base
                square_base_width=0.03,
                square_base_height=0.005,
            )
        self.cylinder_args = dict(cylinder_args)
        self.cylinder_args["joints"] = None

        self.use_lightbulb = lightbulb_args is not None
        if self.use_lightbulb:
            self.lightbulb_args = dict(lightbulb_args)
            self.lightbulb_args["joints"] = None

        # materials
        box_geom_material = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"},
        )
        stove_base_material = CustomMaterial(
            texture="WoodLight",
            tex_name="lightwood",
            mat_name="lightwood_mat",
            tex_attrib={"type": "cube"},
            mat_attrib={"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"},
        )

        # params for ball joints used

        # <joint name="J1" type="ball" pos="0 0 0" frictionloss="&friction;"/>
        ball_joint_spec = {
            "type": "ball",
            "pos": "0 {} 0".format(self.wire_box_geom_size[1]),
            "springref": "0",
            "springdamper": "0.1 1.0",  # mass-spring system with 0.1 time constant, 1.0 damping ratio
            "limited": "true",
            "range": "0 {}".format(np.pi / 4),
        }

        # Create objects
        objects = []
        object_locations = []
        object_quats = []
        object_parents = []
        object_joints = dict()

        # NOTE: For absolute object locations (objects not defined relative to parent) we will use the stove base frame
        #       as a frame of reference, and add an offset from the center of the object (approximated via full width)
        #       to it.

        # get an approximate x-y bounding box below, and place the center there, and define offset relative to stove base cente
        approx_full_width = self.num_box_geoms_left * (2.0 * self.wire_box_geom_size[1]) + (
            2.0 * self.stove_base_size[1]
        )
        approx_full_height = (self.num_box_geoms_vert + 2) * (2.0 * self.wire_box_geom_size[1])

        stove_base_x_off = -(approx_full_height / 2.0) + self.stove_base_size[0]
        stove_base_y_off = (approx_full_width / 2.0) - self.stove_base_size[1]

        # base of stove
        self.stove_base = BoxObject(
            name="stove_base",
            size=list(self.stove_base_size),
            rgba=rgba,
            material=stove_base_material,
            joints=None,
        )
        objects.append(self.stove_base)
        object_locations.append([stove_base_x_off, stove_base_y_off, 0.0])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        if self.use_lightbulb:
            # lightbulb
            self.lightbulb = LightbulbObject(
                name="lightbulb",
                **self.lightbulb_args,
            )
            objects.append(self.lightbulb)
            object_locations.append(
                [
                    stove_base_x_off,
                    stove_base_y_off,
                    (self.stove_base_size[2] + self.lightbulb.z_center),
                ]
            )
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(None)
        else:
            # stove
            self.stove = StoveObjectNew(
                name="new_stove",
                joints=None,
            )
            objects.append(self.stove)
            object_locations.append(
                [
                    stove_base_x_off,
                    stove_base_y_off,
                    (self.stove_base_size[2] + self.stove_z_half_size),
                ]
            )
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(None)

        # chain to the left of stove base
        chain_ind = 0
        left_chain_size = list(self.wire_box_geom_size)
        num_geoms_iter = self.num_box_geoms_left - 1
        if self.merge_box_geoms:
            # only one big geom instead of chain of geoms
            left_chain_size[1] *= self.num_box_geoms_left
            left_chain_size[1] /= self.merge_size
            # number of additional geoms to add
            num_geoms_iter = self.merge_size - 1
        left_chain_obj = BoxObject(
            name="chain_{}".format(chain_ind),
            size=list(left_chain_size),
            rgba=list(self.wire_box_geom_rgba),
            material=box_geom_material,
            joints=None,
        )
        chain_ind += 1
        objects.append(left_chain_obj)
        object_locations.append(
            [
                stove_base_x_off - 0.75 * self.stove_base_size[0],
                stove_base_y_off - (self.stove_base_size[1] + left_chain_size[1]),
                0.0,
            ]
        )
        # object_locations.append([0., -(self.stove_base_size[1] + left_chain_size[1]), 0.])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)
        if self.merge_box_geoms:
            # add ball joint and make sure to place it at the edge of the geom
            ball_joint = dict(ball_joint_spec)
            ball_joint["name"] = "ball_joint_{}".format(chain_ind)
            ball_joint["pos"] = "0 {} 0".format(left_chain_size[1])
            object_joints[left_chain_obj.root_body] = [ball_joint]
        for i in range(num_geoms_iter):
            left_chain_obj = BoxObject(
                name="chain_{}".format(chain_ind),
                size=list(left_chain_size),
                rgba=list(self.wire_box_geom_rgba),
                material=box_geom_material,
                joints=None,
            )
            chain_ind += 1
            parent = objects[-1].root_body
            objects.append(left_chain_obj)
            object_locations.append([0.0, -2.0 * left_chain_size[1], 0.0])
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(parent)

            # add ball joint and make sure to place it at the edge of the geom
            ball_joint = dict(ball_joint_spec)
            ball_joint["name"] = "ball_joint_{}".format(chain_ind)
            ball_joint["pos"] = "0 {} 0".format(left_chain_size[1])
            object_joints[left_chain_obj.root_body] = [ball_joint]

        # add chain in downward direction
        rot_quat = T.convert_quat(
            T.axisangle2quat(np.array([0.0, 0.0, 1.0]) * (np.pi / 2.0)), to="wxyz"
        )
        vert_chain_size = list(self.wire_box_geom_size)
        num_geoms_iter = self.num_box_geoms_vert - 1
        if self.merge_box_geoms:
            # only one big geom instead of chain of geoms
            vert_chain_size[1] *= self.num_box_geoms_vert
            vert_chain_size[1] /= self.merge_size
            # number of additional geoms to add
            num_geoms_iter = self.merge_size - 1
        vert_chain_obj = BoxObject(
            name="chain_{}".format(chain_ind),
            size=list(vert_chain_size),
            rgba=list(self.wire_box_geom_rgba),
            material=box_geom_material,
            joints=None,
        )
        chain_ind += 1
        parent = objects[-1].root_body
        objects.append(vert_chain_obj)
        object_locations.append([vert_chain_size[1], -left_chain_size[1], 0.0])
        object_quats.append(rot_quat)
        object_parents.append(parent)

        ball_joint = dict(ball_joint_spec)
        ball_joint["name"] = "ball_joint_{}".format(chain_ind)
        ball_joint["pos"] = "0 {} 0".format(vert_chain_size[1])
        object_joints[vert_chain_obj.root_body] = [ball_joint]

        for i in range(num_geoms_iter):
            vert_chain_obj = BoxObject(
                name="chain_{}".format(chain_ind),
                size=list(vert_chain_size),
                rgba=list(self.wire_box_geom_rgba),
                material=box_geom_material,
                joints=None,
            )
            chain_ind += 1
            parent = objects[-1].root_body
            objects.append(vert_chain_obj)
            object_locations.append([0.0, -2.0 * vert_chain_size[1], 0.0])
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(parent)

            ball_joint = dict(ball_joint_spec)
            ball_joint["name"] = "ball_joint_{}".format(chain_ind)
            ball_joint["pos"] = "0 {} 0".format(vert_chain_size[1])
            object_joints[vert_chain_obj.root_body] = [ball_joint]

        # add chain in rightward direction
        rot_quat = T.convert_quat(
            T.axisangle2quat(np.array([0.0, 0.0, 1.0]) * (np.pi / 2.0)), to="wxyz"
        )
        right_chain_size = list(self.wire_box_geom_size)
        num_geoms_iter = self.num_box_geoms_right - 1
        if self.merge_box_geoms:
            # only one big geom instead of chain of geoms
            right_chain_size[1] *= self.num_box_geoms_right
            right_chain_size[1] /= self.merge_size
            # number of additional geoms to add
            num_geoms_iter = self.merge_size - 1
        right_chain_obj = BoxObject(
            name="chain_{}".format(chain_ind),
            size=list(right_chain_size),
            rgba=list(self.wire_box_geom_rgba),
            material=box_geom_material,
            joints=None,
        )
        chain_ind += 1
        parent = objects[-1].root_body
        objects.append(right_chain_obj)
        object_locations.append([right_chain_size[1], -vert_chain_size[1], 0.0])
        object_quats.append(rot_quat)
        object_parents.append(parent)

        ball_joint = dict(ball_joint_spec)
        ball_joint["name"] = "ball_joint_{}".format(chain_ind)
        ball_joint["pos"] = "0 {} 0".format(right_chain_size[1])
        object_joints[right_chain_obj.root_body] = [ball_joint]

        for i in range(num_geoms_iter):
            right_chain_obj = BoxObject(
                name="chain_{}".format(chain_ind),
                size=list(right_chain_size),
                rgba=list(self.wire_box_geom_rgba),
                material=box_geom_material,
                joints=None,
            )
            chain_ind += 1
            parent = objects[-1].root_body
            objects.append(right_chain_obj)
            object_locations.append([0.0, -2.0 * right_chain_size[1], 0.0])
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(parent)

            ball_joint = dict(ball_joint_spec)
            ball_joint["name"] = "ball_joint_{}".format(chain_ind)
            ball_joint["pos"] = "0 {} 0".format(right_chain_size[1])
            object_joints[right_chain_obj.root_body] = [ball_joint]

        # add cylinder object
        self.cylinder_obj = InverseStackedCylinderObject(
            name="cylinder_obj",
            **self.cylinder_args,
        )

        x_off = 0.0
        y_off = -(right_chain_size[1] + self.cylinder_obj.square_base_width)
        z_off = ((self.cylinder_obj.h1 + self.cylinder_obj.h2) / 2.0) - right_chain_size[2]
        parent = objects[-1].root_body
        objects.append(self.cylinder_obj)
        object_locations.append([x_off, y_off, z_off])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(parent)

        # # add debug site to see object center
        # sites = [
        #     dict(
        #         name="TMP",
        #         pos=array_to_string([0., 0., 0.]),
        #         size="{}".format(0.1),
        #         rgba=array_to_string([1., 0., 0., 1.]),
        #     )
        # ]

        # Run super init
        super().__init__(
            name=name,
            objects=objects,
            object_locations=object_locations,
            object_quats=object_quats,
            object_parents=object_parents,
            joints=joints,
            body_joints=object_joints,
            # sites=sites,
        )
