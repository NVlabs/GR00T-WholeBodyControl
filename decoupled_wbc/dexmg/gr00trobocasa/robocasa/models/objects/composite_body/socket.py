from robosuite.models.objects import CompositeBodyObject, BoxObject, CylinderObject
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import array_to_string
from robosuite.utils.mjcf_utils import RED, BLUE, CustomMaterial
from .stacked_cylinder import StackedCylinderObject


class SocketObject(CompositeBodyObject):
    """
    Box geom connected to series of geoms with ball joints (wire) and connected to
    cylinder representing plug.
    """

    def __init__(
        self,
        name,
        joints="default",
        rgba=None,
        material=None,
        density=100.0,
        friction=None,
        socket_base_size=(0.02, 0.02, 0.03),
        wire_box_geom_size=(0.02, 0.005, 0.005),
        wire_box_geom_rgba=(0.0, 0.0, 0.0, 1.0),
        num_box_geoms_up=6,
        num_box_geoms_left=2,
        num_box_geoms_down=9,
        merge_box_geoms=False,
        merge_size=1,
        cylinder_args=None,
    ):

        # Object properties

        # box geom used for socket base
        self.socket_base_size = socket_base_size

        # box geoms used for wire
        self.wire_box_geom_size = wire_box_geom_size
        self.wire_box_geom_rgba = wire_box_geom_rgba

        # wire parameters - number of geoms to use for up, left, and down portions
        self.num_box_geoms_up = num_box_geoms_up
        self.num_box_geoms_left = num_box_geoms_left
        self.num_box_geoms_down = num_box_geoms_down

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
                height_1=0.075,
                # top cylinder radius and half-height
                radius_2=0.02,
                height_2=0.025,
                rgba=[0.839, 0.839, 0.839, 1],
                density=1000.0,
                # add square base
                square_base_width=0.03,
                square_base_height=0.005,
            )
        self.cylinder_args = dict(cylinder_args)
        self.cylinder_args["joints"] = None

        # materials
        box_geom_material = CustomMaterial(
            texture="Metal",
            tex_name="metal",
            mat_name="MatMetal",
            tex_attrib={"type": "cube"},
            mat_attrib={"specular": "1", "shininess": "0.3", "rgba": "0.9 0.9 0.9 1"},
        )
        socket_material = CustomMaterial(
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
            "pos": "{} 0 0".format(self.wire_box_geom_size[0]),
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

        # NOTE: For absolute object locations (objects not defined relative to parent) we will use the socket base frame
        #       as a frame of reference, and add an offset from the center of the object (approximated via full width)
        #       to it.

        # get an approximate x-y bounding box below, and place the center there, and define offset relative to stove base center
        approx_full_width = self.num_box_geoms_left * (2.0 * self.wire_box_geom_size[0]) + (
            2.0 * self.socket_base_size[1]
        )
        approx_full_height = (self.num_box_geoms_down + 2) * (2.0 * self.wire_box_geom_size[0])

        base_x_off = (
            self.socket_base_size[0] + (self.num_box_geoms_up) * (2.0 * self.wire_box_geom_size[0])
        ) - (approx_full_height / 2.0)
        base_y_off = (self.num_box_geoms_left * self.wire_box_geom_size[0]) + self.socket_base_size[
            1
        ]

        # base of socket
        self.socket_base = BoxObject(
            name="stove_base",
            size=list(self.socket_base_size),
            rgba=rgba,
            material=socket_material,
            joints=None,
        )
        objects.append(self.socket_base)
        object_locations.append([base_x_off, base_y_off, 0.0])
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)

        # chain above base
        chain_ind = 0
        up_chain_size = list(self.wire_box_geom_size)
        num_geoms_iter = self.num_box_geoms_up - 1
        if self.merge_box_geoms:
            # only one big geom instead of chain of geoms
            up_chain_size[0] *= self.num_box_geoms_up
            up_chain_size[0] /= self.merge_size
            # number of additional geoms to add
            num_geoms_iter = self.merge_size - 1
        chain_obj = BoxObject(
            name="chain_{}".format(chain_ind),
            size=list(up_chain_size),
            rgba=list(self.wire_box_geom_rgba),
            material=box_geom_material,
            joints=None,
        )
        chain_ind += 1
        objects.append(chain_obj)
        object_locations.append(
            [
                base_x_off - (self.socket_base_size[0] + up_chain_size[0]),
                base_y_off,
                0.0,
            ]
        )
        object_quats.append([1.0, 0.0, 0.0, 0.0])
        object_parents.append(None)
        if self.merge_box_geoms:
            # add ball joint and make sure to place it at the edge of the geom
            ball_joint = dict(ball_joint_spec)
            ball_joint["name"] = "ball_joint_{}".format(chain_ind)
            ball_joint["pos"] = "{} 0 0".format(up_chain_size[0])
            object_joints[chain_obj.root_body] = [ball_joint]

        for i in range(num_geoms_iter):
            chain_obj = BoxObject(
                name="chain_{}".format(chain_ind),
                size=list(up_chain_size),
                rgba=list(self.wire_box_geom_rgba),
                material=box_geom_material,
                joints=None,
            )
            chain_ind += 1
            parent = objects[-1].root_body
            objects.append(chain_obj)
            object_locations.append([-2.0 * up_chain_size[0], 0.0, 0.0])
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(parent)

            # add ball joint and make sure to place it at the edge of the geom
            ball_joint = dict(ball_joint_spec)
            ball_joint["name"] = "ball_joint_{}".format(chain_ind)
            ball_joint["pos"] = "{} 0 0".format(up_chain_size[0])
            object_joints[chain_obj.root_body] = [ball_joint]

        # add chain in left direction
        rot_quat = T.convert_quat(
            T.axisangle2quat(np.array([0.0, 0.0, 1.0]) * (np.pi / 2.0)), to="wxyz"
        )
        left_chain_size = list(self.wire_box_geom_size)
        num_geoms_iter = self.num_box_geoms_left - 1
        if self.merge_box_geoms:
            # only one big geom instead of chain of geoms
            left_chain_size[0] *= self.num_box_geoms_left
            # HACK: merge entire left chain since this side is small
            left_merge_size = 1
            left_chain_size[0] /= left_merge_size
            # number of additional geoms to add
            num_geoms_iter = left_merge_size - 1
        chain_obj = BoxObject(
            name="chain_{}".format(chain_ind),
            size=list(left_chain_size),
            rgba=list(self.wire_box_geom_rgba),
            material=box_geom_material,
            joints=None,
        )
        chain_ind += 1
        parent = objects[-1].root_body
        objects.append(chain_obj)
        object_locations.append([-up_chain_size[0], -left_chain_size[0], 0.0])
        object_quats.append(rot_quat)
        object_parents.append(parent)

        ball_joint = dict(ball_joint_spec)
        ball_joint["name"] = "ball_joint_{}".format(chain_ind)
        ball_joint["pos"] = "{} 0 0".format(left_chain_size[0])
        object_joints[chain_obj.root_body] = [ball_joint]

        for i in range(num_geoms_iter):
            chain_obj = BoxObject(
                name="chain_{}".format(chain_ind),
                size=list(left_chain_size),
                rgba=list(self.wire_box_geom_rgba),
                material=box_geom_material,
                joints=None,
            )
            chain_ind += 1
            parent = objects[-1].root_body
            objects.append(chain_obj)
            object_locations.append([-2.0 * left_chain_size[0], 0.0, 0.0])
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(parent)

            ball_joint = dict(ball_joint_spec)
            ball_joint["name"] = "ball_joint_{}".format(chain_ind)
            ball_joint["pos"] = "{} 0 0".format(left_chain_size[0])
            object_joints[chain_obj.root_body] = [ball_joint]

        # add chain in downward direction
        rot_quat = T.convert_quat(
            T.axisangle2quat(np.array([0.0, 0.0, 1.0]) * (np.pi / 2.0)), to="wxyz"
        )
        down_chain_size = list(self.wire_box_geom_size)
        num_geoms_iter = self.num_box_geoms_down - 1
        if self.merge_box_geoms:
            # only one big geom instead of chain of geoms
            down_chain_size[0] *= self.num_box_geoms_down
            down_chain_size[0] /= self.merge_size
            # number of additional geoms to add
            num_geoms_iter = self.merge_size - 1
        chain_obj = BoxObject(
            name="chain_{}".format(chain_ind),
            size=list(down_chain_size),
            rgba=list(self.wire_box_geom_rgba),
            material=box_geom_material,
            joints=None,
        )
        chain_ind += 1
        parent = objects[-1].root_body
        objects.append(chain_obj)
        object_locations.append([-left_chain_size[0], -down_chain_size[0], 0.0])
        object_quats.append(rot_quat)
        object_parents.append(parent)

        ball_joint = dict(ball_joint_spec)
        ball_joint["name"] = "ball_joint_{}".format(chain_ind)
        ball_joint["pos"] = "{} 0 0".format(down_chain_size[0])
        object_joints[chain_obj.root_body] = [ball_joint]

        for i in range(num_geoms_iter):
            chain_obj = BoxObject(
                name="chain_{}".format(chain_ind),
                size=list(down_chain_size),
                rgba=list(self.wire_box_geom_rgba),
                material=box_geom_material,
                joints=None,
            )
            chain_ind += 1
            parent = objects[-1].root_body
            objects.append(chain_obj)
            object_locations.append([-2.0 * down_chain_size[0], 0.0, 0.0])
            object_quats.append([1.0, 0.0, 0.0, 0.0])
            object_parents.append(parent)

            ball_joint = dict(ball_joint_spec)
            ball_joint["name"] = "ball_joint_{}".format(chain_ind)
            ball_joint["pos"] = "{} 0 0".format(down_chain_size[0])
            object_joints[chain_obj.root_body] = [ball_joint]

        # add cylinder object
        self.cylinder_obj = StackedCylinderObject(
            name="cylinder_obj",
            **self.cylinder_args,
        )

        x_off = -(down_chain_size[0] + self.cylinder_obj.square_base_width)
        y_off = 0.0
        z_off = ((self.cylinder_obj.h1 + self.cylinder_obj.h2) / 2.0) - down_chain_size[2]
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
