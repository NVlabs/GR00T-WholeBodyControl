import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
import mujoco
import mujoco.viewer

from simulation.mujoco_utils import joint_names_to_joint_ids
from simulation.mujoco_utils import joints_to_limits, joints_to_qpos_dof_ids
from simulation.mujoco_utils import get_geoms_from_group, geoms_in_contact
from simulation.mujoco_utils import sample_qpos

# Upper-body planning DOFs for the fixed-base G1 planning model (waist + arms).
JOINT_NAMES_LEFT = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
JOINT_NAMES_RIGHT = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
JOINT_NAMES_BIMANUAL = JOINT_NAMES_LEFT + JOINT_NAMES_RIGHT
JOINT_NAMES_UP = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
] + JOINT_NAMES_BIMANUAL


class MujocoRobot:
    """Generic robot wrapper for MuJoCo model/data access."""

    def __init__(
        self,
        model,
        joint_names,
        root_link,
        data=None,
        collision_geom_group=3,
        ee_names=None,
        visualize=False,
    ):
        """Initialize MujocoRobot"""
        self.model = model
        if data is None:
            self.data = mujoco.MjData(model)
        else:
            self.data = data
        self.joint_names = joint_names
        self.root_link = root_link
        self.viewer = None
        if visualize:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
            self.viewer.sync()

        # Resolve joint ids and corresponding qpos ids
        self.joint_ids = joint_names_to_joint_ids(model, self.joint_names)
        self.joint_qpos_ids, self.joint_dof_ids = joints_to_qpos_dof_ids(
            model, joint_names=self.joint_names
        )
        self.n_joints = len(self.joint_ids)
        # get joint limits
        self.joint_limits = np.array(joints_to_limits(model, self.joint_ids))

        # get robot geoms from subtree
        self.robot_geoms = self.get_robot_geoms(collision_geom_group)

        self.ee_names = ee_names

    def get_joint_qpos(self):
        """Get joint angles"""
        # allow passing in a different data object
        return self.data.qpos[self.joint_qpos_ids].copy()

    def set_joint_qpos(self, q):
        """Set joint angles"""
        # set joint values
        q = np.asarray(q)
        if len(q) != self.n_joints:
            raise ValueError("Expected q length %d" % self.n_joints)
        self.data.qpos[self.joint_qpos_ids] = q

        mujoco.mj_forward(self.model, self.data)

    def ctrl_joint_qpos(self, q):
        """Control joint angles"""
        self.data.ctrl[self.joint_dof_ids] = q
        mujoco.mj_step(self.model, self.data)

    def get_ee_pose(self, idx=0):
        """Get end-effector pose"""
        ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_names[idx]
        )
        pos = self.data.site_xpos[ee_id]
        mat = self.data.site_xmat[ee_id]
        quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(quat, mat)
        return np.concatenate([pos.copy(), quat])

    def in_contact(self, verbose=False):
        """Check if the robot is in contact with the environment"""
        in_contact = geoms_in_contact(
            self.model, self.data, self.robot_geoms, 1e-3, verbose
        )
        return in_contact

    def get_robot_geoms(self, geom_group):
        """Get robot geoms"""
        return get_geoms_from_group(self.model, geom_group, self.root_link)

    def teleport_base(self, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]):
        """Teleport the robot base to the given position and orientation"""
        bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.root_link
        )
        if bid < 0:
            raise ValueError(f"unknown body root: '{self.root_link}'")

        self.model.body_pos[bid] = pos
        self.model.body_quat[bid] = quat
        mujoco.mj_forward(self.model, self.data)

    def in_limits(self, q):
        """Check if a configuration is within joint limits"""
        lo, hi = self.joint_limits
        return np.all(q >= lo) and np.all(q <= hi)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def sample_qpos(self):
        """Sample a random joint configuration"""
        lo, hi = self.joint_limits
        # convert all the -np.inf to -2pi and np.inf to 2pi
        lo = np.where(lo == -np.inf, -2 * np.pi, lo)
        hi = np.where(hi == np.inf, 2 * np.pi, hi)
        return np.random.uniform(lo, hi)


class G1Up(MujocoRobot):
    """Fetch specialization."""

    FINGER = [
        # left hand
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        # right hand
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
    ]
    FINGER_CLOSED = []
    FINGER_OPEN = np.zeros(len(FINGER))
    HOME_POS = np.zeros(len(JOINT_NAMES_UP))
    HOME_POS[[4, 11]] = (0.2, -0.2)
    HOME_POS[[6, 13]] = (1.5708, 1.5708)

    def __init__(self, model, data=None, visualize=False):
        """Initialize FetchRobot"""
        MujocoRobot.__init__(
            self,
            model,
            joint_names=JOINT_NAMES_UP,
            root_link=None,
            data=data,
            collision_geom_group=3,
            ee_names=["left_palm", "right_palm"],
            visualize=visualize,
        )
        # Send to home
        self.set_joint_qpos(self.HOME_POS)
        self.ctrl_joint_qpos(self.HOME_POS)

        # Open the gripper
        for i, finger in enumerate(self.FINGER):
            j_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, finger)
            self.data.qpos[model.jnt_qposadr[j_id]] = self.FINGER_OPEN[i]
            self.data.ctrl[model.jnt_dofadr[j_id]] = self.FINGER_OPEN[i]
        mujoco.mj_forward(model, self.data)


if __name__ == "__main__":
    # Test G1
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    xml = open(os.path.join(curr_dir, "g1_up.xml")).read()
    model = mujoco.MjModel.from_xml_string(xml)
    robot = G1Up(model, visualize=True)

    # test collision checking
    in_contact = geoms_in_contact(model, robot.data, robot.robot_geoms, True)
    print(robot.robot_geoms)
    print("in_contact:", in_contact)
    robot.viewer.sync()

    # Keep the viewer
    try:
        while True:
            time.sleep(0.01)
    except KeyboardInterrupt:
        robot.close()
