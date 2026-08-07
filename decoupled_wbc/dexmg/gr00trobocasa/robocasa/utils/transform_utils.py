from robosuite.utils.transform_utils import *


def unmake_pose(pose):
    """
    Split homogenous pose matrices back into translation vectors and rotation matrices.

    Args:
        pose (np.array): batch of pose matrices with last 2 dimensions of (4, 4)

    Returns:
        pos (np.array): batch of position vectors with last dimension of 3
        rot (np.array): batch of rotation matrices with last 2 dimensions of (3, 3)
    """
    return pose[..., :3, 3], pose[..., :3, :3]


def standardize_quat(quaternions):
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last (e.g. xyzw),
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    # if isinstance(quaternions, torch.Tensor):
    #     return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)
    return np.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def pose_in_world_to_pose_in_ref(pos_in_world, rot_in_world, ref_pos, ref_rot):
    """
    Takes a pose in world frame and a reference pose (in world frame) and
    transforms the pose to be with respect to the reference frame.
    """

    # Let O be the frame of the item of interest, R be the reference frame,
    # and W be the world frame. Then,
    #
    #   T^O_R = (T^R_W)^-1 T^O_W
    pose_in_world = make_pose(pos_in_world, rot_in_world)
    ref_pose = make_pose(ref_pos, ref_rot)
    world_in_ref = pose_inv(ref_pose)
    return np.matmul(world_in_ref, pose_in_world)


def extract_top_down_angle(rotation_matrix):
    """
    Gets top-down angle about z-axis for a given 3D rotation matrix, assuming
    that the rotation matrix corresponds to a simple rotation about z-axis.
    """
    return np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
