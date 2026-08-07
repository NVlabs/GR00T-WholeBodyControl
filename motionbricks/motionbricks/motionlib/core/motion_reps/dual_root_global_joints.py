from .motion_reps_base.dual_root_local_body import DualRootLocalBody
from .motion_reps_base.global_root_local_body import GlobalRootLocalBody
from .motion_reps_base.local_root_local_body import LocalRootLocalBody

COMPUTE_KWARGS = {
    "local_vel_without_root": False,
    "local_root_vel_with_y": False,
    "compute_heading_method": "hips_pos",  # to compute the root heading
    "removing_heading": False,
}

DEFAULT_JOINT_POSITIONS_FROM = "global_rot_data"


class DualRootGlobalJoints(DualRootLocalBody):
    """Motion representation with global rotation but without removing the heading."""

    _name_ = "dual_root_global_joints"
    compute_kwargs = COMPUTE_KWARGS
    default_joint_positions_from = DEFAULT_JOINT_POSITIONS_FROM

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            global_class=GlobalRootGlobalJoints,
            local_class=LocalRootGlobalJoints,
            **kwargs,
        )


def get_body_keys_dim(self, nbjoints: int):
    # as removing heading is set to False, we does not remove the y rotation to any data in both root and body
    return {
        "ric_data": [
            (nbjoints - 1) * 3
        ],  # xyz without the root: careful it is actually not rotation invariant
        "global_rot_data": [
            nbjoints * 6
        ],  # 6D rot with root (y rotation is not removed)
        "local_vel": [nbjoints * 3],  # local_vel xyz (with root)
        "foot_contacts": [4],  # Left: Foot + Toe / Right: Foot + Toe
    }


class GlobalRootGlobalJoints(GlobalRootLocalBody):
    """Motion representation with global root."""

    dual_class = DualRootGlobalJoints
    get_body_keys_dim = get_body_keys_dim
    compute_kwargs = COMPUTE_KWARGS
    default_joint_positions_from = DEFAULT_JOINT_POSITIONS_FROM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LocalRootGlobalJoints(LocalRootLocalBody):
    """Motion representation with local root."""

    dual_class = DualRootGlobalJoints
    get_body_keys_dim = get_body_keys_dim
    compute_kwargs = COMPUTE_KWARGS
    default_joint_positions_from = DEFAULT_JOINT_POSITIONS_FROM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
