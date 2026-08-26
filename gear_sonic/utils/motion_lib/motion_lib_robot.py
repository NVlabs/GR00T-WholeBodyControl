from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase
from gear_sonic.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch


class MotionLibRobot(MotionLibBase):
    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(motion_lib_cfg=motion_lib_cfg, num_envs=num_envs, device=device)
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        return
