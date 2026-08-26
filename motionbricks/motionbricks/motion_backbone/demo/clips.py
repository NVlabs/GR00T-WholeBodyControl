import torch as t
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
import os
from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix, matrix_to_cont6d
from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from motionbricks.helper.mujoco_helper import motion_feature_to_mujoco_qpos
from copy import deepcopy
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
import time
from motionbricks.helper.mujoco_helper import mujoco_qpos_converter
from typing import Union

NUM_FRAMES_PER_TOKEN = 4
time_stamp = time.strftime("%Y%m%d_%H%M%S")

def get_clip_data(clip_id: Union[str, int], train_dataloader: DataLoader):
    """ @brief: get the clip data from the dataloader """
    if type(clip_id) == int:
        clip_data = train_dataloader.dataset[clip_id]['motion']
    else:
        # get the actual data index from the clip_id
        meta_original_path = train_dataloader.dataset.meta['original_path']
        clip_key_id = np.where(meta_original_path.str.endswith("/" + clip_id))[0].item()
        clip_data = train_dataloader.dataset.__getitem__(keyid=clip_key_id)['motion']

    return clip_data

class clip_holder(t.nn.Module):
    """ @brief: hold the clips to use for interactive demo, such as walking clip, running clip, etc.
    """

    def __init__(self, train_dataloader: DataLoader = None, visualize_clips: bool = False, ckpt_path: str = None,
                 converter: mujoco_qpos_converter = None, reprocess_clips: bool = False,
                 val_dataloader: DataLoader = None):
        super(clip_holder, self).__init__()
        self._converter = converter
        if ckpt_path is not None and os.path.exists(ckpt_path) and not reprocess_clips:
            self._preprocess_clips_from_ckpt(ckpt_path)
        elif train_dataloader is not None:
            self._preprocess_clips_from_dataloader(train_dataloader, val_dataloader, visualize_clips, ckpt_path)
        else:
            raise ValueError("Either train_dataloader or ckpt_path must be provided")
        self._apply_root_headings_correction()

    def _preprocess_clips_from_ckpt(self, ckpt_path: str):
        """ @brief: create and load the tensors according to the keys /shapes in the ckpt. """
        state_dict = t.load(ckpt_path)
        # Remap legacy key names
        key_remap = {'mfm_feature': 'motion_feature'}
        for key, value in state_dict.items():
            key = key_remap.get(key, key)
            self.register_buffer(key, value)

    def _preprocess_clips_from_dataloader(self, train_dataloader: DataLoader, val_dataloader: DataLoader,
                                          visualize_clips: bool = False, ckpt_path: str = None):
        """ @brief: preprocess the clips from the dataloader:

            1. figure out which clips are for walking, running, idle
            2. Process them into global motion representation
            3. store into np files and avoid loading them again
        """
        train_dataloader.dataset.motion_sampler.max_seconds = 50.0    # no limit for samples
        train_dataloader.dataset.motion_sampler = None                # disable the sampler now
        train_dataloader.dataset.motion_loader.motion_sampler = None  # disable the sampler here too
        motion_rep = train_dataloader.dataset.motion_rep
        num_joints = motion_rep.skeleton.nbjoints

        max_num_frames, DEFAULT_MAX_NUM_FRAMES = 20, 20
        for clip_name, clip_info in self.CLIPS.items():
            clip_id, start_frame, end_frame = clip_info['clip_id'], clip_info['start_frame'], clip_info['end_frame']
            # clip_data = train_dataloader.dataset[clip_id]['motion'][None, start_frame: end_frame]
            clip_data = get_clip_data(clip_id, train_dataloader)[None, start_frame: end_frame]
            clip_data = motion_rep.change_first_heading(clip_data, 0.0, is_normalized=True, to_normalize=False)
            self.CLIPS[clip_name]['motion_feature'] = clip_data.clone()[0]  # remove batch dim

            # get the mujoco qpos
            device = self.CLIPS[clip_name]['motion_feature'].device
            self.CLIPS[clip_name]['mujoco_qpos'] = \
                self._converter.convert_motion_features_to_mujoco_qpos(self.CLIPS[clip_name]['motion_feature'][None],
                                                                    motion_rep.to(device), False)[0]
            root_rot = self.CLIPS[clip_name]['mujoco_qpos'][:, 3: 7].clone()
            self.CLIPS[clip_name]['mujoco_qpos'][:, 3: 7] = root_rot[:, [3, 0, 1, 2]]
            max_num_frames = max(max_num_frames, self.CLIPS[clip_name]['mujoco_qpos'].shape[0])

            # for it to be used, only 1) global root information (num_frames, 3),
            # 2) global joint positions wrt to root translation (num_frames, num_joints, 3),
            # 3) global joint orientation (num_frames, num_joints, 3, 3)
            global_joint_positions, global_joint_rotations = \
                self._converter.convert_mujoco_qpos_to_motion_transforms(self.CLIPS[clip_name]['mujoco_qpos'][None])
            self.CLIPS[clip_name]['global_root_positions'] = \
                global_joint_positions[0, :, 0] * t.tensor([[1.0, 0.0, 1.0]])
            self.CLIPS[clip_name]['global_joint_positions'] = \
                global_joint_positions[0] - self.CLIPS[clip_name]['global_root_positions'][:, None, :]
            self.CLIPS[clip_name]['global_joint_rotations'] = global_joint_rotations[0]

            # also cache the heading direction
            root_direction = t.matmul(self.CLIPS[clip_name]['global_joint_rotations'][:, 0, :, :],
                                      t.tensor([0.0, 0.0, 1.0]).view([1, -1, 1]))  # y up and z forward
            root_direction = root_direction.view([-1, 3]) * t.tensor([1.0, 0.0, 1.0]).view([1, -1])
            assert (root_direction.norm(dim=1, keepdim=True) > 1e-5).all().item(), \
                f"Clip with ill defined heading found at clip_id {clip_id}"
            root_direction = root_direction / t.norm(root_direction, dim=1, keepdim=True)
            self.CLIPS[clip_name]['global_headings'] = t.atan2(root_direction[:, 0], root_direction[:, 2])

        # register the clips as parameters so that it will be later used in onnx / trt model
        motion_feature_shape = self.CLIPS[list(self.CLIPS.keys())[0]]['motion_feature'].shape[-1]
        for data_buffer_name, feat_shape in zip(['global_root_positions', 'global_joint_positions',
                                                 'global_joint_rotations', 'global_headings',
                                                 'motion_feature', 'mujoco_qpos'],
                                                [[3], [num_joints, 3], [num_joints, 3, 3], [],
                                                 [motion_feature_shape], [36]]):
            data_buffer = t.zeros([len(self.CLIPS), max_num_frames, *feat_shape])
            num_frames_per_clip = t.zeros([len(self.CLIPS)], dtype=t.int32)
            for clip_idx, clip_name in enumerate(self.CLIPS):
                clip_length = self.CLIPS[clip_name][data_buffer_name].shape[0]
                num_frames_per_clip[clip_idx] = clip_length
                data_buffer[clip_idx, :clip_length] = self.CLIPS[clip_name][data_buffer_name]
            self.register_buffer(data_buffer_name, data_buffer)
        self.register_buffer('num_frames_per_clip', num_frames_per_clip)
        t.save(self.state_dict(), ckpt_path)

    def _apply_root_headings_correction(self):
        """ @brief: apply the root headings correction to the clips; since cetain root project is ill defined."""
        pass

class clip_holder_G1(clip_holder):
    LOAD_FROM_CLIP_NAME = True
    CLIPS = {
        "idle": {
            "clip_id": 'neutral_idle_loop_001__A076',
            "start_frame": 0, "end_frame": 30, 'avg_root_vel': 0.0,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },
        "slow_walk": {
            "clip_id": 'neutral_idle_loop_001__A076',
            "start_frame": 0, "end_frame": 30, 'avg_root_vel': 0.3 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },
        "walk": {
            "clip_id": 'neutral_idle_loop_001__A076',
            "start_frame": 0, "end_frame": 30, 'avg_root_vel': 1.0 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },
        "hand_crawling": {
            "clip_id": "mohak_backward_stop_001__A031",
            "start_frame": 0, "end_frame": 30, 'avg_root_vel': 0.5 * 2.0,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },  # crawling
        "walk_boxing": {
            "clip_id": "shadow_boxing_R_003__A360_M",
            "start_frame": 25, "end_frame": 35, 'avg_root_vel': 1.0 * 2.0,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        },
        "elbow_crawling": {
            "clip_id": "crawl_ff_loop_270_001__A130_M",
            "start_frame": 13, "end_frame": 18, 'avg_root_vel': 0.8 * 2.0,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        },  # crawling
        "stealth_walk": {
            "clip_id": "stealth_ff_start_360_001__A125",
            "start_frame": 50, "end_frame": 70, 'avg_root_vel': 1.0 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]},
        "injured_walk": {
            "clip_id": "dancecards1_AB_injured_L_leg_001__A005",
            "start_frame": 211, "end_frame": 219, 'avg_root_vel': 0.5 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },
        "walk_stealth": {
            "clip_id": "crouch_ff_loop_180_R_001__A196",
            "start_frame": 50, "end_frame": 70, 'avg_root_vel': 0.7 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },
        "walk_happy_dance": {
            "clip_id": "dance_vouge_vogue_sequence_180_R_002__A316",
            "start_frame": 30, "end_frame": 50, 'avg_root_vel': 1.0 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },
        "walk_zombie": {
            "clip_id": "zombie_walk_180_R_003__A330",
            "start_frame": 10, "end_frame": 100, 'avg_root_vel': 0.6 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },
        "walk_gun": {
            "clip_id": "angry_gun_walk_ff_loop_090_R_001__A393_M",
            "start_frame": 10, "end_frame": 100, 'avg_root_vel': 0.6 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },
        "walk_scared": {
            "clip_id": "scared_walk_ff_start_225_R_002__A423",
            "start_frame": 10, "end_frame": 100, 'avg_root_vel': 0.6 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        },

        "walk_left": {
            "clip_id": 'dance_sakuras_victory_sway_001__A464',
            "start_frame": 35, "end_frame": 40, 'avg_root_vel': 0.2 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        },  # for robot deployment safety; can be turned off
        "walk_right": {
            "clip_id": 'dance_sakuras_victory_sway_001__A464_M',
            "start_frame": 35, "end_frame": 40, 'avg_root_vel': 0.2 * 2,
            'allowed_pred_num_tokens': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
        },  # for robot deployment safety; can be turned off
    }  # the actual velocity is 0.5 of the `avg_root_vel` because of the spring model

    DEFAULT_KEYS = {
        "idle": "",
        "slow_walk": "v",
        "walk": "",
        "hand_crawling": "z",
        "walk_boxing": "x",
        "elbow_crawling": "b",
        "stealth_walk": "r",
        "injured_walk": "t",
        "walk_stealth": "c",
        "walk_happy_dance": "e",
        "walk_zombie": "f",
        "walk_gun": "g",
        "walk_scared": "q",
        "walk_left": "",
        "walk_right": "",
    }

    def _apply_root_headings_correction(self):
        """ @brief: apply the root headings correction to the clips; since certain root projection is ill defined."""
        hand_crawling_id = list(self.CLIPS.keys()).index('hand_crawling')
        elbow_crawling_id = list(self.CLIPS.keys()).index('elbow_crawling')
        self.global_headings[hand_crawling_id, 0] = 0.0  # the crawling pose's root heading is ill defined
        self.global_headings[elbow_crawling_id, :] -= 0.95  # the elbow crawling pose's root heading is ill defined

    def blendspace_modes_remap_from_velocity(self, mode: t.Tensor,
                                             target_movement_direction: t.Tensor, target_heading: t.Tensor):
        """ @brief: This is the getto version of 2D blend space
        if no velocity, don't swap
        if velocity angle > heading angle, positive, swap to the right
        if velocity angle < heading angle, negative, swap to the left

        This is helpful for the robot deployment safety, but can be removed as well.
        """
        # note target_movement_direction is in mujoco space, but facing_direction was in the motion space;
        # thus the atan2 is not the same as the target_heading is calculated
        movement_heading = t.atan2(target_movement_direction[:, 0], target_movement_direction[:, 1])
        indices_of_slow_walk = list(self.CLIPS.keys()).index('slow_walk')
        indices_of_walk = list(self.CLIPS.keys()).index('walk')

        indices_of_walk_left = list(self.CLIPS.keys()).index('walk_left')
        indices_of_walk_right = list(self.CLIPS.keys()).index('walk_right')

        heading_diff = (movement_heading - target_heading + t.pi) % (2 * t.pi) - t.pi  # > 0 left, < 0 right
        if heading_diff.item() < -1.0 and mode.item() == indices_of_slow_walk:
            pass

        going_right = t.logical_and(heading_diff < -t.pi / 4.0, heading_diff > -3 * t.pi / 4.0)
        going_left = t.logical_and(heading_diff > t.pi / 4.0, heading_diff < 3 * t.pi / 4.0)
        is_slow_walk_or_walk = t.logical_or(mode == indices_of_slow_walk, mode == indices_of_walk)

        mode = mode * t.logical_or(t.logical_and(~going_right, ~going_left), ~is_slow_walk_or_walk) + \
            t.full_like(mode, indices_of_walk_right) * t.logical_and(is_slow_walk_or_walk, going_right) + \
            t.full_like(mode, indices_of_walk_left) * t.logical_and(is_slow_walk_or_walk, going_left)
        return mode
