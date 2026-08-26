from motionbricks.motion_backbone.inference.motion_inference import motion_inference
from copy import deepcopy
import torch as t
from torch.utils.data import DataLoader
from motionbricks.motion_backbone.demo.clips import clip_holder_G1
from motionbricks.helper.mujoco_helper import get_mujoco_converter
import time
from scipy.spatial.transform import Rotation as R
from motionbricks.motionlib.core.utils.rotations import angle_to_Y_rotation_matrix, matrix_to_cont6d, quat_apply, quat_mul
from motionbricks.motionlib.core.utils.rotations import quaternion_to_matrix

# using this matrix_to_quaternion instead of the one in motionbricks.motionlib.core.utils.rotations
# to avoid some tensorrt issues
from motionbricks.geometry.quaternions import matrix_to_quaternion

import os

def angle_to_Z_rotation_matrix(angle):
    """Create rotation matrix around Z-axis for Z-up coordinate system"""
    cos, sin = t.cos(angle), t.sin(angle)
    one, zero = t.ones_like(angle), t.zeros_like(angle)
    # Z-axis rotation matrix:
    # [cos(θ) -sin(θ)  0]
    # [sin(θ)  cos(θ)  0]
    # [  0       0     1]
    mat = t.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), -1)
    mat = mat.reshape(angle.shape + (3, 3))
    return mat

class full_navigation_agent(t.nn.Module):
    """ @brief: this is the agent class which handles everything.
    """
    DEFAULT_PLANING_HORIZON = 1.0  # 1 second
    NUM_FRAMES_PER_TOKEN = 4
    DEFAULT_PRED_OFFSETS = 4

    def __init__(self, inferencer: motion_inference, train_dataloader: DataLoader, device: str = 'cuda',
                 speed_scale: list[float] = [1.0, 1.0], target_root_realignment: bool = True,
                 source_root_realignment: bool = True,
                 pred_offsets: int = DEFAULT_PRED_OFFSETS,
                 skeleton_xml: str = "assets/skeletons/g1/g1.xml",
                 filter_qpos: bool = True,
                 force_canonicalization: bool = True, clips: str = "G1",
                 bypass_spring_model: bool = False,
                 skip_ending_target_cond: bool = True,
                 ckpt_path: str = None,
                 reprocess_clips: bool = False,
                 val_dataloader: DataLoader = None,
                 use_spring_root_instead: bool = False):
        super(full_navigation_agent, self).__init__()
        self._inferencer = inferencer.eval().to(device)
        self._motion_rep = deepcopy(inferencer.motion_rep).to(device)  # make a copy to avoid gpu cpu transfer
        self._converter = get_mujoco_converter(self._motion_rep, skeleton_xml).to(device)
        self._clip_holder = clip_holder_G1(train_dataloader=train_dataloader, ckpt_path=ckpt_path,
                                                converter=self._converter, reprocess_clips=reprocess_clips,
                                                val_dataloader=val_dataloader)
        self._train_dataloader = train_dataloader
        self._device = device
        self._fps = self._motion_rep.fps
        self._target_root_realignment = target_root_realignment
        self._source_root_realignment = source_root_realignment

        self.PRED_OFFSETS = pred_offsets
        self.FILTER_QPOS = filter_qpos
        self.FORCE_CANONICALIZATION = force_canonicalization
        self.BYPASS_SPRING_MODEL = bypass_spring_model
        self.SKIP_ENDING_TARGET_COND = skip_ending_target_cond

        self.frames = {
            # model features are the output from the model inference. The actual inference runs here
            "model_features": None,  # [batch_size, num_frames, feature_dim (390)]

            # qpos feature is the feature understood by the mujoco simulator
            "mujoco_qpos": None,   # [batch_size, num_frames, 32]
            "mode": None,
        }
        self._speed_scale_min, self._speed_scale_max = speed_scale  # in case you want to perturb the speed
        self._initialize_frames()
        self._has_prebaked_inference_engine = False

    def set_prebaked_inference_engine(self):
        raise NotImplementedError("Prebaked inference engine is not implemented yet")

    def reset(self):
        self._current_frame_idx = 0
        self._initialize_frames()

    def _initialize_frames(self):

        # the initial frames
        self._current_frame_idx = 0
        # self.frames['model_features'] = self._clip_holder.CLIPS['idle']['motion_feature']  # unnormalized, [batch, F, D=390]
        # fetch the motion features for the idle clip
        self.frames['model_features'] = self._clip_holder.motion_feature[0, :self._clip_holder.num_frames_per_clip[0]][None]

        self.frames['mujoco_qpos'] = self._converter.convert_motion_features_to_mujoco_qpos(
            self.frames['model_features'], self._motion_rep.to(self.frames['model_features'].device), False
        )
        root_rot = self.frames['mujoco_qpos'][:, :, 3: 7].clone()
        self.frames['mujoco_qpos'][:, :, 3: 7] = root_rot[:, :, [3, 0, 1, 2]]
        NUM_MIN_FRAMES_IN_BUFFER = 64
        if self.frames['mujoco_qpos'].shape[1] < NUM_MIN_FRAMES_IN_BUFFER:
            self.frames['mujoco_qpos'] = t.cat(
                [self.frames['mujoco_qpos'],
                 self.frames['mujoco_qpos'][:, -1:].repeat(1, NUM_MIN_FRAMES_IN_BUFFER -
                                                           self.frames['mujoco_qpos'].shape[1], 1)], dim=1
            )

    def generate_new_frames(self, input: dict, controller_dt: float = 0.25, force_generation: bool = False):
        """ @brief: call the model inference to generate the new frames.

            input:
                context_global_joint_positions: [batch_size, num_frames = 4, num_joints (including root), 3]
                context_global_joint_rotations: [batch_size, num_frames = 4, num_joints (including root), 3, 3]

                mode: [batch] int which corresponds to ['walk', 'run', 'idle', 'slow_walk']

                movement_direction: [batch_size, 3] in mujoco coordinate system (global)
                facing_direction: [batch_size, 3] in mujoco coordinate system (global)

        """
        if not force_generation:
            if self._current_frame_idx < controller_dt * self._fps and \
                    self._current_frame_idx < self.frames['mujoco_qpos'].shape[1] - 1:  # replan frequency
                return self.frames['model_features'], self.frames['mujoco_qpos']

            if not self._should_regenerate(input):
                return self.frames['model_features'], self.frames['mujoco_qpos']

        self._prev_input = input.copy()
        if 'specific_target_positions' in input and 'has_specific_target' not in input:
            input['has_specific_target'] = t.tensor([[True]]).int()  # compatibility if not provided
        input = {i: input[i].to(self._device) for i in input if i}

        if self._has_prebaked_inference_engine:
            raise NotImplementedError("Prebaked inference engine is not implemented yet")
        else:

            input['context_global_joint_positions'], input['context_global_joint_rotations'] = \
                self._process_input_to_joint_transforms(input)

            # use the spring model to generate the realistic target global root position and heading
            input['target_root_position'], input['target_root_positions'], \
                input['target_root_headings'], input['target_root_heading'], \
                    input['start_root_positions'], input['start_root_headings'] = \
                self._generate_spring_model_position_and_heading(input)

            if 'has_specific_target' in input and self.BYPASS_SPRING_MODEL:
                self._override_target_transforms(input)

            # construct the target joint transforms for the model inference
            input['target_global_joint_positions'], input['target_global_joint_rotations'], \
                input['target_global_root_positions'] = self._generate_target_joint_transforms(input)

            # the inference
            model_features, mujoco_qpos, num_pred_frames = self._generate_inbetween_frames(input)

        self.frames['mode'] = input['mode']

        self.frames['mujoco_qpos_notrunc'] = mujoco_qpos

        # truncate the frames; if using onnx model in C++, you should also manually truncate the frames
        self.frames['model_features'] = model_features[:, :num_pred_frames.item(), :]
        self.frames['mujoco_qpos'] = mujoco_qpos[:, :num_pred_frames.item(), :]

        return self.frames['mujoco_qpos'], num_pred_frames

    def _process_input_to_joint_transforms(self, input: dict):
        """ @brief: process the input to joint transforms
        """
        if 'context_mujoco_qpos' in input:  # should always use this if possible
            if self.FORCE_CANONICALIZATION:
                self._canonicalize_mujoco_qpos(input)
            else:
                input['raw_context_mujoco_qpos'] = input['context_mujoco_qpos'].clone()

            context_global_joint_positions, context_global_joint_rotations = \
                self._converter.convert_mujoco_qpos_to_motion_transforms(input['context_mujoco_qpos'])
        elif 'context_global_joint_positions' in input and 'context_global_joint_rotations' in input:
            # this is the expected input for onnx / trt model
            assert not self.FORCE_CANONICALIZATION, "not implemented yet."
            context_global_joint_positions = input['context_global_joint_positions']
            context_global_joint_rotations = input['context_global_joint_rotations']
        elif 'context_motion_features' in input:
            assert not self.FORCE_CANONICALIZATION, "not implemented yet."
            output_results = self._motion_rep.inverse(input['context_motion_features'],
                                                      is_normalized=False, return_quat=False, return_all=False)
            context_global_joint_positions, context_global_joint_rotations = \
                output_results['posed_joints'], output_results['global_joint_rots']  # [batch, numF, numJ, ...]
        else:
            raise ValueError("Invalid input: context_global_joint_positions or motion_features not found")

        return context_global_joint_positions, context_global_joint_rotations

    def _generate_spring_model_position_and_heading(self, input: dict):
        """ @brief: do the critical damping spring model for the position and heading
        """
        batch_size, device = input['context_global_joint_positions'].shape[0], input['mode'].device

        # default parameters and helper functions for the spring model
        ln2, eps = 0.69314718056, 1e-5
        def fast_neg_exp_func(x):
            return 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x *x)

        # generate position from the spring model
        root_joint_idx = 0
        curr_root_pos = input['context_global_joint_positions'][:, 0, root_joint_idx, [0, 2]]
        curr_root_vel = (input['context_global_joint_positions'][:, 1, root_joint_idx, [0, 2]] -
                         input['context_global_joint_positions'][:, 0, root_joint_idx, [0, 2]]) * self._fps

        input['mode'] = t.min(input['mode'],
                              t.tensor([len(self._clip_holder.CLIPS) - 1]).to(device).int())  # safety check

        translation_movement_in_1s = (input['mode'] == 0) * curr_root_vel.norm(dim=-1, keepdim=False) / 2.0
        for i in range(1, len(self._clip_holder.CLIPS)):
            translation_movement_in_1s += (input['mode'] == i) * \
                self._clip_holder.CLIPS[list(self._clip_holder.CLIPS.keys())[i]]['avg_root_vel']

        # add perturbation to the speed
        random_seed = input.get('random_seed', t.randint(0, 10000, (1,))).to(self._device)  # map this to float
        random_ratio = (random_seed.float() % 100) / 100.0  # [0, 1]
        translation_movement_in_1s *= \
            (random_ratio * (self._speed_scale_max - self._speed_scale_min) + self._speed_scale_min)
        translation_movement_in_1s = (translation_movement_in_1s > 0.1).float() * translation_movement_in_1s

        # enforce the target velocity
        target_vel = (input['mode'] != 0) * \
            input.get('target_vel', -1.0) * 2.0  # 2.0 since the actual speed is usually 0.5 of the target speed
        if type(target_vel) == t.Tensor:
            target_vel = target_vel.view([batch_size, 1])  # a float tensor
        translation_movement_in_1s = (target_vel <= 0.0) * translation_movement_in_1s + \
            (target_vel > 0.0) * target_vel
        target_movement_direction = input['movement_direction'][:, [1, 0]]

        # movement for inplace turning
        target_movement_direction = (target_movement_direction.norm(dim=-1, keepdim=False) > 1e-5) * \
            target_movement_direction + (target_movement_direction.norm(dim=-1, keepdim=False) <= 1e-5) * \
            input['facing_direction'][:, [1, 0]] * 0.1
        target_root_pos = curr_root_pos + translation_movement_in_1s * target_movement_direction

        if 'specific_target_positions' in input:
            has_specific_target = input['has_specific_target']
            target_root_pos = target_root_pos * (1.0 - has_specific_target.float()) + \
                input['specific_target_positions'][:, -1, [1, 0]] * has_specific_target.float()

        y = (4.0 * ln2) / (0.8 + eps) / 2.0  # a typical halflife = 0.4 for the positions; 0.6 for slower changes
        dts = t.cat([(t.arange(self.NUM_FRAMES_PER_TOKEN).float() * 1.0 / self._fps).to(self._device),
                     (1.0 + t.arange(self.NUM_FRAMES_PER_TOKEN).float() * 1.0 / self._fps).to(self._device)], dim=-1)
        # dts = t.arange(self.NUM_FRAMES_PER_TOKEN + self._fps).float() * 1.0 / self._fps
        dts = dts[None, None, :]  # [b, F, DTs]
        eydt = fast_neg_exp_func(y * dts)
        j0 = curr_root_pos - target_root_pos
        j1 = curr_root_vel + j0 * y
        root_positions = (j0[:, :, None] + j1[:, :, None] * dts) * eydt + target_root_pos[:, :, None]
        start_root_positions = root_positions[:, :, :self.NUM_FRAMES_PER_TOKEN]
        target_root_positions = root_positions[:, :, -self.NUM_FRAMES_PER_TOKEN:]

        target_root_positions = (input['mode'][..., None] == 0) * target_root_positions[:, :, :1] + \
            (input['mode'][..., None] != 0) * target_root_positions
        target_root_position = target_root_positions[:, :, 0]

        # generate heading from the spring model
        curr_heading = t.atan2(input['context_global_joint_rotations'][:, 0, root_joint_idx, 0, 2],
                               input['context_global_joint_rotations'][:, 0, root_joint_idx, 2, 2])  # y-axis rotation
        next_heading = t.atan2(input['context_global_joint_rotations'][:, 1, root_joint_idx, 0, 2],
                               input['context_global_joint_rotations'][:, 1, root_joint_idx, 2, 2])
        curr_heading_vel = ((next_heading - curr_heading + t.pi) % (2 * t.pi) - t.pi) * self._fps
        target_heading = t.atan2(input['facing_direction'][:, 1], input['facing_direction'][:, 0])  # mujoco coordinate
        if 'specific_target_headings' in input:
            # make it in the range of [-pi, pi]
            specific_target_heading = (input['specific_target_headings'][:, -1] + t.pi) % (2 * t.pi) - t.pi
            target_heading = specific_target_heading * input['has_specific_target'].view([batch_size]).float() + \
                target_heading * (1.0 - input['has_specific_target'].view([batch_size]).float())

        target_heading[target_heading.isnan()] = 0.0
        target_heading = target_heading + 2 * t.pi * (curr_heading - target_heading > t.pi) \
            -2 * t.pi * (curr_heading - target_heading < -t.pi)

        # use halflife = 0.17 for the heading
        y = (4.0 * ln2) / (0.17 + eps) / 2.0
        # dts = 1.0
        eydt = fast_neg_exp_func(y * dts)
        j0 = curr_heading - target_heading
        j1 = curr_heading_vel + j0 * y
        headings = (j0 + j1 * dts) * eydt + target_heading  # [batch, 1, 4]
        start_headings = headings[:, :, :self.NUM_FRAMES_PER_TOKEN].view([batch_size, -1])
        target_headings = headings[:, :, -self.NUM_FRAMES_PER_TOKEN:].view([batch_size, -1])
        target_heading = target_headings[:, 0]

        if self.FORCE_CANONICALIZATION:
            input['mode'] = self._clip_holder.blendspace_modes_remap_from_velocity(
                input['mode'], target_movement_direction, target_heading
            )

        return target_root_position, target_root_positions, target_headings, target_heading, \
            start_root_positions, start_headings

    def _override_target_transforms(self, input: dict):
        """ @brief: override the target transforms
        """
        if 'specific_target_positions' not in input or 'specific_target_headings' not in input:
            return

        # root positions
        specific_target_root_positions = input['target_root_positions'].clone()
        specific_target_root_positions[:, 0, :] = input['specific_target_positions'][:, :, 1]
        specific_target_root_positions[:, 1, :] = input['specific_target_positions'][:, :, 0]
        input['target_root_positions'] = \
            specific_target_root_positions * input['has_specific_target'][:, None, :].float() + \
            input['target_root_positions'] * (1.0 - input['has_specific_target'][:, None, :].float())
        input['target_root_position'] = input['target_root_positions'][:, :, 0]

        # root headings
        input['target_root_headings'] = \
            input['specific_target_headings'].reshape(input['target_root_headings'].shape) * \
            input['has_specific_target'].float() + \
            input['target_root_headings'] * (1.0 - input['has_specific_target'].float())
        input['target_root_heading'] = input['target_root_headings'][:, 0]
        return input

    def _generate_target_joint_transforms(self, input: dict):
        """ @brief: generate the target joint transforms
        """
        # based on the mode and random seeds, fetch the target poses
        NUM_FRAMES_PER_TOKEN, batch_size = self.NUM_FRAMES_PER_TOKEN, input['mode'].shape[0]
        random_seed = input.get('random_seed', t.randint(0, 10000, (1,))).to(self._device)

        onehot_mode = t.nn.functional.one_hot(input['mode'].view(-1), len(self._clip_holder.CLIPS))
        num_frames_per_clip = (self._clip_holder.num_frames_per_clip[None] * onehot_mode).sum(dim=1)
        frame_idx = random_seed % (num_frames_per_clip - NUM_FRAMES_PER_TOKEN)

        chunks = t.arange(NUM_FRAMES_PER_TOKEN).to(self._device) + frame_idx

        global_root_positions = self._clip_holder.global_root_positions[None,:, chunks]
        global_joint_positions = self._clip_holder.global_joint_positions[None,:, chunks]
        global_joint_rotations = self._clip_holder.global_joint_rotations[None,:, chunks]
        global_headings = self._clip_holder.global_headings[None,:, chunks]
        qpos = self._clip_holder.mujoco_qpos[None,:, chunks]

        global_root_positions = (global_root_positions * onehot_mode[:, :, None, None]).sum(dim=1)
        global_joint_positions = (global_joint_positions * onehot_mode[:, :, None, None, None]).sum(dim=1)
        global_joint_rotations = (global_joint_rotations * onehot_mode[:, :, None, None, None, None]).sum(dim=1)
        global_headings = (global_headings * onehot_mode[:, :, None]).sum(dim=1)
        qpos = (qpos * onehot_mode[:, :, None, None]).sum(dim=1)

        # rotate the orientation to the target heading
        if self._target_root_realignment:
            # set the target rotations based on the target headings

            diff_heading = (input['target_root_headings'] -
                            global_headings + t.pi) % (2 * t.pi) - t.pi  # [-pi, pi]
            corrective_mat = angle_to_Y_rotation_matrix(diff_heading).float()  # [batch, numF, 3, 3]
            global_headings = global_headings + diff_heading
            global_joint_rotations = t.matmul(corrective_mat[:, :, None], global_joint_rotations)
            global_joint_positions = \
                t.matmul(corrective_mat[:, :, None], global_joint_positions[:, :, :, :, None])[..., 0]

            # move the target positions based on the momentum of the spring model
            global_root_positions[:, :, [0, 2]] = input['target_root_positions'].transpose(1, 2).float()
        else:

            diff_heading = (input['target_root_heading'] -
                            global_headings[:, 0] + t.pi) % (2 * t.pi) - t.pi  # [-pi, pi]
            corrective_mat = angle_to_Y_rotation_matrix(diff_heading).float()
            global_headings = global_headings + diff_heading
            global_joint_rotations = t.matmul(corrective_mat[:, None, None, :, :], global_joint_rotations)
            global_joint_positions = \
                t.matmul(corrective_mat[:, None, None, :, :], global_joint_positions[:, :, :, :, None])[..., 0]

            # recenter the target positions and rotate the rest of the root positions
            global_root_positions = \
                t.matmul(corrective_mat[: ,None, :, :], global_root_positions[:, :, :, None])[..., 0]
            global_root_positions = global_root_positions - global_root_positions[:, :1, :]
            global_root_positions[:, :, 0] += input['target_root_position'][:, 0]
            global_root_positions[:, :, 2] += input['target_root_position'][:, 1]

        if self._source_root_realignment:
            context_headings = t.atan2(input['context_global_joint_rotations'][:, :, 0, 0, 2],
                                       input['context_global_joint_rotations'][:, :, 0, 2, 2])  # y-axis rotation
            corrective_mat = angle_to_Y_rotation_matrix(input['start_root_headings'] - context_headings).float()
            input['context_global_joint_rotations'] = \
                t.matmul(corrective_mat[:, :, None, :, :], input['context_global_joint_rotations'])
            input['context_global_joint_positions'] = \
                t.matmul(corrective_mat[:, :, None, :, :],
                         input['context_global_joint_positions'][:, :, :, :, None])[..., 0]
            input['context_global_joint_positions'][:, :, :, [0, 2]] = \
                input['context_global_joint_positions'][:, :, :, [0, 2]] - \
                input['context_global_joint_positions'][:, :, :1, [0, 2]] + \
                input['start_root_positions'].transpose(1, 2)[:, :, None, :].float()

        return global_joint_positions, global_joint_rotations, global_root_positions

    def _generate_inbetween_frames(self, input: dict):
        start_time = time.time()
        batch_size, MASKED_NUM_TOKENS = 1, self._inferencer._root_model.backbone_net.MASKED_NUM_TOKENS
        fps = self._inferencer.local_motion_rep.fps
        root_joint_idx = 0

        # prepare the values for the context frames
        context_global_root_pos = input['context_global_joint_positions'][:, :, root_joint_idx, :]
        context_rotation_angle = t.atan2(input['context_global_joint_rotations'][:, :, root_joint_idx, 0, 2],
                                         input['context_global_joint_rotations'][:, :, root_joint_idx, 2, 2])
        context_global_root_values = t.cat([context_global_root_pos, t.cos(context_rotation_angle)[..., None],
                                            t.sin(context_rotation_angle)[..., None]], dim=-1)  # [B, numF, 5]
        context_local_root_values = \
            t.zeros([batch_size, self.NUM_FRAMES_PER_TOKEN, 4]).to(self._device)  # [B, numF, 4]
        context_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 0] = \
            (((context_rotation_angle[:, 1:] - context_rotation_angle[:, :-1] + t.pi) % (2 * t.pi)) - t.pi) * fps
        context_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 1: 3] = \
            (context_global_root_pos[:, 1:, [0, 2]] - context_global_root_pos[:, :-1, [0, 2]]) * fps
        context_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 3] = \
            context_global_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 1]

        context_global_joint_positions = input['context_global_joint_positions'].clone()
        joint_positions = context_global_joint_positions[:, :, 1:, :]
        joint_positions[..., 0] = \
            context_global_joint_positions[:, :, 1:, 0] - context_global_joint_positions[:, :, :1, 0]
        joint_positions[..., 2] = \
            context_global_joint_positions[:, :, 1:, 2] - context_global_joint_positions[:, :, :1, 2]

        joint_rotation_ortho6d = matrix_to_cont6d(input['context_global_joint_rotations'])
        context_local_poses = t.cat([joint_positions.view([batch_size, self.NUM_FRAMES_PER_TOKEN, -1]),
                                     joint_rotation_ortho6d.view([batch_size, self.NUM_FRAMES_PER_TOKEN, -1])], dim=-1)

        # prepare the values for the target frames
        target_global_root_pos = input['target_global_root_positions'] + \
            input['target_global_joint_positions'][:, :, root_joint_idx, :]
        target_rotation_angle = t.atan2(input['target_global_joint_rotations'][:, :, root_joint_idx, 0, 2],
                                        input['target_global_joint_rotations'][:, :, root_joint_idx, 2, 2])
        if 'target_root_headings' in input:
            target_rotation_angle = input['target_root_headings']  # avoid double counting AND avoid ill defined angles
        target_rotation_angle = target_rotation_angle.float()

        target_global_root_values = t.cat([target_global_root_pos, t.cos(target_rotation_angle)[..., None],
                                           t.sin(target_rotation_angle)[..., None]], dim=-1)  # [B, num_frames, 5]
        target_local_root_values = t.zeros_like(context_local_root_values).to(self._device)  # [b=1, num_frames, 4]
        target_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 0] = \
            (((target_rotation_angle[:, 1:] - target_rotation_angle[:, :-1] + t.pi) % (2 * t.pi)) - t.pi) * fps
        target_local_root_values[:, :self.NUM_FRAMES_PER_TOKEN - 1, 1: 3] = \
            (target_global_root_pos[:, 1:, [0, 2]] - target_global_root_pos[:, :-1, [0, 2]]) * fps
        target_local_root_values[:, -1, 0: 3] = target_local_root_values[:, -2, 0: 3]  # add the last velocity
        target_local_root_values[:, :, 3] = target_global_root_values[:, :, 1]

        joint_positions = input['target_global_joint_positions'][:, :, 1:, :]
        joint_rotation_ortho6d = matrix_to_cont6d(input['target_global_joint_rotations'])
        target_local_poses = t.cat([joint_positions.view([batch_size, self.NUM_FRAMES_PER_TOKEN, -1]),
                                    joint_rotation_ortho6d.view([batch_size, self.NUM_FRAMES_PER_TOKEN, -1])], dim=-1)

        # merge the constraints
        local_root_values = t.cat([context_local_root_values, target_local_root_values], dim=1)
        global_root_values = t.cat([context_global_root_values, target_global_root_values], dim=1)
        local_poses = t.cat([context_local_poses, target_local_poses], dim=1)

        has_global_root_values = t.ones_like(global_root_values[:, :, 0], dtype=t.bool)
        has_local_root_values = t.ones_like(local_root_values[:, :, 0], dtype=t.bool)
        has_local_poses = t.ones_like(local_poses[:, :, 0], dtype=t.bool)
        has_local_root_values[:, self.NUM_FRAMES_PER_TOKEN - 1] = False  # the last velocity is incorrect

        if not self._target_root_realignment:
            # if root is not realigned, disable the following info since they might be misleading
            has_local_root_values[:, -self.NUM_FRAMES_PER_TOKEN:] = False
            has_global_root_values[:, -self.NUM_FRAMES_PER_TOKEN + 1:] = False
            has_local_poses[:, -self.NUM_FRAMES_PER_TOKEN + 1:] = False

        num_tokens = t.full([batch_size], MASKED_NUM_TOKENS).int().to(self._device)

        # pred the motions
        config = {'num_inference_step': 1, 'smooth_root_traj': False, 'allow_pred_out_of_reach_num_tokens': False,
                  'pose_token_sampling_use_argmax': True, 'skip_ending_target_cond': self.SKIP_ENDING_TARGET_COND}
        info = {}
        pred_global_motions, num_pred_tokens = self._inferencer.predict(
            global_root_values, has_global_root_values, local_root_values, has_local_root_values,
            local_poses, has_local_poses, num_tokens, config=config, info=info,
            allowed_pred_num_tokens=input.get('allowed_pred_num_tokens', None)
        )

        self.frames['model_features'] = pred_global_motions
        self.frames['num_pred_frames'] = self.NUM_FRAMES_PER_TOKEN * num_pred_tokens

        self.frames['mujoco_qpos'] = \
            self._converter.convert_motion_features_to_mujoco_qpos(self.frames['model_features'], self._motion_rep, False)
        root_rot = self.frames['mujoco_qpos'][:, :, 3: 7].clone()
        self.frames['mujoco_qpos'][:, :, 3: 7] = root_rot[:, :, [3, 0, 1, 2]]
        if self.FORCE_CANONICALIZATION:
            input['mujoco_qpos'] = self.frames['mujoco_qpos']
            self.frames['mujoco_qpos'] = self._uncanonicalize_mujoco_qpos(input)
        self._current_frame_idx = self.NUM_FRAMES_PER_TOKEN - self.PRED_OFFSETS

        if self.FILTER_QPOS:
            # blend the generated first frames with the context frames for smooth transitions
            # can remove since it does not cause visual difference in the motion
            self.frames['raw_mujoco_qpos'] = self.frames['mujoco_qpos'].clone()
            ctx = input['raw_context_mujoco_qpos']
            num_ctx = ctx.shape[1]
            blend = t.linspace(0.3, 0.7, num_ctx)[None, :, None].to(ctx.device)
            self.frames['mujoco_qpos'][:, :num_ctx, :3] = \
                ctx[:, :, :3] * (1 - blend) + self.frames['mujoco_qpos'][:, :num_ctx, :3] * blend
            self.frames['mujoco_qpos'][:, :num_ctx, 7:] = \
                ctx[:, :, 7:] * (1 - blend) + self.frames['mujoco_qpos'][:, :num_ctx, 7:] * blend

        return self.frames['model_features'], self.frames['mujoco_qpos'], self.frames['num_pred_frames']

    def get_next_frame(self):
        current_frame_idx = self._current_frame_idx
        self._current_frame_idx = max(0, min(current_frame_idx + 1, self.frames['mujoco_qpos'].shape[1] - 1))
        next_qpos = self.frames['mujoco_qpos'][0, current_frame_idx]
        if type(next_qpos) == t.Tensor:
            next_qpos = next_qpos.detach().cpu().numpy()
        return next_qpos

    def get_context_motion_features(self):
        indices = [max(0, min(self._current_frame_idx - self.NUM_FRAMES_PER_TOKEN + i + self.PRED_OFFSETS,
                              self.frames['model_features'].shape[1] - 1))
                   for i in range(self.NUM_FRAMES_PER_TOKEN)]
        return self.frames['model_features'][:, indices, :].to(self._device)

    def get_context_mujoco_qpos(self):
        indices = [max(0, min(self._current_frame_idx - self.NUM_FRAMES_PER_TOKEN + i + self.PRED_OFFSETS,
                              self.frames['mujoco_qpos'].shape[1] - 1))
                   for i in range(self.NUM_FRAMES_PER_TOKEN)]
        return self.frames['mujoco_qpos'][:, indices, :].to(self._device)

    def _canonicalize_mujoco_qpos(self, input: dict):
        mujoco_qpos = input['context_mujoco_qpos']
        input['raw_context_mujoco_qpos'] = input['context_mujoco_qpos'].clone()

        # first frame information
        first_frame_position = mujoco_qpos[:, 0, :3].clone() * t.tensor([[1.0, 1.0, 0.0]]).to(mujoco_qpos.device)
        first_frame_rot = quaternion_to_matrix(mujoco_qpos[:, 0, 3: 7].clone())  # the rotation of first frame
        first_frame_heading_angle = t.atan2(first_frame_rot[:, 1, 0], first_frame_rot[:, 0, 0])
        first_frame_heading_angle[first_frame_heading_angle.isnan()] = 0.0
        first_frame_rot_heading = angle_to_Z_rotation_matrix(first_frame_heading_angle)
        inverse_first_frame_rot_heading = first_frame_rot_heading.transpose(-2, -1)

        # get the canonicalized root info
        canonicalized_root_position = \
            t.matmul(inverse_first_frame_rot_heading[:, None, :, :],
                     (mujoco_qpos[:, :, :3].clone() - first_frame_position)[..., None])[..., 0]

        canonicalized_rot_matrix = t.matmul(inverse_first_frame_rot_heading[:, None, :, :],
                                            quaternion_to_matrix(mujoco_qpos[:, :, 3: 7]))

        mujoco_qpos[:, :, 3: 7] = matrix_to_quaternion(canonicalized_rot_matrix)
        mujoco_qpos[:, :, :3] = canonicalized_root_position

        # canonicalize the movement & facing direction
        input['movement_direction'] = t.matmul(inverse_first_frame_rot_heading,
                                               input['movement_direction'][..., None].float())[..., 0]
        input['facing_direction'] = t.matmul(inverse_first_frame_rot_heading,
                                             input['facing_direction'][:, :, None].float())[..., 0]
        input['first_frame_heading_angle'] = first_frame_heading_angle
        input['first_frame_position'] = first_frame_position
        input['context_mujoco_qpos'] = mujoco_qpos

        # also if specific target headings are provided, canonicalize them
        if 'specific_target_headings' in input:
            input['specific_target_headings'] = \
                input['specific_target_headings'] - first_frame_heading_angle.view([-1, 1])
            input['specific_target_positions'] = \
                t.matmul(inverse_first_frame_rot_heading[:, None, :, :],
                         (input['specific_target_positions'] - first_frame_position[:, None, :])[..., None])[..., 0]

    def _uncanonicalize_mujoco_qpos(self, input: dict):
        mujoco_qpos = input['mujoco_qpos']
        first_frame_heading_angle = input['first_frame_heading_angle']
        first_frame_position = input['first_frame_position']

        # the first frame
        first_frame_rot_heading = angle_to_Z_rotation_matrix(first_frame_heading_angle)

        # get the uncanonicalized root information
        current_first_frame_rotation = quaternion_to_matrix(mujoco_qpos[:, :1, 3: 7])
        current_first_frame_heading_angle = t.atan2(current_first_frame_rotation[:, :, 1, 0],
                                                    current_first_frame_rotation[:, :, 0, 0])
        current_first_frame_rot_heading = angle_to_Z_rotation_matrix(current_first_frame_heading_angle)
        rot_matrix = quaternion_to_matrix(mujoco_qpos[:, :, 3: 7])
        rot_matrix = t.matmul(first_frame_rot_heading[:, None, :, :],
                              t.matmul(current_first_frame_rot_heading.transpose(-2, -1), rot_matrix))
        root_positions = t.matmul(first_frame_rot_heading[:, None, :, :],
                                  t.matmul(current_first_frame_rot_heading.transpose(-2, -1),
                                           mujoco_qpos[:, :, :3, None]))[..., 0]
        root_positions = root_positions - \
            root_positions[:, :1, :] * t.tensor([[[1.0, 1.0, 0.0]]]).to(mujoco_qpos.device) + first_frame_position

        mujoco_qpos[:, :, 3: 7] = matrix_to_quaternion(rot_matrix)
        mujoco_qpos[:, :, :3] = root_positions
        return mujoco_qpos

    def _should_regenerate(self, input: dict):

        idle_mode_id = list(self._clip_holder.CLIPS.keys()).index('idle')
        if self.frames['mode'] is not None and self.frames['mode'].item() == idle_mode_id and \
                input['mode'].item() == idle_mode_id:
            return False

        return True
