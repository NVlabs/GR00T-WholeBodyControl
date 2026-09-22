import enum
import gc
import glob
import os
import os.path as osp
from pathlib import Path
import random
import re
import resource

import easydict
import joblib
from loguru import logger
import numpy as np
from rich import progress
from scipy.spatial import transform
import torch
import torch.multiprocessing as mp

from gear_sonic.isaac_utils import rotations
from gear_sonic.trl.utils import common
from gear_sonic.utils.motion_lib import skeleton


class FixHeightMode(enum.Enum):
    no_fix = 0
    full_fix = 1
    ankle_fix = 2


class MotionlibMode(enum.Enum):
    file = 1
    directory = 2


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


def is_navigation_motion(motion_key):
    return (
        motion_key.startswith("2025")
        or motion_key.startswith("walking_2025")
        or motion_key.startswith("running_2025")
        or motion_key.startswith("slow_walk_2025")
    )


def interpolate_translation_data(
    data,
    source_fps,
    target_fps,
    num_frames,
    max_num_objects=1,
    pad_value=0.0,
):
    """Interpolate translation-like data (e.g., root_pos, contact_points) to target FPS.

    Args:
        data: Tensor of shape (T, N, D) where T=frames, N=num_objects, D=dims (e.g., 3 for pos)
        source_fps: Original frame rate
        target_fps: Target frame rate
        num_frames: Target number of frames
        max_num_objects: Maximum number of objects to pad to
        pad_value: Value to use for padding

    Returns:
        Interpolated tensor of shape (num_frames, max_num_objects, D)
    """
    from gear_sonic.trl.utils import math

    data = torch.tensor(data).float() if not torch.is_tensor(data) else data.float()
    N_objects = data.shape[1]
    D = data.shape[2]

    # Interpolate to target FPS if needed
    if source_fps != target_fps:
        # Reshape to (T, N_objects*D) for batch interpolation
        data_flat = data.reshape(data.shape[0], -1)
        data_interp = math.interpolate_pose(
            data_flat,
            source_fps=source_fps,
            target_fps=target_fps,
            device=data.device,
            interpolation_type="linear",
        )
        data = data_interp.reshape(-1, N_objects, D)

    # Trim or pad to match num_frames
    if data.shape[0] > num_frames:
        data = data[:num_frames]
    elif data.shape[0] < num_frames:
        padding = data[-1:].repeat(num_frames - data.shape[0], 1, 1)
        data = torch.cat([data, padding], dim=0)

    # Pad or trim to max_num_objects
    if data.shape[1] < max_num_objects:
        padding = torch.full(
            (data.shape[0], max_num_objects - data.shape[1], D),
            pad_value,
            dtype=data.dtype,
            device=data.device,
        )
        data = torch.cat([data, padding], dim=1)
    else:
        data = data[:, :max_num_objects]

    return data


def interpolate_quaternion_data(
    data,
    source_fps,
    target_fps,
    num_frames,
    max_num_objects=1,
):
    """Interpolate quaternion data (e.g., root_quat) to target FPS using slerp.

    Args:
        data: Tensor of shape (T, N, 4) where T=frames, N=num_objects
        source_fps: Original frame rate
        target_fps: Target frame rate
        num_frames: Target number of frames
        max_num_objects: Maximum number of objects to pad to

    Returns:
        Interpolated tensor of shape (num_frames, max_num_objects, 4)
    """
    from gear_sonic.trl.utils import math

    data = torch.tensor(data).float() if not torch.is_tensor(data) else data.float()
    N_objects = data.shape[1]

    # Interpolate to target FPS if needed
    if source_fps != target_fps:
        # Reshape to (T, N_objects*4) for batch interpolation
        data_flat = data.reshape(data.shape[0], -1)
        data_interp = math.interpolate_pose(
            data_flat,
            source_fps=source_fps,
            target_fps=target_fps,
            device=data.device,
            interpolation_type="slerp",
            rot_type="quat",
        )
        data = data_interp.reshape(-1, N_objects, 4)

    # Trim or pad to match num_frames
    if data.shape[0] > num_frames:
        data = data[:num_frames]
    elif data.shape[0] < num_frames:
        padding = data[-1:].repeat(num_frames - data.shape[0], 1, 1)
        data = torch.cat([data, padding], dim=0)

    # Pad or trim to max_num_objects
    if data.shape[1] < max_num_objects:
        padding = torch.zeros(
            data.shape[0],
            max_num_objects - data.shape[1],
            4,
            dtype=data.dtype,
            device=data.device,
        )
        padding[:, :, 0] = 1.0  # w=1 for identity quaternion
        data = torch.cat([data, padding], dim=1)
    else:
        data = data[:, :max_num_objects]

    return data


def interpolate_contact_center(
    contact_points_dict,
    source_fps,
    target_fps,
    num_frames,
):
    """Compute contact center and in_contact label from raw contact points.

    Directly scales source frame indices to target frame space, avoiding
    dense array interpolation that would blend real positions with zeros.

    Args:
        contact_points_dict: Dict mapping frame_idx -> (N_points, 3) array
        source_fps: Original frame rate
        target_fps: Target frame rate
        num_frames: Target number of frames

    Returns:
        Tuple of:
            contact_center: Tensor of shape (num_frames, 3)
            in_contact: Tensor of shape (num_frames,) with binary labels
    """
    if not contact_points_dict:
        return torch.zeros(num_frames, 3), torch.zeros(num_frames)

    fps_ratio = target_fps / source_fps
    contact_center = torch.zeros(num_frames, 3)
    in_contact = torch.zeros(num_frames)

    for src_idx, points in contact_points_dict.items():
        if not (hasattr(points, "shape") and len(points) > 0):
            continue
        # Scale source frame range to target frame space
        t_start = max(0, int(src_idx * fps_ratio))
        t_end = min(num_frames, int((src_idx + 1) * fps_ratio) + 1)
        center = torch.from_numpy(points.mean(axis=0).astype(np.float32))
        contact_center[t_start:t_end] = center
        in_contact[t_start:t_end] = 1.0

    return contact_center, in_contact


class MotionLibBase:
    def __init__(self, motion_lib_cfg, num_envs, device):
        self.m_cfg = motion_lib_cfg
        self.motion_fps_scale = self.m_cfg.get("motion_fps_scale", 1.0)
        self._sim_fps = 1 / self.m_cfg.get("step_dt", 1 / 50)
        self.target_fps = self.m_cfg.get("target_fps", 50)
        self.adaptive_sampling_cfg = self.m_cfg.get("adaptive_sampling", {})
        self.all_motions_loaded = False

        self.debug = motion_lib_cfg.get("debug", False)
        self.use_parallel_fk = motion_lib_cfg.get("use_parallel_fk", False)
        self.num_envs = num_envs
        self._device = device
        self.mesh_parsers = None
        self.has_action = False
        skeleton_file = Path(self.m_cfg.asset.assetRoot) / self.m_cfg.asset.assetFileName
        self.skeleton_tree = skeleton.SkeletonTree.from_mjcf(skeleton_file)
        logger.info(f"Loaded skeleton from {skeleton_file}")
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        self.load_data(self.m_cfg.motion_file)
        self.use_adaptive_sampling = self.adaptive_sampling_cfg.get("enable", False)
        if self.use_adaptive_sampling:
            self.init_adaptive_sampling()
        self.setup_constants(
            fix_height=motion_lib_cfg.get("fix_height", FixHeightMode.no_fix),
            multi_thread=self.m_cfg.get("multi_thread", True),
        )

        self.vid_smpl_pose = None
        self.vid_smpl_joints = None
        self.smpl_data = None
        smpl_motion_file = motion_lib_cfg.get("smpl_motion_file", None)
        self.smpl_data_keys = set()
        if smpl_motion_file is not None:
            if smpl_motion_file in ("dummy", "zeros"):
                # Generate dummy zero SMPL data so SMPL observation terms work
                # without needing to null them out in the config.
                self.smpl_data = [None] * len(self._motion_data_keys)
            elif osp.exists(smpl_motion_file):
                if osp.isfile(smpl_motion_file):
                    self.smpl_data = joblib.load(smpl_motion_file)
                    self.smpl_data_keys = set(self.smpl_data.keys())
                    self.smpl_data = [
                        (self.smpl_data[k] if k in self.smpl_data else None)  # noqa: SIM401
                        for k in self._motion_data_keys
                    ]
                else:
                    self.smpl_data = []
                    smpl_pkl_files = set(
                        glob.glob(osp.join(smpl_motion_file, "**", "*.pkl"), recursive=True)
                    )
                    for k in self._motion_data_keys:
                        seq = os.path.basename(k)
                        smpl_path = osp.join(smpl_motion_file, seq + ".pkl")
                        if self.debug or smpl_path in smpl_pkl_files:
                            self.smpl_data.append({"seq": seq, "path": smpl_path})
                            self.smpl_data_keys.add(seq)
                        else:
                            self.smpl_data.append(None)
            else:
                self.smpl_data = [None] * len(self._motion_data_keys)

        self.smpl_y_up = motion_lib_cfg.get("smpl_y_up", False)

        # SOMA skeleton data loading (parallel to SMPL)
        self.soma_data = None
        soma_motion_file = motion_lib_cfg.get("soma_motion_file", None)
        self.soma_data_keys = set()
        self.soma_y_up = motion_lib_cfg.get("soma_y_up", True)  # BVH is Y-up by default
        self.num_soma_joints = motion_lib_cfg.get("num_soma_joints", 26)
        if soma_motion_file is not None:
            if soma_motion_file in ("dummy", "zeros"):
                self.soma_data = [None] * len(self._motion_data_keys)
            elif osp.exists(soma_motion_file):
                if osp.isfile(soma_motion_file):
                    self.soma_data = joblib.load(soma_motion_file)
                    self.soma_data_keys = set(self.soma_data.keys())
                    self.soma_data = [
                        (self.soma_data[k] if k in self.soma_data else None)  # noqa: SIM401
                        for k in self._motion_data_keys
                    ]
                else:
                    # Directory mode: per-motion PKL files (may be nested in subdirs)
                    soma_index = {
                        osp.splitext(osp.basename(f))[0]: f
                        for f in glob.glob(
                            osp.join(soma_motion_file, "**", "*.pkl"), recursive=True
                        )
                    }
                    self.soma_data = []
                    for k in self._motion_data_keys:
                        seq = os.path.basename(k)
                        soma_path = soma_index.get(seq)
                        if soma_path is not None or self.debug:
                            self.soma_data.append(
                                {
                                    "seq": seq,
                                    "path": soma_path or osp.join(soma_motion_file, seq + ".pkl"),
                                }
                            )
                            self.soma_data_keys.add(seq)
                        else:
                            self.soma_data.append(None)
            else:
                self.soma_data = [None] * len(self._motion_data_keys)

        # Object data loading (similar to SMPL data)
        self.object_data = None
        object_motion_file = motion_lib_cfg.get("object_motion_file", None)
        self.object_data_keys = set()
        self.max_num_objects = motion_lib_cfg.get("max_num_objects", 1)
        if object_motion_file is not None:
            if osp.isfile(object_motion_file):
                self.object_data = joblib.load(object_motion_file)
                self.object_data_keys = set(self.object_data.keys())
                self.object_data = [
                    (self.object_data[k] if k in self.object_data else None)  # noqa: SIM401
                    for k in self._motion_data_keys
                ]
            else:
                self.object_data = []
                # TODO: osp.exists() can be very expensive, consider using a set of all object pkl files
                # like in the smpl data loading above.
                for k in self._motion_data_keys:
                    seq = os.path.basename(k)
                    object_path = osp.join(object_motion_file, seq + ".pkl")
                    if self.debug or osp.exists(object_path):
                        self.object_data.append({"seq": seq, "path": object_path})
                        self.object_data_keys.add(seq)
                    else:
                        self.object_data.append(None)
        # randomize the upper body poses condition
        self.randomize_upper_body_poses = self.m_cfg.get("cat_upper_body_poses", False)
        self.cat_upper_body_poses_prob = self.m_cfg.get("cat_upper_body_poses_prob", 0.0)
        # The default prefixes for the upper body augmentation -- generated by the kinematic planner.
        self.upper_body_augment_prefixes = self.m_cfg.get(
            "upper_body_augment_prefixes",
            ["2025", "walking_2025", "running_2025", "slow_walk_2025"],
        )
        # Wrist joint noise augmentation config
        self.randomize_wrist_poses = self.m_cfg.get("randomize_wrist_poses", False)
        self.randomize_wrist_prob = self.m_cfg.get("randomize_wrist_prob", 0.3)
        self.randomize_wrist_std = self.m_cfg.get("randomize_wrist_std", 0.1)
        # MuJoCo DOF indices for wrist joints (L/R roll/pitch/yaw)
        self.wrist_mujoco_dof_indices = [19, 20, 21, 26, 27, 28]

    def load_data(self, motion_file):
        if osp.isfile(motion_file):
            self.mode = MotionlibMode.file
            self._motion_data_load = joblib.load(motion_file)
        else:
            assert osp.isdir(
                motion_file
            ), f"Expected motion_file to be a directory, got: {motion_file}"
            self.mode = MotionlibMode.directory
            if self.debug:
                self._motion_data_load = {}
            else:
                self._motion_data_load = {
                    osp.splitext(osp.basename(f))[0]: {"path": f}
                    for f in glob.glob(osp.join(motion_file, "**", "*.pkl"), recursive=True)
                    if not f.endswith("metadata.pkl")
                }

            metadata_files = []
            # Check for metadata.pkl directly in motion_file directory
            direct_metadata = osp.join(motion_file, "metadata.pkl")
            if osp.exists(direct_metadata):
                metadata_files.append(direct_metadata)
            # Also check subdirectories for metadata.pkl
            all_sub_dirs = os.listdir(motion_file)
            if self.debug:
                all_sub_dirs = all_sub_dirs[:1]
            for sub_dir in all_sub_dirs:
                sub_dir_path = osp.join(motion_file, sub_dir)
                if osp.isdir(sub_dir_path):
                    sub_meta = osp.join(sub_dir_path, "metadata.pkl")
                    if osp.exists(sub_meta):
                        metadata_files.append(sub_meta)

            for metadata_file in metadata_files:
                metadata = joblib.load(metadata_file)
                if self.debug:
                    metadata = {
                        k: v
                        for k, v in list(metadata.items())[:5]
                        if osp.exists(f"{sub_dir_path}/{k}.pkl")
                    }
                for k, v in metadata.items():
                    if self.debug:
                        self._motion_data_load[k] = {"path": f"{sub_dir_path}/{k}.pkl"}
                    if (
                        k in self._motion_data_load
                    ):  # metadata file can have more motion sequences than in the directory. Only load the necessary ones.  # noqa: E501
                        self._motion_data_load[k].update(v)

            print(f"Loaded {len(self._motion_data_load)} motion files")  # noqa: T201

        data_list = self._motion_data_load

        filter_motion_keys = self.m_cfg.get("filter_motion_keys", None)
        if filter_motion_keys is not None:
            if isinstance(filter_motion_keys, str):  # noqa: SIM108
                patterns = [filter_motion_keys]
            else:
                patterns = list(filter_motion_keys)

            if all(pattern in data_list for pattern in patterns):
                matched_keys = [pattern for pattern in patterns if pattern in data_list]
            else:
                compiled = []
                for pattern in patterns:
                    try:
                        compiled.append(re.compile(pattern))
                    except re.error as exc:
                        raise ValueError(f"Invalid filter_motion_keys regex: {pattern}") from exc

                matched_keys = [
                    k for k in data_list if any(regex.fullmatch(k) for regex in compiled)
                ]
                matched_keys.sort()
            data_list = {k: data_list[k] for k in matched_keys}

        remove_motion_keys = self.m_cfg.get("remove_motion_keys", None)
        if remove_motion_keys is not None:
            # Remove any motion whose key starts with any of the remove_motion_keys prefixes
            keys_to_remove = [
                k for k in data_list if any(k.startswith(prefix) for prefix in remove_motion_keys)
            ]
            for k in keys_to_remove:
                del data_list[k]

        max_unique_motions = self.m_cfg.get("max_unique_motions", None)
        if max_unique_motions is not None and len(data_list) > max_unique_motions:
            import random

            keys = sorted(data_list.keys())  # Sort for determinism, then sample
            selected = random.sample(keys, max_unique_motions)
            data_list = {k: data_list[k] for k in selected}
            print(  # noqa: T201
                f"Limited to {max_unique_motions} random motions (from {len(keys)})"
            )  # noqa: RUF100, T201

        self._motion_data_list = np.array(list(data_list.values()))
        self._motion_data_keys = np.array(list(data_list.keys()))

        # # HACK: Force specific motion only
        # _FORCE_MOTION_KEY = "canned_food_31_jason_rigged_001_indoor2-v4_rand00063_000065"
        # if _FORCE_MOTION_KEY in self._motion_data_keys:
        #     idx = list(self._motion_data_keys).index(_FORCE_MOTION_KEY)
        #     self._motion_data_list = np.array([self._motion_data_list[idx]])
        #     self._motion_data_keys = np.array([_FORCE_MOTION_KEY])

        self._num_unique_motions = len(self._motion_data_list)
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def _should_augment_upper_body(self, motion_key):
        """Check if motion key matches any prefix for upper body augmentation"""  # noqa: D415
        return any(motion_key.startswith(prefix) for prefix in self.upper_body_augment_prefixes)

    def setup_constants(self, fix_height=FixHeightMode.full_fix, multi_thread=True):
        self.fix_height = fix_height
        self.multi_thread = multi_thread

        #### Termination history
        self._curr_motion_ids = None
        self._termination_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._success_rate = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_history = torch.zeros(self._num_unique_motions).to(self._device)
        self._sampling_prob = (
            torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
        )  # For use in sampling batches

    def update_soft_sampling_weight(self, failed_keys):
        # sampling weight based on evaluation, only "mostly" trained on "failed" sequences. Auto PMCP.
        if len(failed_keys) > 0:
            all_keys = self._motion_data_keys.tolist()
            indexes = [all_keys.index(k) for k in failed_keys]
            self._termination_history[indexes] += 1
            self.update_sampling_prob(self._termination_history)

            print(  # noqa: T201
                "############################################################ Auto PMCP ############################################################"  # noqa: E501
            )
            print(  # noqa: T201
                f"Training mostly on {len(self._sampling_prob.cpu().nonzero())} seqs "
            )  # noqa: RUF100, T201
            print(  # noqa: T201
                self._motion_data_keys[self._sampling_prob.cpu().nonzero()].flatten()
            )  # noqa: RUF100, T201
            print(  # noqa: T201
                "###############################################################################################################################"
            )
        else:
            all_keys = self._motion_data_keys.tolist()
            self._sampling_prob = (
                torch.ones(self._num_unique_motions).to(self._device) / self._num_unique_motions
            )  # For use in sampling batches

    def update_sampling_prob(self, termination_history):
        if (
            len(termination_history) == len(self._termination_history)
            and termination_history.sum() > 0
        ):
            self._sampling_prob[:] = termination_history / termination_history.sum()
            if self._sampling_prob[self._curr_motion_ids].sum() == 0:
                self._sampling_prob[self._curr_motion_ids] += 1e-6
                self._sampling_prob[:] = self._sampling_prob[:] / self._sampling_prob[:].sum()
            self._sampling_batch_prob = (
                self._sampling_prob[self._curr_motion_ids]
                / self._sampling_prob[self._curr_motion_ids].sum()
            )
            self._termination_history = termination_history
            return True
        else:
            return False

    def get_motion_actions(self, motion_ids, motion_times):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]
        # import ipdb; ipdb.set_trace()
        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]  # noqa: F841

        action = self._motion_actions[f0l]
        return action

    def get_time_step_total(self, motion_ids):
        return self._motion_num_frames[motion_ids]

    @property
    def body_indexes(self):
        return self.m_cfg.get("body_indexes_data", None)

    def get_dof_pos(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]

        return self.dof_pos[motion_steps + length_starts]

    def get_dof_vel(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.dof_vel[motion_steps + length_starts]

    def get_hand_dof_pos(self, motion_ids, motion_steps):
        """Get hand DOF positions if available (for 43-DOF motion on 43-DOF robot)."""
        if not hasattr(self, "hand_dof_pos") or self.hand_dof_pos is None:
            return None
        length_starts = self.length_starts[motion_ids]
        return self.hand_dof_pos[motion_steps + length_starts]

    def get_body_pos_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]

        return self.body_pos_w[motion_steps + length_starts]

    def get_body_quat_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.body_quat_w[motion_steps + length_starts]

    def get_body_lin_vel_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.body_lin_vel_w[motion_steps + length_starts]

    def get_body_ang_vel_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.body_ang_vel_w[motion_steps + length_starts]

    # Full body data getters (all bodies, not sliced by body_indexes)
    def get_body_pos_w_full(self, motion_ids, motion_steps):
        """Get full body positions (all bodies, IsaacLab order)."""
        length_starts = self.length_starts[motion_ids]
        return self.body_pos_w_full[motion_steps + length_starts]

    def get_body_quat_w_full(self, motion_ids, motion_steps):
        """Get full body quaternions (all bodies, IsaacLab order, wxyz)."""
        length_starts = self.length_starts[motion_ids]
        return self.body_quat_w_full[motion_steps + length_starts]

    def get_body_lin_vel_w_full(self, motion_ids, motion_steps):
        """Get full body linear velocities (all bodies, IsaacLab order)."""
        length_starts = self.length_starts[motion_ids]
        return self.body_lin_vel_w_full[motion_steps + length_starts]

    def get_body_ang_vel_w_full(self, motion_ids, motion_steps):
        """Get full body angular velocities (all bodies, IsaacLab order)."""
        length_starts = self.length_starts[motion_ids]
        return self.body_ang_vel_w_full[motion_steps + length_starts]

    def get_root_pos_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.body_pos_w[motion_steps + length_starts, 0, :]

    def get_root_quat_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.body_quat_w[motion_steps + length_starts, 0, :]

    def get_root_lin_vel_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.body_lin_vel_w[motion_steps + length_starts, 0, :]

    def get_root_ang_vel_w(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.body_ang_vel_w[motion_steps + length_starts, 0, :]

    def get_smpl_pose(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_smpl_poses[motion_steps + length_starts]

    def get_smpl_joints(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_smpl_joints[motion_steps + length_starts]

    def get_smpl_transl(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_smpl_transl[motion_steps + length_starts]

    @staticmethod
    def _resample_soma_tensor(data, fps_source, fps_target):
        """Resample a SOMA tensor along dim 0 matching interploate_pose frame count.

        Uses the same arange(0, duration, 1/fps_target) formula as
        torch_humanoid_batch.interploate_pose so robot and SOMA frame counts align.
        """
        n_src = data.shape[0]
        duration = (n_src - 1) / fps_source
        tgt_times = torch.arange(0, duration, 1 / fps_target, dtype=torch.float32)
        n_tgt = len(tgt_times)
        if n_tgt <= 1:
            return data[:1]
        # Compute blend weights (same logic as _compute_frame_blend)
        phase = tgt_times / duration
        idx0 = (phase * (n_src - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(n_src - 1))
        blend = (phase * (n_src - 1) - idx0).float()
        # Reshape blend for broadcasting with arbitrary trailing dims
        for _ in range(data.dim() - 1):
            blend = blend.unsqueeze(-1)
        return data[idx0] * (1 - blend) + data[idx1] * blend

    def get_soma_joints(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_soma_joints[motion_steps + length_starts]

    def get_soma_root_quat(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_soma_root_quat[motion_steps + length_starts]

    def get_soma_transl(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_soma_transl[motion_steps + length_starts]

    def get_object_root_pos(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_object_root_pos[motion_steps + length_starts]

    def get_object_root_quat(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self._motion_object_root_quat[motion_steps + length_starts]

    def get_object_lin_vel(self, motion_ids, motion_steps):
        """Get object linear velocity from motion library."""
        length_starts = self.length_starts[motion_ids]
        return self._motion_object_lin_vel[motion_steps + length_starts]

    def get_object_ang_vel(self, motion_ids, motion_steps):
        """Get object angular velocity from motion library."""
        length_starts = self.length_starts[motion_ids]
        return self._motion_object_ang_vel[motion_steps + length_starts]

    def get_object_contact_center(self, motion_ids, motion_steps, hand="right_hand"):
        """Get object contact center from motion library.

        Contact center is the mean of all contact points per frame for the given hand.

        Args:
            motion_ids: (N,) tensor of motion indices
            motion_steps: (N,) tensor of frame indices within each motion
            hand: Which hand's contact center to return ("left_hand" or "right_hand")

        Returns:
            Tensor of shape (N, 3) with contact center positions in object-local frame,
            or None if not available.
        """
        attr = f"_motion_object_contact_center_{'left' if hand == 'left_hand' else 'right'}"
        if not hasattr(self, attr):
            return None
        length_starts = self.length_starts[motion_ids]
        return getattr(self, attr)[motion_steps + length_starts]

    def get_object_in_contact(self, motion_ids, motion_steps, hand="right_hand"):
        """Get binary in_contact label for the given hand.

        Args:
            motion_ids: (N,) tensor of motion indices
            motion_steps: (N,) tensor of frame indices within each motion
            hand: Which hand ("left_hand" or "right_hand")

        Returns:
            Tensor of shape (N,) with 1.0 if in contact, 0.0 otherwise,
            or None if not available.
        """
        attr = f"_motion_object_in_contact_{'left' if hand == 'left_hand' else 'right'}"
        if not hasattr(self, attr):
            return None
        length_starts = self.length_starts[motion_ids]
        return getattr(self, attr)[motion_steps + length_starts]

    def get_hand_action(self, motion_ids, motion_steps, hand="right_hand"):
        """Get discrete hand action (open/closed) for the given hand.

        Args:
            motion_ids: (N,) tensor of motion indices
            motion_steps: (N,) tensor of frame indices within each motion
            hand: Which hand ("left_hand" or "right_hand")

        Returns:
            Tensor of shape (N,) with -1.0 = open, +1.0 = closed,
            or None if not available.
        """
        attr = f"_motion_hand_action_{'left' if hand == 'left_hand' else 'right'}"
        if not hasattr(self, attr):
            return None
        length_starts = self.length_starts[motion_ids]
        return getattr(self, attr)[motion_steps + length_starts]

    def get_feet_l(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.feet_l[motion_steps + length_starts]

    def get_feet_r(self, motion_ids, motion_steps):
        length_starts = self.length_starts[motion_ids]
        return self.feet_r[motion_steps + length_starts]

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        if "dof_pos" in self.__dict__:
            local_rot0 = self.dof_pos[f0l]
            local_rot1 = self.dof_pos[f1l]
        else:
            local_rot0 = self.body_pos_b[f0l]
            local_rot1 = self.body_pos_b[f1l]

        body_lin_vel_w0 = self.body_lin_vel_w[f0l]
        body_lin_vel_w1 = self.body_lin_vel_w[f1l]

        body_ang_vel0 = self.body_ang_vel_w[f0l]
        body_ang_vel1 = self.body_ang_vel_w[f1l]

        body_pos_w0 = self.body_pos_w[f0l, :]
        body_pos_w1 = self.body_pos_w[f1l, :]

        dof_vel0 = self.dof_vel[f0l]
        dof_vel1 = self.dof_vel[f1l]

        vals = [
            local_rot0,
            local_rot1,
            body_lin_vel_w0,
            body_lin_vel_w1,
            body_ang_vel0,
            body_ang_vel1,
            body_pos_w0,
            body_pos_w1,
            dof_vel0,
            dof_vel1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            body_pos_w = (
                1.0 - blend_exp
            ) * body_pos_w0 + blend_exp * body_pos_w1  # ZL: apply offset
        else:
            body_pos_w = (
                (1.0 - blend_exp) * body_pos_w0 + blend_exp * body_pos_w1 + offset[..., None, :]
            )  # ZL: apply offset

        body_lin_vel_w = (1.0 - blend_exp) * body_lin_vel_w0 + blend_exp * body_lin_vel_w1
        body_ang_vel_w = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        if "dof_pos" in self.__dict__:  # Robot Joints
            dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
            dof_pos = (1.0 - blend) * local_rot0 + blend * local_rot1
        else:
            dof_vel = (1.0 - blend_exp) * dof_vel0 + blend_exp * dof_vel1
            local_rot = rotations.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
            dof_pos = self._local_rotation_to_dof_smpl(local_rot)

        body_quat_w0 = self.body_quat_w[f0l]
        body_quat_w1 = self.body_quat_w[f1l]
        body_quat_w = rotations.slerp(body_quat_w0, body_quat_w1, blend_exp)
        return_dict = {}

        if "gts_t" in self.__dict__:
            body_pos_w_t0 = self.body_pos_t_w[f0l]
            body_pos_w_t1 = self.body_pos_t_w[f1l]

            body_quat_t0 = self.body_quat_t_w[f0l]
            body_quat_t1 = self.body_quat_t_w[f1l]

            body_lin_vel_w_t0 = self.body_lin_vel_t_w[f0l]
            body_lin_vel_w_t1 = self.body_lin_vel_t_w[f1l]

            body_ang_vel_t0 = self.body_ang_vel_t_w[f0l]
            body_ang_vel_t1 = self.body_ang_vel_t_w[f1l]
            if offset is None:
                body_pos_t_w = (1.0 - blend_exp) * body_pos_w_t0 + blend_exp * body_pos_w_t1
            else:
                body_pos_t_w = (
                    (1.0 - blend_exp) * body_pos_w_t0
                    + blend_exp * body_pos_w_t1
                    + offset[..., None, :]
                )
            body_quat_t_w = rotations.slerp(body_quat_t0, body_quat_t1, blend_exp)
            body_lin_vel_t_w = (1.0 - blend_exp) * body_lin_vel_w_t0 + blend_exp * body_lin_vel_w_t1
            body_ang_vel_t_w = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            body_pos_t_w = body_pos_w
            body_quat_t_w = body_quat_w
            body_lin_vel_t_w = body_lin_vel_w
            body_ang_vel_t_w = body_ang_vel_w

        if self.smpl_data is not None:
            smpl_pose0 = self._motion_smpl_poses[f0l]
            smpl_pose1 = self._motion_smpl_poses[f1l]
            smpl_pose = (1.0 - blend) * smpl_pose0 + blend * smpl_pose1
            return_dict.update({"smpl_pose": smpl_pose.clone()})

            if hasattr(self, "_motion_smpl_joints"):
                smpl_joints0 = self._motion_smpl_joints[f0l]
                smpl_joints1 = self._motion_smpl_joints[f1l]
                smpl_joints = (1.0 - blend_exp) * smpl_joints0 + blend_exp * smpl_joints1
                return_dict.update({"smpl_joints": smpl_joints.clone()})

            if hasattr(self, "_motion_smpl_transl"):
                smpl_transl0 = self._motion_smpl_transl[f0l]
                smpl_transl1 = self._motion_smpl_transl[f1l]
                smpl_transl = (1.0 - blend_exp) * smpl_transl0 + blend_exp * smpl_transl1
                return_dict.update({"smpl_transl": smpl_transl.clone()})

        if self.soma_data is not None:
            if hasattr(self, "_motion_soma_joints"):
                soma_joints0 = self._motion_soma_joints[f0l]
                soma_joints1 = self._motion_soma_joints[f1l]
                soma_joints = (1.0 - blend_exp) * soma_joints0 + blend_exp * soma_joints1
                return_dict.update({"soma_joints": soma_joints.clone()})

            if hasattr(self, "_motion_soma_root_quat"):
                # For quaternions, use slerp (approximate with linear blend + normalize)
                soma_rq0 = self._motion_soma_root_quat[f0l]
                soma_rq1 = self._motion_soma_root_quat[f1l]
                soma_root_quat = (1.0 - blend_exp) * soma_rq0 + blend_exp * soma_rq1
                soma_root_quat = soma_root_quat / (soma_root_quat.norm(dim=-1, keepdim=True) + 1e-8)
                return_dict.update({"soma_root_quat": soma_root_quat.clone()})

            if hasattr(self, "_motion_soma_transl"):
                soma_transl0 = self._motion_soma_transl[f0l]
                soma_transl1 = self._motion_soma_transl[f1l]
                soma_transl = (1.0 - blend_exp) * soma_transl0 + blend_exp * soma_transl1
                return_dict.update({"soma_transl": soma_transl.clone()})

        if self.object_data is not None:
            if hasattr(self, "_motion_object_root_pos"):
                object_root_pos0 = self._motion_object_root_pos[f0l]
                object_root_pos1 = self._motion_object_root_pos[f1l]
                object_root_pos = (
                    1.0 - blend_exp
                ) * object_root_pos0 + blend_exp * object_root_pos1
                return_dict.update({"object_root_pos": object_root_pos.clone()})

            if hasattr(self, "_motion_object_root_quat"):
                object_root_quat0 = self._motion_object_root_quat[f0l]
                object_root_quat1 = self._motion_object_root_quat[f1l]
                # Use slerp for quaternion interpolation
                object_root_quat = rotations.slerp(object_root_quat0, object_root_quat1, blend_exp)
                return_dict.update({"object_root_quat": object_root_quat.clone()})

        return_dict.update(
            {
                "root_pos": body_pos_w[..., 0, :].clone(),
                "root_rot": body_quat_w[..., 0, :].clone(),
                "dof_pos": dof_pos.clone(),
                "root_vel": body_lin_vel_w[..., 0, :].clone(),
                "root_ang_vel": body_ang_vel_w[..., 0, :].clone(),
                "dof_vel": dof_vel.clone(),
                "motion_aa": self._motion_aa[f0l].clone(),
                "motion_bodies": self._motion_bodies[motion_ids].clone(),
                "body_pos_w": body_pos_w.clone(),
                "body_quat_w": body_quat_w.clone(),
                "body_lin_vel_w": body_lin_vel_w.clone(),
                "body_ang_vel_w": body_ang_vel_w.clone(),
                "body_pos_w_t": body_pos_t_w.clone(),
                "body_quat_t": body_quat_t_w.clone(),
                "body_lin_vel_w_t": body_lin_vel_t_w.clone(),
                "body_ang_vel_t": body_ang_vel_t_w.clone(),
            }
        )
        if "feet_l" in self.__dict__:
            blend_int = blend.round().int()
            feet_l = torch.where(blend_int == 0, self.feet_l[f0l], self.feet_l[f1l])
            feet_r = torch.where(blend_int == 0, self.feet_r[f0l], self.feet_r[f1l])
            return_dict.update(
                {
                    "feet_l": feet_l.clone().bool(),
                    "feet_r": feet_r.clone().bool(),
                }
            )
        return return_dict

    def load_all_motions(self):
        self.all_motions_loaded = True
        self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)

    def load_motions_for_training(self, max_num_seqs=None):
        if self.all_motions_loaded:
            print("All motions already loaded!!! No need to resample.")  # noqa: T201
            return False

        if self.m_cfg.get("override_num_motions_to_load", None) is not None:
            max_num_seqs = self.m_cfg.override_num_motions_to_load

        # Option to load unique motions (no duplicates) - useful for replay/evaluation
        load_unique = self.m_cfg.get("load_unique_motions", False)

        if (
            max_num_seqs is None
        ):  # if not specified, load all motions, can OOM if the dataset is too large.
            max_num_seqs = self._num_unique_motions
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
        elif (
            max_num_seqs >= self._num_unique_motions
        ):  # if specified but more than the number of unique motions, load all motions as well.
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
        else:  # if there are more motions than specified, then randomly sample the requested number of motions.
            self.all_motions_loaded = False
            # Use random_sample=False when load_unique=True to avoid duplicates
            self.load_motions(random_sample=not load_unique, num_motions_to_load=max_num_seqs)
            if load_unique:
                print(  # noqa: T201
                    f"[MotionLib] Loaded {max_num_seqs} unique motions (no duplicates)"
                )  # noqa: RUF100, T201
        return True

    def load_motions_for_evaluation(self, start_idx=0):
        # disable this check to avoid upper body poses randomization in evaluation
        # if self.all_motions_loaded:
        #     print("All motions already loaded!!! No need to resample.")
        #     return

        if (
            self._num_unique_motions > self.num_envs
        ):  # if number of motions is more than number of envs, then we should only partially load the motions.
            self.all_motions_loaded = False
            self.load_motions(
                random_sample=False,
                num_motions_to_load=self.num_envs,
                start_idx=start_idx,
                is_evaluation=True,
            )
        else:
            self.all_motions_loaded = True
            self.load_motions(
                random_sample=False,
                num_motions_to_load=self._num_unique_motions,
                start_idx=start_idx,
                is_evaluation=True,
            )

    def load_motions(
        self,
        random_sample=True,
        start_idx=0,
        max_len=-1,
        target_heading=None,
        num_motions_to_load=None,
        is_evaluation=False,
    ):

        if "gts" in self.__dict__:
            del (
                self.body_pos_w,
                self.body_quat_w,
                self.body_pos_b,
                self.root_linv_vel_w,
                self.root_ang_vel_w,
                self.body_ang_vel_w,
                self.body_lin_vel_w,
                self.dof_vels,
                self.dof_pos,
            )
            if "gts_t" in self.__dict__:
                del (
                    self.body_pos_t_w,
                    self.body_quat_t_w,
                    self.body_lin_vel_t_w,
                    self.body_ang_vel_t_w,
                )

        motions = []
        _motion_lengths = []
        _motion_fps = []
        _motion_dt = []
        _motion_num_frames = []
        _motion_bodies = []
        _motion_aa = []
        has_action = False  # noqa: F841
        _motion_actions = []
        _motion_smpl_poses = []
        _motion_smpl_joints = []
        _motion_smpl_transl = []
        _motion_soma_joints = []
        _motion_soma_root_quat = []
        _motion_soma_transl = []
        _motion_object_root_pos = []
        _motion_object_root_quat = []
        _motion_object_contact_center_left = []
        _motion_object_contact_center_right = []
        _motion_object_in_contact_left = []
        _motion_object_in_contact_right = []
        _motion_hand_action_left = []
        _motion_hand_action_right = []

        total_len = 0.0
        self.num_joints = len(self.skeleton_tree.node_names)
        if num_motions_to_load is None:  # noqa: SIM108
            num_motion_to_load = self.num_envs
        else:
            num_motion_to_load = num_motions_to_load

        if self.use_adaptive_sampling:
            self.update_adaptive_sampling_motion_sequences()

        if random_sample:
            sample_idxes = torch.multinomial(
                self._sampling_prob, num_samples=num_motion_to_load, replacement=True
            ).to(self._device)
        else:  # start_idx only used for non-random sampling.
            sample_idxes = torch.clamp(
                torch.arange(num_motion_to_load) + start_idx, max=self._num_unique_motions - 1
            ).to(self._device)

        # sample_idxes = torch.tensor([self._motion_data_keys.tolist().index("0-KIT_8_WalkInClockwiseCircle04_poses")]).to(self._device)  # noqa: E501
        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = (
            [self._motion_data_keys[sample_idxes.cpu()]]
            if sample_idxes.numel() == 1
            else self._motion_data_keys[sample_idxes.cpu()].tolist()
        )
        self._sampling_batch_prob = (
            self._sampling_prob[self._curr_motion_ids]
            / self._sampling_prob[self._curr_motion_ids].sum()
        )

        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:10]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:10]}, ....")

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        if self.smpl_data is not None:
            smpl_data_list = [self.smpl_data[idx] for idx in sample_idxes.cpu().numpy()]
        else:
            smpl_data_list = None
        if self.object_data is not None:
            object_data_list = [self.object_data[idx] for idx in sample_idxes.cpu().numpy()]
        else:
            object_data_list = None
        if self.soma_data is not None:
            soma_data_list = [self.soma_data[idx] for idx in sample_idxes.cpu().numpy()]
        else:
            soma_data_list = None
        torch.set_num_threads(1)

        # Increase file descriptor limit to prevent "too many open files" error
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            target_limit = 1048576

            # Try to set both soft and hard limits
            if soft_limit < target_limit:
                try:
                    # First try to increase hard limit (requires root)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, target_limit))
                    logger.info(
                        f"Increased file descriptor limits from {soft_limit}/{hard_limit} to {target_limit}/{target_limit}"  # noqa: E501
                    )
                except PermissionError:
                    # Fallback to increasing only soft limit up to hard limit
                    new_soft = min(target_limit, hard_limit)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard_limit))
                    logger.info(
                        f"Increased soft file descriptor limit from {soft_limit} to {new_soft} (hard limit: {hard_limit})"  # noqa: E501
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Could not increase file descriptor limit: {e}")

        manager = mp.Manager()
        queue = manager.Queue()
        num_jobs = min(min(mp.cpu_count(), 32), len(motion_data_list))  # noqa: PLW3301

        if num_jobs <= 8 or not self.multi_thread or len(motion_data_list) <= 128:
            num_jobs = 1

        logger.info(f"Loading motions with {num_jobs} jobs...")
        self.res_non_nav_dataset = {}
        res_acc = {}  # using dictionary ensures order of the results.
        workers = []

        # if self.randomize_upper_body_poses:
        # self.cat_upper_body_poses_prob = 1.0
        if self.randomize_upper_body_poses and not is_evaluation:

            # get indices that are in navigation dataset
            nav_indices = [
                i
                for i in range(len(motion_data_list))
                if self._should_augment_upper_body(self.curr_motion_keys[i])
            ]
            other_indices = [
                i
                for i in range(len(motion_data_list))
                if not self._should_augment_upper_body(self.curr_motion_keys[i])
            ]

            nav_motion_data_list = [motion_data_list[i] for i in nav_indices]
            other_motion_data_list = [motion_data_list[i] for i in other_indices]

            if self.smpl_data is not None:
                nav_smpl_data_list = [smpl_data_list[i] for i in nav_indices]
                other_smpl_data_list = [smpl_data_list[i] for i in other_indices]
            else:
                nav_smpl_data_list = None
                other_smpl_data_list = None

            if self.object_data is not None:
                nav_object_data_list = [object_data_list[i] for i in nav_indices]
                other_object_data_list = [object_data_list[i] for i in other_indices]
            else:
                nav_object_data_list = None
                other_object_data_list = None

            if soma_data_list is not None:
                nav_soma_data_list = [soma_data_list[i] for i in nav_indices]
                other_soma_data_list = [soma_data_list[i] for i in other_indices]
            else:
                nav_soma_data_list = None
                other_soma_data_list = None

            # load non-navigation dataset first
            if len(other_motion_data_list) > 0:
                jobs = other_motion_data_list
                chunk = np.ceil(len(jobs) / num_jobs).astype(int)
                ids = np.array(other_indices)  # Use original indices, not sequential

                jobs = [
                    (
                        ids[i : i + chunk],
                        jobs[i : i + chunk],
                        (
                            None
                            if other_smpl_data_list is None
                            else other_smpl_data_list[i : i + chunk]
                        ),
                        (
                            None
                            if other_object_data_list is None
                            else other_object_data_list[i : i + chunk]
                        ),
                        (
                            None
                            if other_soma_data_list is None
                            else other_soma_data_list[i : i + chunk]
                        ),
                        self.fix_height,
                        target_heading,
                        max_len,
                        is_evaluation,
                    )
                    for i in range(0, len(jobs), chunk)
                ]

                job_args = [jobs[i] for i in range(len(jobs))]
                for i in range(1, len(jobs)):
                    worker_args = (*job_args[i], queue, i)
                    worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
                    worker.start()
                    workers.append(worker)
                res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))

                # Wait for all workers to complete and clean them up
                for worker in workers:
                    worker.join()
                    worker.close()
                workers = []

                for i in progress.track(  # noqa: B007
                    range(len(jobs) - 1), "Gathering results for non-navigation dataset..."
                ):
                    res = queue.get()
                    res_acc.update(res)

                self.res_non_nav_dataset = res_acc.copy()

            # load navigation dataset
            if len(nav_motion_data_list) > 0:

                jobs = nav_motion_data_list
                chunk = np.ceil(len(jobs) / num_jobs).astype(int)
                ids = np.array(nav_indices)  # Use original indices, not sequential

                jobs = [
                    (
                        ids[i : i + chunk],
                        jobs[i : i + chunk],
                        nav_smpl_data_list[
                            i : i + chunk
                        ],  # navigation dataset would never have smpl data. This is always empty.
                        (
                            None
                            if nav_object_data_list is None
                            else nav_object_data_list[i : i + chunk]
                        ),
                        (None if nav_soma_data_list is None else nav_soma_data_list[i : i + chunk]),
                        self.fix_height,
                        target_heading,
                        max_len,
                        is_evaluation,
                    )
                    for i in range(0, len(jobs), chunk)
                ]
                job_args = [jobs[i] for i in range(len(jobs))]
                for i in range(1, len(jobs)):
                    worker_args = (*job_args[i], queue, i)
                    worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
                    worker.start()
                    workers.append(worker)
                res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))

                for i in progress.track(  # noqa: B007
                    range(len(jobs) - 1), "Gathering results for navigation dataset..."
                ):
                    res = queue.get()
                    res_acc.update(res)

                # Wait for all workers to complete and clean them up
                for worker in workers:
                    worker.join()
                    worker.close()
                workers = []

        else:
            jobs = motion_data_list
            chunk = np.ceil(len(jobs) / num_jobs).astype(int)
            ids = np.arange(len(jobs))

            jobs = [
                (
                    ids[i : i + chunk],
                    jobs[i : i + chunk],
                    None if smpl_data_list is None else smpl_data_list[i : i + chunk],
                    None if object_data_list is None else object_data_list[i : i + chunk],
                    None if soma_data_list is None else soma_data_list[i : i + chunk],
                    self.fix_height,
                    target_heading,
                    max_len,
                    is_evaluation,
                )
                for i in range(0, len(jobs), chunk)
            ]

            job_args = [jobs[i] for i in range(len(jobs))]
            for i in range(1, len(jobs)):
                worker_args = (*job_args[i], queue, i)
                worker = mp.Process(target=self.load_motion_with_skeleton, args=worker_args)
                worker.start()
                workers.append(worker)
            res_acc.update(self.load_motion_with_skeleton(*jobs[0], None, 0))

            for i in progress.track(range(len(jobs) - 1), "Gathering results..."):  # noqa: B007
                res = queue.get()
                res_acc.update(res)

            nav_indices = []
            other_indices = list(range(len(motions)))

            # Wait for all workers to complete and clean them up
            for worker in workers:
                worker.join()
                worker.close()
            workers = []

        for f in progress.track(range(len(res_acc)), description="Processing motions..."):
            motion_file_data, curr_motion = res_acc[f]
            motion_fps = int(curr_motion.fps * self.motion_fps_scale)
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]

            curr_len = 1.0 / motion_fps * (num_frames - 1)

            if "beta" in motion_file_data:
                _motion_aa.append(motion_file_data["pose_aa"].reshape(-1, self.num_joints * 3))
                _motion_bodies.append(curr_motion.gender_beta)
            else:
                _motion_aa.append(np.zeros((num_frames, self.num_joints * 3)))
                _motion_bodies.append(torch.zeros(17))

            _motion_fps.append(motion_fps)
            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            if self.has_action:
                _motion_actions.append(curr_motion.action)
            if self.smpl_data is not None:
                _motion_smpl_poses.append(curr_motion["smpl_pose"])
                if "smpl_joints" in curr_motion:
                    _motion_smpl_joints.append(curr_motion["smpl_joints"])
                if "smpl_transl" in curr_motion:
                    _motion_smpl_transl.append(curr_motion["smpl_transl"])
            if self.soma_data is not None:
                if "soma_joints" in curr_motion:
                    _motion_soma_joints.append(curr_motion["soma_joints"])
                if "soma_root_quat" in curr_motion:
                    _motion_soma_root_quat.append(curr_motion["soma_root_quat"])
                if "soma_transl" in curr_motion:
                    _motion_soma_transl.append(curr_motion["soma_transl"])
            if self.object_data is not None:
                if "object_root_pos" in curr_motion:
                    _motion_object_root_pos.append(curr_motion["object_root_pos"])
                if "object_root_quat" in curr_motion:
                    _motion_object_root_quat.append(curr_motion["object_root_quat"])
                if "object_contact_center_left" in curr_motion:
                    _motion_object_contact_center_left.append(
                        curr_motion["object_contact_center_left"]
                    )
                if "object_in_contact_left" in curr_motion:
                    _motion_object_in_contact_left.append(curr_motion["object_in_contact_left"])
                if "object_contact_center_right" in curr_motion:
                    _motion_object_contact_center_right.append(
                        curr_motion["object_contact_center_right"]
                    )
                if "object_in_contact_right" in curr_motion:
                    _motion_object_in_contact_right.append(curr_motion["object_in_contact_right"])
                if "hand_action_left" in motion_file_data:
                    raw_action = motion_file_data["hand_action_left"]
                    # Nearest-neighbor interpolation to match target fps
                    src_len = len(raw_action)
                    if src_len != num_frames:
                        indices = np.round(np.linspace(0, src_len - 1, num_frames)).astype(int)
                        raw_action = raw_action[indices]
                    _motion_hand_action_left.append(raw_action)
                if "hand_action_right" in motion_file_data:
                    raw_action = motion_file_data["hand_action_right"]
                    # Nearest-neighbor interpolation to match target fps
                    src_len = len(raw_action)
                    if src_len != num_frames:
                        indices = np.round(np.linspace(0, src_len - 1, num_frames)).astype(int)
                        raw_action = raw_action[indices]
                    _motion_hand_action_right.append(raw_action)
            del curr_motion

        self._motion_lengths = torch.tensor(
            _motion_lengths, device=self._device, dtype=torch.float32
        )
        self._motion_fps = torch.tensor(_motion_fps, device=self._device, dtype=torch.float32)
        self._motion_bodies = torch.stack(_motion_bodies).to(self._device).type(torch.float32)
        self._motion_aa = torch.tensor(
            np.concatenate(_motion_aa), device=self._device, dtype=torch.float32
        )

        if self.smpl_data is not None:
            self._motion_smpl_poses = torch.cat(_motion_smpl_poses, dim=0).float().to(self._device)
            if len(_motion_smpl_joints) > 0:
                self._motion_smpl_joints = (
                    torch.cat(_motion_smpl_joints, dim=0).float().to(self._device)
                )
            if len(_motion_smpl_transl) > 0:
                self._motion_smpl_transl = (
                    torch.cat(_motion_smpl_transl, dim=0).float().to(self._device)
                )
        if self.soma_data is not None:
            if len(_motion_soma_joints) > 0:
                self._motion_soma_joints = (
                    torch.cat(_motion_soma_joints, dim=0).float().to(self._device)
                )
            if len(_motion_soma_root_quat) > 0:
                self._motion_soma_root_quat = (
                    torch.cat(_motion_soma_root_quat, dim=0).float().to(self._device)
                )
            if len(_motion_soma_transl) > 0:
                self._motion_soma_transl = (
                    torch.cat(_motion_soma_transl, dim=0).float().to(self._device)
                )
        if self.object_data is not None:
            if len(_motion_object_root_pos) > 0:
                self._motion_object_root_pos = (
                    torch.cat(_motion_object_root_pos, dim=0).float().to(self._device)
                )
            if len(_motion_object_root_quat) > 0:
                self._motion_object_root_quat = (
                    torch.cat(_motion_object_root_quat, dim=0).float().to(self._device)
                )
            # Store per-hand contact centers and in_contact labels
            if len(_motion_object_contact_center_left) > 0:
                self._motion_object_contact_center_left = (
                    torch.cat(_motion_object_contact_center_left, dim=0).float().to(self._device)
                )
            if len(_motion_object_in_contact_left) > 0:
                self._motion_object_in_contact_left = (
                    torch.cat(_motion_object_in_contact_left, dim=0).float().to(self._device)
                )
            if len(_motion_object_contact_center_right) > 0:
                self._motion_object_contact_center_right = (
                    torch.cat(_motion_object_contact_center_right, dim=0).float().to(self._device)
                )
            if len(_motion_object_in_contact_right) > 0:
                self._motion_object_in_contact_right = (
                    torch.cat(_motion_object_in_contact_right, dim=0).float().to(self._device)
                )
            if len(_motion_hand_action_left) > 0:
                self._motion_hand_action_left = (
                    torch.from_numpy(np.concatenate(_motion_hand_action_left, axis=0))
                    .float()
                    .to(self._device)
                )
            if len(_motion_hand_action_right) > 0:
                self._motion_hand_action_right = (
                    torch.from_numpy(np.concatenate(_motion_hand_action_right, axis=0))
                    .float()
                    .to(self._device)
                )
        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)

        # Compute object velocities from position/quaternion using finite differences
        if self.object_data is not None and hasattr(self, "_motion_object_root_pos"):
            self._compute_object_velocities(_motion_num_frames, _motion_dt)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)

        if self.has_action:
            self._motion_actions = torch.cat(_motion_actions, dim=0).float().to(self._device)
        self._num_motions = len(motions)

        self.body_pos_w = (
            torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        )
        self.body_quat_w = (
            torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        )
        self.body_pos_b = (
            torch.cat([m.local_rotation for m in motions], dim=0).float().to(self._device)
        )
        self.root_linv_vel_w = (
            torch.cat([m.global_root_velocity for m in motions], dim=0).float().to(self._device)
        )
        self.root_ang_vel_w = (
            torch.cat([m.global_root_angular_velocity for m in motions], dim=0)
            .float()
            .to(self._device)
        )
        self.body_ang_vel_w = (
            torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        )
        self.body_lin_vel_w = (
            torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        )
        self.dof_vel = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        self.feet_l = torch.cat([m.feet_l for m in motions], dim=0).float().to(self._device)
        self.feet_r = torch.cat([m.feet_r for m in motions], dim=0).float().to(self._device)

        # if "global_translation_extend" in motions[0].__dict__:
        #     self.body_pos_t_w = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)  # noqa: E501
        #     self.body_quat_t_w = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)  # noqa: E501
        #     self.body_lin_vel_t_w = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)  # noqa: E501
        #     self.body_ang_vel_t_w = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)  # noqa: E501
        #     self.feet_l = torch.cat([m.feet_l for m in motions], dim=0).float().to(self._device)
        #     self.feet_r = torch.cat([m.feet_r for m in motions], dim=0).float().to(self._device)

        if "dof_pos" in motions[0].__dict__:
            self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)

        # Store hand DOF positions if available (for 43-DOF motion)
        if "hand_dof_pos" in motions[0].__dict__:
            self.hand_dof_pos = (
                torch.cat([m.hand_dof_pos for m in motions], dim=0).float().to(self._device)
            )
        else:
            self.hand_dof_pos = None

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        # Zero out initial root XY so all motions start at origin
        if self.m_cfg.get("zero_root_xy", False):
            print(  # noqa: T201
                f"[zero_root_xy] Zeroing initial root XY for {len(motions)} motions"
            )  # noqa: RUF100, T201
            for i in range(len(motions)):
                start = self.length_starts[i]
                end = start + self._motion_num_frames[i]
                init_xy = self.body_pos_w[start, 0, :2].clone()  # root body, XY
                print(  # noqa: T201
                    f"  Motion {i}: init_xy=[{init_xy[0]:.3f}, {init_xy[1]:.3f}], frames={self._motion_num_frames[i]}"  # noqa: E501
                )
                self.body_pos_w[start:end, :, :2] -= init_xy
                if (
                    hasattr(self, "_motion_object_root_pos")
                    and self._motion_object_root_pos is not None
                ):
                    self._motion_object_root_pos[start:end, :, :2] -= init_xy

        self.motion_ids = torch.arange(len(motions), dtype=torch.long, device=self._device)

        motion_has_smpl = [
            self.curr_motion_keys[i] in self.smpl_data_keys for i in range(len(motions))
        ]
        self.motion_has_smpl = torch.tensor(motion_has_smpl, dtype=torch.bool, device=self._device)

        motion_has_soma = [
            self.curr_motion_keys[i] in self.soma_data_keys for i in range(len(motions))
        ]
        self.motion_has_soma = torch.tensor(motion_has_soma, dtype=torch.bool, device=self._device)

        motion_has_object = [
            self.curr_motion_keys[i] in self.object_data_keys for i in range(len(motions))
        ]
        self.motion_has_object = torch.tensor(
            motion_has_object, dtype=torch.bool, device=self._device
        )

        motion = motions[0]  # noqa: F841
        self.num_bodies = self.num_joints

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        if self.use_adaptive_sampling:
            self.update_adaptive_sampling_motion_frames()

        logger.info(
            f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.body_pos_w.shape[0]} frames."  # noqa: E501
        )

        del (
            motions,
            _motion_lengths,
            _motion_fps,
            _motion_dt,
            _motion_num_frames,
            _motion_bodies,
            _motion_aa,
            _motion_actions,
            _motion_smpl_poses,
            _motion_smpl_joints,
            _motion_smpl_transl,
            _motion_object_root_pos,
            _motion_object_root_quat,
        )
        gc.collect()
        torch.cuda.empty_cache()

        if "mujoco_to_isaaclab_body" in self.m_cfg.keys():  # noqa: SIM118
            self.dof_pos = self.dof_pos[:, self.m_cfg.mujoco_to_isaaclab_dof]
            self.dof_vel = self.dof_vel[:, self.m_cfg.mujoco_to_isaaclab_dof]

            # Keep full body data (all bodies, IsaacLab order) before slicing
            self.num_bodies_full = len(self.m_cfg.mujoco_to_isaaclab_body)
            self.body_pos_w_full = self.body_pos_w[:, self.m_cfg.mujoco_to_isaaclab_body]
            self.body_quat_w_full = rotations.xyzw_to_wxyz(
                self.body_quat_w[:, self.m_cfg.mujoco_to_isaaclab_body]
            )
            self.body_lin_vel_w_full = self.body_lin_vel_w[:, self.m_cfg.mujoco_to_isaaclab_body]
            self.body_ang_vel_w_full = self.body_ang_vel_w[:, self.m_cfg.mujoco_to_isaaclab_body]

            # Slice to only selected body_indexes
            self.body_pos_w = self.body_pos_w_full[:, self.body_indexes]
            self.body_quat_w = self.body_quat_w_full[:, self.body_indexes]
            self.body_lin_vel_w = self.body_lin_vel_w_full[:, self.body_indexes]
            self.body_ang_vel_w = self.body_ang_vel_w_full[:, self.body_indexes]
            assert (
                self.m_cfg.get("anchor_body_idx_full", 0) == 0 and self.body_indexes[0] == 0
            ), "The anchor body has to be 0; otherwise will cause issues in the sliced body_indexes data's anchor."
        else:
            # No body reordering — full body data is the same as the original data
            self.body_pos_w_full = self.body_pos_w
            self.body_quat_w_full = self.body_quat_w
            self.body_lin_vel_w_full = self.body_lin_vel_w
            self.body_ang_vel_w_full = self.body_ang_vel_w
            self.num_bodies_full = self.body_pos_w.shape[2]

    def foot_detect(self, positions, vel_thres, height_thresh):
        fid_l = self.m_cfg.get("left_foot_body_idx", [6])
        fid_r = self.m_cfg.get("right_foot_body_idx", [12])
        # fid_l, fid_r = [6], [12]
        velfactor = torch.tensor(
            [vel_thres] * len(fid_l), device=positions.device, dtype=positions.dtype
        )
        heightfactor = torch.tensor(
            [height_thresh] * len(fid_l), device=positions.device, dtype=positions.dtype
        )

        feet_l_xyz = (positions[1:, fid_l] - positions[:-1, fid_l]) ** 2
        feet_l_xyz = torch.cat([feet_l_xyz, feet_l_xyz[[-1]]], dim=0)
        feet_l_h = positions[:, fid_l, 2]
        feet_l = torch.logical_and(
            (feet_l_xyz.sum(dim=-1)) < velfactor, feet_l_h < heightfactor
        ).float()
        # feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).float()

        feet_r_xyz = (positions[1:, fid_r] - positions[:-1, fid_r]) ** 2
        feet_r_xyz = torch.cat([feet_r_xyz, feet_r_xyz[[-1]]], dim=0)
        feet_r_h = positions[:, fid_r, 2]
        feet_r = torch.logical_and(
            (feet_r_xyz.sum(dim=-1)) < velfactor, feet_r_h < heightfactor
        ).float()
        # feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).float()
        return feet_l, feet_r

    def _compute_object_velocities(self, motion_num_frames, motion_dt):
        """Compute object linear and angular velocities from position and quaternion data.
        Uses finite differences: v = (p_{t+1} - p_t) / dt
        Handles motion boundaries properly (first frame uses forward difference).
        """  # noqa: D205
        total_frames = self._motion_object_root_pos.shape[0]
        num_objects = self._motion_object_root_pos.shape[1]

        # Initialize velocity tensors
        self._motion_object_lin_vel = torch.zeros_like(self._motion_object_root_pos)
        self._motion_object_ang_vel = torch.zeros(
            total_frames, num_objects, 3, device=self._device, dtype=torch.float32
        )

        # Compute length_starts for indexing (cumsum of frame counts)
        num_frames_tensor = torch.tensor(motion_num_frames, device=self._device)
        lengths_shifted = num_frames_tensor.roll(1)
        lengths_shifted[0] = 0
        length_starts = lengths_shifted.cumsum(0)

        # Compute velocities for each motion sequence separately
        for i, (start, num_frames, dt) in enumerate(  # noqa: B007
            zip(length_starts, motion_num_frames, motion_dt)  # noqa: B905
        ):
            start = start.item()  # noqa: PLW2901
            end = start + num_frames

            if num_frames < 2:
                continue  # Cannot compute velocity with less than 2 frames

            # Get position and quaternion for this motion
            pos = self._motion_object_root_pos[start:end]  # (T, N_obj, 3)
            quat = self._motion_object_root_quat[start:end]  # (T, N_obj, 4)

            # Compute linear velocity: v = (p_{t+1} - p_t) / dt
            lin_vel = (pos[1:] - pos[:-1]) / dt
            # First frame uses same velocity as second frame
            lin_vel = torch.cat([lin_vel[:1], lin_vel], dim=0)
            self._motion_object_lin_vel[start:end] = lin_vel

            # Compute angular velocity from quaternion difference using same method as robot body
            # ω = axis * angle / dt (same as _compute_angular_velocity in torch_humanoid_batch.py)
            q_curr = quat[:-1]  # (T-1, N_obj, 4)
            q_next = quat[1:]  # (T-1, N_obj, 4)

            # Compute quaternion difference: q_diff = q_next * q_curr^{-1}
            # Using quat_mul_norm and quat_inverse (w_last=False for xyzw format)
            diff_quat = rotations.quat_mul_norm(
                q_next, rotations.quat_inverse(q_curr, w_last=False), w_last=False
            )

            # Extract angle and axis from quaternion difference
            diff_angle, diff_axis = rotations.quat_angle_axis(diff_quat, w_last=False)

            # Angular velocity: ω = axis * angle / dt
            ang_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
            # First frame uses same velocity as second frame
            ang_vel = torch.cat([ang_vel[:1], ang_vel], dim=0)
            self._motion_object_ang_vel[start:end] = ang_vel

        logger.info(f"Computed object velocities for {len(motion_num_frames)} motions")

    def fix_trans_height(self, pose_aa, trans, fix_height_mode):
        if fix_height_mode == FixHeightMode.no_fix:
            return trans, 0
        with torch.no_grad():

            mesh_obj = self.mesh_parsers.mesh_fk(pose_aa[None, :1], trans[None, :1])
            height_diff = np.asarray(mesh_obj.vertices)[..., 2].min()
            trans[..., 2] -= height_diff

            return trans, height_diff

    def load_motion_with_skeleton(
        self,
        ids,
        motion_data_list,
        smpl_data_list,
        object_data_list,
        soma_data_list,
        fix_height,
        target_heading,  # noqa: ARG002
        max_len,
        is_evaluation,
        queue,
        pid,
    ):
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}

        if pid == 0:  # noqa: SIM108
            pbar = progress.track(range(len(ids)), description="Loading motions...")
        else:
            pbar = range(len(ids))

        for f in pbar:

            curr_id = ids[f]  # id for this datasample

            curr_file = motion_data_list[f]
            if "path" in curr_file:
                curr_file, *_ = joblib.load(
                    curr_file["path"]
                ).values()  # First value since it's a single item dictionary

            seq_len = curr_file["root_trans_offset"].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file["root_trans_offset"]).clone()[start:end]
            pose_aa = to_torch(curr_file["pose_aa"][start:end]).clone()

            # import ipdb; ipdb.set_trace()
            if "action" in curr_file.keys():  # noqa: SIM118
                self.has_action = True

            if "fps" not in curr_file.keys():  # noqa: SIM118
                curr_file["fps"] = 30.0
            dt = 1 / curr_file["fps"]  # noqa: F841

            B, J, N = pose_aa.shape
            freeze_frame_aug, freeze_idx = False, 0

            # self.m_cfg.freeze_frame_aug=True; is_evaluation=False; self.m_cfg.freeze_frame_prob=1
            # Debugging, force freeze frame augmentation

            if not is_evaluation and self.m_cfg.get("freeze_frame_aug", False):
                freeze_prob = self.m_cfg.get("freeze_frame_prob", 0.1)
                if np.random.random() < freeze_prob:  # noqa: NPY002
                    # Freeze the sequence at a random index
                    freeze_frame_aug = True
                    freeze_idx = np.random.randint(0, B)  # noqa: NPY002
                    # Repeat the frozen frame for all subsequent frames
                    pose_aa[freeze_idx:] = pose_aa[freeze_idx : freeze_idx + 1].clone()
                    trans[freeze_idx:] = trans[freeze_idx : freeze_idx + 1].clone()

            if not is_evaluation and self.m_cfg.get("randomize_heading", False):
                # ZL: this randomization is not combatiable with SMPL
                random_rot = np.zeros(3)
                random_rot[2] = np.pi * (2 * np.random.random() - 1.0)  # noqa: NPY002
                random_heading_rot = transform.Rotation.from_euler("xyz", random_rot)
                pose_aa = pose_aa.reshape(B, -1)
                pose_aa[:, :3] = torch.tensor(
                    (
                        random_heading_rot * transform.Rotation.from_rotvec(pose_aa[:, :3])
                    ).as_rotvec()
                )
                trans = torch.matmul(
                    trans, torch.from_numpy(random_heading_rot.as_matrix().T).float()
                )
                pose_aa = pose_aa.reshape(B, J, N)

            # self.cat_upper_body_poses_prob of the time, randomize the upper body poses and only for the motions are generated kinematically.  # noqa: E501
            randomize_upper_body_poses = (
                self.randomize_upper_body_poses
                and random.random() < self.cat_upper_body_poses_prob
                and (self._should_augment_upper_body(self.curr_motion_keys[curr_id]))
            )

            # only randomize the upper body poses if the non-navigation dataset is loaded
            if (
                randomize_upper_body_poses
                and self.res_non_nav_dataset is not None
                and len(self.res_non_nav_dataset) > 0
            ):
                # ZL: this randomization is not combatiable with SMPL, so only for kinematic generated data.
                # find the index for the upper body, skip the first index in pose_aa as it is the root.
                upper_body_indices = [
                    i for i in range(1, J) if i - 1 not in self.m_cfg.lower_joint_indices_mujoco
                ]
                # randomly select a motion from the non-navigation dataset
                selected_file, selected_motion = random.choice(
                    list(self.res_non_nav_dataset.values())
                )
                selected_pose_aa = to_torch(selected_file["pose_aa"])

                # Sample a matching slice from the selected motion
                # Use the same method as main code to determine sequence length
                selected_seq_len = selected_file["root_trans_offset"].shape[0]
                current_seq_len = pose_aa.shape[0]
                if selected_seq_len >= current_seq_len:
                    selected_start = random.randint(0, selected_seq_len - current_seq_len)
                    selected_end = selected_start + current_seq_len
                    selected_slice = selected_pose_aa[selected_start:selected_end]
                else:
                    # If selected motion is shorter, create a ping-pong (forward then backward) sequence
                    forward = selected_pose_aa
                    backward = selected_pose_aa.flip(dims=[0])  # reverse the sequence
                    # Concatenate forward and backward, excluding the last frame of forward to avoid duplication
                    extended = torch.cat([forward, backward[1:]], dim=0)

                    # If still not long enough, repeat the extended sequence
                    if extended.shape[0] < current_seq_len:
                        repeats = (current_seq_len + extended.shape[0] - 1) // extended.shape[
                            0
                        ]  # ceiling division
                        extended = extended.repeat(repeats, 1, 1)

                    selected_slice = extended[:current_seq_len]

                pose_aa[:, upper_body_indices] = selected_slice[:, upper_body_indices]

            # Wrist joint noise augmentation
            if (
                not is_evaluation
                and self.randomize_wrist_poses
                and random.random() < self.randomize_wrist_prob
            ):
                wrist_pose_aa_indices = [d + 1 for d in self.wrist_mujoco_dof_indices]
                noise = torch.randn(B, len(wrist_pose_aa_indices), N) * self.randomize_wrist_std
                pose_aa[:, wrist_pose_aa_indices] = pose_aa[:, wrist_pose_aa_indices] + noise

            if self.mesh_parsers is not None:
                trans, trans_fix = self.fix_trans_height(pose_aa, trans, fix_height_mode=fix_height)
                curr_motion = self.mesh_parsers.fk_batch(
                    pose_aa[None,],
                    trans[None,],
                    return_full=True,
                    fps=curr_file["fps"],
                    target_fps=self.target_fps,
                    interpolate_data=True,
                    use_parallel_fk=self.use_parallel_fk,
                )
                if self.smpl_data is not None:
                    curr_smpl_data = smpl_data_list[f]
                    if curr_smpl_data is not None:
                        if "path" in curr_smpl_data:
                            curr_smpl_data = joblib.load(curr_smpl_data["path"])

                        if curr_smpl_data["fps"] != self.target_fps:
                            smpl_pose = torch.tensor(curr_smpl_data["pose_aa"][start:end]).float()
                            smpl_pose[:, -6:] = 0.0
                            curr_motion["smpl_pose"] = self.mesh_parsers.interploate_pose(
                                None, smpl_pose[None,], curr_smpl_data["fps"], self.target_fps
                            )[1][0]
                        else:
                            smpl_pose = torch.tensor(curr_smpl_data["pose_aa"]).float()
                            smpl_pose[:, -6:] = 0.0
                            # new_seq_len = curr_motion['global_translation'].shape[1]
                            curr_motion["smpl_pose"] = smpl_pose
                        if "smpl_joints" in curr_smpl_data:
                            smpl_joints = torch.tensor(curr_smpl_data["smpl_joints"]).float()
                            curr_motion["smpl_joints"] = smpl_joints
                            if (
                                curr_motion["smpl_joints"].shape[0]
                                != curr_motion["global_translation"].shape[1]
                            ):
                                print(  # noqa: T201
                                    f"Length mismatch: smpl_joints={curr_motion['smpl_joints'].shape[0]}, "
                                    f"global_translation={curr_motion['global_translation'].shape[1]}"
                                )
                                print(smpl_data_list[f], motion_data_list[f])  # noqa: T201

                            assert (
                                curr_motion["smpl_joints"].shape[0]
                                == curr_motion["global_translation"].shape[1]
                            )
                        else:
                            num_frames = curr_motion["global_translation"].shape[1]
                            curr_motion["smpl_joints"] = torch.zeros(num_frames, 24, 3).to(
                                curr_motion["global_translation"]
                            )
                        if "transl" in curr_smpl_data:
                            transl = torch.tensor(curr_smpl_data["transl"]).float()
                            curr_motion["smpl_transl"] = transl
                            assert (
                                curr_motion["smpl_transl"].shape[0]
                                == curr_motion["global_translation"].shape[1]
                            )
                        else:
                            num_frames = curr_motion["global_translation"].shape[1]
                            curr_motion["smpl_transl"] = torch.zeros(num_frames, 3).to(
                                curr_motion["global_translation"]
                            )
                        assert (
                            curr_motion["smpl_pose"].shape[0]
                            == curr_motion["global_translation"].shape[1]
                        )

                        if freeze_frame_aug:
                            freeze_idx_new_fps = int(
                                freeze_idx * self.target_fps / curr_file["fps"]
                            )
                            curr_motion["smpl_pose"][freeze_idx_new_fps:] = curr_motion[
                                "smpl_pose"
                            ][freeze_idx_new_fps : freeze_idx_new_fps + 1].clone()
                            curr_motion["smpl_joints"][freeze_idx_new_fps:] = curr_motion[
                                "smpl_joints"
                            ][freeze_idx_new_fps : freeze_idx_new_fps + 1].clone()
                            curr_motion["smpl_transl"][freeze_idx_new_fps:] = curr_motion[
                                "smpl_transl"
                            ][freeze_idx_new_fps : freeze_idx_new_fps + 1].clone()
                    else:
                        curr_motion["smpl_pose"] = torch.zeros(
                            curr_motion["global_translation"].shape[1], 72
                        ).to(curr_motion["global_translation"])
                        curr_motion["smpl_joints"] = torch.zeros(
                            curr_motion["global_translation"].shape[1], 24, 3
                        ).to(curr_motion["global_translation"])
                        curr_motion["smpl_transl"] = torch.zeros(
                            curr_motion["global_translation"].shape[1], 3
                        ).to(curr_motion["global_translation"])
                    # print(curr_motion['smpl_pose'].shape, curr_motion['global_translation'].shape)

                # Load SOMA skeleton data if available
                if soma_data_list is not None:
                    curr_soma_data = soma_data_list[f]
                    if curr_soma_data is not None:
                        if "path" in curr_soma_data:
                            loaded = joblib.load(curr_soma_data["path"])
                            curr_soma_data, *_ = loaded.values()

                        num_frames = curr_motion["global_translation"].shape[1]
                        n_soma = self.num_soma_joints

                        # Resample SOMA data using the canonical interploate_pose formula
                        # to match robot frame count from fk_batch.
                        soma_fps = curr_soma_data.get("fps", self.target_fps)

                        if "soma_joints" in curr_soma_data:
                            soma_joints = torch.tensor(curr_soma_data["soma_joints"]).float()
                            soma_joints_orig_len = soma_joints.shape[0]
                            if soma_fps != self.target_fps:
                                soma_joints = self._resample_soma_tensor(
                                    soma_joints, soma_fps, self.target_fps
                                )
                            curr_motion["soma_joints"] = soma_joints
                            assert soma_joints.shape[0] == num_frames, (
                                f"SOMA soma_joints length {soma_joints.shape[0]} != "
                                f"robot frames {num_frames} "
                                f"(soma_orig={soma_joints_orig_len} @ {soma_fps}fps, "
                                f"robot_orig={seq_len} @ {curr_file['fps']}fps, "
                                f"target_fps={self.target_fps})"
                            )
                        else:
                            curr_motion["soma_joints"] = torch.zeros(num_frames, n_soma, 3).to(
                                curr_motion["global_translation"]
                            )

                        if "soma_root_quat" in curr_soma_data:
                            soma_root_quat = torch.tensor(curr_soma_data["soma_root_quat"]).float()
                            if soma_fps != self.target_fps:
                                soma_root_quat = self._resample_soma_tensor(
                                    soma_root_quat, soma_fps, self.target_fps
                                )
                                # Renormalize quaternions after linear interpolation
                                soma_root_quat = soma_root_quat / (
                                    soma_root_quat.norm(dim=-1, keepdim=True) + 1e-8
                                )
                            curr_motion["soma_root_quat"] = soma_root_quat
                            assert soma_root_quat.shape[0] == num_frames, (
                                f"SOMA soma_root_quat length {soma_root_quat.shape[0]} != "
                                f"robot frames {num_frames}"
                            )
                        else:
                            curr_motion["soma_root_quat"] = torch.zeros(num_frames, 4).to(
                                curr_motion["global_translation"]
                            )
                            curr_motion["soma_root_quat"][:, 0] = 1.0  # identity in wxyz

                        if "soma_transl" in curr_soma_data:
                            soma_transl = torch.tensor(curr_soma_data["soma_transl"]).float()
                            if soma_fps != self.target_fps:
                                soma_transl = self._resample_soma_tensor(
                                    soma_transl, soma_fps, self.target_fps
                                )
                            curr_motion["soma_transl"] = soma_transl
                            assert soma_transl.shape[0] == num_frames, (
                                f"SOMA soma_transl length {soma_transl.shape[0]} != "
                                f"robot frames {num_frames}"
                            )
                        else:
                            curr_motion["soma_transl"] = torch.zeros(num_frames, 3).to(
                                curr_motion["global_translation"]
                            )

                        if freeze_frame_aug:
                            freeze_idx_new_fps = int(
                                freeze_idx * self.target_fps / curr_file["fps"]
                            )
                            for key in ("soma_joints", "soma_root_quat", "soma_transl"):
                                curr_motion[key][freeze_idx_new_fps:] = curr_motion[key][
                                    freeze_idx_new_fps : freeze_idx_new_fps + 1
                                ].clone()
                    else:
                        num_frames = curr_motion["global_translation"].shape[1]
                        n_soma = self.num_soma_joints
                        curr_motion["soma_joints"] = torch.zeros(num_frames, n_soma, 3).to(
                            curr_motion["global_translation"]
                        )
                        curr_motion["soma_root_quat"] = torch.zeros(num_frames, 4).to(
                            curr_motion["global_translation"]
                        )
                        curr_motion["soma_root_quat"][:, 0] = 1.0  # identity in wxyz
                        curr_motion["soma_transl"] = torch.zeros(num_frames, 3).to(
                            curr_motion["global_translation"]
                        )

                # Load object data if available
                if self.object_data is not None:
                    curr_object_data = object_data_list[f] if object_data_list is not None else None
                    if curr_object_data is not None:
                        if "path" in curr_object_data:
                            loaded = joblib.load(curr_object_data["path"])
                            curr_object_data, *_ = loaded.values()

                        num_frames = curr_motion["global_translation"].shape[1]
                        original_fps = curr_object_data.get("fps", curr_file["fps"])

                        if "root_pos" in curr_object_data:
                            curr_motion["object_root_pos"] = interpolate_translation_data(
                                curr_object_data["root_pos"][start:end],
                                source_fps=original_fps,
                                target_fps=self.target_fps,
                                num_frames=num_frames,
                                max_num_objects=self.max_num_objects,
                                pad_value=0.0,
                            )

                        if "root_quat" in curr_object_data:
                            curr_motion["object_root_quat"] = interpolate_quaternion_data(
                                curr_object_data["root_quat"][start:end],
                                source_fps=original_fps,
                                target_fps=self.target_fps,
                                num_frames=num_frames,
                                max_num_objects=self.max_num_objects,
                            )

                        # Load per-hand contact points, compute contact centers and in_contact labels
                        for hand in ("left_hand", "right_hand"):
                            key = f"contact_points_{hand}"
                            side = hand.split("_")[0]  # "left" or "right"
                            if key in curr_object_data:
                                # Remap dict keys to [start:end] slice so contact frames
                                # align with the sliced root_pos/root_quat data
                                raw_dict = curr_object_data[key]
                                sliced_dict = {
                                    k - start: v for k, v in raw_dict.items() if start <= k < end
                                }
                                center, label = interpolate_contact_center(
                                    sliced_dict,
                                    source_fps=original_fps,
                                    target_fps=self.target_fps,
                                    num_frames=num_frames,
                                )
                                curr_motion[f"object_contact_center_{side}"] = center
                                curr_motion[f"object_in_contact_{side}"] = label
                    else:
                        # Fill with zeros if no object data available
                        num_frames = curr_motion["global_translation"].shape[1]
                        curr_motion["object_root_pos"] = torch.zeros(
                            num_frames, self.max_num_objects, 3
                        ).to(curr_motion["global_translation"])
                        curr_motion["object_root_quat"] = torch.zeros(
                            num_frames, self.max_num_objects, 4
                        ).to(curr_motion["global_translation"])
                        curr_motion["object_root_quat"][
                            :, :, 0
                        ] = 1.0  # w=1 for identity quaternion

                curr_motion = easydict.EasyDict(
                    {
                        k: v.squeeze(dim=-1).squeeze(dim=0) if torch.is_tensor(v) else v
                        for k, v in curr_motion.items()
                    }
                )
                # add "action" to curr_motion
                if self.has_action:
                    curr_motion.action = to_torch(curr_file["action"]).clone()[start:end]

                # Extract hand DOFs if motion file has more than 29 DOFs
                hand_dof_count = self.m_cfg.get("hand_dof_count", 0)
                if hand_dof_count > 0 and "dof" in curr_file:
                    raw_dof = to_torch(curr_file["dof"]).clone()[start:end]
                    if raw_dof.shape[-1] > 29:
                        # Extract hand DOFs (indices 29 onwards) and interpolate to target FPS
                        hand_dof = raw_dof[:, 29 : 29 + hand_dof_count]
                        if curr_file["fps"] != self.target_fps:
                            # Simple linear interpolation for hand DOFs
                            num_target_frames = curr_motion["dof_pos"].shape[0]
                            hand_dof_interp = (
                                torch.nn.functional.interpolate(
                                    hand_dof.T.unsqueeze(0),  # (1, C, T)
                                    size=num_target_frames,
                                    mode="linear",
                                    align_corners=True,
                                )
                                .squeeze(0)
                                .T
                            )  # (T, C)
                            curr_motion.hand_dof_pos = hand_dof_interp
                        else:
                            curr_motion.hand_dof_pos = hand_dof

                if self.vid_smpl_pose is not None:  # for cross embodiment tracking
                    vid_smpl_pose = self.vid_smpl_pose[f]
                    vid_smpl_pose = self.mesh_parsers.interploate_pose(
                        None, vid_smpl_pose[None,], 30.0, self.target_fps
                    )[1][0]
                    if curr_motion["smpl_pose"].shape[0] < vid_smpl_pose.shape[0]:
                        for key in curr_motion.keys():  # noqa: SIM118
                            if isinstance(curr_motion[key], torch.Tensor):
                                curr_motion[key] = torch.cat(
                                    [
                                        curr_motion[key],
                                        torch.zeros(
                                            vid_smpl_pose.shape[0] - curr_motion[key].shape[0],
                                            *curr_motion[key].shape[1:],
                                        ).to(curr_motion[key]),
                                    ],
                                    dim=0,
                                )
                    curr_motion["smpl_pose"][:, :] = vid_smpl_pose[
                        : curr_motion["smpl_pose"].shape[0], :
                    ]
                feet_l, feet_r = self.foot_detect(curr_motion["global_translation"], 0.0005, 0.05)
                curr_motion["feet_l"] = feet_l
                curr_motion["feet_r"] = feet_r
                res[curr_id] = (curr_file, curr_motion)
            else:
                logger.error("No mesh parser found")

        if queue is not None:
            queue.put(res)
        else:
            return res

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    def get_motion_num_steps(self, motion_ids=None):
        if motion_ids is None:
            return (
                (self._motion_num_frames * self._sim_fps / self._motion_fps).floor().int()
            )  # don't use ceil as it will cause frames to be missed.
        else:
            return (
                (self._motion_num_frames[motion_ids] * self._sim_fps / self._motion_fps[motion_ids])
                .floor()
                .int()
            )

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)  # noqa: F841
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_time_steps(self, motion_ids, truncate_time=None):
        motion_time = self.sample_time(motion_ids, truncate_time)
        motion_time_steps = (motion_time * self._sim_fps).floor().int()
        return motion_time_steps

    def sample_motions(self, n):
        motion_ids = torch.multinomial(
            self._sampling_batch_prob, num_samples=n, replacement=True
        ).to(self._device)

        return motion_ids

    def get_motion_ids_in_dataset(self, motion_ids):
        return self._curr_motion_ids[motion_ids]

    def get_motion_length(self, motion_ids=None):
        if motion_ids is None:
            return self._motion_lengths
        else:
            return self._motion_lengths[motion_ids]

    def _calc_frame_blend(self, time, len, num_frames, dt):  # noqa: A002
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip(
            (time - frame_idx0 * dt) / dt, 0.0, 1.0
        )  # clip blend to be within 0 and 1

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _local_rotation_to_dof_smpl(self, local_rot):
        B, J, _ = local_rot.shape
        dof_pos = rotations.quat_to_exp_map(local_rot[:, 1:])
        return dof_pos.reshape(B, -1)

    def init_adaptive_sampling(self):
        """Initialize adaptive sampling data structures over all unique motions.

        Divides every motion clip into fixed-size bins (``bin_size`` frames each) and
        creates per-bin tracking tensors for failure rates and sampling probabilities.
        This enables fine-grained, time-segment-level curriculum learning: bins with
        higher failure rates are sampled more frequently during training.

        NOTE: This operates over ALL unique motions in the dataset (not just the
        currently loaded batch), so bin indices are stable across reloads.
        """
        self.adp_samp_num_frames = torch.zeros(
            self._num_unique_motions, device=self._device, dtype=torch.long
        )
        # Compute motion lengths and frame counts for all unique motions using self._motion_data_keys
        for i, motion_key in enumerate(self._motion_data_keys):
            motion_data = self._motion_data_load[motion_key]

            # Compute motion length and frame count similar to how it's done in load_motions
            # Need to account for interpolation from original fps to target_fps
            if "fps" not in motion_data.keys():  # noqa: SIM118
                motion_data["fps"] = 30.0

            original_fps = motion_data["fps"]
            # Get frame count: prefer metadata 'length', else 'root_trans_offset' shape,
            # else lazy-load from pkl file (directory mode without metadata)
            if "length" in motion_data:
                original_num_frames = motion_data["length"]
            elif "root_trans_offset" in motion_data:
                original_num_frames = motion_data["root_trans_offset"].shape[0]
            elif "path" in motion_data:
                # Directory mode: lazy-load the pkl file to get frame count and fps
                loaded_data, *_ = joblib.load(motion_data["path"]).values()
                original_num_frames = loaded_data["root_trans_offset"].shape[0]
                if "fps" in loaded_data:
                    original_fps = loaded_data["fps"]
            else:
                raise KeyError(
                    f"Cannot determine frame count for motion '{motion_key}': no 'length', 'root_trans_offset', or 'path' key"  # noqa: E501
                )

            original_duration = (original_num_frames - 1) / original_fps

            # Match fk_batch behavior: when fps == target_fps, interpolation is
            # skipped and raw frames are used. Otherwise use the canonical
            # interploate_pose formula (arange with exclusive end).
            if original_fps == self.target_fps:
                num_frames = original_num_frames
            else:
                num_frames = len(torch.arange(0, original_duration, 1 / self.target_fps))
            self.adp_samp_num_frames[i] = num_frames

        # Compute length_starts similar to how it's done in load_motions (using num_frames, not lengths)
        lengths = self.adp_samp_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.adp_samp_length_starts = lengths_shifted.cumsum(0)
        self.adp_samp_total_frames = self.adp_samp_num_frames.sum()
        self.adp_samp_length_starts_mask = torch.zeros(
            self.adp_samp_total_frames, device=self._device, dtype=torch.bool
        )
        self.adp_samp_length_starts_mask[self.adp_samp_length_starts] = True

        # init bins - batch version
        self.adp_samp_bin_size = self.adaptive_sampling_cfg.get("bin_size", 50)
        self.adp_samp_frame_to_bin = torch.zeros(
            self.adp_samp_total_frames, device=self._device, dtype=torch.long
        )

        # Pre-compute all bin information in batch
        all_bins = []
        all_bin_motion_lengths = []
        all_bin_new_motion_masks = []
        all_num_peer_bins = []
        all_motion_to_bins = []

        cur_bin_idx = 0
        for orig_motion_id in range(self._num_unique_motions):
            num_frames = self.adp_samp_num_frames[orig_motion_id]
            frame_start = self.adp_samp_length_starts[orig_motion_id]
            frame_end = (
                self.adp_samp_length_starts[orig_motion_id + 1]
                if orig_motion_id < self._num_unique_motions - 1
                else self.adp_samp_total_frames
            )

            # Create bin starts and ends in batch
            bin_starts = torch.arange(
                0, num_frames, self.adp_samp_bin_size, device=self._device, dtype=torch.long
            )
            bin_ends = torch.minimum(bin_starts + self.adp_samp_bin_size, num_frames)
            num_bins = len(bin_starts)
            motion_ids = torch.full(
                (num_bins,), orig_motion_id, device=self._device, dtype=torch.long
            )
            motion_bins = torch.stack([motion_ids, bin_starts, bin_ends], dim=1)
            all_bins.append(motion_bins)

            # Calculate bin lengths
            bin_lengths = bin_ends - bin_starts
            all_bin_motion_lengths.append(bin_lengths)

            # Create new motion mask (first bin of each motion is True)
            new_motion_mask = torch.zeros(num_bins, device=self._device, dtype=torch.bool)
            new_motion_mask[0] = True
            all_bin_new_motion_masks.append(new_motion_mask)

            # Number of peer bins (same for all bins in this motion)
            peer_bins = torch.full((num_bins,), num_bins, device=self._device, dtype=torch.long)
            all_num_peer_bins.append(peer_bins)

            bin_ids = torch.zeros(num_frames, device=self._device, dtype=torch.long)
            bin_ids[bin_starts[1:]] = 1
            bin_ids = bin_ids.cumsum(0) + cur_bin_idx
            self.adp_samp_frame_to_bin[frame_start:frame_end] = bin_ids

            # Store motion to bins mapping
            motion_bin_indices = torch.arange(
                cur_bin_idx, cur_bin_idx + num_bins, device=self._device, dtype=torch.long
            )
            all_motion_to_bins.append(motion_bin_indices)

            cur_bin_idx += num_bins

        # Concatenate all batch results
        self.adp_samp_bins = torch.cat(all_bins, dim=0)
        self.adp_samp_bin_motion_length = torch.cat(all_bin_motion_lengths, dim=0)
        self.adp_samp_bin_new_motion_mask = torch.cat(all_bin_new_motion_masks, dim=0)
        self.adp_samp_num_peer_bins = torch.cat(all_num_peer_bins, dim=0)
        self.orig_motion_id_to_bins = all_motion_to_bins
        self.adp_samp_num_bins = len(self.adp_samp_bins)

        self.adp_samp_bin_weights = (
            self.adp_samp_bin_motion_length / self.adp_samp_bin_motion_length.float().mean()
        )
        # this will make sure each sequence is sampled equally.
        if self.adaptive_sampling_cfg.get("sequence_length_agnostic", True):
            self.adp_samp_bin_weights = self.adp_samp_bin_weights / self.adp_samp_num_peer_bins

        init_num_failures = self.adaptive_sampling_cfg.get("init_num_failures", 1)
        self.adp_samp_failure_rate_max_over_mean = self.adaptive_sampling_cfg.get(
            "adp_samp_failure_rate_max_over_mean", 50.0
        )
        self.uniform_sampling_rate = self.adaptive_sampling_cfg.get("uniform_sampling_rate", 0.1)

        # Max probability constraints (None = skip, "auto" = use failure_rate_max_over_mean)
        # These prevent over-concentration on challenging motions. See update_adaptive_sampling_probabilities().
        self.max_prob_per_bin_cfg = self.adaptive_sampling_cfg.get("max_prob_per_bin", None)
        self.max_prob_per_motion_cfg = self.adaptive_sampling_cfg.get("max_prob_per_motion", None)
        self.adp_samp_num_failures = (
            torch.ones(self.adp_samp_num_bins, device=self._device, dtype=torch.float32)
            * init_num_failures
        )
        self.adp_samp_num_episodes = (
            torch.ones(self.adp_samp_num_bins, device=self._device, dtype=torch.float32)
            * init_num_failures
        )
        self.adp_samp_failure_rate = torch.ones(
            self.adp_samp_num_bins, device=self._device, dtype=torch.float32
        )
        self.adp_samp_failure_rate_raw = torch.ones(
            self.adp_samp_num_bins, device=self._device, dtype=torch.float32
        )
        self.adp_sampling_prob = (
            torch.ones(self.adp_samp_num_bins, device=self._device, dtype=torch.float64)
            / self.adp_samp_num_bins
        )

    def get_state_dict(self):
        """Return a serializable state dict for checkpointing adaptive sampling stats.

        Returns:
            Dict containing ``adp_samp_num_episodes`` and ``adp_samp_num_failures``
            tensors if adaptive sampling is enabled, otherwise an empty dict.
        """
        state_dict = {}
        if self.use_adaptive_sampling:
            state_dict.update(
                {
                    "adp_samp_num_episodes": self.adp_samp_num_episodes,
                    "adp_samp_num_failures": self.adp_samp_num_failures,
                }
            )
        return state_dict

    def load_state_dict(self, state_dict):
        """Restore adaptive sampling statistics from a checkpoint.

        Validates that the bin count matches before restoring. If it does not match
        (e.g. dataset changed between runs), the load is silently skipped.

        Args:
            state_dict: Dict previously returned by ``get_state_dict()``.
        """
        if self.use_adaptive_sampling and "adp_samp_num_episodes" in state_dict:
            if len(self.adp_samp_num_failures) != len(state_dict["adp_samp_num_failures"]):
                print("Adaptive sampling state dict does not match. Skipping load.")  # noqa: T201
                return

            self.adp_samp_num_episodes[:] = state_dict["adp_samp_num_episodes"].to(self._device)
            self.adp_samp_num_failures[:] = state_dict["adp_samp_num_failures"].to(self._device)
            self.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
        return

    def update_adaptive_sampling(self, failure, motion_ids, motion_time_steps):
        """Update adaptive sampling statistics based on training outcomes.

        Increments episode counts for all sampled bins, and failure counts for bins
        where the policy terminated early. Uses bincount for efficient batched updates
        when multiple environments hit the same bin.

        Args:
            failure: Boolean tensor of shape ``(N,)`` indicating which environments
                terminated due to failure (not timeout).
            motion_ids: Tensor of shape ``(N,)`` with batch-local motion indices.
            motion_time_steps: Tensor of shape ``(N,)`` with the simulation time step
                at which the episode ended (or was sampled).
        """
        # Convert motion_ids to dataset motion ids if needed
        dataset_motion_ids = self.get_motion_ids_in_dataset(motion_ids)

        time_steps = self.adp_samp_length_starts[dataset_motion_ids] + motion_time_steps

        # Handle non-unique dataset_motion_ids by counting occurrences
        if len(time_steps) > 0:
            # Use bincount to count occurrences of each unique ID
            bin_ids = self.adp_samp_frame_to_bin[time_steps]
            counts = torch.bincount(bin_ids, minlength=self.adp_samp_num_bins)
            counts = counts / self.adp_samp_bin_motion_length
            self.adp_samp_num_episodes += counts

        # Update failure counts for failed motions
        if failure.any():
            failed_time_steps = time_steps[failure]
            # Handle non-unique failed motion IDs by counting occurrences
            if len(failed_time_steps) > 0:
                bin_ids = self.adp_samp_frame_to_bin[failed_time_steps]
                failure_counts = torch.bincount(bin_ids, minlength=self.adp_samp_num_bins)
                failure_counts_multiplier = self.adaptive_sampling_cfg.get(
                    "failure_counts_multiplier", 1
                )
                self.adp_samp_num_failures += failure_counts * failure_counts_multiplier

    def sync_and_compute_adaptive_sampling(self, accelerator=None, sync_across_gpus=False):
        """Synchronize adaptive sampling stats across GPUs and recompute probabilities.

        In multi-GPU training, averages episode/failure counts across all processes
        before recomputing the per-bin sampling distribution. Optionally applies
        failure-rate decay to propagate difficulty information to preceding bins.

        Args:
            accelerator: HuggingFace Accelerator instance for multi-GPU gather.
                Required when ``sync_across_gpus=True``.
            sync_across_gpus: Whether to synchronize statistics across distributed
                processes before computing probabilities.
        """
        if not self.use_adaptive_sampling:
            return

        if sync_across_gpus:
            with common.Timer("sync_adaptive_sampling_across_gpus"):
                adp_samp_stats = torch.cat(
                    [self.adp_samp_num_episodes, self.adp_samp_num_failures], dim=-1
                )
                adp_samp_stats_all = accelerator.gather(adp_samp_stats).reshape(
                    -1, *adp_samp_stats.shape
                )
                adp_samp_stats_all = adp_samp_stats_all.mean(dim=0)
                self.adp_samp_num_episodes, self.adp_samp_num_failures = adp_samp_stats_all.chunk(
                    2, dim=-1
                )

        with common.Timer("compute_sampling_prob"):
            failure_rate = self.adp_samp_num_failures / self.adp_samp_num_episodes
            self.adp_samp_failure_rate_raw = failure_rate.clone()
            self.adp_samp_failure_rate = failure_rate
            # This is to compute the failure rate with decay.
            # However, this is very slow and not necessary. We can just sample an offset before the failure happens.  # noqa: E501
            if self.adaptive_sampling_cfg.get("use_failure_rate_decay", False):
                gamma = self.adaptive_sampling_cfg.get("decay_gamma", 0.99)
                num_steps = self.adp_samp_num_episodes.shape[0]
                failure_rate_w_decay = torch.zeros_like(failure_rate)
                for step in reversed(range(num_steps)):
                    if step == num_steps - 1:
                        next_failure_rate = 0
                        next_is_not_terminal = 0.0
                    else:
                        next_failure_rate = failure_rate_w_decay[step + 1]
                        next_is_not_terminal = (
                            1.0 - self.adp_samp_bin_new_motion_mask[step + 1].float()
                        )
                    failure_rate_w_decay[step] = (
                        failure_rate[step] + next_is_not_terminal * gamma * next_failure_rate
                    )
                self.adp_samp_failure_rate = failure_rate_w_decay

            # Compute the sampling probability based on the failure rate
            self.update_adaptive_sampling_probabilities()
        return

    def update_adaptive_sampling_probabilities(self):
        """Recompute per-bin sampling probabilities for the currently loaded motion batch.

        Blends failure-rate-based probabilities with a uniform baseline (controlled by
        ``uniform_sampling_rate``), then applies optional max-probability constraints
        per bin and per motion to prevent over-concentration on outlier sequences.
        See the inline comments for detailed rationale on the constraint design.
        """
        self.adp_samp_failure_rate = self.adp_samp_failure_rate.double()
        self.adp_samp_active_failure_rate = self.adp_samp_failure_rate[
            self.adp_samp_active_motion_bins
        ]
        adp_samp_failure_rate_upper_bound = (
            self.adp_samp_active_failure_rate.mean() * self.adp_samp_failure_rate_max_over_mean
        )
        adp_samp_active_failure_rate_clipped = torch.clip(
            self.adp_samp_active_failure_rate, 0.0, adp_samp_failure_rate_upper_bound
        )
        failure_based_sampling_prob = (
            adp_samp_active_failure_rate_clipped / adp_samp_active_failure_rate_clipped.sum()
        )
        uniform_sampling_prob = torch.ones_like(failure_based_sampling_prob) / len(
            failure_based_sampling_prob
        )
        self.adp_sampling_active_prob = (
            failure_based_sampling_prob * (1 - self.uniform_sampling_rate)
            + uniform_sampling_prob * self.uniform_sampling_rate
        )
        self.adp_sampling_active_prob *= self.adp_samp_bin_weights[self.adp_samp_active_motion_bins]
        self.adp_sampling_active_prob = (
            self.adp_sampling_active_prob / self.adp_sampling_active_prob.sum()
        )

        # ==========================================================================
        # MAX PROBABILITY CONSTRAINTS: Prevent over-concentration on challenging motions
        # ==========================================================================
        # WHY THESE CONSTRAINTS EXIST:
        # ---------------------------
        # Adaptive sampling focuses training on motions with higher failure rates.
        # Without constraints, this can cause several problems:
        #
        # 1. CATASTROPHIC FORGETTING: If one motion has 90% failure rate while others
        #    have 10%, it could dominate sampling → policy forgets "easy" motions.
        #
        # 2. TRAINING INSTABILITY: Narrow sample distribution causes high gradient
        #    variance, leading to unstable training dynamics.
        #
        # 3. OVERFITTING TO OUTLIERS: Some motions may be impossible (bad mocap data,
        #    kinematic infeasibility) but still get sampled heavily, wasting compute.
        #
        # 4. DIVERSITY LOSS: For policies that need to generalize across many motions
        #    (e.g., CHIP_token compliance training with 18k+ clips), diversity is critical.
        #
        # EFFECTS OF THESE CONSTRAINTS:
        # -----------------------------
        # - max_prob_per_bin: No single time-segment can exceed N× its fair share.
        #   Prevents over-sampling one specific "hard moment" in a motion.
        #
        # - max_prob_per_motion: No single motion clip can exceed N× its fair share.
        #   Prevents a single broken/impossible motion from dominating training.
        #
        # CONFIGURATION:
        # --------------
        # - "auto": Uses adp_samp_failure_rate_max_over_mean to set the multiplier
        #           (e.g., if failure_rate_max=2, then max_prob = 2x uniform)
        # - null/not set: SKIP these constraints entirely (for legacy configs)
        # - 0: Explicitly disable the constraint
        # - float value: Set exact max probability threshold
        #
        # For CHIP_token compliance training, we use conservative values (2x) to maintain
        # motion diversity. For other training, higher values (50x+) may be acceptable.
        # ==========================================================================

        # Skip all max_prob constraints if neither is configured (legacy behavior)
        if self.max_prob_per_bin_cfg is None and self.max_prob_per_motion_cfg is None:
            self.adp_sampling_active_prob = self.adp_sampling_active_prob.float()
            assert (self.adp_sampling_active_prob >= 0).all()
            return

        num_active_bins = len(self.adp_samp_active_motion_bins)
        active_orig_motion_ids = self.adp_samp_bins[self.adp_samp_active_motion_bins, 0]
        num_active_motions = len(active_orig_motion_ids.unique())

        # 1. Max probability per bin: no single bin can exceed this fraction of total samples
        if self.max_prob_per_bin_cfg is not None:
            if self.max_prob_per_bin_cfg == "auto":
                # Auto: use adp_samp_failure_rate_max_over_mean as multiplier
                multiplier = self.adp_samp_failure_rate_max_over_mean
                max_prob_per_bin = multiplier / num_active_bins if num_active_bins > 0 else 1.0
            else:
                max_prob_per_bin = (
                    float(self.max_prob_per_bin_cfg) if self.max_prob_per_bin_cfg else 0.0
                )

            # Only apply if constraint is meaningful (more bins than 1/max_prob)
            if max_prob_per_bin > 0 and num_active_bins > 1.0 / max_prob_per_bin:
                self.adp_sampling_active_prob = torch.clamp(
                    self.adp_sampling_active_prob, max=max_prob_per_bin
                )
                self.adp_sampling_active_prob = (
                    self.adp_sampling_active_prob / self.adp_sampling_active_prob.sum()
                )

        # 2. Max probability per motion: aggregate bins per motion and cap total
        if self.max_prob_per_motion_cfg is not None:
            if self.max_prob_per_motion_cfg == "auto":
                # Auto: use adp_samp_failure_rate_max_over_mean as multiplier
                multiplier = self.adp_samp_failure_rate_max_over_mean
                max_prob_per_motion = (
                    multiplier / num_active_motions if num_active_motions > 0 else 1.0
                )
            else:
                max_prob_per_motion = (
                    float(self.max_prob_per_motion_cfg) if self.max_prob_per_motion_cfg else 0.0
                )

            # Only apply if constraint is meaningful (more motions than 1/max_prob)
            if max_prob_per_motion > 0 and num_active_motions > 1.0 / max_prob_per_motion:
                unique_motions = active_orig_motion_ids.unique()

                for motion_id in unique_motions:
                    motion_mask = active_orig_motion_ids == motion_id
                    motion_total_prob = self.adp_sampling_active_prob[motion_mask].sum()

                    if motion_total_prob > max_prob_per_motion:
                        # Scale down all bins belonging to this motion
                        scale_factor = max_prob_per_motion / motion_total_prob
                        self.adp_sampling_active_prob[motion_mask] *= scale_factor

                # Re-normalize after capping
                self.adp_sampling_active_prob = (
                    self.adp_sampling_active_prob / self.adp_sampling_active_prob.sum()
                )
        # ==========================================================================

        self.adp_sampling_active_prob = self.adp_sampling_active_prob.float()
        assert (self.adp_sampling_active_prob >= 0).all()

    def update_adaptive_sampling_motion_sequences(self):
        """Recompute global (full-dataset) motion-level sampling probabilities.

        Called before ``load_motions()`` to determine which motions to load next.
        Aggregates per-bin failure rates into per-motion probabilities and applies
        the same max-probability constraints as the batch-level update.
        """
        self.adp_samp_failure_rate = self.adp_samp_failure_rate.double()
        adp_samp_failure_rate_upper_bound = (
            self.adp_samp_failure_rate.mean() * self.adp_samp_failure_rate_max_over_mean
        )
        adp_samp_failure_rate_clipped = torch.clip(
            self.adp_samp_failure_rate, 0.0, adp_samp_failure_rate_upper_bound
        )
        failure_based_sampling_prob = (
            adp_samp_failure_rate_clipped / adp_samp_failure_rate_clipped.sum()
        )
        uniform_sampling_prob = torch.ones_like(failure_based_sampling_prob) / len(
            failure_based_sampling_prob
        )
        self.adp_sampling_prob = (
            failure_based_sampling_prob * (1 - self.uniform_sampling_rate)
            + uniform_sampling_prob * self.uniform_sampling_rate
        )
        self.adp_sampling_prob *= self.adp_samp_bin_weights
        self.adp_sampling_prob = self.adp_sampling_prob / self.adp_sampling_prob.sum()

        # ==========================================================================
        # MAX PROBABILITY CONSTRAINTS (applied to global bin probabilities)
        # See update_adaptive_sampling_probabilities() for detailed explanation.
        # Skip if neither constraint is configured (legacy behavior).
        # ==========================================================================
        if self.max_prob_per_bin_cfg is None and self.max_prob_per_motion_cfg is None:
            # Sum up the adp_sampling_prob for each motion's frames (no constraints)
            motion_sampling_probs = torch.zeros(self._num_unique_motions, device=self._device)
            for orig_motion_id in range(self._num_unique_motions):
                motion_sampling_probs[orig_motion_id] = self.adp_sampling_prob[
                    self.orig_motion_id_to_bins[orig_motion_id]
                ].sum()
            self._sampling_prob = motion_sampling_probs / motion_sampling_probs.sum()
            return

        num_bins = self.adp_samp_num_bins
        num_motions = self._num_unique_motions

        # Apply max_prob_per_bin constraint if configured
        if self.max_prob_per_bin_cfg is not None:
            if self.max_prob_per_bin_cfg == "auto":
                # Auto: use adp_samp_failure_rate_max_over_mean as multiplier
                multiplier = self.adp_samp_failure_rate_max_over_mean
                max_prob_per_bin = multiplier / num_bins if num_bins > 0 else 1.0
            else:
                max_prob_per_bin = (
                    float(self.max_prob_per_bin_cfg) if self.max_prob_per_bin_cfg else 0.0
                )

            # Only apply if constraint is meaningful (more bins than 1/max_prob)
            if max_prob_per_bin > 0 and num_bins > 1.0 / max_prob_per_bin:
                self.adp_sampling_prob = torch.clamp(self.adp_sampling_prob, max=max_prob_per_bin)
                self.adp_sampling_prob = self.adp_sampling_prob / self.adp_sampling_prob.sum()

        # Sum up the adp_sampling_prob for each motion's frames
        motion_sampling_probs = torch.zeros(self._num_unique_motions, device=self._device)
        for orig_motion_id in range(self._num_unique_motions):
            motion_sampling_probs[orig_motion_id] = self.adp_sampling_prob[
                self.orig_motion_id_to_bins[orig_motion_id]
            ].sum()

        # Apply max_prob_per_motion constraint if configured
        if self.max_prob_per_motion_cfg is not None:
            if self.max_prob_per_motion_cfg == "auto":
                # Auto: use adp_samp_failure_rate_max_over_mean as multiplier
                multiplier = self.adp_samp_failure_rate_max_over_mean
                max_prob_per_motion = multiplier / num_motions if num_motions > 0 else 1.0
            else:
                max_prob_per_motion = (
                    float(self.max_prob_per_motion_cfg) if self.max_prob_per_motion_cfg else 0.0
                )

            # Only apply if constraint is meaningful (more motions than 1/max_prob)
            if max_prob_per_motion > 0 and num_motions > 1.0 / max_prob_per_motion:
                motion_sampling_probs = torch.clamp(motion_sampling_probs, max=max_prob_per_motion)

        self._sampling_prob = motion_sampling_probs / motion_sampling_probs.sum()

    def update_adaptive_sampling_motion_frames(self):
        """Build the active-bin index for the currently loaded motion batch.

        Maps each loaded motion to its corresponding global bins, creating
        ``adp_samp_active_motion_bins`` which is used by
        ``sample_motion_ids_and_time_steps()`` and
        ``update_adaptive_sampling_probabilities()`` to sample and update only
        the bins that correspond to currently loaded motions.
        """
        self.adp_samp_active_motion_bins = []
        self.orig_motion_id_to_motion_ids = torch.zeros(
            self._num_unique_motions, device=self._device, dtype=torch.long
        )
        for motion_id, orig_motion_id in enumerate(self._curr_motion_ids):
            bins = self.orig_motion_id_to_bins[orig_motion_id.item()]
            self.adp_samp_active_motion_bins.append(bins)
            self.orig_motion_id_to_motion_ids[orig_motion_id.item()] = motion_id

            # Validate adaptive sampling frame count matches actual loaded frames
            adp_frames = self.adp_samp_num_frames[orig_motion_id].item()
            loaded_frames = self._motion_num_frames[motion_id].item()
            assert adp_frames == loaded_frames, (
                f"Adaptive sampling frame count mismatch for motion "
                f"{orig_motion_id.item()} (key={self._motion_data_keys[orig_motion_id]}): "
                f"adp_samp={adp_frames}, loaded={loaded_frames}. "
                f"This means init_adaptive_sampling computed a different frame count "
                f"than fk_batch produced at load time."
            )

        self.adp_samp_active_motion_bins = torch.cat(self.adp_samp_active_motion_bins, dim=0)
        self.update_adaptive_sampling_probabilities()

    def sample_motion_ids_and_time_steps(self, n):
        """Sample motion IDs and time steps using adaptive sampling probabilities.

        Draws bins from the active-bin distribution, then samples a random frame
        within each selected bin. Optionally shifts the sampled frame backward by
        a random offset (``pre_failure_sample_window``) so the policy starts
        practicing before the difficult segment.

        Args:
            n: Number of (motion_id, time_step) pairs to sample.

        Returns:
            Tuple of (motion_ids, motion_time_steps) where:
                - motion_ids: ``(n,)`` long tensor with batch-local motion indices.
                - motion_time_steps: ``(n,)`` int tensor with frame indices.
        """
        sampled_bin_ids = torch.multinomial(
            self.adp_sampling_active_prob, num_samples=n, replacement=True
        ).to(self._device)
        bin_ids = self.adp_samp_active_motion_bins[sampled_bin_ids]
        bins = self.adp_samp_bins[bin_ids]
        orig_motion_ids, bin_start, bin_end = bins[:, 0], bins[:, 1], bins[:, 2]
        motion_ids = self.orig_motion_id_to_motion_ids[orig_motion_ids]

        motion_time_steps = (
            torch.rand(len(bin_start), device=bin_start.device) * (bin_end - bin_start)
        ).floor().long() + bin_start
        # Sample motion time steps before failures makes more sense since we need to take actions before the failure happens.  # noqa: E501
        pre_failure_sample_window = self.adaptive_sampling_cfg.get("pre_failure_sample_window", 0)
        if pre_failure_sample_window > 0:
            offset = torch.randint(pre_failure_sample_window, (n,), device=self._device)
            motion_time_steps = (motion_time_steps - offset).clamp_min(0)
        return motion_ids, motion_time_steps.int()
