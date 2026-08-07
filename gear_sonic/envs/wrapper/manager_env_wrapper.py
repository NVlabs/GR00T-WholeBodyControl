from typing import TYPE_CHECKING  # noqa: I001

import numpy as np
from omegaconf import OmegaConf
import omni
from pxr import Gf, UsdGeom
import torch
from loguru import logger
from gear_sonic.trl.utils.common import custom_instantiate

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_rl_env import ManagerBasedEnv

# Import joint index functions (single source of truth)
from gear_sonic.envs.env_utils.joint_utils import get_body_joint_indices, get_hand_joint_indices

# Import visualization markers for contact point visualization
try:
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    import isaaclab.sim as sim_utils

    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False


class ManagerEnvWrapper:
    def __init__(self, env: "ManagerBasedEnv", config):
        env.wrapper = self
        self.env = env
        self.config = OmegaConf.create(config)
        self.device = env.device
        self.viewer_focused = True
        self.is_manager_env = True
        if hasattr(self.env, "num_envs"):
            self.num_envs = self.env.num_envs
        else:
            self.num_envs = self.env.env.unwrapped.num_envs
        try:
            self.motion_command = env.command_manager.get_term("motion")
            self._motion_lib = self.motion_command.motion_lib
        except:  # noqa: E722
            logger.info("No motion lib found")
            self.motion_command = None
            self._motion_lib = None
        try:
            self.force_command = env.command_manager.get_term("force")
        except:  # noqa: E722
            self.force_command = None
        self.is_evaluating = False
        self.start_idx = 0
        self._last_predicted_object_pos = (
            None  # For debug visualization of predicted object position
        )
        if not self.config.headless:
            self.setup_keyboard()

        # Action visualization toggle and state
        self.turn_on_visualization = bool(self.config.get("turn_on_visualization", False)) and (
            not self.config.get("headless", False)
        )
        self._viz_every_n_steps = int(self.config.get("viz_every_n_steps", 1))
        self._plot_action_dim = int(self.config.get("action_plot_dim", 29))
        clip_default = self.config.get("action_clip_value", 1.0)  # noqa: F841
        self._action_ylim = float(self.config.get("action_plot_ylim", 10.0))
        self._plot_window = int(self.config.get("action_plot_window", 200))
        self._step_counter = 0
        self._action_fig = None
        self._action_lines = None
        self._action_hist = None
        self._hist_idx = 0

        self._blit_background = None
        self._blit_supported = True
        self._blit_refresh_interval = int(self.config.get("plot_blit_refresh_interval", 120))

        # Initialize action transform module from config
        self.action_transform_module = None
        self._needs_policy_atm = False  # Default: use policy obs directly
        self._policy_atm_indices = None

        # Initialize joint index mappings (for DOF mismatch handling in replay and step)
        self._body_joint_indices = None
        self._hand_joint_indices = None
        self._setup_replay_joint_indices()  # Setup for replay mode

        # Initialize finger primitive support
        self._use_finger_primitive = self.config.get("use_finger_primitive", False)
        self._finger_primitive_map = None
        if self._use_finger_primitive:
            self._setup_finger_primitives()

        # Latent residual mode: policy outputs residual added to token latent space
        self._use_latent_residual = self.config.get("use_latent_residual", False)

        # Latent residual options (only used when use_latent_residual=True)
        self._latent_residual_mode = self.config.get("latent_residual_mode", "post_quantization")
        self._latent_residual_scale = self.config.get("latent_residual_scale", 1.0)

        # Student direct latent mode: policy outputs FULL latent (not residual)
        # This is used for vision student policies that learned to output full latent
        # directly, bypassing the ATM encoder entirely at inference time.
        # The 64-dim output goes directly to ATM decoder (no encoding step).
        self._use_student_direct_latent = self.config.get("use_student_direct_latent", False)

        if self._use_latent_residual:
            logger.info(
                f"Latent residual enabled: mode={self._latent_residual_mode}, "
                f"scale={self._latent_residual_scale}"
            )

        if self._use_student_direct_latent:
            logger.info(
                "Student direct latent mode enabled: policy output goes directly to ATM decoder "
                "(no encoding step)"
            )

        # Camera extrinsics randomization state
        self._camera_extrinsics_randomized = False
        self._camera_base_transforms = {}  # Store original transforms per env

        action_transform_module_cfg = self.config.get("action_transform_module_cfg", None)

        if action_transform_module_cfg is not None:
            # Load configs from exported YAML file
            with open(action_transform_module_cfg) as f:
                exported_config = OmegaConf.load(f)

            env_config = exported_config.get("env_config", {})
            algo_config = exported_config.get("algo_config", {})

            self.action_transform_module = custom_instantiate(
                algo_config.actor, env_config=env_config, algo_config=algo_config, _resolve=False
            ).to(self.device)
            logger.info(f"Initialized action_transform_module from config: {action_transform_module_cfg}")

            # Load checkpoint if provided
            action_transform_module_checkpoint = self.config.get(
                "action_transform_module_checkpoint", None
            )
            if action_transform_module_checkpoint is not None:
                # Compatibility shim: checkpoints saved with TRL < 0.28.0 reference
                # trl.trainer.utils.OnlineTrainerState, which was moved in 0.28.0
                try:
                    from trl.experimental.ppo.ppo_trainer import OnlineTrainerState, exact_div
                    import trl.trainer.utils

                    trl.trainer.utils.OnlineTrainerState = OnlineTrainerState
                    trl.trainer.utils.exact_div = exact_div
                except ImportError:
                    pass
                checkpoint = torch.load(
                    action_transform_module_checkpoint, map_location=self.device, weights_only=False
                )
                self.action_transform_module.load_state_dict(checkpoint["policy_state_dict"])
                logger.info(
                    f"Loaded action_transform_module checkpoint: {action_transform_module_checkpoint}"
                )

            # Precompute tokenizer observation indices for meta_action target
            self._tokenizer_obs_indices = self._compute_tokenizer_obs_indices()

            # Setup policy_atm observations for action_transform_module if DOF mismatch exists
            self._setup_policy_atm(env_config, algo_config)

        try:
            self.viewer_focused = True
            self.env.viewport_camera_controller.update_view_to_world()
            self.env.viewport_camera_controller.update_view_to_asset_root("robot")
        except:  # noqa: E722
            self.viewer_focused = False

    def _setup_replay_joint_indices(self):
        """Setup joint indices for replay mode if robot has more DOFs than motion lib (29)."""
        if self.env.scene["robot"].num_joints > 29:
            self._setup_action_joint_indices()

    def _compute_tokenizer_obs_indices(self):
        """Compute start and end indices for each tokenizer observation in the flattened tensor."""
        if self.action_transform_module is None:
            return {}

        tokenizer_obs_names = self.action_transform_module.actor_module.tokenizer_obs_names
        tokenizer_obs_dims = self.action_transform_module.actor_module.tokenizer_obs_dims

        indices = {}
        current_index = 0
        for name in tokenizer_obs_names:
            all_dim = int(np.prod(tokenizer_obs_dims[name]))
            indices[name] = (current_index, current_index + all_dim)
            current_index += all_dim

        return indices

    def _setup_policy_atm(self, env_config, algo_config):  # noqa: ARG002
        """Setup policy_atm for action_transform_module when robot has more DOFs than ATM expects."""
        atm_num_joints = env_config.get("robot", {}).get("actions_dim") or env_config.get(
            "robot", {}
        ).get("num_joints", 29)
        self._needs_policy_atm = self.config.get("needs_policy_atm", True)
        self._atm_num_joints = atm_num_joints
        self._current_num_joints = self.env.scene["robot"].num_joints
        if self._current_num_joints > atm_num_joints:
            assert (
                self._needs_policy_atm
            ), "Robot has more DOFs than ATM expects, but needs_policy_atm is False"

        has_policy_atm = (
            hasattr(self.env, "observation_manager")
            and "policy_atm" in self.env.observation_manager._group_obs_term_names  # noqa: SLF001
        )
        self._use_policy_atm_group = self._needs_policy_atm and has_policy_atm

        if self._use_policy_atm_group:
            self._setup_action_joint_indices()

    def _setup_action_joint_indices(self):
        """Compute joint indices for mapping body (29 DOF) and hand (14 DOF) actions."""
        if self._body_joint_indices is not None:
            return

        robot = self.env.scene["robot"]
        self._body_joint_indices = get_body_joint_indices(robot)
        self._hand_joint_indices = get_hand_joint_indices(robot)

    def _setup_finger_primitives(self):
        """Setup finger primitive action mapping from config.

        Finger primitives allow the policy to output 2 actions (left/right gripper)
        instead of 14 individual finger joint actions. Each primitive action
        interpolates between open (pos_0) and closed (pos_1) positions.
        """
        primitive_cfg = self.config.get("finger_primitive", {})
        primitive_action_map = primitive_cfg.get("primitive_action_map", {})

        if not primitive_action_map:
            logger.info("Warning: use_finger_primitive=True but no primitive_action_map defined")
            self._use_finger_primitive = False
            return

        robot = self.env.scene["robot"]
        joint_names = robot.joint_names

        self._finger_primitive_map = []
        for action_name in sorted(primitive_action_map.keys()):
            prim_cfg = primitive_action_map[action_name]
            dof_names = list(prim_cfg.get("dof_names", []))
            pos_0 = torch.tensor(prim_cfg.get("pos_0", []), device=self.device, dtype=torch.float32)
            pos_1 = torch.tensor(prim_cfg.get("pos_1", []), device=self.device, dtype=torch.float32)

            # Find indices in the hand joints
            dof_idx = []
            for dof_name in dof_names:
                if dof_name in joint_names:
                    dof_idx.append(joint_names.index(dof_name))
                else:
                    logger.info(f"Warning: DOF {dof_name} not found in robot joints")
            # Support both "mode" (new) and "discrete" (legacy) config keys
            mode = prim_cfg.get("mode", None)
            if mode is None:
                # Legacy support: discrete=True → mode="discrete", discrete=False → mode="linear"
                mode = "discrete" if prim_cfg.get("discrete", False) else "linear"

            self._finger_primitive_map.append(
                {
                    "action_name": action_name,
                    "dof_names": dof_names,
                    "dof_idx": dof_idx,
                    "pos_0": pos_0,
                    "pos_1": pos_1,
                    "mode": mode,
                }
            )
            logger.info(f"Finger primitive '{action_name}': {len(dof_names)} DOFs, mode={mode}")

        self._num_finger_primitives = len(self._finger_primitive_map)
        logger.info(f"Initialized {self._num_finger_primitives} finger primitives")

    def _convert_primitive_to_finger_actions(self, primitive_actions: torch.Tensor) -> torch.Tensor:
        """Convert primitive actions (num_envs, num_primitives) to finger joint targets (num_envs, 14).

        Follows groot_backup implementation:
        - "linear" mode: maps [-1, 1] → [0, 1] via (x + 1) / 2, then lerps between pos_0 and pos_1
        - "discrete" mode: action >= 0 → pos_1 (closed), action < 0 → pos_0 (open)

        Args:
            primitive_actions: Tensor of shape (num_envs, num_primitives), values in [-1, 1]

        Returns:
            Tensor of shape (num_envs, num_finger_joints) with joint position targets
        """
        num_envs = primitive_actions.shape[0]
        num_finger_joints = (
            len(self._hand_joint_indices) if self._hand_joint_indices is not None else 14
        )
        finger_targets = torch.zeros(
            num_envs, num_finger_joints, device=self.device, dtype=primitive_actions.dtype
        )

        for i, prim_cfg in enumerate(self._finger_primitive_map):
            action = primitive_actions[:, i]  # (num_envs,), values in [-1, 1]
            p0 = prim_cfg["pos_0"]
            p1 = prim_cfg["pos_1"]
            dof_idx = prim_cfg["dof_idx"]
            mode = prim_cfg.get("mode", "linear")
            if mode == "linear":
                # Clamp to [-1, 1] then map to [0, 1]
                action_clamped = action.clamp(min=-1.0, max=1.0)
                t = (action_clamped + 1.0) / 2.0  # (num_envs,)
                joint_targets = torch.lerp(p0.unsqueeze(0), p1.unsqueeze(0), t.unsqueeze(1))
            elif mode == "discrete":
                # action >= 0 → closed (pos_1), action < 0 → open (pos_0)
                joint_targets = torch.where(
                    action.unsqueeze(1) >= 0, p1.unsqueeze(0), p0.unsqueeze(0)
                )
            else:
                raise ValueError(f"Invalid finger primitive mode: {mode}")

            # Map to the correct indices in finger_targets
            # dof_idx are absolute joint indices, need to convert to relative hand indices
            for j, abs_idx in enumerate(dof_idx):
                if self._hand_joint_indices is not None:
                    # Convert tensor to list if needed for .index() lookup
                    hand_indices_list = (
                        self._hand_joint_indices.tolist()
                        if isinstance(self._hand_joint_indices, torch.Tensor)
                        else self._hand_joint_indices
                    )
                    rel_idx = hand_indices_list.index(abs_idx)
                    finger_targets[:, rel_idx] = joint_targets[:, j]
                else:
                    # Fallback: assume hand joints are at the end
                    finger_targets[:, abs_idx - 29] = joint_targets[:, j]

        return finger_targets

    def _prepare_obs_for_action_transform_module(self, obs_dict):
        """Use policy_atm observations for ATM if DOF mismatch exists, else use policy."""
        if not self._use_policy_atm_group or "policy_atm" not in obs_dict:
            atm_obs_dict = obs_dict.copy()
        else:
            atm_obs_dict = obs_dict.copy()
            atm_obs_dict["actor_obs"] = atm_obs_dict["policy_atm"]

        # Ensure all observations have a sequence dimension [num_envs, seq_len, dim]
        # The action_transform_module expects 3D tensors
        for k, v in atm_obs_dict.items():
            if isinstance(v, torch.Tensor) and v.dim() == 2:
                atm_obs_dict[k] = v.unsqueeze(1)  # Add seq_len=1 dimension

        return atm_obs_dict

    def reset_all(self, global_rank=0):  # noqa: ARG002
        return self.reset()

    def process_raw_obs(self, obs, flatten_dict_obs):
        new_obs = {
            "actor_obs": obs["policy"],
            "critic_obs": obs["critic"],
        }
        for k, v in obs.items():
            if k not in ["policy", "critic"]:
                if isinstance(v, dict) and flatten_dict_obs:
                    if k == "height_map":
                        # Special case: do not flatten height map
                        new_obs[k] = v["height_map"]
                        continue
                    if k == "camera_rgb":
                        # Special case: do not flatten camera RGB image
                        # Keep original shape [B, H, W, C] for vision encoder
                        new_obs[k] = v["camera_rgb"]
                        continue
                    obs_names = self.env.observation_manager._group_obs_term_names[k]  # noqa: SLF001
                    new_obs[k] = torch.cat(
                        [v[obs_name].reshape(v[obs_name].shape[0], -1) for obs_name in obs_names],
                        dim=-1,
                    )
                else:
                    new_obs[k] = v
        return new_obs

    def reset(self, flatten_dict_obs=True):
        obs, info = self.env.reset()
        new_obs = self.process_raw_obs(obs, flatten_dict_obs)
        # Initialize success_lift to False for all envs after reset (used unconditionally in step())
        self.env.success_lift = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=self.env.device
        )
        if self.action_transform_module is not None:
            # Store obs for action_transform_module when obs_dict is not provided in step()
            self._last_obs_dict = new_obs
            # Initialize last meta action buffer (policy output: latent + primitives)
            # meta_action_dim = tokenizer_action_dim + hand_action_dim (e.g., 64 + 2 = 66)
            meta_action_dim = self.config.get("meta_action_dim", 66)
            self.env._last_meta_action = torch.zeros(  # noqa: SLF001
                self.env.num_envs, meta_action_dim, dtype=torch.float32, device=self.env.device
            )
            # Previous meta action buffer for meta_action_rate_l2 reward (token smoothness)
            self.env._prev_meta_action = torch.zeros(  # noqa: SLF001
                self.env.num_envs, meta_action_dim, dtype=torch.float32, device=self.env.device
            )
            # Full latent buffers for full_latent_rate_l2 reward (decoder input smoothness)
            # tokenizer_action_dim = latent_dim (e.g., 64 = num_tokens * token_dim)
            tokenizer_action_dim = self.config.get("tokenizer_action_dim", 64)
            self.env._full_latent = torch.zeros(  # noqa: SLF001
                self.env.num_envs, tokenizer_action_dim, dtype=torch.float32, device=self.env.device
            )
            self.env._prev_full_latent = torch.zeros(  # noqa: SLF001
                self.env.num_envs, tokenizer_action_dim, dtype=torch.float32, device=self.env.device
            )

            # Apply camera extrinsics randomization on every reset
            self.apply_random_camera_extrinsics()

        return new_obs

    def apply_random_camera_extrinsics(self):
        """Apply per-environment random camera extrinsics (position and rotation offsets).

        Reads randomization ranges from config:
        - cameras.camera_extrinsics_randomization: Enable/disable switch (default: False)
        - cameras.camera_pos_rand_range: ±meters for x, y, z position
        - cameras.camera_roll_rand_range: ±radians for roll
        - cameras.camera_pitch_rand_range: ±radians for pitch
        - cameras.camera_yaw_rand_range: ±radians for yaw
        """
        cameras_config = self.config.get("cameras", {})

        # Check if randomization is enabled via switch
        if not cameras_config.get("camera_extrinsics_randomization", False):
            return

        # Get randomization ranges
        pos_range = cameras_config.get("camera_pos_rand_range", 0.0)
        roll_range = cameras_config.get("camera_roll_rand_range", 0.0)
        pitch_range = cameras_config.get("camera_pitch_rand_range", 0.0)
        yaw_range = cameras_config.get("camera_yaw_rand_range", 0.0)

        # Check if any randomization values are non-zero
        if pos_range == 0 and roll_range == 0 and pitch_range == 0 and yaw_range == 0:
            return

        # Get camera attached link from config
        camera_attached_link = cameras_config.get("camera_attached_link", None)
        if camera_attached_link is None:
            logger.info("Skipping camera extrinsics randomization: no camera_attached_link configured")
            return

        # Only print details on first call
        if not self._camera_extrinsics_randomized:
            logger.info("Applying random camera extrinsics per environment:")
            logger.info(f"  Position range: ±{pos_range*100:.1f}cm")
            logger.info(f"  Roll range: ±{roll_range:.3f} rad ({np.degrees(roll_range):.1f}°)")
            logger.info(f"  Pitch range: ±{pitch_range:.3f} rad ({np.degrees(pitch_range):.1f}°)")
            logger.info(f"  Yaw range: ±{yaw_range:.3f} rad ({np.degrees(yaw_range):.1f}°)")

        stage = omni.usd.get_context().get_stage()

        for env_id in range(self.env.scene.num_envs):
            camera_prim_path = f"/World/envs/env_{env_id}/Robot/{camera_attached_link}/ego_camera"
            camera_prim = stage.GetPrimAtPath(camera_prim_path)

            if not camera_prim.IsValid():
                if env_id == 0:
                    logger.info(f"  Warning: Camera prim not found at {camera_prim_path}")
                continue

            # Get current camera transform
            xformable = UsdGeom.Xformable(camera_prim)

            # On first call, store the original/base transforms
            if env_id not in self._camera_base_transforms:
                base_translate = None
                base_orient = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        base_translate = Gf.Vec3d(op.Get())  # Make a copy
                    elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        base_orient = Gf.Quatd(op.Get())  # Make a copy
                self._camera_base_transforms[env_id] = {
                    "translate": base_translate,
                    "orient": base_orient,
                }

            # Get base transforms
            base_translate = self._camera_base_transforms[env_id]["translate"]
            base_orient = self._camera_base_transforms[env_id]["orient"]

            # Sample random deltas
            if pos_range > 0:  # noqa: SIM108
                pos_delta = np.random.uniform(-pos_range, pos_range, 3)  # noqa: NPY002
            else:
                pos_delta = np.zeros(3)

            # Sample rotation deltas (roll, pitch, yaw)
            roll_delta = np.random.uniform(-roll_range, roll_range) if roll_range > 0 else 0.0  # noqa: NPY002
            pitch_delta = np.random.uniform(-pitch_range, pitch_range) if pitch_range > 0 else 0.0  # noqa: NPY002
            yaw_delta = np.random.uniform(-yaw_range, yaw_range) if yaw_range > 0 else 0.0  # noqa: NPY002

            # Apply position delta relative to BASE (not current)
            if base_translate is not None:
                new_pos = Gf.Vec3d(
                    base_translate[0] + pos_delta[0],
                    base_translate[1] + pos_delta[1],
                    base_translate[2] + pos_delta[2],
                )
                # Find and update the translate op
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        op.Set(new_pos)
                        break

            # Apply rotation delta relative to BASE (not current)
            if base_orient is not None and (roll_delta != 0 or pitch_delta != 0 or yaw_delta != 0):
                # Convert euler deltas to quaternion
                # Order: roll (X), pitch (Y), yaw (Z)
                cr, sr = np.cos(roll_delta / 2), np.sin(roll_delta / 2)
                cp, sp = np.cos(pitch_delta / 2), np.sin(pitch_delta / 2)
                cy, sy = np.cos(yaw_delta / 2), np.sin(yaw_delta / 2)

                # Quaternion from euler (ZYX convention)
                delta_quat = Gf.Quatd(
                    cr * cp * cy + sr * sp * sy,  # w
                    sr * cp * cy - cr * sp * sy,  # x
                    cr * sp * cy + sr * cp * sy,  # y
                    cr * cp * sy - sr * sp * cy,  # z
                )

                # Compose: new = delta * BASE (not current!)
                new_orient = delta_quat * base_orient

                # Find and update the orient op
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                        op.Set(new_orient)
                        break

            # Store the random deltas for each environment (for projection in observations)
            if not hasattr(self, "_camera_random_deltas"):
                self._camera_random_deltas = {}
            self._camera_random_deltas[env_id] = {
                "pos_delta": pos_delta.tolist() if isinstance(pos_delta, np.ndarray) else [0, 0, 0],
                "roll_delta": float(roll_delta),
                "pitch_delta": float(pitch_delta),
                "yaw_delta": float(yaw_delta),
            }

            if env_id == 0 and not self._camera_extrinsics_randomized:
                logger.info(
                    f"  Env 0: pos_delta={pos_delta}, rot_delta=[{roll_delta:.3f}, {pitch_delta:.3f}, {yaw_delta:.3f}]"  # noqa: E501
                )

        # Mark as initialized (for print suppression)
        self._camera_extrinsics_randomized = True

    def _decode_direct_latent(self, full_latent, atm_obs_dict):
        """Decode full latent directly using ATM decoder (skip encoder).
        Used for student rollout where policy outputs full latent.

        Args:
            full_latent: (batch, latent_dim) full latent from student policy
            atm_obs_dict: observation dict for ATM

        Returns:
            body_actions: (batch, seq_len, action_dim) decoded actions
        """  # noqa: D205
        # Get proprioception for decoder
        if "policy_atm" in atm_obs_dict:  # noqa: SIM108
            proprioception = atm_obs_dict["policy_atm"]
        else:
            proprioception = atm_obs_dict["actor_obs"]

        # Ensure sequence dimension
        if proprioception.dim() == 2:
            proprioception = proprioception.unsqueeze(1)

        atm = self.action_transform_module.actor_module
        return self._decode_direct_latent_batch(full_latent, proprioception, atm)

    def _decode_direct_latent_batch(self, full_latent, proprioception, atm):
        """Decode a batch of full latents directly using ATM decoder.

        The student policy outputs pre-quantization values (latent + residual).
        We apply quantization here before decoding to match the teacher's flow:
        - Teacher: latent + residual -> quantize -> decode
        - Student: direct_latent -> quantize -> decode

        Args:
            full_latent: (batch, latent_dim) full latent (pre-quantization)
            proprioception: (batch, seq, proprio_dim) proprioception input
            atm: ATM actor module

        Returns:
            body_actions: (batch, seq_len, action_dim) decoded actions
        """
        batch_size = full_latent.shape[0]

        # Reshape latent for quantization: (batch, latent_dim) -> (batch, num_tokens, token_dim)
        latent_reshaped = full_latent.view(batch_size, atm.max_num_tokens, atm.token_dim)

        # Apply quantization (same as teacher's flow)
        # This ensures student inference matches teacher: latent -> quantize -> decode
        if atm.quantizer is not None:
            quantized_codes, _ = atm.quantizer(latent_reshaped)
            tokens_for_decode = quantized_codes.contiguous()
        else:
            tokens_for_decode = latent_reshaped

        # Reshape for decoder: (batch, num_tokens, token_dim) -> (batch, 1, num_tokens, token_dim)
        tokens_reshaped = tokens_for_decode.unsqueeze(1)
        tokens_flattened = tokens_for_decode.view(batch_size, -1).unsqueeze(1)

        # Prepare decode input
        decode_input_dict = {
            "token": tokens_reshaped,
            "token_flattened": tokens_flattened,  # (batch, 1, latent_dim)
            "proprioception": proprioception,
        }

        # Decode directly (skip encoding entirely)
        decoded_output = atm.decode("g1_dyn", decode_input_dict)

        # Get body actions from decoder output
        # Note: "meta_action" is for special hierarchical policies, "action" is standard
        body_actions = decoded_output.get("meta_action", decoded_output.get("action"))
        if body_actions is None:
            raise KeyError(
                f"Decoder output missing 'action' or 'meta_action'. Keys: {decoded_output.keys()}"
            )

        return body_actions

    def step(self, actions):
        if self.action_transform_module is not None:
            # Use provided obs_dict or fall back to stored obs from last reset/step
            if "obs_dict" in actions:
                obs_dict = actions["obs_dict"].copy()
            else:
                # Fallback for callbacks that don't provide obs_dict (e.g., im_eval)
                obs_dict = getattr(self, "_last_obs_dict", None)
                if obs_dict is None:
                    raise ValueError(
                        "action_transform_module requires obs_dict but none was provided or stored"
                    )
                obs_dict = obs_dict.copy() if isinstance(obs_dict, dict) else obs_dict
            meta_actions = actions["actions"]
            # Determine action mode: "direct_latent", "residual", or "mixed"
            # Priority: 1) explicit action_mode in actions dict, 2) config flag
            # During training: trainer sets action_mode explicitly
            # During eval: fallback to config flags
            action_mode = actions.get("action_mode", None)
            if action_mode is None:
                # Fallback for eval scripts that don't set action_mode
                if self._use_student_direct_latent:
                    action_mode = "direct_latent"
                elif self._use_latent_residual:
                    action_mode = "residual"
                else:
                    action_mode = "residual"  # Default to residual if nothing specified

            # Shift meta action buffers for meta_action_rate_l2 reward (token smoothness)
            self.env._prev_meta_action = self.env._last_meta_action.clone()  # noqa: SLF001
            # Store meta action for observation (last policy output)
            self.env._last_meta_action = meta_actions.clone()  # noqa: SLF001

            atm_obs_dict = self._prepare_obs_for_action_transform_module(obs_dict)

            # Split actions: first tokenizer_action_dim for tokenizer, rest for hands
            tokenizer_action_dim = self.config.get("tokenizer_action_dim")
            tokenizer_meta_actions = meta_actions[:, :tokenizer_action_dim]
            hand_actions_raw = meta_actions[:, tokenizer_action_dim:]

            # Override hand actions with motion data if configured
            if self.config.get("use_motion_hand_actions", False):
                motion_cmd = self.env.command_manager.get_term("motion")
                left_action = motion_cmd.get_hand_action("left_hand")
                right_action = motion_cmd.get_hand_action("right_hand")

                if left_action is None or right_action is None:
                    raise ValueError(
                        "use_motion_hand_actions=True but hand_action_left/right not found in motion data. "
                        "Ensure processed_robot_motions.pkl contains 'hand_action_left' and 'hand_action_right' arrays."  # noqa: E501
                    )

                # Use motion data directly: -1.0 = open, +1.0 = closed
                # Threshold at 0 in _convert_primitive_to_finger_actions
                hand_actions_raw = torch.stack([left_action, right_action], dim=-1)

            # Convert primitive actions to finger joint targets if enabled
            if self._use_finger_primitive and self._finger_primitive_map:
                # Store raw primitive actions on env for reward computation (before clamping)
                self.env._finger_primitive_actions_raw = hand_actions_raw  # noqa: SLF001
                hand_actions = self._convert_primitive_to_finger_actions(hand_actions_raw)
            else:
                self.env._finger_primitive_actions_raw = None  # noqa: SLF001
                hand_actions = hand_actions_raw

            if action_mode == "direct_latent":
                # Student direct latent mode: policy outputs FULL latent, not residual
                # Skip ATM encoder entirely - go directly to decoder
                body_actions = self._decode_direct_latent(tokenizer_meta_actions, atm_obs_dict)

            elif action_mode == "residual":
                # Teacher/residual mode: policy outputs residual that's added to ATM encoded tokens
                # Apply scaling to residual before passing to ATM
                scaled_residual = tokenizer_meta_actions * self._latent_residual_scale
                # Add residual in latent/token space (after encoding, before decoding)
                body_actions = self.action_transform_module(
                    atm_obs_dict,
                    latent_residual=scaled_residual,
                    latent_residual_mode=self._latent_residual_mode,
                )

            elif action_mode == "mixed":
                # Mixed rollout: some envs use teacher (residual), some use student (direct_latent)
                # is_teacher_env is a boolean mask: True = teacher/residual, False = student/direct_latent
                is_teacher_env = actions.get("is_teacher_env")
                if is_teacher_env is None:
                    raise ValueError(
                        "action_mode='mixed' requires 'is_teacher_env' mask in actions dict"
                    )

                num_envs = tokenizer_meta_actions.shape[0]
                atm = self.action_transform_module.actor_module

                # Get proprioception for decoder (needed for student/direct_latent mode)
                if "policy_atm" in atm_obs_dict:
                    proprioception = atm_obs_dict["policy_atm"]
                else:
                    proprioception = atm_obs_dict["actor_obs"]
                if proprioception.dim() == 2:
                    proprioception = proprioception.unsqueeze(1)

                teacher_mask = is_teacher_env
                student_mask = ~is_teacher_env
                num_teacher = teacher_mask.sum().item()
                num_student = student_mask.sum().item()

                # Initialize placeholders
                teacher_indices = None
                student_indices = None
                teacher_body_actions = None
                student_body_actions = None

                # Process teacher envs (residual mode) if any
                if num_teacher > 0:
                    teacher_indices = teacher_mask.nonzero(as_tuple=True)[0]
                    teacher_latent = tokenizer_meta_actions[teacher_indices]

                    # Prepare teacher obs dict (subset of envs)
                    # Only include keys that ATM actually needs: tokenizer and actor_obs
                    teacher_atm_obs = {}
                    atm_keys = ["tokenizer", "actor_obs"]
                    for obs_key in atm_keys:
                        if obs_key in atm_obs_dict:
                            teacher_atm_obs[obs_key] = atm_obs_dict[obs_key][teacher_indices]

                    scaled_residual = teacher_latent * self._latent_residual_scale
                    teacher_body_actions = self.action_transform_module(
                        teacher_atm_obs,
                        latent_residual=scaled_residual,
                        latent_residual_mode=self._latent_residual_mode,
                    )

                # Process student envs (direct_latent mode) if any
                if num_student > 0:
                    student_indices = student_mask.nonzero(as_tuple=True)[0]
                    student_latent = tokenizer_meta_actions[student_indices]
                    student_proprio = proprioception[student_indices]

                    student_body_actions = self._decode_direct_latent_batch(
                        student_latent, student_proprio, atm
                    )

                # Merge results - determine output shape from whichever mode ran
                if teacher_body_actions is not None:
                    out_seq_len = teacher_body_actions.shape[1]
                    out_action_dim = teacher_body_actions.shape[2]
                    dtype = teacher_body_actions.dtype
                elif student_body_actions is not None:
                    out_seq_len = student_body_actions.shape[1]
                    out_action_dim = student_body_actions.shape[2]
                    dtype = student_body_actions.dtype
                else:
                    raise RuntimeError("Mixed mode: no envs to process (both masks empty)")

                body_actions = torch.zeros(
                    num_envs, out_seq_len, out_action_dim, device=self.device, dtype=dtype
                )

                if teacher_body_actions is not None and teacher_indices is not None:
                    body_actions[teacher_indices] = teacher_body_actions
                if student_body_actions is not None and student_indices is not None:
                    body_actions[student_indices] = student_body_actions

            else:
                raise ValueError(
                    f"Unknown action_mode: {action_mode}. "
                    f"Valid modes are 'direct_latent', 'residual', or 'mixed'."
                )

            # Store full latent (decoder input) for full_latent_rate_l2 reward
            # Only needed for residual mode (teacher RL training); student modes
            # (direct_latent, mixed) use L2 distillation loss, not RL rewards.
            if action_mode == "residual":
                self.env._prev_full_latent = self.env._full_latent.clone()  # noqa: SLF001
                atm_module = self.action_transform_module.actor_module
                if (
                    hasattr(atm_module, "_last_full_latent_flat")
                    and atm_module._last_full_latent_flat is not None  # noqa: SLF001
                ):
                    fl = atm_module._last_full_latent_flat  # noqa: SLF001
                    if fl.dim() == 3:
                        fl = fl[:, -1, :]  # (batch, latent_dim)
                    self.env._full_latent = fl.to(self.env.device)  # noqa: SLF001

            body_actions = body_actions[:, -1]  # Take last timestep

            if (
                self._body_joint_indices is not None
                and self._hand_joint_indices is not None
                and len(self._body_joint_indices) > 0
                and len(self._hand_joint_indices) > 0
            ):
                num_envs = body_actions.shape[0]
                env_actions = torch.zeros(
                    num_envs, self._current_num_joints, device=self.device, dtype=body_actions.dtype
                )
                env_actions[:, self._body_joint_indices] = body_actions
                env_actions[:, self._hand_joint_indices] = hand_actions
            else:
                env_actions = torch.cat([body_actions, hand_actions], dim=-1)

        else:
            env_actions = actions["actions"]

        action_clip_value = self.config.get("action_clip_value", None)

        if action_clip_value is not None and action_clip_value > 0:
            env_actions = torch.clip(env_actions, -action_clip_value, action_clip_value)

        # Lightweight action plot update (env 0, first N joints)
        if self.turn_on_visualization:
            try:
                if self._action_fig is None:
                    self._init_action_plot(env_actions.shape[-1])
                if (self._step_counter % self._viz_every_n_steps) == 0:
                    self._update_action_plot(env_actions)
                self._step_counter += 1
            except Exception:  # noqa: S110, BLE001
                pass

        obs_dict, rew, terminated, truncated, extras = self.env.step(env_actions)

        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)

        # Zero out action/latent rate buffers for envs that just reset
        # This prevents a false large rate penalty on the first step of a new episode
        # Only applies when action_transform_module is used (buffers created in reset())
        reset_mask = dones.bool()
        if reset_mask.any() and hasattr(self.env, "_prev_meta_action"):
            self.env._prev_meta_action[reset_mask] = 0.0  # noqa: SLF001
            self.env._last_meta_action[reset_mask] = 0.0  # noqa: SLF001
            self.env._prev_full_latent[reset_mask] = 0.0  # noqa: SLF001
            self.env._full_latent[reset_mask] = 0.0  # noqa: SLF001

        # Compute success_lift metric: check if object has no contact with table (lifted)
        # Skip frames before first contact (from contact label data)
        if (
            hasattr(self.env, "scene")
            and "object_to_table_contact_sensor" in self.env.scene.sensors
        ):
            from isaaclab.sensors import ContactSensor

            sensor: ContactSensor = self.env.scene["object_to_table_contact_sensor"]
            contact_force = sensor.data.force_matrix_w  # [num_envs, 1, 1, 3]
            force_magnitude = torch.norm(contact_force, dim=-1).sum(dim=(-1, -2))  # [num_envs]

            contact_force_threshold = self.config.get("lift_contact_force_threshold", 0.1)

            if hasattr(self.env, "success_lift"):
                self.env.success_lift = self.env.success_lift & (~dones.bool())

            # Get first contact frame from motion command (contact label data)
            is_before_contact = torch.ones(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )
            if self.motion_command is not None:
                per_env_first_contact = getattr(self.motion_command, "_per_env_first_contact", None)
                if per_env_first_contact is not None:
                    current_time = (
                        self.motion_command.motion_start_time_steps + self.motion_command.time_steps
                    )
                    is_before_contact = current_time < per_env_first_contact

            # Object is currently lifted if there's no contact force from table
            is_currently_lifted = force_magnitude <= contact_force_threshold

            # Only update success_lift after first contact (cumulative OR)
            # Before first contact, success_lift stays False regardless of contact
            self.env.success_lift = torch.where(
                is_before_contact,
                self.env.success_lift,  # Keep current value (False after reset)
                is_currently_lifted | self.env.success_lift,
            )
        else:
            # If no contact sensor, set to False for all envs
            self.env.success_lift = torch.zeros(
                self.env.num_envs, dtype=torch.bool, device=self.env.device
            )

        extras["time_outs"] = truncated
        extras["episode"] = {}
        extras["to_log"] = {}
        for k, v in extras["log"].items():
            if isinstance(v, torch.Tensor):
                extras["to_log"][k] = v
            else:
                extras["to_log"][k] = torch.tensor(v, dtype=torch.float)
        if self._motion_lib is not None and self._motion_lib.use_adaptive_sampling:
            extras["to_log"][
                "adp_samp/num_episodes_min"
            ] = self._motion_lib.adp_samp_num_episodes.min()
            extras["to_log"][
                "adp_samp/num_episodes_max"
            ] = self._motion_lib.adp_samp_num_episodes.max()
            extras["to_log"][
                "adp_samp/num_episodes_mean"
            ] = self._motion_lib.adp_samp_num_episodes.mean()
            extras["to_log"][
                "adp_samp/num_failures_min"
            ] = self._motion_lib.adp_samp_num_failures.min()
            extras["to_log"][
                "adp_samp/num_failures_max"
            ] = self._motion_lib.adp_samp_num_failures.max()
            extras["to_log"][
                "adp_samp/num_failures_mean"
            ] = self._motion_lib.adp_samp_num_failures.mean()
            extras["to_log"][
                "adp_samp/failure_rate_min"
            ] = self._motion_lib.adp_samp_failure_rate_raw.min()
            extras["to_log"][
                "adp_samp/failure_rate_max"
            ] = self._motion_lib.adp_samp_failure_rate_raw.max()
            extras["to_log"][
                "adp_samp/failure_rate_mean"
            ] = self._motion_lib.adp_samp_failure_rate_raw.mean()

            if hasattr(self._motion_lib, "adp_sampling_active_prob"):
                prob = self._motion_lib.adp_sampling_active_prob
                uniform_prob = 1.0 / len(prob) if len(prob) > 0 else 1.0
                extras["to_log"]["adp_samp/prob_max"] = prob.max()
                extras["to_log"]["adp_samp/prob_min"] = prob.min()
                extras["to_log"]["adp_samp/prob_mean"] = prob.mean()
                extras["to_log"]["adp_samp/prob_max_over_uniform"] = prob.max() / uniform_prob
                extras["to_log"]["adp_samp/effective_num_bins"] = 1.0 / (prob**2).sum()
                # How many bins have prob > 10x uniform (significantly concentrated)
                # Note: max allowed is 50x uniform, so 10x is 20% of the cap
                extras["to_log"]["adp_samp/num_concentrated_bins"] = (
                    (prob > 10 * uniform_prob).sum().float()
                )

            eps_mean = self._motion_lib.adp_samp_num_episodes.mean()
            if eps_mean > 0:
                extras["to_log"]["adp_samp/episodes_max_over_mean"] = (
                    self._motion_lib.adp_samp_num_episodes.max() / eps_mean
                )
        new_obs = self.process_raw_obs(obs_dict, flatten_dict_obs=True)
        # Store obs for action_transform_module when obs_dict is not provided in next step()
        self._last_obs_dict = new_obs
        self.extras = extras
        # Store env_actions for callbacks (e.g., MultiLatentSaveCallback)
        extras["env_actions"] = env_actions.detach().cpu()
        return new_obs, rew, dones, extras

    def get_env_data(self, key):
        if key == "ref_body_pos_extend":
            return self.motion_command.robot_body_pos_w
        elif key == "rigid_body_pos_extend":
            return self.motion_command.body_pos_w
        else:
            return self.env.get_env_data(key)

    def render_results(self):
        pass

    def end_render_results(self):
        if "render_envs" in self.env.recorder_manager._terms:  # noqa: SLF001
            self.env.recorder_manager._terms["render_envs"].close_writers()  # noqa: SLF001
        if "trajectory" in self.env.recorder_manager._terms:  # noqa: SLF001
            self.env.recorder_manager._terms["trajectory"].close_writers()  # noqa: SLF001
        if self._action_fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._action_fig)
            except Exception:  # noqa: S110, BLE001
                pass
            self._action_fig = None
            self._action_lines = None
            self._action_hist = None
            self._hist_idx = 0
            self._blit_background = None

    def set_is_evaluating(self, is_evaluating: bool = True, global_rank=0, **_kwargs):
        self.is_evaluating = is_evaluating
        if self.motion_command is not None:
            self.motion_command.set_is_evaluating(is_evaluating)
        if self.force_command is not None:
            self.force_command.set_is_evaluating(is_evaluating)
        if is_evaluating:
            self.begin_seq_motion_samples(global_rank)

    def begin_seq_motion_samples(self, global_rank=0):
        logger.info("Loading motions for evaluation")
        self.start_idx = global_rank * self.num_envs
        self._motion_lib.load_motions_for_evaluation(start_idx=self.start_idx)
        self.reset_all(global_rank=global_rank)

    def forward_motion_samples(self, global_rank=0, world_size=1):
        old_start_idx = self.start_idx
        self.start_idx += world_size * self.num_envs
        logger.info(
            f"Forward motions for evaluation from {old_start_idx} to {self.start_idx} - rank: {global_rank} - world size: {world_size}"  # noqa: E501
        )
        self._motion_lib.load_motions_for_evaluation(start_idx=self.start_idx)
        self.reset_all(global_rank=global_rank)

    def focusing_viewer(self):
        if not self.viewer_focused:
            # Focus on robot asset (tracking mode)
            self.env.viewport_camera_controller.update_view_to_world()
            self.env.viewport_camera_controller.update_view_to_asset_root("robot")
            self.viewer_focused = True
            logger.info("Focused on robot")
        else:
            # Switch to free camera mode centered on robot
            self.env.viewport_camera_controller.viewer_origin = torch.zeros_like(
                self.env.scene["robot"].data.root_pos_w[0]
            )
            self.env.viewport_camera_controller.cfg.origin_type = "world"
            cam_eye = self.env.scene["robot"].data.root_pos_w[0].cpu().numpy() + np.array(
                [2.0, 2.0, 1.5]
            )
            cam_target = self.env.scene["robot"].data.root_pos_w[0].cpu().numpy()
            self.env.viewport_camera_controller._env.sim.set_camera_view(  # noqa: SLF001
                eye=cam_eye, target=cam_target
            )
            self.viewer_focused = False
            logger.info("Switched to free camera mode")  # noqa: RUF100, T201

    def set_is_training(self, **_kwargs):
        self.is_evaluating = False
        self.resample_motion()

    def resample_motion(self):
        res = self._motion_lib.load_motions_for_training(
            max_num_seqs=min(self.num_envs, self.motion_command.max_num_load_motions)
        )
        if res:
            self.reset_all()
        else:
            logger.info("No new motions loaded, skipping reset")  # noqa: RUF100, T201

    def sync_and_compute_adaptive_sampling(self, accelerator, sync_across_gpus=False):
        if self._motion_lib is not None:
            self._motion_lib.sync_and_compute_adaptive_sampling(
                accelerator, sync_across_gpus=sync_across_gpus
            )

    def load_env_state_dict(self, state_dict):
        if "motion_lib" in state_dict:
            self._motion_lib.load_state_dict(state_dict["motion_lib"])
            if self._motion_lib.use_adaptive_sampling:
                self.resample_motion()

    def get_env_state_dict(self):
        state_dict = {
            "motion_lib": self._motion_lib.get_state_dict(),
        }
        return state_dict

    def reinit_dr(self, **_kwargs):
        pass

    def setup_keyboard(self):
        try:
            from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg

            cfg = Se2KeyboardCfg()
            self.keyboard_interface = Se2Keyboard(cfg)
            self.keyboard_interface.add_callback("R", self.reset_all)
            self.keyboard_interface.add_callback("T", self.forward_motion_samples)
            self.keyboard_interface.add_callback("F", self.focusing_viewer)
            self.keyboard_interface.add_callback("V", self.toggle_debug_vis)
        except Exception as e:  # noqa: BLE001
            logger.info(f"Error setting up keyboard: {e}")  # noqa: RUF100, T201

    def toggle_debug_vis(self):
        if self.motion_command is not None and hasattr(self.motion_command, "_set_debug_vis_impl"):
            self._debug_vis_enabled = not getattr(self, "_debug_vis_enabled", True)
            self.motion_command._set_debug_vis_impl(self._debug_vis_enabled)  # noqa: SLF001
            logger.info(f"Debug visualization: {'ON' if self._debug_vis_enabled else 'OFF'}")  # noqa: RUF100, T201

    # --- Action plotting helpers ---
    def _init_action_plot(self, action_dim: int):
        plot_dim = min(int(self._plot_action_dim), int(action_dim))
        if plot_dim <= 0:
            return
        try:
            import matplotlib.pyplot as plt
        except Exception as e:  # noqa: BLE001, F841
            return
        plt.ion()
        import math

        cols = math.ceil(math.sqrt(plot_dim))
        rows = math.ceil(plot_dim / cols)
        fig, axes = plt.subplots(
            rows, cols, sharex=True, sharey=True, figsize=(cols * 3.0, rows * 2.2)
        )
        axes = np.array(axes).reshape(-1)
        self._plot_window = max(10, int(self._plot_window))
        x_vals = np.arange(self._plot_window)
        self._action_hist = np.zeros((self._plot_window, plot_dim), dtype=np.float32)
        lines = []
        y_min, y_max = -float(self._action_ylim), float(self._action_ylim)
        for i in range(rows * cols):
            ax_i = axes[i]
            if i < plot_dim:
                (ln,) = ax_i.plot(x_vals, self._action_hist[:, i], linewidth=1.0)
                ln.set_animated(True)
                lines.append(ln)
                ax_i.set_ylim((y_min, y_max))
                ax_i.set_xlim((0, self._plot_window - 1))
                ax_i.set_title(f"J{i}", fontsize=8)
                if i // cols == rows - 1:
                    ax_i.set_xlabel("step", fontsize=8)
                if i % cols == 0:
                    ax_i.set_ylabel("act", fontsize=8)
            else:
                ax_i.axis("off")
        fig.suptitle(f"Action time series (env 0) - first {plot_dim} joints", fontsize=10)
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        self._action_fig = fig
        self._action_lines = lines
        self._hist_idx = -1
        try:
            fig.canvas.draw()
            self._blit_background = fig.canvas.copy_from_bbox(fig.bbox)
        except Exception:  # noqa: BLE001
            self._blit_background = None
            self._blit_supported = False
        fig.canvas.flush_events()

    def _update_action_plot(self, env_actions: torch.Tensor):
        if self._action_lines is None or self._action_hist is None:
            return
        plot_dim = len(self._action_lines)
        with torch.no_grad():
            vals = env_actions[0, :plot_dim].detach().to("cpu").numpy().astype(np.float32)
        vals = np.clip(vals, -self._action_ylim, self._action_ylim)
        # advance circular buffer
        self._hist_idx = (self._hist_idx + 1) % self._plot_window
        self._action_hist[self._hist_idx, :plot_dim] = vals
        # display in chronological order using a view with roll
        disp = np.roll(self._action_hist, shift=-(self._hist_idx + 1), axis=0)
        for i, ln in enumerate(self._action_lines):
            ln.set_ydata(disp[:, i])
        if self._action_fig is not None:
            canvas = self._action_fig.canvas
            # Periodically refresh background to handle resizes or overdraw
            if self._blit_supported and (
                self._blit_background is None
                or (self._step_counter % max(1, self._blit_refresh_interval) == 0)
            ):
                try:
                    self._action_fig.canvas.draw()
                    self._blit_background = canvas.copy_from_bbox(self._action_fig.bbox)
                except Exception:  # noqa: BLE001
                    self._blit_background = None
                    self._blit_supported = False
            if self._blit_supported and self._blit_background is not None:
                try:
                    canvas.restore_region(self._blit_background)
                    for ln in self._action_lines:
                        ln.axes.draw_artist(ln)
                    canvas.blit(self._action_fig.bbox)
                    canvas.flush_events()
                    return
                except Exception:  # noqa: BLE001
                    self._blit_supported = False
            # Fallback full redraw
            canvas.draw_idle()
            canvas.flush_events()

    @property
    def motion_ids(self):
        return self.motion_command.motion_ids

    def setup_replay_grid(self, spacing=2.0, rows=None, cols=None):
        """Setup a custom grid layout for environment origins during replay.

        Args:
            spacing: Distance between environments in meters (default: 2.0)
            rows: Number of rows in the grid (default: auto-calculated)
            cols: Number of columns in the grid (default: auto-calculated)

        Returns:
            Tensor of shape (num_envs, 3) with custom origins
        """
        import math

        num_envs = self.num_envs

        # Auto-calculate grid dimensions if not provided
        if rows is None and cols is None:
            # Try to make a square-ish grid
            cols = int(math.ceil(math.sqrt(num_envs)))  # noqa: RUF046
            rows = int(math.ceil(num_envs / cols))  # noqa: RUF046
        elif rows is None:
            rows = int(math.ceil(num_envs / cols))  # noqa: RUF046
        elif cols is None:
            cols = int(math.ceil(num_envs / rows))  # noqa: RUF046

        logger.info(f"Setting up replay grid: {rows} rows x {cols} cols with {spacing}m spacing")

        # Create grid origins
        custom_origins = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)

        for i in range(num_envs):
            row = i // cols
            col = i % cols

            # Center the grid around origin
            x_offset = (col - (cols - 1) / 2.0) * spacing
            y_offset = (row - (rows - 1) / 2.0) * spacing

            custom_origins[i, 0] = x_offset
            custom_origins[i, 1] = y_offset
            custom_origins[i, 2] = 0.0  # Keep z at ground level

        # Store custom origins
        self._replay_custom_origins = custom_origins

        logger.info(
            f"Grid bounds: X=[{custom_origins[:, 0].min():.1f}, {custom_origins[:, 0].max():.1f}], "
            f"Y=[{custom_origins[:, 1].min():.1f}, {custom_origins[:, 1].max():.1f}]"
        )

        return custom_origins

    def run_replay(
        self,
        motion_id=None,
        start_time_step=0,
        speed=1.0,
        loop=True,
        enable_vis=False,
        target_fps=50,
        grid_spacing=2.0,
        grid_rows=None,
        grid_cols=None,
        save_video_path=None,
    ):
        """Run a complete motion replay with automatic rendering loop for all environments.
        This is the high-level convenience function that handles everything.

        Args:
            motion_id: The motion ID(s) to replay. Can be:
                       - None: uses current motion_ids for all envs
                       - int: single motion ID for all envs
                       - list/tensor: motion ID per environment
            start_time_step: Starting time step in the motion (default: 0)
            speed: Playback speed multiplier (default: 1.0)
            loop: Whether to loop the motion (default: True)
            enable_vis: Whether to enable debug visualization (default: False)
            target_fps: Target frames per second for rendering (default: 50)
            grid_spacing: Distance between environments in meters (default: 2.0)
            grid_rows: Number of rows in grid layout (default: auto)
            grid_cols: Number of columns in grid layout (default: auto)
            save_video_path: Path to save video file (default: None, no video saved)
                             Requires render_results=True in config to enable eval_camera.

        Returns:
            List of dictionaries with motion metadata for each environment
        """  # noqa: D205
        import os
        import time

        # Setup custom grid origins
        self.setup_replay_grid(spacing=grid_spacing, rows=grid_rows, cols=grid_cols)

        # Initialize replay
        info = self.setup_replay_motion(
            motion_id=motion_id,
            start_time_step=start_time_step,
            speed=speed,
            loop=loop,
            enable_vis=enable_vis,
        )

        if info is None:
            return None

        # Initialize video writer if save_video_path is specified
        video_writer = None
        if save_video_path is not None:
            if "eval_camera" not in self.env.scene.sensors:
                logger.info("WARNING: save_video_path specified but eval_camera not available.")
                logger.info("         Set ++manager_env.config.render_results=True to enable it.")
            else:
                import imageio

                os.makedirs(os.path.dirname(os.path.abspath(save_video_path)), exist_ok=True)
                video_writer = imageio.get_writer(
                    save_video_path,
                    fps=target_fps,
                    codec="libx264",
                    quality=5,
                    pixelformat="yuv420p",
                )
                logger.info(f"[Video] Recording to: {save_video_path}")

        # Run the rendering loop
        frame_time = 1.0 / target_fps
        last_time = time.time()
        frame_count = 0

        try:
            while self.step_replay():
                # Position camera BEFORE rendering (if recording)
                if video_writer is not None and "eval_camera" in self.env.scene.sensors:
                    root_pos = self.motion_command.robot.data.body_pos_w[:, 0]
                    camera_offset = self.config.get("eval_camera_offset", [-2, -2, 1])
                    eye = root_pos + torch.tensor(camera_offset, device=self.device)
                    self.env.scene["eval_camera"].set_world_poses_from_view(eye, root_pos)

                # Render the scene (triggers camera render in headless mode)
                if hasattr(self.env, "sim"):
                    try:
                        self.env.sim.render()
                    except Exception as e:  # noqa: BLE001
                        logger.info(f"Render error: {e}")

                # Capture video frame if recording
                if video_writer is not None and "eval_camera" in self.env.scene.sensors:
                    # Update camera to refresh its data after render
                    self.env.scene["eval_camera"].update(dt=0.0)

                    # Grab frame from camera
                    rgb_frame = self.env.scene["eval_camera"].data.output["rgb"]
                    frame = rgb_frame[0].cpu().numpy().astype(np.uint8)
                    video_writer.append_data(frame)
                    frame_count += 1

                # Frame rate limiting (only when not headless or not saving video)
                if not self.config.get("headless", False) or video_writer is None:
                    current_time = time.time()
                    elapsed = current_time - last_time
                    sleep_time = max(0, (frame_time / speed) - elapsed)

                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    last_time = time.time()

        except KeyboardInterrupt:
            logger.info("\nReplay interrupted by user")

        # Close video writer
        if video_writer is not None:
            video_writer.close()
            logger.info(f"[Video] Saved {frame_count} frames to: {save_video_path}")

        logger.info("\nReplay complete!")
        return info

    def setup_replay_motion(
        self, motion_id=None, start_time_step=0, speed=1.0, loop=True, enable_vis=False
    ):
        """Initialize motion replay for all environments (low-level function for manual control).
        Use run_replay() instead if you want automatic rendering loop.

        After calling this, you need to manually call step_replay() in a loop
        and handle rendering yourself. For automatic rendering, use run_replay().

        Args:
            motion_id: The motion ID(s) to replay. Can be:
                       - None: uses current motion_ids for all envs
                       - int: single motion ID for all envs
                       - list/tensor: motion ID per environment
            start_time_step: Starting time step in the motion (default: 0)
            speed: Playback speed multiplier (default: 1.0)
            loop: Whether to loop the motion (default: True)
            enable_vis: Whether to enable debug visualization (default: False)

        Returns:
            List of dictionaries with motion metadata for each environment
        """  # noqa: D205
        if self._motion_lib is None:
            logger.info("No motion library available for replay")
            return None

        # Replay all environments
        num_replay_envs = self.num_envs
        replay_env_ids = torch.arange(num_replay_envs, device=self.device, dtype=torch.long)

        # Get motion IDs
        if motion_id is None:
            motion_ids = self.motion_command.motion_ids
        elif isinstance(motion_id, int):
            motion_ids = torch.full(
                (num_replay_envs,), motion_id, device=self.device, dtype=torch.long
            )
        elif isinstance(motion_id, list | tuple):
            if len(motion_id) != num_replay_envs:
                logger.info(
                    f"Error: motion_id list length ({len(motion_id)}) doesn't match number of environments ({num_replay_envs})"  # noqa: E501
                )
                return None
            motion_ids = torch.tensor(motion_id, device=self.device, dtype=torch.long)
        elif isinstance(motion_id, torch.Tensor):
            if motion_id.shape[0] != num_replay_envs:
                logger.info(
                    f"Error: motion_id tensor size ({motion_id.shape[0]}) doesn't match number of environments ({num_replay_envs})"  # noqa: E501
                )
                return None
            motion_ids = motion_id.to(self.device)
        else:
            logger.info(f"Error: Invalid motion_id type: {type(motion_id)}")
            return None

        # Get motion info for display
        num_steps_per_env = self._motion_lib.get_motion_num_steps(motion_ids)
        max_num_steps = num_steps_per_env.max().item()

        unique_motions = torch.unique(motion_ids)
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Replaying {num_replay_envs} Environments")
        logger.info(f"Unique motion IDs: {unique_motions.tolist()}")
        logger.info(f"Max frames: {max_num_steps}")
        logger.info(f"Max duration: {max_num_steps / 50.0:.2f}s (@ 50 FPS)")
        logger.info(f"Speed: {speed}x")
        logger.info(f"{'='*60}\n")

        # Enable visualization if requested
        if enable_vis and hasattr(self.motion_command, "_set_debug_vis_impl"):
            self.motion_command._set_debug_vis_impl(True)  # noqa: SLF001

        # Initialize contact point visualization if available and enabled
        self._replay_contact_visualizer = None
        self._replay_vis_enabled = enable_vis
        if enable_vis and VISUALIZATION_AVAILABLE:
            self._setup_contact_center_visualizer()

        # Replay state
        self._replay_active = True
        self._replay_paused = False
        self._replay_reverse = False
        self._replay_time_steps = torch.full(
            (num_replay_envs,), start_time_step, device=self.device, dtype=torch.long
        )
        self._replay_motion_ids = motion_ids
        self._replay_env_ids = replay_env_ids
        self._replay_speed = speed
        self._replay_loop = loop
        self._replay_num_steps_per_env = num_steps_per_env
        self._replay_max_num_steps = max_num_steps

        # Pre-load table metadata from pkl files for ALL motions (per-env)
        # This supports multi-motion replay where each env can have different table positions
        self._replay_table_pos = None  # Shape: (num_envs, 3)
        self._replay_table_quat = None  # Shape: (num_envs, 4)
        if hasattr(self.env, "scene") and "table" in self.env.scene.rigid_objects:
            try:
                import os

                import joblib

                # Derive meta directory from motion_file path in config
                motion_lib_cfg = getattr(self.motion_command.cfg, "motion_lib_cfg", None)
                motion_file = motion_lib_cfg.get("motion_file", "") if motion_lib_cfg else ""

                if motion_file and "/robot" in motion_file:
                    if os.path.isdir(motion_file):
                        meta_dir = motion_file.replace("/robot", "/meta")
                    else:
                        meta_dir = os.path.dirname(motion_file).replace("/robot", "/meta")
                else:
                    meta_dir = "data/motion_lib_grab/meta"

                # Pre-load motion file data if it's a single file (for table fallback)
                motion_file_data = None
                if motion_file and os.path.isfile(motion_file) and not os.path.isdir(meta_dir):
                    motion_file_data = joblib.load(motion_file)

                # Load table metadata for each environment's assigned motion
                table_pos_list = []
                table_quat_list = []
                table_scale_list = []  # Also load scales for offset calculation
                loaded_count = 0
                loaded_from_motion_file = 0

                for env_idx in range(num_replay_envs):
                    motion_idx = motion_ids[env_idx].item()
                    motion_key = self._motion_lib.curr_motion_keys[motion_idx]
                    meta_file = os.path.join(meta_dir, f"{motion_key}.pkl")

                    if os.path.exists(meta_file):
                        meta = joblib.load(meta_file)
                        pos = torch.tensor(
                            meta.get("table_pos", [0.0, 0.0, 0.8]), device=self.device
                        ).float()
                        quat = torch.tensor(
                            meta.get("table_quat", [1.0, 0.0, 0.0, 0.0]), device=self.device
                        ).float()
                        scale = torch.tensor(
                            meta.get("table_scale", [1.0, 1.0, 1.0]), device=self.device
                        ).float()
                        table_pos_list.append(pos)
                        table_quat_list.append(quat)
                        table_scale_list.append(scale)
                        loaded_count += 1
                    elif motion_file_data is not None and motion_key in motion_file_data:
                        # Fallback: try to get table data from motion file directly
                        motion_data = motion_file_data[motion_key]
                        if "table_pos" in motion_data and "table_quat" in motion_data:
                            pos = torch.tensor(motion_data["table_pos"], device=self.device).float()
                            quat = torch.tensor(
                                motion_data["table_quat"], device=self.device
                            ).float()
                            scale = torch.tensor(
                                motion_data.get("table_scale", [1.0, 1.0, 1.0]), device=self.device
                            ).float()
                            table_pos_list.append(pos)
                            table_quat_list.append(quat)
                            table_scale_list.append(scale)
                            loaded_from_motion_file += 1
                        else:
                            # No table data in motion file, use default
                            table_pos_list.append(
                                torch.tensor([0.0, 0.0, 0.8], device=self.device).float()
                            )
                            table_quat_list.append(
                                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).float()
                            )
                            table_scale_list.append(
                                torch.tensor([1.0, 1.0, 1.0], device=self.device).float()
                            )
                    else:
                        # Fallback: use default table position
                        table_pos_list.append(
                            torch.tensor([0.0, 0.0, 0.8], device=self.device).float()
                        )
                        table_quat_list.append(
                            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).float()
                        )
                        table_scale_list.append(
                            torch.tensor([1.0, 1.0, 1.0], device=self.device).float()
                        )

                self._replay_table_pos = torch.stack(table_pos_list, dim=0)  # (num_envs, 3)
                self._replay_table_quat = torch.stack(table_quat_list, dim=0)  # (num_envs, 4)
                self._replay_table_scale = torch.stack(table_scale_list, dim=0)  # (num_envs, 3)

                # Apply fixed table_offset if configured
                table_offset = self.config.get("table_offset", None)
                if table_offset is not None:
                    offset_tensor = torch.tensor(
                        table_offset, device=self.device, dtype=self._replay_table_pos.dtype
                    )
                    self._replay_table_pos = self._replay_table_pos + offset_tensor

                # Apply maximal X offset based on object starting position
                # This computes the maximum valid offset range such that object stays on table
                x_offset_maximal = self.config.get("replay_table_x_offset_maximal", False)
                x_offset_margin = self.config.get(
                    "replay_table_x_offset_margin", 0.05
                )  # 5cm safety margin

                if x_offset_maximal:
                    # Get base table width from config (required for maximal offset mode)
                    table_size_cfg = self.config.get("table_size")
                    if table_size_cfg is None:
                        raise ValueError(
                            "replay_table_x_offset_maximal requires table_size to be set in config"
                        )
                    base_table_width = table_size_cfg[0]

                    # Get object starting position (step 0) for each env
                    object_start_pos = self._motion_lib.get_object_root_pos(
                        motion_ids,
                        torch.zeros(num_replay_envs, device=self.device, dtype=torch.long),
                    )[
                        :, 0
                    ]  # Shape: (num_envs, 3) -> take first object if multiple
                    object_start_x = object_start_pos[:, 0]  # (num_envs,)

                    # Get table position and scaled width per env
                    table_x = self._replay_table_pos[:, 0]  # (num_envs,)
                    scale_x = self._replay_table_scale[:, 0]  # (num_envs,)
                    actual_width = base_table_width * scale_x  # (num_envs,)
                    half_width = actual_width / 2.0

                    # Compute valid offset range per env:
                    # Object must stay on table after offset dx
                    # dx_min = object_x - table_x - half_width + margin
                    # dx_max = object_x - table_x + half_width - margin
                    relative_obj_x = object_start_x - table_x
                    dx_min = relative_obj_x - half_width + x_offset_margin
                    dx_max = relative_obj_x + half_width - x_offset_margin

                    # Sample X offset uniformly within valid range per env
                    rand_vals = torch.rand(num_replay_envs, device=self.device)
                    x_offsets = dx_min + rand_vals * (dx_max - dx_min)

                    self._replay_table_pos[:, 0] += x_offsets

                    logger.info("Replay table X offset (maximal mode):")
                    logger.info(
                        f"  Base table width: {base_table_width:.3f}m, margin: {x_offset_margin:.3f}m"
                    )
                    logger.info(f"  dx_min range: [{dx_min.min():.3f}, {dx_min.max():.3f}]")
                    logger.info(f"  dx_max range: [{dx_max.min():.3f}, {dx_max.max():.3f}]")
                    logger.info(f"  Applied offsets: [{x_offsets.min():.3f}, {x_offsets.max():.3f}]")

                # Apply fixed XY offset range (legacy mode, used if maximal mode is disabled)
                # Format: [x_min, x_max, y_min, y_max] - per-env randomized
                xy_offset_range = self.config.get("replay_table_xy_offset_range", None)
                if xy_offset_range is not None and not x_offset_maximal:
                    x_min, x_max, y_min, y_max = xy_offset_range

                    # Sample X offset per environment
                    x_offsets = torch.empty(num_replay_envs, device=self.device).uniform_(
                        x_min, x_max
                    )
                    self._replay_table_pos[:, 0] += x_offsets

                    # Sample Y offset per environment (typically y_min = y_max for fixed offset)
                    y_offsets = torch.empty(num_replay_envs, device=self.device).uniform_(
                        y_min, y_max
                    )
                    self._replay_table_pos[:, 1] += y_offsets

                    logger.info(
                        f"Replay table XY offset: X=[{x_min:.3f}, {x_max:.3f}], Y=[{y_min:.3f}, {y_max:.3f}]"
                    )

                if loaded_count > 0:
                    logger.info(
                        f"Loaded table metadata for {loaded_count}/{num_replay_envs} environments from {meta_dir}"
                    )
                elif loaded_from_motion_file > 0:
                    logger.info(
                        f"Loaded table metadata for {loaded_from_motion_file}/{num_replay_envs} environments from motion file"  # noqa: E501
                    )

            except Exception as e:  # noqa: BLE001
                logger.info(f"Warning: Could not load table metadata: {e}")

        # Setup keyboard controls for replay
        if not self.config.headless:
            self._setup_replay_keyboard()

        logger.info("Replay Controls:")
        logger.info("  G: Pause/Resume")
        logger.info("  B: Toggle Reverse Play")
        logger.info("  LEFT/RIGHT: Step backward/forward (when paused)")
        logger.info("  R: Restart from beginning")
        logger.info("  ESC: Exit replay")
        logger.info("  +/-: Increase/Decrease speed")
        logger.info("")

        # Return metadata
        results = []
        for i, (mid, nsteps) in enumerate(
            zip(motion_ids.tolist(), num_steps_per_env.tolist(), strict=False)
        ):
            # Get motion key from motion library (curr_motion_keys is the active list)
            if hasattr(self._motion_lib, "curr_motion_keys") and mid < len(
                self._motion_lib.curr_motion_keys
            ):
                motion_key = self._motion_lib.curr_motion_keys[mid]
            elif hasattr(self._motion_lib, "motion_keys") and mid < len(
                self._motion_lib.motion_keys
            ):
                motion_key = self._motion_lib.motion_keys[mid]
            else:
                motion_key = f"motion_{mid}"
            results.append(
                {
                    "env_id": i,
                    "motion_id": mid,
                    "motion_key": motion_key,
                    "num_steps": nsteps,
                    "duration": nsteps / 50.0,
                }
            )
        return results

    def _setup_replay_keyboard(self):
        """Setup keyboard controls for replay mode"""  # noqa: D415
        try:
            if not hasattr(self, "keyboard_interface"):
                from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg

                cfg = Se2KeyboardCfg()
                self.keyboard_interface = Se2Keyboard(cfg)

            # Add replay-specific callbacks
            self.keyboard_interface.add_callback("G", self._toggle_replay_pause)
            self.keyboard_interface.add_callback("LEFT", self._replay_step_backward)
            self.keyboard_interface.add_callback("RIGHT", self._replay_step_forward)
            self.keyboard_interface.add_callback("R", self._restart_replay)
            self.keyboard_interface.add_callback("ESCAPE", self._exit_replay)
            self.keyboard_interface.add_callback("EQUAL", self._increase_replay_speed)
            self.keyboard_interface.add_callback("MINUS", self._decrease_replay_speed)
            self.keyboard_interface.add_callback("B", self._toggle_reverse_play)
        except Exception as e:  # noqa: BLE001
            logger.info(f"Could not setup replay keyboard controls: {e}")

    def _setup_contact_center_visualizer(self):
        """Setup visualization markers for per-hand contact centers during replay."""
        if not VISUALIZATION_AVAILABLE:
            return

        has_left = hasattr(self._motion_lib, "_motion_object_contact_center_left")
        has_right = hasattr(self._motion_lib, "_motion_object_contact_center_right")
        if not has_left and not has_right:
            logger.info("No contact center available in motion library for visualization")
            return

        try:
            self._replay_contact_visualizers = {}

            if has_left:
                left_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Replay/contact_center_left",
                    markers={
                        "contact": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 0.0, 1.0)  # Blue for left hand
                            ),
                        ),
                    },
                )
                self._replay_contact_visualizers["left_hand"] = VisualizationMarkers(left_cfg)
                self._replay_contact_visualizers["left_hand"].set_visibility(True)

            if has_right:
                right_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Replay/contact_center_right",
                    markers={
                        "contact": sim_utils.SphereCfg(
                            radius=0.03,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.0, 1.0, 1.0)  # Green for right hand
                            ),
                        ),
                    },
                )
                self._replay_contact_visualizers["right_hand"] = VisualizationMarkers(right_cfg)
                self._replay_contact_visualizers["right_hand"].set_visibility(True)

            hands = list(self._replay_contact_visualizers.keys())
            logger.info(f"Contact center visualizer initialized for: {', '.join(hands)}")

            # Keep _replay_contact_visualizer as a truthy check for the update loop
            self._replay_contact_visualizer = True

        except Exception as e:  # noqa: BLE001
            logger.info(f"Could not setup contact center visualizer: {e}")
            self._replay_contact_visualizer = None

    def _update_contact_center_visualization(self):
        """Update per-hand contact center visualization during replay."""
        if not getattr(self, "_replay_contact_visualizers", None):
            return

        from gear_sonic.isaac_utils.rotations import quat_rotate

        # Get object pose (shared by both hands)
        object_root_pos = self._motion_lib.get_object_root_pos(
            self._replay_motion_ids, self._replay_time_steps
        )[
            :, 0, :
        ]  # (num_envs, 3)
        object_root_quat = self._motion_lib.get_object_root_quat(
            self._replay_motion_ids, self._replay_time_steps
        )[
            :, 0, :
        ]  # (num_envs, 4)

        # Environment origin offsets
        if hasattr(self, "_replay_custom_origins"):
            env_origins = self._replay_custom_origins[self._replay_env_ids]
        else:
            env_origins = self.env.scene.env_origins[self._replay_env_ids]

        # Hidden position for markers when no contact
        hidden_pos = torch.tensor([[0.0, 0.0, -1000.0]], device=self.device)

        for hand, visualizer in self._replay_contact_visualizers.items():
            contact_center = self._motion_lib.get_object_contact_center(
                self._replay_motion_ids, self._replay_time_steps, hand=hand
            )
            if contact_center is None:
                visualizer.visualize(translations=hidden_pos)
                continue

            # Move markers far away for envs with no contact (zero contact center)
            valid_mask = torch.norm(contact_center, dim=-1) > 1e-6
            rotated = quat_rotate(object_root_quat, contact_center, w_last=False)
            world_center = rotated + object_root_pos + env_origins
            world_center[~valid_mask] = hidden_pos
            visualizer.visualize(translations=world_center)

    def _toggle_replay_pause(self):
        """Toggle pause state for replay"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_paused = not self._replay_paused
            status = "PAUSED" if self._replay_paused else "PLAYING"
            max_frame = self._replay_time_steps.max().item()
            logger.info(f"Replay {status} (max frame {max_frame}/{self._replay_max_num_steps})")

    def _replay_step_backward(self):
        """Step backward one frame when paused"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active and self._replay_paused:
            self._replay_time_steps = torch.clamp(self._replay_time_steps - 1, min=0)
            self._update_replay_frame()
            max_frame = self._replay_time_steps.max().item()
            logger.info(f"Frame {max_frame}/{self._replay_max_num_steps}")
            self.env.sim.render()

    def _replay_step_forward(self):
        """Step forward one frame when paused"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active and self._replay_paused:
            self._replay_time_steps = torch.minimum(
                self._replay_time_steps + 1, self._replay_num_steps_per_env - 1
            )
            self._update_replay_frame()
            max_frame = self._replay_time_steps.max().item()
            logger.info(f"Frame {max_frame}/{self._replay_max_num_steps}")
            self.env.sim.render()

    def _restart_replay(self):
        """Restart replay from beginning"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_time_steps.fill_(0)
            self._replay_paused = False
            self._update_replay_frame()
            logger.info("Replay restarted from beginning")
            self.env.sim.render()

    def _exit_replay(self):
        """Exit replay mode"""  # noqa: D415
        if hasattr(self, "_replay_active"):
            self._replay_active = False
            # Hide contact center visualizers
            for vis in getattr(self, "_replay_contact_visualizers", {}).values():
                vis.set_visibility(False)
            logger.info("Exiting replay mode")
            self.env.sim.render()

    def _increase_replay_speed(self):
        """Increase replay speed"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_speed = min(5.0, self._replay_speed * 1.25)
            logger.info(f"Replay speed: {self._replay_speed:.2f}x")

    def _decrease_replay_speed(self):
        """Decrease replay speed"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_speed = max(0.1, self._replay_speed * 0.8)
            logger.info(f"Replay speed: {self._replay_speed:.2f}x")

    def _toggle_reverse_play(self):
        """Toggle reverse playback mode"""  # noqa: D415
        if hasattr(self, "_replay_active") and self._replay_active:
            self._replay_reverse = not self._replay_reverse
            direction = "REVERSE" if self._replay_reverse else "FORWARD"
            logger.info(f"Replay direction: {direction}")

    def _update_replay_frame(self):
        """Update all robot states to match the current replay frames"""  # noqa: D415
        if not hasattr(self, "_replay_active") or not self._replay_active:
            return

        # Get motion data at current time steps for all environments
        root_pos = self._motion_lib.get_root_pos_w(self._replay_motion_ids, self._replay_time_steps)
        root_quat = self._motion_lib.get_root_quat_w(
            self._replay_motion_ids, self._replay_time_steps
        )
        root_lin_vel = self._motion_lib.get_root_lin_vel_w(
            self._replay_motion_ids, self._replay_time_steps
        )
        root_ang_vel = self._motion_lib.get_root_ang_vel_w(
            self._replay_motion_ids, self._replay_time_steps
        )
        motion_lib_joint_pos = self._motion_lib.get_dof_pos(
            self._replay_motion_ids, self._replay_time_steps
        )
        motion_lib_joint_vel = self._motion_lib.get_dof_vel(
            self._replay_motion_ids, self._replay_time_steps
        )

        # Handle DOF mismatch between motion library (e.g., 29 DOF) and robot (e.g., 43 DOF)
        robot_num_joints = self.motion_command.robot.num_joints
        motion_lib_num_dof = motion_lib_joint_pos.shape[-1]

        if robot_num_joints > motion_lib_num_dof and self._body_joint_indices is not None:
            # Robot has more DOFs than motion lib (e.g., 43 DOF robot with 29 DOF motion data)
            # Use body joint indices for proper mapping
            num_envs = motion_lib_joint_pos.shape[0]

            # Create full joint tensors with zeros for all DOFs
            joint_pos = torch.zeros(
                num_envs, robot_num_joints, device=self.device, dtype=motion_lib_joint_pos.dtype
            )
            joint_vel = torch.zeros(
                num_envs, robot_num_joints, device=self.device, dtype=motion_lib_joint_vel.dtype
            )

            # Map motion lib data to body joint indices (using G1_ISAACLab_ORDER mapping)
            joint_pos[:, self._body_joint_indices] = motion_lib_joint_pos
            joint_vel[:, self._body_joint_indices] = motion_lib_joint_vel

            # Use hand DOFs from motion lib if available, otherwise default to zero
            hand_dof_pos = self._motion_lib.get_hand_dof_pos(
                self._replay_motion_ids, self._replay_time_steps
            )
            if hand_dof_pos is not None:
                # Hand DOFs are the last N joints (in Isaac order, not G1_HAND_JOINTS order)
                num_hand_dof = hand_dof_pos.shape[-1]
                joint_pos[:, -num_hand_dof:] = hand_dof_pos
        else:
            joint_pos = motion_lib_joint_pos
            joint_vel = motion_lib_joint_vel

        # Add environment origin offsets (use custom grid origins if set)
        if hasattr(self, "_replay_custom_origins"):
            root_pos = root_pos + self._replay_custom_origins[self._replay_env_ids]
        else:
            root_pos = root_pos + self.env.scene.env_origins[self._replay_env_ids]

        # Write state to simulation for all environments
        self.motion_command.robot.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=self._replay_env_ids
        )
        self.motion_command.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_quat, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=self._replay_env_ids,
        )

        # Get object motion data from motion library
        if hasattr(self._motion_lib, "_motion_object_root_pos") and hasattr(
            self._motion_lib, "_motion_object_root_quat"
        ):
            object_root_pos = self._motion_lib.get_object_root_pos(
                self._replay_motion_ids, self._replay_time_steps
            )
            object_root_quat = self._motion_lib.get_object_root_quat(
                self._replay_motion_ids, self._replay_time_steps
            )

            # Add environment origin offsets to object position
            if hasattr(self, "_replay_custom_origins"):
                object_root_pos = object_root_pos + self._replay_custom_origins[
                    self._replay_env_ids
                ].unsqueeze(1)
            else:
                object_root_pos = object_root_pos + self.env.scene.env_origins[
                    self._replay_env_ids
                ].unsqueeze(1)

            # Write object state to simulation (handle multiple objects)
            # Shape: object_root_pos is (num_envs, max_num_objects, 3)
            # Shape: object_root_quat is (num_envs, max_num_objects, 4)
            for obj_idx in range(object_root_pos.shape[1]):
                obj_pos = object_root_pos[:, obj_idx, :]
                obj_quat = object_root_quat[:, obj_idx, :]
                object_root_pose = torch.cat([obj_pos, obj_quat], dim=-1)

                self.env.scene["object"].write_root_pose_to_sim(
                    object_root_pose, env_ids=self._replay_env_ids
                )

        if hasattr(self.env, "scene") and "table" in self.env.scene.rigid_objects:
            # Use pre-loaded per-motion table metadata (loaded during setup_replay_motion)
            # _replay_table_pos and _replay_table_quat have shape (num_envs, 3) and (num_envs, 4)
            if self._replay_table_pos is not None:
                # Use per-env table positions directly (already the right shape)
                table_pos = self._replay_table_pos.clone()
                table_quat = self._replay_table_quat.clone()
            else:
                # Fallback: derive from object position with hardcoded offset
                table_pos = self._motion_lib.get_object_root_pos(
                    self._replay_motion_ids, torch.zeros_like(self._replay_time_steps)
                )[:, 0].clone()
                table_pos[:, 2] = 0.76  # Table height
                table_pos[:, 1] -= 0.15
                table_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device).repeat(
                    len(self._replay_env_ids), 1
                )

                # Apply table_offset if configured (for fallback path)
                table_offset = self.config.get("table_offset", None)
                if table_offset is not None:
                    offset_tensor = torch.tensor(
                        table_offset, device=self.device, dtype=table_pos.dtype
                    )
                    table_pos = table_pos + offset_tensor

            # Add environment origin offsets (same as robot and object)
            if hasattr(self, "_replay_custom_origins"):
                table_pos = table_pos + self._replay_custom_origins[self._replay_env_ids]
            else:
                table_pos = table_pos + self.env.scene.env_origins[self._replay_env_ids]

            table_root_pose = torch.cat([table_pos, table_quat], dim=-1)
            self.env.scene["table"].write_root_pose_to_sim(
                table_root_pose, env_ids=self._replay_env_ids
            )

        # Visualize contact points if enabled
        if (
            hasattr(self, "_replay_contact_visualizer")
            and self._replay_contact_visualizer is not None
            and hasattr(self, "_replay_vis_enabled")
            and self._replay_vis_enabled
        ):
            self._update_contact_center_visualization()

        self.env.sim.forward()

    def step_replay(self):
        """Step the replay forward for all environments. Call this in a loop to animate the motions.
        Returns False when replay is complete.
        """  # noqa: D205
        if not hasattr(self, "_replay_active") or not self._replay_active:
            return False

        # Update frames based on speed and direction
        if not self._replay_paused:
            if self._replay_reverse:
                # Playing in reverse
                self._replay_time_steps -= int(self._replay_speed)

                # Handle beginning of motions per environment
                at_start = self._replay_time_steps < 0
                if at_start.any():
                    if self._replay_loop:
                        # Loop to end when reaching start
                        self._replay_time_steps[at_start] = (
                            self._replay_num_steps_per_env[at_start] - 1
                        ).to(torch.long)
                    else:
                        # If any motion reached start and not looping, end replay
                        logger.info("Replay complete (reversed to start)!")
                        self._replay_active = False
                        return False
            else:
                # Playing forward (normal)
                self._replay_time_steps += int(self._replay_speed)

                # Handle end of motions per environment
                completed = self._replay_time_steps >= self._replay_num_steps_per_env
                if completed.any():
                    if self._replay_loop:
                        # Loop completed motions
                        self._replay_time_steps[completed] = 0
                        # Decrement loop counter if it's an integer (countdown mode)
                        if not isinstance(self._replay_loop, bool):
                            self._replay_loop -= 1
                            logger.info(f"Loop completed, {self._replay_loop} loops remaining")
                            if self._replay_loop <= 0:
                                logger.info("All loops complete!")
                                self._replay_active = False
                                return False
                    else:
                        # If any motion is complete and not looping, end replay
                        logger.info("Replay complete!")
                        self._replay_active = False
                        return False

        # Update the robot states
        self._update_replay_frame()

        # Print progress every 50 frames (based on max time step)
        max_time_step = self._replay_time_steps.max().item()
        if max_time_step % 50 == 0 and max_time_step > 0:
            progress = (max_time_step / self._replay_max_num_steps) * 100
            logger.info(
                f"Progress: {progress:.1f}% (max frame {max_time_step}/{self._replay_max_num_steps})"
            )

        return True
