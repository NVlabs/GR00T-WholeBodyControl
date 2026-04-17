"""Loss functions for token-based models."""

from pathlib import Path

import omegaconf
import torch
from torch import nn
import torch.nn.functional as F

from gear_sonic.isaac_utils import rotations
from gear_sonic.trl.utils import order_converter, torch_transform
from gear_sonic.utils import batch_normalizer
from gear_sonic.utils.motion_lib import torch_humanoid_batch


def create_humanoid(
    skeleton_name: str, device: torch.device = None
) -> torch_humanoid_batch.Humanoid_Batch:
    """Create a Humanoid_Batch for FK computations.

    Args:
        skeleton_name: Name of the skeleton config file (without .yaml extension)
                       e.g., "motion_g1_extended_toe" for the extended G1 skeleton
        device: Device to place the humanoid on

    Returns:
        Humanoid_Batch instance
    """
    if device is None:
        device = torch.device("cpu")

    groot_root = Path(__file__).parent.parent.parent.parent
    motion_yaml = (
        groot_root
        / "rl"
        / "config"
        / "manager_env"
        / "commands"
        / "terms"
        / f"{skeleton_name}.yaml"
    )

    cfg = omegaconf.OmegaConf.load(motion_yaml).motion.motion_lib_cfg
    return torch_humanoid_batch.Humanoid_Batch(cfg, device=device)


def decoder_output_to_egocentric_transforms(
    decoder_output: dict,
    decoder_cfg: dict,
    humanoid: torch_humanoid_batch.Humanoid_Batch,
    dof_converter: order_converter.IsaacLabMuJoCoConverter = None,
    include_extended: bool = False,
):
    """Convert decoder output to joint positions and 6D rotations (ortho6d).

    Always returns 6D rotations. For geodesic loss, convert to matrices in the loss function.

    Args:
        decoder_output: Dict with decoder output tensors
        decoder_cfg: Decoder config with 'outputs' key
        humanoid: Humanoid_Batch instance (already on correct device)
        dof_converter: DOF order converter (if None, assumes qpos is in MuJoCo order)
        include_extended: If True, compute and append extended body transforms (e.g., head, toes)

    Returns:
        egocentric_pos: Joint positions [..., num_bodies, 3]
        egocentric_rot_6d: Joint rotations in 6D format [..., num_bodies, 6]
    """
    output_keys = list(decoder_cfg["outputs"])

    if set(output_keys) == {"command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"}:
        """NOTE: there's a bug in command_multi_future_nonflat, where the temporal axis is incorrectly flattened.
        command_multi_future_nonflat: [..., num_future, 58] = [dof_pos(29), dof_vel(29)] in IsaacLab order
        motion_anchor_ori_b_mf_nonflat: [..., num_future, 6] = relative 6D orientation
        6D format: rot_mat[..., :2].reshape(6) = [R00, R01, R10, R11, R20, R21] (row-major of first 2 cols)

        Since we don't use the raltive motion anchor, we are considering the egocentric transforms.
        """
        orig_shape = decoder_output["command_multi_future_nonflat"].shape[:-1]  # [..., num_future]
        num_timesteps = orig_shape[-1]

        # The below processing to obtain dof_qpos is needed because of the observation bug
        dof_pos = decoder_output["command_multi_future_nonflat"][
            ..., : num_timesteps // 2, : humanoid.num_dof * 2
        ]
        dof_pos = dof_pos.reshape(-1, humanoid.num_dof)

        root_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(dof_pos.shape[0], 1).to(dof_pos)
        root_trans = torch.zeros(dof_pos.shape[0], 3).to(dof_pos)
        qpos = torch.cat([root_trans, root_quat, dof_pos], dim=-1)

        # Convert from IsaacLab to MuJoCo DOF order if converter provided
        if dof_converter is not None:
            qpos = dof_converter.to_mujoco(qpos)

        egocentric_pos, egocentric_rot = humanoid.qpos_to_global_transforms(
            qpos, include_extended=include_extended
        )

        egocentric_pos = egocentric_pos.view(*orig_shape, egocentric_pos.shape[-2], 3)
        egocentric_rot = egocentric_rot.view(*orig_shape, egocentric_rot.shape[-3], 3, 3)

        # Convert 3x3 rotation matrix to 6D (first two columns flattened)
        egocentric_rot_6d = rotations.mat_to_rot6d_first_two_cols(egocentric_rot)

        return egocentric_pos, egocentric_rot_6d

    elif set(output_keys) == {
        "command_multi_future_egocentric_joint_transforms_nonflat",
        "command_multi_future_root_transforms_nonflat",
    }:
        """
        command_multi_future_egocentric_joint_transforms_nonflat: [..., num_future, num_bodies, 9]
            - 9 = 3 (position) + 6 (6D rotation)
            - Joint positions and rotations relative to each frame's projected root
        command_multi_future_root_transforms_nonflat: [..., num_future, 9]
            - 9 = 3 (position) + 6 (6D rotation)
            - Root position and rotation relative to first reference frame
        """
        joint_transforms = decoder_output[
            "command_multi_future_egocentric_joint_transforms_nonflat"
        ]
        joint_transforms = joint_transforms.reshape(*joint_transforms.shape[:-1], -1, 9)

        egocentric_pos = joint_transforms[..., :3]  # [..., num_future, num_bodies, 3]
        egocentric_rot_6d = joint_transforms[..., 3:]  # [..., num_future, num_bodies, 6]

        # Convert from IsaacLab to MuJoCo DOF order if converter provided
        if dof_converter is not None:
            egocentric_pos = dof_converter.to_mujoco(egocentric_pos)
            egocentric_rot_6d = dof_converter.to_mujoco(egocentric_rot_6d)

        # Compute extended body transforms if requested
        if include_extended and humanoid.num_bodies_augment > humanoid.num_bodies:
            # Convert to matrices for FK computation (only for extended joints)
            egocentric_rot_mat = rotations.rot6d_to_mat_first_two_cols(egocentric_rot_6d)
            full_pos, full_rot_mat = humanoid.append_extended_transforms(
                egocentric_pos, egocentric_rot_mat
            )

            # Extract only extended joints and convert to 6D
            extended_pos = full_pos[..., humanoid.num_bodies :, :]
            extended_rot_mat = full_rot_mat[..., humanoid.num_bodies :, :, :]
            extended_rot_6d = rotations.mat_to_rot6d_first_two_cols(extended_rot_mat)

            # Concatenate: keep original 6D (no normalization) + extended 6D
            egocentric_pos = torch.cat([egocentric_pos, extended_pos], dim=-2)
            egocentric_rot_6d = torch.cat([egocentric_rot_6d, extended_rot_6d], dim=-2)

        return egocentric_pos, egocentric_rot_6d

    raise NotImplementedError(f"Unsupported decoder output format: {output_keys}")


def decoder_output_to_world_transforms(
    decoder_output: dict,
    decoder_cfg: dict,
    humanoid: torch_humanoid_batch.Humanoid_Batch,
    dof_converter: order_converter.IsaacLabMuJoCoConverter = None,
    include_extended: bool = False,
):
    """Convert decoder output to joint positions and rotation matrices in a consistent
    world coordinate frame (first reference frame's projected root, heading-aligned).

    For formats with per-frame egocentric transforms and root transforms, this
    reconstructs world transforms by applying each frame's root rotation/translation.

    Coordinate system: first frame's heading-aligned, ground-projected root.

    Args:
        decoder_output: Dict with decoder output tensors
        decoder_cfg: Decoder config with 'outputs' key
        humanoid: Humanoid_Batch instance (already on correct device)
        dof_converter: DOF order converter (if None, assumes qpos is in MuJoCo order)
        include_extended: If True, compute and append extended body transforms

    Returns:
        world_pos: Joint positions [..., num_future, num_bodies, 3]
        world_rot: Joint rotation matrices [..., num_future, num_bodies, 3, 3]
    """  # noqa: D205
    output_keys = list(decoder_cfg["outputs"])

    if set(output_keys) == {"command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"}:
        # qpos-based format with identity root — egocentric IS the world frame
        egocentric_pos, egocentric_rot_6d = decoder_output_to_egocentric_transforms(
            decoder_output,
            decoder_cfg,
            humanoid,
            dof_converter,
            include_extended=include_extended,
        )
        egocentric_rot_mat = rotations.rot6d_to_mat_first_two_cols(egocentric_rot_6d)
        return egocentric_pos, egocentric_rot_mat

    elif set(output_keys) == {
        "command_multi_future_egocentric_joint_transforms_nonflat",
        "command_multi_future_root_transforms_nonflat",
    }:
        # Get egocentric transforms and root transforms
        egocentric_pos, egocentric_rot_6d = decoder_output_to_egocentric_transforms(
            decoder_output,
            decoder_cfg,
            humanoid,
            dof_converter,
            include_extended=include_extended,
        )
        root_pos, root_rot_6d = decoder_output_to_root_transforms(
            decoder_output,
            decoder_cfg,
        )

        # root_rot = heading_0_inv * q_current_t  (full rotation relative to first frame heading)
        # egocentric transforms are in each frame's heading-aligned frame
        # To go from egocentric to world (first frame heading-aligned):
        #   - Extract heading-only (yaw) from root_rot for position/rotation transform
        #   - ego_pos is already in heading frame, so only yaw rotation is needed
        root_rot_mat = rotations.rot6d_to_mat_first_two_cols(root_rot_6d)  # [..., num_future, 3, 3]
        ego_rot_mat = rotations.rot6d_to_mat_first_two_cols(
            egocentric_rot_6d
        )  # [..., num_future, num_bodies, 3, 3]

        # Variable-frame masking can zero out padded root rotations.
        # matrix_to_quaternion assumes valid SO(3); sanitize before heading extraction.
        root_rot_mat = _sanitize_rotation_matrices(root_rot_mat)

        # Extract heading (yaw-only) rotation from root_rot_mat
        root_quat = rotations.matrix_to_quaternion(root_rot_mat)  # [..., num_future, 4] (wxyz)
        heading_quat = torch_transform.get_heading_q(
            root_quat
        )  # [..., num_future, 4] (wxyz, yaw only)
        heading_rot_mat = rotations.quaternion_to_matrix(heading_quat)  # [..., num_future, 3, 3]

        # Positions: [..., num_future, num_bodies, 3]
        # Zero out height (z) from root_pos — egocentric positions already encode
        # height relative to ground; adding root height would double-count it.
        root_pos_xy = root_pos.clone()
        root_pos_xy[..., 2] = 0.0
        world_pos = torch.matmul(
            heading_rot_mat.unsqueeze(-3), egocentric_pos.unsqueeze(-1)
        ).squeeze(-1) + root_pos_xy.unsqueeze(-2)

        # Rotations: [..., num_future, num_bodies, 3, 3]
        world_rot = torch.matmul(heading_rot_mat.unsqueeze(-3), ego_rot_mat)

        return world_pos, world_rot

    raise NotImplementedError(f"Unsupported decoder output format: {output_keys}")


def decoder_output_to_root_transforms(
    decoder_output: dict,
    decoder_cfg: dict,
):
    """Extract root position and rotation (as 6D) from decoder output.

    Always returns 6D rotations. For geodesic loss, convert to matrices in the loss function.

    Args:
        decoder_output: Dict with decoder output tensors
        decoder_cfg: Decoder config with 'outputs' key

    Returns:
        root_pos: Root position [..., num_future, 3] (zeros if not available in format)
        root_rot_6d: Root rotation in 6D format [..., num_future, 6]
    """
    output_keys = list(decoder_cfg["outputs"])

    if set(output_keys) == {"command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"}:
        # Root rotation from motion_anchor_ori_b_mf_nonflat (already 6D)
        root_rot_6d = decoder_output["motion_anchor_ori_b_mf_nonflat"]  # [..., num_future, 6]

        # Root position not directly available - return zeros with matching shape
        root_pos = torch.zeros(
            *root_rot_6d.shape[:-1], 3, device=root_rot_6d.device, dtype=root_rot_6d.dtype
        )

        return root_pos, root_rot_6d

    elif set(output_keys) == {
        "command_multi_future_egocentric_joint_transforms_nonflat",
        "command_multi_future_root_transforms_nonflat",
    }:
        # Root transforms are in command_multi_future_root_transforms_nonflat
        root_transforms = decoder_output[
            "command_multi_future_root_transforms_nonflat"
        ]  # [..., num_future, 9]

        root_pos = root_transforms[..., :3]  # [..., num_future, 3]
        root_rot_6d = root_transforms[..., 3:]  # [..., num_future, 6]

        return root_pos, root_rot_6d

    raise NotImplementedError(f"Unsupported decoder output format: {output_keys}")


def _extract_dof_pos(
    decoder_output: dict,
    decoder_cfg: dict,
    humanoid: torch_humanoid_batch.Humanoid_Batch,
    dof_converter: order_converter.IsaacLabMuJoCoConverter = None,
    egocentric_pos: torch.Tensor = None,
    egocentric_rot_6d: torch.Tensor = None,
):
    """Extract DOF angles from decoder output via inverse kinematics.

    Runs ``humanoid.global_transforms_to_qpos`` on egocentric transforms to
    recover DOF angles.  If ``egocentric_pos`` and ``egocentric_rot_6d`` are
    provided, uses them directly (avoids re-parsing decoder output).  Otherwise,
    calls ``decoder_output_to_egocentric_transforms`` to obtain them.

    Only supports the ``command_multi_future_egocentric_joint_transforms_nonflat``
    format; returns ``None`` for other formats.

    Returns:
        dof_pos: [..., num_future, num_dof] or None if format unsupported.
    """
    if egocentric_pos is None or egocentric_rot_6d is None:
        output_keys = list(decoder_cfg["outputs"])
        if set(output_keys) != {
            "command_multi_future_egocentric_joint_transforms_nonflat",
            "command_multi_future_root_transforms_nonflat",
        }:
            return None

        # Parse decoder output (no extended bodies — they have no qpos)
        egocentric_pos, egocentric_rot_6d = decoder_output_to_egocentric_transforms(
            decoder_output,
            decoder_cfg,
            humanoid,
            dof_converter,
            include_extended=False,
        )

    # Convert 6D -> 3x3 rotation matrices for IK
    rot_mat = rotations.rot6d_to_mat_first_two_cols(egocentric_rot_6d)  # [..., F, J, 3, 3]

    # Variable-frame masking can zero out padded frames — sanitize for IK.
    rot_mat = _sanitize_rotation_matrices(rot_mat)

    # Flatten leading dims for global_transforms_to_qpos which expects [B, T, J, 3, 3]
    num_frames = rot_mat.shape[-4]
    num_joints = rot_mat.shape[-3]
    lead_shape = rot_mat.shape[:-4]

    rot_mat_5d = rot_mat.reshape(-1, num_frames, num_joints, 3, 3)
    pos_4d = egocentric_pos.reshape(-1, num_frames, num_joints, 3)

    # IK: egocentric transforms -> qpos (heading cancels in local rotation computation)
    qpos = humanoid.global_transforms_to_qpos(rot_mat_5d, pos_4d)  # [flat_B, F, D]
    dof_pos = qpos[..., 7:]  # [flat_B, F, num_dof]

    return dof_pos.reshape(*lead_shape, num_frames, humanoid.num_dof)


def _sanitize_rotation_matrices(rot_mat: torch.Tensor) -> torch.Tensor:
    """Replace degenerate (near-zero) rotation matrices with identity.

    Variable-frame masking can zero out padded frames, producing zero 3x3
    matrices.  Downstream operations (matrix_to_quaternion, IK) assume valid
    SO(3) and produce NaN on such inputs.
    """
    norm_sq = (rot_mat * rot_mat).sum(dim=(-2, -1))
    eye = torch.eye(3, device=rot_mat.device, dtype=rot_mat.dtype)
    return torch.where((norm_sq < 1e-6)[..., None, None], eye, rot_mat)


def _apply_normalizer(normalizer, gt, pred, mask=None):
    """Update normalizer from gt, normalize both gt and pred.

    Flattens trailing dims to match normalizer's expected feature dim,
    updates running stats from valid gt samples, normalizes both tensors.

    Returns:
        (gt_normed, pred_normed) with same shapes as inputs.
    """
    orig_shape = gt.shape
    flat_dim = normalizer.num_features
    gt_flat = gt.reshape(-1, flat_dim)
    pred_flat = pred.reshape(-1, flat_dim)
    if mask is not None:
        # Broadcast mask to match gt shape (mask may have fewer trailing dims)
        m = mask
        while m.ndim < gt.ndim:
            m = m.unsqueeze(-1)
        valid_mask = m.expand_as(gt).reshape(-1, flat_dim)[:, 0].bool()
        if valid_mask.any():
            normalizer.update(gt_flat[valid_mask])
    else:
        normalizer.update(gt_flat)
    return (
        normalizer.normalize(gt_flat).reshape(orig_shape),
        normalizer.normalize(pred_flat).reshape(orig_shape),
    )


# =============================================================================
# Helper Functions
# =============================================================================


def compute_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Compute loss between prediction and target using specified loss type.

    Args:
        pred: Predicted tensor
        target: Target tensor
        loss_type: One of "mse", "l1", "huber", "cosine"

    Returns:
        Scalar loss tensor
    """
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    elif loss_type == "l1":
        return F.l1_loss(pred, target)
    elif loss_type == "huber":
        return F.huber_loss(pred, target)
    elif loss_type == "cosine":
        cosine_sim = F.cosine_similarity(pred, target, dim=-1)
        return (1 - cosine_sim).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def zero_loss(device: torch.device) -> torch.Tensor:
    """Return a zero loss tensor with requires_grad=True."""
    return torch.tensor(0.0, device=device, requires_grad=True)


def _get_device_from_loss_inputs(loss_inputs: dict) -> torch.device:
    """Get device from loss_inputs, handling kinematic-only mode where action_mean is None."""
    if loss_inputs.get("action_mean") is not None:
        return loss_inputs["action_mean"].device
    for val in loss_inputs.get("tokenizer_obs", {}).values():
        if isinstance(val, torch.Tensor):
            return val.device
    return torch.device("cpu")


def _build_frame_mask_for_loss(frame_mask, num_frames_in_tensor):
    """Build a frame mask for a loss tensor.

    Args:
        frame_mask: [..., max_frames] bool (from loss_inputs), or None
        num_frames_in_tensor: actual frame count in the loss tensor (may be < max_frames)

    Returns:
        float mask [..., num_frames], or None.  Consumers (_compute_masked_loss,
        _masked_geodesic_angle) auto-broadcast to the target shape.
    """
    if frame_mask is None:
        return None
    return frame_mask[..., :num_frames_in_tensor].float()


def _masked_geodesic_angle(pred_rot, gt_rot, mask=None, dt=1.0, eps=1e-6):  # noqa: D417
    """Compute geodesic angle loss on rotation matrices with optional masking.

    Args:
        pred_rot, gt_rot: [..., 3, 3] rotation matrices
        mask: float mask (1=valid) with frame dim, auto-broadcast to angles shape, or None
        dt: time step divisor (for velocity normalization)
        eps: clamp epsilon for numerical stability

    Returns:
        scalar loss
    """
    R_diff = torch.matmul(gt_rot.transpose(-1, -2), pred_rot)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    cos_angle = torch.clamp((trace - 1) / 2, -1.0 + eps, 1.0 - eps)
    angles = torch.acos(cos_angle) / dt  # [..., (B)]
    if mask is not None:
        # Auto-expand mask [..., F] to match angles [..., F, (B)]
        while mask.dim() < angles.dim():
            mask = mask.unsqueeze(-1)
        return (angles * mask).sum() / mask.expand_as(angles).sum().clamp(min=1)
    return angles.mean()


def _build_vel_frame_mask(frame_mask, num_frames_in_tensor):
    """Build a frame mask for velocity (finite-difference) tensors.

    Velocity at frame i uses frames i and i+1, so frame i is valid only if
    both frame i and frame i+1 are valid.

    Args:
        frame_mask: [..., max_frames] bool, or None
        num_frames_in_tensor: number of frames in the position tensor (vel has num-1)

    Returns:
        float mask [..., num_frames-1], or None.  Consumers auto-broadcast.
    """
    if frame_mask is None:
        return None
    mask = frame_mask[..., :num_frames_in_tensor]
    return (mask[..., :-1] & mask[..., 1:]).float()


def _compute_masked_loss(pred, gt, mask, loss_type="mse"):
    """Compute loss with optional mask, supporting all loss types.

    Uses torch loss functions with reduction='none' so masking works uniformly.
    mask is auto-broadcast to match the element-wise loss shape.
    """
    if loss_type == "mse":
        elem = F.mse_loss(pred, gt, reduction="none")
    elif loss_type == "l1":
        elem = F.l1_loss(pred, gt, reduction="none")
    elif loss_type == "huber":
        elem = F.huber_loss(pred, gt, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")
    if mask is None:
        return elem.mean()
    # Auto-expand mask [..., F] to match elem [..., F, ...]
    while mask.dim() < elem.dim():
        mask = mask.unsqueeze(-1)
    masked = elem * mask
    num_valid = mask.expand_as(elem).sum().clamp(min=1)
    return masked.sum() / num_valid


# =============================================================================
# Reconstruction Loss
# =============================================================================


class G1ReconLoss(nn.Module):

    def __init__(self, loss_type="mse", **kwargs):  # noqa: ARG002
        super().__init__()
        self.loss_type = loss_type

    def forward(self, loss_inputs):
        tokenizer_obs = loss_inputs["tokenizer_obs"]
        decoders_cfg = loss_inputs["decoders_cfg"]
        decoded_outputs = loss_inputs["decoded_outputs"]
        frame_mask = loss_inputs.get("frame_mask", None)

        g1_motion_output = torch.cat(
            [tokenizer_obs[key] for key in decoders_cfg["g1_kin"]["outputs"]], dim=-1
        )
        g1_motion_output_pred = torch.cat(
            [decoded_outputs["g1_kin"][key] for key in decoded_outputs["g1_kin"]], dim=-1
        )

        # Align temporal dim: truncate the longer to the shorter
        if g1_motion_output_pred.shape[-2] < g1_motion_output.shape[-2]:
            g1_motion_output = g1_motion_output[..., : g1_motion_output_pred.shape[-2], :]
        elif g1_motion_output_pred.shape[-2] > g1_motion_output.shape[-2]:
            g1_motion_output_pred = g1_motion_output_pred[..., : g1_motion_output.shape[-2], :]

        # Build frame mask: shape [..., num_frames, 1] for broadcasting
        num_frames = g1_motion_output.shape[-2]
        mask = _build_frame_mask_for_loss(frame_mask, num_frames)

        return _compute_masked_loss(g1_motion_output_pred, g1_motion_output, mask, self.loss_type)


class G1ReconLossAligned(G1ReconLoss):
    """Same as G1ReconLoss but uses a single key list for target and pred so that
    encoder input = g1_kin output = loss. Expects loss_inputs["recon_target_keys"].
    """  # noqa: D205

    def forward(self, loss_inputs):
        tokenizer_obs = loss_inputs["tokenizer_obs"]
        decoded_outputs = loss_inputs["decoded_outputs"]
        frame_mask = loss_inputs.get("frame_mask", None)
        keys = list(loss_inputs["recon_target_keys"])
        g1_motion_output = torch.cat([tokenizer_obs[k] for k in keys], dim=-1)
        g1_motion_output_pred = torch.cat([decoded_outputs["g1_kin"][k] for k in keys], dim=-1)

        if g1_motion_output_pred.shape[-2] < g1_motion_output.shape[-2]:
            g1_motion_output = g1_motion_output[..., : g1_motion_output_pred.shape[-2], :]
        elif g1_motion_output_pred.shape[-2] > g1_motion_output.shape[-2]:
            g1_motion_output_pred = g1_motion_output_pred[..., : g1_motion_output.shape[-2], :]

        num_frames = g1_motion_output.shape[-2]
        mask = _build_frame_mask_for_loss(frame_mask, num_frames)

        return _compute_masked_loss(g1_motion_output_pred, g1_motion_output, mask, self.loss_type)


# =============================================================================
# G1-SMPL Latent Alignment Loss
# =============================================================================


class G1SmplLatentLoss(nn.Module):
    """Loss that compares the encoded latents between g1 and smpl encoders.
    This encourages the latent representations to be similar across different
    motion representation formats.
    """  # noqa: D205

    def __init__(self, loss_type="mse", **kwargs):  # noqa: ARG002
        super().__init__()
        self.loss_type = loss_type

    def forward(self, loss_inputs):
        encoded_latents = loss_inputs["encoded_latents"]
        encoder_masks = loss_inputs["encoder_masks"]

        # Get g1 latents that have corresponding smpl data
        g1_latents = encoded_latents["g1"]
        smpl_latents = encoded_latents["smpl"]

        # Extract only the g1 samples that have corresponding smpl
        g1_latents_matched = g1_latents[encoder_masks["g1_has_smpl"]]
        if g1_latents_matched.shape[0] == 0:
            return torch.tensor(0.0, device=g1_latents.device)

        # Compute loss based on loss_type
        if self.loss_type == "mse":
            loss = F.mse_loss(g1_latents_matched, smpl_latents)
        elif self.loss_type == "l1":
            loss = F.l1_loss(g1_latents_matched, smpl_latents)
        elif self.loss_type == "huber":
            loss = F.huber_loss(g1_latents_matched, smpl_latents)
        elif self.loss_type == "cosine":
            # Cosine distance: 1 - cosine_similarity
            cosine_sim = F.cosine_similarity(g1_latents_matched, smpl_latents, dim=-1)
            loss = (1 - cosine_sim).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


class TeleopSmplLatentLoss(nn.Module):
    """Loss that compares the encoded latents between teleop and smpl encoders.
    This encourages the latent representations to be similar across different
    motion representation formats.
    """  # noqa: D205

    def __init__(self, loss_type="mse", **kwargs):  # noqa: ARG002
        super().__init__()
        self.loss_type = loss_type

    def forward(self, loss_inputs):
        encoded_latents = loss_inputs["encoded_latents"]
        encoder_masks = loss_inputs["encoder_masks"]

        # Get teleop and smpl latents
        tlp_latents = encoded_latents["teleop"]
        smpl_latents = encoded_latents["smpl"]

        # teleop_has_smpl selects teleop samples that also have smpl active
        tlp_latents_matched = tlp_latents[encoder_masks["teleop_has_smpl"]]
        # smpl_has_teleop selects smpl samples that also have teleop active
        smpl_latents_matched = smpl_latents[encoder_masks["smpl_has_teleop"]]

        # Return 0 loss if no matching samples
        if tlp_latents_matched.shape[0] == 0:
            return torch.tensor(0.0, device=tlp_latents.device)

        # Compute loss based on loss_type
        if self.loss_type == "mse":
            loss = F.mse_loss(tlp_latents_matched, smpl_latents_matched)
        elif self.loss_type == "l1":
            loss = F.l1_loss(tlp_latents_matched, smpl_latents_matched)
        elif self.loss_type == "huber":
            loss = F.huber_loss(tlp_latents_matched, smpl_latents_matched)
        elif self.loss_type == "cosine":
            # Cosine distance: 1 - cosine_similarity
            cosine_sim = F.cosine_similarity(tlp_latents_matched, smpl_latents_matched, dim=-1)
            loss = (1 - cosine_sim).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


# =============================================================================
# G1-Teleop Latent Alignment Loss
# =============================================================================


class G1TeleopLatentLoss(nn.Module):
    """Loss that compares the encoded latents between g1 and teleop encoders.
    This encourages the latent representations to be similar across different
    motion representation formats.
    """  # noqa: D205

    def __init__(self, loss_type="mse", **kwargs):  # noqa: ARG002
        super().__init__()
        self.loss_type = loss_type

    def forward(self, loss_inputs):
        encoded_latents = loss_inputs["encoded_latents"]
        encoder_masks = loss_inputs["encoder_masks"]

        # Get g1 and teleop latents
        g1_latents = encoded_latents["g1"]
        tlp_latents = encoded_latents["teleop"]

        # g1_has_teleop selects g1 samples that also have teleop active
        g1_latents_matched = g1_latents[encoder_masks["g1_has_teleop"]]
        # teleop_has_g1 selects teleop samples that also have g1 active
        tlp_latents_matched = tlp_latents[encoder_masks["teleop_has_g1"]]

        # Return 0 loss if no matching samples
        if g1_latents_matched.shape[0] == 0:
            return torch.tensor(0.0, device=g1_latents.device)

        # Compute loss based on loss_type
        if self.loss_type == "mse":
            loss = F.mse_loss(g1_latents_matched, tlp_latents_matched)
        elif self.loss_type == "l1":
            loss = F.l1_loss(g1_latents_matched, tlp_latents_matched)
        elif self.loss_type == "huber":
            loss = F.huber_loss(g1_latents_matched, tlp_latents_matched)
        elif self.loss_type == "cosine":
            # Cosine distance: 1 - cosine_similarity
            cosine_sim = F.cosine_similarity(g1_latents_matched, tlp_latents_matched, dim=-1)
            loss = (1 - cosine_sim).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


# =============================================================================
# SOMA Latent Consistency Loss
# =============================================================================


class G1SomaLatentLoss(nn.Module):
    """Loss that compares the encoded latents between g1 and soma encoders.
    This encourages the latent representations to be similar across different
    motion representation formats.
    """  # noqa: D205

    def __init__(self, loss_type="mse", **kwargs):  # noqa: ARG002
        super().__init__()
        self.loss_type = loss_type

    def forward(self, loss_inputs):
        encoded_latents = loss_inputs["encoded_latents"]
        encoder_masks = loss_inputs["encoder_masks"]

        # Get g1 latents that have corresponding soma data
        g1_latents = encoded_latents["g1"]
        soma_latents = encoded_latents["soma"]

        # Extract only the g1 samples that have corresponding soma
        g1_latents_matched = g1_latents[encoder_masks["g1_has_soma"]]
        if g1_latents_matched.shape[0] == 0:
            return torch.tensor(0.0, device=g1_latents.device)

        assert g1_latents_matched.shape == soma_latents.shape, (
            f"Shape mismatch: g1_matched={g1_latents_matched.shape}, soma={soma_latents.shape}. "
            "Ensure g1 encoder is co-activated when soma is sampled."
        )

        # Compute loss based on loss_type
        if self.loss_type == "mse":
            loss = F.mse_loss(g1_latents_matched, soma_latents)
        elif self.loss_type == "l1":
            loss = F.l1_loss(g1_latents_matched, soma_latents)
        elif self.loss_type == "huber":
            loss = F.huber_loss(g1_latents_matched, soma_latents)
        elif self.loss_type == "cosine":
            cosine_sim = F.cosine_similarity(g1_latents_matched, soma_latents, dim=-1)
            loss = (1 - cosine_sim).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


# =============================================================================
# Cycle Consistency Loss
# =============================================================================


class ReencodedSmplG1LatentLoss(nn.Module):
    """Loss that compares the reencoded g1 latents (from smpl-to-g1 reconstruction)
    with the original g1 latents. This encourages the reconstructed g1 motion
    to be encodable back to the same latent space as the original g1 motion.
    """  # noqa: D205

    def __init__(self, loss_type="mse", **kwargs):  # noqa: ARG002
        super().__init__()
        self.loss_type = loss_type

    def forward(self, loss_inputs):
        reencoded_smpl_g1_latents = loss_inputs["reencoded_smpl_g1_latents"]
        encoded_latents = loss_inputs["encoded_latents"]
        encoder_masks = loss_inputs["encoder_masks"]

        # Get g1 latents that have corresponding smpl data
        g1_latents = encoded_latents["g1"]
        g1_latents_matched = g1_latents[encoder_masks["g1_has_smpl"]]
        if g1_latents_matched.shape[0] == 0:
            return torch.tensor(0.0, device=g1_latents.device)

        # Compute loss based on loss_type
        if self.loss_type == "mse":
            loss = F.mse_loss(reencoded_smpl_g1_latents, g1_latents_matched)
        elif self.loss_type == "l1":
            loss = F.l1_loss(reencoded_smpl_g1_latents, g1_latents_matched)
        elif self.loss_type == "huber":
            loss = F.huber_loss(reencoded_smpl_g1_latents, g1_latents_matched)
        elif self.loss_type == "cosine":
            # Cosine distance: 1 - cosine_similarity
            cosine_sim = F.cosine_similarity(reencoded_smpl_g1_latents, g1_latents_matched, dim=-1)
            loss = (1 - cosine_sim).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return loss


# =============================================================================
# Compliance-Aware Latent Alignment Losses
# =============================================================================


class G1SmplComplianceLatentLoss(nn.Module):
    """G1→SMPL latent alignment loss with compliance-aware filtering.

    This loss ONLY applies when compliance ≈ 0 (stiff mode).

    Rationale:
    - G1 encoder encodes pure kinematics (no compliance input)
    - SMPL encoder receives compliance as input
    - Only in stiff mode (compliance=0) should both produce similar latents
    - In compliant mode, latents may legitimately diverge

    Uses paired_g1_smpl_latents which ensures both latents are for the SAME environments.

    Note: G1 latents are pre-detached in UniversalTokenModule for memory optimization.
    The detach_g1_target parameter is kept for API compatibility but has no effect
    when using the default module configuration.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        compliance_threshold: float = 0.01,
        detach_g1_target: bool = True,  # Note: pre-detached in module
        debug: bool = False,
        debug_print_every: int = 500,
        **kwargs,  # noqa: ARG002
    ):
        super().__init__()
        self.loss_type = loss_type
        self.compliance_threshold = compliance_threshold
        self.detach_g1_target = detach_g1_target
        self.debug = debug
        self.debug_print_every = debug_print_every
        self._call_count = 0

    def forward(self, loss_inputs: dict) -> torch.Tensor:
        paired_g1_smpl_latents = loss_inputs.get("paired_g1_smpl_latents")

        if paired_g1_smpl_latents is None:
            return zero_loss(_get_device_from_loss_inputs(loss_inputs))

        # Note: g1 latents are pre-detached in UniversalTokenModule
        g1_latents = paired_g1_smpl_latents["g1"]
        smpl_latents = paired_g1_smpl_latents["smpl"]

        if g1_latents.shape[0] == 0 or smpl_latents.shape[0] == 0:
            return zero_loss(g1_latents.device)

        # Filter to stiff samples only (compliance ≈ 0)
        compliance_values = paired_g1_smpl_latents.get("compliance")
        if self.debug and compliance_values.max() > 0.001:
            print(  # noqa: T201
                f"Compliance values: {compliance_values.max()}, {compliance_values.min()}"
            )  # noqa: T201

        if compliance_values is not None:
            # All 3 compliance dimensions must be near zero
            is_stiff = (compliance_values.abs() <= self.compliance_threshold).all(dim=-1)

            self._call_count += 1
            if self.debug and self._call_count % self.debug_print_every == 1:
                num_stiff = is_stiff.sum().item()
                total = is_stiff.shape[0]
                print(  # noqa: T201
                    f"[G1SmplComplianceLatentLoss] Stiff samples: {num_stiff}/{total} "
                    f"({100*num_stiff/max(1,total):.1f}%)"
                )

            g1_latents = g1_latents[is_stiff]
            smpl_latents = smpl_latents[is_stiff]

            if g1_latents.shape[0] == 0:
                return zero_loss(_get_device_from_loss_inputs(loss_inputs))

        return compute_loss(smpl_latents, g1_latents, self.loss_type)


class TeleopSmplComplianceLatentLoss(nn.Module):
    """Teleop→SMPL latent alignment for compliance-aware training.

    Compares latents between teleop and smpl encoders
    for the SAME motion under the SAME compliance level.

    Key Design Points:
    - Applied to ALL compliance values (enables compliant mode learning!)
    - teleop is the BRIDGE (uses G1-space joints, closer to G1)
    - smpl is further from G1, needs more learning
    - detach_teleop_target=True: teleop is teacher, smpl learns

    This loss enables smpl encoder to learn compliance-aware
    behavior from the teleop bridge encoder.

    Note: Teleop latents are pre-detached in UniversalTokenModule for memory optimization.
    The detach_teleop_target parameter is kept for API compatibility.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        detach_teleop_target: bool = True,  # Note: pre-detached in module
        debug: bool = False,
        debug_print_every: int = 1000,
        **kwargs,  # noqa: ARG002
    ):
        super().__init__()
        self.loss_type = loss_type
        self.detach_teleop_target = detach_teleop_target
        self.debug = debug
        self.debug_print_every = debug_print_every
        self._call_count = 0

    def forward(self, loss_inputs: dict) -> torch.Tensor:
        # Use paired_compliance_latents computed for same motion + same compliance
        paired_compliance_latents = loss_inputs.get("paired_compliance_latents")

        if paired_compliance_latents is None:
            return zero_loss(_get_device_from_loss_inputs(loss_inputs))

        # Note: teleop latents are pre-detached in UniversalTokenModule
        teleop_latents = paired_compliance_latents["teleop"]
        smpl_latents = paired_compliance_latents["smpl"]

        # Debug logging
        self._call_count += 1
        if self.debug and self._call_count % self.debug_print_every == 1:
            self._print_debug_info(loss_inputs, teleop_latents, smpl_latents)

        if teleop_latents.shape[0] == 0 or smpl_latents.shape[0] == 0:
            return zero_loss(teleop_latents.device)

        return compute_loss(smpl_latents, teleop_latents, self.loss_type)

    def _print_debug_info(
        self, loss_inputs: dict, teleop_latents: torch.Tensor, smpl_latents: torch.Tensor
    ):
        """Print debug information about paired latents."""
        debug_info = loss_inputs.get("paired_compliance_debug_info")
        if debug_info is None:
            return

        print("\n" + "=" * 80)  # noqa: T201
        print(f"[TeleopSmplComplianceLatentLoss DEBUG] Call #{self._call_count}")  # noqa: T201
        print("=" * 80)  # noqa: T201

        total_envs = debug_info.get("total_envs", "unknown")
        num_paired = debug_info["num_paired_envs"]
        ratio = f"{num_paired}/{total_envs}" if total_envs != "unknown" else str(num_paired)
        pct = (
            f" ({100*num_paired/total_envs:.1f}%)"
            if isinstance(total_envs, int) and total_envs > 0
            else ""
        )

        print(f"  Paired envs: {ratio}{pct}")  # noqa: T201
        print(f"  teleop_latents shape: {teleop_latents.shape}")  # noqa: T201
        print(f"  smpl_latents shape: {smpl_latents.shape}")  # noqa: T201

        if num_paired > 0:
            print("\n  --- LATENT COMPARISON ---")  # noqa: T201
            print(f"  teleop[0,:5]: {teleop_latents[0,:5].detach()}")  # noqa: T201
            print(f"  smpl[0,:5]:   {smpl_latents[0,:5].detach()}")  # noqa: T201
            l2_diff = ((teleop_latents - smpl_latents) ** 2).mean(dim=-1)
            print(f"  L2 diff: {l2_diff[:min(4, l2_diff.shape[0])].detach()}")  # noqa: T201
        else:
            print("\n  WARNING: No paired envs!")  # noqa: T201
            print("  Possible causes: no SMPL data or too few envs")  # noqa: T201

        print("=" * 80 + "\n")  # noqa: T201


class ReencodedSmplG1ComplianceLatentLoss(nn.Module):
    """Cycle consistency loss for decoder regularization.

    Re-encodes the decoded G1 motion back to latent space and compares
    with the original G1 latent.

    Compliance-Aware Design:
    - compliance_aware=True (default): Only apply in stiff mode (compliance ≈ 0)
      In stiff mode, decoder should produce pure kinematic motion that
      can be re-encoded to the same latent space.
    - compliance_aware=False: Apply to all samples

    Note: Target G1 latents (original_g1_latents_for_reencode) are pre-detached
    in UniversalTokenModule. The detach_target parameter is kept for API compatibility.
    """

    def __init__(
        self,
        loss_type: str = "mse",
        detach_target: bool = True,  # Note: pre-detached in module
        compliance_aware: bool = True,
        compliance_threshold: float = 0.01,
        debug: bool = False,
        debug_print_every: int = 500,
        **kwargs,  # noqa: ARG002
    ):
        super().__init__()
        self.loss_type = loss_type
        self.detach_target = detach_target
        self.compliance_aware = compliance_aware
        self.compliance_threshold = compliance_threshold
        self.debug = debug
        self.debug_print_every = debug_print_every
        self._call_count = 0

    def forward(self, loss_inputs: dict) -> torch.Tensor:
        reencoded_latents = loss_inputs.get("reencoded_smpl_g1_latents")
        # Note: g1 latents are pre-detached in UniversalTokenModule
        g1_latents = loss_inputs.get("original_g1_latents_for_reencode")

        if reencoded_latents is None or g1_latents is None:
            return zero_loss(_get_device_from_loss_inputs(loss_inputs))

        if reencoded_latents.shape[0] == 0 or g1_latents.shape[0] == 0:
            return zero_loss(g1_latents.device)

        # Compliance-aware filtering
        if self.compliance_aware:
            paired_g1_smpl_latents = loss_inputs.get("paired_g1_smpl_latents")
            compliance_values = None
            if paired_g1_smpl_latents is not None:
                compliance_values = paired_g1_smpl_latents.get("compliance")

            if compliance_values is not None:
                is_stiff = (compliance_values.abs() < self.compliance_threshold).all(dim=-1)

                self._call_count += 1
                if self.debug and self._call_count % self.debug_print_every == 1:
                    num_stiff = is_stiff.sum().item()
                    total = is_stiff.shape[0]
                    print(  # noqa: T201
                        f"[ReencodedSmplG1LatentLoss] Stiff samples: {num_stiff}/{total} "
                        f"({100*num_stiff/max(1,total):.1f}%)"
                    )

                reencoded_latents = reencoded_latents[is_stiff]
                g1_latents = g1_latents[is_stiff]

                if reencoded_latents.shape[0] == 0:
                    return zero_loss(_get_device_from_loss_inputs(loss_inputs))

        return compute_loss(reencoded_latents, g1_latents, self.loss_type)


class LatentL2Loss(nn.Module):
    """L2 regularization on the latent residual (first latent_dim dimensions of action_mean).

    Encourages the policy to make small, focused adjustments to the pretrained ATM
    rather than completely overriding its behavior.

    loss = mean(latent_residual^2)
    """

    def __init__(self, latent_dim=64, **kwargs):  # noqa: ARG002
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, loss_inputs):
        action_mean = loss_inputs["action_mean"]
        latent_residual = action_mean[..., : self.latent_dim]
        return (latent_residual**2).mean()


class LatentL1Loss(nn.Module):
    """L1 regularization on the latent residual (sparsity-inducing).

    Encourages sparse modifications to the pretrained ATM.
    """

    def __init__(self, latent_dim=64, **kwargs):  # noqa: ARG002
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, loss_inputs):
        action_mean = loss_inputs["action_mean"]
        latent_residual = action_mean[..., : self.latent_dim]
        return torch.abs(latent_residual).mean()


# =============================================================================
# FK-based Kinematic Losses
# Use decoder_output_to_egocentric_transforms to get egocentric positions/rotations
# =============================================================================


class G1JointPositionLoss(nn.Module):
    """Loss on joint body positions via FK with optional velocity loss."""

    def __init__(
        self,
        loss_type="mse",
        vel_weight=0.0,
        dt=0.1,
        include_extended=False,
        coordinate_frame="egocentric",
        normalize=False,
        **kwargs,
    ):
        super().__init__()
        self._skeleton_name = kwargs.get("skeleton_name", "motion_g1_extended_toe")
        self.loss_type = loss_type
        self.vel_weight = vel_weight
        self.dt = (
            dt  # Time step between future frames (from env.commands.motion.dt_future_ref_frames)
        )
        self._include_extended = include_extended
        assert coordinate_frame in (
            "egocentric",
            "world",
        ), f"coordinate_frame must be 'egocentric' or 'world', got '{coordinate_frame}'"
        self.coordinate_frame = coordinate_frame
        self._humanoid = create_humanoid(self._skeleton_name)
        self._dof_converter = order_converter.get_converter_for_skeleton(self._skeleton_name)
        if normalize:
            num_b = (
                self._humanoid.num_bodies_augment
                if self._include_extended
                else self._humanoid.num_bodies
            )
            self._normalizer = batch_normalizer.BatchNormNormalizer((num_b * 3,))
            self._vel_normalizer = batch_normalizer.BatchNormNormalizer((num_b * 3,))
        else:
            self._normalizer = None
            self._vel_normalizer = None

    def _get_positions(self, decoder_output, decoder_cfg):
        if self.coordinate_frame == "world":
            pos, _ = decoder_output_to_world_transforms(
                decoder_output,
                decoder_cfg,
                self._humanoid,
                self._dof_converter,
                include_extended=self._include_extended,
            )
        else:
            pos, _ = decoder_output_to_egocentric_transforms(
                decoder_output,
                decoder_cfg,
                self._humanoid,
                self._dof_converter,
                include_extended=self._include_extended,
            )
        return pos

    def forward(self, loss_inputs):
        tokenizer_obs = loss_inputs["tokenizer_obs"]
        decoders_cfg = loss_inputs["decoders_cfg"]
        decoded_outputs = loss_inputs["decoded_outputs"]
        frame_mask = loss_inputs.get("frame_mask", None)

        device = _get_device_from_loss_inputs(loss_inputs)
        if self._humanoid.device != device:
            self._humanoid = self._humanoid.to(device=device)

        pos_gt = self._get_positions(tokenizer_obs, decoders_cfg["g1_kin"])
        pos_pred = self._get_positions(decoded_outputs["g1_kin"], decoders_cfg["g1_kin"])

        # Position loss
        # TCN/conv decoders may produce fewer frames; truncate gt to match pred (time dim -3)
        if pos_pred.shape[-3] < pos_gt.shape[-3]:
            pos_gt = pos_gt[..., : pos_pred.shape[-3], :, :]

        num_frames = pos_gt.shape[-3]
        pos_mask = _build_frame_mask_for_loss(frame_mask, num_frames)

        # Compute velocity from raw positions BEFORE normalization
        vel_gt = vel_pred = None
        if self.vel_weight > 0 and num_frames > 1:
            vel_gt = (pos_gt[..., 1:, :, :] - pos_gt[..., :-1, :, :]) / self.dt
            vel_pred = (pos_pred[..., 1:, :, :] - pos_pred[..., :-1, :, :]) / self.dt

        # Apply running normalization if enabled
        if self._normalizer is not None:
            pos_gt, pos_pred = _apply_normalizer(self._normalizer, pos_gt, pos_pred, pos_mask)

        pos_loss = _compute_masked_loss(pos_pred, pos_gt, pos_mask, self.loss_type)

        # Velocity loss (finite difference along future frames, normalized by dt)
        if self.vel_weight > 0 and vel_gt is not None:
            vel_mask = _build_vel_frame_mask(frame_mask, num_frames)
            if self._vel_normalizer is not None:
                vel_gt, vel_pred = _apply_normalizer(
                    self._vel_normalizer, vel_gt, vel_pred, vel_mask
                )
            vel_loss = _compute_masked_loss(vel_pred, vel_gt, vel_mask, self.loss_type)
            return pos_loss + self.vel_weight * vel_loss

        return pos_loss


class G1JointRotationLoss(nn.Module):
    """Loss on joint rotations via FK with optional velocity loss. Uses geodesic or Frobenius norm."""

    def __init__(
        self,
        loss_type="frobenius",
        vel_weight=0.0,
        dt=0.1,
        include_extended=False,
        normalize=False,
        **kwargs,
    ):
        super().__init__()
        self._skeleton_name = kwargs.get("skeleton_name", "motion_g1_extended_toe")
        self.loss_type = loss_type
        self.vel_weight = vel_weight
        self.dt = (
            dt  # Time step between future frames (from env.commands.motion.dt_future_ref_frames)
        )
        self._include_extended = include_extended
        self._humanoid = create_humanoid(self._skeleton_name)
        self._dof_converter = order_converter.get_converter_for_skeleton(self._skeleton_name)
        # Normalization only makes sense for frobenius (element-wise); geodesic operates on SO(3)
        if normalize and loss_type == "frobenius":
            num_b = (
                self._humanoid.num_bodies_augment
                if self._include_extended
                else self._humanoid.num_bodies
            )
            self._normalizer = batch_normalizer.BatchNormNormalizer((num_b * 6,))
            # Velocity normalizer operates on DOF angles (extended joints have no qpos)
            self._vel_normalizer = batch_normalizer.BatchNormNormalizer((self._humanoid.num_dof,))
        else:
            self._normalizer = None
            self._vel_normalizer = None

    def forward(self, loss_inputs):
        tokenizer_obs = loss_inputs["tokenizer_obs"]
        decoders_cfg = loss_inputs["decoders_cfg"]
        decoded_outputs = loss_inputs["decoded_outputs"]
        frame_mask = loss_inputs.get("frame_mask", None)

        device = _get_device_from_loss_inputs(loss_inputs)
        if self._humanoid.device != device:
            self._humanoid = self._humanoid.to(device=device)

        # Get egocentric rotations (with extended bodies if configured)
        _, egocentric_rot_6d_gt = decoder_output_to_egocentric_transforms(
            tokenizer_obs,
            decoders_cfg["g1_kin"],
            self._humanoid,
            self._dof_converter,
            include_extended=self._include_extended,
        )
        _, egocentric_rot_6d_pred = decoder_output_to_egocentric_transforms(
            decoded_outputs["g1_kin"],
            decoders_cfg["g1_kin"],
            self._humanoid,
            self._dof_converter,
            include_extended=self._include_extended,
        )

        # TCN/conv decoders may produce fewer frames; truncate gt to match pred (time dim -3)
        if egocentric_rot_6d_pred.shape[-3] < egocentric_rot_6d_gt.shape[-3]:
            egocentric_rot_6d_gt = egocentric_rot_6d_gt[
                ..., : egocentric_rot_6d_pred.shape[-3], :, :
            ]

        num_frames = egocentric_rot_6d_gt.shape[-3]

        if self.loss_type == "frobenius":
            rot_mask = _build_frame_mask_for_loss(frame_mask, num_frames)

            # Compute velocity from DOF positions (qpos) BEFORE normalization
            vel_gt = vel_pred = None
            if self.vel_weight > 0 and num_frames > 1:
                dof_pos_gt = _extract_dof_pos(
                    tokenizer_obs,
                    decoders_cfg["g1_kin"],
                    self._humanoid,
                    self._dof_converter,
                )
                dof_pos_pred = _extract_dof_pos(
                    decoded_outputs["g1_kin"],
                    decoders_cfg["g1_kin"],
                    self._humanoid,
                    self._dof_converter,
                )
                if dof_pos_gt is not None and dof_pos_pred is not None:
                    # Truncate gt to match pred frames (same as egocentric transforms above)
                    if dof_pos_pred.shape[-2] < dof_pos_gt.shape[-2]:
                        dof_pos_gt = dof_pos_gt[..., : dof_pos_pred.shape[-2], :]
                    vel_gt = (dof_pos_gt[..., 1:, :] - dof_pos_gt[..., :-1, :]) / self.dt
                    vel_pred = (dof_pos_pred[..., 1:, :] - dof_pos_pred[..., :-1, :]) / self.dt

            # Apply running normalization if enabled
            if self._normalizer is not None:
                egocentric_rot_6d_gt, egocentric_rot_6d_pred = _apply_normalizer(
                    self._normalizer, egocentric_rot_6d_gt, egocentric_rot_6d_pred, rot_mask
                )

            rot_loss = _compute_masked_loss(
                egocentric_rot_6d_pred, egocentric_rot_6d_gt, rot_mask, "mse"
            )

            # Velocity loss on DOF positions
            if self.vel_weight > 0 and vel_gt is not None:
                vel_mask = _build_vel_frame_mask(frame_mask, num_frames)
                if self._vel_normalizer is not None:
                    vel_gt, vel_pred = _apply_normalizer(
                        self._vel_normalizer, vel_gt, vel_pred, vel_mask
                    )
                vel_loss = _compute_masked_loss(vel_pred, vel_gt, vel_mask, "mse")
                return rot_loss + self.vel_weight * vel_loss

            return rot_loss

        elif self.loss_type == "geodesic":
            # Convert 6D to 3x3 rotation matrices for geodesic loss
            egocentric_rot_gt = rotations.rot6d_to_mat_first_two_cols(egocentric_rot_6d_gt)
            egocentric_rot_pred = rotations.rot6d_to_mat_first_two_cols(egocentric_rot_6d_pred)

            geo_mask = _build_frame_mask_for_loss(frame_mask, num_frames)
            rot_loss = _masked_geodesic_angle(egocentric_rot_pred, egocentric_rot_gt, mask=geo_mask)

            # Angular velocity loss (relative rotation between consecutive frames)
            if (
                self.vel_weight > 0 and egocentric_rot_gt.shape[-4] > 1
            ):  # [..., num_future, num_bodies, 3, 3]
                rot_vel_gt = torch.matmul(
                    egocentric_rot_gt[..., 1:, :, :, :],
                    egocentric_rot_gt[..., :-1, :, :, :].transpose(-1, -2),
                )
                rot_vel_pred = torch.matmul(
                    egocentric_rot_pred[..., 1:, :, :, :],
                    egocentric_rot_pred[..., :-1, :, :, :].transpose(-1, -2),
                )
                vel_geo_mask = _build_vel_frame_mask(frame_mask, num_frames)
                vel_loss = _masked_geodesic_angle(
                    rot_vel_pred, rot_vel_gt, mask=vel_geo_mask, dt=self.dt
                )
                return rot_loss + self.vel_weight * vel_loss

            return rot_loss

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class G1RootPositionLoss(nn.Module):
    """Loss on root (pelvis) position relative to first reference frame.

    Only available for decoder output formats that include root transforms
    (e.g., command_multi_future_root_transforms_nonflat).
    """

    def __init__(
        self, loss_type="mse", vel_weight=0.0, dt=0.1, normalize=False, **kwargs  # noqa: ARG002
    ):  # noqa: ARG002
        super().__init__()
        self.loss_type = loss_type
        self.vel_weight = vel_weight
        self.dt = dt
        self._normalizer = batch_normalizer.BatchNormNormalizer((3,)) if normalize else None
        self._vel_normalizer = batch_normalizer.BatchNormNormalizer((3,)) if normalize else None

    def forward(self, loss_inputs):
        tokenizer_obs = loss_inputs["tokenizer_obs"]
        decoders_cfg = loss_inputs["decoders_cfg"]
        decoded_outputs = loss_inputs["decoded_outputs"]
        frame_mask = loss_inputs.get("frame_mask", None)

        root_pos_gt, _ = decoder_output_to_root_transforms(tokenizer_obs, decoders_cfg["g1_kin"])
        root_pos_pred, _ = decoder_output_to_root_transforms(
            decoded_outputs["g1_kin"], decoders_cfg["g1_kin"]
        )

        # TCN/conv decoders may produce fewer frames; truncate gt to match pred (time dim -2)
        if root_pos_pred.shape[-2] < root_pos_gt.shape[-2]:
            root_pos_gt = root_pos_gt[..., : root_pos_pred.shape[-2], :]

        num_frames = root_pos_gt.shape[-2]
        pos_mask = _build_frame_mask_for_loss(frame_mask, num_frames)

        # Compute velocity from raw positions BEFORE normalization
        vel_gt = vel_pred = None
        if self.vel_weight > 0 and num_frames > 1:
            vel_gt = (root_pos_gt[..., 1:, :] - root_pos_gt[..., :-1, :]) / self.dt
            vel_pred = (root_pos_pred[..., 1:, :] - root_pos_pred[..., :-1, :]) / self.dt

        # Apply running normalization if enabled
        if self._normalizer is not None:
            root_pos_gt, root_pos_pred = _apply_normalizer(
                self._normalizer, root_pos_gt, root_pos_pred, pos_mask
            )

        pos_loss = _compute_masked_loss(root_pos_pred, root_pos_gt, pos_mask, self.loss_type)

        # Velocity loss (finite difference along future frames)
        if self.vel_weight > 0 and vel_gt is not None:
            vel_mask = _build_vel_frame_mask(frame_mask, num_frames)
            if self._vel_normalizer is not None:
                vel_gt, vel_pred = _apply_normalizer(
                    self._vel_normalizer, vel_gt, vel_pred, vel_mask
                )
            vel_loss = _compute_masked_loss(vel_pred, vel_gt, vel_mask, self.loss_type)
            return pos_loss + self.vel_weight * vel_loss

        return pos_loss


class G1RootRotationLoss(nn.Module):
    """Loss on root (pelvis) rotation in robot frame.

    Supports both Frobenius (on 6D representation) and geodesic (on 3x3 matrices) losses.
    """

    def __init__(
        self,
        loss_type="frobenius",
        vel_weight=0.0,
        dt=0.1,
        normalize=False,
        **kwargs,  # noqa: ARG002
    ):  # noqa: ARG002
        super().__init__()
        self.loss_type = loss_type
        self.vel_weight = vel_weight
        self.dt = dt
        # Normalization only makes sense for frobenius (element-wise); geodesic operates on SO(3)
        self._normalizer = (
            batch_normalizer.BatchNormNormalizer((6,))
            if (normalize and loss_type == "frobenius")
            else None
        )
        self._vel_normalizer = (
            batch_normalizer.BatchNormNormalizer((6,))
            if (normalize and loss_type == "frobenius")
            else None
        )

    def forward(self, loss_inputs):
        tokenizer_obs = loss_inputs["tokenizer_obs"]
        decoders_cfg = loss_inputs["decoders_cfg"]
        decoded_outputs = loss_inputs["decoded_outputs"]
        frame_mask = loss_inputs.get("frame_mask", None)

        # Get 6D rotations from decoder output
        _, root_rot_6d_gt = decoder_output_to_root_transforms(tokenizer_obs, decoders_cfg["g1_kin"])
        _, root_rot_6d_pred = decoder_output_to_root_transforms(
            decoded_outputs["g1_kin"], decoders_cfg["g1_kin"]
        )

        # TCN/conv decoders may produce fewer frames; truncate gt to match pred (time dim -2)
        if root_rot_6d_pred.shape[-2] < root_rot_6d_gt.shape[-2]:
            root_rot_6d_gt = root_rot_6d_gt[..., : root_rot_6d_pred.shape[-2], :]

        num_frames = root_rot_6d_gt.shape[-2]

        if self.loss_type == "frobenius":
            rot_mask = _build_frame_mask_for_loss(frame_mask, num_frames)

            # Compute velocity from off-diagonal elements of rotation matrix BEFORE normalization
            vel_gt = vel_pred = None
            if self.vel_weight > 0 and num_frames > 1:
                # Convert 6D -> 3x3 rotation matrix
                R_gt = rotations.rot6d_to_mat_first_two_cols(root_rot_6d_gt)  # [..., F, 3, 3]
                R_pred = rotations.rot6d_to_mat_first_two_cols(root_rot_6d_pred)  # [..., F, 3, 3]
                # Extract 6 off-diagonal elements
                idx_r = [1, 2, 0, 2, 0, 1]
                idx_c = [0, 0, 1, 1, 2, 2]
                off_gt = R_gt[..., idx_r, idx_c]  # [..., F, 6]
                off_pred = R_pred[..., idx_r, idx_c]  # [..., F, 6]
                # Finite difference velocity
                vel_gt = (off_gt[..., 1:, :] - off_gt[..., :-1, :]) / self.dt
                vel_pred = (off_pred[..., 1:, :] - off_pred[..., :-1, :]) / self.dt

            # Apply running normalization if enabled
            if self._normalizer is not None:
                root_rot_6d_gt, root_rot_6d_pred = _apply_normalizer(
                    self._normalizer, root_rot_6d_gt, root_rot_6d_pred, rot_mask
                )

            rot_loss = _compute_masked_loss(root_rot_6d_pred, root_rot_6d_gt, rot_mask, "mse")

            # Velocity loss on off-diagonal elements of rotation matrix
            if self.vel_weight > 0 and vel_gt is not None:
                vel_mask = _build_vel_frame_mask(frame_mask, num_frames)
                if self._vel_normalizer is not None:
                    vel_gt, vel_pred = _apply_normalizer(
                        self._vel_normalizer, vel_gt, vel_pred, vel_mask
                    )
                vel_loss = _compute_masked_loss(vel_pred, vel_gt, vel_mask, "mse")
                return rot_loss + self.vel_weight * vel_loss

            return rot_loss

        elif self.loss_type == "geodesic":
            # Convert 6D to 3x3 rotation matrices for geodesic loss
            root_rot_gt = rotations.rot6d_to_mat_first_two_cols(root_rot_6d_gt)
            root_rot_pred = rotations.rot6d_to_mat_first_two_cols(root_rot_6d_pred)

            geo_mask = _build_frame_mask_for_loss(frame_mask, num_frames)
            rot_loss = _masked_geodesic_angle(root_rot_pred, root_rot_gt, mask=geo_mask)

            # Angular velocity loss (relative rotation between consecutive frames)
            if self.vel_weight > 0 and root_rot_gt.shape[-3] > 1:  # [..., num_future, 3, 3]
                rot_vel_gt = torch.matmul(
                    root_rot_gt[..., 1:, :, :],
                    root_rot_gt[..., :-1, :, :].transpose(-1, -2),
                )
                rot_vel_pred = torch.matmul(
                    root_rot_pred[..., 1:, :, :],
                    root_rot_pred[..., :-1, :, :].transpose(-1, -2),
                )
                vel_geo_mask = _build_vel_frame_mask(frame_mask, num_frames)
                vel_loss = _masked_geodesic_angle(
                    rot_vel_pred, rot_vel_gt, mask=vel_geo_mask, dt=self.dt
                )
                return rot_loss + self.vel_weight * vel_loss

            return rot_loss

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class G1FootContactLoss(nn.Module):
    """Foot skating / foot contact loss.

    Penalizes foot velocity when the foot is in contact with the ground.
    Contact is determined by a height threshold: feet below the threshold
    are considered in contact and should have zero horizontal velocity.

    Note: This loss requires the new egocentric observation format
    (command_multi_future_egocentric_joint_transforms + root_transforms).
    With the old qpos-based format (command_multi_future_nonflat), root is
    identity so z-coordinates are not in world frame — contact detection
    will produce incorrect results.
    """

    def __init__(
        self,
        dt=0.05,
        foot_joint_names=("left_ankle_link", "right_ankle_link"),
        contact_height_threshold=0.05,
        vel_threshold=0.15,
        dt_warning_threshold=0.05,
        **kwargs,
    ):
        super().__init__()
        self._skeleton_name = kwargs.get("skeleton_name", "motion_g1_extended_toe")
        self.dt = dt
        self.foot_joint_names = list(foot_joint_names)
        self.contact_height_threshold = contact_height_threshold
        self.vel_threshold = vel_threshold
        self._humanoid = create_humanoid(self._skeleton_name)
        self._dof_converter = order_converter.get_converter_for_skeleton(self._skeleton_name)
        self._foot_indices = [
            self._humanoid.body_names_augment.index(name) for name in self.foot_joint_names
        ]

        if dt > dt_warning_threshold:
            import warnings

            warnings.warn(
                f"G1FootContactLoss: dt={dt:.3f}s is large (>{dt_warning_threshold}s). "
                f"Foot sliding loss may be less effective with coarse temporal resolution.",
                stacklevel=2,
            )

    def forward(self, loss_inputs):
        tokenizer_obs = loss_inputs["tokenizer_obs"]
        decoders_cfg = loss_inputs["decoders_cfg"]
        decoded_outputs = loss_inputs["decoded_outputs"]
        frame_mask = loss_inputs.get("frame_mask", None)

        device = _get_device_from_loss_inputs(loss_inputs)
        if self._humanoid.device != device:
            self._humanoid = self._humanoid.to(device=device)

        # Get joint positions in world frame for GT and predicted (include extended for toe joints)
        world_pos_gt, _ = decoder_output_to_world_transforms(
            tokenizer_obs,
            decoders_cfg["g1_kin"],
            self._humanoid,
            self._dof_converter,
            include_extended=True,
        )
        world_pos_pred, _ = decoder_output_to_world_transforms(
            decoded_outputs["g1_kin"],
            decoders_cfg["g1_kin"],
            self._humanoid,
            self._dof_converter,
            include_extended=True,
        )
        feet_pos_gt = world_pos_gt[..., self._foot_indices, :]
        feet_pos_pred = world_pos_pred[..., self._foot_indices, :]

        # TCN/conv decoders may produce fewer frames; truncate gt to match pred (time dim -3)
        if feet_pos_pred.shape[-3] < feet_pos_gt.shape[-3]:
            feet_pos_gt = feet_pos_gt[..., : feet_pos_pred.shape[-3], :, :]

        # Need at least 2 frames for velocity
        if feet_pos_pred.shape[-3] < 2:
            return torch.tensor(0.0, device=feet_pos_pred.device)

        num_frames = feet_pos_gt.shape[-3]

        # Contact mask from GT: foot height (z) below threshold AND velocity below threshold
        gt_foot_height = feet_pos_gt[..., :-1, :, 2]  # [..., num_future-1, num_feet]
        gt_foot_vel = (
            torch.norm(feet_pos_gt[..., 1:, :, :] - feet_pos_gt[..., :-1, :, :], dim=-1) / self.dt
        )
        contact_mask = (
            (gt_foot_height < self.contact_height_threshold) & (gt_foot_vel < self.vel_threshold)
        ).float()

        # Apply variable frame mask to contact_mask (velocity uses frame pairs)
        # contact_mask shape: [..., F-1, num_feet] — expand vel mask to match
        vel_frame_mask = _build_vel_frame_mask(frame_mask, num_frames)
        if vel_frame_mask is not None:
            contact_mask = contact_mask * vel_frame_mask.unsqueeze(-1)

        # Predicted foot velocity: ||pos[t+1] - pos[t]|| / dt
        pred_foot_vel = (
            torch.norm(feet_pos_pred[..., 1:, :, :] - feet_pos_pred[..., :-1, :, :], dim=-1)
            / self.dt
        )

        # Penalize predicted velocity when GT says foot is in contact
        vel_err = pred_foot_vel * contact_mask

        # Mean over contacting frames (avoid division by zero)
        total_contact = contact_mask.sum() + 1e-6
        skating_loss = vel_err.sum() / total_contact

        return skating_loss
