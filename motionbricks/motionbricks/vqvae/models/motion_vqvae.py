import numpy as np
import torch as t
import torch
import logging
from typing import Callable, Optional, Union, Dict
from pytorch_lightning import LightningModule
from motionbricks.motionlib.core.motion_reps import MotionRepBase
from motionbricks.motionlib.core.motion_reps.dual_root_global_joints import (
    GlobalRootGlobalJoints,
    LocalRootGlobalJoints,
)
from motionbricks.helper.data_training_util import extract_feature_from_motion_rep
from motionbricks.helper.data_training_util import (
    sample_motion_segments_from_motion_clips,
)
from motionbricks.helper.data_training_util import sample_keyframes
from motionbricks.vqvae.neural_modules import vqvae as vqvae_module

log = logging.getLogger(__name__)


class MotionVQVAEModel(LightningModule):
    """VQVAE model for motion generation, wrapped in PyTorch Lightning.

    The pose VQVAE encodes local/global motion representations into discrete
    codes and reconstructs them with optional keyframe conditioning and
    external root-motion conditioning.
    """

    def __init__(
        self,
        pose_vqvae_network: vqvae_module.VQVAE,
        root_vqvae_network: Optional[vqvae_module.VQVAE],
        motion_rep: MotionRepBase,
        optimizer: Callable[[list], torch.optim.Optimizer],
        scheduler: Optional[
            Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
        ],
        device: Optional[Union[str, torch.device]] = None,
        args: Dict = None,
        **kwargs,
    ):
        super().__init__()

        self.optimizer = optimizer
        self.scheduler = scheduler

        assert (
            motion_rep.dual_rep.local_motion_rep.num_joints
            == motion_rep.dual_rep.global_motion_rep.num_joints
        ), "The number of joints should be the same."
        self.NUM_JOINTS = motion_rep.num_joints

        self.pose_net = pose_vqvae_network
        self.root_net = root_vqvae_network
        self.motion_rep: GlobalRootGlobalJoints = motion_rep
        self.global_motion_rep: GlobalRootGlobalJoints = (
            motion_rep.dual_rep.global_motion_rep
        )
        self.local_motion_rep: LocalRootGlobalJoints = (
            motion_rep.dual_rep.local_motion_rep
        )
        self._args = args

        if device is not None:
            self.pose_net = self.pose_net.to(device)
            self.root_net = (
                self.root_net.to(device) if self.root_net is not None else None
            )

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters())
        if not self.scheduler:
            return optimizer

        lt_kwargs = dict(self.scheduler.keywords.pop("lt_kwargs", {}))
        lt_kwargs["scheduler"] = self.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lt_kwargs}

    def training_step(self, batch, batch_idx):
        batch_size, device = batch["batch_size"], batch["motion"].device
        motions, motion_lengths, _ = (
            batch.pop("motion"),
            batch.pop("motion_len"),
            batch.pop("motion_pad_mask"),
        )

        num_codes = np.random.choice(
            np.arange(self._args["min_tokens"], self._args["max_tokens"] + 1)
        )
        num_frames = num_codes * self.get_num_frames_per_code()

        # filter out short samples and truncate
        valid_samples_id = motion_lengths >= num_frames + 1
        num_invalid_samples = batch_size - valid_samples_id.sum()
        if num_invalid_samples > batch_size // 2:
            return None

        motions = sample_motion_segments_from_motion_clips(
            motions,
            motion_lengths,
            num_frames,
            self.args["batchsize_mul_factor"],
            motion_rep=self.global_motion_rep,
        )
        actual_batch_size = int(batch_size * self.args["batchsize_mul_factor"])

        # prepare global & local motion representations
        first_frame_heading_angle = (
            t.rand(actual_batch_size).to(device) * np.pi * 2.0
            if not self.motion_rep.compute_kwargs["removing_heading"]
            else 0.0
        )
        global_motions = self.global_motion_rep.change_first_heading(
            motions,
            first_frame_heading_angle,
            is_normalized=True,
            to_normalize=True,
        )
        local_motions = self.motion_rep.dual_rep.global_to_local(
            global_motions,
            is_normalized=True,
            to_normalize=True,
            lengths=t.full([actual_batch_size], motions.shape[1]).to(device),
        )
        local_motions, global_motions = (
            local_motions[:, :num_frames, :],
            global_motions[:, :num_frames, :],
        )

        # keyframe conditioning
        assert self.pose_net.motion_rep.name in [
            "local",
            "global",
        ], "should be using local or global rep."
        prob_pose_num_keyframes, _ = self._construct_keyframe_prob()
        pose_motions = (
            local_motions
            if self.pose_net.motion_rep.name == "local"
            else global_motions
        )
        pose_has_target_cond, pose_target_cond = sample_keyframes(
            pose_motions,
            self._args["pose_vqvae_max_num_keyframes"],
            prob_pose_num_keyframes,
        )
        pose_external_cond = extract_feature_from_motion_rep(
            pose_motions,
            self.pose_net.motion_rep,
            self.pose_net.decoder_external_cond_feature_mode,
        )

        batch["local_motions"], batch["global_motions"] = local_motions, global_motions
        batch["pose_has_target_cond"] = pose_has_target_cond
        pose_external_cond = self._construct_masked_pose_external_cond_if_needed(
            pose_external_cond, batch
        )

        # network forward
        if self.root_net is not None:
            raise NotImplementedError("Root VQVAE is not implemented.")
        else:
            root_net_output = None

        pose_net_output = self.pose_net(
            batch["local_motions"]
            if self.pose_net.motion_rep.name == "local"
            else batch["global_motions"],
            target_cond=pose_target_cond,
            has_target_cond=pose_has_target_cond,
            external_cond=pose_external_cond,
        )
        losses = self.loss(batch, pose_net_output, root_net_output)

        for key, val in losses.items():
            self.log(
                f"loss/train_{key}",
                val,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch["batch_size"],
            )
        return losses["loss"]

    def loss(self, batch, pose_net_output, root_net_output):
        if self.root_net is not None:
            raise NotImplementedError("Root VQVAE is not implemented.")
        else:
            global_root_recons_loss = local_root_recons_loss = 0.0

        # pose reconstruction loss
        if self.pose_net.motion_rep.name == "local":
            local_pose_recons_loss = t.nn.SmoothL1Loss()(
                pose_net_output["recon_state"], batch["local_motions"][:, :, :]
            )
            global_pose_recons_loss = 0.0
            pose_recons_loss = local_pose_recons_loss
        else:
            global_pose_recons_loss = t.nn.SmoothL1Loss()(
                pose_net_output["recon_state"], batch["global_motions"][:, :, :]
            )
            pred_local_motions = self.motion_rep.dual_rep.global_to_local(
                pose_net_output["recon_state"],
                is_normalized=True,
                to_normalize=True,
                lengths=t.full(
                    [pose_net_output["recon_state"].shape[0]],
                    pose_net_output["recon_state"].shape[1],
                ).to(pose_net_output["recon_state"].device),
            )
            pred_local_motions = t.concat(
                [
                    pred_local_motions[:, :-1, :],
                    t.concat(
                        [
                            batch["local_motions"][
                                :, -1:, self.local_motion_rep.indices["root"]
                            ],
                            pred_local_motions[
                                :, -1:, self.local_motion_rep.indices["body"]
                            ],
                        ],
                        dim=-1,
                    ),
                ],
                dim=1,
            )
            local_pose_recons_loss = t.nn.SmoothL1Loss()(
                pred_local_motions, batch["local_motions"]
            )
            pose_recons_loss = (
                self.args["global_root_loss_coeff"] * global_pose_recons_loss
                + self.args["local_root_loss_coeff"] * local_pose_recons_loss
            )

            global_root_recons_loss = t.nn.SmoothL1Loss()(
                pose_net_output["recon_state"][
                    :, :, self.global_motion_rep.indices["root"]
                ],
                batch["global_motions"][:, :, self.global_motion_rep.indices["root"]],
            )
            local_root_recons_loss = t.nn.SmoothL1Loss()(
                pred_local_motions[:, :, self.local_motion_rep.indices["root"]],
                batch["local_motions"][:, :, self.local_motion_rep.indices["root"]],
            )

        # foot contact loss
        if self.pose_net.motion_rep.name == "local":
            pred_joints_output = self.local_motion_rep.inverse(
                pose_net_output["recon_state"],
                is_normalized=True,
                return_quat=True,
                return_all=True,
            )
        else:
            pred_joints_output = self.global_motion_rep.inverse(
                pose_net_output["recon_state"],
                is_normalized=True,
                return_quat=True,
                return_all=True,
            )

        pred_joints_pos = pred_joints_output["posed_joints"]
        pred_foot_contacts = pred_joints_output.get("foot_contacts")

        fidx = self.motion_rep.skeleton.foot_joint_idx
        feet_pos = pred_joints_pos[:, :, fidx]
        dt = 1.0 / self.motion_rep.fps
        foot_vel = torch.norm(feet_pos[:, 1:] - feet_pos[:, :-1], dim=-1) / dt

        foot_contacts = pred_foot_contacts[:, :-1]
        vel_err = foot_vel * foot_contacts
        mean_vel = torch.sum(vel_err, (1, 2)) / (
            torch.sum(foot_contacts, (1, 2)) + 1e-6
        )
        mean_vel = mean_vel.mean()

        # joint velocity loss
        joint_vel_loss_coeff = self._args.get("joint_vel_loss_coeff", 0.0)
        if joint_vel_loss_coeff > 0.0:
            batch_size, num_frames = (
                pred_joints_pos.shape[0],
                pred_joints_pos.shape[1],
            )
            pred_joints_vel = (
                (pred_joints_pos[:, 1:] - pred_joints_pos[:, :-1]) / dt
            ).view([batch_size, num_frames - 1, -1])
            gt_joints_pos = self.global_motion_rep.inverse(
                batch["global_motions"],
                is_normalized=True,
                return_quat=False,
                return_all=False,
                joint_positions_from="ric_data",
            )["posed_joints"]
            gt_joints_vel = (
                (gt_joints_pos[:, 1:] - gt_joints_pos[:, :-1]) / dt
            ).view([batch_size, num_frames - 1, -1])
            joint_vel_indices = self.global_motion_rep.indices["local_vel"]
            vel_mean = self.global_motion_rep.stats.mean[joint_vel_indices]
            vel_std = self.global_motion_rep.stats.std[joint_vel_indices]

            joint_vel_loss = t.nn.SmoothL1Loss()(
                (pred_joints_vel - vel_mean) / torch.sqrt(vel_std**2 + 1e-5),
                (gt_joints_vel - vel_mean) / torch.sqrt(vel_std**2 + 1e-5),
            )
            joint_vel_loss = (
                joint_vel_loss
                / len(self.local_motion_rep.indices["all"])
                * len(joint_vel_indices)
            )
        else:
            joint_vel_loss = 0.0

        losses = {}
        losses["perplexity_pose"] = pose_net_output["perplexity"]
        losses["l_commit_pose"] = pose_net_output["l_commit"]
        losses["l_recons_pose"] = pose_recons_loss
        losses["l_recons_root_global"] = global_root_recons_loss
        losses["l_recons_root_local"] = local_root_recons_loss
        losses["l_joint_vel"] = joint_vel_loss
        losses["l_recons_pose_global"] = global_pose_recons_loss
        losses["l_recons_pose_local"] = local_pose_recons_loss
        losses["l_skate_contact"] = mean_vel
        skate_contact_loss_coeff = self._args.get("skate_contact_loss_coeff", 0.0)

        losses["loss"] = (
            pose_recons_loss
            + self._args["commit_loss_coeff"] * losses["l_commit_pose"]
            + skate_contact_loss_coeff * losses["l_skate_contact"]
            + joint_vel_loss_coeff * losses["l_joint_vel"]
        )
        return losses

    @property
    def args(self):
        return self._args

    def get_num_frames_per_code(self):
        return 2 ** self._args["down_t"]

    def _construct_keyframe_prob(self):
        probs = dict()
        for module in ["pose", "root"]:
            max_num_keyframes = self.args[f"{module}_vqvae_max_num_keyframes"] * (
                self.trainer.global_step / self.args["keyframe_num_warmup_steps"]
            )
            max_num_keyframes = int(
                max(
                    1,
                    min(
                        max_num_keyframes,
                        self.args[f"{module}_vqvae_max_num_keyframes"],
                    ),
                )
            )

            prob_num_keyframes = [
                1.0 if i > 0 and i <= max_num_keyframes else 0.0
                for i in range(self.args[f"{module}_vqvae_max_num_keyframes"] + 1)
            ]
            prob_no_keyframe = self.args[f"{module}_vqvae_no_keyframe_prob"]
            prob_num_keyframes[0] = (
                sum(prob_num_keyframes) / (1 - prob_no_keyframe) * prob_no_keyframe
            )
            prob_num_keyframes = np.array(prob_num_keyframes)
            prob_num_keyframes /= prob_num_keyframes.sum()
            probs[module] = prob_num_keyframes

        return probs["pose"], probs["root"]

    def _construct_masked_pose_external_cond_if_needed(
        self, pose_external_cond: t.Tensor, batch: dict
    ):
        if self.pose_net.motion_rep.name != "global":
            return pose_external_cond
        if (
            self.pose_net.decoder_external_cond_feature_mode
            != "root_without_hip_height_without_heading_with_mask"
        ):
            return pose_external_cond

        batch_size, motion_length = (
            batch["local_motions"].shape[0],
            batch["local_motions"].shape[1],
        )
        device = batch["local_motions"].device
        unnorm_gt_local_motion = self.local_motion_rep.unnormalize(
            batch["local_motions"]
        )

        max_perturb_angle = self.args.get("max_perturb_angle", 5.0)
        max_vel_norm_perturb_ratio = self.args.get("max_vel_norm_perturb_ratio", 0.2)
        norm_ratio = (
            torch.rand([batch_size, unnorm_gt_local_motion.shape[1], 1])
            * max_vel_norm_perturb_ratio
            * 2
            + (1 - max_vel_norm_perturb_ratio)
        )
        angle = (
            torch.rand([batch_size, unnorm_gt_local_motion.shape[1]])
            * max_perturb_angle
            * 2
            - max_perturb_angle
        ) * np.pi / 180.0
        norm_ratio, angle = norm_ratio.to(device), angle.to(device)

        accumulated_angle = torch.cumsum(angle, dim=1)
        unnorm_gt_local_motion[
            :, :, self.local_motion_rep.indices["local_root_vel"]
        ] *= norm_ratio[:, :, :]
        cos, sin = torch.cos(accumulated_angle), torch.sin(accumulated_angle)
        rotated_x = (
            unnorm_gt_local_motion[
                :, :, self.local_motion_rep.indices["local_root_vel"][0]
            ]
            * cos
            - unnorm_gt_local_motion[
                :, :, self.local_motion_rep.indices["local_root_vel"][1]
            ]
            * sin
        )
        rotated_z = (
            unnorm_gt_local_motion[
                :, :, self.local_motion_rep.indices["local_root_vel"][0]
            ]
            * sin
            + unnorm_gt_local_motion[
                :, :, self.local_motion_rep.indices["local_root_vel"][1]
            ]
            * cos
        )
        unnorm_gt_local_motion[
            :, :, self.local_motion_rep.indices["local_root_vel"][0]
        ] = rotated_x
        unnorm_gt_local_motion[
            :, :, self.local_motion_rep.indices["local_root_vel"][1]
        ] = rotated_z

        perturbed_pose_external_cond = extract_feature_from_motion_rep(
            self.motion_rep.dual_rep.local_to_global(
                unnorm_gt_local_motion,
                is_normalized=False,
                to_normalize=True,
                lengths=torch.full([batch_size], motion_length).to(device),
            ),
            self.pose_net.motion_rep,
            self.pose_net.decoder_external_cond_feature_mode,
        )

        perturbed = torch.rand([batch_size]) < self.args.get(
            "percentage_of_perturbed_samples", 0.2
        )
        perturbed_pose_external_cond = t.where(
            batch["pose_has_target_cond"][:, :, None],
            pose_external_cond,
            perturbed_pose_external_cond,
        )
        pose_external_cond = t.where(
            perturbed[:, None, None].to(device),
            perturbed_pose_external_cond,
            pose_external_cond,
        )
        pose_external_cond[:, :, -1] = batch["pose_has_target_cond"].float()

        batch["perturbed"] = perturbed
        return pose_external_cond
