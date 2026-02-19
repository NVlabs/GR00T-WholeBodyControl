"""
Visualization utilities for VPlanner.

Creates prediction plots for training visualization and WandB logging.
"""

import io
import numpy as np
import torch
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

from groot.rl.isaac_utils.rotations import quat_rotate
from groot.rl.trl.utils.fk_utils import FKHelper

# Lazy import cv2 - only used in SkeletonVisualizer for real-time display
cv2 = None


def _get_cv2():
    global cv2
    if cv2 is None:
        import cv2 as _cv2

        cv2 = _cv2
    return cv2


class SkeletonVisualizer:
    """
    Real-time skeleton visualizer for VPlanner evaluation.

    Renders a root-centric 3D skeleton from DOF predictions using matplotlib,
    then converts to OpenCV image for display.
    """

    def __init__(self, motion_lib, img_size: int = 400):
        """
        Initialize skeleton visualizer.

        Args:
            motion_lib: MotionLibRobot instance for FK
            img_size: Output image size in pixels
        """
        self.fk_helper = FKHelper(motion_lib)
        self.img_size = img_size
        self.device = motion_lib.mesh_parsers.dof_axis.device

        # Quaternion for root-centric rendering
        # The skeleton rest pose may be in Y-up, so rotate -90 deg around X to make Z-up
        # Rotation of -90 deg around X axis: quat = [cos(-45°), sin(-45°), 0, 0] in wxyz
        import math

        angle = -math.pi / 2  # -90 degrees
        self.upright_quat = torch.tensor(
            [math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0],  # w  # x  # y  # z
            device=self.device,
            dtype=torch.float32,
        )

        # Create persistent figure for faster rendering
        self.fig = plt.figure(figsize=(4, 4), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")

        logger.info(f"SkeletonVisualizer initialized: {img_size}x{img_size}")

    def render(
        self,
        dof_pos: torch.Tensor,
        title: str = "Predicted Pose",
    ) -> np.ndarray:
        """
        Render a single DOF pose as a root-centric skeleton.

        Args:
            dof_pos: [29] DOF positions (single frame)
            title: Title to display on the image

        Returns:
            OpenCV BGR image [img_size, img_size, 3]
        """
        # Ensure batch dimension
        if dof_pos.dim() == 1:
            dof_pos = dof_pos.unsqueeze(0)  # [1, 29]

        dof_pos = dof_pos.to(self.device)

        # Root at origin, identity rotation for FK
        root_pos = torch.zeros(1, 3, device=self.device)
        root_rot6d = torch.tensor([[1, 0, 0, 0, 1, 0]], device=self.device, dtype=torch.float)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # Compute body positions via FK
        try:
            body_pos = self.fk_helper.dof_to_body_pos(
                dof_pos, root_pos, root_rot6d, identity_quat
            )  # [1, num_keypoints, 3]
            body_pos = body_pos[0].cpu().numpy()  # [num_keypoints, 3]

            # Rotate so pelvis-to-torso direction becomes +Z (upward)
            # Pelvis is index 0, torso is index 7
            pelvis = body_pos[0]
            torso = body_pos[7]
            up_vec = torso - pelvis

            # Find which axis has the largest component in up_vec - that's the current "up"
            up_axis = np.argmax(np.abs(up_vec))

            # Swap axes so that axis becomes Z
            if up_axis == 0:  # X is up -> swap X and Z
                body_pos = body_pos[:, [2, 1, 0]]  # XYZ -> ZYX
                if up_vec[0] < 0:  # pointing in -X, flip Z
                    body_pos[:, 2] = -body_pos[:, 2]
            elif up_axis == 1:  # Y is up -> swap Y and Z
                body_pos = body_pos[:, [0, 2, 1]]  # XYZ -> XZY
                if up_vec[1] < 0:  # pointing in -Y, flip Z
                    body_pos[:, 2] = -body_pos[:, 2]
            # else: Z is already up, check sign
            elif up_vec[2] < 0:  # Z is up but pointing down
                body_pos[:, 2] = -body_pos[:, 2]
        except Exception as e:
            logger.warning(f"FK failed: {e}")
            # Return blank image on failure
            blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            _get_cv2().putText(
                blank,
                "FK Failed",
                (10, self.img_size // 2),
                _get_cv2().FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            return blank

        # Clear and redraw
        self.ax.clear()

        # Plot directly - upright_quat already rotated skeleton to Z-up
        plot_x = body_pos[:, 0]
        plot_y = body_pos[:, 1]
        plot_z = body_pos[:, 2]

        # Draw skeleton bones
        for start, end in FKHelper.SKELETON_BONES:
            self.ax.plot(
                [plot_x[start], plot_x[end]],
                [plot_y[start], plot_y[end]],
                [plot_z[start], plot_z[end]],
                c="cyan",
                linewidth=2,
            )

        # Draw joints
        regular_mask = np.ones(len(body_pos), dtype=bool)
        regular_mask[FKHelper.FOOT_INDICES + FKHelper.HAND_INDICES] = False

        # Regular joints
        self.ax.scatter(
            plot_x[regular_mask], plot_y[regular_mask], plot_z[regular_mask], c="white", s=30
        )

        # Feet (orange)
        self.ax.scatter(
            plot_x[FKHelper.FOOT_INDICES],
            plot_y[FKHelper.FOOT_INDICES],
            plot_z[FKHelper.FOOT_INDICES],
            c="orange",
            s=50,
            marker="^",
        )

        # Hands (purple)
        self.ax.scatter(
            plot_x[FKHelper.HAND_INDICES],
            plot_y[FKHelper.HAND_INDICES],
            plot_z[FKHelper.HAND_INDICES],
            c="magenta",
            s=50,
            marker="o",
        )

        # Auto-scale axes based on data
        origin = body_pos.mean(axis=0)
        radius = max(0.5 * (body_pos.max(axis=0) - body_pos.min(axis=0)).max(), 0.5)

        self.ax.set_xlim([origin[0] - radius, origin[0] + radius])
        self.ax.set_ylim([origin[1] - radius, origin[1] + radius])
        self.ax.set_zlim([origin[2] - radius, origin[2] + radius])

        # Set view angle: looking from front-right, slightly above
        self.ax.view_init(elev=20, azim=-135)

        # Style
        self.ax.set_facecolor((0.1, 0.1, 0.1))
        self.ax.set_xlabel("X", color="gray", fontsize=8)
        self.ax.set_ylabel("Y", color="gray", fontsize=8)
        self.ax.set_zlabel("Z (up)", color="gray", fontsize=8)
        self.ax.set_title(title, color="white", fontsize=10)
        try:
            self.ax.set_box_aspect([1, 1, 1])
        except AttributeError:
            pass  # Older matplotlib
        self.ax.tick_params(colors="gray")

        # Convert figure to OpenCV image
        self.fig.tight_layout()
        self.fig.canvas.draw()

        # Get RGBA buffer (compatible with newer matplotlib)
        w, h = self.fig.canvas.get_width_height()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        img = buf[:, :, :3]  # Drop alpha channel, keep RGB

        # Resize to target size
        img = _get_cv2().resize(img, (self.img_size, self.img_size))

        # Convert RGB to BGR for OpenCV
        img = _get_cv2().cvtColor(img, _get_cv2().COLOR_RGB2BGR)

        return img

    def close(self):
        """Close the matplotlib figure."""
        plt.close(self.fig)


class VPlannerVisualizer:
    """
    Visualization utilities for VPlanner predictions.

    Creates multi-panel figures showing:
    - Input images
    - BEV trajectory with heading arrows
    - 3D skeleton trajectories (GT and Pred)
    - Comparison plots
    """

    def __init__(self, fk_helper: FKHelper):
        """
        Initialize visualizer.

        Args:
            fk_helper: FKHelper instance for forward kinematics
        """
        self.fk_helper = fk_helper

    def create_prediction_plots(
        self,
        batch: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        num_samples: int = 4,
    ) -> plt.Figure:
        """
        Create a figure with prediction visualizations.

        Shows:
        - Input image
        - Top-down trajectory with heading arrows
        - 3D skeleton trajectory (GT)
        - 3D skeleton trajectory (Pred)
        - 3D comparison (GT + Pred together)

        Args:
            batch: Input batch with images and metadata
            predictions: Model predictions dict
            labels: Ground truth labels dict
            num_samples: Number of samples to visualize

        Returns:
            matplotlib Figure
        """
        num_cols = 5
        fig = plt.figure(figsize=(5 * num_cols, 5 * num_samples))

        for i in range(num_samples):
            self._plot_sample(fig, i, num_samples, num_cols, batch, predictions, labels)

        plt.tight_layout()
        return fig

    def _plot_sample(
        self,
        fig: plt.Figure,
        i: int,
        num_samples: int,
        num_cols: int,
        batch: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ):
        """Plot visualizations for a single sample."""
        # Get data for this sample
        image = batch["image"][i]
        seq_name = batch["seq_name"][i]
        frame_idx = batch["frame_idx"][i].item()
        bev_bounds = batch["bev_bounds"][i]

        # Get predictions and labels
        pred_pos = predictions["future_root_pos"][i]
        gt_pos = labels["future_root_pos"][i]
        pred_rot6d = predictions["future_root_rot6d"][i]
        gt_rot6d = labels["future_root_rot6d"][i]
        pred_dof = predictions["future_dof_pos"][i]
        gt_dof = labels["future_dof_pos"][i]

        # Current frame reference
        current_root_pos = labels["current_root_pos"][i]
        current_root_quat = labels["current_root_quat"][i]

        # Transform root positions to world frame
        gt_pos_world = (
            self.fk_helper.transform_to_world(gt_pos, current_root_quat) + current_root_pos
        )
        pred_pos_world = (
            self.fk_helper.transform_to_world(pred_pos, current_root_quat) + current_root_pos
        )

        # Compute body positions via FK (same path for GT and Pred)
        try:
            gt_body_world = self.fk_helper.dof_to_body_pos(
                gt_dof, gt_pos, gt_rot6d, current_root_quat
            )
            pred_body_world = self.fk_helper.dof_to_body_pos(
                pred_dof, pred_pos, pred_rot6d, current_root_quat
            )
            gt_body_pos = (gt_body_world + current_root_pos).cpu().numpy()
            pred_body_pos = (pred_body_world + current_root_pos).cpu().numpy()
            fk_success = True
        except Exception as e:
            logger.warning(f"FK failed for sample {i}: {e}")
            fk_success = False

        # Convert to numpy
        gt_pos_np = gt_pos_world.cpu().numpy()
        pred_pos_np = pred_pos_world.cpu().numpy()
        gt_rot_np = gt_rot6d.cpu().numpy()
        pred_rot_np = pred_rot6d.cpu().numpy()
        current_pos_np = current_root_pos.cpu().numpy()
        num_future = len(gt_pos_np)

        # --- Plot 1: Input image ---
        ax = fig.add_subplot(num_samples, num_cols, i * num_cols + 1)
        self._plot_image(ax, image, seq_name, frame_idx)

        # --- Plot 2: Top-down trajectory ---
        ax = fig.add_subplot(num_samples, num_cols, i * num_cols + 2)
        self._plot_bev_trajectory(
            ax,
            gt_pos_np,
            pred_pos_np,
            gt_rot_np,
            pred_rot_np,
            current_root_quat,
            current_pos_np,
            num_future,
            bev_bounds,
        )

        # --- Plot 3: 3D skeleton GT ---
        ax = fig.add_subplot(num_samples, num_cols, i * num_cols + 3, projection="3d")
        if fk_success:
            self._plot_skeleton_3d(ax, gt_body_pos, color="green", title="GT")
        else:
            ax.set_title("FK failed", fontsize=8)

        # --- Plot 4: 3D skeleton Pred ---
        ax = fig.add_subplot(num_samples, num_cols, i * num_cols + 4, projection="3d")
        if fk_success:
            self._plot_skeleton_3d(ax, pred_body_pos, color="red", title="Pred")
        else:
            ax.set_title("FK failed", fontsize=8)

        # --- Plot 5: 3D comparison ---
        ax = fig.add_subplot(num_samples, num_cols, i * num_cols + 5, projection="3d")
        if fk_success:
            self._plot_skeleton_comparison_3d(ax, gt_body_pos, pred_body_pos)
        else:
            ax.set_title("FK failed", fontsize=8)

    def _plot_image(self, ax, image: torch.Tensor, seq_name: str, frame_idx: int):
        """Plot input image (oldest history on top, current on bottom)."""
        # Denormalize from ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        if image.dim() == 4:  # [T, C, H, W] - multiple frames
            if image.shape[0] == 0:
                # No images - show placeholder
                ax.text(
                    0.5,
                    0.5,
                    "No images\n(num_history_frames_img=0)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                oldest = (image[0].cpu() * std + mean).permute(1, 2, 0).numpy()
                current = (image[-1].cpu() * std + mean).permute(1, 2, 0).numpy()
                # Concatenate vertically: oldest on top, current on bottom
                img = np.concatenate([oldest, current], axis=0)
                ax.imshow(np.clip(img, 0, 1))
        else:  # [C, H, W] - single frame
            img = (image.cpu() * std + mean).permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"{seq_name}\nframe {frame_idx}", fontsize=8)
        ax.axis("off")

    def _plot_bev_trajectory(
        self,
        ax,
        gt_pos: np.ndarray,
        pred_pos: np.ndarray,
        gt_rot: np.ndarray,
        pred_rot: np.ndarray,
        current_root_quat: torch.Tensor,
        current_pos: np.ndarray,
        num_future: int,
        bev_bounds: Dict[str, float],
    ):
        """Plot top-down trajectory with heading arrows."""
        # Plot trajectories
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], "g-", linewidth=2, label="GT", alpha=0.7)
        ax.plot(pred_pos[:, 0], pred_pos[:, 1], "r--", linewidth=2, label="Pred", alpha=0.7)
        ax.scatter(
            [current_pos[0]],
            [current_pos[1]],
            c="blue",
            s=100,
            marker="s",
            zorder=5,
            label="Current",
        )

        # Heading arrows
        gt_fwd_world = (
            quat_rotate(
                current_root_quat.unsqueeze(0).expand(num_future, -1),
                torch.tensor(gt_rot[:, :3], device=current_root_quat.device, dtype=torch.float),
                w_last=False,
            )
            .cpu()
            .numpy()
        )
        pred_fwd_world = (
            quat_rotate(
                current_root_quat.unsqueeze(0).expand(num_future, -1),
                torch.tensor(pred_rot[:, :3], device=current_root_quat.device, dtype=torch.float),
                w_last=False,
            )
            .cpu()
            .numpy()
        )

        # Axis limits from full motion bounds
        x_center = (bev_bounds["x_min"] + bev_bounds["x_max"]) / 2
        y_center = (bev_bounds["y_min"] + bev_bounds["y_max"]) / 2
        extent = (
            max(
                bev_bounds["x_max"] - bev_bounds["x_min"],
                bev_bounds["y_max"] - bev_bounds["y_min"],
                0.5,
            )
            * 1.1
        )
        arrow_len = extent * 0.02

        for t in range(num_future):
            alpha = 1.0 - 0.7 * (t / max(num_future - 1, 1))

            # GT arrow
            gt_fwd = gt_fwd_world[t, :2]
            if np.linalg.norm(gt_fwd) > 0.1:
                gt_fwd = gt_fwd / np.linalg.norm(gt_fwd)
                ax.arrow(
                    gt_pos[t, 0],
                    gt_pos[t, 1],
                    gt_fwd[0] * arrow_len,
                    gt_fwd[1] * arrow_len,
                    head_width=arrow_len * 0.4,
                    head_length=arrow_len * 0.3,
                    fc="green",
                    ec="green",
                    alpha=alpha,
                    zorder=4,
                )

            # Pred arrow
            pred_fwd = pred_fwd_world[t, :2]
            if np.linalg.norm(pred_fwd) > 0.1:
                pred_fwd = pred_fwd / np.linalg.norm(pred_fwd)
                ax.arrow(
                    pred_pos[t, 0],
                    pred_pos[t, 1],
                    pred_fwd[0] * arrow_len,
                    pred_fwd[1] * arrow_len,
                    head_width=arrow_len * 0.4,
                    head_length=arrow_len * 0.3,
                    fc="red",
                    ec="red",
                    alpha=alpha,
                    zorder=4,
                )

        ax.set_xlim(x_center - extent / 2, x_center + extent / 2)
        ax.set_ylim(y_center + extent / 2, y_center - extent / 2)  # Flipped: larger Y at bottom
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("BEV", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    def _plot_skeleton_3d(self, ax, body_pos_seq: np.ndarray, color: str, title: str):
        """Plot 3D skeleton trajectory."""
        num_frames = len(body_pos_seq)
        frames_to_show = list(range(0, num_frames, 5))
        if (num_frames - 1) not in frames_to_show:
            frames_to_show.append(num_frames - 1)

        for t in frames_to_show:
            body_pos = body_pos_seq[t]
            alpha = 1.0 - 0.7 * (t / max(num_frames - 1, 1))

            # Joints
            regular_mask = np.ones(len(body_pos), dtype=bool)
            regular_mask[FKHelper.FOOT_INDICES + FKHelper.HAND_INDICES] = False
            ax.scatter(
                body_pos[regular_mask, 0],
                body_pos[regular_mask, 1],
                body_pos[regular_mask, 2],
                c=color,
                s=15,
                alpha=alpha,
            )
            ax.scatter(
                body_pos[FKHelper.FOOT_INDICES, 0],
                body_pos[FKHelper.FOOT_INDICES, 1],
                body_pos[FKHelper.FOOT_INDICES, 2],
                c="orange",
                s=25,
                alpha=alpha,
                marker="^",
            )
            ax.scatter(
                body_pos[FKHelper.HAND_INDICES, 0],
                body_pos[FKHelper.HAND_INDICES, 1],
                body_pos[FKHelper.HAND_INDICES, 2],
                c="purple",
                s=25,
                alpha=alpha,
                marker="o",
            )

            # Bones
            for start, end in FKHelper.SKELETON_BONES:
                ax.plot(
                    [body_pos[start, 0], body_pos[end, 0]],
                    [body_pos[start, 1], body_pos[end, 1]],
                    [body_pos[start, 2], body_pos[end, 2]],
                    c=color,
                    linewidth=1.5,
                    alpha=alpha,
                )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title, fontsize=8)
        self._set_3d_axes_equal(ax, body_pos_seq.reshape(-1, 3))

    def _plot_skeleton_comparison_3d(self, ax, gt_body_pos: np.ndarray, pred_body_pos: np.ndarray):
        """Plot GT and Pred skeletons together."""
        num_frames = len(gt_body_pos)
        frames_to_show = list(range(0, num_frames, 5))
        if (num_frames - 1) not in frames_to_show:
            frames_to_show.append(num_frames - 1)

        for t in frames_to_show:
            alpha = 1.0 - 0.7 * (t / max(num_frames - 1, 1))

            for pos, color, style in [
                (gt_body_pos[t], "green", "-"),
                (pred_body_pos[t], "red", "--"),
            ]:
                for start, end in FKHelper.SKELETON_BONES:
                    ax.plot(
                        [pos[start, 0], pos[end, 0]],
                        [pos[start, 1], pos[end, 1]],
                        [pos[start, 2], pos[end, 2]],
                        c=color,
                        linewidth=1.5,
                        alpha=alpha,
                        linestyle=style,
                    )

            # Markers on GT only
            ax.scatter(
                gt_body_pos[t][FKHelper.FOOT_INDICES, 0],
                gt_body_pos[t][FKHelper.FOOT_INDICES, 1],
                gt_body_pos[t][FKHelper.FOOT_INDICES, 2],
                c="orange",
                s=20,
                alpha=alpha,
                marker="^",
            )
            ax.scatter(
                gt_body_pos[t][FKHelper.HAND_INDICES, 0],
                gt_body_pos[t][FKHelper.HAND_INDICES, 1],
                gt_body_pos[t][FKHelper.HAND_INDICES, 2],
                c="purple",
                s=20,
                alpha=alpha,
                marker="o",
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("GT (green) vs Pred (red)", fontsize=8)
        all_pos = np.concatenate([gt_body_pos.reshape(-1, 3), pred_body_pos.reshape(-1, 3)])
        self._set_3d_axes_equal(ax, all_pos)

    def _set_3d_axes_equal(self, ax, positions: np.ndarray):
        """Set 3D axes to equal aspect ratio with Y-axis flipped (larger Y at bottom)."""
        origin = positions.mean(axis=0)
        radius = 0.5 * max(positions.max(axis=0) - positions.min(axis=0))
        radius = max(radius, 0.2)
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] + radius, origin[1] - radius])  # Flipped: larger Y at bottom
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
        ax.set_box_aspect([1, 1, 1])

    def create_terminal_prediction_plots(
        self,
        batch: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
        num_samples: int = 4,
    ) -> plt.Figure:
        """
        Create a figure with terminal pose prediction visualizations.

        Shows:
        - Input image
        - BEV with current position and terminal position (GT + Pred) with arrows
        - 3D skeleton comparison (GT + Pred terminal pose)

        Args:
            batch: Input batch with images and metadata
            predictions: Model predictions dict (terminal_*)
            labels: Ground truth labels dict
            num_samples: Number of samples to visualize

        Returns:
            matplotlib Figure
        """
        num_cols = 3  # Image, BEV, 3D skeleton
        fig = plt.figure(figsize=(5 * num_cols, 5 * num_samples))

        for i in range(num_samples):
            self._plot_terminal_sample(fig, i, num_samples, num_cols, batch, predictions, labels)

        plt.tight_layout()
        return fig

    def _plot_terminal_sample(
        self,
        fig: plt.Figure,
        i: int,
        num_samples: int,
        num_cols: int,
        batch: Dict[str, Any],
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ):
        """Plot visualizations for a single terminal prediction sample."""
        # Get data for this sample
        image = batch["image"][i]
        seq_name = batch["seq_name"][i]
        frame_idx = batch["frame_idx"][i].item()
        bev_bounds = batch["bev_bounds"][i]

        # Get predictions and labels (terminal = single frame, not trajectory)
        pred_pos = predictions["terminal_root_pos"][i]  # [3]
        gt_pos = labels["terminal_root_pos"][i]  # [3]
        pred_rot6d = predictions["terminal_root_rot6d"][i]  # [6]
        gt_rot6d = labels["terminal_root_rot6d"][i]  # [6]
        pred_dof = predictions["terminal_dof_pos"][i]  # [num_dofs]
        gt_dof = labels["terminal_dof_pos"][i]  # [num_dofs]

        # Current frame reference
        current_root_pos = labels["current_root_pos"][i]
        current_root_quat = labels["current_root_quat"][i]

        # Transform terminal positions to world frame
        gt_pos_world = (
            self.fk_helper.transform_to_world(gt_pos.unsqueeze(0), current_root_quat).squeeze(0)
            + current_root_pos
        )
        pred_pos_world = (
            self.fk_helper.transform_to_world(pred_pos.unsqueeze(0), current_root_quat).squeeze(0)
            + current_root_pos
        )

        # Compute body positions via FK (add batch dim for FK)
        try:
            gt_body_world = self.fk_helper.dof_to_body_pos(
                gt_dof.unsqueeze(0), gt_pos.unsqueeze(0), gt_rot6d.unsqueeze(0), current_root_quat
            )
            pred_body_world = self.fk_helper.dof_to_body_pos(
                pred_dof.unsqueeze(0),
                pred_pos.unsqueeze(0),
                pred_rot6d.unsqueeze(0),
                current_root_quat,
            )
            gt_body_pos = (gt_body_world[0] + current_root_pos).cpu().numpy()  # [num_keypoints, 3]
            pred_body_pos = (
                (pred_body_world[0] + current_root_pos).cpu().numpy()
            )  # [num_keypoints, 3]
            fk_success = True
        except Exception as e:
            logger.warning(f"FK failed for sample {i}: {e}")
            fk_success = False

        # Convert to numpy
        gt_pos_np = gt_pos_world.cpu().numpy()
        pred_pos_np = pred_pos_world.cpu().numpy()
        gt_rot_np = gt_rot6d.cpu().numpy()
        pred_rot_np = pred_rot6d.cpu().numpy()
        current_pos_np = current_root_pos.cpu().numpy()

        # --- Plot 1: Input image ---
        ax = fig.add_subplot(num_samples, num_cols, i * num_cols + 1)
        self._plot_image(ax, image, seq_name, frame_idx)

        # --- Plot 2: BEV with terminal positions and arrows ---
        ax = fig.add_subplot(num_samples, num_cols, i * num_cols + 2)
        self._plot_terminal_bev(
            ax,
            gt_pos_np,
            pred_pos_np,
            gt_rot_np,
            pred_rot_np,
            current_root_quat,
            current_pos_np,
            bev_bounds,
        )

        # --- Plot 3: 3D skeleton comparison ---
        ax = fig.add_subplot(num_samples, num_cols, i * num_cols + 3, projection="3d")
        if fk_success:
            self._plot_terminal_skeleton_3d(ax, gt_body_pos, pred_body_pos)
        else:
            ax.set_title("FK failed", fontsize=8)

    def _plot_terminal_bev(
        self,
        ax,
        gt_pos: np.ndarray,
        pred_pos: np.ndarray,
        gt_rot: np.ndarray,
        pred_rot: np.ndarray,
        current_root_quat: torch.Tensor,
        current_pos: np.ndarray,
        bev_bounds: Dict[str, float],
    ):
        """Plot BEV with current position, terminal GT and terminal Pred with heading arrows."""
        # Plot current position
        ax.scatter(
            [current_pos[0]],
            [current_pos[1]],
            c="blue",
            s=150,
            marker="s",
            zorder=5,
            label="Current",
        )

        # Plot terminal positions
        ax.scatter(
            [gt_pos[0]], [gt_pos[1]], c="green", s=150, marker="*", zorder=5, label="GT Terminal"
        )
        ax.scatter(
            [pred_pos[0]],
            [pred_pos[1]],
            c="red",
            s=150,
            marker="*",
            zorder=5,
            label="Pred Terminal",
        )

        # Draw lines from current to terminal
        ax.plot(
            [current_pos[0], gt_pos[0]], [current_pos[1], gt_pos[1]], "g--", linewidth=2, alpha=0.5
        )
        ax.plot(
            [current_pos[0], pred_pos[0]],
            [current_pos[1], pred_pos[1]],
            "r--",
            linewidth=2,
            alpha=0.5,
        )

        # Axis limits from full motion bounds
        x_center = (bev_bounds["x_min"] + bev_bounds["x_max"]) / 2
        y_center = (bev_bounds["y_min"] + bev_bounds["y_max"]) / 2
        extent = (
            max(
                bev_bounds["x_max"] - bev_bounds["x_min"],
                bev_bounds["y_max"] - bev_bounds["y_min"],
                0.5,
            )
            * 1.1
        )
        arrow_len = extent * 0.05

        # Heading arrows for GT
        gt_fwd_world = (
            quat_rotate(
                current_root_quat.unsqueeze(0),
                torch.tensor(
                    gt_rot[:3], device=current_root_quat.device, dtype=torch.float
                ).unsqueeze(0),
                w_last=False,
            )
            .cpu()
            .numpy()[0]
        )
        gt_fwd = gt_fwd_world[:2]
        if np.linalg.norm(gt_fwd) > 0.1:
            gt_fwd = gt_fwd / np.linalg.norm(gt_fwd)
            ax.arrow(
                gt_pos[0],
                gt_pos[1],
                gt_fwd[0] * arrow_len,
                gt_fwd[1] * arrow_len,
                head_width=arrow_len * 0.4,
                head_length=arrow_len * 0.3,
                fc="green",
                ec="green",
                zorder=6,
                linewidth=2,
            )

        # Heading arrows for Pred
        pred_fwd_world = (
            quat_rotate(
                current_root_quat.unsqueeze(0),
                torch.tensor(
                    pred_rot[:3], device=current_root_quat.device, dtype=torch.float
                ).unsqueeze(0),
                w_last=False,
            )
            .cpu()
            .numpy()[0]
        )
        pred_fwd = pred_fwd_world[:2]
        if np.linalg.norm(pred_fwd) > 0.1:
            pred_fwd = pred_fwd / np.linalg.norm(pred_fwd)
            ax.arrow(
                pred_pos[0],
                pred_pos[1],
                pred_fwd[0] * arrow_len,
                pred_fwd[1] * arrow_len,
                head_width=arrow_len * 0.4,
                head_length=arrow_len * 0.3,
                fc="red",
                ec="red",
                zorder=6,
                linewidth=2,
            )

        ax.set_xlim(x_center - extent / 2, x_center + extent / 2)
        ax.set_ylim(y_center + extent / 2, y_center - extent / 2)  # Flipped: larger Y at bottom
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Terminal BEV", fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    def _plot_terminal_skeleton_3d(self, ax, gt_body_pos: np.ndarray, pred_body_pos: np.ndarray):
        """Plot GT and Pred terminal skeletons together."""
        for pos, color, label in [(gt_body_pos, "green", "GT"), (pred_body_pos, "red", "Pred")]:
            # Bones
            for start, end in FKHelper.SKELETON_BONES:
                ax.plot(
                    [pos[start, 0], pos[end, 0]],
                    [pos[start, 1], pos[end, 1]],
                    [pos[start, 2], pos[end, 2]],
                    c=color,
                    linewidth=2,
                    alpha=0.8,
                )

            # Joints
            regular_mask = np.ones(len(pos), dtype=bool)
            regular_mask[FKHelper.FOOT_INDICES + FKHelper.HAND_INDICES] = False
            ax.scatter(
                pos[regular_mask, 0],
                pos[regular_mask, 1],
                pos[regular_mask, 2],
                c=color,
                s=20,
                alpha=0.8,
            )

        # Special markers on GT
        ax.scatter(
            gt_body_pos[FKHelper.FOOT_INDICES, 0],
            gt_body_pos[FKHelper.FOOT_INDICES, 1],
            gt_body_pos[FKHelper.FOOT_INDICES, 2],
            c="orange",
            s=40,
            marker="^",
            label="Feet",
        )
        ax.scatter(
            gt_body_pos[FKHelper.HAND_INDICES, 0],
            gt_body_pos[FKHelper.HAND_INDICES, 1],
            gt_body_pos[FKHelper.HAND_INDICES, 2],
            c="purple",
            s=40,
            marker="o",
            label="Hands",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("GT (green) vs Pred (red)", fontsize=8)
        all_pos = np.concatenate([gt_body_pos, pred_body_pos])
        self._set_3d_axes_equal(ax, all_pos)

    def save_figure(self, fig: plt.Figure, save_dir: Path, step: int, wandb_log: bool = False):
        """Save figure to disk and optionally log to wandb."""
        import wandb
        from groot.rl.trl.utils.common import wandb_run_exists

        vis_dir = save_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(vis_dir / f"predictions_step_{step:06d}.png", dpi=100, bbox_inches="tight")

        if wandb_log and wandb_run_exists():
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            wandb.log(
                {"vis/predictions": wandb.Image(Image.open(buf), caption=f"Step {step}")}, step=step
            )
            buf.close()

        plt.close(fig)
        logger.info(f"Saved visualization for step {step}")
