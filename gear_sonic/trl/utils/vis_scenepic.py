"""ScenePic skeleton visualizer. Generates interactive 3D HTML animations."""

import numpy as np
import torch
from typing import Union, List, Optional
from pathlib import Path

try:
    import scenepic as sp

    SCENEPIC_AVAILABLE = True
except ImportError:
    SCENEPIC_AVAILABLE = False


def _normalize_vec(v, eps=1e-8):
    return v / (torch.norm(v, dim=-1, keepdim=True) + eps)


def _quat_to_exp_map(q):
    """Quaternion (wxyz) to axis-angle."""
    w, xyz = q[..., 0:1], q[..., 1:4]
    sin_half = torch.norm(xyz, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, w)
    axis = xyz / (sin_half + 1e-8)
    return axis * angle


def _quat_between_two_vec(v1, v2, eps=1e-6):
    """Quaternion (wxyz) rotating v1 to v2."""
    v1, v2 = v1.reshape(-1, 3), v2.reshape(-1, 3)
    dot = (v1 * v2).sum(-1)
    cross = torch.cross(v1, v2, dim=-1)
    out = _normalize_vec(torch.cat([(1 + dot).unsqueeze(-1), cross], dim=-1))
    out[dot > 1 - eps] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=v1.device)
    return out


def _make_floor_texture():
    from PIL import ImageColor

    c1 = np.tile(np.asarray(ImageColor.getcolor("#81C6EB", "RGB"), dtype=np.uint8), (5, 5, 1))
    c2 = np.tile(np.asarray(ImageColor.getcolor("#D4F1F7", "RGB"), dtype=np.uint8), (5, 5, 1))
    return np.tile(np.block([[[c1], [c2]], [[c2], [c1]]]), (15, 15, 1))


class SkeletonActor:
    """Skeleton with joints (spheres) and bones (cones)."""

    def __init__(
        self, scene, name, joint_parents, num_base_bodies=None, joint_radius=0.06, bone_radius=0.04
    ):
        self.joint_parents = joint_parents
        self.num_base = num_base_bodies or len(joint_parents)

        # Floor
        self.floor_img = scene.create_image(image_id="floor")
        self.floor_img.from_numpy(_make_floor_texture())
        self.floor_mesh = scene.create_mesh(texture_id="floor", layer_id="floor")
        self.floor_mesh.add_image(transform=sp.Transforms.Scale(20))

        # Joints and bones
        self.joint_meshes, self.bone_pairs = [], []
        for j, pa in enumerate(joint_parents):
            is_ext = j >= self.num_base
            jcolor = sp.Colors.Red if is_ext else (sp.Colors.Yellow if j == 0 else sp.Colors.Yellow)
            joint_mesh = scene.create_mesh(f"{name}_j{j}", layer_id=name)
            joint_mesh.add_sphere(color=jcolor, transform=sp.Transforms.scale(joint_radius))
            self.joint_meshes.append(joint_mesh)

            if pa >= 0:
                bcolor = sp.Colors.Orange if is_ext else sp.Colors.Green
                bone_mesh = scene.create_mesh(f"{name}_b{j}", layer_id=name)
                bone_mesh.add_cone(
                    color=bcolor,
                    transform=sp.Transforms.scale(np.array([1, bone_radius, bone_radius])),
                )
                self.bone_pairs.append((j, pa, bone_mesh))

    def add_to_frame(self, frame, pos):
        frame.add_mesh(self.floor_mesh)
        for j, p in enumerate(pos):
            frame.add_mesh(self.joint_meshes[j], transform=sp.Transforms.translate(p))

        if not self.bone_pairs:
            return
        vecs = np.stack([pos[j] - pos[pa] for j, pa, _ in self.bone_pairs])
        dists = np.linalg.norm(vecs, axis=-1)
        aa = _quat_to_exp_map(
            _quat_between_two_vec(
                torch.tensor([-1.0, 0.0, 0.0]).expand(len(vecs), 3),
                torch.tensor(vecs / (dists[:, None] + 1e-8)),
            )
        ).numpy()
        angles, axes = np.linalg.norm(aa, axis=-1, keepdims=True), aa / (
            np.linalg.norm(aa, axis=-1, keepdims=True) + 1e-8
        )

        for (j, pa, mesh), ang, ax, d in zip(self.bone_pairs, angles, axes, dists):
            t = sp.Transforms.translate((pos[pa] + pos[j]) * 0.5)
            t = (
                t
                @ sp.Transforms.RotationMatrixFromAxisAngle(ax, ang)
                @ sp.Transforms.Scale(np.array([d, 1, 1]))
            )
            frame.add_mesh(mesh, transform=t)


class ScenepicVisualizer:
    """Interactive 3D skeleton visualizer using ScenePic."""

    def __init__(self, joint_parents, num_base_bodies=None):
        if not SCENEPIC_AVAILABLE:
            raise ImportError("pip install scenepic")
        if isinstance(joint_parents, torch.Tensor):
            joint_parents = joint_parents.cpu().numpy().tolist()
        self.joint_parents = joint_parents
        self.num_base = num_base_bodies or len(joint_parents)

    def visualize(self, joint_positions, output_path, fps=30, title="Skeleton"):
        """Render skeleton animation to HTML. joint_positions: [T, J, 3]"""
        if isinstance(joint_positions, torch.Tensor):
            joint_positions = joint_positions.cpu().numpy()
        if joint_positions.ndim == 4:
            joint_positions = joint_positions[0]

        T = joint_positions.shape[0]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        scene = sp.Scene()
        scene.framerate = fps
        canvas = scene.create_canvas_3d(width=600, height=600)
        canvas.set_layer_settings({title: {"filled": True}})

        skeleton = SkeletonActor(scene, title, self.joint_parents, self.num_base)
        camera = sp.Camera(
            center=(4, 0, 1.5), look_at=(0, 0, 0.8), up_dir=(0, 0, 1), fov_y_degrees=45.0
        )

        for t in range(T):
            frame = canvas.create_frame()
            frame.camera = camera
            skeleton.add_to_frame(frame, joint_positions[t])

        scene.save_as_html(str(output_path))
        print(f"Saved: {output_path}")
        return output_path


if __name__ == "__main__":
    from pathlib import Path
    from omegaconf import OmegaConf
    from groot.rl.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
    from groot.rl.trl.utils.mujoco_fk_utils import MuJoCoFKHelper, load_qpos_from_csv

    groot_root = Path(__file__).parent.parent.parent.parent
    motion_yaml = (
        groot_root
        / "rl"
        / "config"
        / "manager_env"
        / "commands"
        / "terms"
        / "motion_g1_extended_toe.yaml"
    )
    cfg = OmegaConf.load(motion_yaml).motion.motion_lib_cfg
    fk_helper = MuJoCoFKHelper(Humanoid_Batch(cfg, device=torch.device("cpu")))
    print(
        f"Loaded: {fk_helper.num_bodies} + {fk_helper.num_bodies_augment - fk_helper.num_bodies} extended bodies"
    )

    csv_path = groot_root / ".." / "data" / "example_csv_g1_navigation.csv"
    if csv_path.exists():
        qpos = load_qpos_from_csv(str(csv_path))
    else:
        T, t = 60, torch.linspace(0, 4 * 3.14159, 60)
        dof = torch.zeros(T, fk_helper.num_dof)
        dof[:, 0], dof[:, 3] = 0.3 * torch.sin(t), 0.5 * torch.sin(t)
        dof[:, 6], dof[:, 9] = 0.3 * torch.sin(t + 3.14), 0.5 * torch.sin(t + 3.14)
        qpos = torch.cat(
            [
                torch.zeros(T, 3) + torch.tensor([0, 0, 1.0]),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(T, 4),
                dof,
            ],
            dim=-1,
        )

    global_pos, _ = fk_helper.qpos_to_global_transforms(
        qpos.unsqueeze(0), False, include_extended=True
    )

    viz = ScenepicVisualizer(fk_helper._parents, fk_helper.num_bodies)
    viz.visualize(
        global_pos[0],
        Path(__file__).parent.parent.parent.parent / "scenepic_demo.html",
        title="G1 Robot",
    )
    print("Yellow = base joints, Red = extended (head/toes)")
