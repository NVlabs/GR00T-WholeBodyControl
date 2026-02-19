"""SMPL/SMPLX body model utilities: downloading, creation, and pose decomposition."""

import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import smplx
import torch
import torch.nn.functional as F
from smplx import SMPL, SMPLX, SMPLXLayer

from groot.rl.trl.utils.smplx.body_model import BodyModelSMPLH, BodyModelSMPLX
from groot.rl.trl.utils.smplx.body_model.smplx_lite import (
    SmplxLiteCoco17,
    SmplxLiteSmplN24,
    SmplxLiteV437Coco17,
)

# fmt: off
SMPLH_PARENTS = torch.tensor([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                              16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
                              35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50])
# fmt: on


def _ensure_body_models_downloaded(model_path: str) -> Path:
    """
    Ensure body_models directory exists, downloading from gear data if needed.

    Args:
        model_path: Path to body_models directory (e.g., "data/body_models")

    Returns:
        Path object to the body_models directory
    """
    model_path_obj = Path(model_path)

    # Check if path exists and has required subdirectories (smplx is the main one we need)
    if model_path_obj.exists():
        smplx_subdir = model_path_obj / "smplx"
        if smplx_subdir.exists() and any(smplx_subdir.iterdir()):
            return model_path_obj

    # Try relative to project root (gr00t/), not groot/
    # File is at: groot/rl/trl/utils/smplx/smplx_utils.py
    # Need to go up 6 levels to get to project root
    project_root = Path(__file__).parent.parent.parent.parent.parent.parent
    full_path = project_root / model_path
    if full_path.exists():
        smplx_subdir = full_path / "smplx"
        if smplx_subdir.exists() and any(smplx_subdir.iterdir()):
            return full_path

    # Need to download - create directory
    full_path.mkdir(parents=True, exist_ok=True)

    print(f"Body models not found at {model_path}. Downloading from gear data storage...")
    try:
        # Use exact command format that works manually: gear data download wbc_data/body_models/** data/body_models/
        result = subprocess.run(
            [
                "gear",
                "data",
                "download",
                "wbc_data/body_models/**",
                str(full_path) + "/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Print output for debugging
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)

        # Verify download succeeded by checking if directory has contents
        if not any(full_path.iterdir()):
            raise RuntimeError(
                f"Download completed but directory is empty. "
                f"Check if 'wbc_data/body_models' exists in gear data storage.\n"
                f"Output: {result.stdout}\n"
                f"Error: {result.stderr}"
            )

        print(f"Successfully downloaded body models to {full_path}")
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if e.stderr else (e.stdout if e.stdout else "Unknown error")
        raise RuntimeError(
            f"Failed to download body models from wbc_data/body_models.\n"
            f"Error: {error_msg}\n"
            f"Please ensure 'gear' CLI is installed and configured, or manually download:\n"
            f"  gear data download wbc_data/body_models/** {full_path}/"
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"'gear' command not found. Please install gear CLI or manually download body models:\n"
            f"  gear data download wbc_data/body_models/** {full_path}/"
        )

    return full_path


def make_smplx(type="neu_fullpose", **kwargs):
    if type == "neu_fullpose":
        model = smplx.create(
            model_path="inputs/models/smplx/SMPLX_NEUTRAL.npz",
            use_pca=False,
            flat_hand_mean=True,
            **kwargs,
        )
    elif type == "supermotion":
        # SuperMotion is trained on BEDLAM dataset, the smplx config is the same except only 10 betas are used
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": False,
        }
        bm_kwargs.update(kwargs)
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = BodyModelSMPLX(model_path=str(body_models_path), **bm_kwargs)
    elif type == "supermotion_EVAL3DPW":
        # SuperMotion is trained on BEDLAM dataset, the smplx config is the same except only 10 betas are used
        bm_kwargs = {
            "model_type": "smplx",
            "gender": "neutral",
            "num_pca_comps": 12,
            "flat_hand_mean": True,
        }
        bm_kwargs.update(kwargs)
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = BodyModelSMPLX(model_path=str(body_models_path), **bm_kwargs)
    elif type == "supermotion_coco17":
        # Fast but only predicts 17 joints
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = SmplxLiteCoco17(model_path=str(body_models_path / "smplx"))
    elif type == "supermotion_v437coco17":
        # Predicts 437 verts and 17 joints
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = SmplxLiteV437Coco17(model_path=str(body_models_path / "smplx"))
    elif type == "supermotion_smpl24":
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = SmplxLiteSmplN24(model_path=str(body_models_path / "smplx"))
    elif type == "supermotion_smpl24_male":
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = SmplxLiteSmplN24(gender="male", model_path=str(body_models_path / "smplx"))
    elif type == "supermotion_smpl24_female":
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = SmplxLiteSmplN24(gender="female", model_path=str(body_models_path / "smplx"))
    elif type == "rich-smplx":
        # https://github.com/paulchhuang/rich_toolkit/blob/main/smplx2images.py
        bm_kwargs = {
            "model_type": "smplx",
            "gender": kwargs.get("gender", "male"),
            "num_pca_comps": 12,
            "flat_hand_mean": False,
            # create_expression=True, create_jaw_pose=Ture
        }
        # A /smplx folder should exist under the model_path
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = BodyModelSMPLX(model_path=str(body_models_path), **bm_kwargs)
    elif type == "rich-smplh":
        bm_kwargs = {
            "model_type": "smplh",
            "gender": kwargs.get("gender", "male"),
            "use_pca": False,
            "flat_hand_mean": True,
        }
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = BodyModelSMPLH(model_path=str(body_models_path), **bm_kwargs)

    elif type in ["smplx-circle", "smplx-groundlink"]:
        # don't use hand
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        bm_kwargs = {
            "model_path": str(body_models_path),
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 16,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type == "smplx-motionx":
        layer_args = {
            "create_global_orient": False,
            "create_body_pose": False,
            "create_left_hand_pose": False,
            "create_right_hand_pose": False,
            "create_jaw_pose": False,
            "create_leye_pose": False,
            "create_reye_pose": False,
            "create_betas": False,
            "create_expression": False,
            "create_transl": False,
        }

        body_models_path = _ensure_body_models_downloaded("data/body_models")
        bm_kwargs = {
            "model_type": "smplx",
            "model_path": str(body_models_path),
            "gender": "neutral",
            "use_pca": False,
            "use_face_contour": True,
            **layer_args,
        }
        model = smplx.create(**bm_kwargs)

    elif type == "smplx-samp":
        # don't use hand
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        bm_kwargs = {
            "model_path": str(body_models_path),
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 10,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type == "smplx-bedlam":
        # don't use hand
        bm_kwargs = {
            "model_path": "data/body_models",
            "model_type": "smplx",
            "gender": kwargs.get("gender"),
            "num_betas": 11,
            "num_expression": 0,
        }
        model = BodyModelSMPLX(**bm_kwargs)

    elif type in ["smplx-layer", "smplx-fit3d"]:
        # Use layer
        if type == "smplx-fit3d":
            assert (
                kwargs.get("gender") == "neutral"
            ), "smplx-fit3d use neutral model: https://github.com/sminchisescu-research/imar_vision_datasets_tools/blob/e8c8f83ffac23cc36adf8ec8d0fd1c55679484ef/util/smplx_util.py#L15C34-L15C34"

        body_models_path = _ensure_body_models_downloaded("data/body_models")
        bm_kwargs = {
            "model_path": str(body_models_path / "smplx"),
            "gender": kwargs.get("gender"),
            "num_betas": 10,
            "num_expression": 10,
        }
        model = SMPLXLayer(**bm_kwargs)

    elif type == "smpl":
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        bm_kwargs = {
            "model_path": str(body_models_path),
            "model_type": "smpl",
            "gender": "neutral",
            "num_betas": 10,
            "create_body_pose": False,
            "create_betas": False,
            "create_global_orient": False,
            "create_transl": False,
        }
        bm_kwargs.update(kwargs)
        # model = SMPL(**bm_kwargs)
        model = BodyModelSMPLH(**bm_kwargs)
    elif type == "smplh":
        bm_kwargs = {
            "model_type": "smplh",
            "gender": kwargs.get("gender", "male"),
            "use_pca": False,
            "flat_hand_mean": False,
        }
        body_models_path = _ensure_body_models_downloaded("data/body_models")
        model = BodyModelSMPLH(model_path=str(body_models_path), **bm_kwargs)

    else:
        raise NotImplementedError

    if hasattr(model, "create_parents_list"):
        model.create_parents_list()

    return model


def load_parents(npz_path="models/smplx/SMPLX_NEUTRAL.npz"):
    smplx_struct = np.load("models/smplx/SMPLX_NEUTRAL.npz", allow_pickle=True)
    parents = smplx_struct["kintree_table"][0].astype(np.long)
    parents[0] = -1
    return parents


def load_smpl_faces(npz_path="models/smplh/SMPLH_FEMALE.pkl"):
    with open(npz_path, "rb") as f:
        smpl_model = pickle.load(f, encoding="latin1")
    faces = np.array(smpl_model["f"].astype(np.int64))
    return faces


def decompose_fullpose(fullpose, model_type="smplx"):
    assert model_type == "smplx"

    fullpose_dict = {
        "global_orient": fullpose[..., :3],
        "body_pose": fullpose[..., 3:66],
        "jaw_pose": fullpose[..., 66:69],
        "leye_pose": fullpose[..., 69:72],
        "reye_pose": fullpose[..., 72:75],
        "left_hand_pose": fullpose[..., 75:120],
        "right_hand_pose": fullpose[..., 120:165],
    }

    return fullpose_dict


def compose_fullpose(fullpose_dict, model_type="smplx"):
    assert model_type == "smplx"
    fullpose = torch.cat(
        [
            fullpose_dict[k]
            for k in [
                "global_orient",
                "body_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "left_hand_pose",
                "right_hand_pose",
            ]
        ],
        dim=-1,
    )
    return fullpose


def compute_R_from_kinetree(rot_mats, parents):
    """operation of lbs/batch_rigid_transform, focus on 3x3 R only
    Parameters
    ----------
    rot_mats: torch.tensor BxNx3x3
        Tensor of rotation matrices
    parents : torch.tensor BxN
        The kinematic tree of each object

    Returns
    -------
    R : torch.tensor BxNx3x3
        Tensor of rotation matrices
    """
    rot_mat_chain = [rot_mats[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(rot_mat_chain[parents[i]], rot_mats[:, i])
        rot_mat_chain.append(curr_res)

    R = torch.stack(rot_mat_chain, dim=1)
    return R


def compute_relR_from_kinetree(R, parents):
    """Inverse operation of lbs/batch_rigid_transform, focus on 3x3 R only
    Parameters
    ----------
    R : torch.tensor BxNx4x4 or BxNx3x3
        Tensor of rotation matrices
    parents : torch.tensor BxN
        The kinematic tree of each object

    Returns
    -------
    rot_mats: torch.tensor BxNx3x3
        Tensor of rotation matrices
    """
    R = R[:, :, :3, :3]

    Rp = R[:, parents]  # Rp[:, 0] is invalid
    rot_mats = Rp.transpose(2, 3) @ R
    rot_mats[:, 0] = R[:, 0]

    return rot_mats


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    # res = np.concatenate(
    #     [
    #         y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
    #         y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
    #         y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
    #         y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
    #     ],
    #     axis=-1,
    # )
    res = torch.cat(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        axis=-1,
    )

    return res


def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    # res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    res = torch.tensor([1, -1, -1, -1], device=q.device).float() * q
    return res


def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    # t = 2.0 * np.cross(q[..., 1:], x)
    t = 2.0 * torch.cross(q[..., 1:], x)
    # res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)
    res = x + q[..., 0][..., None] * t + torch.cross(q[..., 1:], t)

    return res


def inverse_kinematics_motion(
    global_pos,
    global_rot,
    parents=SMPLH_PARENTS,
):
    """
    Args:
        global_pos : (B, T, J-1, 3)
        global_rot (q) : (B, T, J-1, 4)
        parents : SMPLH_PARENTS
    Returns:
        local_pos : (B, T, J-1, 3)
        local_rot (q) : (B, T, J-1, 4)
    """
    J = 22
    local_pos = quat_mul_vec(
        quat_inv(global_rot[..., parents[1:J], :]),
        global_pos - global_pos[..., parents[1:J], :],
    )
    local_rot = (quat_mul(quat_inv(global_rot[..., parents[1:J], :]), global_rot),)
    return local_pos, local_rot


def transform_mat(R, t):
    """Creates a batch of transformation matrices
    Args:
        - R: Bx3x3 array of a batch of rotation matrices
        - t: Bx3x1 array of a batch of translation vectors
    Returns:
        - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def normalize_joints(joints):
    """
    Args:
        joints: (B, *, J, 3)
    """
    LR_hips_xy = joints[..., 2, [0, 1]] - joints[..., 1, [0, 1]]
    LR_shoulders_xy = joints[..., 17, [0, 1]] - joints[..., 16, [0, 1]]
    LR_xy = (LR_hips_xy + LR_shoulders_xy) / 2  # (B, *, J, 2)

    x_dir = F.pad(F.normalize(LR_xy, 2, -1), (0, 1), "constant", 0)  # (B, *, 3)
    z_dir = torch.zeros_like(x_dir)  # (B, *, 3)
    z_dir[..., 2] = 1
    y_dir = torch.cross(z_dir, x_dir, dim=-1)

    joints_normalized = (joints - joints[..., [0], :]) @ torch.stack([x_dir, y_dir, z_dir], dim=-1)
    return joints_normalized


@torch.no_grad()
def compute_Rt_af2az(joints, inverse=False):
    """Assume z coord is upward
    Args:
        joints: (B, J, 3), in the start-frame
    Returns:
        R_af2az: (B, 3, 3)
        t_af2az: (B, 3)
    """
    t_af2az = joints[:, 0, :].detach().clone()
    t_af2az[:, 2] = 0  # do not modify z

    LR_xy = joints[:, 2, [0, 1]] - joints[:, 1, [0, 1]]  # (B, 2)
    I_mask = LR_xy.pow(2).sum(-1) < 1e-4  # do not rotate, when can't decided the face direction
    x_dir = F.pad(F.normalize(LR_xy, 2, -1), (0, 1), "constant", 0)  # (B, 3)
    z_dir = torch.zeros_like(x_dir)
    z_dir[..., 2] = 1
    y_dir = torch.cross(z_dir, x_dir, dim=-1)
    R_af2az = torch.stack([x_dir, y_dir, z_dir], dim=-1)  # (B, 3, 3)
    R_af2az[I_mask] = torch.eye(3).to(R_af2az)

    if inverse:
        R_az2af = R_af2az.transpose(1, 2)
        t_az2af = -(R_az2af @ t_af2az.unsqueeze(2)).squeeze(2)
        return R_az2af, t_az2af
    else:
        return R_af2az, t_af2az


def finite_difference_forward(x, dim_t=1, dup_last=True):
    if dim_t == 1:
        v = x[:, 1:] - x[:, :-1]
        if dup_last:
            v = torch.cat([v, v[:, [-1]]], dim=1)
    else:
        raise NotImplementedError

    return v


def compute_joints_zero(betas, gender):
    """
    Args:
        betas: (16)
        gender: 'male' or 'female'
    Returns:
        joints_zero: (22, 3)
    """
    body_model = {
        "male": make_smplx(type="humor", gender="male"),
        "female": make_smplx(type="humor", gender="female"),
    }

    smpl_params = {
        "root_orient": torch.zeros((1, 3)),
        "pose_body": torch.zeros((1, 63)),
        "betas": betas[None],
        "trans": torch.zeros(1, 3),
    }
    joints_zero = body_model[gender](**smpl_params).Jtr[0, :22]
    return joints_zero
