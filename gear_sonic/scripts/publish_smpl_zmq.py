#!/usr/bin/env python3
"""
Stream SMPL pose parameters to the SONIC G1 deployment over ZMQ. Written by Xianghui Xie, March 15, 2026. 

Dependency: the human joints file gear_sonic/data/human/human_joints_info.pkl, from https://github.com/NVlabs/GR00T-WholeBodyControl/blob/main/gear_sonic/data/human/human_joints_info.pkl

Standalone version of examples/zmq_publisher.py — all local Python
dependencies (examples.torch_transform, examples.rotations, gear_sonic.*)
have been inlined.  External deps only:

    pip install zmq numpy scipy torch smplx open3d

------------------------------------------------------------------------------
STEP 1 — Visualise in the MuJoCo simulator  (3 terminals, run from repo root)
------------------------------------------------------------------------------

  Terminal 1 — MuJoCo simulator:
    source .venv_sim/bin/activate
    python gear_sonic/scripts/run_sim_loop.py

  Terminal 2 — C++ deployment (from gear_sonic_deploy/):
    bash deploy.sh sim --input-type zmq   --zmq-host localhost --zmq-port 5556   --zmq-topic pose 

  Terminal 3 — This publisher:
    python gear_sonic/scripts/publish_smpl_zmq.py

  In Terminal 2:
    Press  ]      to start the control system
    In the MuJoCo window press  9  to drop the robot to the ground
    Press  ENTER  to enable ZMQ streaming mode
    Press  O      for emergency stop

------------------------------------------------------------------------------
STEP 2 — Deploy on the real G1 robot  (2 terminals)
------------------------------------------------------------------------------

  Terminal 1 — C++ deployment ON THE ROBOT (from gear_sonic_deploy/):
    bash deploy.sh real --input-type zmq --zmq-host <IP-of-publisher-machine> --zmq-port 5556   --zmq-topic pose 

  Terminal 2 — This publisher (on your workstation):
    python gear_sonic/scripts/publish_smpl_zmq.py

  In Terminal 1:
    Press  ]      to start the control system
    Press  ENTER  to enable ZMQ streaming mode
    Press  O      for emergency stop

  NOTE: replace <IP-of-publisher-machine> with the IP address of the machine
  running this script.  If both terminals are on the same machine, use
  localhost.
"""

import argparse
import json
import queue
import threading
import time

import numpy as np
import zmq

from scipy.spatial.transform import Rotation as R, Rotation as sRot
import torch
import torch.nn.functional as F

# ===========================================================================
# Inlined rotation math — replaces examples.torch_transform,
# examples.rotations, and gear_sonic.* imports.
# All quaternions use scalar-first [w, x, y, z] convention.
# ===========================================================================

def _normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps).unsqueeze(-1)


def _safe_zero_div(num: torch.Tensor, den: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    den = den.clone()
    den = torch.where(den.abs() < eps, den + eps, den)
    return num / den


def _torch_safe_atan2(y: torch.Tensor, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    y = y.clone()
    y[(y.abs() < eps) & (x.abs() < eps)] += eps
    return torch.atan2(y, x)


# ---------------------------------------------------------------------------
# angle_axis → rotation matrix  (from kornia_transform)
# ---------------------------------------------------------------------------

def _rotation_matrix_from_aa(angle_axis: torch.Tensor, theta2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    theta = torch.sqrt(theta2.clamp_min(eps))
    wxyz = angle_axis / (theta + eps)
    wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
    c, s = torch.cos(theta), torch.sin(theta)
    k = 1.0
    r00 = c + wx*wx*(k-c);  r01 = wx*wy*(k-c) - wz*s; r02 = wy*s  + wx*wz*(k-c)
    r10 = wz*s + wx*wy*(k-c); r11 = c + wy*wy*(k-c);  r12 = -wx*s + wy*wz*(k-c)
    r20 = -wy*s + wx*wz*(k-c); r21 = wx*s + wy*wz*(k-c); r22 = c + wz*wz*(k-c)
    return torch.cat([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1).view(-1, 3, 3)


def _rotation_matrix_taylor(angle_axis: torch.Tensor) -> torch.Tensor:
    rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
    o = torch.ones_like(rx)
    return torch.cat([o, -rz, ry, rz, o, -rx, -ry, rx, o], dim=1).view(-1, 3, 3)


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    orig_shape = angle_axis.shape
    aa = angle_axis.reshape(-1, 3)
    theta2 = (aa.unsqueeze(1) @ aa.unsqueeze(2)).squeeze(1)
    rot_normal = _rotation_matrix_from_aa(aa, theta2)
    rot_taylor = _rotation_matrix_taylor(aa)
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device).type_as(theta2)
    rot = torch.eye(3, device=aa.device, dtype=aa.dtype).view(1, 3, 3).repeat(aa.shape[0], 1, 1)
    rot[..., :3, :3] = mask * rot_normal + (1 - mask) * rot_taylor
    return rot.view(orig_shape[:-1] + (3, 3))


# ---------------------------------------------------------------------------
# angle_axis ↔ quaternion  (from kornia_transform)
# ---------------------------------------------------------------------------

def angle_axis_to_quaternion(angle_axis: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a0, a1, a2 = angle_axis[..., 0:1], angle_axis[..., 1:2], angle_axis[..., 2:3]
    theta_sq = a0*a0 + a1*a1 + a2*a2
    theta = torch.sqrt(theta_sq.clamp_min(eps))
    half_theta = theta * 0.5
    mask = (theta_sq > 0.0)
    k = torch.where(mask, _safe_zero_div(torch.sin(half_theta), theta, eps), 0.5 * torch.ones_like(half_theta))
    w = torch.where(mask, torch.cos(half_theta), torch.ones_like(half_theta))
    q = torch.zeros(angle_axis.shape[:-1] + (4,), dtype=angle_axis.dtype, device=angle_axis.device)
    q[..., 0:1] = w
    q[..., 1:2] = a0 * k
    q[..., 2:3] = a1 * k
    q[..., 3:4] = a2 * k
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    cos_theta = quaternion[..., 0]
    q1, q2, q3 = quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    sin_sq = q1*q1 + q2*q2 + q3*q3
    sin_theta = torch.sqrt(sin_sq.clamp_min(eps))
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        _torch_safe_atan2(-sin_theta, -cos_theta),
        _torch_safe_atan2(sin_theta,  cos_theta),
    )
    k = torch.where(sin_sq > 0.0, _safe_zero_div(two_theta, sin_theta, eps), 2.0 * torch.ones_like(sin_theta))
    aa = torch.zeros_like(quaternion)[..., :3]
    aa[..., 0] = q1 * k
    aa[..., 1] = q2 * k
    aa[..., 2] = q3 * k
    return aa


# ---------------------------------------------------------------------------
# quaternion ↔ rotation matrix  (from kornia_transform)
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    q = F.normalize(quaternion, p=2.0, dim=-1, eps=1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    tx, ty, tz = 2*x, 2*y, 2*z
    twx, twy, twz = tx*w, ty*w, tz*w
    txx, txy, txz = tx*x, ty*x, tz*x
    tyy, tyz, tzz = ty*y, tz*y, tz*z
    one = torch.tensor(1.0, device=q.device, dtype=q.dtype)
    mat = torch.stack([
        one-(tyy+tzz), txy-twz,       txz+twy,
        txy+twz,       one-(txx+tzz), tyz-twx,
        txz-twy,       tyz+twx,       one-(txx+tyy),
    ], dim=-1).view(quaternion.shape[:-1] + (3, 3))
    return mat


def rotation_matrix_to_quaternion(rotation_matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Returns [w, x, y, z]."""
    m = rotation_matrix
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
    trace = m00 + m11 + m22

    sq = torch.sqrt((trace + 1.0).clamp_min(eps)) * 2.0
    tp = torch.stack((_safe_zero_div(sq*0.25, torch.ones_like(sq)),
                      _safe_zero_div(m21-m12, sq),
                      _safe_zero_div(m02-m20, sq),
                      _safe_zero_div(m10-m01, sq)), dim=-1)

    sq1 = torch.sqrt((1+m00-m11-m22).clamp_min(eps)) * 2.0
    c1  = torch.stack((_safe_zero_div(m21-m12, sq1),
                       sq1*0.25*torch.ones_like(sq1),
                       _safe_zero_div(m01+m10, sq1),
                       _safe_zero_div(m02+m20, sq1)), dim=-1)

    sq2 = torch.sqrt((1+m11-m00-m22).clamp_min(eps)) * 2.0
    c2  = torch.stack((_safe_zero_div(m02-m20, sq2),
                       _safe_zero_div(m01+m10, sq2),
                       sq2*0.25*torch.ones_like(sq2),
                       _safe_zero_div(m12+m21, sq2)), dim=-1)

    sq3 = torch.sqrt((1+m22-m00-m11).clamp_min(eps)) * 2.0
    c3  = torch.stack((_safe_zero_div(m10-m01, sq3),
                       _safe_zero_div(m02+m20, sq3),
                       _safe_zero_div(m12+m21, sq3),
                       sq3*0.25*torch.ones_like(sq3)), dim=-1)

    w23  = torch.where((m11 > m22).unsqueeze(-1), c2, c3)
    w1   = torch.where(((m00 > m11) & (m00 > m22)).unsqueeze(-1), c1, w23)
    return torch.where((trace > 0.0).unsqueeze(-1), tp, w1)


# ---------------------------------------------------------------------------
# Quaternion ops  (from torch_transform)
# ---------------------------------------------------------------------------

def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    """Conjugate of [w,x,y,z] → [w,-x,-y,-z]."""
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


def quat_inv(a: torch.Tensor) -> torch.Tensor:
    return _normalize(quat_conjugate(a))


def quat_apply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Rotate vector b by quaternion a (both batched). a=[w,x,y,z], b=[x,y,z]."""
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:].clone()
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 0:1].clone() * t + xyz.cross(t, dim=-1)).view(shape)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of [w,x,y,z] quaternions."""
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1+x1)*(x2+y2); yy = (w1-y1)*(w2+z2); zz = (w1+y1)*(w2-z2)
    xx = ww + yy + zz
    qq = 0.5*(xx + (z1-x1)*(x2-y2))
    w  = qq - ww + (z1-y1)*(y2-z2)
    x  = qq - xx + (x1+w1)*(x2+w2)
    y  = qq - yy + (w1-x1)*(y2+z2)
    z  = qq - zz + (z1+y1)*(w2-x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


# ---------------------------------------------------------------------------
# SMPL root orientation helpers  (from examples/rotations.py)
# ---------------------------------------------------------------------------

def smpl_root_ytoz_up(root_quat_y_up: torch.Tensor) -> torch.Tensor:
    """Convert SMPL root quaternion from Y-up to Z-up (90° rotation about X)."""
    base_rot = angle_axis_to_quaternion(
        torch.tensor([[np.pi / 2, 0.0, 0.0]]).to(root_quat_y_up)
    )
    return quat_mul(base_rot.repeat(root_quat_y_up.shape[0], 1), root_quat_y_up)


def remove_smpl_base_rot(quat: torch.Tensor) -> torch.Tensor:
    """Remove SMPL default rest pose — conjugate of [0.5,0.5,0.5,0.5]."""
    base_rot = quat_conjugate(torch.tensor([[0.5, 0.5, 0.5, 0.5]]).to(quat))
    return quat_mul(quat, base_rot.repeat(quat.shape[0], 1))


# ---------------------------------------------------------------------------
# SMPL forward kinematics (compute_human_joints from torch_transform)
# Loads rest-pose data from gear_sonic/data/human/human_joints_info.pkl
# ---------------------------------------------------------------------------

_human_joints_info = None


def compute_human_joints(
    body_pose: torch.Tensor,
    global_orient: torch.Tensor,
    human_joints_info_path: str = "gear_sonic/data/human/human_joints_info.pkl",
    use_thumb_joints: bool = True,
) -> torch.Tensor:
    global _human_joints_info
    if _human_joints_info is None:
        _human_joints_info = torch.load(human_joints_info_path)
    J = _human_joints_info["J"]
    parents_list = _human_joints_info["parents_list"]
    device = body_pose.device
    J = J.to(device)

    other_pose = torch.zeros(*body_pose.shape[:-1], 99, device=device)
    full_pose = torch.cat([global_orient, body_pose, other_pose], dim=-1)
    rot_mats = angle_axis_to_rotation_matrix(full_pose.reshape(*full_pose.shape[:-1], 55, 3))

    J = J.expand(*rot_mats.shape[:-3], -1, -1)
    rel_joints = J.clone()
    rel_joints[..., 1:, :] -= J[..., parents_list[1:], :]

    transforms_mat = F.pad(
        torch.cat([rot_mats, rel_joints[..., :, None]], dim=-1), [0, 0, 0, 1], value=0.0
    )
    transforms_mat[..., 3, 3] = 1.0

    chain = [transforms_mat[..., 0, :, :]]
    for i in range(1, len(parents_list)):
        chain.append(torch.matmul(chain[parents_list[i]], transforms_mat[..., i, :, :]))

    joints = torch.stack(chain, dim=-3)[..., :3, 3]
    idx = np.arange(22)
    if use_thumb_joints:
        idx = np.concatenate([idx, np.array([39, 54])])
    return joints[:, idx]


# ===========================================================================
# Visualization
# ===========================================================================

# Colors for the 24 SMPL joints (RGB, 0–1).
# Left side = blue, right side = red, spine/center = white/yellow.
# Joint order: 0:pelvis 1:L_hip 2:R_hip 3:spine1 4:L_knee 5:R_knee
#              6:spine2 7:L_ankle 8:R_ankle 9:spine3 10:L_foot 11:R_foot
#              12:neck 13:L_collar 14:R_collar 15:head
#              16:L_shoulder 17:R_shoulder 18:L_elbow 19:R_elbow
#              20:L_wrist 21:R_wrist 22:L_hand 23:R_hand
SMPL_JOINT_COLORS = [
    [1.00, 1.00, 0.20],  #  0 pelvis       (yellow)
    [0.20, 0.60, 1.00],  #  1 L_hip        (blue)
    [1.00, 0.30, 0.20],  #  2 R_hip        (red)
    [1.00, 1.00, 0.50],  #  3 spine1       (light yellow)
    [0.10, 0.50, 1.00],  #  4 L_knee       (blue)
    [1.00, 0.20, 0.10],  #  5 R_knee       (red)
    [1.00, 1.00, 0.70],  #  6 spine2       (pale yellow)
    [0.10, 0.40, 0.90],  #  7 L_ankle      (blue)
    [0.90, 0.20, 0.10],  #  8 R_ankle      (red)
    [0.95, 0.95, 0.95],  #  9 spine3       (white)
    [0.05, 0.35, 0.80],  # 10 L_foot       (dark blue)
    [0.80, 0.15, 0.05],  # 11 R_foot       (dark red)
    [0.80, 0.95, 0.80],  # 12 neck         (light green)
    [0.30, 0.70, 1.00],  # 13 L_collar     (blue)
    [1.00, 0.45, 0.30],  # 14 R_collar     (red)
    [0.50, 1.00, 0.50],  # 15 head         (green)
    [0.20, 0.65, 1.00],  # 16 L_shoulder   (blue)
    [1.00, 0.35, 0.20],  # 17 R_shoulder   (red)
    [0.15, 0.55, 0.95],  # 18 L_elbow      (blue)
    [0.95, 0.25, 0.15],  # 19 R_elbow      (red)
    [0.10, 0.45, 0.85],  # 20 L_wrist      (blue)
    [0.85, 0.20, 0.10],  # 21 R_wrist      (red)
    [0.05, 0.35, 0.75],  # 22 L_hand       (dark blue)
    [0.75, 0.15, 0.05],  # 23 R_hand       (dark red)
]

# Parent joint index for each of the 24 visualized joints (-1 = root).
SMPL_BONE_PARENTS = [
    -1,  #  0 pelvis (root)
     0,  #  1 L_hip
     0,  #  2 R_hip
     0,  #  3 spine1
     1,  #  4 L_knee
     2,  #  5 R_knee
     3,  #  6 spine2
     4,  #  7 L_ankle
     5,  #  8 R_ankle
     6,  #  9 spine3
     7,  # 10 L_foot
     8,  # 11 R_foot
     9,  # 12 neck
     9,  # 13 L_collar
     9,  # 14 R_collar
    12,  # 15 head
    13,  # 16 L_shoulder
    14,  # 17 R_shoulder
    16,  # 18 L_elbow
    17,  # 19 R_elbow
    18,  # 20 L_wrist
    19,  # 21 R_wrist
    20,  # 22 L_hand
    21,  # 23 R_hand
]

# (child_idx, parent_idx) for every bone — skip root.
SMPL_BONE_PAIRS = [(i, SMPL_BONE_PARENTS[i]) for i in range(24) if SMPL_BONE_PARENTS[i] != -1]

_viz_queue = queue.Queue(maxsize=2)  # (joint_positions np.ndarray (24,3), frame_index int)


def _align_cyl_verts(template: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """Transform unit-Z cylinder template vertices so the cylinder connects p0 → p1."""
    d = p1 - p0
    length = np.linalg.norm(d)
    if length < 1e-8:
        return np.tile(p0, (len(template), 1))
    scaled = template.copy()
    scaled[:, 2] *= length
    u = d / length
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, u)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, u)
    if s < 1e-8:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))
    return scaled @ rot.T + p0


def _viz_worker() -> None:
    """Visualization thread: renders 24 SMPL body joints as colored spheres
    and 23 bones as colored cylinders connecting parent–child joints."""
    import open3d as o3d

    _CAM_FRONT = np.array([0.0, 1.0, 0.2])   # look from -Y toward scene, Z-up
    _CAM_FRONT /= np.linalg.norm(_CAM_FRONT)
    _CAM_UP    = np.array([0.0, 0.0, 1.0])   # Z is up

    vis = o3d.visualization.Visualizer()
    vis.create_window("SMPL Joints", width=500, height=500, top=200, left=2000)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.array([0.15, 0.15, 0.15])
    opt.light_on = False

    # Ground plane grid (LineSet) — Z-up, XY plane at fixed Z, follows character via 1 m snapping
    _GROUND_Z  = -1.0
    _GRID_HALF = 20
    _GRID_STEP = 1.0
    _grid_coords = np.arange(-_GRID_HALF, _GRID_HALF + _GRID_STEP, _GRID_STEP)
    _gpts_local, _glines = [], []
    _gi = 0
    for y in _grid_coords:
        _gpts_local += [[-_GRID_HALF, y, _GROUND_Z], [_GRID_HALF, y, _GROUND_Z]]
        _glines.append([_gi, _gi + 1]); _gi += 2
    for x in _grid_coords:
        _gpts_local += [[x, -_GRID_HALF, _GROUND_Z], [x, _GRID_HALF, _GROUND_Z]]
        _glines.append([_gi, _gi + 1]); _gi += 2
    _gpts_local   = np.array(_gpts_local, dtype=np.float64)
    _glines_np    = np.array(_glines)
    _ground_color = np.full((len(_glines_np), 3), 0.35)
    geom_ground = o3d.geometry.LineSet()
    geom_ground.points = o3d.utility.Vector3dVector(_gpts_local)
    geom_ground.lines  = o3d.utility.Vector2iVector(_glines_np)
    geom_ground.colors = o3d.utility.Vector3dVector(_ground_color)
    vis.add_geometry(geom_ground)

    # Build merged sphere mesh: 24 spheres fused into one geometry for efficiency.
    # Topology (faces) is fixed; only vertices move each frame.
    _sphere_tmpl  = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    _sv = np.asarray(_sphere_tmpl.vertices).copy()   # (N_v, 3) unit sphere
    _sf = np.asarray(_sphere_tmpl.triangles).copy()  # (N_f, 3)
    N_v, N_f = len(_sv), len(_sf)

    all_faces = np.zeros((24 * N_f, 3), dtype=np.int32)
    for j in range(24):
        all_faces[j*N_f:(j+1)*N_f] = _sf + j * N_v

    # Pre-assign per-joint colors (constant across frames)
    _jcolors = np.array(SMPL_JOINT_COLORS, dtype=np.float64)  # (24, 3)
    all_colors = np.zeros((24 * N_v, 3))
    for j in range(24):
        all_colors[j*N_v:(j+1)*N_v] = _jcolors[j]

    geom_joints = o3d.geometry.TriangleMesh()
    geom_joints.triangles    = o3d.utility.Vector3iVector(all_faces)
    geom_joints.vertex_colors = o3d.utility.Vector3dVector(all_colors)

    # Build merged cylinder mesh for 23 bones (same batching strategy as spheres).
    _BONE_RADIUS = 0.05 / 3
    _cyl_tmpl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=_BONE_RADIUS, height=1.0, resolution=10, split=1,
    )
    _cv = np.asarray(_cyl_tmpl.vertices).copy()
    _cv[:, 2] += 0.5  # shift from z ∈ [-0.5, 0.5] to z ∈ [0, 1]
    _cf = np.asarray(_cyl_tmpl.triangles).copy()
    N_cv, N_cf = len(_cv), len(_cf)
    N_bones = len(SMPL_BONE_PAIRS)

    all_bone_faces = np.zeros((N_bones * N_cf, 3), dtype=np.int32)
    for b in range(N_bones):
        all_bone_faces[b * N_cf:(b + 1) * N_cf] = _cf + b * N_cv

    _bcolors = np.zeros((N_bones, 3))
    for b, (child, parent) in enumerate(SMPL_BONE_PAIRS):
        _bcolors[b] = (np.array(SMPL_JOINT_COLORS[child]) +
                        np.array(SMPL_JOINT_COLORS[parent])) / 2.0
    all_bone_colors = np.zeros((N_bones * N_cv, 3))
    for b in range(N_bones):
        all_bone_colors[b * N_cv:(b + 1) * N_cv] = _bcolors[b]

    geom_bones = o3d.geometry.TriangleMesh()
    geom_bones.triangles     = o3d.utility.Vector3iVector(all_bone_faces)
    geom_bones.vertex_colors = o3d.utility.Vector3dVector(all_bone_colors)

    joints_added  = False
    bones_added   = False
    cam_init      = False
    vis_frame_idx = -1

    try:
        while True:
            # Drain queue — keep only the newest frame
            joints = None
            while True:
                try:
                    _jpos, _fidx = _viz_queue.get_nowait()
                    if _fidx > vis_frame_idx:
                        joints = _jpos
                        vis_frame_idx = _fidx
                except queue.Empty:
                    break

            if joints is not None:
                # Place each sphere at its joint position
                all_verts = np.empty((24 * N_v, 3))
                for j in range(24):
                    all_verts[j*N_v:(j+1)*N_v] = _sv + joints[j]
                geom_joints.vertices = o3d.utility.Vector3dVector(all_verts)

                if not joints_added:
                    vis.add_geometry(geom_joints)
                    joints_added = True
                else:
                    vis.update_geometry(geom_joints)

                # Place each cylinder along its bone
                all_bone_verts = np.empty((N_bones * N_cv, 3))
                for b, (child, parent) in enumerate(SMPL_BONE_PAIRS):
                    all_bone_verts[b * N_cv:(b + 1) * N_cv] = _align_cyl_verts(
                        _cv, joints[parent], joints[child],
                    )
                geom_bones.vertices = o3d.utility.Vector3dVector(all_bone_verts)

                if not bones_added:
                    vis.add_geometry(geom_bones)
                    bones_added = True
                else:
                    vis.update_geometry(geom_bones)

                # Snap ground grid to character's XY (1 m discrete steps)
                centroid = joints.mean(axis=0)
                snap_x = round(centroid[0] / _GRID_STEP) * _GRID_STEP
                snap_y = round(centroid[1] / _GRID_STEP) * _GRID_STEP
                geom_ground.points = o3d.utility.Vector3dVector(
                    _gpts_local + np.array([snap_x, snap_y, 0.0])
                )
                vis.update_geometry(geom_ground)

                # Camera: init front/up/zoom once; track lookat XY every frame.
                # Keep lookat Z fixed so the ground plane doesn't appear to
                # move as the character's height bounces during motion.
                vc = vis.get_view_control()
                vc.set_lookat([centroid[0], centroid[1], _GROUND_Z + 1.0])
                if not cam_init:
                    vc.set_front(_CAM_FRONT.tolist())
                    vc.set_up(_CAM_UP.tolist())
                    vc.set_zoom(0.3)
                    cam_init = True

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.016)  # ~60 Hz

    finally:
        vis.destroy_window()


# ===========================================================================
# ZMQ Protocol v3 message packing
# ===========================================================================

_HEADER_SIZE = 1280
_NUM_ROBOT_JOINTS = 29


def _pack_pose_v3(
    body_quat: np.ndarray,   # [N, 4]
    frame_index: np.ndarray, # [N]
    smpl_joints: np.ndarray, # [N, 24, 3]
    smpl_pose: np.ndarray,   # [N, 21, 3]
    joint_pos: np.ndarray,   # [N, 29]
    joint_vel: np.ndarray,   # [N, 29]
    topic: str = "pose",
) -> bytes:
    arrays = [
        ("body_quat",   body_quat.astype(np.float32),   "f32"),
        ("frame_index", frame_index.astype(np.int32),    "i32"),
        ("joint_pos",   joint_pos.astype(np.float32),    "f32"),
        ("joint_vel",   joint_vel.astype(np.float32),    "f32"),
        ("smpl_joints", smpl_joints.astype(np.float32),  "f32"),
        ("smpl_pose",   smpl_pose.astype(np.float32),    "f32"),
    ]
    fields, payloads = [], []
    for name, arr, dtype in arrays:
        arr = np.ascontiguousarray(arr)
        fields.append({"name": name, "dtype": dtype, "shape": list(arr.shape)})
        payloads.append(arr.tobytes())

    header_json = json.dumps(
        {"v": 3, "endian": "le", "count": 1, "fields": fields},
        separators=(",", ":"),
    ).encode("utf-8")
    assert len(header_json) <= _HEADER_SIZE, f"Header too large ({len(header_json)} bytes)"
    header_bytes = header_json.ljust(_HEADER_SIZE, b"\x00")
    return topic.encode("utf-8") + header_bytes + b"".join(payloads)


# ===========================================================================
# SMPL processing
# ===========================================================================

def process_smpl_joints(body_pose, global_orient):
    """
    body_pose: (B, 69), relative local body pose in SMPL axis angle format
    global_orient: (B, 3), global orientation, axis angle 

    """
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new  = quaternion_to_angle_axis(global_orient_quat)

    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
    )  # (*, 24, 3)

    global_orient_quat = remove_smpl_base_rot(global_orient_quat)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d  = global_orient_mat[..., :2].reshape(-1, 6)

    return {
        "smpl_pose":          body_pose,
        "joints":             joints,
        "smpl_joints_local":  smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d":   global_orient_6d
    }


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish SMPL pose parameters over ZMQ to the SONIC G1 deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host",  default="0.0.0.0",
                        help="ZMQ bind address (* = all interfaces)")
    parser.add_argument("--port",  type=int, default=5556, help="ZMQ publisher port")
    parser.add_argument("--topic", default="pose",         help="ZMQ topic prefix")
    parser.add_argument("--hz",    type=float, default=50,
                        help="Target publish rate in Hz")
    parser.add_argument("--motion_file", help="Path to AMASS .npz motion file",
                        default='/mnt/data/AMASS/DFaust_67/50002/50002_running_on_spot_poses.npz')
    args = parser.parse_args()

    interval = 1.0 / args.hz

    ctx  = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://{args.host}:{args.port}")

    print(f"[smpl_publisher] Bound tcp://{args.host}:{args.port}  "
          f"topic={args.topic!r}  rate={args.hz:.1f} Hz")
    print("[smpl_publisher] Waiting 300 ms for ZMQ subscribers to connect …")
    time.sleep(0.3)
    print("[smpl_publisher] Streaming. Press Ctrl-C to stop.")

    # ---- Playback gate: press 'y' + Enter to start advancing frames ----
    _playback_started = threading.Event()

    def _wait_for_y():
        print("[smpl_publisher] Sending first-frame pose. Press 'y' + Enter to start playback.")
        while not _playback_started.is_set():
            try:
                line = input()
            except EOFError:
                break
            if line.strip().lower() == "y":
                _playback_started.set()
                print("[smpl_publisher] Playback started.")

    threading.Thread(target=_wait_for_y, daemon=True).start()

    pose_data = np.load(args.motion_file)
    pose_data_new = {k: pose_data[k] for k in pose_data}
    pose_data = pose_data_new
    pose_data['trans'] = pose_data['trans'] - pose_data['trans'][0:1]

    def _run_publisher() -> None:
        HISTORY_SIZE = 15
        frame_index  = 0

        amass2yup = np.array([
            [-1, 0, 0.],
            [ 0, 0, 1 ],
            [ 0, 1, 0 ],
        ])
        L = len(pose_data['poses'])

        def get_data_idx(fi: int) -> int:
            r = fi // L
            return fi % L if r % 2 == 0 else L - 1 - (fi % L)

        try:
            while True:
                t0 = time.monotonic()

                if _playback_started.is_set():
                    window_abs = [max(0, frame_index - HISTORY_SIZE + 1 + i) for i in range(HISTORY_SIZE)]
                else:
                    window_abs = [0] * HISTORY_SIZE

                data_idxs = [get_data_idx(fi) for fi in window_abs]

                go_list, bp_list, tr_list = [], [], []
                for di in data_idxs:
                    go = np.matmul(amass2yup, R.from_rotvec(pose_data['poses'][di, :3]).as_matrix())
                    roty120 = sRot.from_euler("y", 120, degrees=True).as_matrix()
                    go = np.matmul(roty120, go)
                    go_list.append(R.from_matrix(go).as_rotvec())
                    bp_list.append(pose_data['poses'][di, 3:72])
                    t = np.matmul(amass2yup, pose_data['trans'][di][:, None])[:, 0]
                    tr_list.append(t)

                global_orient = torch.from_numpy(np.array(go_list)).float()  # (H, 3)
                body_pose      = torch.from_numpy(np.array(bp_list)).float()  # (H, 69)
                trans          = torch.from_numpy(np.array(tr_list)).float()  # (H, 3)

                joint_pos = np.zeros((HISTORY_SIZE, 29))
                joint_vel = np.zeros((HISTORY_SIZE, 29))

                # body_joints    = smpl_output.joints[:, :24].cpu().numpy()  # (H, 24, 3)
                body_pose_full = torch.cat([global_orient, body_pose], 1)  # (H, 72)

                all_data = process_smpl_joints(body_pose_full[:, 3:], body_pose_full[:, :3])
                global_orient_quat = all_data['global_orient_quat']
                smpl_joints_local = all_data['smpl_joints_local']
                smpl_pose_stacked = all_data['smpl_pose']

                true_frame_indices = np.array(window_abs, dtype=np.int32)

                msg_dict = {
                    "body_quat":   global_orient_quat.cpu().numpy(),
                    "frame_index": true_frame_indices,
                    "smpl_joints": smpl_joints_local.cpu().numpy(),
                    "smpl_pose":   smpl_pose_stacked[:, :63].reshape(-1, 21, 3).cpu().numpy(),
                    "joint_pos":   joint_pos,
                    "joint_vel":   joint_vel,
                    "topic":       args.topic,
                }
                msg = _pack_pose_v3(**msg_dict)
                sock.send(msg)

                try:
                    # for visualization: use global translation 
                    joints_np = all_data['joints'][-1].cpu().numpy()
                    _viz_queue.put_nowait((joints_np, frame_index))
                except queue.Full:
                    pass

                if _playback_started.is_set():
                    frame_index += 1

                elapsed = time.monotonic() - t0
                rem = interval - elapsed
                if rem > 0:
                    time.sleep(rem)

        except KeyboardInterrupt:
            print("\n[smpl_publisher] Stopped.")
        finally:
            sock.close()
            ctx.term()

    threading.Thread(target=_viz_worker, daemon=True).start()
    _run_publisher()


if __name__ == "__main__":
    main()
