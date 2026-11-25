import numpy as np
from scipy.spatial.transform import Rotation as R

from gear_sonic.scripts import capture_body_skeleton as capture

N = capture.NUM_JOINTS
SAMPLES = 40


def _save(path, rots, valid=None, label="x"):
    """Write a static capture whose every sample holds ``rots`` (one Rotation per joint)."""
    rng = np.random.default_rng(0)
    quat = np.stack([rots.as_quat()] * SAMPLES)
    # Natural sub-degree jitter, so the capture reads as held still, not frozen.
    quat = (
        (
            R.from_rotvec(rng.normal(scale=0.002, size=(SAMPLES * N, 3)))
            * R.from_quat(quat.reshape(-1, 4))
        )
        .as_quat()
        .reshape(SAMPLES, N, 4)
    )
    if valid is None:
        valid = np.ones((SAMPLES, N), dtype=bool)
    np.savez(
        path,
        pos=np.zeros((SAMPLES, N, 3)),
        quat=quat,
        valid=valid,
        t=np.arange(SAMPLES, dtype=float),
        label=label,
    )
    return str(path)


def test_a_fixed_joint_local_correction_reads_the_same_in_every_pose(tmp_path):
    """B = A * C for one fixed C per joint: the offset must come back as C.

    Measured in the world frame (B * A^-1) it would come back as A C A^-1,
    which changes with the pose, and a valid fixed correction would look
    pose-dependent.
    """
    correction = R.from_euler("xyz", [30, -50, 70], degrees=True)
    for seed in (1, 2):
        pose = R.random(N, random_state=seed)
        offsets = capture.rest_pose(
            _save(tmp_path / f"a{seed}.npz", pose),
            _save(tmp_path / f"b{seed}.npz", pose * correction),
        )
        for j, off in offsets.items():
            assert np.degrees((off.inv() * correction).magnitude()) < 1.0, j


def test_samples_marked_invalid_are_left_out(tmp_path, capsys):
    pose = R.random(N, random_state=3)
    path_a = _save(tmp_path / "a.npz", pose)

    # Same pose in B, except a quarter of one joint's samples are garbage the
    # tracker flagged invalid.
    path_b = _save(tmp_path / "b.npz", pose)
    data = dict(np.load(path_b))
    bad = np.arange(0, SAMPLES, 4)
    data["quat"][bad, 5] = R.random(len(bad), random_state=4).as_quat()
    data["valid"][bad, 5] = False
    np.savez(path_b, **data)

    offsets = capture.rest_pose(path_a, path_b)

    assert "TOO MUCH MOTION" not in capsys.readouterr().out
    assert np.degrees(offsets[5].magnitude()) < 1.0
