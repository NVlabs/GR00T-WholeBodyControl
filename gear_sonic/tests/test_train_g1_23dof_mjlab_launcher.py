"""Pure launcher tests that do not require MJLab or a CUDA device."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gear_sonic.scripts.train_g1_23dof_mjlab import (
    DEFAULT_SAMPLE_NPZ,
    NORMAL_REFERENCE_PROFILE,
    _source_files,
    build_parser,
    motion_report,
)


def _write_motion(path: Path, *, frames: int = 47, dof: int = 23) -> None:
    np.savez(
        path,
        fps=np.asarray([50], dtype=np.int64),
        joint_pos=np.zeros((frames, dof), dtype=np.float32),
        joint_vel=np.zeros((frames, dof), dtype=np.float32),
        body_pos_w=np.zeros((frames, 24, 3), dtype=np.float32),
        body_quat_w=np.zeros((frames, 24, 4), dtype=np.float32),
        body_lin_vel_w=np.zeros((frames, 24, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((frames, 24, 3), dtype=np.float32),
    )


def test_motion_report_accepts_exact_50hz_true23_npz(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    _write_motion(motion_path)

    report = motion_report(motion_path)

    assert report["ok"] is True
    assert report["frames"] == 47
    assert report["fps"] == 50.0
    assert report["shapes"]["joint_pos"] == [47, 23]


def test_motion_report_rejects_29dof_or_short_motion(tmp_path: Path) -> None:
    motion_path = tmp_path / "bad_motion.npz"
    _write_motion(motion_path, frames=46, dof=29)

    report = motion_report(motion_path)

    assert report["ok"] is False
    assert any("at least 47" in problem for problem in report["problems"])
    assert any("joint_pos" in problem for problem in report["problems"])
    assert any("joint_vel" in problem for problem in report["problems"])


def test_launcher_low_vram_defaults_are_explicit() -> None:
    parser = build_parser()

    smoke = parser.parse_args(["smoke"])
    train = parser.parse_args(["train"])

    assert smoke.num_envs == 4
    assert smoke.iterations == 2
    assert smoke.reference_profile == NORMAL_REFERENCE_PROFILE
    assert smoke.motion_file == DEFAULT_SAMPLE_NPZ
    assert train.num_envs == 128
    assert train.iterations == 10_001


def test_lineage_source_selection_includes_exact_boundary() -> None:
    files = _source_files()

    assert "gear_sonic/envs/mjlab/sonic_true23.py" in files
    assert "gear_sonic/trl/mjlab/true23_actor.py" in files
    assert "gear_sonic/trl/mjlab/runner.py" in files
    assert "gear_sonic/utils/g1_23dof_contract.py" in files
    assert all(path.is_file() for path in files.values())
