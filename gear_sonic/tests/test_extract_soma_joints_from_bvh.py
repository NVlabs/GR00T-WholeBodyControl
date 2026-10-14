from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pytest

from gear_sonic.data_process.extract_soma_joints_from_bvh import (
    SOMA_JOINTS,
    compute_fk_selected,
    parse_bvh,
)


def write_bvh(path, names, offsets, root_frames, end_site="End Site"):
    lines = [
        "HIERARCHY",
        f"ROOT {names[0]}",
        "{",
        "OFFSET 0 0 0",
        "CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation",
    ]
    for name, offset in zip(names[1:], offsets[1:]):
        lines.extend(
            [
                f"JOINT {name}",
                "{",
                "OFFSET " + " ".join(map(str, offset)),
                "CHANNELS 3 Zrotation Xrotation Yrotation",
                end_site,
                "{",
                "OFFSET " + " ".join(str(value * 10) for value in offset),
                "}",
                "}",
            ]
        )
    lines.extend(["}", "MOTION", f"Frames: {len(root_frames)}", "Frame Time: 0.0333333333"])
    for frame in root_frames:
        lines.append(" ".join(map(str, [*frame, *([0] * (3 * (len(names) - 1)))])))
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("end_site", ["End Site", "End\tSite"])
def test_end_sites_preserve_joint_offsets_and_parent_transforms(tmp_path, end_site):
    path = tmp_path / "branches.bvh"
    names = ["Hips", "LeftArm", "RightArm"]
    offsets = [[0, 0, 0], [1, 0, 0], [0, 2, 0]]
    write_bvh(
        path,
        names,
        offsets,
        [[0, 0, 0, 0, 0, 0], [10, 20, 30, 90, 0, 0]],
        end_site,
    )

    joints, channels, motion_data, n_frames, frame_time = parse_bvh(path)
    assert [joint["name"] for joint in joints] == names
    assert [joint["parent_idx"] for joint in joints] == [-1, 0, 0]
    np.testing.assert_array_equal([joint["offset"] for joint in joints], offsets)
    assert n_frames == 2
    assert frame_time == pytest.approx(1 / 30)
    assert motion_data.shape == (2, 12)
    assert len(channels) == 12

    positions, _ = compute_fk_selected(joints, channels, motion_data, names)
    np.testing.assert_allclose(
        positions,
        [
            [[0, 0, 0], [1, 0, 0], [0, 2, 0]],
            [[10, 20, 30], [10, 21, 30], [8, 20, 30]],
        ],
        atol=1e-6,
    )


def test_cli_preserves_soma_joint_positions_with_end_sites(tmp_path):
    input_dir = tmp_path / "session"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    offsets = np.array([[i, 2 * i, 0] for i in range(len(SOMA_JOINTS))])
    write_bvh(
        input_dir / "motion.bvh",
        SOMA_JOINTS,
        offsets,
        [[100, 200, 300, 0, 0, 0], [400, 500, 600, 0, 0, 0]],
    )
    script = Path(__file__).resolve().parents[1] / "data_process" / "extract_soma_joints_from_bvh.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--fps",
            "30",
            "--num_workers",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "1 converted, 0 failed" in result.stdout
    motion = joblib.load(output_dir / "session" / "motion.pkl")["motion"]
    expected_positions = offsets[:, [0, 2, 1]] / 100
    np.testing.assert_allclose(motion["soma_joints"], [expected_positions] * 2, atol=1e-6)
    np.testing.assert_allclose(motion["soma_transl"], [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(motion["soma_root_quat"], [[1, 0, 0, 0]] * 2)
    assert motion["joint_names"] == SOMA_JOINTS
    assert motion["fps"] == 30
