import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.joint_mapping import (
    DEX3_DDS_NAMES,
    G1_ISAACLAB_NAMES,
    G1_MUJOCO_NAMES,
    NOMINAL_G1_MUJOCO,
    dex3_dds_to_robot_model,
    reorder_by_name,
)

EXPECTED_G1_MUJOCO_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

EXPECTED_G1_ISAACLAB_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

EXPECTED_ISAACLAB_SOURCE_INDICES = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)

EXPECTED_NOMINAL_G1_MUJOCO = np.array(
    (
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        -0.312,
        0.0,
        0.0,
        0.669,
        -0.363,
        0.0,
        0.0,
        0.0,
        0.0,
        0.2,
        0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        0.2,
        -0.2,
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
    ),
    dtype=np.float64,
)

EXPECTED_DEX3_DDS_NAMES = {
    "left": ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"),
    "right": ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"),
}


def test_nominal_g1_mujoco_has_exact_absolute_targets() -> None:
    assert NOMINAL_G1_MUJOCO.shape == (29,)
    assert NOMINAL_G1_MUJOCO.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(NOMINAL_G1_MUJOCO, EXPECTED_NOMINAL_G1_MUJOCO)
    assert not NOMINAL_G1_MUJOCO.flags.writeable


def test_g1_name_orders_are_exact_unique_permutations() -> None:
    assert G1_MUJOCO_NAMES == EXPECTED_G1_MUJOCO_NAMES
    assert G1_ISAACLAB_NAMES == EXPECTED_G1_ISAACLAB_NAMES
    assert len(G1_MUJOCO_NAMES) == len(set(G1_MUJOCO_NAMES)) == 29
    assert len(G1_ISAACLAB_NAMES) == len(set(G1_ISAACLAB_NAMES)) == 29
    assert set(G1_MUJOCO_NAMES) == set(G1_ISAACLAB_NAMES)


def test_reorder_by_name_matches_exact_isaaclab_source_indices() -> None:
    reordered = reorder_by_name(np.arange(29), G1_MUJOCO_NAMES, G1_ISAACLAB_NAMES)

    np.testing.assert_array_equal(reordered, EXPECTED_ISAACLAB_SOURCE_INDICES)


def test_reorder_by_name_preserves_leading_dimensions_and_dtype() -> None:
    values = np.arange(2 * 3 * 29, dtype=np.float32).reshape(2, 3, 29)

    reordered = reorder_by_name(values, G1_MUJOCO_NAMES, G1_ISAACLAB_NAMES)

    assert reordered.shape == (2, 3, 29)
    assert reordered.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(reordered, values[..., EXPECTED_ISAACLAB_SOURCE_INDICES])


@pytest.mark.parametrize("values", [np.arange(28), np.array(1.0), np.empty((2, 30))])
def test_reorder_by_name_rejects_wrong_final_dimension(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"values must have final dimension 29; got shape"):
        reorder_by_name(values, G1_MUJOCO_NAMES, G1_ISAACLAB_NAMES)


def test_reorder_by_name_rejects_duplicate_source_names() -> None:
    with pytest.raises(ValueError, match=r"duplicate source names: \['joint_a'\]"):
        reorder_by_name(np.arange(3), ("joint_a", "joint_b", "joint_a"), ("joint_a", "joint_b", "joint_c"))


def test_reorder_by_name_rejects_duplicate_target_names() -> None:
    with pytest.raises(ValueError, match=r"duplicate target names: \['joint_b'\]"):
        reorder_by_name(np.arange(3), ("joint_a", "joint_b", "joint_c"), ("joint_a", "joint_b", "joint_b"))


def test_reorder_by_name_reports_sorted_missing_and_extra_names() -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"source and target name sets differ: "
            r"missing from source=\['joint_c', 'joint_d'\], "
            r"extra in source=\['joint_x', 'joint_z'\]"
        ),
    ):
        reorder_by_name(
            np.arange(3),
            ("joint_z", "joint_a", "joint_x"),
            ("joint_d", "joint_a", "joint_c"),
        )


def test_dex3_dds_orders_are_explicitly_asymmetric() -> None:
    assert dict(DEX3_DDS_NAMES) == EXPECTED_DEX3_DDS_NAMES
    assert DEX3_DDS_NAMES["left"][3:] == ("middle_0", "middle_1", "index_0", "index_1")
    assert DEX3_DDS_NAMES["right"][3:] == ("index_0", "index_1", "middle_0", "middle_1")
    assert DEX3_DDS_NAMES["left"] != DEX3_DDS_NAMES["right"]


@pytest.mark.parametrize("side", ["left", "right"])
def test_dex3_one_hot_indices_map_to_exact_robot_model_joint(side: str) -> None:
    source_order = EXPECTED_DEX3_DDS_NAMES[side]
    expected_keys = tuple(f"{side}_hand_{semantic}_joint" for semantic in source_order)

    for source_index, expected_key in enumerate(expected_keys):
        raw = np.zeros(7, dtype=np.int16)
        raw[source_index] = 1

        mapped = dex3_dds_to_robot_model(side, raw)

        assert tuple(mapped) == expected_keys
        assert mapped[expected_key] == 1
        assert all(value == 0 for key, value in mapped.items() if key != expected_key)


def test_dex3_dds_to_robot_model_preserves_numpy_scalar_dtype_and_values() -> None:
    raw = np.linspace(-0.75, 0.75, 7, dtype=np.float32)

    mapped = dex3_dds_to_robot_model("right", raw)

    np.testing.assert_array_equal(np.array(tuple(mapped.values())), raw)
    assert all(isinstance(value, np.float32) for value in mapped.values())


@pytest.mark.parametrize("side", ["LEFT", "", "middle", None, 0])
def test_dex3_dds_to_robot_model_rejects_unknown_side(side: object) -> None:
    with pytest.raises(ValueError, match=r"side must be exactly 'left' or 'right'"):
        dex3_dds_to_robot_model(side, np.zeros(7))


@pytest.mark.parametrize(
    "raw",
    [np.zeros(6), np.zeros(8), np.zeros((1, 7)), np.zeros((7, 1)), np.array(0.0)],
)
def test_dex3_dds_to_robot_model_requires_exact_vector_shape(raw: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"raw must have shape \(7,\); got"):
        dex3_dds_to_robot_model("left", raw)
