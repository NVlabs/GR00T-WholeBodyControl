"""Offline model checks; requires the sim extra and G1 mesh LFS contents."""

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import yaml

from gear_sonic.data.robot_model.instantiation import instantiate_g1_rh56dfx_robot_model
from gear_sonic.data.robot_model.instantiation.g1_inspire_assets import load_asset_paths
from gear_sonic.data.robot_model.supplemental_info.g1.g1_supplemental_info import (
    G1SupplementalInfo,
)
from gear_sonic.utils.hand_control.inspire.config import DEFAULT_MAPPING_PATH, load_mapping
from gear_sonic.utils.hand_control.inspire.contract import JOINT_ORDER, ContractError


@pytest.mark.parametrize("through_sim_entry", [False, True])
def test_custom_urdf_is_used_by_fk_and_sim_entry(tmp_path, monkeypatch, through_sim_entry):
    assets = load_asset_paths()
    document = yaml.safe_load(DEFAULT_MAPPING_PATH.read_text())
    urdf = ET.parse(assets.urdf)
    urdf.getroot().set("name", "custom_inspire_fk")
    for mesh in urdf.findall(".//mesh"):
        mesh.set("filename", str((assets.urdf.parent / mesh.attrib["filename"]).resolve()))
    custom_urdf = tmp_path / "custom.urdf"
    urdf.write(custom_urdf)
    document["simulation"]["urdf_path"] = str(custom_urdf)
    custom_config = tmp_path / "hand.yaml"
    custom_config.write_text(yaml.safe_dump(document, sort_keys=False))
    monkeypatch.chdir(tmp_path)

    if through_sim_entry:
        from gear_sonic.scripts import run_sim_loop as entry

        captured = {}

        def capture_wrapper(robot_model, env_name, config, **kwargs):
            captured.update(robot=robot_model, scene=config["ROBOT_SCENE"])
            return SimpleNamespace(sim=object())

        # Run the actual entry and FK constructor without starting DDS or a viewer.
        monkeypatch.setattr(entry, "SimWrapper", capture_wrapper)
        monkeypatch.setattr(entry.SimulatorFactory, "start_simulator", lambda *a, **kw: None)
        entry.main(entry.ArgsConfig(interface="sim", hand="inspire", hand_config=custom_config))
        robot = captured["robot"]
        assert Path(captured["scene"]) == load_asset_paths(custom_config).scene
    else:
        robot = instantiate_g1_rh56dfx_robot_model(config_path=custom_config)

    assert robot.pinocchio_wrapper.model.name == "custom_inspire_fk"
    assert robot.num_joints == 41


def test_robot_groups_limits_and_paths_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    robot = instantiate_g1_rh56dfx_robot_model()
    mapping = load_mapping()
    assert robot.num_joints == 41
    body = robot.get_body_actuated_joint_indices()
    assert len(body) == 29
    assert [robot.joint_names[i] for i in body] == G1SupplementalInfo().body_actuated_joints
    hands = []
    for side, joints in (("left", mapping.left), ("right", mapping.right)):
        indices = robot.get_hand_actuated_joint_indices(side)
        assert len(indices) == 6
        assert [robot.joint_names[i] for i in indices] == [joint.urdf_joint for joint in joints]
        assert robot.get_joint_group_indices(f"{side}_hand") == sorted(indices)
        np.testing.assert_allclose(robot.lower_joint_limits[indices], [j.lower for j in joints])
        np.testing.assert_allclose(robot.upper_joint_limits[indices], [j.upper for j in joints])
        assert tuple(j.name for j in joints) == JOINT_ORDER
        hands.extend(indices)
    assert set(body).isdisjoint(hands)
    assert sorted(body + hands) == list(range(41))


@pytest.mark.parametrize("pose", ["zero", "default", "bent"])
def test_wrist_to_palm_transform_matches_mujoco(pose):
    robot = instantiate_g1_rh56dfx_robot_model()
    q = robot.q_zero if pose == "zero" else robot.default_body_pose.copy()
    if pose == "bent":
        for side in ("left", "right"):
            q[robot.dof_index(f"{side}_elbow_joint")] = 0.8
            q[robot.dof_index(f"{side}_wrist_yaw_joint")] = 0.2
            q[robot.get_hand_actuated_joint_indices(side)] = [0.4, 0.5, 0.6, 0.7, 0.2, 0.3]
    robot.cache_forward_kinematics(q, auto_clip=False)
    model = mujoco.MjModel.from_xml_path(str(load_asset_paths().model))
    data = mujoco.MjData(model)
    for name in robot.joint_names:
        data.joint(name).qpos[0] = q[robot.dof_index(name)]
    mujoco.mj_forward(model, data)
    for side in ("left", "right"):
        wrist, palm = f"{side}_wrist_roll_link", f"{side}_inspire_base"
        pin_relative = robot.frame_placement(wrist).inverse() * robot.frame_placement(palm)
        wrist_rotation = data.body(wrist).xmat.reshape(3, 3)
        mj_translation = wrist_rotation.T @ (data.body(palm).xpos - data.body(wrist).xpos)
        mj_rotation = wrist_rotation.T @ data.body(palm).xmat.reshape(3, 3)
        np.testing.assert_allclose(pin_relative.translation, mj_translation, atol=1e-4)
        np.testing.assert_allclose(pin_relative.rotation, mj_rotation, atol=1e-4)


@pytest.mark.parametrize(
    "defect", ["order", "duplicate_joint", "nonfinite", "limit", "domain_index"]
)
def test_invalid_mapping_is_rejected(tmp_path, defect):
    document = yaml.safe_load(DEFAULT_MAPPING_PATH.read_text())
    left = document["joints"]["left"]
    if defect == "order":
        document["joint_order"].reverse()
    elif defect == "duplicate_joint":
        left["index"]["urdf_joint"] = left["middle"]["urdf_joint"]
    elif defect == "nonfinite":
        left["index"]["scale"] = float("nan")
    elif defect == "limit":
        left["index"]["position_limit"] = [0, 2]
    else:
        left["index"]["domain_index"] = 0
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ContractError):
        load_mapping(path)


@pytest.mark.parametrize("mesh_contents", [None, b"version https://git-lfs.github.com/spec/v1\n"])
def test_missing_or_pointer_mesh_has_actionable_error(tmp_path, mesh_contents):
    # Tiny asset fixture exercises diagnostics only; it is never used as a robot.
    mesh = tmp_path / "missing.STL"
    if mesh_contents is not None:
        mesh.write_bytes(mesh_contents)
    urdf = tmp_path / "test.urdf"
    urdf.write_text('<robot name="test"><mesh filename="missing.STL"/></robot>')
    config = tmp_path / "paths.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "simulation": {
                    "model_path": str(urdf),
                    "scene_path": str(urdf),
                    "urdf_path": str(urdf),
                }
            }
        )
    )
    expected = FileNotFoundError if mesh_contents is None else ValueError
    message = "Missing Inspire mesh" if mesh_contents is None else "git lfs pull"
    with pytest.raises(expected, match=message):
        load_asset_paths(config)
