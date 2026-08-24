from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_23dof_artifact import (
    canonical_json_bytes,
    sha256_bytes,
)
from gear_sonic.utils.g1_23dof_contract import (
    HARDWARE_23_ACTION_SCALE,
    HARDWARE_23_JOINT_NAMES,
    SOURCE_IL29_EXCLUDED_INDICES,
)
from gear_sonic.utils.g1_23dof_live_shadow import (
    HARDWARE_DEFAULT_Q,
    native_to_hardware,
)
from gear_sonic.utils.g1_23dof_mujoco_sim2sim import (
    APPROVED_CONFIG_SHA256,
    DEFAULT_CONFIG_RELPATH,
    DEFAULT_MJCF_RELPATH,
    PHYSICS_CONTRACT_KIND,
    TRACE_KIND,
    TRACE_SCHEMA_VERSION,
    NeutralReference,
    True23PolicyRuntime,
    _term_major_history,
    _write_trace,
    apply_world_velocity_disturbance,
    approved_offline_input_paths,
    deterministic_disturbance,
    load_sim2sim_config,
    prepare_mujoco_model,
    recompute_metrics,
    run_episode,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / DEFAULT_CONFIG_RELPATH
MJCF_PATH = REPO_ROOT / DEFAULT_MJCF_RELPATH
INIT_CHECKPOINT = REPO_ROOT / "sonic_release/g1_23dof_rev_1_0_init.pt"


@pytest.fixture(scope="module")
def config():
    return load_sim2sim_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def model_bundle(config):
    return prepare_mujoco_model(mjcf_path=MJCF_PATH, config=config)


def test_approved_config_and_exact_model_physics(config, model_bundle):
    module, model, physics = model_bundle
    assert APPROVED_CONFIG_SHA256 == sha256_bytes(CONFIG_PATH.read_bytes())
    assert approved_offline_input_paths(CONFIG_PATH, MJCF_PATH)
    assert (model.nq, model.nv, model.nu) == (30, 29, 23)
    assert physics["kind"] == PHYSICS_CONTRACT_KIND
    assert physics["control_hz"] == 50
    assert physics["control_decimation"] == 10
    assert physics["effort_limit_hardware_nm"][4:6] == [35.0, 35.0]
    assert physics["armature_hardware"][4:6] == [0.00721945, 0.00721945]
    joint_names = [
        module.mj_id2name(model, module.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    ]
    actuator_names = [
        module.mj_id2name(model, module.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]
    assert joint_names == list(HARDWARE_23_JOINT_NAMES)
    assert actuator_names == joint_names
    assert np.all(model.jnt_actfrclimited[1:] == 1)
    assert model.jnt_actfrcrange[5, 1] == 35.0
    assert model.jnt_actfrcrange[6, 1] == 35.0


def test_neutral_reference_is_semantic_fk_not_zero_vector(config, model_bundle):
    module, model, _physics = model_bundle
    reference = NeutralReference(model, module, config["initial_state"])
    encoder_input = reference.encoder_input((1.0, 0.0, 0.0, 0.0))
    assert len(encoder_input) == 267
    # commands.py first converts MJ->native, then selects the first 12
    # ISAACLAB_TO_MUJOCO entries. That reconstructs HW/MJ lower-body order.
    assert encoder_input[:120] == list(HARDWARE_DEFAULT_Q[:12]) * 10
    assert encoder_input[120:240] == [0.0] * 120
    for start in (249, 253, 257):
        assert math.isclose(
            np.linalg.norm(encoder_input[start : start + 4]),
            1.0,
            abs_tol=1e-12,
        )
    assert encoder_input[261:] == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert reference.descriptor()["zeros_only_command"] is False


def test_term_major_history_and_missing_slots_stay_fixed():
    frames = []
    for frame_index in range(10):
        frame = [0.0] * 93
        frame[0:3] = [frame_index + 0.1, frame_index + 0.2, frame_index + 0.3]
        frame[3] = frame_index + 10.0
        frame[32] = frame_index + 20.0
        frame[61] = frame_index + 30.0
        frame[90:93] = [frame_index + 40.1, frame_index + 40.2, frame_index + 40.3]
        frames.append(frame)
    history = _term_major_history(frames)
    assert len(history) == 930
    assert history[:6] == [0.1, 0.2, 0.3, 1.1, 1.2, 1.3]
    assert history[30] == 10.0
    assert history[320] == 20.0
    assert history[610] == 30.0
    assert history[900:903] == [40.1, 40.2, 40.3]
    for block_offset in (30, 320, 610):
        for frame_index in range(10):
            for missing in SOURCE_IL29_EXCLUDED_INDICES:
                assert history[block_offset + frame_index * 29 + missing] == 0.0


def test_disturbance_is_deterministic_and_world_angular_converts_local(config):
    first = deterministic_disturbance(
        config=config,
        seed=1729,
        episode=0,
        scale=1.0,
    )
    assert first == deterministic_disturbance(
        config=config,
        seed=1729,
        episode=0,
        scale=1.0,
    )
    assert first != deterministic_disturbance(
        config=config,
        seed=1729,
        episode=1,
        scale=1.0,
    )
    qvel = np.zeros(29)
    yaw_90_wxyz = (
        math.cos(math.pi / 4),
        0.0,
        0.0,
        math.sin(math.pi / 4),
    )
    apply_world_velocity_disturbance(
        qvel,
        yaw_90_wxyz,
        (1.0, 2.0, 3.0, 1.0, 0.0, 0.0),
    )
    assert np.allclose(qvel[:3], (1.0, 2.0, 3.0), atol=1e-12)
    assert np.allclose(qvel[3:6], (0.0, -1.0, 0.0), atol=1e-12)


class _FixedRuntime:
    def infer(self, encoder_input, proprio_history):
        assert len(encoder_input) == 267
        assert len(proprio_history) == 930
        return np.zeros(64), np.linspace(-0.5, 0.5, 23)


def test_short_closed_loop_uses_native23_mapping(config, model_bundle):
    module, model, _physics = model_bundle
    short = deepcopy(config)
    short["coverage"] = dict(short["coverage"])
    short["coverage"]["steps_per_episode"] = 3
    reference = NeutralReference(model, module, short["initial_state"])
    records = run_episode(
        module=module,
        model=model,
        runtime=_FixedRuntime(),
        reference=reference,
        config=short,
        scenario="nominal",
        seed=1729,
        episode=0,
        disturbance_scale=0.0,
    )
    assert len(records) == 3
    raw_native = np.linspace(-0.5, 0.5, 23)
    expected_target = np.asarray(HARDWARE_DEFAULT_Q) + np.asarray(
        native_to_hardware(raw_native)
    ) * np.asarray(HARDWARE_23_ACTION_SCALE)
    assert np.allclose(
        records[0]["target_position_hardware_rad"],
        expected_target,
        atol=1e-10,
    )
    assert len(records[0]["action_native"]) == 23
    assert len(records[0]["target_position_hardware_rad"]) == 23
    assert records[0]["action_saturated_count"] == 0
    metrics = recompute_metrics(
        records,
        config=short,
        disturbance_scale=0.0,
    )
    assert metrics["record_count"] == 3
    assert metrics["episode_count"] == 1


def test_trace_jsonl_has_no_blank_lines_and_payload_hash(tmp_path):
    record = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "kind": TRACE_KIND,
        "value": 1,
    }
    output = tmp_path / "report.json"
    descriptor = _write_trace(
        output_path=output,
        scenario="nominal",
        seed=1729,
        records=[record, record],
    )
    trace_path = tmp_path / descriptor["file"]
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(line for line in lines)
    parsed = [json.loads(line) for line in lines]
    assert descriptor["payload_sha256"] == sha256_bytes(
        canonical_json_bytes(parsed)
    )


def test_custom_paths_cannot_be_promotion_inputs(tmp_path):
    copied_config = tmp_path / CONFIG_PATH.name
    copied_config.write_bytes(CONFIG_PATH.read_bytes())
    copied_mjcf_dir = tmp_path / "model"
    copied_mjcf_dir.mkdir()
    copied_mjcf = copied_mjcf_dir / MJCF_PATH.name
    copied_mjcf.write_bytes(MJCF_PATH.read_bytes())
    assert not approved_offline_input_paths(copied_config, MJCF_PATH)
    assert not approved_offline_input_paths(CONFIG_PATH, copied_mjcf)


def test_initialization_checkpoint_is_explicitly_nonpromotable():
    runtime = True23PolicyRuntime(checkpoint_path=INIT_CHECKPOINT)
    source = runtime.source_artifact()
    assert runtime.stage == "checkpoint_initialization"
    assert runtime.uses_onnx is False
    assert runtime.promotion_source_complete is False
    assert source["artifact_kind"] == "checkpoint_initialization_diagnostic"
    assert source["inference_runtime"] == "pytorch_checkpoint_cpu"
    assert source["encoder_onnx_sha256"] is None
    assert source["candidate_manifest_sha256"] is None


def test_config_rejects_weakened_coverage(tmp_path):
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    value["coverage"]["episodes_per_seed"] = 1
    path = tmp_path / "weakened.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="66 episodes/scenario"):
        load_sim2sim_config(path)
