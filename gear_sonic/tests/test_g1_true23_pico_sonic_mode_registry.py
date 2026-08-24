from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.g1_true23_clean_mujoco_teleop import (
    RELEASED_RETAINED_KD,
    RELEASED_RETAINED_KP,
    CleanSonicPolicy,
    CleanTrue23MujocoController,
    motion_reference_terms,
    pico_reference_policy_history,
    reference_initial_state,
)
from gear_sonic.utils.g1_true23_pico_sonic_mode_registry import (
    PROFILE_NAMES,
    REGISTRY_RELATIVE_PATH,
    load_native23_mode_profile,
)
from gear_sonic.utils.g1_true23_sonic_library_replay import _reference_policy_frame

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("profile_name", PROFILE_NAMES)
def test_registry_loads_only_proven_native23_profiles(profile_name: str) -> None:
    profile = load_native23_mode_profile(ROOT, profile_name)
    assert profile.name == profile_name
    assert profile.encoder_path.is_file()
    assert profile.decoder_path.is_file()
    assert profile.evidence_paths
    assert profile.sustained_deep_crouch_authorized is False


def test_pico_profile_is_simulator_only_and_builds_exact_policy() -> None:
    profile = load_native23_mode_profile(ROOT, "pico_fullbody")
    assert profile.simulator_modes == ("saved", "zmq")
    assert profile.live_transport_proven is True
    assert profile.live_headset_source_proven is False
    assert profile.minimum_base_height_m == 0.30
    assert profile.maximum_base_tilt_rad == 1.0
    policy = CleanSonicPolicy(
        profile.encoder_path,
        profile.decoder_path,
        expected_decoder_sha256=profile.decoder_sha256,
    )
    assert policy.decoder.get_outputs()[0].shape == [1, 23]


def test_authentic_pico_walk_profile_is_distinct_and_transport_proven() -> None:
    profile = load_native23_mode_profile(ROOT, "pico_internet_fullbody_walk")
    assert profile.simulator_modes == ("saved", "zmq")
    assert profile.live_transport_proven is True
    assert profile.live_headset_source_proven is False
    assert profile.minimum_base_height_m == 0.30
    assert profile.maximum_base_tilt_rad == 1.0
    assert len(profile.evidence_paths) == 8
    assert profile.decoder_sha256 == "f66408ae9a10720a3aff717269d0e2a4e07ab471e449a6fe8f5bae5e8607ef63"
    policy = CleanSonicPolicy(
        profile.encoder_path,
        profile.decoder_path,
        expected_decoder_sha256=profile.decoder_sha256,
    )
    assert policy.decoder.get_outputs()[0].shape == [1, 23]


def test_registry_rejects_29dof_or_hardware_claim(tmp_path: Path) -> None:
    source = json.loads((ROOT / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    source["physical_dof"] = 29
    path = tmp_path / REGISTRY_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError, match="native23 boundary"):
        load_native23_mode_profile(tmp_path, "pico_fullbody")


def test_pico_profile_selects_exact_passing_physical_gains() -> None:
    profile = load_native23_mode_profile(ROOT, "pico_fullbody")
    controller = CleanTrue23MujocoController(
        model_path=ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        policy=CleanSonicPolicy(
            profile.encoder_path,
            profile.decoder_path,
            expected_decoder_sha256=profile.decoder_sha256,
        ),
    )
    controller.use_released_retained_gains()
    assert (controller.physics.kp == RELEASED_RETAINED_KP).all()
    assert (controller.physics.kd == RELEASED_RETAINED_KD).all()


def test_pico_packet_rebuilds_exact_passing_q0_q9_policy_history() -> None:
    packet_bundle = json.loads(
        (
            ROOT / "artifacts/g1_true23/pico_saved_clip_replay_v1/upright/"
            "causal_packets_neutral_calibrated_v1.json"
        ).read_text(encoding="utf-8")
    )
    with np.load(
        ROOT / "artifacts/g1_true23/pico_fullbody_motion_v1/upright.true23.npz",
        allow_pickle=False,
    ) as archive:
        motion = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    actual = pico_reference_policy_history(packet_bundle["robot_independent_reference_packets"][0])
    expected = [_reference_policy_frame(motion, index) for index in range(10)]
    assert np.array_equal(np.asarray(actual), np.asarray(expected))
    packets = packet_bundle["robot_independent_reference_packets"]
    q10, _ = reference_initial_state(packets[0])
    _, qd10 = reference_initial_state(packets[1])
    assert np.array_equal(q10.astype(np.float32), motion["joint_pos"][10])
    assert np.array_equal(qd10.astype(np.float32), motion["joint_vel"][10])

    profile = load_native23_mode_profile(ROOT, "pico_fullbody")
    controller = CleanTrue23MujocoController(
        model_path=ROOT / "gear_sonic/data/robots/g1/g1_23dof_rev_1_0.xml",
        physics_path=ROOT / "gear_sonic/config/sim_validation/g1_23dof_mujoco_sim2sim.json",
        policy=CleanSonicPolicy(
            profile.encoder_path,
            profile.decoder_path,
            expected_decoder_sha256=profile.decoder_sha256,
        ),
    )
    retargeted = controller.retarget_pico_reference_packet(packets[0])
    expected_packet = motion_reference_terms(motion, 9)
    assert np.allclose(
        retargeted["vr_3point_local_target"],
        expected_packet["vr_3point_local_target"],
        rtol=0.0,
        atol=1.0e-7,
    )
    assert np.allclose(
        retargeted["vr_3point_local_orn_target"],
        expected_packet["vr_3point_local_orn_target"],
        rtol=0.0,
        atol=1.0e-7,
    )
