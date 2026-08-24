from __future__ import annotations

import json

import pytest

from gear_sonic.scripts import (
    validate_g1_23dof_xr24_standing_capture as standing_cli,
)
from gear_sonic.tests.test_g1_23dof_pico_retargeted_producer import _capture


def test_standing_capture_fixture_passes_opening_hold() -> None:
    report = standing_cli.assess_standing_capture(_capture(12))

    assert report["pass"] is True
    assert report["status"] == "neutral_standing_acquisition_pass"
    assert report["hold_frame_count"] == 10
    assert report["failed_frames"] == []
    assert report["authorization"]["robot_channel_opened"] is False
    assert report["authorization"]["actuation_authorized"] is False


def test_seated_capture_fixture_is_rejected_for_every_hold_frame() -> None:
    capture = _capture(12)
    for frame in capture["frames"]:
        poses = frame["body_poses"]
        poses[0][1] = 0.45
        poses[1][:3] = [-0.10, 0.42, 0.00]
        poses[2][:3] = [0.10, 0.42, 0.00]
        poses[4][:3] = [-0.10, 0.50, 0.35]
        poses[5][:3] = [0.10, 0.50, 0.35]

    report = standing_cli.assess_standing_capture(capture)

    assert report["pass"] is False
    assert report["status"] == "neutral_standing_acquisition_reject"
    assert report["checks"]["pelvis_above_knees_m"] is False
    assert len(report["failed_frames"]) == 10
    assert report["next_step"] == "stand_upright_recalibrate_and_recapture"


def test_cli_exit_code_distinguishes_pass_and_reject(
    tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    capture_path = tmp_path / "capture.json"
    capture_path.write_text(json.dumps(_capture(12)), encoding="utf-8")

    assert standing_cli.main(["--capture", str(capture_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pass"] is True


def test_capture_too_short_for_hold_fails_closed() -> None:
    with pytest.raises(ValueError, match="too short"):
        standing_cli.assess_standing_capture(_capture(5))
