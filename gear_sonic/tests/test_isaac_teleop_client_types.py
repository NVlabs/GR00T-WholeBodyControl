import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

_SDK = next(
    (name for name in ("isaaccapture", "isaacteleop") if importlib.util.find_spec(name)), None
)

_PAYLOAD_TYPING_PROBE = """
from gear_sonic.utils.teleop.isaac_teleop_client import TrackerSnapshot
from {sdk}.schema import ControllerSnapshot, FullBodyPose, HandPose, HeadPose

def real(b: FullBodyPose, h: HeadPose, hand: HandPose, c: ControllerSnapshot) -> None:
    TrackerSnapshot(full_body=b, head=h, left_hand=hand, left_controller=c)
    reveal_type(TrackerSnapshot().full_body)

def wrong(h: HeadPose, hand: HandPose) -> None:
    TrackerSnapshot(full_body=123)  # wrong
    TrackerSnapshot(head="incorrect")  # wrong
    TrackerSnapshot(full_body=h)  # wrong
    TrackerSnapshot(full_body=hand)  # wrong
    TrackerSnapshot(left_controller=h)  # wrong

def wrong_body_for_hand(b: FullBodyPose) -> None:
    TrackerSnapshot(left_hand=b)  # wrong

def nested(s: TrackerSnapshot) -> None:
    assert s.left_controller and s.head and s.full_body and s.left_hand
    reveal_type(s.left_controller.inputs.trigger_value)
    reveal_type(s.left_controller.aim_pose.pose.orientation.w)
    reveal_type(s.head.pose.position.x)
    reveal_type(s.full_body.joints.joints(0).pose.position.x)
    reveal_type(s.left_hand.joints.poses(0).radius)
    trigger: str = s.left_controller.inputs.trigger_value  # wrong
    tracked: str = s.head.is_tracked  # wrong
    joint_x: str = s.full_body.joints.joints(0).pose.position.x  # wrong
    radius: str = s.left_hand.joints.poses(0).radius  # wrong
"""


@pytest.mark.skipif(_SDK is None, reason="no IsaacCapture SDK is installed")
@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright is not installed")
def test_snapshot_fields_reject_the_wrong_payload_on_the_installed_sdk(tmp_path):
    """The payload types hold on whichever SDK is installed, with nothing Unknown.

    Real payloads type-check, and each wrong one -- a number, a string, one
    payload passed for another -- is an error rather than silently accepted.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(_PAYLOAD_TYPING_PROBE.format(sdk=_SDK))
    result = subprocess.run(
        ["pyright", "--outputjson", "--pythonpath", sys.executable, str(probe)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics = json.loads(result.stdout)["generalDiagnostics"]
    errors = [d for d in diagnostics if d["severity"] == "error"]
    revealed = [d["message"] for d in diagnostics if d["severity"] == "information"]

    assert revealed[0] == 'Type of "TrackerSnapshot().full_body" is "FullBodyPose | None"'
    # Nested reads resolve down to the scalar, rather than stopping at object.
    assert len(revealed) == 6
    assert all(message.endswith('is "float"') for message in revealed[1:])
    wrong_lines = {i for i, line in enumerate(probe.read_text().splitlines()) if "# wrong" in line}
    assert len(wrong_lines) == 10
    assert {d["range"]["start"]["line"] for d in errors} == wrong_lines
    assert all("assign" in d["message"] for d in errors)
