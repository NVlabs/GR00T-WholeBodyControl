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

    assert revealed == ['Type of "TrackerSnapshot().full_body" is "FullBodyPose | None"']
    wrong_lines = {i for i, line in enumerate(probe.read_text().splitlines()) if "# wrong" in line}
    assert len(wrong_lines) == 5
    assert {d["range"]["start"]["line"] for d in errors} == wrong_lines
    assert all("cannot be assigned" in d["message"] for d in errors)
