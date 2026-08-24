from pathlib import Path


PATCH_PATH = (
    Path(__file__).resolve().parents[2]
    / "install_scripts"
    / "xrobotoolkit_pico_health_v1.patch"
)


def test_xr24_role_array_index_is_canonical_without_weakening_health_gates():
    patch = PATCH_PATH.read_text(encoding="utf-8")

    assert "roleDatas array order as the canonical XR24 joint" in patch
    assert "roleData.role != expectedRoleIndex" not in patch
    assert "trackingData.roleDatas[expectedRoleIndex]" in patch

    # Keep exact XR24 cardinality and every existing numeric validity gate.
    assert "bodyRoleCount == (int)BodyTrackerRole.ROLE_NUM" in patch
    assert "trackingData.roleDatas.Length" in patch
    assert "pose.TimeStamp < 0" in patch
    for field in (
        "pose.PosX",
        "pose.PosY",
        "pose.PosZ",
        "pose.RotQx",
        "pose.RotQy",
        "pose.RotQz",
        "pose.RotQw",
        "velocity[component]",
        "acceleration[component]",
        "angularVelocity[component]",
        "angularAcceleration[component]",
    ):
        assert f"!IsFinite({field})" in patch
    assert "quaternionNormSquared < 0.25" in patch
    assert "quaternionNormSquared > 2.25" in patch
    assert "connectedBandCount >= 2" in patch
    assert "bodyDataResult == 0" in patch
    assert "sampleSequence" in patch
