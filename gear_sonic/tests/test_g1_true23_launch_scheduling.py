from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPERS = (
    REPO_ROOT / "install_scripts/run_g1_true23_v12_promoted_shadow.sh",
    REPO_ROOT / "install_scripts/run_g1_true23_v12_stage1_gantry.sh",
)


def test_v12_launchers_isolate_control_from_publisher_without_realtime() -> None:
    for wrapper in WRAPPERS:
        source = wrapper.read_text(encoding="utf-8")
        assert 'control_cpu_set="0-3"' in source
        assert 'service_cpu_set="4,5"' in source
        assert 'publisher_cpu_set="6-15"' in source
        assert 'expected_online_cpus="0-15"' in source
        assert "RoboticsServiceProcess" in source
        assert "8654b4f3552e36e1223f6589491ebe6c82002a07a09520fae7f257465ce0bbbc" in source
        assert '"$cls" != "TS"' in source
        assert 'setsid taskset -c "$control_cpu_set"' in source
        assert 'setsid taskset -c "$publisher_cpu_set"' in source
        assert '"$python" "$publisher"' in source
        assert "chrt" not in source
        assert '[[ "$actual_pgid" == "$leader_pid" ]]' in source
        assert 'kill -KILL -- "-$group_pid"' in source
        assert "stream_g1_23dof_pico_raw_worker.py" in source
        assert "assert_no_runtime_tree" in source
        assert "for _ in {1..150}" in source


def test_v12_launchers_preserve_40ms_lowstate_gates() -> None:
    shadow_source = (
        REPO_ROOT
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/"
        "g1_true23_live_shadow.cpp"
    ).read_text(encoding="utf-8")
    active_core = (
        REPO_ROOT
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/"
        "true23_active_gantry_core.hpp"
    ).read_text(encoding="utf-8")
    assert "kStateFreshnessNs = 40'000'000" in shadow_source
    assert "kStateFreshnessNs = 40'000'000" in active_core
