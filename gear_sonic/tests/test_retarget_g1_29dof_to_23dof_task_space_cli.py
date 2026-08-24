from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.scripts import retarget_g1_29dof_to_23dof_task_space as cli


def _write_large_file(path: Path, byte: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(byte * 2048)


def test_manifest_resolves_relative_sources_and_preserves_provenance(tmp_path: Path) -> None:
    manifest = tmp_path / "selection" / "heldout.json"
    manifest.parent.mkdir()
    payload = {
        "schema": cli.MANIFEST_SCHEMA,
        "source_root": "../corpus",
        "fps_source": 120,
        "clips": [
            {
                "clip_id": "walk_001",
                "source_csv": "session/walk.csv",
                "category": "walk",
            },
            {
                "clip_id": "turn_001",
                "source_csv": "session/turn.csv",
                "category": "turn",
                "fps_source": 60,
            },
        ],
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    specs, provenance = cli._load_manifest_specs(
        manifest,
        cli_source_root=None,
        default_fps_source=30.0,
    )

    assert [spec.clip_id for spec in specs] == ["walk_001", "turn_001"]
    assert specs[0].csv_path == (tmp_path / "corpus/session/walk.csv").resolve()
    assert specs[0].category == "walk"
    assert specs[0].fps_source == 120.0
    assert specs[1].fps_source == 60.0
    assert provenance["manifest_sha256"] == cli._sha256(manifest)
    assert provenance["qualification_categories"] == list(cli.QUALIFICATION_CATEGORIES)
    assert specs[0].qualification_categories == cli.QUALIFICATION_CATEGORIES


def test_manifest_declared_categories_bind_selection_and_cache_provenance(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "heldout.json"
    manifest.write_text(
        json.dumps(
            {
                "qualification_categories": ["idle"],
                "clips": [
                    {
                        "clip_id": "idle_001",
                        "source_csv": "idle.csv",
                        "category": "idle",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    specs, selection = cli._load_manifest_specs(
        manifest, cli_source_root=None, default_fps_source=120.0
    )
    expected = cli._cache_expectations(
        spec=specs[0],
        outputs=cli._output_triplet(tmp_path / "output", specs[0].clip_id),
        source_hash="a" * 64,
        source_model_path=tmp_path / "source.xml",
        source_model_hash="b" * 64,
        target_model_path=tmp_path / "target.xml",
        target_model_hash="c" * 64,
        fps_target=50.0,
        retarget_config={},
        retiming_config={"enabled": False},
    )

    assert selection["qualification_categories"] == ["idle"]
    assert expected["qualification_categories"] == ["idle"]


def test_default_six_categories_preserve_legacy_cache_provenance(
    tmp_path: Path,
) -> None:
    spec = cli.ClipSpec("idle_001", tmp_path / "idle.csv", "idle", 120.0)
    expected = cli._cache_expectations(
        spec=spec,
        outputs=cli._output_triplet(tmp_path / "output", spec.clip_id),
        source_hash="a" * 64,
        source_model_path=tmp_path / "source.xml",
        source_model_hash="b" * 64,
        target_model_path=tmp_path / "target.xml",
        target_model_hash="c" * 64,
        fps_target=50.0,
        retarget_config={},
        retiming_config={"enabled": False},
    )

    assert "qualification_categories" not in expected


@pytest.mark.parametrize(
    "declaration",
    [[], ["idle", "idle"], ["unknown"], ["idle", 3]],
)
def test_manifest_rejects_malformed_qualification_categories(
    tmp_path: Path, declaration: list[object]
) -> None:
    manifest = tmp_path / "heldout.json"
    manifest.write_text(
        json.dumps({"qualification_categories": declaration, "clips": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="qualification_categories"):
        cli._load_manifest_specs(
            manifest, cli_source_root=None, default_fps_source=120.0
        )


def test_manifest_rejects_clip_category_outside_declaration(tmp_path: Path) -> None:
    manifest = tmp_path / "heldout.json"
    manifest.write_text(
        json.dumps(
            {
                "qualification_categories": ["idle"],
                "clips": [
                    {
                        "clip_id": "walk_001",
                        "source_csv": "walk.csv",
                        "category": "walk",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside qualification_categories"):
        cli._load_manifest_specs(
            manifest, cli_source_root=None, default_fps_source=120.0
        )


@pytest.mark.parametrize(
    ("clips", "message"),
    [
        (
            [
                {"clip_id": "Walk", "source_csv": "one.csv"},
                {"clip_id": "walk", "source_csv": "two.csv"},
            ],
            "duplicate clip_id",
        ),
        (
            [
                {"clip_id": "one", "source_csv": "same.csv"},
                {"clip_id": "two", "source_csv": "./same.csv"},
            ],
            "same source CSV",
        ),
        ([{"clip_id": "../escape", "source_csv": "one.csv"}], "clip_id must match"),
    ],
)
def test_manifest_rejects_unsafe_or_ambiguous_selection(
    tmp_path: Path,
    clips: list[dict[str, str]],
    message: str,
) -> None:
    manifest = tmp_path / "heldout.json"
    manifest.write_text(json.dumps({"clips": clips}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        cli._load_manifest_specs(
            manifest,
            cli_source_root=None,
            default_fps_source=120.0,
        )


def test_directory_input_rejects_casefold_duplicate_stems(tmp_path: Path) -> None:
    _write_large_file(tmp_path / "a/same.csv")
    _write_large_file(tmp_path / "b/SAME.csv")

    with pytest.raises(ValueError, match="use --manifest with unique clip_id"):
        cli._load_input_specs(tmp_path, fps_source=120.0)


def test_partial_triplet_recomputes_then_complete_triplet_verifies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "motion.csv"
    source_model = tmp_path / "source.xml"
    target_model = tmp_path / "target.xml"
    output = tmp_path / "output"
    _write_large_file(source)
    _write_large_file(source_model, b"s")
    _write_large_file(target_model, b"t")
    for directory in ("motions", "experts", "reports"):
        (output / directory).mkdir(parents=True)
    outputs = cli._output_triplet(output, "heldout_001")
    outputs.motion.write_bytes(b"stale partial triplet member")

    class FakeResult:
        task_names = ("left_hand",)

        @staticmethod
        def adaptation_arrays() -> dict[str, np.ndarray]:
            return {
                "joint_pos": np.ones((2, 1), dtype=np.float32),
                "root_offset_w": np.zeros((2, 3), dtype=np.float32),
                "trajectory_derivative_convention": np.asarray(
                    ["free_initial_velocity_equal_first_interval"], dtype=np.str_
                ),
            }

        @staticmethod
        def summary() -> dict[str, object]:
            return {
                "schema": cli.RETARGET_RESULT_SCHEMA,
                "schema_version": cli.TASK_SPACE_RETARGET_SCHEMA_VERSION,
                "frame_count": 2,
                "constraints": {},
            }

    monkeypatch.setattr(cli, "load_models", lambda *_: (object(), object()))
    monkeypatch.setattr(cli, "_model_joint_names", lambda _: ("joint",))
    monkeypatch.setattr(
        cli,
        "safe_target_joint_bounds",
        lambda *_args, **_kwargs: (
            np.asarray([-1.0]),
            np.asarray([1.0]),
        ),
    )
    monkeypatch.setattr(
        cli,
        "_load_csv_trajectory",
        lambda *_, **__: (
            np.zeros((2, 3)),
            np.zeros((2, 4)),
            np.zeros((2, 1)),
        ),
    )
    call_count = 0

    def fake_retarget(**_: object) -> FakeResult:
        nonlocal call_count
        call_count += 1
        return FakeResult()

    monkeypatch.setattr(cli, "retarget_trajectory", fake_retarget)
    monkeypatch.setattr(
        cli,
        "build_mjlab_motion_arrays",
        lambda *_: {"joint_pos": np.zeros((2, 1), dtype=np.float32)},
    )

    spec = cli.ClipSpec("heldout_001", source, "walk", 120.0)
    arguments = (
        spec,
        output,
        source_model,
        target_model,
        cli._sha256(source_model),
        cli._sha256(target_model),
        50.0,
        2,
        {"max_iterations": 2},
    )
    clip_id, status, first_summary = cli._convert_one(*arguments)
    assert clip_id == spec.clip_id
    assert status == "ok: 2 frames"
    assert first_summary is not None
    assert first_summary["cache_disposition"] == "recomputed: partial"
    assert "qualification_categories" not in first_summary
    assert all(path.is_file() for path in outputs.paths())
    assert not list(output.rglob("*.partial"))

    _, status, cached_summary = cli._convert_one(*arguments)
    assert status == "skipped: verified cache"
    assert cached_summary is not None
    assert call_count == 1

    report_payload = json.loads(outputs.report.read_text(encoding="utf-8"))
    report_payload.pop("serialization_constraint_audit_passed")
    outputs.report.write_text(json.dumps(report_payload), encoding="utf-8")
    _, status, rebuilt_summary = cli._convert_one(*arguments)
    assert status == "ok: 2 frames"
    assert rebuilt_summary is not None
    assert rebuilt_summary["cache_disposition"] == (
        "recomputed: missing serialization constraint certificate"
    )
    assert call_count == 2

    outputs.expert.write_bytes(b"corrupt")
    _, status, rebuilt_summary = cli._convert_one(*arguments)
    assert status == "ok: 2 frames"
    assert rebuilt_summary is not None
    assert rebuilt_summary["cache_disposition"] == "recomputed: expert output hash mismatch"
    assert call_count == 3


def test_aggregate_counts_only_ok_and_verified_skip_as_completed(tmp_path: Path) -> None:
    specs = [
        cli.ClipSpec(f"clip_{index}", tmp_path / f"{index}.csv", "walk", 120.0)
        for index in range(5)
    ]
    results = {
        "clip_0": "ok: 10 frames",
        "clip_1": "skipped: verified cache",
        "clip_2": "pending: input file is empty/incomplete",
        "clip_3": "failed: RuntimeError: boom",
        "clip_4": "rejected: only 2 frames after resampling",
    }

    aggregate = cli._aggregate_report(
        specs=specs,
        selection={"mode": "manifest"},
        results=results,
        summaries={},
    )

    assert aggregate["ok_count"] == 1
    assert aggregate["skipped_count"] == 1
    assert aggregate["pending_count"] == 1
    assert aggregate["failed_count"] == 1
    assert aggregate["rejected_count"] == 1
    assert aggregate["completed_count"] == 2
    assert aggregate["all_complete"] is False


def test_auto_retime_is_continuous_and_meets_reserved_derivative_budgets() -> None:
    frame_count = 40
    phase = np.linspace(0.0, 4.0 * np.pi, frame_count)
    root_pos = np.zeros((frame_count, 3), dtype=np.float64)
    root_pos[:, 0] = np.linspace(0.0, 1.0, frame_count)
    root_quat = np.zeros((frame_count, 4), dtype=np.float64)
    root_quat[:, 0] = 1.0
    joints = np.sin(phase)[:, None]

    retimed_root, retimed_quat, retimed_joints, metadata = (
        cli._auto_retime_trajectory(
            root_pos,
            root_quat,
            joints,
            source_joint_names=("joint",),
            target_joint_names=("joint",),
            fps=50.0,
            max_velocity_rad_s=8.0,
            max_acceleration_rad_s2=80.0,
            retained_lower_bounds=np.asarray([-2.0]),
            retained_upper_bounds=np.asarray([2.0]),
            velocity_fraction=0.75,
            acceleration_fraction=0.5,
            max_time_scale=8.0,
        )
    )

    assert metadata["time_scale"] > 1.0
    assert len(retimed_joints) > frame_count
    np.testing.assert_allclose(retimed_root[[0, -1]], root_pos[[0, -1]])
    np.testing.assert_allclose(
        retimed_joints[[0, -1]], joints[[0, -1]], atol=1.0e-12
    )
    np.testing.assert_allclose(np.linalg.norm(retimed_quat, axis=1), 1.0)
    velocity_max, acceleration_max = cli._trajectory_derivative_maxima(
        retimed_joints, 50.0
    )
    assert velocity_max <= 6.0 * 1.000001
    assert acceleration_max <= 40.0 * 1.000001


def test_auto_retime_measures_post_clip_safe_target_acceleration() -> None:
    joints = np.asarray([[0.0], [0.2], [0.4], [0.6], [0.8]], dtype=np.float64)
    root_pos = np.zeros((len(joints), 3), dtype=np.float64)
    root_quat = np.repeat(
        np.asarray([[1.0, 0.0, 0.0, 0.0]]), len(joints), axis=0
    )
    _, _, retimed, metadata = cli._auto_retime_trajectory(
        root_pos,
        root_quat,
        joints,
        source_joint_names=("joint",),
        target_joint_names=("joint",),
        fps=50.0,
        max_velocity_rad_s=20.0,
        max_acceleration_rad_s2=80.0,
        retained_lower_bounds=np.asarray([-1.0]),
        retained_upper_bounds=np.asarray([0.45]),
        velocity_fraction=1.0,
        acceleration_fraction=0.75,
        max_time_scale=8.0,
    )

    safe = np.clip(retimed, -1.0, 0.45)
    _, acceleration_max = cli._trajectory_derivative_maxima(safe, 50.0)
    assert metadata["constraint_signal"] == (
        "safe_action_reachable_clipped_retained23"
    )
    assert acceleration_max <= 60.0 * 1.000001


def test_auto_retime_can_converge_on_sixth_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joints = np.zeros((3, 1), dtype=np.float64)
    root_pos = np.zeros((3, 3), dtype=np.float64)
    root_quat = np.repeat(
        np.asarray([[1.0, 0.0, 0.0, 0.0]]), 3, axis=0
    )
    derivative_results = iter(
        [(1.0, 4.0)]
        + [(1.01, 1.0)] * 5
        + [(1.0, 1.0)]
    )
    interpolation_scales: list[float] = []

    def fake_derivative_maxima(
        _joint_pos: np.ndarray, _fps: float
    ) -> tuple[float, float]:
        return next(derivative_results)

    def fake_interpolate(
        input_root: np.ndarray,
        input_quat: np.ndarray,
        input_joints: np.ndarray,
        time_scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        interpolation_scales.append(time_scale)
        return input_root, input_quat, input_joints, time_scale

    monkeypatch.setattr(cli, "_trajectory_derivative_maxima", fake_derivative_maxima)
    monkeypatch.setattr(cli, "_interpolate_time_scaled_trajectory", fake_interpolate)

    _, _, _, metadata = cli._auto_retime_trajectory(
        root_pos,
        root_quat,
        joints,
        source_joint_names=("joint",),
        target_joint_names=("joint",),
        fps=50.0,
        max_velocity_rad_s=1.0,
        max_acceleration_rad_s2=1.0,
        retained_lower_bounds=np.asarray([-1.0]),
        retained_upper_bounds=np.asarray([1.0]),
        velocity_fraction=1.0,
        acceleration_fraction=1.0,
        max_time_scale=10.0,
    )

    assert len(interpolation_scales) == 6
    assert metadata["retimed_velocity_abs_max_rad_s"] == 0.0
    assert metadata["retimed_acceleration_abs_max_rad_s2"] == 0.0


def test_auto_retime_iteration_bound_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joints = np.zeros((3, 1), dtype=np.float64)
    root_pos = np.zeros((3, 3), dtype=np.float64)
    root_quat = np.repeat(
        np.asarray([[1.0, 0.0, 0.0, 0.0]]), 3, axis=0
    )
    derivative_call_count = 0

    def never_converges(
        _joint_pos: np.ndarray, _fps: float
    ) -> tuple[float, float]:
        nonlocal derivative_call_count
        derivative_call_count += 1
        if derivative_call_count == 1:
            return 1.0, 1.0
        return 1.01, 1.0

    monkeypatch.setattr(cli, "_trajectory_derivative_maxima", never_converges)
    monkeypatch.setattr(
        cli,
        "_interpolate_time_scaled_trajectory",
        lambda root, quat, values, scale: (root, quat, values, scale),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "automatic retiming did not converge after "
            f"{cli.AUTO_RETIME_MAX_CONVERGENCE_ITERATIONS} iterations"
        ),
    ):
        cli._auto_retime_trajectory(
            root_pos,
            root_quat,
            joints,
            source_joint_names=("joint",),
            target_joint_names=("joint",),
            fps=50.0,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=1.0,
            retained_lower_bounds=np.asarray([-1.0]),
            retained_upper_bounds=np.asarray([1.0]),
            velocity_fraction=1.0,
            acceleration_fraction=1.0,
            max_time_scale=100.0,
        )

    assert derivative_call_count == 1 + cli.AUTO_RETIME_MAX_CONVERGENCE_ITERATIONS


def test_auto_retime_final_derivative_audit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    joints = np.asarray([[0.0], [1.0], [0.0]], dtype=np.float64)
    root_pos = np.zeros((3, 3), dtype=np.float64)
    root_quat = np.repeat(
        np.asarray([[1.0, 0.0, 0.0, 0.0]]), 3, axis=0
    )
    derivative_results = iter([(1.0, 1.0), (1.0, 1.0)])
    monkeypatch.setattr(
        cli,
        "_trajectory_derivative_maxima",
        lambda _values, _fps: next(derivative_results),
    )

    with pytest.raises(
        RuntimeError, match="automatic retiming failed final derivative audit"
    ):
        cli._auto_retime_trajectory(
            root_pos,
            root_quat,
            joints,
            source_joint_names=("joint",),
            target_joint_names=("joint",),
            fps=50.0,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=1.0,
            retained_lower_bounds=np.asarray([-1.0]),
            retained_upper_bounds=np.asarray([1.0]),
            velocity_fraction=1.0,
            acceleration_fraction=1.0,
            max_time_scale=10.0,
        )


def test_main_catches_future_exception_and_returns_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.csv"
    source_model = tmp_path / "source.xml"
    target_model = tmp_path / "target.xml"
    report = tmp_path / "batch.json"
    _write_large_file(source)
    _write_large_file(source_model, b"s")
    _write_large_file(target_model, b"t")

    class BrokenExecutor:
        def __init__(self, *, max_workers: int) -> None:
            assert max_workers == 1

        def __enter__(self) -> BrokenExecutor:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        @staticmethod
        def submit(*_: object, **__: object) -> concurrent.futures.Future[object]:
            future: concurrent.futures.Future[object] = concurrent.futures.Future()
            future.set_exception(RuntimeError("worker died"))
            return future

    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", BrokenExecutor)

    exit_code = cli.main(
        [
            "--input",
            str(source),
            "--output",
            str(tmp_path / "output"),
            "--source-model",
            str(source_model),
            "--target-model",
            str(target_model),
            "--workers",
            "1",
            "--report",
            str(report),
        ]
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["failed_count"] == 1
    assert payload["completed_count"] == 0
    assert payload["results"]["input"].startswith("failed: worker exception: RuntimeError")


def test_parser_requires_exactly_one_input_mode(tmp_path: Path) -> None:
    parser = cli._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input",
                "one.csv",
                "--manifest",
                "selection.json",
                "--output",
                str(tmp_path),
            ]
        )
