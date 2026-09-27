from dataclasses import FrozenInstanceError, fields, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gear_sonic.data.unitree_conversion.contracts import DiagnosticReport, SourceSpec
from gear_sonic.data.unitree_conversion.inspire_diagnostics import (
    INSPIRE_GATE_REASONS,
    diagnose_inspire_arrays,
    diagnose_inspire_episode,
)
from gear_sonic.data.unitree_conversion.provenance import load_source_lock

DATASET_PATH = "G1_WB_Dex5_Pickup_Pillow"
CURRENT_KEY = "observation.state.robot_q_current"
DESIRED_KEY = "action.robot_q_desired"
HAND_STATE_KEY = "observation.state.hand_state"
HAND_CMD_KEY = "action.hand_cmd"
CAMERAS = ("observation.images.cam_0", "observation.images.cam_1")
SMOKE_LOCK = Path("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")


def _motion_arrays(n: int = 5) -> dict[str, np.ndarray]:
    current = np.arange(n * 36, dtype=np.float64).reshape(n, 36) / 10.0
    desired = current + 100.0
    current[:, 3:7] = (1.0, 0.0, 0.0, 0.0)
    desired[:, 3:7] = (1.0, 0.0, 0.0, 0.0)
    hand_state = np.arange(n * 12, dtype=np.float64).reshape(n, 12) / 20.0
    hand_cmd = hand_state + 10.0
    return {
        "current": current,
        "desired": desired,
        "hand_state": hand_state,
        "hand_cmd": hand_cmd,
        "timestamps": np.arange(n, dtype=np.float64) / 30.0,
    }


def test_valid_shapes_remain_blocked_with_exact_three_reasons_and_deterministic_audit() -> None:
    arrays = _motion_arrays()

    report = diagnose_inspire_arrays(**arrays)

    assert report.status == "blocked_unverified"
    assert report.gate_reasons == (
        "missing_authoritative_29_joint_order",
        "missing_robot_q_desired_semantics",
        "missing_inspire_to_dex3_calibration",
    )
    assert report.gate_reasons == INSPIRE_GATE_REASONS
    assert report.encoder_invoked is False
    assert report.frame_count == 5
    assert report.source_repo_id is None
    assert report.source_revision is None
    assert report.source_episode_id is None
    assert report.source_task_indices is None
    assert report.timestamp_audit.finite_count == 5
    assert report.timestamp_audit.strictly_increasing is True
    assert report.timestamp_audit.start == 0.0
    assert report.timestamp_audit.end == 4 / 30
    assert report.finite_counts == {
        "robot_q_current": 180,
        "robot_q_desired": 180,
        "hand_state": 60,
        "hand_cmd": 60,
        "timestamps": 5,
    }
    current_norms = np.linalg.norm(arrays["current"][:, 3:7], axis=1)
    desired_norms = np.linalg.norm(arrays["desired"][:, 3:7], axis=1)
    assert report.current_root_quaternion_norm.minimum == pytest.approx(current_norms.min())
    assert report.current_root_quaternion_norm.maximum == pytest.approx(current_norms.max())
    assert report.desired_root_quaternion_norm.minimum == pytest.approx(desired_norms.min())
    assert report.desired_root_quaternion_norm.maximum == pytest.approx(desired_norms.max())
    first = report.column_statistics["robot_q_current"][0]
    assert first.minimum == pytest.approx(arrays["current"][:, 0].min())
    assert first.maximum == pytest.approx(arrays["current"][:, 0].max())
    assert first.mean == pytest.approx(arrays["current"][:, 0].mean())
    assert first.std == pytest.approx(arrays["current"][:, 0].std(ddof=0))


def test_report_is_deeply_immutable() -> None:
    report = diagnose_inspire_arrays(**_motion_arrays())

    with pytest.raises(FrozenInstanceError):
        report.status = "ready"
    with pytest.raises(TypeError):
        report.finite_counts["timestamps"] = 0
    with pytest.raises(TypeError):
        report.column_statistics["robot_q_current"] = ()


def test_report_serializes_deterministically_to_json_primitives() -> None:
    report = diagnose_inspire_arrays(**_motion_arrays(n=2))

    payload = report.to_dict()

    assert payload["gate_reasons"] == list(INSPIRE_GATE_REASONS)
    assert payload["source_task_indices"] is None
    assert payload["timestamp_audit"]["finite_count"] == 2
    assert len(payload["column_statistics"]["robot_q_current"]) == 36
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert encoded == json.dumps(report.to_dict(), sort_keys=True, allow_nan=False)


def test_source_schema_error_status_constructs_directly_and_via_replace() -> None:
    report = diagnose_inspire_arrays(**_motion_arrays(n=2))
    values = {field.name: getattr(report, field.name) for field in fields(DiagnosticReport)}
    values["status"] = "source_schema_error"

    direct = DiagnosticReport(**values)
    replaced = replace(report, status="source_schema_error")

    assert direct.status == "source_schema_error"
    assert replaced.status == "source_schema_error"
    assert direct.gate_reasons == INSPIRE_GATE_REASONS
    assert direct.to_dict()["status"] == "source_schema_error"
    assert replaced.to_dict()["status"] == "source_schema_error"


@pytest.mark.parametrize("status", ["ready", "error", "", None, 1])
def test_diagnostic_report_rejects_every_other_status(status: object) -> None:
    report = diagnose_inspire_arrays(**_motion_arrays(n=2))

    with pytest.raises(ValueError, match="diagnostic status"):
        replace(report, status=status)


@pytest.mark.parametrize(
    ("field_name", "shape", "source_name"),
    [
        ("current", (5, 35), "robot_q_current"),
        ("desired", (5, 37), "robot_q_desired"),
        ("hand_state", (5, 11), "hand_state"),
        ("hand_cmd", (5, 13), "hand_cmd"),
    ],
)
def test_pure_diagnostics_reject_wrong_motion_shapes(
    field_name: str,
    shape: tuple[int, ...],
    source_name: str,
) -> None:
    arrays = _motion_arrays()
    arrays[field_name] = np.zeros(shape)

    with pytest.raises(ValueError, match=rf"{source_name}.*\[N,{36 if 'robot_q' in source_name else 12}\]"):
        diagnose_inspire_arrays(**arrays)


@pytest.mark.parametrize("field_name", ["current", "desired", "hand_state", "hand_cmd"])
def test_pure_diagnostics_reject_non_numeric_or_nonfinite_motion(field_name: str) -> None:
    arrays = _motion_arrays()
    arrays[field_name] = np.full(arrays[field_name].shape, "not-numeric")
    with pytest.raises(ValueError, match="numeric array"):
        diagnose_inspire_arrays(**arrays)

    arrays = _motion_arrays()
    arrays[field_name][-1, -1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        diagnose_inspire_arrays(**arrays)


def test_pure_diagnostics_reject_mismatched_or_short_rows() -> None:
    arrays = _motion_arrays()
    arrays["hand_cmd"] = arrays["hand_cmd"][:-1]
    with pytest.raises(ValueError, match="common row count"):
        diagnose_inspire_arrays(**arrays)

    with pytest.raises(ValueError, match="at least two rows"):
        diagnose_inspire_arrays(**_motion_arrays(n=1))


@pytest.mark.parametrize(
    "timestamps",
    [
        np.zeros((5, 1)),
        np.array([0.0, 1 / 30, np.nan, 3 / 30, 4 / 30]),
        np.array([0.0, 1 / 30, 1 / 30, 3 / 30, 4 / 30]),
        np.array([0.0, 2 / 30, 1 / 30, 3 / 30, 4 / 30]),
    ],
)
def test_pure_diagnostics_require_finite_strictly_increasing_timestamp_vector(
    timestamps: np.ndarray,
) -> None:
    arrays = _motion_arrays()
    arrays["timestamps"] = timestamps

    with pytest.raises(ValueError, match="timestamps"):
        diagnose_inspire_arrays(**arrays)


def _named_feature(prefix: str, size: int) -> dict[str, object]:
    return {
        "dtype": "float32",
        "shape": [size],
        "names": [f"{prefix}_{index}" for index in range(size)],
    }


def _video_feature() -> dict[str, object]:
    return {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channel"],
    }


def _scalar_feature(dtype: str) -> dict[str, object]:
    return {"dtype": dtype, "shape": [1], "names": None}


def _metadata_info(**changes: object) -> dict[str, object]:
    info: dict[str, object] = {
        "codebase_version": "v3.0",
        "robot_type": "unitree_g1",
        "chunks_size": 1000,
        "fps": 30,
        "total_episodes": 609,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            CAMERAS[0]: _video_feature(),
            CAMERAS[1]: _video_feature(),
            "observation.state.ee_state": _named_feature("ee_state", 12),
            HAND_STATE_KEY: _named_feature("hand_state", 12),
            CURRENT_KEY: _named_feature("robot_q_current", 36),
            "action.ee_action": _named_feature("ee_action", 12),
            HAND_CMD_KEY: _named_feature("hand_cmd", 12),
            DESIRED_KEY: _named_feature("robot_q_desired", 36),
            "timestamp": _scalar_feature("float32"),
            "frame_index": _scalar_feature("int64"),
            "episode_index": _scalar_feature("int64"),
            "index": _scalar_feature("int64"),
            "task_index": _scalar_feature("int64"),
        },
    }
    info.update(changes)
    return info


def _episode_metadata_row(episode_id: int = 152) -> dict[str, object]:
    return {
        "episode_index": episode_id,
        "data/chunk_index": 0,
        "data/file_index": 3,
        "length": 3,
        **{
            f"videos/{camera}/{part}": value
            for camera in CAMERAS
            for part, value in (
                ("chunk_index", 0),
                ("file_index", 4),
                ("from_timestamp", 0.0),
                ("to_timestamp", 3 / 30),
            )
        },
    }


def _source_rows(episode_id: int = 152, n: int = 3) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame_index in range(n):
        current = np.zeros(36, dtype=np.float32)
        desired = np.zeros(36, dtype=np.float32)
        current[3] = 1.0
        desired[3] = 1.0
        current[7:] = frame_index
        desired[7:] = frame_index + 0.5
        rows.append(
            {
                CURRENT_KEY: current.tolist(),
                DESIRED_KEY: desired.tolist(),
                HAND_STATE_KEY: np.full(12, frame_index, dtype=np.float32).tolist(),
                HAND_CMD_KEY: np.full(12, frame_index + 1, dtype=np.float32).tolist(),
                "timestamp": frame_index / 30,
                "task_index": 0,
                "frame_index": frame_index,
                "episode_index": episode_id,
            }
        )
    return rows


class PrefixSnapshotDownloader:
    def __init__(
        self,
        *,
        info: dict[str, object] | None = None,
        episode_rows: list[dict[str, object]] | None = None,
        data_rows: list[dict[str, object]] | None = None,
    ) -> None:
        self.info = _metadata_info() if info is None else info
        self.episode_rows = [_episode_metadata_row()] if episode_rows is None else episode_rows
        self.data_rows = _source_rows() if data_rows is None else data_rows
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> str:
        self.calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        allow_patterns = kwargs["allow_patterns"]
        if len(self.calls) == 1:
            assert allow_patterns == [f"{DATASET_PATH}/meta/**"]
            info_path = local_dir / DATASET_PATH / "meta/info.json"
            info_path.parent.mkdir(parents=True, exist_ok=True)
            info_path.write_text(json.dumps(self.info))
            episode_path = local_dir / DATASET_PATH / "meta/episodes/chunk-000/file-000.parquet"
            episode_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(self.episode_rows), episode_path)
        else:
            assert allow_patterns == [f"{DATASET_PATH}/data/chunk-000/file-003.parquet"]
            data_path = local_dir / allow_patterns[0]
            data_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist(self.data_rows), data_path)
        return str(local_dir)


def _source_spec(**changes: object) -> SourceSpec:
    values: dict[str, object] = {
        "approved": True,
        "repo_id": "unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly",
        "revision": "2" * 40,
        "dataset_path": DATASET_PATH,
        "episode_count": 609,
        "episodes": (0, 152, 304, 456, 608),
        "primary_camera": CAMERAS[0],
        "camera_map": {CAMERAS[0]: "diagnostic.primary_camera"},
        "label": "inspire",
    }
    values.update(changes)
    return SourceSpec(**values)


def test_episode_diagnostic_reads_selected_nested_v3_rows_without_video_download(tmp_path: Path) -> None:
    unrelated = _source_rows(episode_id=0, n=2)
    downloader = PrefixSnapshotDownloader(data_rows=unrelated + _source_rows())

    report = diagnose_inspire_episode(
        _source_spec(),
        152,
        root=tmp_path,
        snapshot_downloader=downloader,
    )

    assert report.status == "blocked_unverified"
    assert report.gate_reasons == INSPIRE_GATE_REASONS
    assert report.encoder_invoked is False
    assert report.source_repo_id == "unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly"
    assert report.source_revision == "2" * 40
    assert report.source_episode_id == 152
    assert report.source_task_indices == (0, 0, 0)
    assert report.frame_count == 3
    assert report.primary_camera == CAMERAS[0]
    assert report.camera_map == {CAMERAS[0]: "diagnostic.primary_camera"}
    assert len(downloader.calls) == 2
    assert all("videos/" not in pattern for pattern in downloader.calls[1]["allow_patterns"])


def test_episode_diagnostic_rejects_gapped_frame_sequence(tmp_path: Path) -> None:
    rows = _source_rows()
    rows[1]["frame_index"] = 2

    with pytest.raises(ValueError, match="expected frame_index 1, got 2"):
        diagnose_inspire_episode(
            _source_spec(),
            152,
            root=tmp_path,
            snapshot_downloader=PrefixSnapshotDownloader(data_rows=rows),
        )


@pytest.mark.parametrize(
    ("source", "episode_id", "message"),
    [
        (_source_spec(approved=False), 152, "approved SourceSpec"),
        (_source_spec(), 1, "selected in source_spec.episodes"),
    ],
)
def test_episode_diagnostic_requires_approved_selected_source(
    tmp_path: Path,
    source: SourceSpec,
    episode_id: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        diagnose_inspire_episode(
            source,
            episode_id,
            root=tmp_path,
            snapshot_downloader=PrefixSnapshotDownloader(),
        )


@pytest.mark.parametrize(
    "source",
    [
        _source_spec(repo_id="unitreerobotics/unsupported"),
        _source_spec(dataset_path="another_dataset"),
        _source_spec(label="dex3"),
        _source_spec(label=None),
    ],
)
def test_episode_diagnostic_rejects_unsupported_adapter_identity_before_download(
    tmp_path: Path,
    source: SourceSpec,
) -> None:
    downloader = PrefixSnapshotDownloader()

    with pytest.raises(ValueError, match="supported Inspire adapter identity"):
        diagnose_inspire_episode(
            source,
            152,
            root=tmp_path,
            snapshot_downloader=downloader,
        )

    assert downloader.calls == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"total_episodes": 608}, "total_episodes"),
        ({"fps": 29}, "fps"),
    ],
)
def test_episode_diagnostic_rejects_mismatched_pinned_metadata(
    tmp_path: Path,
    change: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        diagnose_inspire_episode(
            _source_spec(),
            152,
            root=tmp_path,
            snapshot_downloader=PrefixSnapshotDownloader(info=_metadata_info(**change)),
        )


@pytest.mark.parametrize(
    ("feature_key", "field_name", "bad_value"),
    [
        (CURRENT_KEY, "shape", [35]),
        (DESIRED_KEY, "names", [f"wrong_{index}" for index in range(36)]),
        (HAND_STATE_KEY, "dtype", "float64"),
        (HAND_CMD_KEY, "shape", [13]),
        (CAMERAS[0], "dtype", "image"),
    ],
)
def test_episode_diagnostic_rejects_malformed_exact_feature_contract(
    tmp_path: Path,
    feature_key: str,
    field_name: str,
    bad_value: object,
) -> None:
    info = _metadata_info()
    info["features"][feature_key][field_name] = bad_value

    with pytest.raises(ValueError, match="metadata feature"):
        diagnose_inspire_episode(
            _source_spec(),
            152,
            root=tmp_path,
            snapshot_downloader=PrefixSnapshotDownloader(info=info),
        )


def test_episode_diagnostic_rejects_missing_or_extra_feature_keys(tmp_path: Path) -> None:
    missing_info = _metadata_info()
    del missing_info["features"][CURRENT_KEY]
    with pytest.raises(ValueError, match="feature keys must exactly match"):
        diagnose_inspire_episode(
            _source_spec(),
            152,
            root=tmp_path,
            snapshot_downloader=PrefixSnapshotDownloader(info=missing_info),
        )

    extra_info = _metadata_info()
    extra_info["features"]["unexpected"] = _scalar_feature("int64")
    with pytest.raises(ValueError, match="feature keys must exactly match"):
        diagnose_inspire_episode(
            _source_spec(),
            152,
            root=tmp_path,
            snapshot_downloader=PrefixSnapshotDownloader(info=extra_info),
        )


def test_episode_diagnostic_requires_manifest_primary_camera(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="primary camera"):
        diagnose_inspire_episode(
            _source_spec(primary_camera=CAMERAS[1], camera_map={CAMERAS[1]: "diagnostic.primary_camera"}),
            152,
            root=tmp_path,
            snapshot_downloader=PrefixSnapshotDownloader(),
        )


def test_all_five_inspire_diagnostic_ids_are_manifest_backed() -> None:
    source = load_source_lock(SMOKE_LOCK).inspire
    assert source.dataset_path == DATASET_PATH
    assert source.episodes == (0, 152, 304, 456, 608)
    assert len(source.episodes) == 5


def test_diagnostic_module_contains_no_forbidden_conversion_dependencies() -> None:
    module_path = Path("gear_sonic/data/unitree_conversion/inspire_diagnostics.py")
    source = module_path.read_text()
    for forbidden in ("sonic_encoder", "onnxruntime", "target_frames", "RobotModel"):
        assert forbidden not in source

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gear_sonic.data.unitree_conversion.inspire_diagnostics; "
            "assert 'gear_sonic.data.unitree_conversion.sonic_encoder' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
