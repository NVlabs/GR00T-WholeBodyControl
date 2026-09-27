from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.contracts import (
    INSPIRE_GATE_REASONS,
    CollectionMembership,
    SourceLock,
)
from gear_sonic.data.unitree_conversion.pipeline import (
    ClassifiedFailure,
    DiagnosticIdentity,
    PipelineComponents,
    RepositoryPreflight,
    _adapt_dex3,
    _construct_payload,
    _resolve_source_task,
    _source_load_error_class,
    _validate_dex3_row_indices,
    _validate_inspire_root_finiteness,
    _verify_preflight_source_hashes,
    _write_diagnostic,
    run_dex3_pipeline,
    run_inspire_diagnostics,
)
from gear_sonic.data.unitree_conversion.provenance import (
    SMOKE_SOURCE_LOCK,
    load_source_lock,
)
from gear_sonic.scripts.convert_unitree_unifolm_to_sonic import ConvertConfig, _validate_config


class FakeComponents:
    def __init__(self, *, resumed_ids: tuple[int, ...] = ()) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.adapter_ids: list[int] = []
        self.encoder_ids: list[int] = []
        self.encoder_constructions = 0
        self.resumed_ids = frozenset(resumed_ids)

    def build(self) -> PipelineComponents:
        return PipelineComponents(
            validate_artifacts=self.validate_artifacts,
            validate_source_metadata=self.validate_source_metadata,
            preflight_repository=self.preflight_repository,
            make_stage_identity=self.make_stage_identity,
            resume_stage=self.resume_stage,
            adapt_dex3_episode=self.adapt_dex3_episode,
            resample_episode=self.resample_episode,
            encoder_factory=self.encoder_factory,
            encode_episode=self.encode_episode,
            construct_payload=self.construct_payload,
            validate_payload=self.validate_payload,
            write_stage=self.write_stage,
            merge_stages=self.merge_stages,
            diagnose_inspire_episode=self.diagnose_inspire_episode,
            make_diagnostic_identity=self.make_diagnostic_identity,
            write_diagnostic=self.write_diagnostic,
        )

    def validate_artifacts(self, lock, cache_dir):
        self.calls.append(("artifacts", cache_dir))
        return SimpleNamespace(lock=lock)

    def validate_source_metadata(self, source, episode_ids, cache_dir, kind):
        self.calls.append(("metadata", source.repo_id, episode_ids, kind))
        return SimpleNamespace(source=source, episode_ids=episode_ids)

    def preflight_repository(self, source, episode_ids, metadata, cache_dir, kind):
        assert metadata.source is source
        self.calls.append(("preflight", source.repo_id, episode_ids, kind))
        return SimpleNamespace(source=source, episode_ids=episode_ids)

    def make_stage_identity(self, lock_sha256, source, episode_id, artifacts, preflight):
        self.calls.append(("identity", episode_id))
        return (source.repo_id, episode_id, lock_sha256)

    def resume_stage(self, output_root, identity):
        episode_id = identity[1]
        self.calls.append(("resume", episode_id))
        if episode_id in self.resumed_ids:
            return SimpleNamespace(path=output_root / f"stage-{episode_id}", identity=identity)
        return None

    def adapt_dex3_episode(self, source, episode_id, preflight, cache_dir):
        self.calls.append(("adapt", episode_id))
        self.adapter_ids.append(episode_id)
        return SimpleNamespace(source_episode_id=episode_id)

    def resample_episode(self, episode, preflight):
        self.calls.append(("resample", episode.source_episode_id))
        return episode

    def encoder_factory(self, artifacts):
        self.calls.append(("encoder_factory",))
        self.encoder_constructions += 1
        return object()

    def encode_episode(self, encoder, episode):
        del encoder
        self.calls.append(("encode", episode.source_episode_id))
        self.encoder_ids.append(episode.source_episode_id)
        return SimpleNamespace(episode_id=episode.source_episode_id, invocation_count=1)

    def construct_payload(self, episode, encoded, preflight):
        self.calls.append(("construct", episode.source_episode_id))
        return SimpleNamespace(episode_id=encoded.episode_id)

    def validate_payload(self, payload):
        self.calls.append(("validate", payload.episode_id))
        return SimpleNamespace(valid=True)

    def write_stage(self, output_root, identity, payload):
        self.calls.append(("stage", payload.episode_id))
        return SimpleNamespace(path=output_root / f"stage-{payload.episode_id}", identity=identity)

    def merge_stages(self, stages, target_path, resume):
        self.calls.append(("merge", tuple(stage.identity[1] for stage in stages), resume))
        target_path.mkdir(parents=True)
        return target_path

    def diagnose_inspire_episode(self, source, episode_id, preflight, cache_dir):
        self.calls.append(("diagnose", episode_id))
        return SimpleNamespace(
            status="blocked_unverified",
            gate_reasons=INSPIRE_GATE_REASONS,
            encoder_invoked=False,
            source_repo_id=source.repo_id,
            source_episode_id=episode_id,
            to_dict=lambda: {
                "status": "blocked_unverified",
                "gate_reasons": list(INSPIRE_GATE_REASONS),
                "source_episode_id": episode_id,
            },
        )

    def make_diagnostic_identity(self, lock_sha256, source, episode_id, preflight):
        self.calls.append(("diagnostic_identity", episode_id))
        return (lock_sha256, source.repo_id, episode_id)

    def write_diagnostic(self, output_root, identity, report):
        assert identity[2] == report.source_episode_id
        self.calls.append(("write_diagnostic", report.source_episode_id))
        path = output_root / f"episode-{report.source_episode_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
        return path


@pytest.fixture
def smoke_lock():
    return load_source_lock(SMOKE_SOURCE_LOCK)


def test_dex3_smoke_processes_only_five_locked_ids(smoke_lock, tmp_path: Path) -> None:
    fake = FakeComponents()

    report = run_dex3_pipeline(
        lock=smoke_lock,
        output_root=tmp_path,
        components=fake.build(),
        smoke=True,
    )

    assert report.source_episode_ids == (0, 78, 155, 233, 310)
    assert report.validated_episode_count == 5
    assert report.failed_episode_count == 0
    assert fake.adapter_ids == [0, 78, 155, 233, 310]
    assert report.target_dataset_paths == (tmp_path / "unitreerobotics--G1_Dex3_Pouring_Dataset",)


def test_dex3_pipeline_preserves_required_order(smoke_lock, tmp_path: Path) -> None:
    fake = FakeComponents()

    run_dex3_pipeline(
        lock=smoke_lock,
        output_root=tmp_path,
        components=fake.build(),
        smoke=True,
    )

    names = [call[0] for call in fake.calls]
    assert names[:3] == ["artifacts", "metadata", "preflight"]
    for episode_id in (0, 78, 155, 233, 310):
        indices = {
            name: fake.calls.index((name, episode_id))
            for name in ("identity", "resume", "adapt", "resample", "encode", "construct", "validate", "stage")
        }
        assert list(indices.values()) == sorted(indices.values())
    assert names[-1] == "merge"


def test_resume_invokes_only_three_adapters_and_encoders(smoke_lock, tmp_path: Path) -> None:
    fake = FakeComponents(resumed_ids=(0, 78))

    report = run_dex3_pipeline(
        lock=smoke_lock,
        output_root=tmp_path,
        components=fake.build(),
        smoke=True,
        resume=True,
    )

    assert fake.adapter_ids == [155, 233, 310]
    assert fake.encoder_ids == [155, 233, 310]
    assert report.encoder_invocation_count == 3
    assert report.validated_episode_count == 5


def test_episode_failure_is_classified_and_prevents_merge(smoke_lock, tmp_path: Path) -> None:
    fake = FakeComponents()
    original = fake.adapt_dex3_episode

    def fail_one(source, episode_id, preflight, cache_dir):
        if episode_id == 78:
            raise ValueError("bad source row")
        return original(source, episode_id, preflight, cache_dir)

    components = replace(fake.build(), adapt_dex3_episode=fail_one)

    report = run_dex3_pipeline(
        lock=smoke_lock,
        output_root=tmp_path,
        components=components,
        smoke=True,
    )

    assert report.validated_episode_count == 4
    assert report.failed_episode_count == 1
    assert report.status_counts == {"failed": 1, "validated": 4}
    assert report.episode_reports[1].error_class == "source_schema_error"
    assert report.target_dataset_paths == ()
    assert all(call[0] != "merge" for call in fake.calls)
    assert fake.adapter_ids == [0, 155, 233, 310]


def test_failed_encoder_reports_attempted_invocations(smoke_lock, tmp_path: Path) -> None:
    fake = FakeComponents()

    class FailingEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def encode(self, _tensor):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("inference failed")
            return object()

    def encode_twice(encoder, episode):
        encoder.encode(object())
        encoder.encode(object())
        raise AssertionError("unreachable")

    components = replace(
        fake.build(),
        encoder_factory=lambda _artifacts: FailingEncoder(),
        encode_episode=encode_twice,
    )

    report = run_dex3_pipeline(
        lock=smoke_lock,
        output_root=tmp_path,
        components=components,
        smoke=True,
    )

    assert report.episode_reports[0].error_class == "encoder_contract_error"
    assert report.episode_reports[0].encoder_invocation_count == 2


def test_dex3_adapter_rejects_out_of_order_source_frame_indices() -> None:
    rows = (
        {"episode_index": 7, "frame_index": 0},
        {"episode_index": 7, "frame_index": 99},
    )

    with pytest.raises(ValueError) as captured:
        _validate_dex3_row_indices(rows, 7)

    assert captured.value.error_class == "source_schema_error"


def test_production_dex3_adapter_rejects_out_of_order_indices(
    monkeypatch: pytest.MonkeyPatch,
    smoke_lock,
    tmp_path: Path,
) -> None:
    from gear_sonic.data.unitree_conversion import lerobot_v3_source, pipeline as pipeline_module
    from gear_sonic.data.unitree_conversion.dex3_adapter import DEX3_FEATURE_NAMES

    source = smoke_lock.dex3
    rows = (
        {"episode_index": 0, "frame_index": 0},
        {"episode_index": 0, "frame_index": 99},
    )
    meta = SimpleNamespace(
        revision=source.revision,
        total_episodes=source.episode_count,
        fps=30,
        features={
            "observation.state": {"names": DEX3_FEATURE_NAMES},
            "action": {"names": DEX3_FEATURE_NAMES},
        },
    )
    fresh = SimpleNamespace(root=tmp_path, revision=source.revision, meta=meta, hf_dataset=rows)
    original = SimpleNamespace(root=tmp_path, video_segments={})
    preflight = SimpleNamespace(datasets={0: original}, source_hashes={0: {"data": "1" * 64}})
    monkeypatch.setattr(pipeline_module, "_verify_preflight_source_hashes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lerobot_v3_source, "load_pinned_v3_episode", lambda *_args, **_kwargs: fresh)

    with pytest.raises(ValueError) as captured:
        _adapt_dex3(source, 0, preflight, tmp_path)

    assert captured.value.error_class == "source_schema_error"


def test_production_dex3_adapter_detects_mutation_after_fresh_read(
    monkeypatch: pytest.MonkeyPatch,
    smoke_lock,
    tmp_path: Path,
) -> None:
    from gear_sonic.data.unitree_conversion import lerobot_v3_source, pipeline as pipeline_module
    from gear_sonic.data.unitree_conversion.dex3_adapter import DEX3_FEATURE_NAMES

    source = smoke_lock.dex3
    rows = tuple(
        {
            "episode_index": 0,
            "frame_index": index,
            "timestamp": index / 30.0,
            "task_index": 0,
            "observation.state": np.zeros(28, dtype=np.float32),
            "action": np.zeros(28, dtype=np.float32),
        }
        for index in range(2)
    )
    meta = SimpleNamespace(
        revision=source.revision,
        total_episodes=source.episode_count,
        fps=30,
        features={
            "observation.state": {"names": DEX3_FEATURE_NAMES},
            "action": {"names": DEX3_FEATURE_NAMES},
        },
    )
    fresh = SimpleNamespace(root=tmp_path, revision=source.revision, meta=meta, hf_dataset=rows)
    original = SimpleNamespace(root=tmp_path, video_segments={})
    expected = {"data/episode.parquet": "1" * 64}
    preflight = SimpleNamespace(datasets={0: original}, source_hashes={0: expected})
    observed_hashes = iter((expected, expected, {"data/episode.parquet": "2" * 64}))
    monkeypatch.setattr(
        pipeline_module,
        "_source_hashes",
        lambda _dataset, _episode_id: next(observed_hashes),
    )
    monkeypatch.setattr(lerobot_v3_source, "load_pinned_v3_episode", lambda *_args, **_kwargs: fresh)

    with pytest.raises(ValueError) as captured:
        _adapt_dex3(source, 0, preflight, tmp_path)

    assert captured.value.error_class == "provenance_error"
    assert "during Dex3 adaptation" in str(captured.value)


def test_source_mutation_after_preflight_is_provenance_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from gear_sonic.data.unitree_conversion import pipeline as pipeline_module

    preflight = SimpleNamespace(
        datasets={7: object()},
        source_hashes={7: {"data/episode.parquet": "1" * 64}},
    )
    monkeypatch.setattr(
        pipeline_module,
        "_source_hashes",
        lambda _dataset, _episode_id: {"data/episode.parquet": "2" * 64},
    )

    with pytest.raises(ValueError) as captured:
        _verify_preflight_source_hashes(preflight, 7, operation="during test")

    assert captured.value.error_class == "provenance_error"

    monkeypatch.setattr(
        pipeline_module,
        "_source_hashes",
        lambda _dataset, _episode_id: (_ for _ in ()).throw(OSError("source deleted")),
    )
    with pytest.raises(ValueError) as captured:
        _verify_preflight_source_hashes(preflight, 7, operation="during test")
    assert captured.value.error_class == "provenance_error"


def test_fresh_dex3_read_failure_reaches_pipeline_as_provenance(
    monkeypatch: pytest.MonkeyPatch,
    smoke_lock,
    tmp_path: Path,
) -> None:
    from gear_sonic.data.unitree_conversion import lerobot_v3_source, pipeline as pipeline_module

    fake = FakeComponents()
    preflight = SimpleNamespace(datasets={}, source_hashes={})
    components = replace(
        fake.build(),
        preflight_repository=lambda *_args: preflight,
        adapt_dex3_episode=_adapt_dex3,
    )
    monkeypatch.setattr(pipeline_module, "_verify_preflight_source_hashes", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        lerobot_v3_source,
        "load_pinned_v3_episode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("selected parquet disappeared")),
    )

    report = run_dex3_pipeline(
        lock=smoke_lock,
        output_root=tmp_path,
        components=components,
        smoke=True,
    )

    assert report.failed_episode_count == 5
    assert {episode.error_class for episode in report.episode_reports} == {"provenance_error"}


def test_video_mutation_cleans_temporary_output_and_is_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from gear_sonic.data.unitree_conversion import (
        pipeline as pipeline_module,
        target_frames as target_frames_module,
    )

    class FakeBuilder:
        def build(self, _episode, _index, _token):
            return {}

    calls = 0

    def verify(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise pipeline_module._EpisodePreflightError(
                "provenance_error",
                "source video changed during construction",
            )

    temporary_video = tmp_path / "target.mp4"

    def encode(_timeline):
        temporary_video.write_bytes(b"temporary video")
        return temporary_video

    monkeypatch.setattr(target_frames_module, "TargetFrameBuilder", FakeBuilder)
    monkeypatch.setattr(pipeline_module, "_verify_preflight_source_hashes", verify)
    monkeypatch.setattr(pipeline_module, "_encode_target_video", encode)
    episode = SimpleNamespace(frame_count=2, source_episode_id=7)
    preflight = SimpleNamespace(
        timelines={7: {"camera": object()}},
        camera_schema=SimpleNamespace(source_to_target={"camera": "observation.images.ego_view"}),
    )

    with pytest.raises(ValueError) as captured:
        _construct_payload(episode, np.zeros((2, 64), dtype=np.float32), preflight)

    assert captured.value.error_class == "provenance_error"
    assert not temporary_video.exists()


def test_preflight_failure_is_episode_classified_and_later_episodes_continue(smoke_lock, tmp_path: Path) -> None:
    fake = FakeComponents()
    original = fake.preflight_repository

    def fail_one(source, episode_ids, metadata, cache_dir, kind):
        context = original(source, episode_ids, metadata, cache_dir, kind)
        return RepositoryPreflight(
            context=context,
            episode_failures={78: ClassifiedFailure("timeline_error", "required primary camera frame mismatch")},
        )

    report = run_dex3_pipeline(
        lock=smoke_lock,
        output_root=tmp_path,
        components=replace(fake.build(), preflight_repository=fail_one),
        smoke=True,
    )

    assert report.episode_reports[1].error_class == "timeline_error"
    assert fake.adapter_ids == [0, 155, 233, 310]
    assert report.target_dataset_paths == ()


def test_missing_selected_source_episode_is_provenance_not_schema() -> None:
    assert (
        _source_load_error_class(ValueError("no episode metadata row for selected episode 78"))
        == "provenance_error"
    )
    assert _source_load_error_class(ValueError("episode metadata parquet is malformed")) == "source_schema_error"


def test_nonfinite_inspire_root_is_quaternion_error() -> None:
    current = [[0.0] * 36, [0.0] * 36]
    desired = [[0.0] * 36, [0.0] * 36]
    current[1][3] = float("nan")

    with pytest.raises(ValueError) as captured:
        _validate_inspire_root_finiteness(current, desired)

    assert captured.value.error_class == "quaternion_error"


def test_missing_inspire_task_is_source_schema_error() -> None:
    with pytest.raises(ValueError) as captured:
        _resolve_source_task({0: "pick up pillow"}, 9)

    assert captured.value.error_class == "source_schema_error"


def test_inspire_smoke_never_constructs_encoder(smoke_lock, tmp_path: Path) -> None:
    fake = FakeComponents()

    def forbidden_encoder(_artifacts):
        raise AssertionError("encoder constructed for gated Inspire path")

    components = replace(fake.build(), encoder_factory=forbidden_encoder)
    report = run_inspire_diagnostics(
        lock=smoke_lock,
        output_root=tmp_path,
        components=components,
        smoke=True,
    )

    assert report.source_episode_ids == (0, 152, 304, 456, 608)
    assert report.status_counts == {"blocked_unverified": 5}
    assert report.encoder_invocation_count == 0
    assert report.target_dataset_paths == ()
    assert report.gate_reasons == frozenset(INSPIRE_GATE_REASONS)
    assert report.succeeded is True
    assert fake.encoder_constructions == 0


def test_inspire_preflight_does_not_download_or_hash_unread_video_payloads(
    monkeypatch: pytest.MonkeyPatch,
    smoke_lock,
    tmp_path: Path,
) -> None:
    from gear_sonic.data.unitree_conversion import lerobot_v3_source, pipeline as pipeline_module

    source = smoke_lock.inspire
    calls: list[tuple[str, object]] = []
    dataset = SimpleNamespace(
        root=tmp_path,
        revision=source.revision,
        meta=SimpleNamespace(
            revision=source.revision,
            total_episodes=source.episode_count,
            fps=30,
        ),
    )

    def load(*args: object, **kwargs: object) -> object:
        del args
        calls.append(("download_videos", kwargs["download_videos"]))
        return dataset

    def source_hashes(*args: object, **kwargs: object) -> dict[str, str]:
        del args
        calls.append(("include_videos", kwargs["include_videos"]))
        return {"data/chunk-000/file-000.parquet": "1" * 64}

    monkeypatch.setattr(lerobot_v3_source, "load_pinned_v3_episode", load)
    monkeypatch.setattr(pipeline_module, "_load_task_catalog", lambda root: {0: "pick up pillow"})
    monkeypatch.setattr(pipeline_module, "_source_hashes", source_hashes)

    result = pipeline_module._validate_source_metadata(
        source,
        (0,),
        tmp_path,
        "inspire",
    )

    assert result.episode_failures == {}
    assert calls == [("download_videos", False), ("include_videos", False)]


def test_source_task_catalog_reads_lerobot_v3_parquet_index_layout(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from gear_sonic.data.unitree_conversion import pipeline as pipeline_module

    metadata = tmp_path / "meta"
    metadata.mkdir()
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0, 1], type=pa.int64()),
                "__index_level_0__": pa.array(["pick up pillow", "place pillow"], type=pa.string()),
            }
        ),
        metadata / "tasks.parquet",
    )

    assert pipeline_module._load_task_catalog(tmp_path) == {
        0: "pick up pillow",
        1: "place pillow",
    }


def test_source_video_size_uses_named_chw_dimensions_from_public_v3_metadata() -> None:
    from gear_sonic.data.unitree_conversion import pipeline as pipeline_module

    feature = {
        "dtype": "video",
        "shape": [3, 480, 640],
        "names": ["channels", "height", "width"],
    }

    assert pipeline_module._feature_video_size(feature, key="camera") == (640, 480)


def test_data_only_source_paths_do_not_resolve_absent_video_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pathlib import PurePosixPath

    from gear_sonic.data.unitree_conversion import lerobot_v3_source, pipeline as pipeline_module

    data = tmp_path / "data/chunk-000/file-000.parquet"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"selected data")
    monkeypatch.setattr(
        lerobot_v3_source,
        "_read_info",
        lambda root: SimpleNamespace(video_keys=("observation.images.cam_0",)),
    )
    monkeypatch.setattr(lerobot_v3_source, "_selected_episode_row", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        lerobot_v3_source,
        "_selected_paths",
        lambda row, info: (
            PurePosixPath("data/chunk-000/file-000.parquet"),
            PurePosixPath("videos/observation.images.cam_0/chunk-000/file-000.mp4"),
        ),
    )

    assert pipeline_module._selected_source_paths(
        SimpleNamespace(root=tmp_path),
        0,
        include_videos=False,
    ) == (data.resolve(),)


def test_inspire_repository_preflight_skips_camera_decode_for_diagnostic_only(
    smoke_lock,
) -> None:
    from gear_sonic.data.unitree_conversion import pipeline as pipeline_module

    source = smoke_lock.inspire
    dataset = SimpleNamespace(video_segments={})
    metadata = pipeline_module._SourceMetadataContext(
        source=source,
        datasets={0: dataset},
        tasks={0: "pick up pillow"},
        source_hashes={0: {"data/chunk-000/file-000.parquet": "1" * 64}},
    )

    result = pipeline_module._preflight_repository(
        source,
        (0,),
        metadata,
        None,
        "inspire",
    )

    assert result.episode_failures == {}
    assert result.context.timelines == {}
    assert result.context.camera_schema is None


def test_diagnostic_stage_is_immutable_and_provenance_bound(tmp_path: Path) -> None:
    identity = DiagnosticIdentity(
        source_repo_id="unitreerobotics/inspire",
        source_revision="1" * 40,
        source_episode_id=7,
        source_file_sha256={"data/episode.parquet": "2" * 64},
        source_lock_sha256="3" * 64,
        diagnostic_contract_sha256="4" * 64,
        converter_version="test-v1",
    )
    report = SimpleNamespace(
        source_repo_id=identity.source_repo_id,
        source_revision=identity.source_revision,
        source_episode_id=identity.source_episode_id,
        to_dict=lambda: {"status": "blocked_unverified", "value": 1},
    )

    first = _write_diagnostic(tmp_path, identity, report)
    second = _write_diagnostic(tmp_path, identity, report)

    assert first == second
    assert first.name == "diagnostic.json"
    assert (first.parent / "identity.json").is_file()
    assert (first.parent / "checksums.sha256").is_file()
    changed = SimpleNamespace(
        source_repo_id=identity.source_repo_id,
        source_revision=identity.source_revision,
        source_episode_id=identity.source_episode_id,
        to_dict=lambda: {"status": "blocked_unverified", "value": 2},
    )
    with pytest.raises(FileExistsError, match="immutable"):
        _write_diagnostic(tmp_path, identity, changed)


def test_smoke_lock_rejects_full_mode_and_unapproved_sources(smoke_lock, tmp_path: Path) -> None:
    fake = FakeComponents()
    with pytest.raises(ValueError, match="scope.*smoke"):
        run_dex3_pipeline(
            lock=smoke_lock,
            output_root=tmp_path,
            components=fake.build(),
            smoke=False,
        )

    unapproved = replace(smoke_lock.dex3, approved=False)
    bad_lock = replace(smoke_lock, sources=(unapproved, smoke_lock.inspire))
    with pytest.raises(ValueError, match="not approved"):
        run_dex3_pipeline(
            lock=bad_lock,
            output_root=tmp_path,
            components=fake.build(),
            smoke=True,
        )

    unrelated_unapproved = replace(smoke_lock.inspire, approved=False)
    bad_lock = replace(smoke_lock, sources=(smoke_lock.dex3, unrelated_unapproved))
    with pytest.raises(ValueError, match="not approved"):
        run_dex3_pipeline(
            lock=bad_lock,
            output_root=tmp_path,
            components=fake.build(),
            smoke=True,
        )

    wrong_cohort = replace(smoke_lock.dex3, episodes=(0, 1, 2, 3, 4))
    bad_lock = replace(smoke_lock, sources=(wrong_cohort, smoke_lock.inspire))
    with pytest.raises(ValueError, match="deterministic cohort"):
        run_dex3_pipeline(
            lock=bad_lock,
            output_root=tmp_path,
            components=fake.build(),
            smoke=True,
        )

    extra_source = replace(
        smoke_lock.dex3,
        repo_id="unitreerobotics/extra",
        label="dex3-extra",
    )
    bad_lock = replace(smoke_lock, sources=(*smoke_lock.sources, extra_source))
    with pytest.raises(ValueError, match="exactly one Dex3"):
        run_dex3_pipeline(
            lock=bad_lock,
            output_root=tmp_path,
            components=fake.build(),
            smoke=True,
        )


def test_full_lock_merges_repositories_independently_in_repo_order(smoke_lock, tmp_path: Path) -> None:
    source_z = replace(
        smoke_lock.dex3,
        repo_id="unitreerobotics/Z_Dex3",
        episode_count=2,
        episodes=(0, 1),
        label="dex3-z",
    )
    source_a = replace(
        smoke_lock.dex3,
        repo_id="unitreerobotics/A_Dex3",
        episode_count=2,
        episodes=(0, 1),
        label="dex3-a",
    )
    lock = SourceLock(
        version=1,
        scope="full",
        semantic_repo_commit=smoke_lock.semantic_repo_commit,
        encoder=smoke_lock.encoder,
        observation_config=smoke_lock.observation_config,
        sources=(source_z, source_a),
        collections=(
            CollectionMembership(
                slug="unitreerobotics/unifolm-g1-dex3-dataset",
                repo_ids=(source_z.repo_id, source_a.repo_id),
            ),
        ),
    )
    fake = FakeComponents()

    report = run_dex3_pipeline(
        lock=lock,
        output_root=tmp_path,
        components=fake.build(),
        smoke=False,
    )

    assert report.target_dataset_paths == (
        tmp_path / "unitreerobotics--A_Dex3",
        tmp_path / "unitreerobotics--Z_Dex3",
    )
    merge_calls = [call for call in fake.calls if call[0] == "merge"]
    assert merge_calls == [("merge", (0, 1), True), ("merge", (0, 1), True)]


def test_cli_help_exposes_only_reviewed_controls() -> None:
    script = Path("gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(Path.cwd()), environment.get("PYTHONPATH"))))

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    for option in ("source-lock", "output-root", "kind", "smoke", "cache-dir", "resume", "workers"):
        assert option in result.stdout
    assert "unverified" not in result.stdout


def test_cli_rejects_invalid_workers_and_full_mode_on_smoke_lock() -> None:
    with pytest.raises(ValueError, match="workers"):
        _validate_config(ConvertConfig(workers=0))
    with pytest.raises(ValueError, match="scope.*smoke"):
        _validate_config(ConvertConfig(smoke=False))
