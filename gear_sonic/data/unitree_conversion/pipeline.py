"""Fail-closed orchestration for pinned Unitree-to-SONIC conversion cohorts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import Any

from gear_sonic.data.unitree_conversion.contracts import (
    INSPIRE_GATE_REASONS,
    SourceLock,
    SourceSpec,
    validate_revision,
    validate_sha256,
)
from gear_sonic.data.unitree_conversion.provenance import (
    SourceLockSnapshot,
    dump_source_lock,
)

_ERROR_CLASSES = frozenset(
    {
        "provenance_error",
        "source_schema_error",
        "timeline_error",
        "quaternion_error",
        "semantic_gate_error",
        "encoder_contract_error",
        "target_validation_error",
        "replay_acceptance_error",
    }
)


@dataclass(frozen=True)
class ClassifiedFailure:
    """Machine-readable failure captured before or during one episode."""

    error_class: str
    detail: str

    def __post_init__(self) -> None:
        if self.error_class not in _ERROR_CLASSES:
            raise ValueError(f"unsupported error class: {self.error_class!r}")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("classified failure detail must be nonempty")


class _EpisodePreflightError(ValueError):
    def __init__(self, error_class: str, detail: str) -> None:
        if error_class not in _ERROR_CLASSES:
            raise ValueError(f"unsupported error class: {error_class!r}")
        super().__init__(detail)
        self.error_class = error_class


@dataclass(frozen=True)
class RepositoryPreflight:
    """Repository-wide context plus failures assigned to exact episode IDs."""

    context: object
    episode_failures: Mapping[int, ClassifiedFailure]

    def __post_init__(self) -> None:
        failures = dict(sorted(self.episode_failures.items()))
        if any(type(index) is not int or index < 0 for index in failures):
            raise ValueError("preflight failure episode IDs must be nonnegative integers")
        if not all(isinstance(failure, ClassifiedFailure) for failure in failures.values()):
            raise ValueError("preflight failures must be ClassifiedFailure values")
        object.__setattr__(self, "episode_failures", MappingProxyType(failures))


@dataclass(frozen=True)
class DiagnosticIdentity:
    """Immutable source and lock identity for one gated Inspire report."""

    source_repo_id: str
    source_revision: str
    source_episode_id: int
    source_file_sha256: Mapping[str, str]
    source_lock_sha256: str
    diagnostic_contract_sha256: str
    converter_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_repo_id, str) or "/" not in self.source_repo_id:
            raise ValueError("diagnostic source_repo_id must be a repository identifier")
        validate_revision(self.source_revision, field_name="diagnostic source_revision")
        if type(self.source_episode_id) is not int or self.source_episode_id < 0:
            raise ValueError("diagnostic source_episode_id must be a nonnegative integer")
        if not isinstance(self.source_file_sha256, Mapping) or not self.source_file_sha256:
            raise ValueError("diagnostic identity requires source file hashes")
        hashes = dict(sorted(self.source_file_sha256.items()))
        for source_path, digest in hashes.items():
            if (
                not isinstance(source_path, str)
                or not source_path
                or source_path.startswith("/")
                or ".." in Path(source_path).parts
            ):
                raise ValueError("diagnostic source paths must be safe relative paths")
            validate_sha256(digest, field_name=f"diagnostic source hash {source_path}")
        validate_sha256(self.source_lock_sha256, field_name="diagnostic source_lock_sha256")
        validate_sha256(self.diagnostic_contract_sha256, field_name="diagnostic contract SHA-256")
        if not isinstance(self.converter_version, str) or not self.converter_version.strip():
            raise ValueError("diagnostic converter_version must be nonempty")
        object.__setattr__(self, "source_file_sha256", MappingProxyType(hashes))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_repo_id": self.source_repo_id,
            "source_revision": self.source_revision,
            "source_episode_id": self.source_episode_id,
            "source_file_sha256": dict(sorted(self.source_file_sha256.items())),
            "source_lock_sha256": self.source_lock_sha256,
            "diagnostic_contract_sha256": self.diagnostic_contract_sha256,
            "converter_version": self.converter_version,
        }

    @property
    def digest(self) -> str:
        data = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class EncodedEpisode:
    """Episode-level encoding result and exact number of ONNX invocations."""

    value: object
    invocation_count: int

    def __post_init__(self) -> None:
        if type(self.invocation_count) is not int or self.invocation_count < 0:
            raise ValueError("invocation_count must be a nonnegative integer")


class _CountingEncoder:
    """Count attempted frame-level encoder calls, including failed calls."""

    def __init__(self, encoder: object) -> None:
        self._encoder = encoder
        self.invocation_count = 0

    def encode(self, tensor: object) -> object:
        self.invocation_count += 1
        return self._encoder.encode(tensor)

    def __getattr__(self, name: str) -> object:
        return getattr(self._encoder, name)


@dataclass(frozen=True)
class EpisodePipelineReport:
    """One selected source episode's terminal pipeline classification."""

    source_repo_id: str
    source_episode_id: int
    status: str
    detail: str | None = None
    error_class: str | None = None
    artifact_path: Path | None = None
    encoder_invocation_count: int = 0

    def __post_init__(self) -> None:
        if self.status not in {"validated", "reused", "failed", "blocked_unverified"}:
            raise ValueError(f"unsupported episode status: {self.status!r}")
        if type(self.source_episode_id) is not int or self.source_episode_id < 0:
            raise ValueError("source_episode_id must be a nonnegative integer")
        if type(self.encoder_invocation_count) is not int or self.encoder_invocation_count < 0:
            raise ValueError("encoder_invocation_count must be a nonnegative integer")
        if self.status == "failed" and self.error_class not in _ERROR_CLASSES:
            raise ValueError("failed episode reports require an approved error_class")
        if self.status == "blocked_unverified" and self.error_class != "semantic_gate_error":
            raise ValueError("blocked Inspire reports require semantic_gate_error")
        if self.status in {"validated", "reused"} and self.error_class is not None:
            raise ValueError("successful episode reports must not contain an error_class")

    def to_dict(self) -> dict[str, object]:
        return {
            "source_repo_id": self.source_repo_id,
            "source_episode_id": self.source_episode_id,
            "status": self.status,
            "detail": self.detail,
            "error_class": self.error_class,
            "artifact_path": None if self.artifact_path is None else str(self.artifact_path),
            "encoder_invocation_count": self.encoder_invocation_count,
        }


@dataclass(frozen=True)
class PipelineReport:
    """Deterministic summary of one source-specific pipeline invocation."""

    source_episode_ids: tuple[int, ...]
    validated_episode_count: int
    failed_episode_count: int
    status_counts: Mapping[str, int]
    gate_reasons: frozenset[str]
    encoder_invocation_count: int
    target_dataset_paths: tuple[Path, ...]
    output_root: Path
    episode_reports: tuple[EpisodePipelineReport, ...]

    def __post_init__(self) -> None:
        if tuple(report.source_episode_id for report in self.episode_reports) != self.source_episode_ids:
            raise ValueError("episode_reports must preserve exact source episode order")
        expected = dict(sorted(Counter(report.status for report in self.episode_reports).items()))
        if dict(self.status_counts) != expected:
            raise ValueError("status_counts must exactly summarize episode_reports")
        validated = sum(report.status in {"validated", "reused"} for report in self.episode_reports)
        failed = sum(report.status == "failed" for report in self.episode_reports)
        if self.validated_episode_count != validated or self.failed_episode_count != failed:
            raise ValueError("episode counts must exactly summarize episode_reports")
        invocations = sum(report.encoder_invocation_count for report in self.episode_reports)
        if self.encoder_invocation_count != invocations:
            raise ValueError("encoder_invocation_count must exactly summarize episode_reports")

    @property
    def succeeded(self) -> bool:
        if self.failed_episode_count:
            return False
        terminal = {report.status for report in self.episode_reports}
        if terminal == {"blocked_unverified"}:
            return len(self.episode_reports) == len(self.source_episode_ids) and bool(self.episode_reports)
        return self.validated_episode_count == len(self.source_episode_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_episode_ids": list(self.source_episode_ids),
            "validated_episode_count": self.validated_episode_count,
            "failed_episode_count": self.failed_episode_count,
            "status_counts": dict(self.status_counts),
            "gate_reasons": sorted(self.gate_reasons),
            "encoder_invocation_count": self.encoder_invocation_count,
            "target_dataset_paths": [str(path) for path in self.target_dataset_paths],
            "output_root": str(self.output_root),
            "succeeded": self.succeeded,
            "episode_reports": [report.to_dict() for report in self.episode_reports],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        target.write_text(serialized, encoding="utf-8")
        return target


@dataclass(frozen=True)
class PipelineComponents:
    """Injected operations that keep orchestration independently testable."""

    validate_artifacts: Callable[[SourceLock, Path | None], object]
    validate_source_metadata: Callable[[SourceSpec, tuple[int, ...], Path | None, str], object]
    preflight_repository: Callable[[SourceSpec, tuple[int, ...], object, Path | None, str], object]
    make_stage_identity: Callable[[str, SourceSpec, int, object, object], object]
    resume_stage: Callable[[Path, object], object | None]
    adapt_dex3_episode: Callable[[SourceSpec, int, object, Path | None], object]
    resample_episode: Callable[[object, object], object]
    encoder_factory: Callable[[object], object]
    encode_episode: Callable[[object, object], object]
    construct_payload: Callable[[object, object, object], object]
    validate_payload: Callable[[object], object]
    write_stage: Callable[[Path, object, object], object]
    merge_stages: Callable[[Sequence[object], Path, bool], Path]
    diagnose_inspire_episode: Callable[[SourceSpec, int, object, Path | None], object]
    make_diagnostic_identity: Callable[[str, SourceSpec, int, object], object]
    write_diagnostic: Callable[[Path, object, object], Path]


def _lock_and_sha256(lock: SourceLock | SourceLockSnapshot) -> tuple[SourceLock, str]:
    if isinstance(lock, SourceLockSnapshot):
        return lock.lock, lock.sha256
    if not isinstance(lock, SourceLock):
        raise TypeError("lock must be a SourceLock or SourceLockSnapshot")
    serialized = dump_source_lock(lock).encode("utf-8")
    return lock, hashlib.sha256(serialized).hexdigest()


def _selected_sources(lock: SourceLock, *, kind: str, smoke: bool) -> tuple[SourceSpec, ...]:
    if type(smoke) is not bool:
        raise TypeError("smoke must be a boolean")
    if lock.scope == "smoke" and not smoke:
        raise ValueError("a scope: smoke source lock requires smoke=True")
    if lock.scope == "full" and smoke:
        raise ValueError("a scope: full source lock requires smoke=False")
    if lock.scope == "smoke" and (
        len(lock.sources) != 2 or {source.label for source in lock.sources} != {"dex3", "inspire"}
    ):
        raise ValueError("scope: smoke source lock must contain exactly one Dex3 and one Inspire source")
    for source in lock.sources:
        if not source.approved:
            raise ValueError(f"source {source.repo_id!r} is not approved")
        if lock.scope == "smoke":
            from gear_sonic.data.unitree_conversion.provenance import stratified_episode_ids

            expected = stratified_episode_ids(source.episode_count)
            if len(source.episodes) != 5 or source.episodes != expected:
                raise ValueError(
                    f"scope: smoke source {source.repo_id!r} episodes must equal deterministic cohort {expected}"
                )
    if lock.scope == "full":
        collection_slug = {
            "dex3": "unitreerobotics/unifolm-g1-dex3-dataset",
            "inspire": "unitreerobotics/unifolm-wbt-dataset",
        }[kind]
        selected_repo_ids = {
            repo_id
            for membership in lock.collections
            if membership.slug == collection_slug
            for repo_id in membership.repo_ids
        }
        selected = tuple(
            sorted(
                (source for source in lock.sources if source.repo_id in selected_repo_ids),
                key=lambda item: item.repo_id,
            )
        )
    else:
        selected = tuple(source for source in lock.sources if source.label == kind)
    if not selected:
        raise ValueError(f"source lock has no {kind!r} sources")
    for source in selected:
        if not source.episodes:
            raise ValueError(f"source {source.repo_id!r} has no explicitly eligible episodes")
    return selected


def _repo_output_path(output_root: Path, source: SourceSpec) -> Path:
    safe_repo = source.repo_id.replace("/", "--")
    return output_root / safe_repo


def _encoded_parts(encoded: object) -> tuple[object, int]:
    if isinstance(encoded, EncodedEpisode):
        return encoded.value, encoded.invocation_count
    count = getattr(encoded, "invocation_count", None)
    if type(count) is not int or count < 0:
        raise ValueError("encode_episode must return a value with nonnegative invocation_count")
    return encoded, count


def _preflight_parts(value: object) -> tuple[object, Mapping[int, ClassifiedFailure]]:
    if isinstance(value, RepositoryPreflight):
        return value.context, value.episode_failures
    return value, MappingProxyType({})


def _failure(error_class: str, error: BaseException) -> ClassifiedFailure:
    return ClassifiedFailure(
        error_class=error_class,
        detail=f"{type(error).__name__}: {error}",
    )


def _phase_error_class(phase: str, error: BaseException) -> str:
    if isinstance(error, _EpisodePreflightError):
        return error.error_class
    return {
        "identity": "provenance_error",
        "resume": "provenance_error",
        "adapt": "source_schema_error",
        "resample": "timeline_error",
        "encoder": "encoder_contract_error",
        "construct": "target_validation_error",
        "validate": "target_validation_error",
        "stage": "target_validation_error",
    }.get(phase, "target_validation_error")


def _source_load_error_class(error: BaseException) -> str:
    if str(error).startswith("no episode metadata row for selected episode "):
        return "provenance_error"
    if type(error).__name__ in {"EntryNotFoundError", "RevisionNotFoundError"}:
        return "provenance_error"
    return "source_schema_error"


def _resolve_source_task(tasks: Mapping[int, str], task_index: int) -> str:
    try:
        task = tasks[task_index]
    except KeyError as error:
        raise _EpisodePreflightError(
            "source_schema_error",
            f"source task_index {task_index} is absent from the task catalog",
        ) from error
    return task


def _pipeline_report(
    *,
    output_root: Path,
    episode_reports: list[EpisodePipelineReport],
    target_paths: list[Path],
    gate_reasons: frozenset[str] = frozenset(),
) -> PipelineReport:
    reports = tuple(episode_reports)
    return PipelineReport(
        source_episode_ids=tuple(report.source_episode_id for report in reports),
        validated_episode_count=sum(report.status in {"validated", "reused"} for report in reports),
        failed_episode_count=sum(report.status == "failed" for report in reports),
        status_counts=dict(sorted(Counter(report.status for report in reports).items())),
        gate_reasons=gate_reasons,
        encoder_invocation_count=sum(report.encoder_invocation_count for report in reports),
        target_dataset_paths=tuple(target_paths),
        output_root=output_root,
        episode_reports=reports,
    )


def run_dex3_pipeline(
    *,
    lock: SourceLock | SourceLockSnapshot,
    output_root: str | Path,
    components: PipelineComponents | None = None,
    smoke: bool = True,
    cache_dir: str | Path | None = None,
    resume: bool = True,
) -> PipelineReport:
    """Convert every explicitly selected Dex3 episode, grouped by repository."""
    parsed_lock, lock_sha256 = _lock_and_sha256(lock)
    sources = _selected_sources(parsed_lock, kind="dex3", smoke=smoke)
    root = Path(output_root)
    cache = None if cache_dir is None else Path(cache_dir)
    operations = _default_dex3_components(parsed_lock, cache) if components is None else components

    artifacts = operations.validate_artifacts(parsed_lock, cache)
    episode_reports: list[EpisodePipelineReport] = []
    target_paths: list[Path] = []
    encoder: _CountingEncoder | None = None

    for source in sources:
        episode_ids = tuple(source.episodes)
        metadata_result = operations.validate_source_metadata(source, episode_ids, cache, "dex3")
        preflight_result = operations.preflight_repository(source, episode_ids, metadata_result, cache, "dex3")
        preflight, preflight_failures = _preflight_parts(preflight_result)
        stages: list[object] = []
        repository_reports: list[EpisodePipelineReport] = []
        for episode_id in episode_ids:
            invocation_count = 0
            encoder_start_count = 0 if encoder is None else encoder.invocation_count
            preflight_failure = preflight_failures.get(episode_id)
            if preflight_failure is not None:
                report = EpisodePipelineReport(
                    source_repo_id=source.repo_id,
                    source_episode_id=episode_id,
                    status="failed",
                    detail=preflight_failure.detail,
                    error_class=preflight_failure.error_class,
                )
                repository_reports.append(report)
                episode_reports.append(report)
                continue
            phase = "identity"
            try:
                identity = operations.make_stage_identity(
                    lock_sha256,
                    source,
                    episode_id,
                    artifacts,
                    preflight,
                )
                phase = "resume"
                resumed = operations.resume_stage(root, identity) if resume else None
                if resumed is not None:
                    stages.append(resumed)
                    report = EpisodePipelineReport(
                        source_repo_id=source.repo_id,
                        source_episode_id=episode_id,
                        status="reused",
                        artifact_path=Path(resumed.path),
                    )
                else:
                    phase = "adapt"
                    canonical = operations.adapt_dex3_episode(source, episode_id, preflight, cache)
                    phase = "resample"
                    resampled = operations.resample_episode(canonical, preflight)
                    phase = "encoder"
                    if encoder is None:
                        encoder = _CountingEncoder(operations.encoder_factory(artifacts))
                        encoder_start_count = 0
                    encoded = operations.encode_episode(encoder, resampled)
                    encoded_value, reported_invocations = _encoded_parts(encoded)
                    counted_invocations = encoder.invocation_count - encoder_start_count
                    invocation_count = counted_invocations or reported_invocations
                    if counted_invocations and counted_invocations != reported_invocations:
                        raise ValueError("encode_episode invocation_count differs from observed encoder calls")
                    phase = "construct"
                    payload = operations.construct_payload(resampled, encoded_value, preflight)
                    phase = "validate"
                    operations.validate_payload(payload)
                    phase = "stage"
                    stage = operations.write_stage(root, identity, payload)
                    stages.append(stage)
                    report = EpisodePipelineReport(
                        source_repo_id=source.repo_id,
                        source_episode_id=episode_id,
                        status="validated",
                        artifact_path=Path(stage.path),
                        encoder_invocation_count=invocation_count,
                    )
            except Exception as error:  # classify per episode and continue the fixed cohort
                if encoder is not None:
                    invocation_count = max(
                        invocation_count,
                        encoder.invocation_count - encoder_start_count,
                    )
                report = EpisodePipelineReport(
                    source_repo_id=source.repo_id,
                    source_episode_id=episode_id,
                    status="failed",
                    detail=f"{type(error).__name__}: {error}",
                    error_class=_phase_error_class(phase, error),
                    encoder_invocation_count=invocation_count,
                )
            repository_reports.append(report)
            episode_reports.append(report)

        if all(report.status in {"validated", "reused"} for report in repository_reports):
            target = _repo_output_path(root, source)
            target_paths.append(operations.merge_stages(tuple(stages), target, resume))

    return _pipeline_report(
        output_root=root,
        episode_reports=episode_reports,
        target_paths=target_paths,
    )


def run_inspire_diagnostics(
    *,
    lock: SourceLock | SourceLockSnapshot,
    output_root: str | Path,
    components: PipelineComponents | None = None,
    smoke: bool = True,
    cache_dir: str | Path | None = None,
) -> PipelineReport:
    """Audit Inspire sources without importing, constructing, or invoking SONIC."""
    parsed_lock, lock_sha256 = _lock_and_sha256(lock)
    sources = _selected_sources(parsed_lock, kind="inspire", smoke=smoke)
    root = Path(output_root)
    cache = None if cache_dir is None else Path(cache_dir)
    operations = _default_inspire_components() if components is None else components

    episode_reports: list[EpisodePipelineReport] = []
    for source in sources:
        episode_ids = tuple(source.episodes)
        metadata_result = operations.validate_source_metadata(source, episode_ids, cache, "inspire")
        preflight_result = operations.preflight_repository(source, episode_ids, metadata_result, cache, "inspire")
        preflight, preflight_failures = _preflight_parts(preflight_result)
        for episode_id in episode_ids:
            preflight_failure = preflight_failures.get(episode_id)
            if preflight_failure is not None:
                episode_reports.append(
                    EpisodePipelineReport(
                        source_repo_id=source.repo_id,
                        source_episode_id=episode_id,
                        status="failed",
                        detail=preflight_failure.detail,
                        error_class=preflight_failure.error_class,
                    )
                )
                continue
            phase = "identity"
            try:
                identity = operations.make_diagnostic_identity(
                    lock_sha256,
                    source,
                    episode_id,
                    preflight,
                )
                phase = "diagnose"
                diagnostic = operations.diagnose_inspire_episode(source, episode_id, preflight, cache)
                phase = "semantic_gate"
                if getattr(diagnostic, "status", None) == "source_schema_error":
                    raise _EpisodePreflightError(
                        "quaternion_error",
                        "Inspire root quaternion validation failed",
                    )
                if getattr(diagnostic, "status", None) != "blocked_unverified":
                    raise ValueError("Inspire diagnostic must remain blocked_unverified")
                if getattr(diagnostic, "encoder_invoked", None) is not False:
                    raise ValueError("Inspire diagnostic must prove encoder_invoked is false")
                if tuple(getattr(diagnostic, "gate_reasons", ())) != INSPIRE_GATE_REASONS:
                    raise ValueError("Inspire diagnostic gate reasons differ from the exact contract")
                phase = "write"
                artifact = operations.write_diagnostic(root, identity, diagnostic)
                report = EpisodePipelineReport(
                    source_repo_id=source.repo_id,
                    source_episode_id=episode_id,
                    status="blocked_unverified",
                    error_class="semantic_gate_error",
                    artifact_path=artifact,
                )
            except Exception as error:
                error_class = (
                    error.error_class
                    if isinstance(error, _EpisodePreflightError)
                    else {
                        "identity": "provenance_error",
                        "diagnose": "source_schema_error",
                        "semantic_gate": "semantic_gate_error",
                        "write": "target_validation_error",
                    }[phase]
                )
                report = EpisodePipelineReport(
                    source_repo_id=source.repo_id,
                    source_episode_id=episode_id,
                    status="failed",
                    detail=f"{type(error).__name__}: {error}",
                    error_class=error_class,
                )
            episode_reports.append(report)

    return _pipeline_report(
        output_root=root,
        episode_reports=episode_reports,
        target_paths=[],
        gate_reasons=frozenset(INSPIRE_GATE_REASONS),
    )


def _unavailable(*_args: object, **_kwargs: object) -> Any:
    raise RuntimeError("operation is unavailable in this source-specific pipeline")


@dataclass(frozen=True)
class _ResolvedArtifacts:
    encoder_path: Path
    observation_config_path: Path


@dataclass(frozen=True)
class _SourceMetadataContext:
    source: SourceSpec
    datasets: Mapping[int, object]
    tasks: Mapping[int, str]
    source_hashes: Mapping[int, Mapping[str, str]]


@dataclass(frozen=True)
class _RepositoryPreflight:
    metadata: _SourceMetadataContext
    timelines: Mapping[int, Mapping[str, object]]
    camera_schema: object

    @property
    def source(self) -> SourceSpec:
        return self.metadata.source

    @property
    def datasets(self) -> Mapping[int, object]:
        return self.metadata.datasets

    @property
    def tasks(self) -> Mapping[int, str]:
        return self.metadata.tasks

    @property
    def source_hashes(self) -> Mapping[int, Mapping[str, str]]:
        return self.metadata.source_hashes


@dataclass(frozen=True)
class _ProductionPayload:
    payload: object
    temporary_videos: tuple[Path, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_downloader(cache_dir: Path | None) -> Callable[..., str | Path]:
    def download(**kwargs: object) -> str | Path:
        from huggingface_hub import hf_hub_download

        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        return hf_hub_download(**kwargs)

    return download


def _validate_default_artifacts(lock: SourceLock, cache_dir: Path | None) -> _ResolvedArtifacts:
    from gear_sonic.data.unitree_conversion.provenance import materialize_artifact

    downloader = _artifact_downloader(cache_dir)
    return _ResolvedArtifacts(
        encoder_path=materialize_artifact(lock.encoder, downloader=downloader),
        observation_config_path=materialize_artifact(lock.observation_config, downloader=downloader),
    )


def _feature_video_size(feature: object, *, key: str) -> tuple[int, int]:
    if not isinstance(feature, Mapping) or feature.get("dtype") != "video":
        raise ValueError(f"source camera {key!r} must be a video feature")
    shape = feature.get("shape")
    names = feature.get("names")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
        or not isinstance(names, Sequence)
        or isinstance(names, (str, bytes))
        or len(names) != 3
        or any(not isinstance(value, str) for value in names)
    ):
        raise ValueError(f"source camera {key!r} must declare named positive RGB dimensions")
    normalized_names = tuple("channels" if name == "channel" else name for name in names)
    if set(normalized_names) != {"channels", "height", "width"}:
        raise ValueError(f"source camera {key!r} must name channel, height, and width dimensions")
    dimensions = dict(zip(normalized_names, shape, strict=True))
    if dimensions["channels"] != 3:
        raise ValueError(f"source camera {key!r} must declare exactly three RGB channels")
    return int(dimensions["width"]), int(dimensions["height"])


def _load_task_catalog(dataset_root: Path) -> Mapping[int, str]:
    jsonl_path = dataset_root / "meta" / "tasks.jsonl"
    parquet_path = dataset_root / "meta" / "tasks.parquet"
    existing = tuple(path for path in (jsonl_path, parquet_path) if path.is_file())
    if len(existing) != 1:
        raise ValueError("source metadata must contain exactly one supported task catalog")

    records: list[dict[str, object]] = []
    if existing[0] == jsonl_path:
        try:
            lines = jsonl_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise ValueError("source meta/tasks.jsonl must be readable UTF-8") from error
        if not lines:
            raise ValueError("source task catalog must not be empty")
        for line_number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"source tasks line {line_number} must be valid JSON") from error
            if not isinstance(record, dict) or set(record) != {"task_index", "task"}:
                raise ValueError("source task records must contain exactly task_index and task")
            records.append(record)
    else:
        import pyarrow.parquet as pq

        try:
            table = pq.read_table(parquet_path)
        except Exception as error:
            raise ValueError("source meta/tasks.parquet must be readable") from error
        column_names = set(table.column_names)
        if column_names == {"task_index", "task"}:
            task_column = "task"
        elif column_names == {"task_index", "__index_level_0__"}:
            task_column = "__index_level_0__"
        else:
            raise ValueError("source tasks.parquet has an unsupported exact schema")
        records = [
            {
                "task_index": table["task_index"][row_index].as_py(),
                "task": table[task_column][row_index].as_py(),
            }
            for row_index in range(table.num_rows)
        ]
        if not records:
            raise ValueError("source task catalog must not be empty")

    tasks: dict[int, str] = {}
    for record in records:
        index = record["task_index"]
        text = record["task"]
        if type(index) is not int or index < 0 or not isinstance(text, str) or not text.strip():
            raise ValueError("source task records require a nonnegative integer index and nonempty task")
        if index in tasks:
            raise ValueError("source task records must have unique task_index values")
        tasks[index] = text
    if tuple(sorted(tasks)) != tuple(range(len(tasks))):
        raise ValueError("source task indices must be contiguous from zero")
    return MappingProxyType(dict(sorted(tasks.items())))


def _selected_source_paths(
    dataset: object,
    episode_id: int,
    *,
    include_videos: bool = True,
) -> tuple[Path, ...]:
    from gear_sonic.data.unitree_conversion.lerobot_v3_source import (
        _read_info,
        _selected_episode_row,
        _selected_paths,
    )

    root = Path(dataset.root)
    info = _read_info(root)
    row = _selected_episode_row(root, episode_id=episode_id, video_keys=info.video_keys)
    relative = _selected_paths(row, info)
    if not include_videos:
        relative = relative[:1]
    paths = tuple((root / path).resolve(strict=True) for path in relative)
    if any(root.resolve() not in path.parents for path in paths):
        raise ValueError("selected source artifact escaped the pinned dataset root")
    return paths


def _source_hashes(
    dataset: object,
    episode_id: int,
    *,
    include_videos: bool = True,
) -> Mapping[str, str]:
    root = Path(dataset.root).resolve()
    selected = _selected_source_paths(dataset, episode_id, include_videos=include_videos)
    metadata_root = root / "meta"
    metadata = tuple(path for path in sorted(metadata_root.rglob("*")) if path.is_file())
    if not metadata:
        raise ValueError("pinned source metadata tree must contain regular files")
    paths = tuple(dict.fromkeys((*metadata, *selected)))
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError("source artifacts must not be symbolic links")
        resolved = path.resolve(strict=True)
        if root not in resolved.parents:
            raise ValueError("source artifact escaped the pinned dataset root")
        relative = resolved.relative_to(root).as_posix()
        hashes[relative] = _sha256_file(resolved)
    return MappingProxyType(dict(sorted(hashes.items())))


def _validate_dex3_metadata(source: SourceSpec, dataset: object) -> None:
    from gear_sonic.data.unitree_conversion.dex3_adapter import DEX3_FEATURE_NAMES

    meta = dataset.meta
    if dataset.revision != source.revision or meta.revision != source.revision:
        raise _EpisodePreflightError(
            "provenance_error",
            "loaded source revision differs from the pinned revision",
        )
    if meta.total_episodes != source.episode_count:
        raise _EpisodePreflightError(
            "provenance_error",
            "source episode count differs from the pinned contract",
        )
    if meta.fps != 30:
        raise _EpisodePreflightError("timeline_error", "source fps differs from exact 30 Hz")
    for key in ("observation.state", "action"):
        feature = meta.features.get(key) if isinstance(meta.features, Mapping) else None
        names = feature.get("names") if isinstance(feature, Mapping) else None
        if isinstance(names, Sequence) and len(names) == 1 and isinstance(names[0], Sequence):
            names = names[0]
        if tuple(names or ()) != DEX3_FEATURE_NAMES:
            raise _EpisodePreflightError(
                "source_schema_error",
                f"source {key} names differ from the exact Dex3 order",
            )


def _validate_source_metadata(
    source: SourceSpec,
    episode_ids: tuple[int, ...],
    cache_dir: Path | None,
    kind: str,
) -> RepositoryPreflight:
    from gear_sonic.data.unitree_conversion.lerobot_v3_source import (
        DEX3_DATA_SCHEMA,
        default_lerobot_cache_base,
        load_pinned_v3_episode,
    )

    if kind not in {"dex3", "inspire"}:
        raise ValueError(f"unsupported source kind: {kind!r}")
    schema = DEX3_DATA_SCHEMA
    if kind == "inspire":
        from gear_sonic.data.unitree_conversion.inspire_diagnostics import _SOURCE_SCHEMA

        schema = _SOURCE_SCHEMA
    cache_base = default_lerobot_cache_base() if cache_dir is None else cache_dir / "lerobot"
    datasets: dict[int, object] = {}
    source_hashes: dict[int, Mapping[str, str]] = {}
    task_catalog: Mapping[int, str] | None = None
    failures: dict[int, ClassifiedFailure] = {}

    for episode_id in episode_ids:
        phase = "load"
        try:
            dataset = load_pinned_v3_episode(
                source,
                episode_id,
                cache_base=cache_base,
                schema=schema,
                download_videos=kind == "dex3",
            )
            phase = "identity"
            if dataset.revision != source.revision or dataset.meta.revision != source.revision:
                raise _EpisodePreflightError(
                    "provenance_error",
                    "loaded source revision differs from the pinned revision",
                )
            if dataset.meta.total_episodes != source.episode_count:
                raise _EpisodePreflightError(
                    "provenance_error",
                    "source episode count differs from the pinned contract",
                )
            if dataset.meta.fps != 30:
                raise _EpisodePreflightError("timeline_error", "source fps differs from exact 30 Hz")
            if kind == "dex3":
                _validate_dex3_metadata(source, dataset)
            phase = "tasks"
            current_tasks = _load_task_catalog(Path(dataset.root))
            if task_catalog is None:
                task_catalog = current_tasks
            elif dict(task_catalog) != dict(current_tasks):
                raise ValueError("source task catalog changed within one repository preflight")

            datasets[episode_id] = dataset
            phase = "hash"
            source_hashes[episode_id] = _source_hashes(
                dataset,
                episode_id,
                include_videos=kind == "dex3",
            )
        except Exception as error:
            if isinstance(error, _EpisodePreflightError):
                failures[episode_id] = ClassifiedFailure(error.error_class, str(error))
            else:
                error_class = (
                    "provenance_error"
                    if phase == "hash"
                    else _source_load_error_class(error)
                    if phase == "load"
                    else "source_schema_error"
                )
                failures[episode_id] = _failure(error_class, error)

    if task_catalog is None:
        task_catalog = MappingProxyType({})
    context = _SourceMetadataContext(
        source=source,
        datasets=MappingProxyType(datasets),
        tasks=task_catalog,
        source_hashes=MappingProxyType(source_hashes),
    )
    return RepositoryPreflight(context=context, episode_failures=failures)


def _preflight_repository(
    source: SourceSpec,
    episode_ids: tuple[int, ...],
    metadata_result: object,
    cache_dir: Path | None,
    kind: str,
) -> RepositoryPreflight:
    del cache_dir
    from gear_sonic.data.unitree_conversion.video_timeline import (
        CameraStreamReport,
        EpisodeCameraReport,
        choose_target_camera_schema,
        inspect_video,
    )

    metadata_value, existing_failures = _preflight_parts(metadata_result)
    if not isinstance(metadata_value, _SourceMetadataContext):
        raise TypeError("source metadata validation must return _SourceMetadataContext")
    failures = dict(existing_failures)
    if kind == "inspire":
        return RepositoryPreflight(
            context=_RepositoryPreflight(
                metadata=metadata_value,
                timelines=MappingProxyType({}),
                camera_schema=None,
            ),
            episode_failures=failures,
        )
    if kind != "dex3":
        raise ValueError(f"unsupported source kind: {kind!r}")
    timelines: dict[int, Mapping[str, object]] = {}
    reports: list[EpisodeCameraReport] = []

    for episode_id in episode_ids:
        if episode_id in failures:
            continue
        try:
            dataset = metadata_value.datasets[episode_id]
            streams: dict[str, CameraStreamReport] = {}
            exact_timelines: dict[str, object] = {}
            for source_key in source.camera_map:
                segment = dataset.video_segments.get(source_key)
                if segment is None:
                    streams[source_key] = CameraStreamReport(status="missing", reason="source stream absent")
                    continue
                try:
                    size = _feature_video_size(dataset.meta.features.get(source_key), key=source_key)
                    timeline = inspect_video(
                        segment.path,
                        expected_frames=len(dataset.hf_dataset),
                        expected_size=size,
                        segment=segment,
                    )
                except ValueError as error:
                    streams[source_key] = CameraStreamReport(status="invalid", reason=str(error))
                else:
                    streams[source_key] = CameraStreamReport(status="exact", timeline=timeline)
                    exact_timelines[source_key] = timeline
            primary = streams.get(source.primary_camera)
            if primary is None or primary.status != "exact":
                reason = "source stream absent" if primary is None else primary.reason
                error_class = (
                    "source_schema_error" if primary is None or primary.status == "missing" else "timeline_error"
                )
                raise _EpisodePreflightError(
                    error_class,
                    f"required primary camera {source.primary_camera!r} failed: {reason}",
                )
            reports.append(
                EpisodeCameraReport(
                    source_repo_id=source.repo_id,
                    source_episode_id=episode_id,
                    source_frame_count=len(dataset.hf_dataset),
                    primary_camera=source.primary_camera,
                    streams=streams,
                )
            )
            timelines[episode_id] = MappingProxyType(dict(sorted(exact_timelines.items())))
        except Exception as error:
            failures[episode_id] = (
                ClassifiedFailure(error.error_class, str(error))
                if isinstance(error, _EpisodePreflightError)
                else _failure("timeline_error", error)
            )

    camera_schema: object | None = None
    if reports:
        try:
            camera_schema = choose_target_camera_schema(reports, source.camera_map)
        except Exception as error:
            failure = _failure("source_schema_error", error)
            for report in reports:
                failures[report.source_episode_id] = failure
    context = _RepositoryPreflight(
        metadata=metadata_value,
        timelines=MappingProxyType(timelines),
        camera_schema=camera_schema,
    )
    return RepositoryPreflight(context=context, episode_failures=failures)


def _make_stage_identity(
    lock_sha256: str,
    source: SourceSpec,
    episode_id: int,
    artifacts: _ResolvedArtifacts,
    preflight: _RepositoryPreflight,
) -> object:
    from gear_sonic.data.unitree_conversion.staging import StageIdentity

    conversion_contract = json.dumps(
        {
            "body_fps": [30, 50],
            "encoder_layout": 1247,
            "motion_token": 64,
            "source_kind": "dex3",
            "source_to_target_camera": dict(preflight.camera_schema.source_to_target),
            "target_schema": "g1-sonic-vla-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return StageIdentity(
        source_repo_id=source.repo_id,
        source_revision=source.revision,
        source_episode_id=episode_id,
        source_file_sha256=preflight.source_hashes[episode_id],
        conversion_config_sha256=hashlib.sha256(conversion_contract).hexdigest(),
        converter_version="unitree-sonic-conversion-v1",
        encoder_sha256=_sha256_file(artifacts.encoder_path),
        encoder_config_sha256=_sha256_file(artifacts.observation_config_path),
        source_lock_sha256=lock_sha256,
    )


def _resume_stage(output_root: Path, identity: object) -> object | None:
    from gear_sonic.data.unitree_conversion.staging import (
        StageIdentity,
        StageResult,
        can_resume,
        stage_path,
    )

    if not isinstance(identity, StageIdentity):
        raise TypeError("stage identity must be StageIdentity")
    path = stage_path(output_root, identity)
    if not can_resume(path, identity):
        return None
    manifest_digest = hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
    return StageResult(path=path, identity=identity, manifest_digest=manifest_digest, reused=True)


def _validate_dex3_row_indices(rows: object, episode_id: int) -> None:
    import numpy as np

    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise _EpisodePreflightError("source_schema_error", f"Dex3 row {row_number} must be a mapping")
        for field_name, expected in (("episode_index", episode_id), ("frame_index", row_number)):
            if field_name not in row:
                raise _EpisodePreflightError(
                    "source_schema_error",
                    f"Dex3 row {row_number} is missing {field_name}",
                )
            value = np.asarray(row[field_name])
            if value.shape != () or value.dtype.kind not in "iu" or int(value) != expected:
                raise _EpisodePreflightError(
                    "source_schema_error",
                    f"Dex3 row {row_number} {field_name} must equal {expected}",
                )


def _verify_preflight_source_hashes(
    preflight: _RepositoryPreflight,
    episode_id: int,
    *,
    operation: str,
) -> None:
    try:
        dataset = preflight.datasets[episode_id]
        expected = dict(preflight.source_hashes[episode_id])
        observed = dict(_source_hashes(dataset, episode_id))
    except Exception as error:
        raise _EpisodePreflightError(
            "provenance_error",
            f"source files could not be reverified {operation}: {type(error).__name__}: {error}",
        ) from error
    if observed != expected:
        raise _EpisodePreflightError("provenance_error", f"source files changed {operation}")


def _adapt_dex3(
    source: SourceSpec,
    episode_id: int,
    preflight: _RepositoryPreflight,
    cache_dir: Path | None,
) -> object:
    import numpy as np

    from gear_sonic.data.unitree_conversion.dex3_adapter import (
        DEX3_FEATURE_NAMES,
        adapt_dex3_arrays,
    )
    from gear_sonic.data.unitree_conversion.lerobot_v3_source import (
        default_lerobot_cache_base,
        load_pinned_v3_episode,
    )

    _verify_preflight_source_hashes(preflight, episode_id, operation="before Dex3 adaptation")
    cache_base = default_lerobot_cache_base() if cache_dir is None else cache_dir / "lerobot"
    try:
        dataset = load_pinned_v3_episode(
            source,
            episode_id,
            cache_base=cache_base,
            download_videos=False,
        )
    except Exception as error:
        raise _EpisodePreflightError(
            "provenance_error",
            f"fresh Dex3 source read failed: {type(error).__name__}: {error}",
        ) from error
    original_dataset = preflight.datasets[episode_id]
    if Path(dataset.root).resolve() != Path(original_dataset.root).resolve():
        raise _EpisodePreflightError(
            "provenance_error",
            "fresh Dex3 read resolved to a different pinned dataset root",
        )
    _validate_dex3_metadata(source, dataset)
    _verify_preflight_source_hashes(preflight, episode_id, operation="during fresh Dex3 read")
    rows = dataset.hf_dataset
    _validate_dex3_row_indices(rows, episode_id)
    if len(rows) < 2:
        raise _EpisodePreflightError("timeline_error", "Dex3 episode contains fewer than two source frames")
    try:
        timestamps = np.asarray([row["timestamp"] for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise _EpisodePreflightError("timeline_error", "Dex3 timestamps must be finite numeric values") from error
    if not np.isfinite(timestamps).all() or not np.all(np.diff(timestamps) > 0.0):
        raise _EpisodePreflightError(
            "timeline_error",
            "Dex3 timestamps must be finite and strictly increasing",
        )
    episode = adapt_dex3_arrays(
        source_repo_id=source.repo_id,
        source_revision=source.revision,
        source_episode_id=episode_id,
        observed=np.asarray([row["observation.state"] for row in rows]),
        desired=np.asarray([row["action"] for row in rows]),
        feature_names=DEX3_FEATURE_NAMES,
        timestamps=timestamps,
        task_indices=np.asarray([row["task_index"] for row in rows]),
        video_segments=original_dataset.video_segments,
    )
    _verify_preflight_source_hashes(preflight, episode_id, operation="during Dex3 adaptation")
    return episode


def _resample_episode(episode: object, preflight: _RepositoryPreflight) -> object:
    import numpy as np

    from gear_sonic.data.unitree_conversion.contracts import CanonicalEpisode, ResampledEpisode
    from gear_sonic.data.unitree_conversion.joint_mapping import G1_MUJOCO_NAMES
    from gear_sonic.data.unitree_conversion.resampling import (
        audit_source_timestamps,
        nearest_image_indices,
        resample_positions,
        resample_quaternions,
    )

    if not isinstance(episode, CanonicalEpisode):
        raise _EpisodePreflightError("source_schema_error", "adapter must return CanonicalEpisode")
    audit = audit_source_timestamps(episode.timestamps)
    if audit.rejected:
        raise ValueError("source timestamps exceed the exact 30 Hz rejection threshold")
    observed_body, _ = resample_positions(episode.observed_body_q)
    desired_body, desired_velocity = resample_positions(episode.desired_body_q)
    observed_left, _ = resample_positions(episode.observed_left_hand)
    observed_right, _ = resample_positions(episode.observed_right_hand)
    desired_left, _ = resample_positions(episode.desired_left_hand)
    desired_right, _ = resample_positions(episode.desired_right_hand)
    discrete_indices = nearest_image_indices(episode.timestamps.shape[0])
    task_indices = episode.task_indices[discrete_indices]
    task_texts = tuple(_resolve_source_task(preflight.tasks, int(index)) for index in task_indices)
    try:
        observed_roots = resample_quaternions(episode.observed_root_wxyz)
        reference_roots = resample_quaternions(episode.reference_root_wxyz)
    except ValueError as error:
        raise _EpisodePreflightError("quaternion_error", str(error)) from error
    return ResampledEpisode(
        source_repo_id=episode.source_repo_id,
        source_revision=episode.source_revision,
        source_episode_id=episode.source_episode_id,
        body_joint_names=G1_MUJOCO_NAMES,
        task_indices=np.asarray(task_indices, dtype=np.int64),
        task_texts=task_texts,
        observed_root_wxyz=observed_roots,
        reference_root_wxyz=reference_roots,
        observed_body_q=observed_body,
        desired_body_q=desired_body,
        desired_body_velocity=desired_velocity,
        observed_left_hand=observed_left,
        observed_right_hand=observed_right,
        desired_left_hand=desired_left,
        desired_right_hand=desired_right,
    )


def _encoder_factory(artifacts: _ResolvedArtifacts) -> object:
    from gear_sonic.data.unitree_conversion.sonic_encoder import (
        SonicEncoder,
        _parse_observation_config,
    )

    _parse_observation_config(artifacts.observation_config_path)
    import onnxruntime

    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    provider = "CPUExecutionProvider"
    session = onnxruntime.InferenceSession(
        str(artifacts.encoder_path),
        providers=[provider],
        sess_options=options,
    )
    return SonicEncoder.from_session(
        session,
        encoder_path=artifacts.encoder_path,
        observation_config_path=artifacts.observation_config_path,
        provider=provider,
        runtime_version=str(onnxruntime.__version__),
    )


def _encode_episode(encoder: object, episode: object) -> EncodedEpisode:
    import numpy as np

    from gear_sonic.data.unitree_conversion.sonic_encoder import build_frame_encoder_input

    tokens: list[np.ndarray] = []
    for frame_index in range(episode.frame_count):
        token = encoder.encode(build_frame_encoder_input(episode, frame_index))
        array = np.asarray(token)
        if array.shape != (1, 64) or array.dtype != np.dtype(np.float32):
            raise ValueError("encoder must return exact float32 shape [1,64]")
        tokens.append(np.array(array[0], dtype=np.float32, copy=True))
    return EncodedEpisode(
        value=np.stack(tokens),
        invocation_count=episode.frame_count,
    )


def _h264_encoder_name() -> str:
    import av

    for name in ("libx264", "libopenh264", "h264"):
        try:
            av.codec.Codec(name, "w")
        except (av.error.FFmpegError, ValueError):
            continue
        return name
    raise RuntimeError("PyAV has no usable H.264 encoder")


def _encode_target_video(timeline: object) -> Path:
    import av

    from gear_sonic.data.unitree_conversion.video_timeline import iter_resampled_video

    handle = tempfile.NamedTemporaryFile(prefix="unitree-sonic-", suffix=".mp4", delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream(_h264_encoder_name(), rate=50)
            stream.width = 640
            stream.height = 480
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, 50)
            for frame in iter_resampled_video(timeline):
                video_frame = av.VideoFrame.from_ndarray(frame.rgb, format="rgb24")
                video_frame = video_frame.reformat(width=640, height=480, format="yuv420p")
                video_frame.pts = frame.target_index
                video_frame.time_base = Fraction(1, 50)
                for packet in stream.encode(video_frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _construct_payload(episode: object, encoded: object, preflight: _RepositoryPreflight) -> _ProductionPayload:
    import numpy as np

    from gear_sonic.data.unitree_conversion.staging import StagePayload
    from gear_sonic.data.unitree_conversion.target_frames import TargetFrameBuilder

    tokens = np.asarray(encoded)
    if tokens.shape != (episode.frame_count, 64) or tokens.dtype != np.dtype(np.float32):
        raise ValueError("encoded episode must have exact float32 shape [T50,64]")
    _verify_preflight_source_hashes(
        preflight,
        episode.source_episode_id,
        operation="before target/video construction",
    )
    builder = TargetFrameBuilder()
    rows = tuple(builder.build(episode, index, tokens[index]) for index in range(episode.frame_count))
    timelines = preflight.timelines[episode.source_episode_id]
    videos: dict[str, Path] = {}
    temporary: list[Path] = []
    try:
        for source_key, target_key in preflight.camera_schema.source_to_target.items():
            path = _encode_target_video(timelines[source_key])
            temporary.append(path)
            videos[target_key] = path
        _verify_preflight_source_hashes(
            preflight,
            episode.source_episode_id,
            operation="during target/video construction",
        )
        payload = StagePayload(rows=rows, videos=videos)
        return _ProductionPayload(payload=payload, temporary_videos=tuple(temporary))
    except BaseException as error:
        for path in temporary:
            path.unlink(missing_ok=True)
        try:
            _verify_preflight_source_hashes(
                preflight,
                episode.source_episode_id,
                operation="during failed target/video construction",
            )
        except _EpisodePreflightError as integrity_error:
            raise integrity_error from error
        raise


def _validate_payload(payload: _ProductionPayload) -> object:
    from gear_sonic.data.unitree_conversion.validation import (
        probe_video_50fps_stream,
        validation_report_from_inspections,
    )

    try:
        inspections: dict[str, tuple[int, tuple[int, int]]] = {}
        for key, path in payload.payload.videos.items():
            with Path(path).open("rb") as stream:
                inspections[key] = probe_video_50fps_stream(stream, field_name=f"video {key}")
        return validation_report_from_inspections(payload.payload.rows, inspections)
    except BaseException:
        for path in payload.temporary_videos:
            path.unlink(missing_ok=True)
        raise


def _write_stage(output_root: Path, identity: object, payload: _ProductionPayload) -> object:
    from gear_sonic.data.unitree_conversion.staging import write_stage

    try:
        return write_stage(output_root, identity, payload.payload)
    finally:
        for path in payload.temporary_videos:
            path.unlink(missing_ok=True)


def _merge_stages(stages: Sequence[object], target_path: Path, resume: bool) -> Path:
    from gear_sonic.data.unitree_conversion.staging import merge_stages

    return merge_stages(stages, target_path, resume_existing=resume)


def _validate_inspire_root_finiteness(current: object, desired: object) -> None:
    import numpy as np

    for field_name, values in (("robot_q_current", current), ("robot_q_desired", desired)):
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if array.ndim == 2 and array.shape[1] >= 7 and not np.isfinite(array[:, 3:7]).all():
            raise _EpisodePreflightError(
                "quaternion_error",
                f"{field_name} root quaternion contains NaN or Inf",
            )


def _diagnose_inspire(
    source: SourceSpec,
    episode_id: int,
    preflight: _RepositoryPreflight,
    cache_dir: Path | None,
) -> object:
    del cache_dir
    import numpy as np

    from gear_sonic.data.unitree_conversion.inspire_diagnostics import (
        _CURRENT_KEY,
        _DESIRED_KEY,
        _HAND_CMD_KEY,
        _HAND_STATE_KEY,
        _diagnose_arrays,
        _integer,
        _timestamp,
        _validate_metadata,
        _validate_source_spec,
    )

    source = _validate_source_spec(source, episode_id)
    dataset = preflight.datasets[episode_id]
    _validate_metadata(source, dataset.meta)
    expected_hashes = dict(preflight.source_hashes[episode_id])
    if dict(_source_hashes(dataset, episode_id, include_videos=False)) != expected_hashes:
        raise _EpisodePreflightError(
            "provenance_error",
            "Inspire source files changed after preflight",
        )

    current: list[object] = []
    desired: list[object] = []
    hand_state: list[object] = []
    hand_cmd: list[object] = []
    timestamps: list[float] = []
    task_indices: list[int] = []
    for row_number, row in enumerate(dataset.hf_dataset):
        source_episode_id = _integer(row["episode_index"], field_name=f"row {row_number} episode_index")
        if source_episode_id != episode_id:
            raise ValueError(f"row {row_number} episode_index does not match requested episode")
        frame_index = _integer(row["frame_index"], field_name=f"row {row_number} frame_index")
        if frame_index != row_number:
            raise ValueError(f"expected frame_index {row_number}, got {frame_index}")
        task_index = _integer(
            row["task_index"],
            field_name=f"row {row_number} task_index",
            nonnegative=True,
        )
        _resolve_source_task(preflight.tasks, task_index)
        current.append(row[_CURRENT_KEY])
        desired.append(row[_DESIRED_KEY])
        hand_state.append(row[_HAND_STATE_KEY])
        hand_cmd.append(row[_HAND_CMD_KEY])
        try:
            timestamp = _timestamp(row["timestamp"], field_name=f"row {row_number} timestamp")
        except (KeyError, ValueError) as error:
            raise _EpisodePreflightError("timeline_error", str(error)) from error
        timestamps.append(timestamp)
        task_indices.append(task_index)
    if len(timestamps) < 2:
        raise _EpisodePreflightError(
            "timeline_error",
            "Inspire episode contains fewer than two source frames",
        )
    if not np.all(np.diff(np.asarray(timestamps, dtype=np.float64)) > 0.0):
        raise _EpisodePreflightError(
            "timeline_error",
            "Inspire timestamps must be strictly increasing",
        )
    _validate_inspire_root_finiteness(current, desired)
    report = _diagnose_arrays(
        current=current,
        desired=desired,
        hand_state=hand_state,
        hand_cmd=hand_cmd,
        timestamps=timestamps,
        source_repo_id=source.repo_id,
        source_revision=source.revision,
        source_episode_id=episode_id,
        source_task_indices=tuple(task_indices),
        primary_camera=source.primary_camera,
        camera_map=source.camera_map,
    )
    if dict(_source_hashes(dataset, episode_id, include_videos=False)) != expected_hashes:
        raise _EpisodePreflightError(
            "provenance_error",
            "Inspire source files changed during diagnostics",
        )
    return report


def _make_diagnostic_identity(
    lock_sha256: str,
    source: SourceSpec,
    episode_id: int,
    preflight: _RepositoryPreflight,
) -> DiagnosticIdentity:
    contract = json.dumps(
        {
            "gate_reasons": list(INSPIRE_GATE_REASONS),
            "output_kind": "diagnostic-only",
            "status": "blocked_unverified",
            "version": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return DiagnosticIdentity(
        source_repo_id=source.repo_id,
        source_revision=source.revision,
        source_episode_id=episode_id,
        source_file_sha256=preflight.source_hashes[episode_id],
        source_lock_sha256=lock_sha256,
        diagnostic_contract_sha256=hashlib.sha256(contract).hexdigest(),
        converter_version="unitree-sonic-conversion-v1",
    )


def _write_diagnostic(output_root: Path, identity: object, report: object) -> Path:
    from gear_sonic.data.unitree_conversion.staging import (
        _artifact_paths,
        _atomic_publish,
        _checksums_bytes,
        _ensure_safe_directory,
        _fsync_directory,
        _fsync_tree,
        _secure_read,
        _verify_checksum_tree,
        _write_file,
    )

    if not isinstance(identity, DiagnosticIdentity):
        raise TypeError("diagnostic identity must be DiagnosticIdentity")
    if (
        getattr(report, "source_repo_id", None) != identity.source_repo_id
        or getattr(report, "source_revision", None) != identity.source_revision
        or getattr(report, "source_episode_id", None) != identity.source_episode_id
    ):
        raise ValueError("diagnostic report identity differs from immutable provenance")
    safe_repo = identity.source_repo_id.replace("/", "--")
    final = (
        output_root
        / safe_repo
        / identity.source_revision
        / f"episode-{identity.source_episode_id:06d}"
        / identity.digest
    ).absolute()
    _ensure_safe_directory(final.parent)
    identity_data = (json.dumps(identity.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    diagnostic_data = (json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    if final.exists():
        _verify_checksum_tree(final, checksum_name="checksums.sha256")
        if (
            _secure_read(final / "identity.json") != identity_data
            or _secure_read(final / "diagnostic.json") != diagnostic_data
        ):
            raise FileExistsError("immutable diagnostic stage exists with different content")
        return final / "diagnostic.json"

    temporary = Path(tempfile.mkdtemp(prefix=f".{identity.digest}.", dir=final.parent))
    try:
        _write_file(temporary / "identity.json", identity_data)
        _write_file(temporary / "diagnostic.json", diagnostic_data)
        artifacts = _artifact_paths(temporary)
        _write_file(temporary / "checksums.sha256", _checksums_bytes(temporary, artifacts))
        _fsync_tree(temporary)
        _verify_checksum_tree(temporary, checksum_name="checksums.sha256")
        _atomic_publish(temporary, final)
        _fsync_directory(final.parent)
        return final / "diagnostic.json"
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _default_dex3_components(lock: SourceLock, cache_dir: Path | None) -> PipelineComponents:
    del lock, cache_dir
    # Heavy encoder/runtime imports remain inside ``_encoder_factory``.
    return PipelineComponents(
        validate_artifacts=_validate_default_artifacts,
        validate_source_metadata=_validate_source_metadata,
        preflight_repository=_preflight_repository,
        make_stage_identity=_make_stage_identity,
        resume_stage=_resume_stage,
        adapt_dex3_episode=_adapt_dex3,
        resample_episode=_resample_episode,
        encoder_factory=_encoder_factory,
        encode_episode=_encode_episode,
        construct_payload=_construct_payload,
        validate_payload=_validate_payload,
        write_stage=_write_stage,
        merge_stages=_merge_stages,
        diagnose_inspire_episode=_unavailable,
        make_diagnostic_identity=_unavailable,
        write_diagnostic=_unavailable,
    )


def _default_inspire_components() -> PipelineComponents:
    # No ONNX or SonicEncoder import is reachable from this constructor or path.
    return PipelineComponents(
        validate_artifacts=_unavailable,
        validate_source_metadata=_validate_source_metadata,
        preflight_repository=_preflight_repository,
        make_stage_identity=_unavailable,
        resume_stage=_unavailable,
        adapt_dex3_episode=_unavailable,
        resample_episode=_unavailable,
        encoder_factory=_unavailable,
        encode_episode=_unavailable,
        construct_payload=_unavailable,
        validate_payload=_unavailable,
        write_stage=_unavailable,
        merge_stages=_unavailable,
        diagnose_inspire_episode=_diagnose_inspire,
        make_diagnostic_identity=_make_diagnostic_identity,
        write_diagnostic=_write_diagnostic,
    )
