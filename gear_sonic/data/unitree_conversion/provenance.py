"""Revision-lock loading, discovery, and byte verification."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from gear_sonic.data.unitree_conversion.contracts import (
    ArtifactSpec,
    CollectionMembership,
    SourceLock,
    SourceSpec,
    validate_sha256,
)

HASH_CHUNK_SIZE = 1024 * 1024
SMOKE_SOURCE_LOCK = Path(__file__).parent / "manifests" / "smoke_sources.yaml"

Downloader = Callable[..., str | Path]


@dataclass(frozen=True)
class SourceLockSnapshot:
    """Parsed source lock and identity of the exact bytes parsed."""

    lock: SourceLock
    sha256: str


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """SafeLoader variant that rejects ambiguous mappings at every depth."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable YAML mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate YAML mapping key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def stratified_episode_ids(episode_count: int) -> tuple[int, ...]:
    """Select five deterministic IDs using exact half-up integer rounding."""
    if isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count < 5:
        raise ValueError("episode_count must be an integer of at least 5")
    last_episode = episode_count - 1
    denominator = 4
    return tuple((2 * k * last_episode + denominator) // (2 * denominator) for k in range(5))


def verify_file(
    path: str | Path,
    expected_size: int,
    expected_sha256: str,
) -> Path:
    """Verify exact size then SHA-256 and return the resolved path."""
    candidate = Path(path)
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size <= 0:
        raise ValueError("expected_size must be a positive integer")

    actual_size = candidate.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"size mismatch for {candidate}: expected {expected_size} bytes, got {actual_size}")

    validate_sha256(expected_sha256, field_name="expected_sha256")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch for {candidate}: expected {expected_sha256}, got {actual_sha256}")
    return candidate.resolve()


def materialize_artifact(
    spec: ArtifactSpec,
    *,
    downloader: Downloader | None = None,
) -> Path:
    """Download an artifact at its pinned revision and verify its exact bytes."""
    if not isinstance(spec, ArtifactSpec):
        raise ValueError("spec must be an ArtifactSpec")
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download
    downloaded = downloader(
        repo_id=spec.repo_id,
        filename=spec.filename,
        revision=spec.revision,
    )
    return verify_file(downloaded, spec.size, spec.sha256)


def source_lock_sha256(path: str | Path) -> str:
    """Hash the exact tracked source-lock bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: object, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    context: str,
) -> None:
    unknown = sorted(repr(key) for key in value if not isinstance(key, str) or key not in allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def _require_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    context: str,
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def _parse_artifact(value: object, *, context: str) -> ArtifactSpec:
    data = _as_mapping(value, context=context)
    fields = {"repo_id", "revision", "filename", "size", "sha256"}
    _reject_unknown_fields(data, allowed=fields, context=context)
    _require_fields(data, required=fields, context=context)
    return ArtifactSpec(
        repo_id=data["repo_id"],
        revision=data["revision"],
        filename=data["filename"],
        size=data["size"],
        sha256=data["sha256"],
    )


def _parse_source(
    value: object,
    *,
    context: str,
    label: str | None = None,
) -> SourceSpec:
    data = _as_mapping(value, context=context)
    fields = {
        "approved",
        "repo_id",
        "revision",
        "dataset_path",
        "episode_count",
        "episodes",
        "primary_camera",
        "camera_map",
        "name",
    }
    _reject_unknown_fields(data, allowed=fields, context=context)
    _require_fields(data, required={"approved", "repo_id", "revision"}, context=context)
    serialized_label = data.get("name")
    if label is not None and serialized_label is not None:
        raise ValueError(f"{context} cannot specify a name inside a named source mapping")
    return SourceSpec(
        approved=data["approved"],
        repo_id=data["repo_id"],
        revision=data["revision"],
        dataset_path=data.get("dataset_path", "."),
        episode_count=data.get("episode_count"),
        episodes=data.get("episodes", ()),
        primary_camera=data.get("primary_camera"),
        camera_map=data.get("camera_map", {}),
        label=label if label is not None else serialized_label,
    )


def _parse_sources(value: object) -> tuple[SourceSpec, ...]:
    if isinstance(value, Mapping):
        return tuple(
            _parse_source(
                source_data,
                context=f"sources.{label}",
                label=label,
            )
            for label, source_data in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(
            _parse_source(source_data, context=f"sources[{index}]") for index, source_data in enumerate(value)
        )
    raise ValueError("sources must be a named mapping or an ordered sequence")


def _parse_collections(value: object) -> tuple[CollectionMembership, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("collections must be an ordered sequence")
    memberships: list[CollectionMembership] = []
    for index, membership_value in enumerate(value):
        context = f"collections[{index}]"
        data = _as_mapping(membership_value, context=context)
        fields = {"slug", "repositories"}
        _reject_unknown_fields(data, allowed=fields, context=context)
        _require_fields(data, required=fields, context=context)
        repositories = data["repositories"]
        if not isinstance(repositories, Sequence) or isinstance(repositories, (str, bytes)):
            raise ValueError(f"{context}.repositories must be an ordered sequence")
        memberships.append(
            CollectionMembership(
                slug=data["slug"],
                repo_ids=tuple(repositories),
            )
        )
    return tuple(memberships)


def _parse_source_lock_bytes(lock_bytes: bytes) -> SourceLock:
    try:
        lock_text = lock_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("source lock must be UTF-8 encoded") from error

    loader = _UniqueKeySafeLoader(lock_text)
    try:
        loaded = loader.get_single_data()
    except yaml.YAMLError as error:
        raise ValueError(f"malformed source lock YAML: {error}") from error
    finally:
        loader.dispose()
    data = _as_mapping(loaded, context="source lock")
    fields = {
        "version",
        "scope",
        "semantic_repo_commit",
        "encoder",
        "observation_config",
        "sources",
        "collections",
    }
    required = fields - {"collections"}
    _reject_unknown_fields(data, allowed=fields, context="source lock")
    _require_fields(data, required=required, context="source lock")
    return SourceLock(
        version=data["version"],
        scope=data["scope"],
        semantic_repo_commit=data["semantic_repo_commit"],
        encoder=_parse_artifact(data["encoder"], context="encoder"),
        observation_config=_parse_artifact(
            data["observation_config"],
            context="observation_config",
        ),
        sources=_parse_sources(data["sources"]),
        collections=_parse_collections(data.get("collections", ())),
    )


def load_source_lock_snapshot(path: str | Path) -> SourceLockSnapshot:
    """Read one byte snapshot, then parse and hash that exact snapshot."""
    lock_bytes = Path(path).read_bytes()
    return SourceLockSnapshot(
        lock=_parse_source_lock_bytes(lock_bytes),
        sha256=hashlib.sha256(lock_bytes).hexdigest(),
    )


def load_source_lock(path: str | Path) -> SourceLock:
    """Safely parse and validate a source lock without resolving network state."""
    return load_source_lock_snapshot(path).lock


def _artifact_to_dict(spec: ArtifactSpec) -> dict[str, object]:
    return {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "filename": spec.filename,
        "size": spec.size,
        "sha256": spec.sha256,
    }


def _source_to_dict(spec: SourceSpec, *, include_name: bool) -> dict[str, object]:
    data: dict[str, object] = {
        "approved": spec.approved,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "dataset_path": spec.dataset_path,
        "episode_count": spec.episode_count,
        "episodes": list(spec.episodes),
        "primary_camera": spec.primary_camera,
        "camera_map": dict(spec.camera_map),
    }
    if include_name and spec.label is not None:
        data = {"name": spec.label, **data}
    return data


def source_lock_to_dict(lock: SourceLock) -> dict[str, object]:
    """Return a YAML-safe, order-preserving representation of a source lock."""
    if not isinstance(lock, SourceLock):
        raise ValueError("lock must be a SourceLock")
    data: dict[str, object] = {
        "version": lock.version,
        "scope": lock.scope,
        "semantic_repo_commit": lock.semantic_repo_commit,
        "encoder": _artifact_to_dict(lock.encoder),
        "observation_config": _artifact_to_dict(lock.observation_config),
    }
    if lock.collections:
        data["collections"] = [
            {
                "slug": membership.slug,
                "repositories": list(membership.repo_ids),
            }
            for membership in lock.collections
        ]
    if lock.scope == "smoke" and all(source.label for source in lock.sources):
        data["sources"] = {source.label: _source_to_dict(source, include_name=False) for source in lock.sources}
    else:
        data["sources"] = [_source_to_dict(source, include_name=True) for source in lock.sources]
    return data


def dump_source_lock(lock: SourceLock) -> str:
    """Serialize a source lock as reviewable safe YAML."""
    return yaml.safe_dump(
        source_lock_to_dict(lock),
        sort_keys=False,
        allow_unicode=False,
    )


def _is_dataset_item(item: object) -> bool:
    item_type = getattr(item, "item_type", None)
    return getattr(item_type, "value", item_type) == "dataset"


def discover_collection_lock(
    api: Any,
    collection_slugs: Sequence[str],
) -> SourceLock:
    """Resolve ordered dataset collection members to immutable review-draft SHAs."""
    if not isinstance(collection_slugs, Sequence) or isinstance(collection_slugs, (str, bytes)):
        raise ValueError("collection_slugs must be an ordered sequence")
    if not collection_slugs:
        raise ValueError("collection_slugs must not be empty")

    sources: list[SourceSpec] = []
    memberships: list[CollectionMembership] = []
    seen_repositories: set[str] = set()
    for collection_slug in collection_slugs:
        if not isinstance(collection_slug, str) or not collection_slug.strip():
            raise ValueError("every collection slug must be a nonempty string")
        collection = api.get_collection(collection_slug)
        items = getattr(collection, "items", None)
        if not isinstance(items, Iterable) or isinstance(items, (str, bytes, Mapping)):
            raise ValueError(f"collection {collection_slug!r} items must be a non-string iterable")
        collection_repositories: list[str] = []
        for item in items:
            if not _is_dataset_item(item):
                continue
            repo_id = getattr(item, "item_id", None)
            if not isinstance(repo_id, str) or not repo_id.strip():
                raise ValueError(f"dataset item in {collection_slug!r} has no repository ID")
            if repo_id in seen_repositories:
                raise ValueError(f"duplicate dataset repository in collections: {repo_id}")
            info = api.dataset_info(repo_id)
            revision = getattr(info, "sha", None)
            sources.append(
                SourceSpec(
                    approved=False,
                    repo_id=repo_id,
                    revision=revision,
                )
            )
            collection_repositories.append(repo_id)
            seen_repositories.add(repo_id)
        if not collection_repositories:
            raise ValueError(f"collection {collection_slug!r} must contain at least one dataset item")
        memberships.append(
            CollectionMembership(
                slug=collection_slug,
                repo_ids=tuple(collection_repositories),
            )
        )

    smoke_lock = load_source_lock(SMOKE_SOURCE_LOCK)
    return SourceLock(
        version=1,
        scope="full",
        semantic_repo_commit=smoke_lock.semantic_repo_commit,
        encoder=smoke_lock.encoder,
        observation_config=smoke_lock.observation_config,
        sources=tuple(sources),
        collections=tuple(memberships),
    )
