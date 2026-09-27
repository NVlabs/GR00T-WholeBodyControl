from dataclasses import FrozenInstanceError, replace
import hashlib
from pathlib import Path

import pytest

from gear_sonic.data.unitree_conversion import provenance as provenance_module
from gear_sonic.data.unitree_conversion.contracts import (
    ArtifactSpec,
    CollectionMembership,
    SourceSpec,
)
from gear_sonic.data.unitree_conversion.provenance import (
    discover_collection_lock,
    dump_source_lock,
    load_source_lock,
    load_source_lock_snapshot,
    materialize_artifact,
    source_lock_sha256,
    stratified_episode_ids,
    verify_file,
)
from gear_sonic.scripts.lock_unitree_unifolm_sources import write_discovered_lock

LOCK = Path("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")
DEX3_COLLECTION = "unitreerobotics/unifolm-g1-dex3-dataset"
WBT_COLLECTION = "unitreerobotics/unifolm-wbt-dataset"


class FakeHfApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.collections = {
            DEX3_COLLECTION: self._collection(
                self._item("dataset", "unitreerobotics/dex3-first"),
                self._item("model", "unitreerobotics/not-a-dataset"),
                self._item("dataset", "unitreerobotics/dex3-second"),
            ),
            WBT_COLLECTION: self._collection(self._item("dataset", "unitreerobotics/wbt-first")),
        }
        self.revisions = {
            "unitreerobotics/dex3-first": "a" * 40,
            "unitreerobotics/dex3-second": "B" * 40,
            "unitreerobotics/wbt-first": "c" * 40,
        }

    @staticmethod
    def _item(item_type: str, item_id: str):
        return type(
            "CollectionItem",
            (),
            {"item_type": item_type, "item_id": item_id},
        )()

    @staticmethod
    def _collection(*items):
        return type("Collection", (), {"items": list(items)})()

    def get_collection(self, collection_slug: str):
        self.calls.append(("get_collection", collection_slug))
        return self.collections[collection_slug]

    def dataset_info(self, repo_id: str):
        self.calls.append(("dataset_info", repo_id))
        return type(
            "DatasetInfo",
            (),
            {"id": repo_id, "sha": self.revisions[repo_id]},
        )()


def test_smoke_lock_is_fully_immutable() -> None:
    lock = load_source_lock(LOCK)

    assert lock.version == 1
    assert lock.scope == "smoke"
    assert lock.semantic_repo_commit == "6220f4e210e14c7f804f727da94c8886850d513b"
    assert lock.encoder.repo_id == "nvidia/GEAR-SONIC"
    assert lock.encoder.revision == "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2"
    assert lock.encoder.filename == "low_latency/model_encoder.onnx"
    assert lock.encoder.size == 45933505
    assert lock.encoder.sha256 == "60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b"

    assert lock.observation_config.repo_id == "nvidia/GEAR-SONIC"
    assert lock.observation_config.revision == "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2"
    assert lock.observation_config.filename == "low_latency/observation_config.yaml"
    assert lock.observation_config.size == 3258
    assert lock.observation_config.sha256 == "582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab"

    assert lock.dex3.approved is True
    assert lock.dex3.label == "dex3"
    assert lock.dex3.repo_id == "unitreerobotics/G1_Dex3_Pouring_Dataset"
    assert lock.dex3.revision == "c9552eb3b1cb610cd6227555e9e98b1bde826a77"
    assert lock.dex3.dataset_path == "."
    assert lock.dex3.episode_count == 311
    assert lock.dex3.episodes == (0, 78, 155, 233, 310)
    assert lock.dex3.primary_camera == "observation.images.cam_left_high"
    assert dict(lock.dex3.camera_map) == {
        "observation.images.cam_left_high": "observation.images.ego_view",
        "observation.images.cam_left_wrist": "observation.images.left_wrist",
        "observation.images.cam_right_wrist": "observation.images.right_wrist",
    }

    assert lock.inspire.approved is True
    assert lock.inspire.label == "inspire"
    assert lock.inspire.repo_id == "unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly"
    assert lock.inspire.revision == "24e3e4d88a5020bdb4b3046ec09b09dc56f8d1f1"
    assert lock.inspire.dataset_path == "G1_WB_Dex5_Pickup_Pillow"
    assert lock.inspire.episode_count == 609
    assert lock.inspire.episodes == (0, 152, 304, 456, 608)
    assert lock.inspire.primary_camera == "observation.images.cam_0"
    assert dict(lock.inspire.camera_map) == {"observation.images.cam_0": "diagnostic.primary_camera"}

    assert lock.sources == (lock.dex3, lock.inspire)
    assert lock.collections == ()
    for source in lock.sources:
        assert len(source.revision) == 40
        assert int(source.revision, 16) >= 0
    with pytest.raises(TypeError):
        lock.dex3.camera_map["new.camera"] = "new.target"


def test_dataset_path_round_trips_through_source_lock(tmp_path: Path) -> None:
    lock = load_source_lock(LOCK)
    candidate = tmp_path / "round-trip.yaml"
    candidate.write_text(dump_source_lock(lock))

    round_tripped = load_source_lock(candidate)

    assert round_tripped.dex3.dataset_path == "."
    assert round_tripped.inspire.dataset_path == "G1_WB_Dex5_Pickup_Pillow"
    assert round_tripped == lock


@pytest.mark.parametrize(
    "dataset_path",
    [
        "",
        " ",
        "/absolute",
        "../escape",
        "nested/../escape",
        "./nested",
        "nested/.",
        "nested//child",
        "nested/",
        r"nested\child",
        *(f"nested{chr(code)}child" for code in (*range(0x20), 0x7F)),
        "nested*glob",
        "nested?glob",
        "nested[glob]",
    ],
)
def test_source_spec_rejects_unsafe_or_noncanonical_dataset_path(dataset_path: str) -> None:
    with pytest.raises(ValueError, match="dataset_path.*safe canonical POSIX-relative"):
        SourceSpec(
            approved=False,
            repo_id="unitreerobotics/example",
            revision="a" * 40,
            dataset_path=dataset_path,
        )


def test_load_source_lock_rejects_unknown_or_unsafe_dataset_path(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(LOCK.read_text().replace("    dataset_path: .\n", "    dataset_path: ../escape\n", 1))
    with pytest.raises(ValueError, match="dataset_path.*safe canonical POSIX-relative"):
        load_source_lock(unsafe)

    unknown = tmp_path / "unknown-source-field.yaml"
    unknown.write_text(LOCK.read_text().replace("    dataset_path: .\n", "    source_root: nested\n", 1))
    with pytest.raises(ValueError, match="unknown fields.*source_root"):
        load_source_lock(unknown)


@pytest.mark.parametrize(
    "yaml_path",
    [
        '"nested*glob"',
        '"nested?glob"',
        '"nested[glob]"',
        '"nested\\0child"',
        '"nested\\x7fchild"',
    ],
)
def test_load_source_lock_rejects_control_or_glob_dataset_path(
    tmp_path: Path,
    yaml_path: str,
) -> None:
    candidate = tmp_path / "unsafe-prefix.yaml"
    candidate.write_text(LOCK.read_text().replace("    dataset_path: .\n", f"    dataset_path: {yaml_path}\n", 1))

    with pytest.raises(ValueError, match="dataset_path"):
        load_source_lock(candidate)


def test_stratified_selection_matches_pinned_counts() -> None:
    assert stratified_episode_ids(311) == (0, 78, 155, 233, 310)
    assert stratified_episode_ids(609) == (0, 152, 304, 456, 608)


def test_stratified_selection_uses_half_up_rounding() -> None:
    assert stratified_episode_ids(7) == (0, 2, 3, 5, 6)


@pytest.mark.parametrize("episode_count", [0, -1, True, 1.5])
def test_stratified_selection_rejects_invalid_counts(episode_count) -> None:
    with pytest.raises(ValueError, match="integer of at least 5"):
        stratified_episode_ids(episode_count)


@pytest.mark.parametrize("episode_count", range(1, 5))
def test_stratified_selection_rejects_counts_too_small(episode_count: int) -> None:
    with pytest.raises(ValueError, match="integer of at least 5"):
        stratified_episode_ids(episode_count)


def test_load_source_lock_rejects_floating_revision(tmp_path: Path) -> None:
    lock_text = LOCK.read_text().replace("c9552eb3b1cb610cd6227555e9e98b1bde826a77", "main")
    candidate = tmp_path / "floating.yaml"
    candidate.write_text(lock_text)

    with pytest.raises(ValueError, match="40-character immutable revision"):
        load_source_lock(candidate)


def test_load_source_lock_rejects_unknown_fields(tmp_path: Path) -> None:
    candidate = tmp_path / "unknown.yaml"
    candidate.write_text(f"{LOCK.read_text()}unexpected: true\n")

    with pytest.raises(ValueError, match="unknown fields.*unexpected"):
        load_source_lock(candidate)


@pytest.mark.parametrize(
    ("yaml_version", "message"),
    [
        ("1.0", "source lock version must be an integer"),
        ('"1"', "source lock version must be an integer"),
        ("true", "source lock version must be an integer"),
        ("2", "unsupported source lock version: 2"),
    ],
)
def test_load_source_lock_rejects_invalid_version(
    tmp_path: Path,
    yaml_version: str,
    message: str,
) -> None:
    candidate = tmp_path / "invalid-version.yaml"
    candidate.write_text(LOCK.read_text().replace("version: 1\n", f"version: {yaml_version}\n"))

    with pytest.raises(ValueError, match=message):
        load_source_lock(candidate)


@pytest.mark.parametrize(
    ("duplicate_key", "lock_text"),
    [
        ("scope", f"{LOCK.read_text()}scope: smoke\n"),
        (
            "revision",
            LOCK.read_text().replace(
                "  revision: 9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2\n  filename: low_latency/model_encoder.onnx",
                "  revision: 9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2\n"
                "  revision: 9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2\n"
                "  filename: low_latency/model_encoder.onnx",
            ),
        ),
        (
            "approved",
            LOCK.read_text().replace(
                "  dex3:\n    approved: true",
                "  dex3:\n    approved: true\n    approved: true",
            ),
        ),
    ],
)
def test_load_source_lock_rejects_duplicate_yaml_keys_recursively(
    tmp_path: Path,
    duplicate_key: str,
    lock_text: str,
) -> None:
    candidate = tmp_path / "duplicate.yaml"
    candidate.write_text(lock_text)

    with pytest.raises(
        ValueError,
        match=f"duplicate YAML mapping key.*{duplicate_key}",
    ):
        load_source_lock(candidate)


def test_verify_file_checks_size_before_hashing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"wrong")

    def fail_if_hashed():
        raise AssertionError("hashing must not start after a size mismatch")

    monkeypatch.setattr(provenance_module.hashlib, "sha256", fail_if_hashed)
    with pytest.raises(ValueError, match="size mismatch"):
        verify_file(artifact, expected_size=6, expected_sha256="0" * 64)


def test_verify_file_hashes_in_one_mib_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"x" * (1024 * 1024 + 7))
    chunks: list[int] = []

    class RecordingHash:
        def update(self, data: bytes) -> None:
            chunks.append(len(data))

        @staticmethod
        def hexdigest() -> str:
            return "a" * 64

    monkeypatch.setattr(provenance_module.hashlib, "sha256", RecordingHash)

    resolved = verify_file(
        artifact,
        expected_size=1024 * 1024 + 7,
        expected_sha256="a" * 64,
    )

    assert chunks == [1024 * 1024, 7]
    assert resolved == artifact.resolve()


def test_verify_file_rejects_sha256_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"wrong")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(artifact, expected_size=5, expected_sha256="0" * 64)


def test_materialize_artifact_pins_revision_and_verifies_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"model")
    calls: list[dict[str, str]] = []
    spec = ArtifactSpec(
        repo_id="nvidia/GEAR-SONIC",
        revision="1" * 40,
        filename="low_latency/model_encoder.onnx",
        size=5,
        sha256=hashlib.sha256(b"model").hexdigest(),
    )

    def downloader(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(artifact)

    resolved = materialize_artifact(spec, downloader=downloader)

    assert calls == [
        {
            "repo_id": "nvidia/GEAR-SONIC",
            "filename": "low_latency/model_encoder.onnx",
            "revision": "1" * 40,
        }
    ]
    assert resolved == artifact.resolve()


def test_collection_discovery_resolves_every_dataset_to_sha_in_order() -> None:
    api = FakeHfApi()

    proposed = discover_collection_lock(
        api=api,
        collection_slugs=(DEX3_COLLECTION, WBT_COLLECTION),
    )
    smoke_lock = load_source_lock(LOCK)

    assert proposed.scope == "full"
    assert proposed.encoder == smoke_lock.encoder
    assert proposed.observation_config == smoke_lock.observation_config
    assert [source.repo_id for source in proposed.sources] == [
        "unitreerobotics/dex3-first",
        "unitreerobotics/dex3-second",
        "unitreerobotics/wbt-first",
    ]
    assert [source.revision for source in proposed.sources] == [
        "a" * 40,
        "B" * 40,
        "c" * 40,
    ]
    assert all(source.approved is False for source in proposed.sources)
    assert [membership.slug for membership in proposed.collections] == [
        DEX3_COLLECTION,
        WBT_COLLECTION,
    ]
    assert proposed.collections[0].repo_ids == (
        "unitreerobotics/dex3-first",
        "unitreerobotics/dex3-second",
    )
    assert api.calls == [
        ("get_collection", DEX3_COLLECTION),
        ("dataset_info", "unitreerobotics/dex3-first"),
        ("dataset_info", "unitreerobotics/dex3-second"),
        ("get_collection", WBT_COLLECTION),
        ("dataset_info", "unitreerobotics/wbt-first"),
    ]


def test_collection_discovery_rejects_duplicate_repositories() -> None:
    api = FakeHfApi()
    api.collections[WBT_COLLECTION] = api._collection(api._item("dataset", "unitreerobotics/dex3-first"))

    with pytest.raises(ValueError, match="duplicate dataset repository"):
        discover_collection_lock(
            api=api,
            collection_slugs=(DEX3_COLLECTION, WBT_COLLECTION),
        )


@pytest.mark.parametrize("malformed_items", [None, 7, "not-an-item-sequence"])
def test_collection_discovery_rejects_malformed_items(
    malformed_items,
) -> None:
    api = FakeHfApi()
    api.collections[DEX3_COLLECTION] = type(
        "MalformedCollection",
        (),
        {"items": malformed_items},
    )()

    with pytest.raises(
        ValueError,
        match="unifolm-g1-dex3-dataset.*items.*non-string iterable",
    ):
        discover_collection_lock(
            api=api,
            collection_slugs=(DEX3_COLLECTION, WBT_COLLECTION),
        )


def test_collection_discovery_rejects_missing_items() -> None:
    api = FakeHfApi()
    api.collections[DEX3_COLLECTION] = type("CollectionWithoutItems", (), {})()

    with pytest.raises(
        ValueError,
        match="unifolm-g1-dex3-dataset.*items.*non-string iterable",
    ):
        discover_collection_lock(
            api=api,
            collection_slugs=(DEX3_COLLECTION, WBT_COLLECTION),
        )


def test_collection_discovery_rejects_collection_without_datasets() -> None:
    api = FakeHfApi()
    api.collections[DEX3_COLLECTION] = api._collection(api._item("model", "unitreerobotics/not-a-dataset"))

    with pytest.raises(
        ValueError,
        match="unifolm-g1-dex3-dataset.*at least one dataset item",
    ):
        discover_collection_lock(
            api=api,
            collection_slugs=(DEX3_COLLECTION, WBT_COLLECTION),
        )


def test_discovery_draft_round_trips_and_overwrite_is_fail_closed(
    tmp_path: Path,
) -> None:
    lock = discover_collection_lock(
        api=FakeHfApi(),
        collection_slugs=(DEX3_COLLECTION, WBT_COLLECTION),
    )
    output_lock = tmp_path / "review.yaml"

    assert write_discovered_lock(lock, output_lock) == output_lock.resolve()
    assert load_source_lock(output_lock) == lock

    output_lock.write_text("reviewed lock\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_discovered_lock(lock, output_lock)
    assert output_lock.read_text() == "reviewed lock\n"

    assert write_discovered_lock(lock, output_lock, force=True) == output_lock.resolve()
    assert load_source_lock(output_lock) == lock
    assert list(tmp_path.glob(f".{output_lock.name}.*.tmp")) == []


@pytest.mark.parametrize("collections_yaml", ["", "collections: []\n"])
def test_full_source_lock_requires_collection_membership(
    tmp_path: Path,
    collections_yaml: str,
) -> None:
    candidate = tmp_path / "full.yaml"
    candidate.write_text(LOCK.read_text().replace("scope: smoke", "scope: full") + collections_yaml)

    with pytest.raises(ValueError, match="full source locks require at least one collection"):
        load_source_lock(candidate)


@pytest.mark.parametrize(
    "collections",
    [
        (),
        (
            CollectionMembership(
                slug=DEX3_COLLECTION,
                repo_ids=(),
            ),
        ),
    ],
)
def test_full_source_lock_requires_at_least_one_source(collections) -> None:
    smoke_lock = load_source_lock(LOCK)

    with pytest.raises(ValueError, match="full source locks require at least one source"):
        replace(
            smoke_lock,
            scope="full",
            sources=(),
            collections=collections,
        )


def test_full_source_lock_requires_exact_ordered_collection_membership() -> None:
    smoke_lock = load_source_lock(LOCK)
    reversed_repo_ids = tuple(source.repo_id for source in reversed(smoke_lock.sources))

    with pytest.raises(
        ValueError,
        match="ordered collection membership must exactly match ordered sources",
    ):
        replace(
            smoke_lock,
            scope="full",
            collections=(
                CollectionMembership(
                    slug=DEX3_COLLECTION,
                    repo_ids=reversed_repo_ids,
                ),
            ),
        )


@pytest.mark.parametrize("repo_ids", ["owner/repository", b"owner/repository"])
def test_collection_membership_rejects_scalar_repo_ids(repo_ids) -> None:
    with pytest.raises(
        ValueError,
        match="collection repo_ids must be a non-string iterable",
    ):
        CollectionMembership(
            slug=DEX3_COLLECTION,
            repo_ids=repo_ids,
        )


@pytest.mark.parametrize(
    ("field_name", "different_value"),
    [
        ("repo_id", "nvidia/a-different-model"),
        ("revision", "f" * 40),
    ],
)
def test_source_lock_requires_one_model_source(
    field_name: str,
    different_value: str,
) -> None:
    smoke_lock = load_source_lock(LOCK)
    mismatched_config = replace(
        smoke_lock.observation_config,
        **{field_name: different_value},
    )

    with pytest.raises(
        ValueError,
        match=f"encoder and observation_config must share the same {field_name}",
    ):
        replace(smoke_lock, observation_config=mismatched_config)


@pytest.mark.parametrize(
    ("revision", "sha256", "message"),
    [
        ("g" * 40, "0" * 64, "40-character immutable revision"),
        ("0" * 39, "0" * 64, "40-character immutable revision"),
        ("0" * 40, "z" * 64, "64-character SHA-256"),
    ],
)
def test_artifact_contract_rejects_malformed_hashes(revision: str, sha256: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ArtifactSpec(
            repo_id="owner/repository",
            revision=revision,
            filename="artifact.bin",
            size=1,
            sha256=sha256,
        )


def test_source_contract_allows_unapproved_discovery_draft() -> None:
    source = SourceSpec(
        approved=False,
        repo_id="owner/repository",
        revision="a" * 40,
    )

    assert source.episode_count is None
    assert source.episodes == ()
    assert source.primary_camera is None
    assert dict(source.camera_map) == {}


@pytest.mark.parametrize(
    ("episode_count", "episodes", "primary_camera", "camera_map", "message"),
    [
        (0, (), "observation.images.cam", {"observation.images.cam": "ego"}, "episode_count"),
        (3, (0, 3), "observation.images.cam", {"observation.images.cam": "ego"}, "episode ID"),
        (3, (0, 2), "", {"observation.images.cam": "ego"}, "primary_camera"),
        (3, (0, 2), "missing", {"observation.images.cam": "ego"}, "camera_map"),
    ],
)
def test_approved_source_contract_rejects_invalid_episode_or_camera_data(
    episode_count,
    episodes,
    primary_camera,
    camera_map,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SourceSpec(
            approved=True,
            repo_id="owner/repository",
            revision="a" * 40,
            episode_count=episode_count,
            episodes=episodes,
            primary_camera=primary_camera,
            camera_map=camera_map,
        )


def test_source_lock_sha256_hashes_exact_manifest_bytes() -> None:
    assert source_lock_sha256(LOCK) == hashlib.sha256(LOCK.read_bytes()).hexdigest()


def test_source_lock_snapshot_uses_one_exact_byte_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exact_bytes = LOCK.read_bytes()
    expected_lock = load_source_lock(LOCK)
    virtual_path = Path("virtual-source-lock.yaml")
    read_paths: list[Path] = []

    def read_bytes_once(path: Path) -> bytes:
        read_paths.append(path)
        return exact_bytes

    monkeypatch.setattr(Path, "read_bytes", read_bytes_once)

    snapshot = load_source_lock_snapshot(virtual_path)

    assert read_paths == [virtual_path]
    assert snapshot.lock == expected_lock
    assert snapshot.sha256 == hashlib.sha256(exact_bytes).hexdigest()
    assert not hasattr(snapshot, "raw_bytes")
    with pytest.raises(FrozenInstanceError):
        snapshot.sha256 = "0" * 64

    read_paths.clear()
    assert load_source_lock(virtual_path) == expected_lock
    assert read_paths == [virtual_path]
