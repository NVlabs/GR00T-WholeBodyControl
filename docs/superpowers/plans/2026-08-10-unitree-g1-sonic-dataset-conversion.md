# Unitree G1 to GEAR-SONIC Dataset Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved revision-locked Dex3-to-SONIC dataset converter and the fail-closed five-episode Inspire diagnostic path.

**Architecture:** Resolve every source and model through immutable Hugging Face revisions, adapt documented source fields into a named canonical episode, reproduce deployment's 30-to-50 Hz and 1,247D encoder semantics, then validate and stage each episode before a deterministic LeRobot merge. Keep Inspire in a separate diagnostic path that never imports or invokes the encoder until its three semantic gate artifacts exist.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyArrow, PyAV, Hugging Face Hub, LeRobot v3, ONNX Runtime, Pinocchio RobotModel, MuJoCo, pytest, C++20, TensorRT.

**Design spec:** `docs/superpowers/specs/2026-08-10-unitree-g1-sonic-dataset-conversion-design.md`

**Test runner:** Use the pytest installation from `/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python` with data-collection packages added through `PYTHONPATH=/home/jihun/work/GR00T-WholeBodyControl/.venv_data_collection/lib/python3.10/site-packages`. Set `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, and `-p no:cacheprovider` on every test command.

---

## Current-worktree constraint

The workspace contains unrelated changes in PICO, CMake/ROS, diagnostics, `.codegraph/`, `.superpowers/`, `command.txt`, outputs, and tests. Do not modify or stage them. Every commit below uses exact paths; never use `git add .` or `git add -A`.

The low-latency policy directory is currently untracked because `*.onnx` is ignored. Do not force-add ONNX or TensorRT artifacts. The converter verifies the revision-qualified Hugging Face artifact and materializes it into the existing ignored policy path or Hugging Face cache.

## File and responsibility map

**Create package `gear_sonic/data/unitree_conversion/`:**

- `__init__.py` — public converter types only.
- `contracts.py` — immutable artifact/source specs, canonical episode arrays, diagnostics, and conversion statuses.
- `provenance.py` — source-lock parsing, revision validation, Hugging Face materialization, size/hash verification, and conversion identity.
- `joint_mapping.py` — G1 name orders, nominal posture, side-specific Dex3 DDS tables, and name-based reordering.
- `dex3_adapter.py` — pinned LeRobot Dex3 episode reader and canonical adapter.
- `inspire_diagnostics.py` — shape/statistics scanner and three-reason semantic gate.
- `quaternion.py` — strict WXYZ validation, C++-equivalent SLERP, heading, multiplication, and row-wise 6D packing.
- `resampling.py` — exact 30-to-50 Hz pose/velocity/window and nearest-image formulas.
- `sonic_encoder.py` — immutable artifact loader, config/tensor checks, 1,247D builder, and ONNX Runtime inference.
- `video_timeline.py` — source camera preflight, exact frame counting/decoding, and 50 Hz frame selection.
- `target_frames.py` — complete SONIC VLA output frame, neutral-field, task, and FK construction.
- `staging.py` — immutable episode stages, checksums, resume validation, and deterministic merge.
- `validation.py` — structural and numeric validators plus reports.
- `replay.py` — deterministic MuJoCo replay orchestration and metric aggregation.
- `pipeline.py` — source-specific orchestration without embedding transformation logic.
- `manifests/smoke_sources.yaml` — exact model/source revisions, cameras, counts, and selected episodes.

**Create scripts:**

- `gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py` — conversion/diagnostic CLI.
- `gear_sonic/scripts/lock_unitree_unifolm_sources.py` — collection discovery to reviewable immutable source lock.
- `gear_sonic/scripts/replay_unitree_sonic_smoke.py` — explicit five-episode replay CLI.

**Create tests:**

- `gear_sonic/tests/unitree_conversion/test_provenance.py`
- `gear_sonic/tests/unitree_conversion/test_joint_mapping.py`
- `gear_sonic/tests/unitree_conversion/test_dex3_adapter.py`
- `gear_sonic/tests/unitree_conversion/test_inspire_diagnostics.py`
- `gear_sonic/tests/unitree_conversion/test_quaternion.py`
- `gear_sonic/tests/unitree_conversion/test_resampling.py`
- `gear_sonic/tests/unitree_conversion/test_sonic_encoder.py`
- `gear_sonic/tests/unitree_conversion/test_video_timeline.py`
- `gear_sonic/tests/unitree_conversion/test_target_frames.py`
- `gear_sonic/tests/unitree_conversion/test_staging.py`
- `gear_sonic/tests/unitree_conversion/test_pipeline.py`
- `gear_sonic/tests/unitree_conversion/test_replay.py`
- `gear_sonic/tests/unitree_conversion/test_smoke_network.py`
- `gear_sonic/tests/data/unitree_conversion/golden_encoder_case.npz`
- `gear_sonic/tests/data/unitree_conversion/golden_encoder_token.npy`

**Create deployment parity harness:**

- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/sonic_encoder_parity_test.cpp` — deployment math/layout and TensorRT golden runner.

**Modify:**

- `gear_sonic/pyproject.toml` — direct conversion dependencies.
- `gear_sonic/data/exporter.py` — stable first-occurrence episode task order.
- `docs/source/tutorials/data_conversion.md` — operator commands, gates, and artifacts.
- `docs/source/index.rst` — include the tutorial.

### Task 1: Establish immutable manifests and dependency boundary

**Files:**

- Modify: `gear_sonic/pyproject.toml`
- Create: `gear_sonic/data/unitree_conversion/__init__.py`
- Create: `gear_sonic/data/unitree_conversion/contracts.py`
- Create: `gear_sonic/data/unitree_conversion/provenance.py`
- Create: `gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml`
- Create: `gear_sonic/scripts/lock_unitree_unifolm_sources.py`
- Test: `gear_sonic/tests/unitree_conversion/test_provenance.py`

- [ ] **Step 1: Write the failing provenance and cohort tests**

Create `test_provenance.py` with these contract cases:

```python
from pathlib import Path

import pytest

from gear_sonic.data.unitree_conversion.provenance import (
    discover_collection_lock,
    load_source_lock,
    stratified_episode_ids,
    verify_file,
)


LOCK = Path("gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml")


class FakeHfApi:
    def get_collection(self, collection_slug):
        item = type(
            "CollectionItem",
            (),
            {"item_type": "dataset", "item_id": f"unitreerobotics/{collection_slug.split('/')[-1]}"},
        )()
        return type("Collection", (), {"items": [item]})()

    def dataset_info(self, repo_id):
        return type("DatasetInfo", (), {"id": repo_id, "sha": "a" * 40})()


@pytest.fixture
def fake_hf_api():
    return FakeHfApi()


def test_smoke_lock_is_fully_immutable():
    lock = load_source_lock(LOCK)
    assert lock.encoder.revision == "9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2"
    assert lock.encoder.sha256 == "60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b"
    assert lock.dex3.episodes == (0, 78, 155, 233, 310)
    assert lock.inspire.episodes == (0, 152, 304, 456, 608)
    assert all(len(source.revision) == 40 for source in (lock.dex3, lock.inspire))


def test_stratified_selection_matches_pinned_counts():
    assert stratified_episode_ids(311) == (0, 78, 155, 233, 310)
    assert stratified_episode_ids(609) == (0, 152, 304, 456, 608)


def test_rejects_floating_revision(tmp_path: Path):
    lock_text = LOCK.read_text().replace(
        "c9552eb3b1cb610cd6227555e9e98b1bde826a77", "main"
    )
    candidate = tmp_path / "floating.yaml"
    candidate.write_text(lock_text)
    with pytest.raises(ValueError, match="40-character immutable revision"):
        load_source_lock(candidate)


def test_hash_mismatch_is_fatal(tmp_path: Path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(artifact, expected_size=5, expected_sha256="0" * 64)


def test_collection_discovery_resolves_every_dataset_to_sha(fake_hf_api):
    proposed = discover_collection_lock(
        api=fake_hf_api,
        collection_slugs=(
            "unitreerobotics/unifolm-g1-dex3-dataset",
            "unitreerobotics/unifolm-wbt-dataset",
        ),
    )
    assert proposed.scope == "full"
    assert all(len(source.revision) == 40 for source in proposed.sources)
    assert all(source.approved is False for source in proposed.sources)
```

- [ ] **Step 2: Run the tests and verify the package is missing**

Run:

```bash
PYTHONPATH=/home/jihun/work/GR00T-WholeBodyControl/.venv_data_collection/lib/python3.10/site-packages \
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
gear_sonic/tests/unitree_conversion/test_provenance.py
```

Expected: collection fails with `ModuleNotFoundError: gear_sonic.data.unitree_conversion`.

- [ ] **Step 3: Add the source lock and strict dataclasses**

Write `smoke_sources.yaml` with the exact values:

```yaml
version: 1
scope: smoke
semantic_repo_commit: 6220f4e210e14c7f804f727da94c8886850d513b
encoder:
  repo_id: nvidia/GEAR-SONIC
  revision: 9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2
  filename: low_latency/model_encoder.onnx
  size: 45933505
  sha256: 60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b
observation_config:
  repo_id: nvidia/GEAR-SONIC
  revision: 9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2
  filename: low_latency/observation_config.yaml
  size: 3258
  sha256: 582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab
sources:
  dex3:
    approved: true
    repo_id: unitreerobotics/G1_Dex3_Pouring_Dataset
    revision: c9552eb3b1cb610cd6227555e9e98b1bde826a77
    episode_count: 311
    episodes: [0, 78, 155, 233, 310]
    primary_camera: observation.images.cam_left_high
    camera_map:
      observation.images.cam_left_high: observation.images.ego_view
      observation.images.cam_left_wrist: observation.images.left_wrist
      observation.images.cam_right_wrist: observation.images.right_wrist
  inspire:
    approved: true
    repo_id: unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly
    revision: 24e3e4d88a5020bdb4b3046ec09b09dc56f8d1f1
    episode_count: 609
    episodes: [0, 152, 304, 456, 608]
    primary_camera: observation.images.cam_0
    camera_map:
      observation.images.cam_0: diagnostic.primary_camera
```

In `contracts.py`, define frozen `ArtifactSpec`, `SourceSpec`, and `SourceLock` dataclasses using tuples and `Path`-free scalar values. In `provenance.py`, reject non-hex or non-40-character revisions, calculate hashes in 1 MiB chunks, validate size before hash, and materialize artifacts only with `hf_hub_download(..., revision=spec.revision)`.

`discover_collection_lock()` calls `HfApi.get_collection()` only during discovery, resolves each dataset with `dataset_info(...).sha`, records the ordered collection membership, and emits `scope: full` entries with `approved: false`. The lock command never converts data. An operator must fill explicit camera mappings and set each reviewed entry to `approved: true`; conversion rejects an unapproved entry. This makes collection discovery reproducible without allowing a floating collection to reach conversion.

- [ ] **Step 4: Add direct package dependencies**

Add these entries to `data_collection` in `gear_sonic/pyproject.toml`:

```toml
"huggingface_hub>=0.34,<2",
"onnxruntime>=1.23,<1.24",
"pyarrow>=20,<25",
"pyyaml>=6,<7",
```

Add `lock_unitree_unifolm_sources.py` with fixed default collection slugs from the request, `--output-lock`, and `--force` defaulting false. It uses exclusive creation and refuses to overwrite an existing reviewed lock.

- [ ] **Step 5: Run provenance tests and lint**

Run the Step 2 test command, then:

```bash
ruff check --no-cache \
gear_sonic/data/unitree_conversion/__init__.py \
gear_sonic/data/unitree_conversion/contracts.py \
gear_sonic/data/unitree_conversion/provenance.py \
gear_sonic/scripts/lock_unitree_unifolm_sources.py \
gear_sonic/tests/unitree_conversion/test_provenance.py
```

Expected: provenance tests pass and Ruff exits 0.

- [ ] **Step 6: Commit only Task 1**

```bash
git add gear_sonic/pyproject.toml \
  gear_sonic/data/unitree_conversion/__init__.py \
  gear_sonic/data/unitree_conversion/contracts.py \
  gear_sonic/data/unitree_conversion/provenance.py \
  gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml \
  gear_sonic/scripts/lock_unitree_unifolm_sources.py \
  gear_sonic/tests/unitree_conversion/test_provenance.py
git diff --cached --check
git commit -m "feat: lock Unitree conversion provenance"
```

### Task 2: Implement semantic G1 and side-specific Dex3 mappings

**Files:**

- Create: `gear_sonic/data/unitree_conversion/joint_mapping.py`
- Test: `gear_sonic/tests/unitree_conversion/test_joint_mapping.py`

- [ ] **Step 1: Write mapping and one-hot tests**

```python
import numpy as np

from gear_sonic.data.unitree_conversion.joint_mapping import (
    DEX3_DDS_NAMES,
    G1_ISAACLAB_NAMES,
    G1_MUJOCO_NAMES,
    NOMINAL_G1_MUJOCO,
    dex3_dds_to_robot_model,
    reorder_by_name,
)


def test_nominal_pose_is_absolute_29d():
    assert NOMINAL_G1_MUJOCO.shape == (29,)
    np.testing.assert_allclose(NOMINAL_G1_MUJOCO[:6], [-0.312, 0, 0, 0.669, -0.363, 0])


def test_name_mapping_matches_deployment_arrays():
    source = np.arange(29, dtype=np.float64)
    mapped = reorder_by_name(source, G1_MUJOCO_NAMES, G1_ISAACLAB_NAMES)
    np.testing.assert_array_equal(
        mapped,
        source[[0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]],
    )


def test_all_dex3_one_hot_paths_land_on_semantic_robot_model_joint():
    for side in ("left", "right"):
        for source_index, semantic in enumerate(DEX3_DDS_NAMES[side]):
            raw = np.zeros(7, dtype=np.float64)
            raw[source_index] = 1.0
            named = dex3_dds_to_robot_model(side, raw)
            assert named[f"{side}_hand_{semantic}_joint"] == 1.0
            assert sum(named.values()) == 1.0
```

- [ ] **Step 2: Run and observe the missing module failure**

Run the standard pytest command for `test_joint_mapping.py`.

Expected: import fails for `joint_mapping`.

- [ ] **Step 3: Implement explicit name tables and rejecting reorder helper**

Define the full 29-name MuJoCo and Isaac Lab tuples, exact 29-value nominal array, and:

```python
DEX3_DDS_NAMES = {
    "left": ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"),
    "right": ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"),
}


def reorder_by_name(values, source_names, target_names):
    values = np.asarray(values)
    if values.shape[-1] != len(source_names):
        raise ValueError(f"last dimension {values.shape[-1]} != {len(source_names)}")
    if len(set(source_names)) != len(source_names):
        raise ValueError("source joint names are not unique")
    missing = set(target_names) - set(source_names)
    extra = set(source_names) - set(target_names)
    if missing or extra:
        raise ValueError(f"joint-name mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    indices = [source_names.index(name) for name in target_names]
    return values[..., indices]
```

`dex3_dds_to_robot_model()` must return a complete seven-name dictionary and reject an unknown side or non-`[7]` input.

- [ ] **Step 4: Run tests, Ruff, and commit**

Run the mapping test and Ruff on the two paths. Expected: pass and exit 0.

```bash
git add gear_sonic/data/unitree_conversion/joint_mapping.py \
  gear_sonic/tests/unitree_conversion/test_joint_mapping.py
git diff --cached --check
git commit -m "feat: add semantic G1 Dex3 mappings"
```

### Task 3: Adapt documented Dex3 arrays into canonical episodes

**Files:**

- Modify: `gear_sonic/data/unitree_conversion/contracts.py`
- Create: `gear_sonic/data/unitree_conversion/dex3_adapter.py`
- Test: `gear_sonic/tests/unitree_conversion/test_dex3_adapter.py`

- [ ] **Step 1: Write pure-adapter tests with asymmetric hands**

```python
import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.dex3_adapter import adapt_dex3_arrays
from gear_sonic.data.unitree_conversion.joint_mapping import NOMINAL_G1_MUJOCO


ARM_NAMES = (
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
)
HAND_NAMES = (
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
)


def test_adapts_body_and_preserves_side_specific_dds_order():
    observed = np.tile(np.arange(28, dtype=np.float64), (3, 1))
    desired = observed + 100.0
    episode = adapt_dex3_arrays(
        source_repo_id="synthetic/dex3",
        source_revision="1" * 40,
        source_episode_id=7,
        observed=observed,
        desired=desired,
        feature_names=ARM_NAMES + HAND_NAMES,
        timestamps=np.arange(3, dtype=np.float64) / 30.0,
        task_indices=np.zeros(3, dtype=np.int64),
    )
    np.testing.assert_allclose(episode.observed_body_q[:, :15], NOMINAL_G1_MUJOCO[:15])
    np.testing.assert_allclose(episode.desired_body_q[:, 15:], desired[:, :14])
    np.testing.assert_allclose(episode.desired_left_hand, desired[:, 14:21])
    np.testing.assert_allclose(episode.desired_right_hand, desired[:, 21:28])
    np.testing.assert_allclose(episode.observed_root_wxyz, [[1, 0, 0, 0]] * 3)


def test_rejects_swapped_or_generic_feature_names():
    values = np.zeros((2, 28), dtype=np.float64)
    with pytest.raises(ValueError, match="exact Dex3 feature names"):
        adapt_dex3_arrays(
            source_repo_id="synthetic/dex3",
            source_revision="1" * 40,
            source_episode_id=7,
            observed=values,
            desired=values,
            feature_names=tuple(f"feature_{i}" for i in range(28)),
            timestamps=np.array([0.0, 1 / 30]),
            task_indices=np.zeros(2, dtype=np.int64),
        )
```

- [ ] **Step 2: Run the adapter tests and verify the import/function failure**

Run the standard pytest command on `test_dex3_adapter.py`.

Expected: failure because `dex3_adapter` is absent.

- [ ] **Step 3: Add a validating canonical dataclass**

Add mutable array-owning `CanonicalEpisode` to `contracts.py`. Its `__post_init__` rejects mismatched leading dimensions, non-30 Hz source rate, fewer than two rows, wrong body/hand/root shapes, non-finite numeric arrays, and task/timestamp length mismatches. Arrays are copied into contiguous float64 or int64 storage so later source-buffer mutation cannot alter a stage.

Use these exact fields:

```python
@dataclass
class CanonicalEpisode:
    source_repo_id: str
    source_revision: str
    source_episode_id: int
    source_fps: int
    timestamps: np.ndarray
    task_indices: np.ndarray
    observed_root_wxyz: np.ndarray
    reference_root_wxyz: np.ndarray
    observed_body_q: np.ndarray
    desired_body_q: np.ndarray
    observed_left_hand: np.ndarray
    observed_right_hand: np.ndarray
    desired_left_hand: np.ndarray
    desired_right_hand: np.ndarray
```

- [ ] **Step 4: Implement the source adapter and pinned LeRobot loader**

`adapt_dex3_arrays()` verifies the exact 28 source feature names, fills the first 15 body values from `NOMINAL_G1_MUJOCO`, copies the 14 arms by semantic name into body positions 15:29, splits hands at documented indices, and sets roots to identity.

Add `load_dex3_episode(source_spec, episode_id, root=None)` using:

```python
dataset = LeRobotDataset(
    repo_id=source_spec.repo_id,
    root=root,
    episodes=[episode_id],
    revision=source_spec.revision,
    download_videos=True,
    video_backend="pyav",
)
```

Before iterating rows, compare `dataset.meta.total_episodes` with the pinned count and metadata fps with 30. Extract only rows whose source `episode_index` equals the requested ID; reject gaps, duplicate frame indices, or an episode index that was remapped by the loader without preserving source metadata.

- [ ] **Step 5: Run adapter and provenance tests, lint, and commit**

Run both test files and Ruff on changed Python paths. Expected: pass.

```bash
git add gear_sonic/data/unitree_conversion/contracts.py \
  gear_sonic/data/unitree_conversion/dex3_adapter.py \
  gear_sonic/tests/unitree_conversion/test_dex3_adapter.py
git diff --cached --check
git commit -m "feat: adapt Unitree Dex3 episodes"
```

### Task 4: Add fail-closed Inspire diagnostics

**Files:**

- Modify: `gear_sonic/data/unitree_conversion/contracts.py`
- Create: `gear_sonic/data/unitree_conversion/inspire_diagnostics.py`
- Test: `gear_sonic/tests/unitree_conversion/test_inspire_diagnostics.py`

- [ ] **Step 1: Write diagnostics and negative-capability tests**

```python
import sys

import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.inspire_diagnostics import diagnose_inspire_arrays


GATES = (
    "missing_authoritative_29_joint_order",
    "missing_robot_q_desired_semantics",
    "missing_inspire_to_dex3_calibration",
)


def test_valid_shapes_still_fail_closed_with_three_reasons():
    report = diagnose_inspire_arrays(
        current=np.zeros((5, 36), dtype=np.float32),
        desired=np.zeros((5, 36), dtype=np.float32),
        hand_state=np.zeros((5, 12), dtype=np.float32),
        hand_cmd=np.zeros((5, 12), dtype=np.float32),
        timestamps=np.arange(5, dtype=np.float64) / 30.0,
    )
    assert report.status == "blocked_unverified"
    assert report.gate_reasons == GATES
    assert report.encoder_invoked is False
    assert "gear_sonic.data.unitree_conversion.sonic_encoder" not in sys.modules


def test_wrong_root_plus_joint_shape_is_schema_error():
    with pytest.raises(ValueError, match=r"robot_q_current.*\[N,36\]"):
        diagnose_inspire_arrays(
            current=np.zeros((5, 35), dtype=np.float32),
            desired=np.zeros((5, 36), dtype=np.float32),
            hand_state=np.zeros((5, 12), dtype=np.float32),
            hand_cmd=np.zeros((5, 12), dtype=np.float32),
            timestamps=np.arange(5, dtype=np.float64) / 30.0,
        )
```

- [ ] **Step 2: Run and verify the missing module failure**

Run the standard pytest command on `test_inspire_diagnostics.py`.

- [ ] **Step 3: Implement diagnostic reports without encoder imports**

Add `DiagnosticReport` with `status`, `gate_reasons`, `encoder_invoked`, frame count, timestamp audit, finite counts, quaternion norm extrema, and per-column min/max/mean/std. `inspire_diagnostics.py` must not import `sonic_encoder`, `resampling`, `target_frames`, RobotModel, or ONNX Runtime.

Validate `[N,36]` current/desired arrays, `[N,12]` hand arrays, finite timestamps, and the 7D root split. Apply quaternion validation only after Task 5 exposes it; until then, calculate and report root quaternion norms without interpreting the 29 generic joint columns.

- [ ] **Step 4: Add pinned LeRobot diagnostic loader**

Implement `diagnose_inspire_episode(source_spec, episode_id, root=None)` using revision-pinned `LeRobotDataset`, exact feature keys from the spec, and source task/camera metadata. Return the report only; create no output dataset path.

- [ ] **Step 5: Run tests, scan forbidden imports, and commit**

Run the diagnostics test, then:

```bash
rg -n "sonic_encoder|onnxruntime|target_frames|RobotModel" \
  gear_sonic/data/unitree_conversion/inspire_diagnostics.py
```

Expected: no matches. Run Ruff and commit:

```bash
git add gear_sonic/data/unitree_conversion/contracts.py \
  gear_sonic/data/unitree_conversion/inspire_diagnostics.py \
  gear_sonic/tests/unitree_conversion/test_inspire_diagnostics.py
git diff --cached --check
git commit -m "feat: add gated Inspire diagnostics"
```

### Task 5: Implement strict deployment quaternion math

**Files:**

- Create: `gear_sonic/data/unitree_conversion/quaternion.py`
- Modify: `gear_sonic/data/unitree_conversion/dex3_adapter.py`
- Modify: `gear_sonic/data/unitree_conversion/inspire_diagnostics.py`
- Test: `gear_sonic/tests/unitree_conversion/test_quaternion.py`

- [ ] **Step 1: Write tolerance, SLERP, and packing tests**

```python
import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.quaternion import (
    quat_slerp_deployment,
    quat_to_encoder_rot6d,
    validate_wxyz,
)


def test_rejects_norm_outside_one_e_minus_five():
    with pytest.raises(ValueError, match="unit-norm tolerance"):
        validate_wxyz(np.array([1.00002, 0, 0, 0], dtype=np.float64))


def test_renormalizes_accepted_quaternion():
    q, changed = validate_wxyz(np.array([1.000005, 0, 0, 0], dtype=np.float64))
    np.testing.assert_array_equal(q, [1, 0, 0, 0])
    assert changed is True


def test_slerp_uses_shortest_quaternion_image():
    q = quat_slerp_deployment(
        np.array([1, 0, 0, 0], dtype=np.float64),
        np.array([-1, 0, 0, 0], dtype=np.float64),
        0.4,
    )
    np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-12)


def test_encoder_rot6d_is_row_wise_not_column_contiguous():
    half = np.sqrt(0.5)
    packed = quat_to_encoder_rot6d(np.array([half, 0, 0, half]))
    np.testing.assert_allclose(packed, [0, -1, 1, 0, 0, 0], atol=1e-12)
```

- [ ] **Step 2: Run and verify the missing module failure**

Run the standard pytest command on `test_quaternion.py`.

- [ ] **Step 3: Implement WXYZ Hamilton operations and exact SLERP branch**

Implement `validate_wxyz`, `quat_conjugate`, `quat_mul`, `heading_quat`, `quat_slerp_deployment`, and `quat_to_encoder_rot6d`. Use float64 internally, the `1e-5` input tolerance, `1e-12` degenerate threshold, `0.9995` SLERP branch, and the row-wise rotation output `[R00,R01,R10,R11,R20,R21]`. Do not call SciPy's `Slerp`, because its branch behavior is not the deployed contract.

- [ ] **Step 4: Route both adapters through the policy**

Dex3 identity roots must pass through `validate_wxyz` for uniform counters. Inspire diagnostics applies it to the current and desired root quaternions and changes status to `source_schema_error` when any frame violates the policy; it remains `blocked_unverified` when all root quaternions are valid.

- [ ] **Step 5: Run quaternion, adapter, and diagnostic tests; commit**

Run the four affected test files and Ruff. Expected: pass.

```bash
git add gear_sonic/data/unitree_conversion/quaternion.py \
  gear_sonic/data/unitree_conversion/dex3_adapter.py \
  gear_sonic/data/unitree_conversion/inspire_diagnostics.py \
  gear_sonic/tests/unitree_conversion/test_quaternion.py
git diff --cached --check
git commit -m "feat: match deployment quaternion semantics"
```

### Task 6: Reproduce deployment timeline, velocities, windows, and image indices

**Files:**

- Create: `gear_sonic/data/unitree_conversion/resampling.py`
- Test: `gear_sonic/tests/unitree_conversion/test_resampling.py`

- [ ] **Step 1: Write exact endpoint and terminal-velocity tests**

```python
import numpy as np

from gear_sonic.data.unitree_conversion.resampling import (
    audit_source_timestamps,
    future_windows,
    nearest_image_indices,
    resample_positions,
)


def test_three_source_frames_produce_five_target_frames_and_deployment_velocity():
    positions = np.array([[0.0], [1.0], [2.0]])
    q50, dq50 = resample_positions(positions)
    np.testing.assert_allclose(q50[:, 0], [0.0, 0.6, 1.2, 1.8, 2.0])
    np.testing.assert_allclose(dq50[:, 0], [30.0, 30.0, 30.0, 10.0, 10.0])


def test_final_window_repeats_stored_terminal_velocity():
    q50, dq50 = resample_positions(np.array([[0.0], [1.0], [2.0]]))
    q_window, dq_window = future_windows(q50, dq50, width=10)
    np.testing.assert_allclose(q_window[-1, :, 0], [2.0] * 10)
    np.testing.assert_allclose(dq_window[-1, :, 0], [10.0] * 10)


def test_image_indices_use_round_half_up():
    np.testing.assert_array_equal(nearest_image_indices(3), [0, 1, 1, 2, 2])


def test_nominal_float32_timestamps_pass_audit():
    report = audit_source_timestamps(np.arange(20, dtype=np.float32) / np.float32(30.0))
    assert report.rejected is False
```

- [ ] **Step 2: Run and verify missing implementation**

Run the standard pytest command on `test_resampling.py`.

- [ ] **Step 3: Implement formulas without general-purpose resampling helpers**

Use `T50 = floor(N*50/30)`, `x=j*30/50`, linear interpolation, `quat_slerp_deployment`, forward differences, final-velocity copy, and clamped index matrices. Reject `N < 2`; never concatenate episodes before windowing.

`audit_source_timestamps()` rejects non-finite/non-increasing values and grid error above `1/60`, warns above `0.001`, and reports the maximum error. `nearest_image_indices(N)` returns exactly `T50` int64 indices using `floor(x+0.5)` and clamp.

- [ ] **Step 4: Add randomized deployment-formula comparisons**

Generate deterministic random arrays with `np.random.default_rng(0)` for `N` in `(2,3,10,31)`. In the test, calculate each target sample with a literal scalar implementation of the equations and compare every position, velocity, and image index. This catches vectorization endpoint mistakes independently of the hand-authored examples.

- [ ] **Step 5: Run tests, Ruff, and commit**

```bash
git add gear_sonic/data/unitree_conversion/resampling.py \
  gear_sonic/tests/unitree_conversion/test_resampling.py
git diff --cached --check
git commit -m "feat: match deployment motion resampling"
```

### Task 7: Build and run the exact 1,247D SONIC encoder contract

**Files:**

- Modify: `gear_sonic/data/unitree_conversion/contracts.py`
- Create: `gear_sonic/data/unitree_conversion/sonic_encoder.py`
- Test: `gear_sonic/tests/unitree_conversion/test_sonic_encoder.py`

- [ ] **Step 1: Write tensor slice, inactive-zero, and session-contract tests**

```python
import numpy as np
import pytest

from gear_sonic.data.unitree_conversion.sonic_encoder import (
    SonicEncoder,
    build_encoder_orientation_window,
    build_g1_encoder_input,
)


def test_g1_encoder_layout_is_exactly_1247d():
    positions = np.arange(290, dtype=np.float32).reshape(10, 29)
    velocities = positions + 1000
    orientations = np.arange(60, dtype=np.float32).reshape(10, 6)
    tensor = build_g1_encoder_input(positions, velocities, orientations)
    assert tensor.shape == (1, 1247)
    assert tensor.dtype == np.float32
    np.testing.assert_array_equal(tensor[0, :4], [0, 0, 0, 0])
    np.testing.assert_array_equal(tensor[0, 4:294], positions.reshape(-1))
    np.testing.assert_array_equal(tensor[0, 294:584], velocities.reshape(-1))
    np.testing.assert_array_equal(tensor[0, 584:644], orientations.reshape(-1))
    np.testing.assert_array_equal(tensor[0, 644:], 0)


def test_builder_rejects_wrong_window_shape():
    with pytest.raises(ValueError, match=r"positions.*\[10,29\]"):
        build_g1_encoder_input(
            np.zeros((9, 29), dtype=np.float32),
            np.zeros((10, 29), dtype=np.float32),
            np.zeros((10, 6), dtype=np.float32),
        )


class FakeSession:
    def get_inputs(self):
        return [type("Tensor", (), {"name": "wrong", "shape": [1, 1247], "type": "tensor(float)"})()]

    def get_outputs(self):
        return [type("Tensor", (), {"name": "encoded_tokens", "shape": [1, 64], "type": "tensor(float)"})()]


def test_session_rejects_wrong_tensor_name():
    with pytest.raises(ValueError, match="obs_dict"):
        SonicEncoder.from_session(FakeSession())
```

- [ ] **Step 2: Run and verify missing encoder implementation**

Run the standard pytest command on `test_sonic_encoder.py`.

- [ ] **Step 3: Implement the pure layout and orientation-window builder**

Add `ResampledEpisode` to `contracts.py` with 50 Hz observed/reference roots, observed/desired body, desired velocity, and hands. `build_frame_encoder_input(episode, frame_index)` must:

1. Clamp ten future indices inside this episode.
2. Reorder 29 positions/velocities by name into `G1_ISAACLAB_NAMES`.
3. Calculate `q_apply` from frame-zero roots.
4. Calculate ten current-observed-relative reference rotations.
5. Pack `[0:644]` and leave `[644:1247]` exactly zero.

`build_g1_encoder_input()` remains a shape-validating pure function so slice tests do not depend on RobotModel or ONNX Runtime.

Expose `build_encoder_orientation_window(initial_observed_root_wxyz, current_observed_root_wxyz, initial_reference_root_wxyz, future_reference_root_wxyz)` as a pure helper. It returns exactly `[10,6]` values using the pinned heading-alignment equation and row-wise first-two-column flattening from the specification. The parity fixture must call this helper from raw roots instead of trusting precomputed orientation values.

- [ ] **Step 4: Implement verified artifact loading and ONNX inference**

`SonicEncoder.from_artifacts(lock, cache_dir)` materializes both files through `provenance.py`, parses the YAML to verify G1 mode ID `0` and the four required observations, creates an `onnxruntime.InferenceSession` using `providers=["CPUExecutionProvider"]`, and verifies exactly:

```python
input_contract = ("obs_dict", [1, 1247], "tensor(float)")
output_contract = ("encoded_tokens", [1, 64], "tensor(float)")
```

`encode()` accepts only finite C-contiguous float32 `[1,1247]`, invokes `session.run(["encoded_tokens"], {"obs_dict": tensor})`, and returns finite float32 `[1,64]` without squeezing.

- [ ] **Step 5: Add a local-artifact integration test**

When `gear_sonic_deploy/policy/low_latency/model_encoder.onnx` exists, verify its hash, construct the session, encode an identity/nominal input, and assert finite `[1,64]`. Otherwise use `pytest.skip("pinned low-latency encoder is not materialized")`; do not download in the unit suite.

- [ ] **Step 6: Run tests, Ruff, and commit**

```bash
git add gear_sonic/data/unitree_conversion/contracts.py \
  gear_sonic/data/unitree_conversion/sonic_encoder.py \
  gear_sonic/tests/unitree_conversion/test_sonic_encoder.py
git diff --cached --check
git commit -m "feat: add low latency SONIC encoder contract"
```

### Task 8: Enforce source camera counts and stream the exact 50 Hz video timeline

**Files:**

- Create: `gear_sonic/data/unitree_conversion/video_timeline.py`
- Test: `gear_sonic/tests/unitree_conversion/test_video_timeline.py`

- [ ] **Step 1: Write three-frame synthetic video tests**

Create a helper in the test that writes a 30 Hz H.264 RGB video with three solid frames `[red, green, blue]` using PyAV. Then test:

```python
def test_streams_exact_target_count_with_nearest_indices(rgb_video):
    timeline = inspect_video(rgb_video, expected_frames=3, expected_size=(640, 480))
    frames = list(iter_resampled_video(timeline, target_fps=50))
    assert timeline.decoded_frames == 3
    assert len(frames) == 5
    assert [frame.source_index for frame in frames] == [0, 1, 1, 2, 2]
    assert all(frame.rgb.shape == (480, 640, 3) for frame in frames)
    assert all(frame.rgb.dtype == np.uint8 for frame in frames)


def test_primary_count_mismatch_is_not_repaired(rgb_video):
    with pytest.raises(ValueError, match="decoded frame count 3 != data frame count 4"):
        inspect_video(rgb_video, expected_frames=4, expected_size=(640, 480))
```

- [ ] **Step 2: Run and verify the missing module failure**

Run the standard pytest command on `test_video_timeline.py`.

- [ ] **Step 3: Implement two-pass streaming video access**

`inspect_video()` performs a decode pass to count frames, validate RGB-convertibility, and verify shape. `iter_resampled_video()` opens a second decoder and walks the monotonic `nearest_image_indices(N)` sequence, retaining only the current source frame while yielding repeated target references. It must not load the full video into RAM or use source presentation timestamps to alter the selected index.

- [ ] **Step 4: Implement repository-wide optional camera preflight**

`choose_target_camera_schema(episode_reports, camera_map)` always requires the configured primary. It includes a wrist key only when every selected episode reports the source stream present and exact. Return one immutable schema plus omission reasons so target features cannot vary by episode.

- [ ] **Step 5: Run tests, Ruff, and commit**

```bash
git add gear_sonic/data/unitree_conversion/video_timeline.py \
  gear_sonic/tests/unitree_conversion/test_video_timeline.py
git diff --cached --check
git commit -m "feat: add exact Unitree video timeline"
```

### Task 9: Construct every SONIC VLA target field and task annotation

**Files:**

- Create: `gear_sonic/data/unitree_conversion/target_frames.py`
- Modify: `gear_sonic/data/exporter.py`
- Test: `gear_sonic/tests/unitree_conversion/test_target_frames.py`

- [ ] **Step 1: Write exact field/dtype tests with a fake FK model**

Use a fake RobotModel exposing a 43-name order, semantic assembly, and deterministic left/right wrist poses. Assert:

```python
def test_frame_contains_exact_derived_and_neutral_contract(builder, resampled_episode):
    frame = builder.build(resampled_episode, frame_index=2, token=np.arange(64, dtype=np.float32))
    assert frame["observation.state"].shape == (43,)
    assert frame["observation.state"].dtype == np.float64
    assert frame["observation.eef_state"].shape == (14,)
    assert frame["action.wbc"].shape == (43,)
    assert frame["action.motion_token"].dtype == np.float64
    np.testing.assert_array_equal(frame["teleop.delta_heading"], np.zeros(1, np.float64))
    np.testing.assert_array_equal(frame["teleop.target_body_orientation"], [1, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(frame["teleop.planner_facing"], [1, 0, 0])
    np.testing.assert_array_equal(frame["teleop.planner_speed"], [-1])
    np.testing.assert_array_equal(frame["teleop.smpl_frame_index"], np.array([2], np.int64))
    assert frame["timestamp"] == np.float32(2 / 50.0)
```

Add a second test proving `observation.eef_state` comes from observed state while `action.wbc` contains desired state. Add a third test proving raw left DDS hand order is semantically reordered inside the 43D arrays but unchanged in `teleop.left_hand_joints`.

- [ ] **Step 2: Write deterministic task-order regression test**

Test a two-task episode `['z task', 'a task', 'z task']` and assert episode metadata keeps `['z task', 'a task']` in first occurrence while pre-registered global target indices are lexicographic: `a task -> 0`, `z task -> 1`.

- [ ] **Step 3: Run and verify failures**

Run the standard pytest command on `test_target_frames.py`.

Expected: missing builder plus the existing `list(set(tasks))` behavior fails stable order.

- [ ] **Step 4: Implement `TargetFrameBuilder`**

Instantiate the production RobotModel through `get_g1_robot_model(waist_location="lower_and_upper_body")`. Assemble observed and desired 43D configurations by semantic names. Recompute FK for observed state and emit the exact derived and neutral tables from the spec using explicit NumPy constructors and dtypes. Reject non-finite tokens or wrong frame index.

Set:

```python
frame["observation.cpp_rotation_offset"] = episode.reference_root_wxyz[0].astype(np.float64)
frame["observation.init_base_quat"] = episode.observed_root_wxyz[0].astype(np.float64)
frame["timestamp"] = np.float32(frame_index / 50.0)
frame["task"] = task_text
```

- [ ] **Step 5: Make exporter episode task order stable**

In `gear_sonic/data/exporter.py`, replace only:

```python
episode_tasks = list(set(tasks))
```

with:

```python
episode_tasks = list(dict.fromkeys(tasks))
```

Do not change task-index assignment or unrelated exporter behavior.

- [ ] **Step 6: Run target tests, exporter-adjacent tests, Ruff, and commit**

Run `test_target_frames.py` and any existing exporter tests discovered by `rg -l "Gr00tDataExporter" gear_sonic/tests`. Expected: pass.

```bash
git add gear_sonic/data/unitree_conversion/target_frames.py \
  gear_sonic/data/exporter.py \
  gear_sonic/tests/unitree_conversion/test_target_frames.py
git diff --cached --check
git commit -m "feat: build exact SONIC VLA target frames"
```

### Task 10: Add immutable episode staging, checksum resume, and deterministic merge

**Files:**

- Create: `gear_sonic/data/unitree_conversion/staging.py`
- Create: `gear_sonic/data/unitree_conversion/validation.py`
- Test: `gear_sonic/tests/unitree_conversion/test_staging.py`

- [ ] **Step 1: Write atomic/resume/corruption tests**

```python
def test_valid_stage_resumes_only_with_matching_identity(tmp_path, stage_fixture):
    stage = write_stage(tmp_path, stage_fixture)
    assert can_resume(stage.path, stage_fixture.identity) is True
    assert can_resume(stage.path, replace(stage_fixture.identity, encoder_sha256="0" * 64)) is False


def test_checksum_corruption_forces_rebuild(tmp_path, stage_fixture):
    stage = write_stage(tmp_path, stage_fixture)
    parquet = stage.path / "frame-data.parquet"
    parquet.write_bytes(parquet.read_bytes() + b"corruption")
    assert can_resume(stage.path, stage_fixture.identity) is False


def test_merge_order_does_not_depend_on_completion_order(tmp_path, two_stages):
    first = merge_stages(list(reversed(two_stages)), tmp_path / "first")
    second = merge_stages(two_stages, tmp_path / "second")
    assert dataset_manifest_digest(first) == dataset_manifest_digest(second)
    assert read_source_episode_order(first) == sorted(read_source_episode_order(first))


def test_merge_rejects_stages_from_different_source_repositories(tmp_path, mixed_stages):
    with pytest.raises(ValueError, match="source repository"):
        merge_stages(mixed_stages, tmp_path / "mixed")
```

- [ ] **Step 2: Run and verify missing staging implementation**

Run the standard pytest command on `test_staging.py`.

- [ ] **Step 3: Implement immutable stage writes and validation reports**

Write frame arrays to `frame-data.parquet`, already-resampled videos to `videos/`, exact provenance/config to `manifest.json`, and structural results to `validation.json`. Generate sorted `checksums.sha256` entries over every artifact except the checksum file itself.

Use `tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent)`, fsync files and containing directories, and `temp_path.rename(final)` only after validation. If the immutable final path exists with different checksums, raise `FileExistsError`; never overwrite it.

- [ ] **Step 4: Implement deterministic target merge**

Require every merge invocation to contain exactly one `source_repo_id`, then sort stages by `source_episode_id`. Create a sibling temporary output with `Gr00tDataExporter.create(fps=50, ...)`, pre-register lexicographically sorted task strings, then stream staged rows/videos into the exporter. Verify exact episode/global indices, timestamps, video counts, stats, and source manifest before atomically renaming the output. A full collection run invokes this merge once per approved upstream repository; it never combines repositories into one target dataset.

Do not rely on stage completion order, Python set order, or existing partial output metadata.

- [ ] **Step 5: Run tests, Ruff, and commit**

```bash
git add gear_sonic/data/unitree_conversion/staging.py \
  gear_sonic/data/unitree_conversion/validation.py \
  gear_sonic/tests/unitree_conversion/test_staging.py
git diff --cached --check
git commit -m "feat: stage and merge converted episodes safely"
```

### Task 11: Orchestrate source-specific pipelines and CLI without opening the Inspire gate

**Files:**

- Create: `gear_sonic/data/unitree_conversion/pipeline.py`
- Create: `gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py`
- Test: `gear_sonic/tests/unitree_conversion/test_pipeline.py`

- [ ] **Step 1: Write fake-resolver orchestration tests**

```python
def test_dex3_smoke_processes_only_five_locked_ids(fake_components, smoke_lock, tmp_path):
    report = run_dex3_pipeline(
        lock=smoke_lock,
        output_root=tmp_path,
        components=fake_components,
        smoke=True,
    )
    assert report.source_episode_ids == (0, 78, 155, 233, 310)
    assert report.validated_episode_count == 5
    assert fake_components.resolver.requested_ids == [0, 78, 155, 233, 310]


def test_inspire_smoke_never_constructs_encoder(fake_components, smoke_lock, tmp_path):
    fake_components.encoder_factory = lambda: (_ for _ in ()).throw(
        AssertionError("encoder constructed for gated Inspire path")
    )
    report = run_inspire_diagnostics(
        lock=smoke_lock,
        output_root=tmp_path,
        components=fake_components,
        smoke=True,
    )
    assert report.source_episode_ids == (0, 152, 304, 456, 608)
    assert report.status_counts == {"blocked_unverified": 5}
    assert report.encoder_invocation_count == 0
    assert report.target_dataset_paths == ()
```

Add a resume test where two stages are valid and three are missing; only three adapters/encoders may be invoked before deterministic merge.

- [ ] **Step 2: Run and verify missing pipeline/CLI**

Run the standard pytest command on `test_pipeline.py`.

- [ ] **Step 3: Implement typed pipeline reports and dependency injection**

`run_dex3_pipeline()` performs, in order: lock validation, artifact validation, source metadata validation, optional-camera cohort preflight, episode adaptation, resampling, per-frame encoding/target construction, structural validation, immutable staging, and final merge. It continues to the next selected episode after a classified episode failure but refuses final success unless all five selected IDs validate.

`run_inspire_diagnostics()` imports only provenance, source access, video preflight, and Inspire diagnostics. Keep the encoder import inside `run_dex3_pipeline()` so importing or invoking the Inspire function cannot initialize ONNX Runtime.

Define one `PipelineReport` with `source_episode_ids`, `validated_episode_count`, `failed_episode_count`, `status_counts`, `gate_reasons`, `encoder_invocation_count`, `target_dataset_paths`, `output_root`, and classified episode reports. `target_dataset_paths` is a tuple ordered by source repository and is empty for Inspire diagnostics. A `scope: smoke` lock requires `smoke=True`; a reviewed `scope: full` lock accepts `smoke=False` and processes all explicitly eligible repositories/episodes. Group selected episodes by `source_repo_id`, preflight and merge each group independently, and never combine upstream repositories in one target dataset. Any `approved: false` source is a preflight error.

- [ ] **Step 4: Implement explicit CLI modes**

Use a Tyro dataclass with:

```python
@dataclass(frozen=True)
class ConvertConfig:
    source_lock: Path = Path(
        "gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml"
    )
    output_root: Path = Path("outputs/unitree_sonic_conversion")
    kind: Literal["dex3", "inspire"] = "dex3"
    smoke: bool = True
    cache_dir: Path | None = None
    resume: bool = True
    workers: int = 1
```

Reject `smoke=False` for a `scope: smoke` lock, reject `workers < 1`, and print one final JSON report path. `kind=inspire` writes under `diagnostic-only/`; it has no encoder/model override flag.

- [ ] **Step 5: Run unit tests and CLI help**

Run pipeline tests, then:

```bash
PYTHONPATH=/home/jihun/work/GR00T-WholeBodyControl/.venv_data_collection/lib/python3.10/site-packages \
/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py --help
```

Expected: help lists `source-lock`, `output-root`, `kind`, `smoke`, `cache-dir`, `resume`, and `workers`; no unverified-Inspire override exists.

- [ ] **Step 6: Run Ruff and commit**

```bash
git add gear_sonic/data/unitree_conversion/pipeline.py \
  gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py \
  gear_sonic/tests/unitree_conversion/test_pipeline.py
git diff --cached --check
git commit -m "feat: orchestrate Unitree SONIC conversion"
```

### Task 12: Add independent deployment/TensorRT parity and golden vectors

**Files:**

- Create: `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/sonic_encoder_parity_test.cpp`
- Create: `gear_sonic/tests/data/unitree_conversion/golden_encoder_case.npz`
- Create: `gear_sonic/tests/data/unitree_conversion/golden_encoder_token.npy`
- Modify: `gear_sonic/tests/unitree_conversion/test_sonic_encoder.py`

- [ ] **Step 1: Add a Python golden-fixture test that initially has no fixture**

```python
GOLDEN_DIR = Path("gear_sonic/tests/data/unitree_conversion")


def test_golden_encoder_case_matches_checked_token(pinned_encoder):
    case = np.load(GOLDEN_DIR / "golden_encoder_case.npz")
    orientations = build_encoder_orientation_window(
        case["initial_observed_root_wxyz"],
        case["current_observed_root_wxyz"],
        case["initial_reference_root_wxyz"],
        case["future_reference_root_wxyz"],
    )
    np.testing.assert_allclose(orientations, case["orientations"], atol=1e-12, rtol=0)
    tensor = build_g1_encoder_input(
        case["positions"], case["velocities"], orientations
    )
    np.testing.assert_array_equal(tensor, case["encoder_input"])
    token = pinned_encoder.encode(tensor)
    np.testing.assert_allclose(
        token,
        np.load(GOLDEN_DIR / "golden_encoder_token.npy"),
        atol=1e-5,
        rtol=1e-5,
    )


def test_large_semantic_perturbation_changes_token(pinned_encoder):
    case = np.load(GOLDEN_DIR / "golden_encoder_case.npz")
    baseline = pinned_encoder.encode(case["encoder_input"])
    perturbed_positions = case["positions"].copy()
    perturbed_positions[:, 11] += np.float32(0.5)
    perturbed = build_g1_encoder_input(
        perturbed_positions, case["velocities"], case["orientations"]
    )
    assert np.count_nonzero(pinned_encoder.encode(perturbed) != baseline) >= 1
```

- [ ] **Step 2: Add a deployment unit test controlled by exact environment paths**

The C++ GTest reads:

```text
SONIC_PARITY_CASE_JSON   canonical positions, velocities, and root quaternions
SONIC_PARITY_MODEL       pinned model_encoder.onnx
SONIC_PARITY_INPUT_OUT   raw float32, exactly 1247 packed values
SONIC_PARITY_TOKEN_OUT   raw float32, exactly 64 encoded values
```

The canonical JSON contains ten `[29]` position rows, ten `[29]` velocity rows, `initial_observed_root_wxyz`, `current_observed_root_wxyz`, `initial_reference_root_wxyz`, and ten `future_reference_root_wxyz` rows. The C++ test independently computes `q_apply`, current-relative rotations through `math_utils.hpp`, row-wise 6D packing, and all 1,247 slice offsets before inference. It rejects wrong JSON shapes and non-finite values.

If variables are absent, use `GTEST_SKIP()`. Otherwise initialize `EncoderEngine` with `use_fp16=false`, assert names/dimensions, encode the independently packed vector, assert 64 finite outputs, write exactly 4,988 input bytes and 256 token bytes. The test uses the existing `run_tests` target, which already globs `unit_tests/*.cpp`; do not edit the currently dirty CMake files.

Core test body:

```cpp
EncoderEngine encoder;
ASSERT_TRUE(encoder.Initialize(model_path, false));
ASSERT_EQ(encoder.GetInputTensorName(), "obs_dict");
ASSERT_EQ(encoder.GetOutputTensorName(), "encoded_tokens");
ASSERT_EQ(encoder.GetInputDimension(), 1247u);
ASSERT_EQ(encoder.GetTokenDimension(), 64u);
const auto input = PackCanonicalG1Case(case_json);
ASSERT_EQ(input.size(), 1247u);
std::copy(input.begin(), input.end(), encoder.GetInputBuffer().begin());
ASSERT_TRUE(encoder.Encode());
for (float value : encoder.GetTokenBuffer()) ASSERT_TRUE(std::isfinite(value));
```

- [ ] **Step 3: Generate the canonical fixture deterministically**

Use `np.random.default_rng(20260810)` for a bounded desired arm trajectory, deployed nominal lower body, normalized non-identity observed/reference heading fixtures, and deployment resampling. Save named arrays `positions`, `velocities`, `initial_observed_root_wxyz`, `current_observed_root_wxyz`, `initial_reference_root_wxyz`, `future_reference_root_wxyz`, `orientations`, and `encoder_input`; avoid object arrays and compression-dependent metadata. Export the canonical fields to sorted-key JSON for the C++ run.

Run the Python builder twice and require identical SHA-256 output before retaining the fixture.

- [ ] **Step 4: Build and run TensorRT parity**

Build the existing `run_tests` target using the repository's documented CMake command. Then export the fixture's raw encoder input to `/tmp/unitree_sonic_encoder_input.f32` and run:

```bash
SONIC_PARITY_CASE_JSON=/tmp/unitree_sonic_encoder_case.json \
SONIC_PARITY_MODEL=gear_sonic_deploy/policy/low_latency/model_encoder.onnx \
SONIC_PARITY_INPUT_OUT=/tmp/unitree_sonic_encoder_input.f32 \
SONIC_PARITY_TOKEN_OUT=/tmp/unitree_sonic_encoder_token.f32 \
gear_sonic_deploy/target/release/run_tests \
  --gtest_filter=SonicEncoderParity.EncodesPinnedFloat32Input
```

Expected: one passing GTest, exactly 4,988 packed-input bytes, and exactly 256 token bytes.

- [ ] **Step 5: Compare ORT/TensorRT and freeze the golden token**

First compare the raw C++ packed input with Python `encoder_input` using `atol=1e-7, rtol=0`. Then compare raw TensorRT output with CPU ONNX Runtime using `atol=1e-5, rtol=1e-5`, and save the CPU token as `golden_encoder_token.npy`. Record both tracked file hashes in test constants so accidental fixture regeneration fails loudly.

- [ ] **Step 6: Run golden tests and commit**

Run `test_sonic_encoder.py`, the filtered GTest, and `git diff --check`.

```bash
git add gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/unit_tests/sonic_encoder_parity_test.cpp \
  gear_sonic/tests/data/unitree_conversion/golden_encoder_case.npz \
  gear_sonic/tests/data/unitree_conversion/golden_encoder_token.npy \
  gear_sonic/tests/unitree_conversion/test_sonic_encoder.py
git diff --cached --check
git commit -m "test: pin SONIC encoder deployment parity"
```

### Task 13: Implement deterministic MuJoCo replay metrics and orchestration

**Files:**

- Create: `gear_sonic/data/unitree_conversion/replay.py`
- Create: `gear_sonic/scripts/replay_unitree_sonic_smoke.py`
- Test: `gear_sonic/tests/unitree_conversion/test_replay.py`

- [ ] **Step 1: Write pure metric and cohort aggregation tests**

```python
def test_metrics_exclude_only_first_second_of_two_second_warmup():
    collector = ReplayMetricsCollector(
        sim_dt=0.005,
        exclude_before_s=1.0,
        robot_mass_kg=35.0,
        effort_limits=np.array([10.0, 20.0]),
        joint_limits=np.array([[-1.0, 1.0], [-2.0, 2.0]]),
    )
    collector.add(sample(time_s=0.5, torques=[100, 100], contact_force=99999))
    collector.add(sample(time_s=1.0, torques=[5, 10], contact_force=100))
    report = collector.finalize()
    assert report.torque_ratio_max == 0.5
    assert report.contact_force_max == 100


def test_cohort_uses_maximum_not_mean():
    cohort = aggregate_replay_reports([
        report_with(torque_ratio_max=0.4),
        report_with(torque_ratio_max=1.06),
        report_with(torque_ratio_max=0.3),
    ])
    assert cohort.torque_ratio_max == 1.06
    assert cohort.accepted is False
```

Add boundary tests for root height, roll/pitch, joint overshoot, velocity, torque ratio, contact force, non-finite data, and controller-fault count.

- [ ] **Step 2: Run and verify missing replay implementation**

Run the standard pytest command on `test_replay.py`.

- [ ] **Step 3: Implement exact per-step metrics**

`ReplayMetricsCollector` computes the spec equations, keeps maxima plus p50/p95/p99 samples, and evaluates exact inclusive thresholds. Contact force comes from `mujoco.mj_contactForce` six-vectors; use Euclidean norm per active contact and then the maximum. Robot mass comes from `mujoco.mj_getTotalmass`; effort limits and joint limits come from the simulator's loaded robot/model, not duplicated constants.

- [ ] **Step 4: Implement end-to-end replay context management**

The replay script starts `BaseSimulator` headless with seed `0`, starts the existing low-latency C++ deployment with `--input-type zmq_manager`, waits for readiness with bounded timeouts, and sends staged token/hand frames through the same packed v4 action path used by `run_vla_inference.py`. Its CLI accepts source episode IDs and resolves them through the merged dataset's immutable provenance manifest; target-local episode indices are never assumed to equal upstream IDs.

For each episode it executes exactly 2.0 s first-frame warm-up, all target frames at 50 Hz, and 1.0 s final hold while sampling MuJoCo at simulation rate. On exit, send SIGINT, wait five seconds, then terminate only the owned process if still running. Never kill by executable name or broad process pattern.

- [ ] **Step 5: Run unit tests and one synthetic standing replay**

Run `test_replay.py`, then replay a generated ten-second nominal token/hand stage. Expected: report contains duration, seed, mass, timestep, all maxima/percentiles, and acceptance result; no background process remains owned by the harness.

- [ ] **Step 6: Run Ruff and commit**

```bash
git add gear_sonic/data/unitree_conversion/replay.py \
  gear_sonic/scripts/replay_unitree_sonic_smoke.py \
  gear_sonic/tests/unitree_conversion/test_replay.py
git diff --cached --check
git commit -m "feat: validate converted episodes in MuJoCo"
```

### Task 14: Run the fixed network smoke cohorts and enforce outcomes

**Files:**

- Create: `gear_sonic/tests/unitree_conversion/test_smoke_network.py`

- [ ] **Step 1: Write a network-marked fixed-cohort test**

The test loads only `smoke_sources.yaml`, invokes Dex3 conversion and Inspire diagnostics, and asserts:

```python
assert dex_report.source_episode_ids == (0, 78, 155, 233, 310)
assert dex_report.validated_episode_count == 5
assert dex_report.failed_episode_count == 0
assert inspire_report.source_episode_ids == (0, 152, 304, 456, 608)
assert inspire_report.status_counts == {"blocked_unverified": 5}
assert inspire_report.gate_reasons == {
    "missing_authoritative_29_joint_order",
    "missing_robot_q_desired_semantics",
    "missing_inspire_to_dex3_calibration",
}
assert inspire_report.encoder_invocation_count == 0
assert inspire_report.target_dataset_paths == ()
```

- [ ] **Step 2: Run five Inspire diagnostic episodes first**

Run:

```bash
PYTHONPATH=/home/jihun/work/GR00T-WholeBodyControl/.venv_data_collection/lib/python3.10/site-packages \
/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py \
  --kind inspire \
  --output-root outputs/unitree_sonic_smoke
```

Expected: exactly five `blocked_unverified` diagnostics and no target dataset.

- [ ] **Step 3: Run five Dex3 export/token episodes**

Run the same CLI with `--kind dex3`. Expected: exactly five validated stages, deterministic merge, and no source episode outside `[0,78,155,233,310]` downloaded by the adapter.

- [ ] **Step 4: Run all five Dex3 replays**

```bash
PYTHONPATH=/home/jihun/work/GR00T-WholeBodyControl/.venv_data_collection/lib/python3.10/site-packages \
/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python \
gear_sonic/scripts/replay_unitree_sonic_smoke.py \
  --dataset-root outputs/unitree_sonic_smoke \
  --source-episode-ids 0 78 155 233 310
```

Expected: five per-episode reports plus one cohort report using maxima across episodes. Any numerical gate failure keeps the task open for diagnosis; do not relax a threshold without a spec amendment.

- [ ] **Step 5: Re-run conversion with resume and compare manifests**

Run the Dex3 smoke command again with resume enabled. Expected: zero adapter/encoder work, five checksum-validated stage hits, and a byte-identical canonical dataset manifest digest.

- [ ] **Step 6: Commit the network test only after the fixed cohort passes**

```bash
git add gear_sonic/tests/unitree_conversion/test_smoke_network.py
git diff --cached --check
git commit -m "test: smoke Unitree SONIC conversion cohorts"
```

### Task 15: Document operation and run final verification

**Files:**

- Create: `docs/source/tutorials/data_conversion.md`
- Modify: `docs/source/index.rst`

- [ ] **Step 1: Write the operator guide**

Document immutable artifact/source URIs, exact five-plus-five IDs, full-collection lock discovery/review, cache/output layout, Dex3 conversion command, Inspire diagnostic command, resume behavior, validation report locations, replay command, and the three Inspire gate requirements. State that Inspire output is not training-ready and that there is no override flag.

- [ ] **Step 2: Add the tutorial to the documentation toctree**

Add `tutorials/data_conversion` beside other data/training tutorials in `docs/source/index.rst` without reordering unrelated entries.

- [ ] **Step 3: Run the complete network-free suite**

```bash
PYTHONPATH=/home/jihun/work/GR00T-WholeBodyControl/.venv_data_collection/lib/python3.10/site-packages \
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/jihun/work/GR00T-WholeBodyControl/.venv_teleop/bin/python -m pytest -q -p no:cacheprovider \
gear_sonic/tests/unitree_conversion \
--ignore=gear_sonic/tests/unitree_conversion/test_smoke_network.py
```

Expected: all network-free conversion tests pass with no failures or errors.

- [ ] **Step 4: Run formatting, compilation, and staged-diff checks**

```bash
ruff check --no-cache gear_sonic/data/unitree_conversion \
  gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py \
  gear_sonic/scripts/lock_unitree_unifolm_sources.py \
  gear_sonic/scripts/replay_unitree_sonic_smoke.py \
  gear_sonic/tests/unitree_conversion
ruff format --check --no-cache gear_sonic/data/unitree_conversion \
  gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py \
  gear_sonic/scripts/lock_unitree_unifolm_sources.py \
  gear_sonic/scripts/replay_unitree_sonic_smoke.py \
  gear_sonic/tests/unitree_conversion
```

Compile all new Python modules with `python -m py_compile`, run the filtered C++ parity GTest, run `git diff --check`, and confirm `git status --short` contains no unintended staged path.

- [ ] **Step 5: Verify the spec requirement matrix**

Create a final report mapping every spec section to its test/report evidence: provenance; Dex3 mappings; Inspire gates; quaternion policy; resampling; heading/6D; 1,247D/64D contract; target fields; video/tasks; cohorts; golden parity; staging/resume; and replay thresholds. A section without evidence prevents completion.

- [ ] **Step 6: Commit documentation and final evidence**

```bash
git add docs/source/tutorials/data_conversion.md docs/source/index.rst
git diff --cached --check
git commit -m "docs: add Unitree SONIC conversion workflow"
```

## Execution order and stopping conditions

Tasks 1 through 12 establish network-free correctness and deployment parity. Task 13 adds replay. Task 14 is the only implementation task allowed to download the fixed episode cohorts; it must not broaden to the full collections. Task 15 closes documentation and verification.

Stop Dex3 publication on any hash, schema, golden, episode, or replay failure. Stop Inspire after producing five `blocked_unverified` reports. Enabling Inspire token generation is outside this plan even if plausible mappings are discovered during diagnostics.
