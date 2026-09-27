# Unitree G1 Teleoperation to GEAR-SONIC Dataset Conversion Design

## Status

Approved for specification on 2026-08-10. Dex3 export and SONIC token generation are implementation-ready. Inspire support is explicitly diagnostic-only and fail-closed until its three external semantic requirements are satisfied.

## Goal

Build a reproducible, streaming conversion pipeline for the Unitree G1 + Dex3 datasets in `UnifoLM_G1_Dex3_Dataset` and the Unitree G1 + Inspire datasets in the mixed `UnifoLM_WBT_Dataset` collection.

The training-ready Dex3 output uses the repository's SONIC VLA schema at 50 Hz and contains a 64D low-latency SONIC action token plus two 7D Dex3 hand command vectors. The implementation smoke cohort contains five exported Dex3 episodes and a separate five-episode Inspire diagnostic cohort.

This design does not infer undocumented Inspire semantics and does not permit an unverified Inspire dataset to appear training-ready.

## Non-goals

- Training, fine-tuning, or deploying a VLA policy.
- Changing the deployed low-latency SONIC encoder contract.
- Treating `action.wbc` as a decoded or normalized SONIC policy action.
- Inferring the Inspire 29-joint order from limits, statistics, or forward kinematics.
- Guessing whether Inspire `robot_q_desired` is a controller target, WBC input, WBC output, or reference.
- Inventing an Inspire-to-Dex3 hand calibration from similarly named fingers.
- Combining unrelated source repositories into one output dataset. Each source repository produces one independently resumable target dataset.

## Source and artifact provenance

### Immutable SONIC artifacts

The encoder and its observation config are fetched from one immutable Hugging Face model revision, not from `main` and not merely from an untracked local file:

| Artifact | Immutable source | Local materialization | Bytes | SHA-256 |
|---|---|---|---:|---|
| Encoder | `hf://models/nvidia/GEAR-SONIC@9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2/low_latency/model_encoder.onnx` | `gear_sonic_deploy/policy/low_latency/model_encoder.onnx` | 45,933,505 | `60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b` |
| Observation config | `hf://models/nvidia/GEAR-SONIC@9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2/low_latency/observation_config.yaml` | `gear_sonic_deploy/policy/low_latency/observation_config.yaml` | 3,258 | `582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab` |

The pipeline downloads with `hf_hub_download(repo_id, filename, revision=<full SHA>)`, verifies the exact size and SHA-256 before opening either file, and records both the immutable URI and local resolved path. It refuses an artifact supplied only through a floating revision or a local path without the expected hash. This is required because ONNX files are ignored by `.gitignore` and the observation config is not part of the repository commit below.

Deployment semantics are pinned separately to repository commit:

```text
6220f4e210e14c7f804f727da94c8886850d513b
```

The conversion manifest records the dirty-state flag and SHA-256 of each local deployment source file used for parity testing. A dirty deployment source does not silently inherit the commit's identity.

The Dex3 DDS semantic authority is the official Unitree controller file `teleop/robot_control/robot_hand_unitree.py` at `unitreerobotics/xr_teleoperate` commit `7dc9aa1a6edbf4a9f4f887d8ab6fc449ea5135f6`. Source feature names are read from the pinned dataset metadata below.

### Immutable smoke datasets

The acceptance cohorts use these fixed sources:

| Cohort | Repository | Revision | Episode count at revision |
|---|---|---|---:|
| Dex3 export/token | `unitreerobotics/G1_Dex3_Pouring_Dataset` | `c9552eb3b1cb610cd6227555e9e98b1bde826a77` | 311 |
| Inspire diagnostic-only | `unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly` | `24e3e4d88a5020bdb4b3046ec09b09dc56f8d1f1` | 609 |

Every full-collection conversion begins from a source lock containing the ordered repository list and a full 40-character revision for every repository. Discovery may generate a proposed lock, but conversion never consumes floating `main`, a collection page directly, or a revision resolved after output staging begins. The lock's SHA-256 is part of the conversion identity.

For each source repository, the manifest records:

- repository ID and immutable revision;
- source lock SHA-256;
- source metadata SHA-256 values;
- selected source episode IDs and frame counts;
- source camera keys and decoded frame counts;
- encoder/config immutable URIs, hashes, tensor contract, and semantic repository commit;
- converter version and conversion-configuration hash.

## Architecture

```text
immutable HF/local source resolver
        |
        +-- Dex3 adapter -----------+
        |                            |
        +-- Inspire diagnostics -----+--> CanonicalEpisode / DiagnosticReport
                                         |
                                         +--> deployment-parity 30->50 Hz resampler
                                         +--> deployment-parity 1247D encoder builder
                                         +--> ONNX Runtime encoder
                                         +--> SONIC VLA target-frame builder
                                         +--> immutable per-episode staging
                                         +--> deterministic final LeRobot merge
```

The resolver owns immutable inputs, metadata, and frame/video access. Source adapters map documented fields into named canonical values but do not resample or encode. The canonical resampler reproduces deployment temporal and quaternion behavior. The encoder builder creates and audits the exact 1,247D tensor. The target builder alone owns output fields and dtypes. Validation runs before final merge; staging and merge own resume and global LeRobot indices.

## Collection eligibility

The Dex3 adapter accepts pinned repositories with Unitree G1 metadata, 30 Hz data, documented 28D state/action arrays containing 14 named arms and 14 named Dex3 joints, a required primary camera, and episodes of at least two frames.

The WBT collection is mixed. The Inspire adapter considers only repository IDs and metadata explicitly identifying Inspire hands. BrainCo, Dex1, video-only, malformed, or otherwise unsupported repositories are reported as skipped with a machine-readable reason; they are never coerced into the Inspire schema.

## Canonical episode contract

An exportable episode contains:

```text
source_fps:              exactly 30
source_frame_count:      N >= 2
observed_root_wxyz:      float64 [N, 4]
reference_root_wxyz:     float64 [N, 4]
observed_body_q:         float64 [N, 29], named G1 joints
desired_body_q:          float64 [N, 29], named G1 joints
observed_left_hand:      float64 [N, 7], source/DDS order
observed_right_hand:     float64 [N, 7], source/DDS order
desired_left_hand:       float64 [N, 7], source/DDS order
desired_right_hand:      float64 [N, 7], source/DDS order
source_task_index:       int64 [N]
primary_video:           exactly N decoded frames
optional_videos:         zero or more streams, each exactly N frames
```

Body joints are stored internally by semantic name. Arrays are materialized into hardware/MuJoCo order, Isaac Lab encoder order, or RobotModel order only at an explicit boundary.

## Dex3 transformation

### Body synthesis

The source observation contributes the 14 observed arm joints and the source action contributes the 14 desired arm joints. Legs and waist use the deployed standing values as absolute radians in hardware/MuJoCo name order:

```text
left_hip_pitch_joint       -0.312
left_hip_roll_joint         0.000
left_hip_yaw_joint          0.000
left_knee_joint             0.669
left_ankle_pitch_joint     -0.363
left_ankle_roll_joint       0.000
right_hip_pitch_joint      -0.312
right_hip_roll_joint        0.000
right_hip_yaw_joint         0.000
right_knee_joint            0.669
right_ankle_pitch_joint    -0.363
right_ankle_roll_joint      0.000
waist_yaw_joint             0.000
waist_roll_joint            0.000
waist_pitch_joint           0.000
```

The complete 29-value source is `default_angles` in `policy_parameters.hpp` at semantic commit `6220f4e210e14c7f804f727da94c8886850d513b`. Mapping is by joint name; array-position coincidence is not accepted. Both observed and reference roots are identity WXYZ for Dex3.

### Hand ordering

The official Unitree source/controller motor IDs are side-specific:

| Side | Source index | Semantic name | DDS motor ID | RobotModel semantic joint |
|---|---:|---|---:|---|
| Left | 0 | thumb0 | 0 | `left_hand_thumb_0_joint` |
| Left | 1 | thumb1 | 1 | `left_hand_thumb_1_joint` |
| Left | 2 | thumb2 | 2 | `left_hand_thumb_2_joint` |
| Left | 3 | middle0 | 3 | `left_hand_middle_0_joint` |
| Left | 4 | middle1 | 4 | `left_hand_middle_1_joint` |
| Left | 5 | index0 | 5 | `left_hand_index_0_joint` |
| Left | 6 | index1 | 6 | `left_hand_index_1_joint` |
| Right | 0 | thumb0 | 0 | `right_hand_thumb_0_joint` |
| Right | 1 | thumb1 | 1 | `right_hand_thumb_1_joint` |
| Right | 2 | thumb2 | 2 | `right_hand_thumb_2_joint` |
| Right | 3 | index0 | 3 | `right_hand_index_0_joint` |
| Right | 4 | index1 | 4 | `right_hand_index_1_joint` |
| Right | 5 | middle0 | 5 | `right_hand_middle_0_joint` |
| Right | 6 | middle1 | 6 | `right_hand_middle_1_joint` |

`teleop.left_hand_joints` and `teleop.right_hand_joints` preserve source indices because these are raw per-side DDS arrays. `observation.state` and `action.wbc` map by semantic name into RobotModel order. A one-hot test verifies all 14 paths. The conflicting symmetric local `DEX3_MOTOR_ORDER` tuple is not an authority for conversion.

RobotModel determines the 43D field order, not a hand-coordinate conversion. This matches the deployed feedback path and `run_data_exporter.py`, which place measured/commanded Dex3 DDS values directly in the RobotModel-ordered hand slots. Consequently, the 29 body slots are checked against the exact combined-model URDF limits, while the 14 hand slots are checked in the pinned dataset's asymmetric DDS domains. The narrower supplemental shoulder-roll range is an operational safety posture constraint, not the representable deployed-feedback domain: the pinned Unitree trajectories legitimately cross it while remaining inside the URDF limits. The exact hand command-domain endpoints are the float32 values represented by the pinned dataset/controller convention:

| DDS semantic | Left `[min, max]` radians | Right `[min, max]` radians |
|---|---:|---:|
| `thumb0` | `[-1.0471975803375244, 1.0471975803375244]` | same |
| `thumb1` | `[-1.0471975803375244, 1.0471975803375244]` | same |
| `thumb2` | `[0, 1.7453292608261108]` | `[-1.7453292608261108, 0]` |
| `index0`, `middle0` | `[-1.832595705986023, 0.19198620319366455]` | `[-0.19198620319366455, 1.832595705986023]` |
| `index1`, `middle1` | `[-2.094395160675049, 0]` | `[0, 2.094395160675049]` |

`action.wbc` hand values must remain inside those command domains with only the numerical joint-limit tolerance. Measured `observation.state` hand values receive `0.01` radians of encoder tolerance. Values are never clipped or renormalized. This distinction is required because the local combined visualization URDF contains symmetric hand limits that contradict both valid right-hand DDS values and the runtime export representation.

## Inspire diagnostic-only gate

The public WBT schema provides a 7D root followed by 29 generic joint columns, but does not authoritatively name those columns or define the controller-level semantics of `robot_q_desired`. Inspire's 6D hand action also has no approved calibration to the 7D Dex3 DDS target.

The initial Inspire implementation may resolve pinned sources, validate five selected episodes, report column statistics and FK/limit plausibility as non-authoritative diagnostics, and emit a machine-readable `blocked_unverified` report.

It must not create a 1,247D encoder tensor, 64D token, 7D Dex3 hand command, `action.wbc` presented as absolute targets, or a training-ready dataset.

The gate opens only when all three revision-pinned inputs exist:

1. An authoritative ordered list of the 29 source joint names.
2. An authoritative declaration that makes `robot_q_desired` valid as SONIC's desired reference trajectory.
3. A calibrated, versioned Inspire-6D-to-Dex3-7D transform for each side, including limits and provenance.

Limits and FK remain secondary validation after the gate opens. There is no `--allow-unverified-inspire` path in this scope.

## Quaternion policy

All external and canonical quaternions are WXYZ Hamilton quaternions. Before interpolation or heading extraction, the adapter:

1. Rejects any non-finite component.
2. Computes the float64 Euclidean norm `n`.
3. Rejects `n < 1e-12`.
4. Rejects when `abs(n - 1.0) > 1e-5`.
5. Otherwise replaces `q` with `q / n` and counts any bitwise component change.

Thus, out-of-tolerance inputs are rejected rather than silently repaired. Accepted inputs are normalized once so conjugation is a valid inverse, matching the deployment precondition.

SLERP reproduces `quat_slerp_d`: negative dot negates the second quaternion; dot greater than `0.9995` uses linear interpolation and unit normalization; otherwise it uses the spherical formula. The result must be finite and within `1e-10` of unit norm. Multiplication results used for heading alignment are normalized before heading or matrix conversion. Golden parity uses normalized fixtures on both implementations and does not compare undefined non-unit deployment behavior.

## Deployment-equivalent timeline and resampling

The source motion is governed by its exact nominal 30 Hz frame grid. Recorded source timestamps are diagnostic:

- they must be finite and strictly increasing;
- `timestamp[i] - timestamp[0]` is compared with `i / 30`;
- maximum absolute grid error above `1 ms` produces a warning;
- error above `1/60 s` rejects the episode;
- timestamp jitter never changes the resampling coordinate.

For `N` source frames:

```text
T50 = floor(N * 50 / 30)
target frame j = 0, ..., T50 - 1
x(j) = j * 30 / 50
f0 = floor(x)
f1 = min(f0 + 1, N - 1)
alpha = x - f0
```

Positions and joint angles use `(1-alpha)*v[f0] + alpha*v[f1]`. Root quaternions use the SLERP policy. The formula intentionally treats source duration as `N/30`, matching deployment rather than `(N-1)/30`.

Desired joint velocities use forward differences:

```text
dq50[j] = 50 * (q50[j+1] - q50[j])  for j < T50 - 1
dq50[T50-1] = dq50[T50-2]
```

The last stored velocity is not forced to zero. Ten-frame encoder windows gather `j+k`, `k=0..9`, clamped to `T50-1`. A clamped window repeats the final stored pose, velocity, and reference orientation. Windows never cross episode boundaries.

## Heading alignment and encoder orientation

Let `H(q)` be the deployment yaw-only quaternion, `q_obs[t]` the normalized observed root, and `q_ref[t]` the normalized desired reference root. Offline conversion fixes `delta_heading=0`:

```text
q_apply = H(q_obs[0]) ⊗ inverse(H(q_ref[0]))
q_aligned[k] = q_apply ⊗ q_ref[k]
q_relative[t,k] = inverse(q_obs[t]) ⊗ q_aligned[min(t+k, T50-1)]
```

Inverse means conjugate after normalization. Multiplication order is Hamilton left-to-right as written.

`q_relative` becomes rotation matrix `R`. The encoder serializes its first two columns row-wise:

```text
[R00, R01, R10, R11, R20, R21]
```

For identity this is `[1,0,0,1,0,0]`. This convention is separate from `teleop.target_body_orientation`, whose existing helper uses contiguous column packing.

## SONIC encoder contract

The encoder receives float32 tensor `obs_dict` with shape `[1,1247]` and produces float32 tensor `encoded_tokens` with shape `[1,64]`.

| Slice | Dimension | Value |
|---|---:|---|
| `[0:4]` | 4 | `encoder_mode_4 = [0,0,0,0]` |
| `[4:294]` | 290 | ten future 29D desired positions in Isaac Lab order |
| `[294:584]` | 290 | ten future 29D desired velocities in Isaac Lab order |
| `[584:644]` | 60 | ten future row-wise 6D relative orientations |
| `[644:1247]` | 603 | exact float32 zeros for inactive modes |

The 29D order is produced by name mapping to `ISAACLAB_G1_JOINT_NAMES`; positional casting is forbidden. The builder rejects a nonzero inactive slice.

ONNX Runtime runs without graph mutation. The manifest records provider and version. CPU execution is the acceptance reference; another provider may be used only after the same golden checks pass.

## Target dataset schema

Dex3 uses `get_features_sonic_vla()` and `get_modality_config_sonic_vla()` with G1 + Dex3 RobotModel. `observation.images.ego_view` is required. Optional wrist cameras are added only when preflight proves them complete for every selected episode in that output repository.

### Derived fields

| Field | Dtype/shape | Contract |
|---|---|---|
| `observation.images.ego_view` | video `[480,640,3]` | required primary camera, resampled to 50 fps |
| `observation.state` | float64 `[43]` | observed 29D body plus 14D hands in RobotModel order |
| `observation.eef_state` | float64 `[14]` | wrist XYZ + WXYZ recomputed by FK from observed 43D state |
| `action.wbc` | float64 `[43]` | absolute desired body plus hands in RobotModel order; auxiliary, not SONIC output |
| `observation.root_orientation` | float64 `[4]` | normalized observed WXYZ root |
| `observation.projected_gravity` | float64 `[3]` | world `[0,0,-1]` rotated by inverse observed root |
| `observation.cpp_rotation_offset` | float64 `[4]` | initial reference root `q_ref[0]`, matching current exporter wiring |
| `observation.init_base_quat` | float64 `[4]` | initial observed root `q_obs[0]` |
| `action.motion_token` | float64 `[64]` | finite float32 ONNX result cast to target storage dtype |
| `teleop.left_hand_joints` | float32 `[7]` | desired left hand in left DDS order |
| `teleop.right_hand_joints` | float32 `[7]` | desired right hand in right DDS order |
| `teleop.smpl_frame_index` | int64 `[1]` | target frame index `j` |

### Exact neutral fields

| Field | Dtype/shape | Exact value |
|---|---|---|
| `teleop.delta_heading` | float64 `[1]` | `[0]` |
| `teleop.smpl_joints` | float32 `[72]` | zeros |
| `teleop.smpl_pose` | float32 `[63]` | zeros |
| `teleop.body_quat_w` | float32 `[4]` | `[1,0,0,0]` |
| `teleop.target_body_orientation` | float32 `[6]` | `[1,0,0,0,1,0]`, existing helper convention |
| `teleop.left_wrist_joints` | float32 `[3]` | zeros |
| `teleop.right_wrist_joints` | float32 `[3]` | zeros |
| `teleop.stream_mode` | int32 `[1]` | `[0]` |
| `teleop.planner_mode` | int32 `[1]` | `[0]` |
| `teleop.planner_movement` | float32 `[3]` | `[0,0,0]` |
| `teleop.planner_facing` | float32 `[3]` | `[1,0,0]` |
| `teleop.planner_speed` | float32 `[1]` | `[-1]` |
| `teleop.planner_height` | float32 `[1]` | `[-1]` |
| `teleop.vr_3pt_position` | float32 `[9]` | zeros |
| `teleop.vr_3pt_orientation` | float32 `[18]` | zeros |

### Dataset indices and timestamps

Every output episode contains exactly `T50` rows and `T50` frames in every included video. Output video metadata is exactly 50 fps.

At target frame `j`, logical timestamp is `j/50`. The target LeRobot timestamp dtype is explicitly float32: the writer stores `np.float32(j / 50.0)`, and validation compares the stored value bitwise with that expression. `frame_index` is int64 `j`; final `episode_index`, global `index`, and `task_index` are int64 values assigned only during deterministic merge.

### Task annotations

Each source frame's `task_index` must resolve through pinned `meta/tasks.parquet` to a non-empty UTF-8 string, preserved verbatim. Target task IDs are assigned by lexicographically sorting unique task strings across validated stages. Frame-level task transitions remain frame-level; episode metadata stores unique tasks in first-occurrence order. Missing, out-of-range, or empty tasks reject the episode.

## Video and camera contract

The smoke repositories use explicit mappings:

| Source cohort | Source camera | Target | Requirement |
|---|---|---|---|
| Dex3 Pouring | `observation.images.cam_left_high` | `observation.images.ego_view` | required |
| Dex3 Pouring | `observation.images.cam_left_wrist` | `observation.images.left_wrist` | optional cohort-wide |
| Dex3 Pouring | `observation.images.cam_right_wrist` | `observation.images.right_wrist` | optional cohort-wide |
| Dex3 Pouring | `observation.images.cam_right_high` | none | intentionally dropped |
| Inspire Pickup Pillow | `observation.images.cam_0` | diagnostic primary camera | required |
| Inspire Pickup Pillow | `observation.images.cam_1` | none | intentionally dropped |

Additional repositories declare mappings in the source lock; camera-name heuristics are forbidden. Collection-level aliases such as `head_stereo_left` or `cam_0` are explicit per-repository entries mapping to `ego_view`.

Preflight decodes the complete episode interval:

- Required primary missing, unreadable, non-RGB, wrong shape, or decoded count different from `N`: reject.
- An optional camera is included only if it exists with exactly `N` valid frames in every selected episode; otherwise omit it from the whole output repository and record why.
- No frame insertion or deletion repairs a mismatch.

At target frame `j`, the source image is:

```text
i(j) = min(N - 1, floor(j * 30 / 50 + 0.5))
```

This is round-half-up with at most `1/60 s` nominal error. Frames enter the writer as RGB HWC uint8.

## Acceptance cohorts

### Deterministic selection

For a pinned repository with `E >= 5` episodes, cohort IDs are:

```text
episode_id[k] = floor(k * (E - 1) / 4 + 0.5),  k = 0,1,2,3,4
```

Applied to the immutable sources:

- Dex3 export/token: `[0,78,155,233,310]`.
- Inspire diagnostic-only: `[0,152,304,456,608]`.

Selection occurs before episode payload download and is stored in the source lock. Missing IDs or a changed pinned episode count is a provenance failure, not a reason to select replacements.

### Dex3 acceptance

All five Dex3 episodes must pass source, quaternion, timeline, camera, joint-name, hand-order, target-schema, and golden parity validation; create exactly `T50` rows and video frames; create finite `[1,1247]` tensors and `[1,64]` tokens at every target frame; and complete deterministic MuJoCo replay.

### Inspire acceptance

All five Inspire episodes must complete diagnostic scanning and end in `blocked_unverified` with the same three named gate reasons. They must not create token, target dataset, or Dex3 hand artifacts. A missing gate reason, encoder invocation, or training-ready output directory fails acceptance.

## Golden and numerical validation

### Per-episode structural validation

Every Dex3 episode is checked for:

- exact field presence, dtype, and shape;
- exact frame and timestamp formulas;
- finite state, action, encoder input, and tokens;
- quaternion policy counters and failures;
- exact inactive zeros and neutral fields;
- body/hand name completeness and uniqueness;
- source and target joint limits without clipping;
- FK regeneration of `observation.eef_state` from observed state;
- video frame count, RGB HWC shape, and 50 fps metadata;
- task resolution and deterministic target task index.

Validation never clips values, replaces NaNs, repairs out-of-tolerance quaternions, or drops a required camera.

### Encoder parity

A fixed canonical fixture is fed through the Python builder and a deployment-side harness. Acceptance requires:

- identical encoder input slices within float32 `atol=1e-7`, `rtol=0`;
- input `obs_dict` `[1,1247]` and output `encoded_tokens` `[1,64]`, both float32;
- CPU ONNX Runtime output versus TensorRT FP32 within `atol=1e-5`, `rtol=1e-5`;
- a checked-in golden input/token vector tied to the artifact hashes;
- a deliberate `+0.5 rad` multi-frame shoulder perturbation changes at least one token dimension.

Small perturbations need not change the token because quantization can map them to the same code.

### Replay protocol

All five Dex3 episodes are replayed. Inspire episodes are not replayed while gated.

Each episode runs in a fresh OS process. The Unitree SDK channel factory is process-global and cannot be reliably reinitialized for multiple simulator lifetimes in one Python process; cohort orchestration therefore runs one authenticated single-episode child at a time, validates its report identity and immutable execution provenance, and deterministically merges the five child reports. A child must echo the exact current 64D token and both exact 7D measured-hand vectors with a monotonically increasing controller index. The acknowledgement deadline is one complete 50 Hz control period (`0.02 s`), which accommodates asynchronous telemetry without allowing the next command to be published first.

For each episode:

1. Initialize deterministic headless MuJoCo with seed `0`, nominal standing state, and zero velocity.
2. Hold the first command for `2.0 s` warm-up.
3. Stream all `T50` token/hand frames at exactly 50 Hz.
4. Hold the final command for `1.0 s`.
5. Exclude only the first `1.0 s` of warm-up from acceptance metrics; include the remaining warm-up, full trajectory, and terminal hold.

Per-step aggregation is:

```text
torque_ratio(t) = max_j(abs(tau_j(t)) / effort_limit_j)
contact_force(t) = max_c(norm(contact_force_vector_c(t), 2))
joint_limit_overshoot(t) = max_j(max(lo_j-q_j, q_j-hi_j, 0))
```

Each episode records time-series maximum and p50/p95/p99. Gates use the maximum after the exclusion interval. Cohort aggregation is the maximum of the five episode maxima; averaging cannot hide a failing episode.

Numerical gates are:

- no simulator/controller fault, NaN, or Inf;
- root height within `[0.45,1.05] m`;
- absolute root roll and pitch each at most `0.7 rad`;
- joint-limit overshoot at most `0.02 rad`;
- absolute joint velocity at most `50 rad/s`;
- torque ratio at most `1.05`;
- contact-force maximum at most `10 * m_robot * 9.81 N`.

The report includes robot mass, effort-limit source, simulator/control timestep, included metric interval, and raw per-episode maxima.

## Error handling

Errors are classified and fail closed:

- `provenance_error`: floating/missing revision, hash mismatch, changed episode count.
- `source_schema_error`: missing or ambiguous field, task, joint name, or camera.
- `timeline_error`: too-short episode, invalid timestamp, or video mismatch.
- `quaternion_error`: non-finite, degenerate, or out-of-tolerance quaternion.
- `semantic_gate_error`: Inspire mapping, desired-signal meaning, or hand calibration missing.
- `encoder_contract_error`: model/config/tensor mismatch or non-finite output.
- `target_validation_error`: output field, FK, limit, index, or video failure.
- `replay_acceptance_error`: replay threshold or controller failure.

One bad episode does not corrupt validated stages. A production run may continue collecting error reports, but final publication reports incomplete coverage and never presents skipped failures as success.

## Staging, resume, and deterministic merge

Each source episode has immutable staging rooted by source and conversion identity:

```text
<output>/.staging/
  <repo-id-safe>/
    <source-revision>/
      episode-<source-id>/
        <conversion-config-sha256>/
          frame-data.parquet
          videos/...
          validation.json
          manifest.json
          checksums.sha256
```

Files are written under a unique temporary sibling, flushed, closed, validated, checksummed, and atomically renamed. Resume skips a stage only when source hashes, config, converter version, model/config hashes, source-lock hash, artifact checksums, and success status all match.

Final merge sorts by `(source_repository_id, source_episode_id)`, then rebuilds episode indices, global frame indices, task indices, videos, metadata, and statistics. It writes a new temporary dataset and atomically renames only after complete validation; it never appends to a partial merged dataset.

Inspire diagnostic reports use the same provenance identity under a distinct `diagnostic-only` root and cannot enter the target merge.

## Testing strategy

Implementation follows test-driven increments:

1. Immutable URI/revision/hash enforcement.
2. Dex3 semantic body and one-hot hand mapping.
3. Inspire fail-closed diagnostics and absence of export artifacts.
4. Quaternion validation and exact C++-style SLERP vectors.
5. Frame count, interpolation, terminal velocity, clamp, and image rounding.
6. Exact 1,247D slices and row-wise rotation packing.
7. ONNX contract, golden vector, and large perturbation.
8. Target fields, FK, tasks, and camera timeline.
9. Immutable staging, checksum resume, corruption detection, and deterministic merge.
10. Fixed five-plus-five cohorts and five Dex3 numerical replays.

Network-free unit tests use synthetic LeRobot fixtures and fake resolvers. Networked smoke tests are marked and always use the pinned revisions.

## Deliverables

- Revision-locked source manifest format and smoke lock.
- Dex3 adapter and canonical episode model.
- Inspire diagnostic adapter with three explicit gates.
- Deployment-parity 30-to-50 Hz resampler and root-orientation builder.
- Low-latency artifact loader and 1,247D encoder wrapper.
- SONIC VLA frame/staging/merge writer.
- Structural, golden, and replay validators.
- Conversion/diagnostic CLI with `--smoke` selecting only fixed cohorts.
- Unit/integration tests and operator guide.

## Approval boundary

This specification authorizes Dex3 export/token implementation and Inspire diagnostic-only implementation. It does not authorize Inspire token generation without an evidence-backed revision supplying all three semantic gate inputs.
