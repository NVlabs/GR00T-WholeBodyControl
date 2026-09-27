# Convert Unitree UniFoLM Data to SONIC

This workflow converts the revision-locked Unitree G1 + Dex3 teleoperation source into the repository's 50 Hz SONIC VLA schema. Each target row contains a 64D low-latency SONIC token and the two raw 7D Dex3 hand commands. The converter also audits G1 + Inspire data, but Inspire remains diagnostic-only and is not training-ready.

```{important}
Only Dex3 conversion is enabled. Inspire token generation is fail-closed because the source still lacks an authoritative 29-joint order, confirmed `robot_q_desired` semantics, and an approved Inspire-to-Dex3 hand calibration. There is no override flag.
```

## Prerequisites

Run commands from the repository root. Install the data-collection environment and download/build the normal low-latency deployment artifacts before replay:

```sh
bash install_scripts/install_data_collection.sh
```

The examples below use `.venv_data_collection` for conversion and `.venv_teleop` for MuJoCo replay. The checked-in smoke lock is:

```text
gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml
```

## Immutable smoke inputs

The smoke lock pins every input by immutable revision, byte size, and SHA-256 where applicable.

| Input | Immutable location | Selection or digest |
|---|---|---|
| SONIC encoder | `hf://models/nvidia/GEAR-SONIC@9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2/low_latency/model_encoder.onnx` | 45,933,505 bytes; `60be43157f57d812f38bdbb740a5de5d5d070e8840d9edc16f02a91a6d06255b` |
| Encoder config | `hf://models/nvidia/GEAR-SONIC@9c0ff22b4ffec27c5392e8e284eb2f2df7a5b4e2/low_latency/observation_config.yaml` | 3,258 bytes; `582b9a273a3d69fbf49ae59b39295a3be2b4a295e195ef4cf674b5e2571c90ab` |
| Dex3 source | `hf://datasets/unitreerobotics/G1_Dex3_Pouring_Dataset@c9552eb3b1cb610cd6227555e9e98b1bde826a77/` | episodes `0, 78, 155, 233, 310` |
| Inspire source | `hf://datasets/unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly@24e3e4d88a5020bdb4b3046ec09b09dc56f8d1f1/G1_WB_Dex5_Pickup_Pillow` | episodes `0, 152, 304, 456, 608` |
| Conversion semantics | repository commit `6220f4e210e14c7f804f727da94c8886850d513b` | stored as `semantic_repo_commit` |

The five episode IDs are deterministic, stratified selections over the pinned episode counts. A missing episode or changed count is a provenance failure; the converter does not select a replacement.

## Cache and output layout

Choose separate cache and output roots:

```sh
export UNITREE_SONIC_CACHE=outputs/unitree_sonic_cache
export UNITREE_SONIC_OUTPUT=outputs/unitree_sonic_conversion
```

The cache holds revision-scoped Hugging Face artifacts. The output root contains resumable immutable stages and final reports:

```text
outputs/unitree_sonic_conversion/
├── .staging/                                  # immutable per-episode Dex3 stages
├── pipeline-report.json                       # Dex3 pipeline result
├── diagnostic-only/
│   ├── pipeline-report.json                   # Inspire cohort result
│   └── unitreerobotics--G1_WBT_.../           # per-episode diagnostics
├── unitreerobotics--G1_Dex3_Pouring_Dataset/
│   ├── data/ and videos/                      # merged LeRobot dataset
│   ├── meta/
│   ├── source-manifest.json
│   ├── dataset-checksums.sha256
│   └── merge-validation.json
└── unitree-sonic-replay-report.json           # five-episode MuJoCo report
```

Do not edit a stage or merged dataset in place. Resume authenticates staged checksums and rebuilds the deterministic merge only from matching immutable identities.

## Run Inspire diagnostics

Run the diagnostic path before relying on any Inspire source:

```sh
.venv_data_collection/bin/python \
  gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py \
  --source-lock gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml \
  --output-root "$UNITREE_SONIC_OUTPUT" \
  --cache-dir "$UNITREE_SONIC_CACHE" \
  --kind inspire \
  --smoke
```

Success for this command means all five episodes end as `blocked_unverified`, the encoder invocation count is zero, and no target dataset path is emitted. Each diagnostic must contain exactly these gates:

1. `missing_authoritative_29_joint_order`
2. `missing_robot_q_desired_semantics`
3. `missing_inspire_to_dex3_calibration`

Inspect `outputs/unitree_sonic_conversion/diagnostic-only/pipeline-report.json`. Any missing gate, encoder invocation, or training-ready Inspire output is a failure.

## Convert Dex3

```sh
.venv_data_collection/bin/python \
  gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py \
  --source-lock gear_sonic/data/unitree_conversion/manifests/smoke_sources.yaml \
  --output-root "$UNITREE_SONIC_OUTPUT" \
  --cache-dir "$UNITREE_SONIC_CACHE" \
  --kind dex3 \
  --smoke \
  --resume \
  --workers 1
```

The command succeeds only if all five selected episodes validate and merge. Check `pipeline-report.json` for `validated_episode_count: 5`, `failed_episode_count: 0`, and one target dataset path. The target's `source-manifest.json`, `dataset-checksums.sha256`, and `merge-validation.json` bind the source bytes, lock, encoder/config artifacts, target schema, and merged episode order.

The main target action fields are:

- `action.motion_token`: float32 `[64]` SONIC token generated from nominal standing legs/waist plus the source arm trajectory.
- `teleop.left_hand_joints` and `teleop.right_hand_joints`: float32 `[7]` raw, side-specific Dex3 DDS commands.
- `action.wbc`: float32 `[43]` absolute 29-body + 14-hand target in RobotModel field order.

The converter does not clip joint values, repair quaternions, cross episode boundaries, or substitute cameras.

### Resume check

Run the same Dex3 command again with `--resume`. A clean resume reports `reused: 5`, performs zero encoder invocations, and leaves both `source-manifest.json` and `dataset-checksums.sha256` byte-identical. A checksum or identity mismatch is rejected instead of reused.

## Replay the fixed Dex3 cohort

Replay requires the built C++ low-latency deployment, the decoder/planner models, MuJoCo dependencies, and a DDS-capable local interface. Replace `docker0` if your simulation setup uses a different interface.

```sh
PYTHONPATH=. .venv_teleop/bin/python \
  gear_sonic/scripts/replay_unitree_sonic_smoke.py \
  --dataset-root "$UNITREE_SONIC_OUTPUT" \
  --source-episode-ids 0 78 155 233 310 \
  --network-interface docker0
```

Plural replay intentionally launches one fresh Python/DDS/MuJoCo child process per episode, validates each child's authenticated report, and deterministically merges the results. This avoids reusing the process-global Unitree channel factory. Every 50 Hz command must receive an exact token and hand echo within one 20 ms control period before the next command is published.

Acceptance requires all five reports and no gate failures. The numerical gates include root height `[0.45, 1.05]` m, roll/pitch at most `0.7` rad, joint-limit overshoot at most `0.02` rad, joint velocity at most `50` rad/s, torque ratio at most `1.05`, contact force at most `10 * robot_mass * 9.81`, and no controller fault or non-finite sample.

## Discover and review a full-collection lock

The discovery command resolves the ordered members of both Unitree collections to immutable repository SHAs:

```sh
PYTHONPATH=. .venv_data_collection/bin/python \
  gear_sonic/scripts/lock_unitree_unifolm_sources.py \
  --output-lock outputs/unitree_full_review.yaml
```

Discovery writes a `scope: full` review draft with every source set to `approved: false`. It is not executable as generated. Review each repository and explicitly add or confirm:

- `approved: true` only for a source whose schema has been audited;
- exact `dataset_path`, `episode_count`, and a strictly increasing list of eligible `episodes`;
- the required `primary_camera` and exact `camera_map`;
- any source label needed by local review tooling.

Keep the recorded collection membership and immutable revision unchanged unless conducting a new review. Use `--force` only to atomically replace that exact draft path. Run an approved full lock with `--no-smoke`; a smoke lock cannot be widened with `--no-smoke`, and any unapproved or episode-less source fails closed.

```sh
.venv_data_collection/bin/python \
  gear_sonic/scripts/convert_unitree_unifolm_to_sonic.py \
  --source-lock outputs/unitree_full_reviewed.yaml \
  --output-root outputs/unitree_sonic_full \
  --kind dex3 \
  --no-smoke \
  --resume
```

Inspire sources remain diagnostic-only even in a reviewed full lock.

## Verification evidence

The implementation's requirement-to-evidence map is:

| Requirement | Automated evidence | Runtime artifact |
|---|---|---|
| Immutable source/model provenance | `test_provenance.py`, `test_pipeline.py` | `source-manifest.json`, `dataset-checksums.sha256` |
| Dex3 body and side-specific hand mappings | `test_joint_mapping.py`, `test_dex3_adapter.py`, `test_staging.py` | `merge-validation.json` |
| Inspire semantic gates and zero encoder use | `test_inspire_diagnostics.py`, `test_pipeline.py`, `test_smoke_network.py` | `diagnostic-only/pipeline-report.json` and per-episode diagnostics |
| Quaternion normalization/rejection policy | `test_quaternion.py` | per-episode validation reports |
| Deployment-equivalent 30-to-50 Hz resampling | `test_resampling.py` | target timestamps and episode lengths |
| Heading alignment and row-wise 6D rotation packing | `test_sonic_encoder.py`, `test_encoder_parity_tools.py` | encoder input parity report/golden fixture |
| Exact `[1,1247]` input and `[1,64]` token contract | `test_sonic_encoder.py`, C++ encoder parity GTest | pinned encoder/config hashes and target tokens |
| Target fields, dtypes, limits, FK, and neutral values | `test_target_frames.py`, `test_staging.py` | `merge-validation.json` |
| Camera mapping, frame count, 50 Hz video, and tasks | `test_video_timeline.py`, `test_pipeline.py` | target `meta/`, videos, and pipeline report |
| Fixed five-plus-five cohorts | `test_provenance.py`, `test_smoke_network.py` | Dex3 and Inspire pipeline reports |
| Golden ONNX/TensorRT parity and semantic perturbation | `test_encoder_parity_tools.py`, `test_sonic_encoder.py` | checked-in golden arrays tied to artifact hashes |
| Immutable staging, deterministic merge, and resume | `test_staging.py`, `test_pipeline.py`, `test_smoke_network.py` | stage manifests/checksums and merged manifest |
| Continuous acknowledgements and replay thresholds | `test_replay.py` | `unitree-sonic-replay-report.json` |

Publication remains blocked if any row in this matrix lacks passing evidence or any Dex3 pipeline/replay report is not accepted.
