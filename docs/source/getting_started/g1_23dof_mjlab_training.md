# Exact G1 true23 training with MJLab

This path fine-tunes the exact SONIC G1 rev-1.0 23-DoF policy in MJLab. It is
intended for a CUDA-capable Ubuntu or WSL2 workstation with limited VRAM and
does not require Isaac Lab.

Upstream projects:

- [Unitree RL MJLab](https://github.com/unitreerobotics/unitree_rl_mjlab)
- [MJLab](https://github.com/mujocolab/mjlab)

> **Simulator-only safety boundary**
>
> These commands do not connect to Pico, XRoboToolkit, DDS, or robot hardware.
> They do not send motor commands. Every checkpoint produced here is marked
> `deployment_ready=false` and `promotion_eligible=false`. The included dance
> sample checks conversion, environment, PPO, checkpoint, and resume mechanics
> only. It is not an approved training dataset, a trained balance policy, or
> evidence that a robot is safe to actuate. No gantry or supported physical
> test is performed by this workflow.

## Exact 23-DoF contract

This is native 23-output training, not a 29-output policy with six outputs
discarded at runtime.

- Live Unitree `mode_machine == 4` motor indices are
  `0-12, 15-19, 22-26`.
- Removed and locked motor indices are `13, 14, 20, 21, 27, 28`.
- The policy decoder emits exactly 23 actions in native rev-1.0
  PhysX/IsaacLab breadth-first order. The MJLab action term performs the
  explicit conversion to hardware/MuJoCo order and applies the 23-joint
  defaults and scales.
- The ten-frame proprioception history retains the released canonical
  IsaacLab-29 observation layout. Missing canonical slots are
  `[5, 8, 25, 26, 27, 28]`.
- Missing joint positions are held at their fixed default positions, so their
  `joint_pos_rel` values are zero. Missing joint velocity and previous-action
  values are zero in every history frame.
- The environment observation groups are a 268-value tokenizer group (one
  encoder route plus the exact 267-value teleoperation input), a 930-value
  term-major proprioception group, and a separate 256-value critic group.
- The actor trains the exact `267 -> FSQ64 -> [FSQ64, H10-930] -> 23`
  encoder/decoder topology.

The custom runner rejects stock RSL-RL JIT or ONNX export. A stock MJLab G1
checkpoint cannot be renamed or converted after training into a SONIC true23
checkpoint. See the
[checkpoint boundary](g1_23dof_mjlab_checkpoint_bridge.md) for details.

## Pinned runtime

The installer creates an isolated Python 3.11 virtual environment and verifies
the following runtime:

- Unitree RL MJLab commit
  `1425b15f73bd4095f0df53709d7c389c3eb9e790`
- MJLab 1.2.0 commit
  `5af32e378dcb93c9e881ace83cc5a3f5d373fe60`
- MuJoCo-Warp 3.5.0.2 commit
  `5a86ec28aa07741eb2e000d158f4ca4068ec146e`
- Python 3.11, PyTorch 2.9.0+cu128, TorchVision 0.24.0+cu128
- NumPy 2.3.4, Warp 1.12.0, and RSL-RL 5.0.1
- MuJoCo 3.5.0, SciPy 1.16.2, and TensorDict 0.10.0

MJLab 1.2's upstream lock references a development MuJoCo wheel that is no
longer available from its package index. This environment therefore declares
stable `mujoco==3.5.0` as a compatibility substitution while retaining the
exact MuJoCo-Warp source commit. Unitree's package metadata asks for
`mujoco-warp==3.5.0`; the pinned MJLab commit identifies that same source line
as 3.5.0.2. Both source checkouts are installed with `--no-deps`, and the
launcher records the exact commits and substitution in every immutable
lineage.

## Install in WSL2

Prerequisites:

- WSL2 with Ubuntu and a current Windows NVIDIA driver
- `nvidia-smi` working inside WSL
- `git` and [`uv`](https://docs.astral.sh/uv/) available inside WSL
- the released exact warm start at
  `sonic_release/g1_23dof_rev_1_0_init.pt`

From the repository root inside WSL:

```bash
nvidia-smi
bash install_scripts/setup_g1_23dof_mjlab_wsl.sh
```

The script creates:

```text
$HOME/.venvs/g1_true23_mjlab
external_dependencies/mjlab
external_dependencies/unitree_rl_mjlab
```

It refuses a source checkout at the wrong commit and never resets an existing
checkout.

For the remaining commands:

```bash
PY="$HOME/.venvs/g1_true23_mjlab/bin/python"
LAUNCHER="gear_sonic/scripts/train_g1_23dof_mjlab.py"
```

## Convert the included mechanics sample

Convert Unitree's included G1-23 CSV to the required 50 Hz NPZ:

```bash
"$PY" "$LAUNCHER" convert-sample
```

Default output:

```text
$HOME/.cache/g1_true23_mjlab/dance1_subject2_23dof.npz
```

Use `--overwrite` only when intentionally replacing that derived file. For
real training, pass an approved 50 Hz true23 motion NPZ with
`--motion-file`. The preflight requires finite `joint_pos` and `joint_vel`
arrays shaped `[frames, 23]`, matching full-body arrays, and at least the
47-frame minimum motion length. Do not treat the included sample as
production data.

## Run preflight

```bash
"$PY" "$LAUNCHER" preflight \
  --json-output "$HOME/g1_true23_runs/preflight.json"
```

Preflight fails closed if CUDA, a package version, a source commit, the warm
start, or the motion schema differs from the pinned contract. Read
`ready: true` in the JSON report before starting a smoke run.

For approved data:

```bash
"$PY" "$LAUNCHER" preflight \
  --motion-file /absolute/path/to/approved_true23_motion.npz \
  --json-output "$HOME/g1_true23_runs/approved_preflight.json"
```

## Run a real CUDA smoke test

```bash
SMOKE_RUN="$HOME/g1_true23_runs/true23_smoke"

"$PY" "$LAUNCHER" smoke \
  --num-envs 4 \
  --iterations 2 \
  --save-interval 1 \
  --run-dir "$SMOKE_RUN"
```

Smoke mode performs real CUDA rollout and PPO updates with reduced work:
8 steps per environment, 2 learning epochs, and 2 minibatches. It still uses
the exact actor, native 23-action boundary, disturbances, and domain
randomization.

MJLab's explicit reset initially exposes a stale body target. After the RSL
wrapper performs its final reset and before any PPO rollout, the launcher
refreshes that target directly without advancing physics, command time,
curriculum, or RNG state. Rare randomized terminal starts are rejected as a
full batch, observation histories are rebuilt from one coherent frame, and
the audit is written to `environment_prime.json`. Smoke mode also keeps
initial episode lengths at zero so near-timeouts cannot muddy this diagnostic.
The runner then saves
`checkpoints/model_0.pt`: post-prime policy/trainer state with zero completed
PPO updates. No policy learning occurs before `model_0.pt`. Later numbered
checkpoints represent completed outer PPO updates.

Inspect a checkpoint without arbitrary pickle globals:

```bash
"$PY" -m gear_sonic.scripts.inspect_g1_23dof_mjlab_training \
  --checkpoint "$SMOKE_RUN/checkpoints/model_2.pt" \
  --minimum-update-count 2 \
  --json
```

The expected result still says `deployment_ready=false` and
`promotion_eligible=false`.

RSL's printed `Episode_Termination/*` values are mean environment counts for
steps that produced resets, not percentages or termination rates. Early
stochastic falls are expected from an initialization that has not learned
true23 balance. Judge training health over a larger rollout and never promote
the included two-update smoke checkpoint.

## Start low-VRAM production training

Production defaults target an 8 GB GPU:

- 128 environments
- 16 steps per environment
- 5 PPO epochs
- 8 minibatches
- 10,001 planned updates
- headless EGL rendering and one CUDA device

Use approved motion data, not the included sample:

```bash
TRAIN_RUN="$HOME/g1_true23_runs/approved_true23_v1"
MOTION="/absolute/path/to/approved_true23_motion.npz"

"$PY" "$LAUNCHER" train \
  --motion-file "$MOTION" \
  --num-envs 128 \
  --iterations 10001 \
  --save-interval 10 \
  --run-dir "$TRAIN_RUN"
```

If 128 environments cause CUDA out-of-memory on the local 8 GB GPU, begin a
new run with 64 environments, then 32 if needed. Environment count is part of
immutable lineage; do not change it while resuming an existing run.

`train` refuses fewer than 50 planned updates. Reaching 50 updates only permits
simulation-candidate review when policy weights also changed from
initialization. It does not make a checkpoint deployable.

## Train in chunks and resume

`--iterations` is the immutable total plan. `--session-updates` limits only
the current invocation. This allows overnight-sized chunks without changing
lineage:

```bash
TRAIN_RUN="$HOME/g1_true23_runs/approved_true23_v1"
MOTION="/absolute/path/to/approved_true23_motion.npz"

"$PY" "$LAUNCHER" train \
  --motion-file "$MOTION" \
  --num-envs 128 \
  --iterations 10001 \
  --session-updates 250 \
  --save-interval 10 \
  --run-dir "$TRAIN_RUN"
```

Resume from the final checkpoint written by that chunk:

```bash
"$PY" "$LAUNCHER" train \
  --motion-file "$MOTION" \
  --num-envs 128 \
  --iterations 10001 \
  --session-updates 250 \
  --save-interval 10 \
  --run-dir "$TRAIN_RUN" \
  --resume "$TRAIN_RUN/checkpoints/model_250.pt"
```

Keep motion file, warm start, source trees, robot assets, environment count,
seed, reference profile, save interval, and planned update count unchanged.
Strict resume restores actor, critic, optimizer, adaptive learning rate,
trainer counters, and environment step counter. Any lineage mismatch fails
instead of partially loading state.

Each run directory contains:

- `resolved_training.json`: exact task, runner, package, and source settings
- `lineage.json`: hashes for warm start, source, assets, and motion data
- `environment_prime.json`: reset-target prime audit
- `checkpoints/model_<completed_updates>.pt`: weights-only-safe full resume
  state

## What remains before robot testing

Successful preflight, smoke, resume, and long training mean only that the
training pipeline and simulator checkpoint mechanics work. Before any
sim-to-real claim, a changed policy still needs reviewed data provenance,
ONNX pair parity, deterministic MuJoCo replay for nominal and disturbance
scenarios, dry-run output checks, live shadow operation, and supported
first-actuation testing with `mode_machine == 4` gating.

Without a gantry or other rated restraint and an approved physical test plan,
stop at simulator and live-shadow evidence. Do not enable motor output.
