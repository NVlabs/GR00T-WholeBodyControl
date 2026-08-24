# G1 true-23 simulation promotion

`g1_23dof_rev_1_0.json` is the hash-bound promotion envelope for the true
23-DoF teleop policy. The checked-in IsaacLab producer is
`gear_sonic/scripts/run_g1_23dof_sim_validation.py`. It fixes:

- 50 Hz and 250 steps per episode;
- three deterministic seeds and 22 parallel episodes per seed;
- nominal, 50%, and 100% disturbance scenarios;
- raw per-step/per-environment traces;
- checkpoint, runner, asset, validation-config, and resolved-Hydra-config
  hashes.

The trace validator reopens every trace and recomputes all promotion metrics.
Arbitrary evaluation overrides are not accepted.

Promotion remains intentionally disabled because no trained true-23 checkpoint
has completed this real IsaacLab run:

- `producer.promotion_enabled` is `false`;
- the runner SHA-256 is pinned, but its reports cannot yet authorize export.

## Approved MuJoCo alternative

`g1_23dof_mujoco_sim2sim.json` defines the independent MuJoCo campaign.
`g1_23dof_mujoco_sim2sim_approval.json` pins MuJoCo 3.2.3, the runner,
replay validator, exact rev-1.0 MJCF/URDF, and the complete Unitree asset
manifest. This path does not require IsaacLab.

The workflow is deliberately split:

1. `export_g1_23dof_mujoco_candidate.py` accepts only a genuinely trained
   weights-only `*.promotion.pt` and emits immutable ONNX candidate bytes
   marked non-deployable.
2. `run_g1_23dof_mujoco_sim2sim.py` runs 198 five-second native-23 episodes
   and writes nine raw JSONL traces.
3. `promote_g1_23dof_mujoco_candidate.py` reruns the exact ONNX pair and
   MuJoCo dynamics, compares all records, and emits a self-hashed promotion
   sidecar only when every threshold passes.

The sidecar authorizes the exact ONNX bytes for native read-only shadow
testing. It never authorizes active motors. An initialization checkpoint
cannot enter this workflow.

Dry-run the trained checkpoint outputs before any simulator starts. This
reconstructs the exact 267-input encoder and 994-input decoder on CPU, runs a
finite chained forward pass, and requires exactly 23 float32 outputs. The
normal validation command repeats this output check before launching IsaacLab.
Then run or validate the disturbance campaign:

```bash
python -m gear_sonic.scripts.run_g1_23dof_sim_validation \
  --checkpoint /path/to/trained_true23.promotion.pt \
  --output /path/to/sim-report.json \
  --dry-run

python -m gear_sonic.scripts.run_g1_23dof_sim_validation \
  --checkpoint /path/to/trained_true23.promotion.pt \
  --output /path/to/sim-report.json

python -m gear_sonic.scripts.run_g1_23dof_sim_validation \
  --checkpoint /path/to/trained_true23.promotion.pt \
  --output /path/to/sim-report.json \
  --validate-only
```

The trainer writes a trusted full-resume checkpoint and a separate
weights-only `*.promotion.pt` sidecar. Simulation, readiness, and export accept
only the promotion sidecar; never pass the full trainer checkpoint to them.

Passing offline promotion is still not permission for free-standing hardware
operation. The native `g1_true23_shadow_gate` must first validate the paired
ONNX artifact and five advancing CRC-valid `mode_machine == 4` LowState
samples. Gantry testing remains a separate required stage.
