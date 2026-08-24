# True23 XR24 causal curriculum boundary

Current released policies cannot consume honest live measured XR24 motion under
the 60 ms source-freshness gate. This is a temporal-contract mismatch, not an
IP, coordinate, or tensor-shape problem.

## Hard latency proof

Both released profiles encode ten **future** lower-body position frames and ten
matching forward-difference velocity frames. A measured producer cannot know
those frames before they happen.

| Released profile | Selected future horizon | Minimum measured delay | Current checked adapter delay | 60 ms feasible |
| --- | ---: | ---: | ---: | --- |
| `true23_step5_0p1s` | 900 ms | 920 ms | 940 ms | No |
| `released_low_latency_step1_0p02s` | 180 ms | 200 ms | 220 ms | No |

`step1` means 20 ms spacing between frames. It does not mean 20 ms end-to-end
latency. The tenth selected frame is 180 ms after the first, and one additional
real 20 ms sample is required to prove its forward velocity. Current bridge
validation deliberately retains one more complete semantic proof frame.

Local `low_latency/last.pt` was hash-verified as the approved release
`0031ae7db24747445d6eb7c27697640973a837546f0b8763e775143c47d4507c`.
Its source config explicitly records `target_fps: 50`,
`dt_future_ref_frames: 0.02`, and `num_future_frames: 10`.

Using a current timestamp on the oldest reference, repeating a last pose,
zero-filling unavailable joints, or calling measured past frames "future" would
hide this delay and violate training semantics. Those substitutions remain
forbidden.

## Exact rolling SOMA result

The rolling adapter preserves the pinned SOMA Newton configuration:

- 24 main IK iterations;
- 20 feet-stabilizer IK iterations;
- original objective weights, initialization sequence, CUDA graphs, joint-limit
  clamper, and solver continuity;
- exact bracketed 50 Hz XR24 timestamps with no synthetic future sample.

On the 49-frame CUDA fixture, rolling root7+MJ29 output and synchronized
locked-True23 body terms were bit-for-bit equal to the pinned batch result
(`max_abs_error = 0` for both). On the deployment RTX 3070 Laptop GPU, while
the MJLab training job was also active, steady rolling timing was:

- mean 26.67 ms (37.50 fps);
- p50 20.83 ms;
- p95 45.00 ms;
- p99 100.28 ms;
- maximum 143.19 ms;
- target construction mean 2.76 ms;
- Newton solve/post-processing mean 23.50 ms;
- synchronized locked-True23 body FK mean 0.42 ms.

This run proves current shared-machine deadline failure. It is not an isolated
hardware ceiling measurement. Even a future isolated run above 50 fps would
not repair the 200/920 ms released-policy temporal mismatch.

Reproduce read-only benchmark:

```bash
cd /mnt/z/codex/GR00T-WholeBodyControl
PYTHONPATH=$PWD /root/.venvs/g1_true23_soma/bin/python \
  gear_sonic/scripts/benchmark_g1_23dof_xr24_soma_stream.py \
  --capture /mnt/z/codex/g1_true23_runs/xr24_soma_synthetic_capture_20260803.json \
  --output /mnt/z/codex/g1_true23_runs/xr24_soma_rolling_benchmark.json
```

Add `--require-deadline` when a nonzero exit is wanted for CI deadline failure.
The benchmark opens no XR, ZMQ, DDS, ADB, or robot channel.

## Smallest safe replacement

Create a new profile, not an alias of either released profile:

`true23_causal_step1_history_0p02s_v1`

Keep 267 encoder values, but change and hash-bind their temporal meaning:

1. Retain eleven contiguous IL29 position samples `q0..q10`.
2. Build ten exact semantic frames `q0..q9`, with
   `dq[i] = (q[i+1] - q[i]) / 0.02`.
3. Treat `q9` as anchor. Tensor offsets, oldest to anchor, are
   `[-0.18, -0.16, -0.14, -0.12, -0.10, -0.08, -0.06, -0.04, -0.02, 0.0]`.
4. Keep `q10` only as measured velocity proof. Never put it in the ten-frame
   tensor as a predicted future.
5. Derive wrist/torso three-point terms and reference pelvis orientation from
   the same `q9` SOMA/FK result.
6. Interpolate robot pelvis/IMU orientation from a timestamped LowState ring
   buffer at the same `q9` anchor time. Current robot orientation is not an
   equivalent substitute.

Steady intrinsic anchor age becomes at least 20 ms instead of 200/920 ms:

```text
anchor age = 20 ms velocity proof
           + raw-frame bracket lag
           + retarget compute
           + IPC
           + ONNX inference
```

Every component must remain within the 60 ms gate at p99/max policy chosen for
deployment. A practical retarget target is below 12-15 ms p99, leaving budget
for capture jitter, IPC, and inference. Current pinned SOMA result does not meet
that target.

## Required training and promotion work

- Generate causal-history observations in MJLab with all 29 semantic reference
  slots present. Keep True23's six excluded physical joints locked in the
  embodiment only; do not zero-fill semantic reference slots.
- Fine-tune from the released low-latency checkpoint only as initialization.
  Changed temporal semantics require a new lineage/profile ID and full
  retraining evidence before deployment.
- Train balance and disturbance recovery using synchronized causal commands,
  randomized capture/retarget latency, and dropped-frame termination rather
  than pose repetition.
- Export a new ONNX pair whose metadata binds the causal profile, offsets,
  joint orders, teacher/runtime hashes, and `mode_machine == 4` gate.
- Run MuJoCo disturbances, exact trace replay, deadline tests, dry-run, then
  gantry testing. Existing promotion artifacts cannot authorize this new
  profile.
- Version the live wire with separate semantic-anchor, robot-anchor,
  raw-capture, producer-finish, and inference timestamps. Validate each age and
  cross-source skew independently.

For runtime, either optimize the same pinned math until rolling output remains
batch-equivalent and meets the deadline, or use pinned SOMA as an offline
teacher for a new causal retargeter. A learned/analytic replacement needs its
own accuracy, limits, disturbance, timing, and gantry evidence; it cannot claim
the pinned-SOMA exact-backend status.

Until those steps pass, current released checkpoint plus measured XR24 stream
must remain non-promotable.
