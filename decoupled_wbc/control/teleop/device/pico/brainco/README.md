# Brainco Hand Integration for Gear-Sonic

PICO headset hand-tracking → dex_retargeting → brainco RevoLimb hands over Unitree DDS, integrated into Gear-Sonic's existing teleop stack.

This document records what was added/changed and why, derives the math behind the coordinate transform so it can be audited, and describes how the pipeline interacts with a real PICO + brainco rig.

---

## 1. Goal

Gear-Sonic already drives a Unitree G1 arm with PICO controllers via xrobotoolkit. We want the same headset session to also command brainco RevoLimb dexterous hands using **real** finger tracking from `XR_EXT_hand_tracking` — not synthetic button-encoded fingertip transforms.

Source-of-truth for the original brainco pipeline is xr_teleoperate, which uses Vuer + televuer for hand tracking. We port that retargeting + DDS publishing path into Gear-Sonic and feed it from xrobotoolkit instead of Vuer.

---

## 2. Data flow

```
┌────────────────────────────┐
│  PICO headset (OpenXR app) │
└──────────────┬─────────────┘
               │ JSON over PXREARobotSDK
               ▼
┌────────────────────────────┐
│  xrobotoolkit_sdk          │
│   - 26-joint hand state    │
│   - controller poses       │
│   - body / motion data     │
└──────────────┬─────────────┘
               │ xrt.get_*_hand_tracking_state()
               ▼
┌────────────────────────────────────────────────────┐
│  PicoStreamer  (main process, every get() tick)    │
│                                                    │
│   pico_data["{hand}_hand_tracking_state"]          │
│         │                                          │
│         ├─► hand_state_to_unitree_keypoints()      │
│         │      → (25, 3) wrist-anchored keypoints  │
│         │      in unitree-hand frame               │
│         │                                          │
│         ├─► push_keypoints_to_shared()             │
│         │      writes Array('d', 75)               │
│         │                                          │
│         └─► _generate_finger_data()                │
│                fills (25, 4, 4) translation cols   │
│                with same keypoints (in meters)     │
└────────┬──────────────────────────────────┬────────┘
         │                                  │
         │ shared (25,3) array              │ left_fingers / right_fingers
         ▼                                  ▼
┌────────────────────────────┐   ┌────────────────────────────────┐
│ BraincoController          │   │ G1GripperInverseKinematicsSolver│
│  (subprocess @ 100 Hz)     │   │  (existing arm IK pipeline)    │
│                            │   │                                │
│  dex_retargeting           │   │  thumb-to-tip distances        │
│        │                   │   │  → arm joint pose              │
│        ▼                   │   └────────────────────────────────┘
│  joint angles → normalize  │
│  to [0, 1]                 │
│        │                   │
│        ▼                   │
│  unitree_sdk2py DDS publish│
│   rt/brainco/left/cmd      │
│   rt/brainco/right/cmd     │
└────────────────────────────┘
            │
            ▼
   ┌─────────────────┐
   │ Brainco RevoLimb│
   │  (over DDS)     │
   └─────────────────┘
```

Key property: **one** `xrobotoolkit_sdk` connection in the main process. Both the brainco DDS path and the existing arm IK fingertip path consume the same `(26, 7)` hand state per tick.

---

## 3. Files added

```
decoupled_wbc/control/teleop/device/pico/brainco/
├── __init__.py                       Public exports
├── README.md                         (this file)
├── hand_retargeting.py               BraincoHandRetargeting wrapper
│                                     (loads URDFs + brainco.yml; isolated to
│                                     the BraincoController subprocess so the
│                                     dex_retargeting C++ optimizer never runs
│                                     in the main process)
├── robot_hand_brainco.py             BraincoController:
│                                       - subprocess control loop @ fps Hz
│                                       - reads (75,) shared array
│                                       - retarget → normalize → DDS publish
│                                       - subscribes to state topics
├── brainco_bridge.py                 Pure functions:
│                                       - hand_state_to_unitree_keypoints()
│                                         (26, 7) → (25, 3) frame transform
│                                       - push_keypoints_to_shared()
│                                       - make_brainco_shared_arrays()
│                                     No background process. Designed to be
│                                     called synchronously from PicoStreamer.
└── assets/brainco_hand/              Copied from xr_teleoperate
    ├── brainco.yml                   dex_retargeting config
    ├── brainco_left.urdf
    ├── brainco_right.urdf
    └── meshes/                       STL meshes referenced by the URDFs

decoupled_wbc/control/teleop/main/
└── run_brainco_teleop.py             Standalone CLI: brainco-only path
                                     (no arm IK), useful for first bring-up
                                     and frame validation.
```

---

## 4. Files modified

### 4.1 `streamers/pico_streamer.py`

**`__init__`** — gained `enable_brainco: bool = False, brainco_fps: float = 100.0` kwargs. When enabled, the streamer allocates two shared `Array('d', 75)` and constructs a `BraincoController` (which spawns its own subprocess). No extra xrobotoolkit connection.

**`_get_pico_data`** — after the existing `get_hand_tracking_state(...)` calls, pushes the resulting `(26, 7)` arrays through `hand_state_to_unitree_keypoints` and writes the resulting `(25, 3)` keypoint clouds into the brainco shared arrays. Conditional on `enable_brainco`.

**`_generate_finger_data`** — replaced the synthetic button-encoded path. Now:

- If hand tracking is active for that hand, fills the translation column `[:3, 3]` of the `(25, 4, 4)` array with **real** wrist-relative keypoints in the unitree-hand frame, in meters.
- The downstream `G1GripperInverseKinematicsSolver` (`solver/hand/g1_gripper_ik_solver.py`) reads exactly that translation column to compute thumb-to-fingertip distances against a 5 cm pinch threshold — distances are now anatomically meaningful.
- Falls back to the legacy synthetic signal when tracking drops (`hand_state is None`), so trigger/grip combos still drive a pinch when occlusion happens.

### 4.2 `device/pico/xr_client.py`

`get_hand_tracking_state` docstring: was wrongly "27 × 7"; corrected to **26 × 7** matching `py_bindings.cpp:19` (`std::array<std::array<double, 7>, 26>`), with the Khronos joint enum and OpenXR pose convention spelled out.

---

## 5. Why these specific design choices

### 5.1 Why a parallel pipeline instead of replacing the existing finger path?

The existing `(25, 4, 4)` fingertip transforms feed `G1GripperInverseKinematicsSolver`, which only reads the translation column. Replacing the path would require teaching the IK solver a new shape (joint angles, not fingertip positions) — out of scope.

Solution: leave the IK solver's input shape unchanged but populate it with **real** keypoints; brainco gets its own `(25, 3) → DDS` path. Both pipelines read from the same xrt source. No duplication of the SDK connection, no fan-out via subprocess.

### 5.2 Why a subprocess for `BraincoController` but not for the bridge?

`BraincoController.control_process` runs `dex_retargeting.retarget(...)` — a C++ NLP solver that takes O(ms) per call. Running it in the main loop would jitter the streamer's `get()` cadence (and indirectly the arm IK loop). Putting it in a daemon subprocess at a fixed `fps` Hz isolates it.

The bridge (the (26,7) → (25,3) transform) is pure NumPy, runs in O(µs), and only needs to write into a `multiprocessing.Array`. No subprocess required.

### 5.3 Why no `xrt.init()` inside the brainco code?

An earlier version had `BraincoBridge` spawn its own subprocess that called `xrt.init()` and read hand tracking independently — two parallel xrt connections to the same PICO service, fragile and wasteful. Refactored: `PicoStreamer` is the single xrt owner, hands the data to brainco via shared memory.

### 5.4 Why drop joint 0 (PALM)?

`XR_HAND_JOINT_COUNT_EXT = 26` is the Khronos `XR_EXT_hand_tracking` constant. The enum order is fixed:

| Index   | Joint                       |
|--------:|:----------------------------|
| 0       | `XR_HAND_JOINT_PALM_EXT`    |
| 1       | `XR_HAND_JOINT_WRIST_EXT`   |
| 2..5    | thumb metacarpal → tip      |
| 6..10   | index (5 joints)            |
| 11..15  | middle                      |
| 16..20  | ring                        |
| 21..25  | little                      |

PICO's PXREA service serializes `xrLocateHandJointsEXT` results 1:1 into `HandJointLocations` JSON. We drop PALM so the resulting 25-joint layout matches what xr_teleoperate's `dex_retargeting` config expects (WRIST at 0, fingertips at 4/9/14/19/24).

The brainco YAML at `assets/brainco_hand/brainco.yml:25` has:
```yaml
target_link_human_indices_vector: [ [ 0, 0, 0, 0, 0 ], [ 4, 9, 14, 19, 24 ] ]
```
That mapping is only consistent if PALM was dropped. This independently confirms the convention.

---

## 6. Coordinate-frame derivation (the math)

This is the crux of correctness — get this wrong and the brainco hand mirrors badly or rotates 90°.

### 6.1 What dex_retargeting expects

Vector input `delta_unitree = pos_tip - pos_wrist` in the **brainco URDF base_link frame** (= "unitree-hand" basis). The retargeter optimizes joint angles such that forward kinematics on the URDF reproduces these vectors.

### 6.2 What xr_teleoperate does (the reference pipeline)

From `televuer/src/televuer/tv_wrapper.py:319-344` and `televuer/src/televuer/televuer.py:278`:

1. Hand positions arrive in OpenXR-world basis as `pos_world_openxr` (25, 3).
2. `pos_world_robot = R_ROBOT_OPENXR @ pos` — basis change OpenXR → robot.
3. The "arm pose" used as the reference is **literally** `hand_data[0:16]` — the wrist joint pose from hand tracking, basis-changed to robot:
   `wrist_T_robot = T_ROBOT_OPENXR @ wrist_T_openxr @ T_OPENXR_ROBOT`
4. `pos_arm_local_robot = inv(wrist_T_robot) @ pos_world_robot` — express in wrist frame.
5. `pos_unitree = T_TO_UNITREE_HAND @ pos_arm_local_robot` — basis change robot → unitree-hand.

### 6.3 Algebraic reduction

For the retargeter, only differences `tip - wrist` matter. The wrist subtraction kills the translation, leaving:

```
delta_unitree = T_TO_UNITREE_HAND @ inv(wrist_T_robot)[:3,:3] @ R_ROBOT_OPENXR @ delta_world_openxr
```

Substituting `inv(wrist_T_robot)[:3,:3] = R_ROBOT_OPENXR @ wrist_R_openxr.T @ R_OPENXR_ROBOT` (orthogonal-matrix identities):

```
delta_unitree = T_TO_UNITREE_HAND @ R_ROBOT_OPENXR @ wrist_R_openxr.T @ R_OPENXR_ROBOT @ R_ROBOT_OPENXR @ delta_world_openxr
              = T_TO_UNITREE_HAND @ R_ROBOT_OPENXR @ wrist_R_openxr.T @ delta_world_openxr
                \________________________________/
                          one rotation
```

So the entire chain collapses to **a single 3×3 rotation** applied to the wrist-local OpenXR vector. Define:

```
R_HAND_TO_UNITREE = T_TO_UNITREE_HAND @ R_ROBOT_OPENXR
                  = [[0, 1, 0],
                     [0, 0, 1],
                     [1, 0, 0]]
```

(Verified: `det = 1`, `R @ R.T = I`.)

### 6.4 What the bridge does

`hand_state_to_unitree_keypoints` ([brainco_bridge.py](brainco_bridge.py)) implements exactly this:

```python
keep         = np.delete(arr, PALM_INDEX, axis=0)            # (26,7) → (25,7)
pos_world    = keep[:, :3]
wrist_R      = R.from_quat(keep[0, 3:]).as_matrix()
pos_local    = pos_world - pos_world[0]
pos_wrist    = pos_local @ wrist_R                           # = (wrist_R.T @ delta.T).T
pos_unitree  = pos_wrist @ R_HAND_TO_UNITREE.T               # composed basis change
```

Both `T_TO_UNITREE_HAND` and `R_ROBOT_OPENXR` are copied verbatim from `xr_teleoperate/teleop/televuer/src/televuer/tv_wrapper.py`, so the math matches their proven pipeline 1:1.

### 6.5 An earlier mistake worth recording

A previous version of the bridge applied `T_TO_UNITREE_HAND` directly to the wrist-local OpenXR vector — i.e. skipped the `R_ROBOT_OPENXR` factor. This produced a vector in the wrong basis (OpenXR-local rather than robot-local), and the retargeter would converge to a 90°-off pose. The fix is the composition above; the docstring in `brainco_bridge.py` warns future-you not to undo it.

### 6.6 Why no on-device frame calibration is required

Both unknowns from the prior writeup are fixed by spec:

- **26-joint layout** is fixed by Khronos `XR_EXT_hand_tracking`. PALM=0, WRIST=1, etc.
- **Wrist-quaternion convention** is fixed by the same spec: +X distal, +Y dorsal, +Z RH-complete.

Cross-confirmed in this repo: `R_HEADSET_TO_WORLD` in `pico_streamer.py:10-16` (used for the existing, working arm IK pipeline) is **bitwise identical** to `R_ROBOT_OPENXR` in xr_teleoperate. That proves xrobotoolkit emits standard OpenXR conventions all the way through, so hand-joint quaternions ride the same convention.

The only remaining unknown is mechanical: whether the brainco firmware's motor 0..5 ordering matches `BraincoLeftJointIndex`/`BraincoRightJointIndex`. This is documented at <https://www.brainco-hz.com/docs/revolimb-hand/product/parameters.html> and is hardcoded in `robot_hand_brainco.py:36-50`.

---

## 7. Real-robot integration

### 7.1 Hardware prerequisites

- PICO 4 Ultra Enterprise (or compatible OpenXR headset) running the XRoboToolkit Unity app.
- Host machine with the XRoboToolkit-PC-Service daemon installed (`/opt/apps/roboticsservice/runService.sh` — auto-launched by `PicoStreamer.run_pico_service`).
- Brainco RevoLimb left + right hands wired to the Unitree CycloneDDS bus on the same network as the host (or onboard the G1).
- Unitree G1 (or any DDS-speaking host) reachable on the configured network interface.

### 7.2 Software prerequisites

`dex-retargeting` is now declared in `gear_sonic[teleop]`, so the standard
install path picks it up automatically:

```
bash install_scripts/install_pico.sh
source .venv_teleop/bin/activate
```

That installs `gear_sonic[teleop]` (incl. `dex-retargeting`, `pyyaml`) and
builds the in-tree native modules `unitree_sdk2py` and `xrobotoolkit_sdk`
from `external_dependencies/`.

If you'd rather wire the deps by hand:

```
pip install dex-retargeting pyyaml
cd external_dependencies/unitree_sdk2_python && pip install -e .
cd external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64 && python setup.py install
```

### 7.3 Two ways to run

**A. Standalone (bring-up & frame validation)** — drives only the hands, no arm IK. Good for first bring-up:

```
python -m decoupled_wbc.control.teleop.main.run_brainco_teleop \
    --network-interface eth0
```

What you should see:
- `xrobotoolkit SDK initialised.`
- `Initialising BraincoController` → `[BraincoController] DDS state ready` (this requires the brainco hands to be already powered and publishing on `rt/brainco/{left,right}/state`).
- Open and close your hand: brainco hand mirrors. Pinch thumb-to-index: motor 0 and 2 close.

**B. Integrated with the existing teleop stack:**

```python
from decoupled_wbc.control.teleop.streamers.pico_streamer import PicoStreamer
streamer = PicoStreamer(enable_brainco=True)
```

That's the entire change. The streamer's `get()` keeps returning `StreamerOutput` for the arm IK path; brainco runs as a parallel subprocess publishing to its own DDS topics.

### 7.4 DDS topics

Subscribed (by `BraincoController`):
```
rt/brainco/left/state
rt/brainco/right/state
```

Published:
```
rt/brainco/left/cmd
rt/brainco/right/cmd
```

Both `MotorCmds_` / `MotorStates_` from `unitree_sdk2py.idl.unitree_go.msg.dds_`. 6 motors per hand.

### 7.5 Joint angles emitted

For each hand, `BraincoController` publishes 6 floats in `[0, 1]` per the brainco API contract:

| Motor ID | Joint              | Internal range (rad) |
|---------:|:-------------------|:---------------------|
| 0        | thumb              | [0, 1.52]            |
| 1        | thumb-aux          | [0, 1.05]            |
| 2        | index proximal     | [0, 1.47]            |
| 3        | middle proximal    | [0, 1.47]            |
| 4        | ring proximal      | [0, 1.47]            |
| 5        | pinky proximal     | [0, 1.47]            |

Mapping `0.0 = open, 1.0 = fully closed` is computed in `_normalize` ([robot_hand_brainco.py:55](robot_hand_brainco.py#L55)).

### 7.6 What the existing arm IK now sees

`PicoStreamer.get().ik_data["left_fingers"]["position"]` is now a `(25, 4, 4)` tensor whose translation column `[:3, 3]` is the **real** wrist-anchored hand keypoint cloud, in **meters**, in the unitree-hand frame.

Downstream `G1GripperInverseKinematicsSolver` reads:
```python
positions = np.array([finger[:3, 3] for finger in fingertips])
thumb = positions[4]; index = positions[9]; ...
if np.linalg.norm(thumb - index) < 0.05:
    q_desired = self._get_index_close_q_desired()
```

The 5 cm threshold is now an actual physical pinch, not a button combo. When tracking drops, the synthetic fallback takes over so triggers/grips still work as a safety net.

---

## 8. Safety features

The controller has three runtime safety guards. None require operator action;
they're listed here so on-call engineers know what's happening when they see
the corresponding log lines.

### 8.1 DDS state-init timeout

`BraincoController.__init__` will not block forever if the brainco state
topics aren't publishing. After `state_timeout_s` (default 5 s) it raises
`TimeoutError` with the offending topic names. Old behaviour ("wait
forever") is opt-in via `state_timeout_s=0`.

```
TimeoutError: BraincoController: no DDS state on 'rt/brainco/left/state' or
'rt/brainco/right/state' within 5.0s. Confirm brainco hands are powered, on
the same DDS network, and that ChannelFactoryInitialize was called with the
right interface.
```

### 8.2 Warmup gate

The first `warmup_frames` (default 10) consecutive valid input frames are
**not** retargeted. The controller publishes the safe "open hand" pose
(q = 0 on every motor) until the gate clears. This blocks the cold-start
case where xrobotoolkit hasn't yet locked tracking and emits a few junk
poses that would otherwise drive the hands closed.

Log lines:

```
[BraincoController] warmup complete after 10 valid frames; publishing retargeted commands
```

### 8.3 Tracking-loss revert

If the input becomes invalid (all-zero or non-finite) after warmup
completed, `consecutive_valid` resets to 0, the controller drops back to
the open-hand pose, and logs:

```
[BraincoController] tracking lost; reverting to open hand
```

Once tracking returns, the warmup counter has to fill again before
retargeted commands resume. This means a one-frame glitch costs you
`warmup_frames / fps` seconds of "open hand", but eliminates the failure
mode where a single noisy frame slams the hand closed.

Both `state_timeout_s` and `warmup_frames` are kwargs on the
`BraincoController` constructor (and propagated through
`PicoStreamer._start_brainco`).

---

## 9. Pre-deploy validation runbook

Run these stages in order. Don't skip ahead until each passes.

### Stage 0 — hardware-free smoke test

```
python -m decoupled_wbc.control.teleop.main.smoke_test_brainco
```

Exercises three things:

1. Every brainco source file imports.
2. `dex_retargeting` can build the URDFs + YAML (skipped on dev boxes
   without the dep — install via `bash install_scripts/install_pico.sh`).
3. The bridge math (`hand_state_to_unitree_keypoints`,
   `push_keypoints_to_shared`) on a synthetic `(26, 7)` input.

Exit code 0 = pass. On a dev machine without runtime deps you'll see a
mix of `[OK]` (parsable code) and `[DEP]` lines (missing third-party
modules); that's expected. On a deployment box every line should be
`[OK]`.

### Stage 1 — unit tests

```
pytest decoupled_wbc/tests/control/teleop/brainco/
```

14 tests covering rotation correctness, palm drop, wrist anchoring,
identity-rotation axis mapping, distance preservation under transform,
and shared-array roundtrip. These are pure NumPy — no SDK needed.

### Stage 2 — PICO connected, brainco hands UNPOWERED

Confirms xrobotoolkit conventions match what we assumed.

1. Connect the headset, launch the XRoboToolkit Unity app, enable hand
   tracking.
2. Run the standalone runner with `--no-publish` (not yet — see TODO in
   §10), or just run a one-off script that calls
   `xrt.get_left_hand_tracking_state()` and prints:

   ```python
   import xrobotoolkit_sdk as xrt; xrt.init()
   import time, numpy as np
   while True:
       s = xrt.get_left_hand_tracking_state()
       print("palm[2]=", s[0][2], "wrist[2]=", s[1][2],
             "tip_index[2]=", s[10][2])
       time.sleep(0.5)
   ```

3. Hold your hand at chest height with fingers up. Verify:
   - `wrist[2]` (z) is below `tip_index[2]` (fingertip is higher than wrist).
   - `palm[2]` is between them — the palm joint sits in the meat of the hand.

   If `palm[2] > tip_index[2]` or `wrist[2] > tip_index[2]`, the OpenXR
   joint convention is non-standard on this firmware and the spec
   assumption needs to change.

### Stage 3 — brainco hands powered, parked safe

Power the hands in a position where a sudden close cannot damage them or
the robot.

1. `python -m decoupled_wbc.control.teleop.main.run_brainco_teleop --network-interface eth0`
2. Watch for the warmup line:
   ```
   [BraincoController] warmup complete after 10 valid frames
   ```
3. Hold your hand still and open. Brainco hand should hold open.
4. Slowly close to a fist. Brainco hand mirrors. Open again — hand opens.
5. Pinch thumb-to-index. Verify motor 0 (thumb) and motor 2 (index) close.

If anything looks 90° off, that's the frame transform; see §6.3.

### Stage 4 — full integrated teleop

Only after Stages 0–3 pass:

```python
streamer = PicoStreamer(enable_brainco=True)
```

The arm IK pipeline keeps running unchanged; brainco rides in parallel.

---

## 10. Failure modes & how to debug

| Symptom                                                | Likely cause                                                              | Fix                                                                                                |
|--------------------------------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `TimeoutError: no DDS state on rt/brainco/...`         | Brainco hands not powered, or wrong DDS interface                          | Confirm hands powered, on the same network, `--network-interface` matches. Or pass `state_timeout_s=0` to wait forever. |
| Hand opens to safe pose then never closes              | Warmup gate keeps resetting (transient `is_active=0`)                      | Move hand into camera FOV; check `is_active`. Lower `warmup_frames` (default 10) if you trust the source. |
| Brainco fingers move but inverted/rotated              | Frame composition wrong                                                    | Check `R_HAND_TO_UNITREE` derivation in `brainco_bridge.py`; see §6.3.                             |
| Single finger lags / never closes                      | Brainco firmware motor order differs from `BraincoLeftJointIndex`          | Compare against <https://www.brainco-hz.com/docs/revolimb-hand/product/parameters.html>.           |
| `_generate_finger_data` falls back to button mode      | `xr_client.get_hand_tracking_state` returned `None` (`is_active=0`)        | Hand outside FOV / occlusion. Move into camera view; `is_active` should report `1`.                |
| Distance threshold never triggers                      | Keypoints in non-meter units                                               | Print `np.linalg.norm(positions[4] - positions[9])` in fist; should be ~3 cm.                      |
| Retargeter logs "configuration file not found"         | dex_retargeting doesn't find URDFs                                         | `set_default_urdf_dir` in `BraincoHandRetargeting.__init__` must point at `assets/` (already done).|
| `ImportError: dex_retargeting`                         | Teleop venv not installed / not active                                     | `bash install_scripts/install_pico.sh && source .venv_teleop/bin/activate`                         |

---

## 11. Notes for future work

- `BraincoController` currently spawns at the same `fps` as the streamer's tick rate. Decoupling (e.g. running brainco @ 200 Hz while the streamer runs @ 60 Hz) is straightforward — the shared array is already the buffer.
- The existing `_generate_finger_data` fallback (button-encoded synthetic transforms) could be deleted entirely once we trust hand tracking. It's kept for now as a graceful degradation when occlusion drops `is_active` to 0.
- If/when xrobotoolkit exposes a more native finger pose accessor (joint angles instead of joint poses), this whole bridge becomes a single rename.
