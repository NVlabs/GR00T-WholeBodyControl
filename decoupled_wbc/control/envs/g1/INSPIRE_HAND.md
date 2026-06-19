# G1 + Inspire RH56 Hand — Integration & Runbook

This documents the Inspire RH56 (FTP series) 6-DOF dexterous hand support for the
G1, selectable alongside the default Unitree dex3 three-finger hand.

> **Scope:** real-robot deploy + teleop. Mujoco sim of the inspire hand is **not**
> supported (there is no inspire MJCF; the sim bridge only handles the dex3
> topics). Use `--interface real` with the inspire hand.

---

## 1. Architecture / data flow

The integration is entirely index-driven by the robot model, so the 6-DOF hand
slots into the existing pipeline without dimension hacks:

```
Manus glove
  → run_teleop_policy_loop  (G1InspireGripperInverseKinematicsSolver: pinch → 6 joint angles)
  → writes into body_q at the inspire hand joint indices (upper_body slice)
  → ROS control goal (target_upper_body_pose)
  → run_g1_control_loop → wbc_policy → full q → env.queue_action
  → G1InspireHand: radians → [0,1000] drive units
  → publishes inspire_hand_ctrl on  rt/inspire_hand/ctrl/{l,r}  (angle mode)
  → [Inspire driver process] → Modbus TCP/serial → hardware
  ← reads inspire_hand_state from rt/inspire_hand/state/{l,r} (angle_act → radians)
```

Key files:

| File | Role |
|---|---|
| `utils/inspire_hand_spec.py` | Single source of truth: 6 DOF order, limits, radian↔drive mapping |
| `g1_inspire_hand.py` | `G1InspireHand(Env)` — 6-DOF drop-in for `G1ThreeFingerHand` |
| `utils/command_sender.py` | `InspireHandCommandSender` (publishes ctrl) |
| `utils/state_processor.py` | `InspireHandStateProcessor` (subscribes state) |
| `robot_model/model_data/g1/g1_29dof_with_inspire_hand.urdf` | Minimal deploy URDF (6 revolute joints/hand, no meshes) |
| `robot_model/supplemental_info/g1/g1_supplemental_info.py` | `G1SupplementalInfo(hand_type="inspire")` |
| `teleop/solver/hand/g1_gripper_ik_solver.py` | `G1InspireGripperInverseKinematicsSolver` |

---

## 2. Prerequisites

1. **Install the Inspire SDK** (`inspire_sdkpy`) into the same env as the control
   loop. From `external_dependencies/inspire_hand_ws/` (see its `readme.md`):
   ```bash
   cd inspire_hand_sdk && pip install -e .
   # plus its unitree_sdk2_python if not already installed
   ```
   `inspire_sdkpy` is imported lazily, so dex3-only setups don't need it.

2. **Run the Inspire hand driver** on the host wired to the hands. This bridges
   DDS ↔ Modbus and is what actually drives the hardware (analogous to the
   Unitree firmware for dex3). From `inspire_hand_sdk/example/`:
   ```bash
   python Headless_driver_l.py     # left hand  → rt/inspire_hand/{ctrl,state}/l
   python Headless_driver_r.py     # right hand → rt/inspire_hand/{ctrl,state}/r
   ```
   Default hand connection: Modbus TCP `192.168.11.210:6000` (configurable via the
   driver / hand registers IP_PART1..4 @ 1700–1703).

3. Hands powered (24 V) and reachable on the configured interface.

---

## 3. Running teleop with the inspire hand

Pass `--hand_type inspire` to **both** the teleop policy loop and the control
loop. They are separate processes and the whole-body `q` vector crosses ROS
between them — if they disagree on hand type the slice lengths won't match
(inspire = 41 dofs vs dex3 = 43).

```bash
# Terminal A — control loop (drives the robot + hands). Start this first.
python decoupled_wbc/control/main/teleop/run_g1_control_loop.py \
    --interface real --hand_type inspire --with_hands True

# Terminal B — teleop policy (retargeting from the glove/controllers)
python decoupled_wbc/control/main/teleop/run_teleop_policy_loop.py \
    --hand_type inspire --hand_control_device=pico --body_control_device=pico
```

`--hand_type` is a `BaseConfig` field; it is also merged into the wbc config as
`HAND_TYPE`, which `G1Env` reads to pick `G1InspireHand`, and which
`instantiate_g1_robot_model` uses to load the inspire URDF. One flag drives all
three (robot model, hand IK solver, hand control).

> Note the default teleop hand control device produces pinch gestures; the inspire
> solver maps them to a progressive grasp (see §5).

---

## 4. Joint specification (from the Inspire FTP user manual)

6 drives, in `angle_set`/`angle_act` index order (also the actuator order in the
robot model). Drive units are integers `[0, 1000]`, **`1000` = fully open,
`0` = fully closed** (`-1` = "no action", never emitted here).

| idx | joint suffix | finger | physical range | modeled limit (rad) |
|---|---|---|---|---|
| 0 | `pinky` | little finger | 20°–176° | `[0, 2.72]` |
| 1 | `ring` | ring finger | 20°–176° | `[0, 2.72]` |
| 2 | `middle` | middle finger | 20°–176° | `[0, 2.72]` |
| 3 | `index` | index finger | 20°–176° | `[0, 2.72]` |
| 4 | `thumb_bend` | thumb proximal pitch | −13°–70° | `[0, 1.45]` |
| 5 | `thumb_rot` | thumb proximal yaw | 90°–165° | `[0, 1.31]` |

We model each drive as a single "flexion-from-open" joint: `q = 0` is open,
`q = Q_CLOSED` is closed. The mapping is linear:
`drive = 1000 · (Q_CLOSED − q) / Q_CLOSED`, clipped to `[0, 1000]`.

Relevant registers (Modbus): `ANGLE_SET` @ 1486, `ANGLE_ACT` @ 1546, `POS_ACT`
@ 1534 (note `POS` is 0–2000 and inverted vs `ANGLE`; we use `ANGLE`).

---

## 5. Tuning notes

These are sensible defaults to verify on hardware, all centralized:

- **Grasp amplitudes** (`G1InspireGripperInverseKinematicsSolver`):
  `FINGER_FLEX=2.2`, `THUMB_BEND=1.1`, `THUMB_ROT=1.0` (~80% of travel). The pinch
  → grasp mapping closes progressively from index toward pinky.
- **Open/closed direction**: confirmed `1000=open / 0=closed` against the manual.
  If a real hand moves inverted, flip `Q_OPEN`/`Q_CLOSED` in `inspire_hand_spec.py`.
- **Joint limits**: the modeled radian limits are the manual's physical travel.
  They are kept in sync across `inspire_hand_spec.Q_CLOSED`, the URDF, and
  `G1SupplementalInfo`; change all three together.
- The inspire state message has no joint velocity/accel/torque, so `G1InspireHand`
  reports `dq/ddq/tau_est` as zeros (not used for hand control).

---

## 6. Known limitations

- **Sim unsupported** for the inspire hand (deploy-only). Use `--interface real`.
- **Not yet validated live** end-to-end (no hardware/ROS/pinocchio in CI here).
  Verified offline: URDF parses, radian↔drive round-trips, supplemental info
  builds 6-DOF, solver outputs stay within limits, all modules compile.
- The pinch-based grasp is a heuristic (matching the dex3 teleop fidelity), not a
  fingertip-accurate retarget.
