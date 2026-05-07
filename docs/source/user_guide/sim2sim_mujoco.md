# Sim-to-Sim: Deploying a Trained Policy in MuJoCo

This guide documents how to take a SONIC checkpoint trained in IsaacLab
and run it in MuJoCo for closed-loop evaluation, plus the collection of
subtle IsaacLab↔MuJoCo mismatches we hit doing exactly that for the
Agibot X2 Ultra. Every gotcha listed here was a real bug that produced
visually-plausible but completely broken behavior; check this list first
when porting a new embodiment.

The reference implementation is `gear_sonic/scripts/eval_x2_mujoco.py`.
The script is X2-specific in the sense that it loads X2's MJCF, default
pose, gain table, and DOF mappings — but the structure (RSI →
proprioception buffer → tokenizer obs → actor → PD → auto-reset) is
identical for any embodiment. Use it as the template.

## Quick Start

```bash
conda run -n env_isaaclab --no-capture-output python gear_sonic/scripts/eval_x2_mujoco.py \
    --checkpoint logs_rl/<run>/model_step_NNNNNN.pt \
    --motion gear_sonic/data/motions/<robot>_<motion>.pkl
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--init-frame N` | 0 | RSI motion frame to teleport into at every reset |
| `--fall-height H` | 0.4 m | Pelvis-z below this triggers an auto-reset |
| `--fall-tilt-cos C` | -0.3 | `gravity_body[z]` above this triggers an auto-reset (~72° tilt) |
| `--max-episode T` | 0 | If > 0, force-reset every T simulated seconds |
| `--speed S` | 1.0 | Motion phase speed multiplier |
| `--device` | cpu | Inference device for the actor |

Viewer controls: `SPACE` pause/resume, `R` manual reset, `V` toggle
tracking/free camera.

## How the Script Mirrors IsaacLab

The deployment loop reproduces what IsaacLab does on every episode:

1. **Reference State Initialization (RSI)** — at every reset, the robot
   is teleported to the motion's `--init-frame` state. This sets root
   pos/quat/lin_vel/ang_vel and joint pos/vel from the motion library.
2. **Closed-loop control** — each control tick the script reads MuJoCo
   state, builds the IsaacLab-format proprioception (history of the
   last `HISTORY_LEN` ticks) and tokenizer (next `NUM_FUTURE_FRAMES`
   reference frames in the body frame) observations, runs the actor
   (encoder + FSQ + decoder), and applies explicit PD control to track
   the policy's joint targets.
3. **Auto-reset on fall** — when the pelvis drops below
   `--fall-height` or the body tilts past `--fall-tilt-cos` the
   episode resets back to RSI. This is a coarser, height-based trigger
   than IsaacLab's tracking-error terminations used during training,
   so episode lengths in the viewer are **not** comparable to
   training/eval episode lengths — the deployment script lets the
   policy survive bad-tracking states that training would have
   terminated.

## Critical Gotchas

### G1. MuJoCo free-joint `qvel` is NOT all in world frame

This is the single most important pitfall. For a free joint:

| Slot | Frame |
|---|---|
| `qvel[0:3]` (linear) | **World** frame |
| `qvel[3:6]` (angular) | **Body-local** frame (NOT world!) |

This bites in two places:

1. **RSI write** — motion-lib stores `root_ang_vel_w` in world frame.
   You must convert to body before writing:
   ```python
   mj_data.qvel[3:6] = quat_rotate_inverse(root_quat_w_wxyz, root_ang_vel_w)
   ```
2. **Per-step proprioception** — IsaacLab's `base_ang_vel` term is
   already in the body frame. Read `mj_data.qvel[3:6]` directly and
   do **not** rotate by `quat_rotate_inverse(base_quat, ...)`. The
   wrong-assumption rotation produces a *double* rotation, corrupting
   the angular-velocity history every step. Symptom: policy looks
   confused, drifts off balance after a stride, never recovers — but
   the same checkpoint walks fine in IsaacLab.

Verification recipe: zero-gravity single-body MJCF, set `qvel[3:6]`,
let the sim integrate one timestep, and check how `qpos[3:7]` evolves
relative to body-frame and world-frame hypotheses.

### G2. Quaternion convention: wxyz everywhere, xyzw for SciPy only

- IsaacLab and MuJoCo both use **wxyz** (scalar-first) quaternions.
- `scipy.spatial.transform.Rotation.from_quat` / `as_quat` use **xyzw**
  (scalar-last) — a constant source of off-by-90° rotations.
- Use a `quat_rotate_inverse` that matches IsaacLab's `quat_apply_inverse`
  exactly. Don't substitute a SciPy-derived helper without converting
  the quaternion order, and double-check the sign convention. A sign-flip
  bug will make gravity in the body frame point *up* instead of down;
  symptom: the robot tries to stand on its head.

### G3. DOF reordering is required and the gather-index naming is confusing

IsaacLab and MuJoCo traverse the kinematic tree in different orders.
Maintain two index permutations in your embodiment file (e.g.
`gear_sonic/envs/manager_env/robots/<robot>.py`):

```python
# Gather index: result[il_pos] = source[IL_TO_MJ_DOF[il_pos]]
IL_TO_MJ_DOF = [...]   # used to convert MJ→IL
MJ_TO_IL_DOF = [...]   # used to convert IL→MJ
```

The naming describes "where to *gather from*", not "where to map to":

- **MJ → IL (state read):** `dof_pos_il = dof_pos_mj[IL_TO_MJ_DOF]`
- **IL → MJ (action write):** `action_mj = action_il[MJ_TO_IL_DOF]`

We had these swapped early on. The symptom was huge correlated
joint motion — the wrong permutation is still a meaningful permutation,
just with totally different per-joint meanings. When in doubt set up a
sentinel test: write a known per-joint pattern, round-trip it, and
verify you get back what you started with.

### G4. Default joint pose, action scale, and PD gains all matter

Pull these from the embodiment file:

- **`DEFAULT_JOINT_POS`** is the home pose. Joint targets are
  `target = DEFAULT_DOF + action * ACTION_SCALE`, **not** raw
  `action * ACTION_SCALE`. Forgetting the offset gives the policy a
  zero-initialized pose where (e.g.) knees are straight, leading to
  instant collapse on contact.
- **`ACTION_SCALE`** is per-joint. Typical humanoid configs use
  large values for hip/knee (~0.5) and smaller values for arms
  (~0.25). A single global scalar will keep the robot alive but
  produces noticeably stiff arms and over-driven legs.
- **`KP` / `KD`** must be derived the same way IsaacLab's implicit
  actuator computes them internally (stiffness + armature
  contribution). Do **not** copy raw `effort_limits` or `damping`
  from the actuator config and use them directly as PD gains.

### G5. Implicit (IsaacLab) vs explicit (MuJoCo) actuator

IsaacLab's PD runs implicitly with armature on the inertia matrix.
MuJoCo's `ctrl`-driven PD is explicit. With low arm/waist KP
(weak by design — many SONIC robots have ~14 N·m/rad on upper-body
joints) the difference is observable: explicit MuJoCo cannot hold
the upper body up against gravity by itself, and the policy is
expected to add the gravity-comp torque.

This is **not something you "fix"** in deployment — it's a known
sim gap. Practical implications:

- Do not expect the robot to hold its default pose under a zero
  action. IsaacLab can; MuJoCo cannot.
- Once the policy is trained well, sag disappears (the policy adds
  the needed torque). With a weakly-trained checkpoint you may see
  the upper body lean even though the legs walk.

### G6. Proprioception layout follows the dataclass, not the YAML

The flat proprioception vector is **not** in YAML order — it's in
`PolicyCfg` dataclass attribute order
(`gear_sonic/envs/manager_env/mdp/observations.py`):

```
[base_ang_vel, joint_pos_rel, joint_vel, last_action, gravity_dir]
```

Each term is concatenated across `history_length` frames,
**oldest-first** within the term. For an embodiment with `N` DOF and
`history_length=10` the layout is:

| Term | Per-frame | × history | Block size |
|---|---|---|---|
| `base_ang_vel` | 3 | 10 | 30 |
| `joint_pos_rel` | N | 10 | 10·N |
| `joint_vel` | N | 10 | 10·N |
| `last_action` | N | 10 | 10·N |
| `gravity_dir` | 3 | 10 | 30 |

Total: `60 + 30·N`. (For X2: 60 + 30·31 = **990**.)

A wrong term order produces nicely-numerically-ranged garbage. Use
`gear_sonic/scripts/dump_isaaclab_step0.py` to dump IsaacLab's
proprioception at step 0, then slice each block in your MuJoCo build
and verify the values match the corresponding `env_state` quantity
within the configured noise tolerance.

### G7. CircularBuffer reset broadcast-fills with the first observation

IsaacLab's `CircularBuffer` does **not** initialize history to zeros.
On reset it *broadcast-fills* every history slot with whatever the
first appended observation is. Replicate this:

```python
def append(self, ...):
    if not self._primed:
        for _ in range(HISTORY_LEN):
            self._hist.append(obs)   # for each term
        self._primed = True
    else:
        self._hist.append(obs)       # for each term
```

Wrong behavior (zero-initialized history): the policy makes huge
corrections in the first few control ticks because the `last_action`
history is artificially zero, even though the at-rest action would
have produced a finite value. The robot wobbles for the first
~200 ms of every episode then either settles or explodes.

### G8. FSQ quantization layer in the actor forward path

The universal-token actor's tokenizer encoder output is passed
through a Finite Scalar Quantizer (FSQ) — a `tanh`-then-round-to-bin
step. The checkpoint stores the FSQ levels but a naive forward
through encoder + decoder does **not** apply it automatically. You
must insert it explicitly:

```python
def fsq_quantize(z, levels):
    # z: (B, num_tokens * token_dim)
    bound = (levels - 1) * 0.5
    z_bounded = bound * torch.tanh(z)
    z_quant = torch.round(z_bounded) / bound
    return z_quant
```

Read the FSQ levels from the checkpoint config (look for
`quantizer.levels`). Without quantization the policy works
marginally — actions are continuous and slightly off, the robot
wobbles but doesn't catastrophically fail. With it, the policy
matches its trained behavior.

### G9. Tokenizer observation: future reference in the *current* body frame

The tokenizer encoder consumes a flattened sequence of the next
`num_future_frames` reference frames. Each frame's data
(`command_multi_future_nonflat`, `motion_anchor_ori_b_mf_nonflat`,
etc. — depends on the encoder config) is expressed in the **current**
body frame, not world frame. The reference frame is captured at the
*current* control tick, so each tokenizer obs is a fresh body-frame
projection of the same world-frame motion clip — repeating the
projection every tick is correct.

For X2 with 10 future frames at 0.1 s spacing the total dim is
`10 × 68 = 680`. Frame width depends on the tokenizer observation
group; see `gear_sonic/config/manager_env/observations/tokenizer/`.

### G10. RSI is the proper reset, not a clean spawn

IsaacLab does **not** start episodes with the robot standing in its
default pose at the origin. It teleports the entire robot — joints
+ root — to a sampled motion frame's state. This means:

- The first proprioception observation already has motion-correct
  joint velocities and base angular velocity.
- The "first action" the policy ever sees comes from a mid-stride
  state, not a stationary one.

If you spawn the robot at the default pose and let physics settle,
the first ~10 frames of proprioception will be wildly out-of-distribution
relative to training, and the policy will produce huge actions trying
to "fix" the imagined error. The deployment script uses RSI by default;
keep it that way unless you have a specific reason not to.

## Debugging Recipe: When IsaacLab Works but MuJoCo Doesn't

1. **Dump the IsaacLab step-0 ground truth** with
   `gear_sonic/scripts/dump_isaaclab_step0.py`. It writes the full
   proprio vector, tokenizer obs, encoder/FSQ/decoder activations,
   and the final action to a `.pt` file.
2. In the MuJoCo loop, after RSI but before the first policy call,
   build the same observations and compare:
   - **Proprioception**: each block (per the G6 layout) should match
     the corresponding `env_state` quantity within IsaacLab's
     observation noise tolerance.
   - **Tokenizer obs**: should match exactly (no noise at step 0).
   - **Encoder output, FSQ output, decoder output, final action**:
     each should match within a few `1e-3` (FSQ rounding + float32
     differences).
3. Whichever block diverges first tells you exactly which side of the
   pipeline has the bug.

## Symptom → Likely Cause

| Symptom | First place to look |
|---|---|
| Robot collapses immediately on reset | Default pose + action-scale offset (G4); RSI not applied (G10) |
| Robot stands but tips over slowly | Implicit-vs-explicit actuator (G5); upper-body sag — sometimes expected |
| Robot walks one stride then falls | qvel angular frame (G1); proprio term order (G6); buffer not primed (G7) |
| Wild whole-body motion, looks "possessed" | DOF reorder direction swapped (G3); FSQ missing (G8) |
| Robot tries to stand on its head | Quaternion sign convention or `quat_rotate_inverse` direction (G2) |
| Arms float, legs over-respond | Single global action-scale instead of per-joint (G4) |
| Walks fine for ~1 s then drifts | Tokenizer obs body-frame transform off (G9); motion phase clock not aligned with RSI frame (G10) |
| Same checkpoint walks in IsaacLab, fails in MuJoCo | Always start with the dump-and-compare recipe above |

## See Also

- `gear_sonic/scripts/eval_x2_mujoco.py` — reference deployment script
- `gear_sonic/scripts/dump_isaaclab_step0.py` — IsaacLab GT dumper
- `gear_sonic/envs/manager_env/robots/x2_ultra.py` — example embodiment config
- {doc}`new_embodiments` — adding a new robot from scratch
- {doc}`../references/conventions` — coordinate / quaternion / DOF conventions
- {doc}`../references/observation_config` — full observation-group reference
