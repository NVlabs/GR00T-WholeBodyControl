# Reliable native23 full-body teleop

## Target and approach

Build one controller for the actual 23-DOF Unitree G1: received full-body movement → balanced motion → braking → continuous standing. Use the existing laptop only. Replace the six-hour experiment deadline with staged, sustained development.

Keep every body objective. Allow bounded adjustments when human movement approaches robot limits; brake when those limits cannot accommodate the request.

This is a substantial controller-development programme. Previous failures do not establish that another network or longer run will succeed. Progress must show retained physical capabilities.

## 1. Make the training references physically usable

- Preserve original recordings and acceptance criteria as independent benchmarks.
- Build references using the actual native23 model, jointly adjusting legs, waist, arms and root. Resolve self-intersections, foot clearance, support transitions and joint-speed conflicts.
- Keep original head, hand, foot, pelvis and heading objectives. Joint posture becomes a secondary preference where it conflicts with body tracking.
- Check resulting trajectories through native dynamics. Static IK and floor clearance alone cannot establish feasibility.
- Implement the same bounded, causal reference adjustment for recorded and live input. Use current/past packets only. Report adjustments and rejected requests.
- If a benchmark interval cannot fit existing source-tracking limits, identify that interval and conflict. Do not remove it, slow it down or change its limits silently.

## 2. Build a fully trainable native23 controller

Reuse native MuJoCo physics, original PD law, motor limits, applied-command history, snapshot fixes and independent simulation clock.

- Train task encoder and control network together. No frozen factory backbone, mandatory gait generator or absent-joint padding.
- Start with a compact network: three hidden layers of 512, 256 and 128 units; ten received observation frames; 23 bounded joint-target outputs. Initialize output at native standing targets.
- Inputs contain robot joint state, orientation/velocity feedback, received body goals and causal derivatives, actual applied commands, packet age and controller mode. No recording identity, playback phase or future reference.
- Use PPO with a privileged training critic. The deployed actor remains causal. Successful expert trajectories supply physically valid reset states and achievable examples; proposed expert commands do not replace actual applied history.
- Keep the public interface: **timestamped body reference + robot observation → 23 targets + status**. Extend status with reference adjustment and rejection reasons. Pin observation format, model, native limits and reference settings together.

A standalone small policy was already tried. This proposal combines a different observation/action design with repaired references, broader training data and sustained capability retention.

## 3. Train one policy through a capability sequence

Create varied, physically feasible command sequences covering upper-body movement, pelvis shifts, crouching, steps, turns and stops. Include different transitions and interruptions, beyond the three existing training recordings. Keep walk008 held out.

Advance the same policy through:

1. Standing with coordinated full-body pose changes.
2. Single steps in different directions, followed by braking and standing.
3. Composed steps, turns and body movements with unexpected stop requests.
4. Complete recordings and unseen command sequences.

Retain earlier capabilities in every later training stage. Full episodes continue across PPO rollout boundaries; collecting an optimizer batch must not reset the robot or controller memory.

Run two fixed seeds sequentially. Evaluate every five million control transitions; preserve model and optimizer state. Allocate at most 200 million transitions per seed for the first foundation campaign. Benchmark actual training throughput before publishing its wall-time estimate.

After the first 20 million transitions, three consecutive evaluations with neither improved physical success nor reduced tracking error trigger investigation of the first repeatable failure. No automatic reward changes, new controller family or indefinite extension.

## 4. Prove control, then connect Pico and robot feedback

Each capability stage gets 100 fixed, unseen command sequences, including ±0.03 m/s starts. Require physical completion, declared tracking limits and 30 seconds quiet standing. Later stages must retain earlier successes.

Final simulation qualification retains the existing requirements:

- All four complete recordings at original speed, with original full-body tracking and native physical limits.
- Early, middle and late input loss; controlled stopping and explicit rearm.
- Three isolated timing repetitions with independent 500 Hz physics and 50 Hz control; every missed deadline reported.
- Prefix-invariance tests confirming future packets cannot affect current commands.

Then finish supported-standing estimator initialization and repeat qualification using simulated joint/IMU sensors, declared noise and delay, with ground truth available only to evaluation. Next, connect calibrated live Pico input and replay captured packets through the same controller.

Physical robot progression remains separate: read-only configuration checks, motor-disabled shadow mode, then supervised supported standing and progressively larger movements.

## Deliverables and defaults

Deliver one pinned simulation launcher, resumable training configuration, controller package, recorded qualification results and full-duration videos. Report the furthest demonstrated capability clearly.

Local compute only. No recurring automation, vendor dependency, hidden future input, motion-specific controllers or weakened benchmark criteria. The next meaningful deliverable is **one policy performing varied full-body movements and steps, then reliably returning to standing**.
