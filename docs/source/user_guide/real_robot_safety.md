# Real Robot Teleoperation Safety

This page is the operator checklist for PICO-driven Unitree G1 teleoperation
with GR00T / GEAR-SONIC low-level deploy.

## Supported vs. Current Setup

The official VR teleoperation setup uses a PICO headset, two hand controllers,
and two ankle motion trackers. The controller-only path in this repository uses
the headset as the third VR_3PT point and is therefore outside the official
full-body tracking setup. Treat every first hardware run as experimental until
the same command sequence has been validated in MuJoCo.

## Stop Layers

During GR00T low-level deploy, the process publishes low-level motor commands.
Do not rely on Unitree remote `L1+A` or `L2+B` as an E-stop while this deploy is
running.

Use these stop layers, in order:

1. Safety operator presses keyboard `O` in the deploy terminal.
2. VR operator presses PICO `A+B+X+Y`.
3. If software stop does not work, use the protective harness/frame and physical
   power-off procedure. Expect the robot to become dead weight.

Drill before each real teleop session:

1. Press keyboard `O` three times while the safety operator watches the deploy
   terminal confirm the stop path.
2. Press PICO `A+B+X+Y` three times while the VR operator confirms the manager
   sends the stop command.
3. Simulate XR loss once by pausing/disconnecting the XR stream and confirm the
   watchdog freezes then stops according to the configured thresholds.
4. Attempt one intentionally mismatched VR_3PT entry in sim and confirm the
   entry gate refuses it.

## Required Preflight

Before `./deploy.sh --input-type zmq_manager real`:

- Put the robot on a safety harness/protective frame for early runs.
- Keep a clear 3 m zone around the robot, with only the designated spotter near
  the robot.
- Keep the robot in line of sight on a flat, dry, unobstructed floor.
- Confirm robot, headset, and controller batteries are charged.
- Confirm the operator wears tight-fitting clothing for VR tracking.
- Measure network latency. Do not run real teleop when latency is above 30 ms.
- Stop all MuJoCo/simulation processes.
- Start from the robot zero/CALIB_FULL reference pose and verify wrist axes in
  simulation first.
- If operator pose and robot pose visibly mismatch, do not enter VR_3PT.

The deploy script runs `gear_sonic_deploy/scripts/preflight.sh` for real mode and
requires per-item confirmation before launching.

## Software Safeguards Added for Real Runs

- Deploy-side VR_3PT filter clamps invalid or abrupt position/orientation target
  changes and escalates to the existing stop path after sustained violations.
- PICO manager-side XR watchdog freezes once after 50 ms stale data and sends a
  stop command after 200 ms stale data.
- VR_3PT entry gate compares calibrated PICO targets with current G1 FK and
  refuses entry if wrist position, torso/head-point position, or wrist
  orientation mismatch exceeds the configured limits.
- VR_3PT ramp is only allowed after the entry gate passes and post-gate
  recalibration has run.

Default VR_3PT entry limits:

- `--vr3pt_entry_wrist_pos_max_m 0.25`
- `--vr3pt_entry_torso_pos_max_m 0.20`
- `--vr3pt_entry_wrist_orn_max_deg 45.0`
