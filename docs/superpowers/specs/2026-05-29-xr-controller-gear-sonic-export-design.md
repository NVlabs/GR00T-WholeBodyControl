# XR Controller GEAR-SONIC Export Design

## Goal

Extend the Unitree `xr_teleoperate` GEAR-SONIC export path so Meta Quest 3 and Pico 4 Ultra controllers can drive upper-body IK, Dex3 grippers, recording/pause/quit controls, and GEAR-SONIC planner locomotion commands.

## Scope

This feature applies to `xr_teleoperate --gear-sonic-export --input-mode controller`.

GEAR-SONIC remains the only body `LowCmd` writer. `xr_teleoperate` only publishes JSON intent over ZMQ. The existing `xr_upperbody_bridge` converts that JSON into GEAR-SONIC `planner` and `manager_state` messages.

## Controller Mapping

Both Quest 3 and Pico 4 Ultra expose the needed controls through Vuer/OpenXR normalized controller fields:

- left/right controller poses: arm IK targets
- left/right index triggers: gripper open/close
- left/right middle-finger grip triggers: available for future use, logged/debuggable
- left stick: walking/strafe movement
- right stick X: turn/yaw
- right stick Y: squat/kneel height adjustment
- A, B, X, Y buttons: pause/resume, quit, locomotion mode cycling

Button mappings:

- Left trigger closes left Dex3 hand; release opens it.
- Right trigger closes right Dex3 hand; release opens it.
- A + X toggles pause/resume publishing.
- B + Y toggles pause/resume publishing.
- A + B + X + Y publishes one final idle planner command, stops XR export, and exits.
- A + B cycles to the next locomotion mode.
- X + Y cycles to the previous locomotion mode.

Locomotion modes:

- 0: Idle
- 1: Slow Walk
- 2: Walk
- 3: Run
- 4: Squat
- 5: Kneel two legs
- 6: Kneel

Right stick Y adjusts `height` only in modes 4, 5, and 6. Height is clamped to `0.2..0.8`. Squat/kneel modes publish zero movement, speed `0`, and the current height.

Walking modes publish left-stick movement and accumulated facing yaw. If the selected walking mode is 1, 2, or 3 but the left stick is inside the dead zone, the exported planner `mode` becomes 0 with zero movement.

## JSON Export Schema

Each live frame may include:

```json
{
  "timestamp": 0.0,
  "dual_arm_position": [0.0],
  "dual_hand_joints": [0.0],
  "mode": 1,
  "movement": [0.0, 0.0, 0.0],
  "facing": [1.0, 0.0, 0.0],
  "speed": 0.4,
  "height": -1.0,
  "toggle_data_collection": false,
  "stop": false
}
```

The bridge already accepts `mode`, `movement`, `facing`, `speed`, and `height`. It forwards `toggle_data_collection` into `manager_state`. The `stop` field is source-lifecycle metadata only and must not turn off GEAR-SONIC.

## Safety

Pause/resume affects publishing. On pause, `xr_teleoperate` stops sending arm, hand, and planner frames so GEAR-SONIC holds the latest valid command. Quit publishes a final idle planner command so the robot stands still, then exits the export process without stopping GEAR-SONIC.

Controller gripper presets are conservative binary open/close targets. Analog trigger values are logged and can later be used for analog gripper interpolation.

## Validation

Unit tests cover:

- controller state edge detection
- mode cycling
- movement/facing command generation
- squat/kneel height clamping
- gripper target generation
- bridge forwarding of recording flags and ignoring source-lifecycle stop as a GEAR-SONIC stop command

Runtime validation uses `--gear-sonic-debug` and bridge `--debug-live` to print buttons, sticks, planner fields, hand targets, and publish/pause state.
