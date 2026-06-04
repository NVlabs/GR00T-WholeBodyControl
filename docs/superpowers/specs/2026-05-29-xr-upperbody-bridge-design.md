# XR Upper-Body Bridge Design

## Goal

Control the Unitree G1 lower body with GEAR-SONIC for balance and safe posture while an operator uses Meta Quest 3 or PICO 4 Ultra XR/HOMIE-style upper-body tracking to teleoperate arms and hands for GR00T N1.7 imitation-learning episode collection.

## Selected Approach

Use an in-repository bridge process. GEAR-SONIC remains the only process that writes Unitree body `LowCmd`; the bridge only publishes GEAR-SONIC-compatible ZMQ `command` and `planner` messages. This avoids dual body motor writers and lets existing GEAR-SONIC safety, policy inference, state output, and VLA data export remain the authority.

The bridge will not depend on the previous `genonai/xr_teleoperate` source. It will accept upper-body trajectory samples from a simple local source interface first, so live XR/HOMIE integration can plug in later without changing the GEAR-SONIC deployment side.

## Architecture

Runtime ownership:

1. XR/HOMIE source produces upper-body trajectory data only.
2. `gear_sonic/scripts/xr_upperbody_bridge.py` converts those samples into GEAR-SONIC planner-topic packets.
3. `gear_sonic_deploy --input-type zmq_manager` receives:
   - `command`: start/stop and `planner=true`.
   - `planner`: idle lower-body command plus `upper_body_position`, `upper_body_velocity`, and optional Dex3 hand joints.
4. GEAR-SONIC planner/policy owns lower-body balance and posture.
5. Existing C++ output and Python data exporter record state/action/teleop fields for VLA training.

## Interfaces

The first implementation supports a trajectory replay/input format because the current requirement is upper-body trajectory data, not full live source integration:

- JSONL or NPZ file input with timestamped frames.
- Required frame field: `upper_body_position`, shape `[17]`, radians, matching GEAR-SONIC's existing upper-body order.
- Optional frame fields: `upper_body_velocity` `[17]`, `left_hand_joints` `[7]`, `right_hand_joints` `[7]`, `mode`, `movement`, `facing`, `speed`, `height`.
- Default lower-body command is stable idle: `mode=IDLE`, `movement=[0,0,0]`, `facing=[1,0,0]`, `speed=-1`, `height=-1`.

The bridge publishes the existing packed ZMQ message format used by `pico_manager_thread_server.py` and decoded by `ZMQPackedMessageSubscriber`.

## Safety

- The bridge never writes DDS, `LowCmd`, or hand commands directly.
- Stale input freezes at the last accepted upper-body frame for a short grace window, then drops optional upper-body control so GEAR-SONIC returns to planner/default upper-body behavior.
- Joint values are checked for finite numbers and exact expected dimensions before publish.
- Optional clipping and rate limiting are applied to upper-body joint targets before publish.
- Real robot launch remains manual and requires the repo's existing real-robot safety confirmation.

## Data Collection

The existing `run_data_exporter.py` path remains the recorder. It subscribes to C++ state output and the same `planner` topic, so bridge-published upper-body and hand targets become part of the recorded teleop/action features. The bridge adds source metadata to its console/config output and uses topic fields already represented in `features_sonic_vla.py`.

## Testing

Unit tests should cover:

- Loading JSONL and NPZ trajectory frames.
- Validation failures for wrong dimensions and non-finite values.
- Rate limiting and clipping behavior.
- ZMQ planner packet field names, shapes, dtypes, and defaults.
- A smoke publish loop using a fake publisher or local decode helper, without requiring robot hardware.

Manual validation sequence:

1. Run unit tests for the bridge.
2. Run bridge in dry-run mode on a sample trajectory.
3. Run sim2sim with `--input-type zmq_manager` and bridge publishing idle lower-body plus upper-body targets.
4. Verify data exporter records the expected teleop planner/upper-body/hand fields.
5. Only then consider real robot validation with explicit safety confirmation.

## Non-Goals

- No direct HOMIE or Unitree low-level command writer integration.
- No dual body `LowCmd` writer design.
- No full-body XR retargeting from ankle or lower-body sensors.
- No real robot launch automation.
