# XR + GEAR-SONIC Teleop Dashboard Design

Date: 2026-06-01

## Goal

Build a browser UI for the XR controller + GEAR-SONIC teleoperation stack so the teleoperator can see whether the system is live, stale, publishing expected planner commands, or issuing suspicious commands during simulation and real robot collection.

The first version is read-only. It observes XR export, bridge output, and camera frames, but does not send robot control commands. This keeps command ownership in the headset/controller path and GEAR-SONIC bridge while still giving the operator a practical monitoring screen.

Recording and pause/resume status are not authoritative in V1. The current bridge/data-exporter contract publishes toggle edges (`toggle_data_collection`, `toggle_data_abort`), while the authoritative recording state lives inside `run_data_exporter.py`'s `EpisodeState` and is not published. The dashboard may show "last toggle seen" and an inferred local latch, but it must label that state as inferred and possibly desynchronized.

## Non-Goals

- Do not replace the headset/controller input path.
- Do not add browser-side robot control in V1.
- Do not stop GEAR-SONIC from the dashboard.
- Do not require a Node/Vite/frontend build stack.
- Do not change the GEAR-SONIC planner wire format.
- Do not claim authoritative recording or pause state until an exporter/status topic exists.

## Runtime Shape

Add a Python dashboard process:

```bash
python gear_sonic/scripts/xr_teleop_dashboard.py \
  --host 0.0.0.0 \
  --port 8088 \
  --xr-source-host localhost \
  --xr-source-port 5560 \
  --xr-source-topic xr_teleop \
  --bridge-host localhost \
  --bridge-port 5556 \
  --camera-host localhost \
  --camera-port 5555
```

The dashboard serves a static HTML/CSS/JS page and image endpoints using Python `ThreadingHTTPServer`. The page polls `/api/state` for telemetry and polls camera image endpoints for latest cached frames.

## Data Sources

### XR Export

Subscribe to the export-only `xr_teleoperate` JSON stream:

- endpoint: `tcp://<xr-source-host>:<xr-source-port>`
- topic: `xr_teleop`
- expected payload fields:
  - `timestamp`
  - `dual_arm_position`
  - `dual_hand_joints`
  - `mode`
  - `movement`
  - `facing`
  - `speed`
  - `height`
  - `toggle_data_collection`
  - `toggle_data_abort`
  - `stop`
  - optional `debug`

The dashboard treats `stop` as teleop source lifecycle metadata only. It must not display it as GEAR-SONIC shutdown.

The XR source must be a ZMQ PUB socket so the dashboard can subscribe alongside `xr_upperbody_bridge.py`. If that assumption breaks, the dashboard must remain useful from bridge and camera state alone.

### Bridge Output

Subscribe to the bridge PUB socket:

- endpoint: `tcp://<bridge-host>:<bridge-port>`
- topics:
  - `planner`
  - `manager_state`
  - `command`

Use `gear_sonic.utils.teleop.xr_upperbody_bridge.unpack_bridge_message` for `planner`, `manager_state`, and `command`.

Distinguish edge-triggered and level-like streams:

- `planner` is level-like and self-healing because it repeats.
- `manager_state` repeats, but its toggle fields are edge semantics.
- `command` is edge/once semantics. A late dashboard subscriber can miss startup `command(start=True)`.

The UI must represent edge topics with an explicit `not yet received` state. It must not infer "stopped" from missing `command` messages.

Bridge `command.stop=True` is a real planner-stop intent from the bridge process. XR JSON `stop=True` is only source lifecycle metadata. The UI must display these as different signals.

### Camera

Use `ComposedCameraClientSensor` against the existing camera server:

- endpoint: `tcp://<camera-host>:<camera-port>`
- default port: `5555`
- expected camera names:
  - `ego_view`
  - `head`
  - `left_wrist`
  - `right_wrist`

The dashboard should tolerate missing camera server or missing individual streams and show stale/no-frame states instead of exiting. "Partial" camera state means the composed camera message arrived but one or more expected image keys are absent from that message; it does not imply independent per-camera sockets or authoritative per-camera transport liveness.

Frame age must be computed from dashboard-local monotonic receive time, not camera embedded timestamps, because camera messages may originate on another machine with a different wall clock.

## UI Layout

The first screen is the actual operator dashboard, not a landing page.

### Top Status Band

Always visible:

- XR source: connected, waiting, stale
- bridge output: connected, waiting, stale
- camera: connected, partial, unavailable
- current mode
- XR export active/stale state
- inferred collection latch with "inferred" label
- last XR frame age
- last planner frame age

### Robot Vision

Large primary tile:

- prefer `ego_view` if present
- otherwise prefer `head`
- otherwise show first available camera

Secondary tiles:

- `left_wrist`
- `right_wrist`
- any additional stream returned by camera server

Each tile shows:

- camera name
- frame age
- FPS estimate
- stale/no-frame overlay

### XR Command Monitor

Show the command being produced by the headset/controller path:

- left stick movement vector as a 2D pad
- right stick yaw/facing as a compass indicator
- height command bar with stand/squat/kneel labels
- left hand aperture estimate
- right hand aperture estimate
- latest edge event:
  - collection toggle edge
  - abort/bad episode
  - teleop quit/stand-still

This panel must reuse `normalize_live_source_payload()` and `UpperBodyFrame` for interpreting XR JSON. It must not implement a parallel payload normalizer that can drift from bridge behavior.

Hand aperture is derived from the 7-DOF Dex3 joint vector, not from a boolean gripper field. V1 should show a numeric aperture/closure estimate and raw joint range. It must not claim authoritative binary open/closed state unless a later XR payload adds an explicit gripper state field.

### Bridge Transform State

Show what the bridge changes or constrains before GEAR-SONIC receives commands:

- raw XR dual-arm joint min/max/mean
- mapped SONIC upper-body joint min/max/mean
- filtered/clamped planner upper-body joint min/max/mean
- max absolute joint value
- max per-frame joint step
- left/right hand split min/max/mean
- pass-through locomotion fields (`mode`, `movement`, `facing`, `speed`, `height`) shown once, with XR vs planner mismatch warning only if they diverge
- joint clamp/stale warnings when detected

### Data Collection Panel

Display:

- collection intent: inferred idle/recording latch, explicitly marked inferred
- last collection toggle edge time
- last abort edge time
- time since last inferred record edge
- received planner frame count
- received XR frame count
- task prompt text field, local UI-only in V1

The inferred collection latch and timer can desynchronize if the dashboard starts late or misses a toggle edge. The task prompt is not written into dataset metadata in V1 unless a later data exporter integration is added.

### Safety Panel

Display high-signal warnings:

- XR stale
- bridge stale
- camera stale
- no planner frames
- no camera frames
- non-finite payload rejected
- joint range suspicious
- hand command absent
- edge state unknown because no `command` or toggle edge has been received yet

Also display this explicit notice:

> XR quit stops teleoperation export and returns commands to idle/stand-still. It does not shut down GEAR-SONIC.

Also display:

> Bridge command stop means the bridge requested planner stop. XR stop means only the XR source ended.

## State Model

Create one shared in-process state object updated by background receiver threads:

- latest XR payload
- latest decoded planner payload
- latest decoded manager state
- latest command payload
- latest camera frame per camera name, cached as image bytes and content type
- counters per source
- monotonic receive timestamps per source
- derived stale flags
- inferred edge-state latches, marked as inferred in snapshots

The HTTP handlers only read snapshots from this state object. A lock protects updates and snapshot reads.

## HTTP Endpoints

- `GET /`
  - dashboard HTML
- `GET /static/dashboard.css`
  - dashboard CSS
- `GET /static/dashboard.js`
  - dashboard JS
- `GET /api/state`
  - JSON snapshot of all non-image telemetry
- `GET /api/camera/<camera_name>`
  - latest cached color image with schema-aware content type
- `GET /api/cameras`
  - available camera names and health metadata

Polling is enough for V1:

- telemetry: 5-10 Hz
- camera images: 10-15 Hz

## Error Handling

- ZMQ subscribe timeout must not exit the dashboard.
- Missing camera server must show unavailable camera state.
- Malformed XR JSON increments an error counter and keeps last good state.
- Malformed bridge messages increment an error counter and keep last good state.
- Camera image cache/serve failure returns HTTP 503 for that frame.
- Keyboard interrupt exits cleanly and closes ZMQ/camera sockets.
- Edge-triggered state starts as `unknown` / `not yet received`, not false.

Color frames may be cached as JPEG. Depth/non-color frames must either be omitted from V1 or served with the correct content type; a `.jpg` endpoint must not be used for depth frames.

## Testing

Add focused tests for the non-browser parts:

- XR JSON snapshot normalization
- bridge planner decode into dashboard state
- command decode using `unpack_bridge_message(..., topic="command")`
- stale flag derivation
- edge-topic `not yet received` behavior
- inferred collection latch behavior and desync labeling
- missing camera behavior
- camera endpoint returns 503 when no frame is available
- read-only invariant: dashboard creates no outbound control PUB/PUSH/REQ sockets toward robot or bridge

Manual smoke test:

1. Start dashboard before the bridge and confirm edge states render as `not yet received`.
2. Start XR export-only teleop.
3. Start `xr_upperbody_bridge.py --source zmq-json`.
4. Start camera server if available.
5. Open `http://<operator-pc-ip>:8088`.
6. Verify mode, collection toggle edge, movement, height, hand aperture estimate, transform summaries, and stale warnings update while operating the controller.

## V1 Acceptance Criteria

- Dashboard runs without Node/npm.
- Dashboard opens from another device on the LAN.
- It remains useful when camera server is missing.
- It shows XR live/stale state.
- It shows bridge planner live/stale state.
- It shows current mode, collection toggle edge/inferred latch, movement, facing, speed, height, and hand aperture estimates.
- It labels collection state as inferred, not authoritative.
- It shows raw XR arm ranges, mapped SONIC upper-body ranges, and planner/filter ranges.
- It displays at least one live camera stream when the existing camera server is running.
- It does not send robot control commands.
- It does not interpret XR teleop quit as GEAR-SONIC shutdown.
- It distinguishes XR stop metadata from bridge `command.stop`.

## Later Extensions

- Browser-side operator notes saved into episode metadata.
- Exporter status topic for authoritative recording state.
- Dataset episode browser with good/bad episode tags.
- Latency timeline for XR, bridge, camera, and exporter.
- Real robot safety checklist overlay before collection.
- Optional authenticated read-only dashboard if exposed beyond local LAN.
