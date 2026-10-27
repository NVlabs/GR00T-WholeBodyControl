# Inspire hand support

Inspire RH56DFX support for MuJoCo and serial hardware, integrated with the
existing deployment entry points. SONIC controls the G1 body; a separate stream
controls six finger channels per hand. Omitting `--hand` preserves the original
Dex3 deployment.

VLA, PICO optical retargeting and dataset recording/replay are outside this
Inspire integration. The upstream inference and PICO entry points are unchanged.

For the file layout and instructions for adding another hand, see
[Adding a hand](../../../gear_sonic/utils/hand_control/README.md).
Each implementation directory contains a short overview; this guide owns the
execution instructions.

Use the existing `gear_sonic_deploy/deploy.sh` for both hands. Model selection,
network selection, build, deployment confirmation and body keyboard controls
use the same path. Leave out `--hand` for Dex3, or add `--hand inspire` for Inspire.
The launcher previews the exact command it will execute, including the Inspire
communication supervisor when selected; no separate Python deployment command
is needed.

## Setup

Complete the existing [deployment setup](../getting_started/installation_deploy.md)
and [model download](../getting_started/download_models.md).
For the simulation Python environment, use the existing installer from the
repository root. Run it once when setting up a checkout; it recreates `.venv_sim`:

```bash
bash install_scripts/install_mujoco_sim.sh
source .venv_sim/bin/activate
git lfs pull --include="gear_sonic/data/robots/g1/meshes/**,gear_sonic_deploy/reference/example/**" --exclude=""
```

Reference CSVs must contain numeric motion data, not Git LFS pointer text.
Keep planner models in their downloaded directory layout, including `V2`.
The short deployment command below uses the existing launcher defaults:

```text
gear_sonic_deploy/
├── policy/release/
│   ├── model_encoder.onnx
│   ├── model_decoder.onnx
│   └── observation_config.yaml
├── planner/target_vel/V2/planner_sonic.onnx
└── reference/example/
```

The environment and model binaries are local, ignored files. Do not commit
virtual environments, generated TensorRT engines or machine-specific paths.
Custom model locations can still be supplied with the upstream `--cp`,
`--obs-config`, `--planner` and `--motion-data` options.

## Full-body simulation

In each new terminal, start at the repository root and activate the environment
once with `source .venv_sim/bin/activate`. The launcher uses its `python3` by
default; `--hand-python` is only needed to select a different interpreter.
Then start these in separate terminals on the computer:

```bash
# Terminal 1, repository root: one shared body-and-hands world.
python gear_sonic/scripts/run_sim_loop.py --interface sim --hand inspire

# Terminal 2, repository root: SONIC body controller.
cd gear_sonic_deploy
./deploy.sh sim --hand inspire --input-type keyboard --output-type zmq
```

After `Init Done`, press `]` in the controller terminal to start body control.
Wait for `transitioning to CONTROL state`, then press Enter in that same terminal
to enable the idle planner. Keep the support band attached until
`[Keyboard] motion name is planner_motion` appears; allow three seconds for the
transition, then press `9` once in the MuJoCo window to release the band. Check
that the robot is standing before sending hand targets. Releasing the band
before switching to the planner can cause a fall during the time spent switching
windows. Do not enable the planner before starting control.

Send a small hand trajectory from a third terminal at the repository root:

```bash
python -m gear_sonic.scripts.inspire_hand_example --backend sim-remote --demo
```

Without `--demo`, the example only reads feedback. `--backend sim` instead uses
a standalone hand test scene with a fixed body. For custom endpoints, pass the
same configuration via `--hand-config` to the simulator and deployment launcher,
and via `--config` to the example.

## Hardware: computer controller, robot serial bridge

```text
Computer: SONIC + models ── DDS ──────────────────────> G1 body
          hand client ── ZMQ through SSH ──> Robot: C++ bridge ── serial ──> hands
```

The robot needs only the bridge executable, configuration and launcher. No
Python, ROS, Unitree SDK or model files are required by the bridge. Building
requires CMake, a C++20 compiler, libzmq/cppzmq and MessagePack headers; runtime
uses libzmq and the standard C++ libraries. Reuse installed dependencies.

For a wired setup, connect the computer to the G1 development Ethernet port and
give that computer interface an unused address in the robot's subnet (commonly
`192.168.123.0/24`). This connection does not need a default gateway. Use the
onboard computer's Ethernet address for SSH and verify that `rt/lowstate` is
received on the selected computer interface before starting control. The same
cable carries body DDS, SSH and the hand tunnel; connecting to the robot's Wi-Fi
is not required. Stop other body controllers and hand drivers before this run.

On the computer, package the existing driver source and copy it to a fresh
robot directory. Replace `USER@ROBOT` with your SSH destination:

```bash
python gear_sonic_deploy/src/inspire_hand_bridge/bridge/package.py /tmp/inspire-bridge.tar.gz
ssh USER@ROBOT 'mkdir -p ~/inspire_hand_bridge'
scp /tmp/inspire-bridge.tar.gz USER@ROBOT:~/inspire_hand_bridge/
```

On the robot, build entirely inside that directory:

```bash
cd ~/inspire_hand_bridge
mkdir .build
tar -xzf inspire-bridge.tar.gz -C .build
TMPDIR="$PWD/.build" cmake -S .build/source -B .build/build -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
TMPDIR="$PWD/.build" cmake --build .build/build --target inspire_direct_serial_hand_gateway -j2
install -m 755 .build/build/inspire_direct_serial_hand_gateway ./inspire_hand_bridge
install -m 755 .build/run.sh ./run.sh
cp .build/bridge.env.example ./bridge.env
cp .build/LICENSE .build/LICENSE-*.txt .build/NOTICE.md .build/source-manifest.json .
```

Edit `bridge.env` to match the two local serial devices, IDs and baud rate.
Defaults are 50 Hz, 57600 baud and device ID 1 per hand. Use plain `KEY=value`
without shell quotes. The generated `.build` and archive are not needed at runtime.

```bash
./run.sh --check  # Validate paths/configuration without opening serial devices.
./run.sh          # Start in foreground; sets and verifies the device speed.
```

On the computer, keep this tunnel running in a separate terminal:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:15561:127.0.0.1:5561 \
  -L 127.0.0.1:15562:127.0.0.1:5562 USER@ROBOT
```

Copy [hand_config.yaml](../../../gear_sonic/data/robots/g1/inspire/hand_config.yaml) to a computer-side configuration. Under
`hardware`, change only these endpoints; local device paths may remain null:

```yaml
command_endpoint: tcp://127.0.0.1:15561
state_endpoint: tcp://127.0.0.1:15562
```

Read feedback, then start SONIC on the computer:

```bash
python -m gear_sonic.scripts.inspire_hand_example --backend real --config /path/to/hand.yaml
cd gear_sonic_deploy
./deploy.sh COMPUTER_DDS_INTERFACE --hand inspire --hand-remote \
  --hand-config /path/to/hand.yaml --input-type keyboard --output-type zmq
```

Use the computer's interface connected to the robot's DDS network. The hand SSH
tunnel does not carry body DDS; successful Wi-Fi/SSH access alone is insufficient.
Follow the controller startup order above, without the simulator support-band key.
The launcher requires healthy, disarmed hand feedback before starting the body.
It monitors feedback freshness and gateway identity but does not start or stop
the remote bridge. `--hand-remote` cannot be used with sim, Dex3 or `--hand-driver`.

If body control and hand serial devices are on the same machine, omit
`--hand-remote`, configure the local device paths in YAML, and build the driver
with `cmake -S gear_sonic_deploy/src/inspire_hand_bridge -B build/inspire-native`
and `cmake --build build/inspire-native`. The launcher then owns that driver.

## Hand commands and stopping

The Inspire adapters share [HandBackend](../../../gear_sonic/utils/hand_control/interface.py):
`connect()`, `set_target(left, right)`, `read_state()`, `stop()` and `close()`.
This API is optional for other hand integrations; see the
[Dex3/Inspire integration examples](../../../gear_sonic/utils/hand_control/README.md).
Use `gear_sonic.utils.hand_control.inspire.client.create_backend(config_path)`
for hardware or
`gear_sonic.utils.mujoco_sim.inspire.backend.create_remote_backend(config_path)`
for the shared simulation.

Each target has six angles in radians, ordered
`pinky, ring, middle, index, thumb_bend, thumb_rotation`. Bounds are
`[0, 1.7/1.7/1.7/1.7/0.6/1.3]`; velocity limits are `[2, 2, 2, 2, 1, 1]` rad/s.
Start near measured feedback and ramp within these limits. The first valid target
acquires the session automatically; manual RESET/ARM commands are unnecessary.
`native6` in diagnostic output names the six-channel hand protocol, not a
separate program to start. ARM/DISARM are internal session states handled by
the client; the example acquires ownership on its first target and releases it
on exit.
The example accepts explicit `--left` and `--right` targets with `--duration`;
`--demo` is simulation-only.

To check every finger while the body remains under SONIC control, run this
explicit sequence from another computer terminal after the idle planner is ready:

```bash
# Hardware; the bridge, tunnel and body controller must already be running.
python -m gear_sonic.scripts.inspire_hand_example --backend real \
  --config /path/to/hand.yaml --sequence

# Shared full-body simulation uses the same sequence.
python -m gear_sonic.scripts.inspire_hand_example --backend sim-remote --sequence
```

The sequence closes then opens the left thumb, index, middle, ring and pinky,
then repeats on the right hand. Only the selected finger's targets change.
It starts from measured positions, uses 80% of the configured joint velocity
limits. Each stage publishes for `--duration` (default 2 seconds), or for the
full transition if longer, holding its target after the ramp. Measurements are
recorded without adding an arrival threshold. A full four-finger joint
stroke takes about 1.06 seconds each way; thumb bending takes about 0.75 seconds.
Thumb closure uses bend `0.6` and rotation `0.3` rad to keep clear of the closed
index, rather than the rotation joint's full `1.3` rad limit. On completion the
example disarms the hands. It does not stop the separately launched body: press
`O` in the body terminal, then stop the bridge. No sequence runs automatically
when `--hand inspire` is selected.

Add `--log-file hand-sequence.jsonl` to save the sequence from the example.
The file must be new. Each submitted target is recorded with its stage, local
submission time, measured left/right positions and local feedback receipt time.
Repeated feedback keeps its original receipt time; a completed stage means its
publication interval ended, not that every joint physically reached the target.
For body data, the existing ZMQ output provides `last_action` (body position
commands) and `body_q` (measured joints), both in MuJoCo joint order;
`body_q_target` is the reference motion, not the motor command.
Body CSV recording is off by default. Add `--enable-csv-logs` to `deploy.sh`
to save the existing body state logs; optionally use `--logs-dir /path/to/logs`
to select the output directory. This does not enable or change finger commands.

The target-producing client must call `stop()`/`close()` and confirm DISARM
before exit. Stop the body with `O`, then the foreground bridge with Ctrl+C.
For a local bridge started by `deploy.sh`, the launcher stops its owned body
and bridge on exit, interruption or startup failure. With `--hand-remote`, the
independently started bridge remains running; close the hand client first and
stop that bridge in its own terminal. The simulation process also remains
separate, as in Dex3 deployment.
`close()` releases the connection; it does not close the fingers. Hardware stop
requests a measured-position hold; normal driver exit restores device speed.
The body's stop path applies damping, not active balance.

**Hardware command-stream timeout is disabled:** lost commands or a broken
network can leave the last target active; remote DISARM cannot be guaranteed
when communication is lost. Feedback faults inhibit writes and remain latched.
Simulation instead zeros hand control after 200 ms without a command. The 50 Hz
hardware setting is a target rate; serial overruns are reported without catch-up
bursts. No preset hand opening/closing is sent at startup or shutdown.

## Validation

```bash
ctest --test-dir build/inspire-native --output-on-failure
INSPIRE_TEST_GATEWAY="$PWD/build/inspire-native/inspire_direct_serial_hand_gateway" \
  python -m pytest -q gear_sonic/tests/test_inspire_*.py
```

Offline checks cover session ownership, measured feedback, sequence logging,
serial/process emulation and deployment entry points. The test suite passed
74 Python checks and seven C++ checks.

Full-body MuJoCo GUI validation exercised `deploy.sh`, default endpoints and
pauses between operator actions. The robot completed the 20-stage close/open
sequence while standing, followed by confirmed hand DISARM and normal body
Stop. Use the planner-before-band-release startup order above.

Hardware validation used a 29-DoF G1 with Inspire hands, computer-side SONIC
over wired DDS, and the serial bridge on the onboard computer. The operator
confirmed standing, both complete finger sequences and normal shutdown. Two
recorded runs each contain 20 stages, 2,000 hand samples over approximately
40 seconds, and 1,999 concurrent body position/action frames. Target submission
was approximately 49.99 Hz; this is not a measurement of serial feedback rate.
Stage completion records the end of publication, not a joint-arrival tolerance
check. These trials validate the tested standing and finger sequence, not
locomotion or the full sim-to-real motion envelope.

Model provenance is in [ASSETS.md](../../../gear_sonic/data/robots/g1/inspire/ASSETS.md);
retained credits and licenses are in [third-party notices](../../../legal/NOTICE-inspire-hands.md).
