# Speech to Motion with Riva + KIMODO

Control the Unitree G1 with your voice: speak a sentence, and the robot does it.
No keyboard, gamepad, VR headset, or typed text.

[Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/overview.html)
transcribes the operator's speech, [KIMODO](https://github.com/nv-tlabs/kimodo)
generates G1 joint motion from that text, a safety gate checks every joint of
the generated clip against the robot's real limits before anything streams,
and a closed-loop monitor watches the robot's real executed motion against
what was commanded for the duration of playback.

> KIMODO and a streaming speech-to-text service are **external dependencies**
> and are **not** bundled here. `live_asr_client.py` needs a streaming STT
> source — we used NVIDIA's ASR NIM (formerly branded Riva ASR NIM), deployed
> separately; point this example at its gRPC endpoint with `--riva-server`
> (or `$RIVA_SERVER`). Any other streaming STT service that yields finalized
> transcripts works, swap it in by editing `live_asr_client.py`'s
> transcription loop. KIMODO is cloned and installed separately in its own
> environment; `kimodo_server.py` runs inside that environment.

## How it works

| Script | Role |
|---|---|
| `kimodo_server.py` | Persistent KIMODO inference server (loads the model once, serves generation over TCP) |
| `motion_controller.py` | Orchestrator: receives a transcript, generates via `kimodo_server.py`, safety-checks, streams to GEAR-SONIC, handles live interruption |
| `kimodo_to_sonic.py` | The safety gate and the KIMODO CSV -> GEAR-SONIC ZMQ `pose` conversion, importable and also runnable standalone for a dry run |
| `closed_loop_monitor.py` | Watches GEAR-SONIC's own `g1_debug` telemetry during playback and logs tracking divergence (observation only, no automatic action) |
| `live_asr_client.py` | Microphone -> Riva streaming ASR -> `motion_controller.py` |

```text
  Microphone ──audio──►  Riva ASR (streaming)
                              │ transcript
                              ▼
                     motion_controller.py
                              │ prompt
                              ▼
                       kimodo_server.py
                     (KIMODO text -> motion)
                              │ qpos CSV
                              ▼
                   kimodo_to_sonic.py: safety gate
                   (29 joints x every frame, position + velocity,
                    against the G1's real limits)
                              │ fails -> nothing sent, robot holds pose
                              │ passes
                              ▼
                  ZMQ Protocol v1, topic "pose"
                   tcp://<sonic-host>:5556
                              ▼
                      GEAR-SONIC deployment
                              │
                              ▼
                       Unitree G1 (sim or real)
                              │
                              │  g1_debug telemetry (existing, unmodified)
                              ▼
                     closed_loop_monitor.py
              (body_q vs body_q_target, logged live)
```

### Why a safety gate *and* a closed-loop monitor

The safety gate is a pre-flight check: it validates a generated clip once,
before a single frame is sent, then trusts GEAR-SONIC to execute it
faithfully. That trust turned out to matter: for a period during development,
every generated motion looked subtly wrong on the robot, and it looked like a
KIMODO motion-quality problem. It wasn't — GEAR-SONIC's ZMQ `pose` interface
expects joint data in IsaacLab order, and KIMODO's native output is MuJoCo
order (see the `ISAACLAB_TO_MUJOCO` comment in `kimodo_to_sonic.py` for how
this was tracked down against the two self-contradictory mapping tables in
`policy_parameters.hpp`). A pre-flight gate alone never catches this kind of
problem, because the generated motion is perfectly valid on its own terms; it
only becomes visible by comparing what GEAR-SONIC actually executes against
what was intended, continuously, while it's happening. That's what
`closed_loop_monitor.py` adds.

## Prerequisites

1. **SONIC deployment** built and runnable (see the repo Quick Start).
2. **A GPU machine for KIMODO** — can be a different machine from the SONIC
   deploy (point `motion_controller.py` at each with `--kimodo-host`/
   `--sonic-host`).
3. **An NGC account** (free) if using NVIDIA's ASR NIM as your speech-to-text
   service, as this example does, and a **Hugging Face account with access
   to `meta-llama/Meta-Llama-3-8B-Instruct`** (gated — request access on its
   model page first; KIMODO's text encoder needs it).

See the [full tutorial](../../../docs/source/tutorials/speech_to_motion.md) for KIMODO install commands and a link to NVIDIA's own ASR NIM deployment tutorial, and [KIMODO's own installation docs](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html) for the source-checkout, Docker, or Kimodo-SMPLX routes. KIMODO package-install short version:

```bash
# KIMODO (request HF access to meta-llama/Meta-Llama-3-8B-Instruct first)
pip install --upgrade huggingface_hub && hf auth login
conda create -n kimodo python=3.10 -y && conda activate kimodo
# install PyTorch for your CUDA version first (pytorch.org/get-started/locally), then:
pip install git+https://github.com/nv-tlabs/kimodo.git

# this example's own deps, in the kimodo conda env
pip install nvidia-riva-client sounddevice pyzmq msgpack msgpack-numpy
sudo apt install libportaudio2   # Debian/Ubuntu; system lib sounddevice needs
```

## Setup verification

Verify the pipeline without a microphone or a robot first.

### 1. Start `kimodo_server.py`

```bash
conda activate kimodo
python /path/to/GR00T-WholeBodyControl/gear_sonic/examples/speech_to_motion/kimodo_server.py
```

### 2. Dry-run generate and safety-check one clip (no robot)

```bash
python -c "
import socket, json
s = socket.create_connection(('localhost', 8766), timeout=60)
s.sendall((json.dumps({'prompt': 'walk forward and wave'}) + chr(10)).encode())
print(s.recv(65536).decode())
"
```

Then validate the generated CSV against the safety gate directly:

```bash
python kimodo_to_sonic.py <path from output_csv above> --host localhost --port 5556
```

If GEAR-SONIC isn't running yet, this will fail to connect at the streaming
step, but the `[SAFETY GATE] passed.` / `[SAFETY GATE] REJECTED` line prints
before that, which is what this step is checking.

## Usage

Run the SONIC deploy in manager mode first (sim shown; use `real` for hardware):

```bash
# terminal 1 (repo root): MuJoCo sim
python gear_sonic/scripts/run_sim_loop.py
# terminal 2 (gear_sonic_deploy/): deploy
bash deploy.sh --input-type manager --output-type all --zmq-host localhost sim
```

On the deploy side: press `]` to start, drop the robot (`9` in MuJoCo), `#`
then `ENTER` to switch into ZMQ streaming mode. Use `O` for emergency stop.
**On real hardware, keep a safety operator on the E-stop and start with a
single low-risk command (e.g. a wave), not a walking command, for the first
run.**

```bash
# terminal 3: the orchestrator (safety gate, live interrupt, closed-loop monitor)
python motion_controller.py

# terminal 4: live voice
python live_asr_client.py
```

Speak a command, e.g. "walk forward, turn left, and wave." Unlike the
keyboard/gamepad paths, voice is always listening once `live_asr_client.py`
is running — there's no explicit mode toggle. A new command always replaces
whatever is currently playing immediately; it does not queue.

## Key options

- `motion_controller.py --kimodo-host` / `--sonic-host`: point the three
  services at different machines if they aren't co-located.
- `kimodo_to_sonic.py --force-identity-quat`: diagnostic flag to isolate
  whether orientation data or joint-angle data is responsible for unexpected
  robot behavior.
- `closed_loop_monitor.py` can also be run standalone (without
  `motion_controller.py`) against any already-running GEAR-SONIC session, to
  watch tracking divergence under any input source, not just this one.

## Notes / limitations

- **Latency is generation-dominated.** Measured: roughly 7.5s from end of
  speech to first streamed frame, almost all of it KIMODO's diffusion
  sampling. The safety gate and ZMQ streaming together add on the order of
  0.01s.
- **The safety-gate velocity threshold is empirical, not from an official
  spec** — no public per-joint velocity limit exists for the G1. See the
  comment above `JOINT_VELOCITY_LIMIT_RAD_S` in `kimodo_to_sonic.py` for the
  measured data points it was set from.
- **The closed-loop monitor is observation-only.** It logs real tracking
  divergence but does not yet trigger any automatic stop/corrective action.
  A stable, normal walk can legitimately show close to 0.7 rad of peak
  divergence on a leg joint without anything being wrong, that's real
  measured data, not a guess, and it's also why no hard action threshold is
  wired in yet: there isn't enough measured data across motion types to set
  one responsibly.

See the [full tutorial](../../../docs/source/tutorials/speech_to_motion.md#troubleshooting) for a troubleshooting table (safety gate rejections, ASR mishears, PortAudio errors, and more).
