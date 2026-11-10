# Speech to Motion with Riva + KIMODO

Control the Unitree G1 with your **voice** — no keyboard, gamepad, VR headset, or typed text. [Riva](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/overview.html) transcribes the operator's speech, [KIMODO](https://github.com/nv-tlabs/kimodo) generates G1 joint motion directly from that text, and this example adds a safety gate that checks every joint of the generated clip against the robot's real physical limits before anything streams, plus a closed-loop monitor that watches the robot's real executed motion against what was commanded for the full duration of playback.

<figure style="margin: 1em 0; text-align: center;">
<video controls playsinline style="border-radius: 8px; max-width: 420px; width: 100%; height: auto;">
  <source src="../_static/speech_to_motion/speech_to_motion_real_robot.mp4" type="video/mp4">
</video>
<figcaption style="text-align: center; font-style: italic; margin-top: 0.5em;">Audio on: hear the command, watch the G1 execute it live.</figcaption>
</figure>

```{admonition} Prerequisites
:class: note
Complete the [Quick Start](../getting_started/quickstart.md) so the sim2sim loop runs. This tutorial drives the same interface described in [Streaming Motion Tracking](zmq.md) and [Manager Mode](manager.md) — read those pages first if you have not used `--input-type manager` before, since the keyboard controls and stream protocol are shared.
```

```{admonition} KIMODO and a speech-to-text service are external dependencies
:class: important
Neither is bundled with this repository or installed by any SONIC install script. KIMODO ([nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo), Apache-2.0) is cloned and installed in its own environment; `kimodo_server.py` runs inside that environment, not this repo's. `live_asr_client.py` needs a streaming speech-to-text source — we used [NVIDIA's ASR NIM](https://docs.nvidia.com/nim/speech/latest/asr/) (formerly branded Riva ASR NIM), deployed separately, but any streaming STT service that yields finalized transcripts works; swap it in by editing `live_asr_client.py`'s transcription loop. Step 1 below covers KIMODO and our own ASR setup with concrete commands, not just links.
```

---

## How It Works

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

| Stage | What happens | Output |
|---|---|---|
| Transcribe | Riva's streaming ASR turns speech into text, with word boosting for command vocabulary | finalized transcript |
| Orchestrate | `motion_controller.py` receives the transcript; a new command always interrupts and replaces whatever is currently playing, immediately | generation request |
| Generate | `kimodo_server.py` keeps a KIMODO model resident on the GPU and generates a motion clip from the (possibly compound) command text | G1 qpos CSV (30fps) |
| Validate | `kimodo_to_sonic.py`'s safety gate checks every joint, every frame, against the G1's real position and velocity limits, before anything is sent | pass / reject with the specific joint, frame, and value |
| Stream | Resampled to 50Hz, reordered to IsaacLab joint order, sent as a single ZMQ `pose` burst | GEAR-SONIC tracks it |
| Verify | `closed_loop_monitor.py` watches GEAR-SONIC's own real-time telemetry for the entire execution | logged tracking divergence |

### Why a safety gate *and* a closed-loop monitor

The safety gate is a pre-flight check: it validates a generated clip once, before a single frame is sent, then trusts GEAR-SONIC to execute it faithfully. That trust turned out to matter during development: for a period, every generated motion looked subtly wrong on the robot, and it looked like a KIMODO motion-quality problem. It wasn't. GEAR-SONIC's ZMQ `pose` interface expects joint data in IsaacLab order, and KIMODO's native output is MuJoCo order — see the comment above `ISAACLAB_TO_MUJOCO` in `kimodo_to_sonic.py` for how this was tracked down against two self-contradictory joint-mapping tables in `policy_parameters.hpp`. A pre-flight gate alone never catches this class of problem, because the generated motion is perfectly valid on its own terms; the mismatch only becomes visible by comparing what GEAR-SONIC actually executes against what was intended, continuously, while it's happening. That's what the closed-loop monitor adds, reusing GEAR-SONIC's own existing `g1_debug` telemetry (published whenever `--output-type all` is used) rather than requiring any new instrumentation.

### Files

Everything lives in `gear_sonic/examples/speech_to_motion/`:

| File | Role |
|---|---|
| `kimodo_server.py` | Persistent KIMODO inference server |
| `motion_controller.py` | Orchestrator: generation, safety gate, streaming, live interruption |
| `kimodo_to_sonic.py` | Safety gate + KIMODO CSV -> GEAR-SONIC ZMQ `pose` conversion; also runnable standalone for a dry run |
| `closed_loop_monitor.py` | Watches `g1_debug` telemetry during playback, logs tracking divergence (observation only) |
| `live_asr_client.py` | Microphone -> Riva streaming ASR -> `motion_controller.py` |
| `README.md` | Short reference version of this tutorial |

---

## Prerequisites

1. **SONIC deployment built and runnable** — see [Installation (Deployment)](../getting_started/installation_deploy.md) and [Quick Start](../getting_started/quickstart.md).
2. **A GPU machine for KIMODO** — a second GPU is fine if it isn't the one running the SONIC deploy; KIMODO and SONIC do not need to share a machine (point `motion_controller.py` at each with `--kimodo-host`/`--sonic-host`).
3. **A microphone** — any input device `sounddevice`/PortAudio can see.
4. **An NGC account** (free) if using NVIDIA's ASR NIM as your speech-to-text service, as this example does, and a Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct` (gated — request access on its [model page](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) first, KIMODO's text encoder needs it).

---

## Step 1 — Install KIMODO, Deploy a Speech-to-Text Service, Start Generation

### Install KIMODO

Request access to the gated text-encoder model first — [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) — then log in:

```bash
pip install --upgrade huggingface_hub   # only if the `hf` CLI isn't already installed
hf auth login   # needs a read token: https://huggingface.co/settings/tokens/new?tokenType=read
```

Then create an environment and install KIMODO (PyTorch 2.0+, matched to your own CUDA version — see [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)):

```bash
conda create -n kimodo python=3.10 -y
conda activate kimodo

# install PyTorch for your CUDA version first, per the link above, then:
pip install git+https://github.com/nv-tlabs/kimodo.git
```

This is the package-only install (sufficient for `kimodo_server.py`, which only imports the `kimodo` package). See [KIMODO's own installation docs](https://research.nvidia.com/labs/sil/projects/kimodo/docs/getting_started/installation.html) for the source-checkout, Docker, or Kimodo-SMPLX routes if you need one of those instead.

### Deploy a Streaming Speech-to-Text Service

`live_asr_client.py` needs a streaming ASR source that yields finalized transcripts over gRPC. For this example we used **NVIDIA's ASR NIM** (what used to be branded Riva ASR NIM) — follow NVIDIA's own [Run Your First NVIDIA Speech NIM Microservice for ASR](https://docs.nvidia.com/nim/speech/latest/get-started/tutorials/asr.html) tutorial to deploy it; that page covers the current container image, `NIM_TAGS_SELECTOR` model selection, and GPU/cache configuration, which change more often than this document does, so it's linked rather than copied here. Once it's up and serving a gRPC endpoint (`50051` by default), point this example at it with `--riva-server` (or `$RIVA_SERVER`).

Any other streaming speech-to-text service that can hand you finalized transcript text works too — swap it in by editing `live_asr_client.py`'s transcription loop, the rest of the pipeline (`motion_controller.py` onward) only ever sees plain text.

### Install this example's Python dependencies

In the `kimodo` conda environment from above:

```bash
pip install nvidia-riva-client sounddevice pyzmq msgpack msgpack-numpy
sudo apt install libportaudio2   # Debian/Ubuntu; sounddevice needs this system library
```

### Start the KIMODO server

```bash
conda activate kimodo
python /path/to/GR00T-WholeBodyControl/gear_sonic/examples/speech_to_motion/kimodo_server.py
```

First load takes about a minute (the model is loaded once and kept resident). Leave this running.

---

## Step 2 — Dry Run Without a Microphone or Robot

Do this before involving the robot, the same spirit as verifying a camera before a teleop session.

```bash
python -c "
import socket, json
s = socket.create_connection(('localhost', 8766), timeout=60)
s.sendall((json.dumps({'prompt': 'walk forward and wave'}) + chr(10)).encode())
print(s.recv(65536).decode())
"
```

This prints the generated clip's path (`output_csv`). Then run the safety gate directly against it:

```bash
cd gear_sonic/examples/speech_to_motion
python kimodo_to_sonic.py <output_csv path from above> --host localhost --port 5556
```

The `[SAFETY GATE] passed.` or `[SAFETY GATE] REJECTED -- <reason>` line prints before the streaming step, which is what this check is verifying — GEAR-SONIC doesn't need to be running yet for this part.

---

## Step 3 — Run in Simulation

Run **four terminals**.

```bash
# Terminal 1 (repo root): MuJoCo simulator
python gear_sonic/scripts/run_sim_loop.py

# Terminal 2 (gear_sonic_deploy/): deploy
bash deploy.sh --input-type manager --output-type all --zmq-host localhost sim
```

In Terminal 2: press `]` to start control, `9` in the MuJoCo window to drop the robot, then `#` and `ENTER` to switch into ZMQ streaming mode.

```bash
# Terminal 3: orchestrator (safety gate, live interrupt, closed-loop monitor)
cd gear_sonic/examples/speech_to_motion
python motion_controller.py

# Terminal 4: live voice
python live_asr_client.py
```

Speak a command, e.g. "walk forward, turn left, and wave." Watch the robot.

```{admonition} Voice is always listening
:class: note
Unlike the keyboard/gamepad input paths, there is no explicit mode toggle once `live_asr_client.py` is running — every finalized transcript above a minimum length is sent. A new command always interrupts and replaces whatever is currently playing, immediately; it does not queue behind the current motion.
```

---

## Step 4 — Real Hardware

Same four terminals, with Terminal 2's deploy target changed from `sim` to `real` (see [Streaming Motion Tracking](zmq.md) for interface auto-detection details).

```{admonition} Danger — First Hardware Run
:class: danger
Keep a safety operator on the E-stop for the entire session. Start with a single low-risk spoken command (e.g. "wave" or "turn left"), **not** a walking command, before attempting anything more ambitious. Confirm the robot behaves as expected at rest and on a simple command before moving on.
```

---

## Controls

| Key / Action | Effect |
|---|---|
| `]` | Start control system (Terminal 2) |
| `9` (sim only) | Drop the robot in MuJoCo |
| `#` then `ENTER` | Switch into ZMQ streaming mode |
| `O` | Emergency stop |
| *(speak)* | Send a new command — always interrupts whatever is currently playing |

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `[SAFETY GATE] REJECTED -- position limit violated: ...` | The generated clip genuinely exceeds a joint's real range of motion. This is the gate working as intended — nothing is sent, the robot holds its pose. Try a simpler or shorter command; if it keeps happening on a specific phrase, that's worth reporting, not working around by loosening the gate. |
| `[SAFETY GATE] REJECTED -- velocity limit violated: ...` | A joint moves faster than `JOINT_VELOCITY_LIMIT_RAD_S` in `kimodo_to_sonic.py` between two frames. Often a KIMODO artifact on compound commands with several fast gestures stacked close together — try splitting it into two spoken commands instead of one long one. |
| `Could not reach motion controller at (...)` | `motion_controller.py` isn't running, or `--controller-host`/`--controller-port` don't match where it's actually listening. |
| Command transcribes correctly but the robot never moves | Check `]` → `9` (sim) → `#` → `ENTER` was completed in the deploy terminal — look for `ZMQ STREAMING MODE: ENABLED` in its log. Receiving and parsing pose data (visible in the deploy log) is not the same as streaming mode being armed. |
| ASR consistently mishears a specific command word | Add it to `BOOSTED_WORDS` in `live_asr_client.py`, or add a `KNOWN_CORRECTIONS` entry if it's a consistent, specific substitution (e.g. a homophone) rather than a general accuracy problem. |
| `OSError: PortAudio library not found` | The system library is missing, not the Python package — `sudo apt install libportaudio2` (Debian/Ubuntu) or the equivalent for your OS, then retry. |
| `kimodo_server.py` generation is very slow | Expected to take a few seconds per command (see Notes below) — this is KIMODO's diffusion sampling, not a bug. If it's taking much longer, confirm it's actually running on GPU (`torch.cuda.is_available()` should be `True` in the KIMODO environment). |

---

## Notes / Limitations

- **Latency is generation-dominated.** Measured: roughly 7.5s from end of speech to first streamed frame, almost all of it KIMODO's diffusion sampling. The safety gate and ZMQ streaming together add on the order of 0.01s.
- **The safety-gate velocity threshold is empirical, not from an official spec** — no public per-joint velocity limit exists for the G1. See the comment above `JOINT_VELOCITY_LIMIT_RAD_S` in `kimodo_to_sonic.py` for the measured data points it was set from, and revisit it against real motor datasheets before relying on it for unsupervised operation.
- **The closed-loop monitor is observation-only.** It logs real tracking divergence but does not yet trigger any automatic stop or corrective action. A stable, normal walk can legitimately show close to 0.7 rad of peak divergence on a leg joint without anything being wrong — that's real measured data, not a guess, and it's also why no hard action threshold is wired in yet: there isn't enough measured data across motion types to set one responsibly.
