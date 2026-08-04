# Model Card

SONIC provides two released whole-body controller checkpoints for the Unitree
G1. Choose the model based on whether you want the original general-purpose
controller or reduced reference lookahead for teleoperation.

## Available Models

| Model | Hugging Face location | SMPL reference input | Intended use and comments |
|---|---|---|---|
| **Default SONIC (original release)** | Top-level `model_encoder.onnx`, `model_decoder.onnx`, and `observation_config.yaml`; training checkpoint at `sonic_release/last.pt` | 10 future frames at 20 ms spacing, approximately 200 ms of reference lookahead | Default general-purpose SONIC controller for motion tracking, planning, teleoperation, and compatibility with existing deployments. G1 and teleoperation future-reference observations use `step5`. |
| **Low-latency teleoperation** | [`low_latency/`](https://huggingface.co/nvidia/GEAR-SONIC/tree/main/low_latency) | 4 future frames at 20 ms spacing, approximately 80 ms of reference lookahead | Intended for more responsive whole-body teleoperation and VLA execution. G1 and teleoperation future-reference observations use `step1`. Use its encoder, decoder, and observation config together. |

Both models use the SONIC universal-token controller, produce 64-dimensional
latent motion tokens, run the controller at 50 Hz, and support SMPL pose, G1
motion reference, and VR 3-point inputs. Deployment uses C++ and TensorRT; the
PyTorch checkpoints support Isaac Lab evaluation and continued training.

```{note}
The lookahead values describe the reference horizon presented to the
controller. They are not measurements of total end-to-end teleoperation
latency, which also includes sensing, networking, preprocessing, and inference.
```

## Released Files

| Model | Deployment files | PyTorch and configuration files |
|---|---|---|
| Default SONIC | `model_encoder.onnx`, `model_decoder.onnx`, `observation_config.yaml` | `sonic_release/last.pt`, `sonic_release/config.yaml` |
| Low-latency teleoperation | `low_latency/model_encoder.onnx`, `low_latency/model_decoder.onnx`, `low_latency/observation_config.yaml` | `low_latency/last.pt`, `low_latency/config.yaml`, `low_latency/model_config.yaml` |

All files are hosted in
[`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC). Model weights
are covered by the [NVIDIA Open Model License](resources/license.md).

## Choosing a Model

Use **Default SONIC** when you want the original release, the broadest
compatibility with existing deployment setups, or the standard motion-tracking
and planning controller.

Use **Low-latency teleoperation** when responsiveness to streamed SMPL, VR, or
VLA commands is the priority. Its shorter reference horizon reduces commanded
motion lookahead, but it does not remove latency elsewhere in the system.

## Usage

Install the Hugging Face dependency from the repository root:

```bash
pip install huggingface_hub
```

### Default SONIC

```bash
python download_from_hf.py

cd gear_sonic_deploy
./deploy.sh --input-type zmq_manager real
```

### Low-Latency Teleoperation

```bash
python download_from_hf.py --low-latency

cd gear_sonic_deploy
./deploy.sh \
    --cp policy/low_latency/model \
    --obs-config policy/low_latency/observation_config.yaml \
    --input-type zmq_manager \
    real
```

### Python VLA Launcher

For the default model:

```bash
python gear_sonic/scripts/launch_inference.py \
    --camera-host 192.168.123.164 \
    --prompt "pick up the cup"
```

For the low-latency model:

```bash
python gear_sonic/scripts/launch_inference.py \
    --deploy-checkpoint policy/low_latency/model \
    --deploy-obs-config policy/low_latency/observation_config.yaml \
    --camera-host 192.168.123.164 \
    --prompt "pick up the cup"
```

See [Downloading Model Checkpoints](getting_started/download_models.md) for
PyTorch checkpoint evaluation and additional download options.

## Limitations and Safety

- The low-latency name refers to reduced controller reference lookahead, not a
  benchmark of total system latency.
- Each ONNX encoder and decoder must be used with its matching observation
  configuration.
- These checkpoints target the Unitree G1 embodiment.
- Test in simulation before deployment and keep a safety operator ready to
  stop a physical robot.

