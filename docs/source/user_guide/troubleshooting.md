# Troubleshooting

Common issues and solutions. If your problem isn't listed here, check the
[GitHub issues](https://github.com/NVlabs/GR00T-WholeBodyControl/issues) page.

---

## 1. `ModuleNotFoundError: No module named 'isaaclab'`

**Symptom:** Training or eval script exits immediately with an import error.

**Cause:** Isaac Lab is not installed, or you're running in the wrong Python
environment. Isaac Lab is not a pip dependency — it must be installed separately.

**Fix:**

1. Install Isaac Lab following the
   [official guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
2. Make sure you activate the correct conda/venv environment before running:
   ```bash
   conda activate env_isaaclab  # or whatever you named it
   python -c "import isaaclab; print(isaaclab.__version__)"
   ```

---

## 2. Mesh files are tiny text files (Git LFS not installed)

**Symptom:** Simulation crashes or renders an invisible/broken robot. Mesh files
(`.stl`, `.STL`) are ~130 bytes and contain text like `version https://git-lfs.github.com/spec/v1`.

**Cause:** The repo was cloned without Git LFS. Large files (meshes, ONNX models)
are stored via Git LFS and need to be fetched separately.

**Fix:**

```bash
sudo apt install git-lfs
git lfs install
git lfs pull
```

Verify: `ls -la gear_sonic/data/assets/robot_description/urdf/g1/main.urdf` should
be ~60KB+, not ~130 bytes.

---

## 3. `RuntimeError: size mismatch` when loading a checkpoint

**Symptom:** Training or eval crashes with errors like:
```
size mismatch for actor_module.decoders.g1_dyn.module.0.weight:
  copying a param with shape torch.Size([2048, 994]) from checkpoint,
  the shape in current model is torch.Size([4096, 994])
```

**Cause:** The experiment config defines a different network architecture than
what the checkpoint was trained with. Common when the config overrides
`hidden_dims` to a different size.

**Fix:** Make sure the experiment config matches the checkpoint's architecture.
Check the `config.yaml` saved alongside the checkpoint for the correct
`hidden_dims`, encoder/decoder settings, etc. The released `sonic_release`
checkpoint uses:

```yaml
decoders:
  g1_dyn:
    params:
      module_config_dict:
        layer_config:
          hidden_dims: [2048, 2048, 1024, 1024, 512, 512]
```

---

## 4. `trl` / `transformers` version conflict during pip install

**Symptom:** `pip install -e "gear_sonic/[training]"` fails with a dependency
resolution error about incompatible `transformers` versions.

**Cause:** `trl==0.28.0` requires `transformers>=4.56.2`. If you have an older
`transformers` pinned or installed, pip cannot resolve.

**Fix:**

```bash
pip install -e "gear_sonic/[training]" --upgrade
```

Or install in a fresh environment. If you need a specific `transformers` version
for another project, use a separate venv for SONIC training.

---

## 5. TensorRT build fails (`TensorRT_ROOT` not set)

**Symptom:** CMake error during C++ deployment build:
```
Could not find a package configuration file provided by "TensorRT"
```

**Cause:** The `TensorRT_ROOT` environment variable is not set, or TensorRT is
not installed.

**Fix:**

1. Download the correct TensorRT version (TAR package, not DEB):

   | Platform | TensorRT Version |
   |---|---|
   | x86_64 (Desktop) | **10.13** (required) |
   | Jetson / G1 onboard Orin | **10.7** (required; JetPack 6) |

2. Extract and set the environment variable:
   ```bash
   export TensorRT_ROOT=$HOME/TensorRT
   echo 'export TensorRT_ROOT=$HOME/TensorRT' >> ~/.bashrc
   ```

---

## 6. Motion file path errors (`FileNotFoundError` or empty motion library)

**Symptom:** Training crashes with `FileNotFoundError` on a motion path, or starts
but logs `0 motions loaded`.

**Cause:** The experiment config has placeholder paths (e.g.,
`data/motion_lib_bones_seed/robot_filtered`) that don't exist on your machine.
Motion data paths must be provided on the command line.

**Fix:** Always pass motion data paths explicitly:

```bash
python gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=<path/to/robot_filtered> \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=<path/to/smpl_filtered>
```

For quick testing, download the sample data from HuggingFace:

```bash
hf download nvidia/GEAR-SONIC --include "sample_data/*" --local-dir .
```

---

## 7. Body name errors (`RuntimeError: body 'xxx' not found`)

**Symptom:** Isaac Lab crashes with an error about a body/joint name not found
in the robot's articulation.

**Cause:** A config YAML references a body name that doesn't exist on your robot.
This commonly happens when using G1 configs with a different robot (e.g., H2).

**Fix:** Check which body name failed and find where it's referenced:

```bash
grep -rn "the_failing_body_name" gear_sonic/config/
```

Override the body name in your experiment config, or check the
[Training on New Embodiments](new_embodiments.md) guide for the full list of
config files that reference body names.

---

## 8. Robot explodes or falls immediately on first frame

**Symptom:** The robot ragdolls, flies away, or collapses instantly when
simulation starts.

**Cause:** Usually one of:

- **Init state height is wrong** — the robot spawns inside the ground or too high.
  Check `init_state.pos` in your robot config (the z-value is spawn height).
- **KP/KD values are wrong** — if stiffness (KP) is too low, joints have no
  holding torque. If too high, the simulation becomes unstable. See
  [Training on New Embodiments](new_embodiments.md) for tuning guidance.
- **Action scale is too large** — the policy outputs move joints too aggressively.
  Reduce `action_scale` values.
- **Default joint angles are wrong** — the robot starts in an impossible pose.
  Check `init_state.joint_pos` matches a stable standing configuration.

**Debug:** Run with `num_envs=1 headless=False` and watch the first few frames.

---

## 9. Robot behaves weirdly during deployment (wrong TensorRT version)

**Symptom:** The robot stands but moves erratically, drifts, or produces
unnatural motions during C++ deployment — even though the same checkpoint works
correctly in Isaac Lab or MuJoCo simulation.

**Cause:** You are using a different TensorRT version than required. TensorRT
version mismatches produce **silently wrong inference results** — the model runs
without errors but outputs incorrect actions.

**Fix:** You **must** use the exact TensorRT versions:

| Platform | Required Version |
|---|---|
| x86_64 (Desktop) | **TensorRT 10.13** |
| Jetson / G1 onboard Orin | **TensorRT 10.7** (JetPack 6) |

Verify your version:

```bash
echo $TensorRT_ROOT
ls $TensorRT_ROOT/lib/libnvinfer.so*
```

If the version is wrong, download the correct one from
[NVIDIA Developer](https://developer.nvidia.com/tensorrt/download/10x) and
rebuild the C++ deployment binary.

---

## 10. `ChannelFactory create domain error` in MuJoCo sim

**Symptom:** `run_sim_loop.py` crashes with:
```
[ChannelFactory] create domain error. msg: Occurred upon initialisation
of a cyclonedds.domain.Domain
```

**Cause:** CycloneDDS domain initialization conflict. The SimulatorFactory
reinitializes a channel that was already created.

**Fix:** This is a known issue ([#77](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/77)).
Workaround: comment out the duplicate channel init in the simulator factory,
or ensure no other DDS process is using the same domain on your machine.

---

## 11. SMPL tracking is unstable or drifts

**Symptom:** The robot follows G1 motion tracking well but drifts or becomes
unstable when using SMPL encoder inputs.

**Cause:** SMPL data may have mismatched coordinate conventions (y-up vs z-up),
incorrect joint ordering, or the SMPL-to-robot retargeting quality is poor.

**Fix:**

- Verify `smpl_y_up: true` is set in your config if your SMPL data uses y-up
  coordinates.
- Check that the SMPL PKL files have the correct shape: `smpl_joints` should be
  `(T, 24, 3)`.
- Try training with `smpl_motion_file: dummy` first to confirm the robot
  encoder works before adding SMPL.

---

## 12. MuJoCo viewer renders incorrectly in Docker

**Symptom:** MuJoCo window is black, garbled, or shows rendering artifacts when
running inside Docker on a machine with an Intel display controller.

**Cause:** GPU passthrough or display driver conflict between the Intel iGPU and
NVIDIA dGPU inside Docker.

**Fix:** Force NVIDIA GPU rendering:

```bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

Or run with `--gpus all -e DISPLAY=$DISPLAY` in your Docker run command. See
[#25](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/25) for details.

---

## Still stuck?

- Search [existing issues](https://github.com/NVlabs/GR00T-WholeBodyControl/issues)
- Open a [new issue](https://github.com/NVlabs/GR00T-WholeBodyControl/issues/new)
  with your error message, Python version, and OS
