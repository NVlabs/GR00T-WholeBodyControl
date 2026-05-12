<div align="center">

  <img src="media/groot_wbc.png" width="800" alt="GEAR SONIC Header">

  <!-- --- -->
  
  
</div>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-76B900.svg)](LICENSE)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-orange.svg)](https://github.com/isaac-sim/IsaacLab/releases/tag/v2.3.2)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-76B900.svg)](https://nvlabs.github.io/GR00T-WholeBodyControl/)
[![Demo](https://img.shields.io/badge/Live%20Demo-GEAR--SONIC-blue.svg)](https://nvlabs.github.io/GEAR-SONIC/demo.html)

</div>

---




# GR00T-WholeBodyControl

This is the codebase for the **GR00T Whole-Body Control (WBC)** projects. It hosts model checkpoints and scripts for training, evaluating, and deploying advanced whole-body controllers for humanoid robots. We currently support:

- **Decoupled WBC**: the decoupled controller (RL for lower body, and IK for upper body) used in NVIDIA GR00T [N1.5](https://research.nvidia.com/labs/gear/gr00t-n1_5/) and [N1.6](https://research.nvidia.com/labs/gear/gr00t-n1_6/) models;
- **GEAR-SONIC Series**: our latest iteration of generalist humanoid whole-body controllers (see our [whitepaper](https://nvlabs.github.io/GEAR-SONIC/));
- **MotionBricks**: a real-time latent generative model for interactive motion control in animation and robotics (see the [project page](https://nvlabs.github.io/motionbricks/)).

## News

- **[2026-05-07]** 🤖 **End-to-end VLA workflow on G1** — collect teleop data, fine-tune Isaac-GR00T N1.7, and deploy with SONIC whole-body control. See [Data Collection](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html), [VLA Workflow](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_workflow.html), and [VLA Inference](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html).
- **[2026-04-27]** 🧩 **MotionBricks preview** — interactive G1 demo, pretrained checkpoints (VQVAE · pose · root), synthetic training code, and motion-representation docs. See [`motionbricks/`](motionbricks/) and the [project page](https://nvlabs.github.io/motionbricks/).
- **[2026-04-14]** 🌐 **[Live web demo](https://nvlabs.github.io/GEAR-SONIC/demo.html)** — try SONIC interactively in your browser. Features [Kimodo](https://github.com/nv-tlabs/kimodo) text-to-motion generation.
- **[2026-04-10]** 🚀 Released **SONIC training code and checkpoint** on [HuggingFace](https://huggingface.co/nvidia/GEAR-SONIC). Train from scratch or finetune. **Additional embodiment support** and **VLA data collection pipeline**. See [Training Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training.html).
- **[2026-03-24]** 🔧 C++ inference stack update: motor error monitoring, TTS alerts, ZMQ protocol v4, idle-mode readaptation. **ZMQ header size changed to 1280 bytes.**
- **[2026-03-16]** 📦 [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) open-sourced — 142K+ human motions (~288 hours) with G1 MuJoCo trajectories.
- **[2026-02-19]** 🎉 Released GEAR-SONIC: pretrained checkpoints, C++ inference, VR teleoperation, and documentation.
- **[2025-11-12]** 🏁 Initial release with Decoupled WBC for GR00T N1.5 and N1.6.

## Table of Contents

- [News](#news)
- [GEAR-SONIC](#gear-sonic)
- [VR Whole-Body Teleoperation](#vr-whole-body-teleoperation)
- [Kinematic Planner](#kinematic-planner)
- [SONIC Training](#sonic-training)
- [TODOs](#todos)
- [What's Included](#whats-included)
  - [Setup](#setup)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Support](#support)
- [MotionBricks](#motionbricks)
- [Decoupled WBC](#decoupled-wbc)


## GEAR-SONIC 

<p style="font-size: 1.2em;">
    <a href="https://nvlabs.github.io/GEAR-SONIC/"><strong>Website</strong></a> | 
    <a href="https://huggingface.co/nvidia/GEAR-SONIC"><strong>Model</strong></a> | 
    <a href="https://arxiv.org/abs/2511.07820"><strong>Paper</strong></a> | 
    <a href="https://nvlabs.github.io/GR00T-WholeBodyControl/"><strong>Docs</strong></a>
  </p>

<div align="center">
  <img src="docs/source/_static/sonic-preview-gif-480P.gif" width="800" >
  
</div>

SONIC is a humanoid behavior foundation model that gives robots a core set of motor skills learned from large-scale human motion data. Rather than building separate controllers for predefined motions, SONIC uses motion tracking as a scalable training task, enabling a single unified policy to produce natural, whole-body movement and support a wide range of behaviors — from walking and crawling to teleoperation and multi-modal control. It is designed to generalize beyond the motions it has seen during training and to serve as a foundation for higher-level planning and interaction.

In this repo, we release SONIC's training code, deployment framework, model checkpoints, and teleoperation stack for data collection.


## VR Whole-Body Teleoperation

SONIC supports real-time whole-body teleoperation via PICO VR headset, enabling natural human-to-robot motion transfer for data collection and interactive control.

<div align="center">
<table>
<tr>
<td align="center"><b>Walking</b></td>
<td align="center"><b>Running</b></td>
</tr>
<tr>
<td align="center"><img src="media/teleop_walking.gif" width="400"></td>
<td align="center"><img src="media/teleop_running.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Sideways Movement</b></td>
<td align="center"><b>Kneeling</b></td>
</tr>
<tr>
<td align="center"><img src="media/teleop_sideways.gif" width="400"></td>
<td align="center"><img src="media/teleop_kneeling.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Getting Up</b></td>
<td align="center"><b>Jumping</b></td>
</tr>
<tr>
<td align="center"><img src="media/teleop_getup.gif" width="400"></td>
<td align="center"><img src="media/teleop_jumping.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Bimanual Manipulation</b></td>
<td align="center"><b>Object Hand-off</b></td>
</tr>
<tr>
<td align="center"><img src="media/teleop_bimanual.gif" width="400"></td>
<td align="center"><img src="media/teleop_switch_hands.gif" width="400"></td>
</tr>
</table>
</div>

## Kinematic Planner

SONIC includes a kinematic planner for real-time locomotion generation — choose a movement style, steer with keyboard/gamepad, and adjust speed and height on the fly.

<div align="center">
<table>
<tr>
<td align="center" colspan="2"><b>In-the-Wild Navigation</b></td>
</tr>
<tr>
<td align="center" colspan="2"><img src="media/planner/planner_in_the_wild_navigation.gif" width="800"></td>
</tr>
<tr>
<td align="center"><b>Run</b></td>
<td align="center"><b>Happy</b></td>
</tr>
<tr>
<td align="center"><img src="media/planner/planner_run.gif" width="400"></td>
<td align="center"><img src="media/planner/planner_happy.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Stealth</b></td>
<td align="center"><b>Injured</b></td>
</tr>
<tr>
<td align="center"><img src="media/planner/planner_stealth.gif" width="400"></td>
<td align="center"><img src="media/planner/planner_injured.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Kneeling</b></td>
<td align="center"><b>Hand Crawling</b></td>
</tr>
<tr>
<td align="center"><img src="media/planner/planner_kneeling.gif" width="400"></td>
<td align="center"><img src="media/planner/planner_hand_crawling.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Elbow Crawling</b></td>
<td align="center"><b>Boxing</b></td>
</tr>
<tr>
<td align="center"><img src="media/planner/planner_elbow_crawling.gif" width="400"></td>
<td align="center"><img src="media/planner/planner_boxing.gif" width="400"></td>
</tr>
</table>
</div>

## SONIC Training

SONIC can be trained from scratch on the [Bones-SEED](https://huggingface.co/datasets/bones-studio/seed)
motion capture dataset (142K+ motions, ~288 hours, Unitree G1 retargeted), or finetuned
from the released checkpoint on [Hugging Face](https://huggingface.co/nvidia/GEAR-SONIC).

### Quick start

```bash
# Install training dependencies (Isaac Lab must be installed separately — see docs)
pip install -e "gear_sonic/[training]"

# Download checkpoint + SMPL data from Hugging Face
pip install huggingface_hub
python download_from_hf.py --training

# Download Bones-SEED G1 CSVs from huggingface.co/datasets/bones-studio/seed, then convert and filter
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input /path/to/bones_seed/g1/csv/ \
    --output data/motion_lib_bones_seed/robot --fps 30 --fps_source 120 --individual --num_workers 16
python gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source data/motion_lib_bones_seed/robot --dest data/motion_lib_bones_seed/robot_filtered

# Finetune from released checkpoint (64+ GPUs recommended)
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True \
    ++manager_env.commands.motion.motion_lib_cfg.motion_file=data/motion_lib_bones_seed/robot_filtered \
    ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=data/smpl_filtered
```

For the full guide including multi-node training, evaluation, ONNX export, and SOMA encoder setup:
📖 [Installation (Training)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html) |
[Training Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training.html)


## TODOs

- [x] Release pretrained SONIC policy checkpoints
- [x] Open source C++ inference stack
- [x] Setup documentation
- [x] Open source teleoperation stack and demonstration scripts
- [x] Release training scripts and recipes for motion imitation and fine-tuning
- [ ] Open source large-scale data collection workflows and fine-tuning VLA scripts. 
- [ ] Publish additional preprocessed large-scale human motion datasets



## What's Included

This release includes:

- **`gear_sonic_deploy`**: C++ inference stack for deploying SONIC policies on real hardware
- **`gear_sonic`**: Full SONIC training stack — PPO training, data processing pipeline, and configuration system for training on Bones-SEED and custom motion datasets
- **`motionbricks`**: Preview release of the MotionBricks real-time latent generative stack — interactive G1 demo, pretrained checkpoints, synthetic training code, and motion-representation docs

### Setup

> **Git LFS required.** This repo contains large binary assets (meshes, ONNX
> models). Without Git LFS, you will get small pointer files instead of actual
> data, causing silent failures. Install Git LFS first if you don't have it:
> `sudo apt install git-lfs && git lfs install`
>
> MotionBricks pretrained checkpoints are skipped by default to avoid an extra
> ~2.2 GiB download during normal monorepo setup. MotionBricks GIFs and meshes
> still download normally. Fetch the checkpoints explicitly if you plan to run
> the MotionBricks demo.

```bash
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl
git lfs pull

# Optional: fetch MotionBricks pretrained checkpoints.
git lfs pull --include="motionbricks/out/**" --exclude=""

# Verify your environment
python check_environment.py
```

### Which environment do I need?

| I want to... | Environment | How to install |
|---|---|---|
| **Train / finetune SONIC** | Isaac Lab's Python env | [Install Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html), then `pip install -e "gear_sonic/[training]"` |
| **Run MuJoCo simulation** | `.venv_sim` (auto-created) | `bash install_scripts/install_mujoco_sim.sh` |
| **VR teleoperation** | `.venv_teleop` (auto-created) | `bash install_scripts/install_pico.sh` |
| **Collect data** | `.venv_data_collection` (auto-created) | `bash install_scripts/install_data_collection.sh` |
| **Deploy on real robot** | C++ build | See [deployment docs](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html) |

Each use case has its own lightweight environment. The install scripts use `uv`
and create isolated venvs automatically — you don't need to manage them manually.
Training is the only one that requires Isaac Lab (installed separately).

## Documentation

📚 **[Full Documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/)**

### Getting Started
- [Installation Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)
- [Quick Start](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html)
- [VR Teleoperation Setup](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)

### Tutorials
- [Keyboard Control](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/keyboard.html)
- [Gamepad Control](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/gamepad.html)
- [ZMQ Communication](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/zmq.html)
- [ZMQ Manager / PICO VR](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vr_wholebody_teleop.html)

### Training
- [Installation (Training)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html)
- [Training Guide](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training.html)
- [Training Data](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training_data.html)

### Best Practices
- [Teleoperation](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/teleoperation.html)






---

## Citation

If you use GEAR-SONIC in your research, please cite:

```bibtex
@article{luo2025sonic,
    title={SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control},
    author={Luo, Zhengyi and Yuan, Ye and Wang, Tingwu and Li, Chenran and Chen, Sirui and Casta\~neda, Fernando and Cao, Zi-Ang and Li, Jiefeng and Minor, David and Ben, Qingwei and Da, Xingye and Ding, Runyu and Hogg, Cyrus and Song, Lina and Lim, Edy and Jeong, Eugene and He, Tairan and Xue, Haoru and Xiao, Wenli and Wang, Zi and Yuen, Simon and Kautz, Jan and Chang, Yan and Iqbal, Umar and Fan, Linxi and Zhu, Yuke},
    journal={arXiv preprint arXiv:2511.07820},
    year={2025}
}
```

---

## License

This project uses dual licensing:

- **Source Code**: Licensed under Apache License 2.0 - applies to all code, scripts, and software components in this repository
- **Model Weights**: Licensed under NVIDIA Open Model License - applies to all trained model checkpoints and weights

See [LICENSE](LICENSE) for the complete dual-license text.

Please review both licenses before using this project. The NVIDIA Open Model License permits commercial use with attribution and requires compliance with NVIDIA's Trustworthy AI terms.

All required legal documents, including the Apache 2.0 license, 3rd-party attributions, and DCO language, are consolidated in the /legal folder of this repository.

---

## Support

For questions and issues, please contact the GEAR WBC team at [gear-wbc@nvidia.com](mailto:gear-wbc@nvidia.com) to provide feedback! 

## MotionBricks

<p style="font-size: 1.2em;">
  <a href="https://nvlabs.github.io/motionbricks/"><strong>Project page</strong></a> |
  <a href="motionbricks/README.md"><strong>Subproject README</strong></a>
</p>

<div align="center">
  <img src="motionbricks/assets/gifs/teaser_animation.gif" width="400">
  <img src="motionbricks/assets/gifs/teaser_robotics.gif" width="400">
</div>

MotionBricks is a real-time generative framework that transforms interactive motion control for animation and robotics. It combines a large-scale latent backbone with intuitive "smart primitives" to deliver high-quality, zero-shot motion synthesis at 15,000 FPS — complementing the tracking-based GEAR-SONIC controllers in this repo.

This preview release ships an interactive G1 demo (keyboard-driven, MuJoCo viewer), pretrained checkpoints (VQVAE · pose · root), a synthetic training pipeline, and motion-representation docs. Its pretrained checkpoints are opt-in for monorepo clones; run `git lfs pull --include="motionbricks/out/**" --exclude=""` from the repo root before using the demo. A full release — fully embedded in the GEAR-SONIC pipeline — is targeted for approximately one month out. See [`motionbricks/README.md`](motionbricks/README.md) for setup, demo, and training instructions.

## Decoupled WBC

For the Decoupled WBC used in GR00T N1.5 and N1.6 models, please refer to the [Decoupled WBC documentation](docs/source/references/decoupled_wbc.md).


## Acknowledgments
We would like to acknowledge the following projects from which parts of the code in this repo are derived from:
- [Beyond Mimic](https://github.com/HybridRobotics/whole_body_tracking)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
