# G1 + Inspire model assets

These files combine upstream assets with project-specific model integration.
Inspire Robotics is the hand manufacturer; Unitree is the verified public
publisher of the hand assets referenced below.

| File | Purpose and attribution |
| --- | --- |
| `g1_29dof_rh56dfx.xml` | MuJoCo model with 41 actuators and 12 finger couplings. Uses 26 Unitree-published hand meshes and 24 matching finger inertial elements, plus 34 existing NVIDIA upstream G1 meshes. Assembly, naming, actuation and coupling are project integration. |
| `g1_29dof_rh56dfx.urdf` | Simplified data/FK model with 41 movable joints. Uses 14 of those hand meshes and the 34 G1 meshes. Hand inertials are placeholders: mass 0.005 kg and diagonal inertia 1e-6 kg·m²; they are not vendor physical parameters. |
| `scene_41dof_rh56dfx.xml` | Includes the MuJoCo model and adds a floor, lighting and display settings. It inherits the model's asset attribution. |
| `hand_config.yaml` | Channel/joint mapping, limits and communication settings shared by the Inspire backends; not vendor model data. |

This directory holds data only. The [hand extension overview](../../../../utils/hand_control/README.md)
describes where to add communication, simulation and deployment implementations.

## Sources and licenses

- **Hand meshes:** SHA256-identical to
  [Unitree MuJoCo H1 assets](https://github.com/unitreerobotics/unitree_mujoco/tree/1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d/unitree_robots/h1/assets).
  See [BSD-3-Clause notice](../../../../../legal/LICENSE-unitree_mujoco.txt).
- **MJCF finger inertials:** 24 `inertial` elements match `pos`, `quat`, `mass`
  and `diaginertia` in
  [Unitree ROS h1_2.xml](https://github.com/unitreerobotics/unitree_ros/blob/5994d4faef0a9cadd3287f8de0199a67eeb2a259/robots/h1_2_description/h1_2.xml).
  This match excludes both hand bases and other model settings.
  See [BSD-3-Clause notice](../../../../../legal/LICENSE-unitree_ros.txt).
- **G1 body meshes:** 34 unchanged assets from NVIDIA's
  [GR00T-WholeBodyControl mesh directory](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/a0732b642c0333077e127a2f56ab0014c196bca4/gear_sonic/data/robots/g1/meshes).
  Their direct Unitree origin has not been individually verified.

[ASSET_SOURCES.json](ASSET_SOURCES.json) records pinned revisions, per-file hashes,
per-model mesh references and the 24 inertial mappings. Content matches identify
public distribution sources, not the original CAD author or historical download
route. Upstream notices apply to their respective material; they do not assign
a single new license to the combined model.

Meshes are shared in `../meshes/`. From the repository root, download them with:

```bash
git lfs pull --include="gear_sonic/data/robots/g1/meshes/**" --exclude=""
```

The Pinocchio entry point is
`gear_sonic.data.robot_model.instantiation.instantiate_g1_rh56dfx_robot_model`.
See the [hand backend guide](../../../../../docs/source/tutorials/inspire_hands.md) for setup and usage.
