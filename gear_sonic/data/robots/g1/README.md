# Unitree G1 Description (URDF & MJCF)

## Overview

This package includes a universal humanoid robot description (URDF & MJCF) for the [Unitree G1](https://www.unitree.com/g1/), developed by [Unitree Robotics](https://www.unitree.com/).

<p align="center">
  <img src="images/g1_23dof.png" width="45%"/>
  <img src="images/g1_29dof.png" width="45%"/>
  <img src="images/g1_29dof_with_hand.png" width="45%"/>
  <img src="images/g1_dual_arm.png" width="45%"/>
</p>

As shown, there are a total of 4 versions of MJCF/URDF for the G1 robot:

* `g1_23dof`
* `g1_29dof`
* `g1_29dof_with_hand`
* `g1_dual_arm`

`g1_23dof_rev_1_0.urdf` and `g1_23dof_rev_1_0.xml` are the native
23-actuator rev-1.0 descriptions from Unitree's
[`unitree_ros`](https://github.com/unitreerobotics/unitree_ros/tree/f3772ce54c56ef2d34c6aee8100bc768896c7d19/robots/g1_description)
G1 description, pinned at revision
`f3772ce54c56ef2d34c6aee8100bc768896c7d19`. The checked-in files are
text-identical to that revision after newline normalization. Their local
SHA-256 values are:

* URDF: `e2f55a541a485d486b376b752734cd1912c3b1b6e74f57e89e2e68691b5aa523`
* MJCF: `ea1ce67705253e73a91f9587aa70a34aaa2d17943517b2fe5ac209283b2c9e0c`

Unlike the legacy deployment `g1_23dof.xml`, these files contain no dummy
29-DoF joints.

## Visulization with [MuJoCo](https://github.com/google-deepmind/mujoco)

1. Open MuJoCo Viewer

   ```bash
   pip install mujoco
   python -m mujoco.viewer
   ```

2. Drag and drop the MJCF/URDF model file (`g1_XXX.xml`/`g1_XXX.urdf`) to the MuJoCo Viewer.
