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

## Visulization with [MuJoCo](https://github.com/google-deepmind/mujoco)

1. Open MuJoCo Viewer

   ```bash
   pip install mujoco
   python -m mujoco.viewer
   ```

2. Drag and drop the MJCF/URDF model file (`g1_XXX.xml`/`g1_XXX.urdf`) to the MuJoCo Viewer.
