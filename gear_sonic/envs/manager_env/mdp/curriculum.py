from __future__ import annotations

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils import configclass
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    force_push_curriculum = None
    force_push_linear_curriculum = None


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def step_curriculum(env, env_ids, original_value, values, num_steps):
    # Override after num_steps
    assert len(values) == len(num_steps)
    for i in range(len(values)):
        if env.common_step_counter > num_steps[len(num_steps) - i - 1]:
            return values[len(num_steps) - i - 1]
    return original_value


def linear_curriculum(env, env_ids, original_value, values, num_steps):
    """
    Linearly interpolates training curriculum values based on step counter.

    Args:
        env: IsaacLab environment (must have `common_step_counter`).
        env_ids: Unused here, but kept for API consistency.
        original_value (float): The base value before curriculum starts.
        values (list of float): Target values at milestones.
        num_steps (list of int): Step milestones corresponding to values.
                                 Must be same length as `values`.

    Returns:
        float: interpolated value at current step.
    """
    assert len(values) == len(num_steps), "values and num_steps must match"

    step = env.common_step_counter

    # Between milestones → interpolate
    for i in range(1, len(num_steps)):
        if step <= num_steps[i]:
            t0, t1 = num_steps[i - 1], num_steps[i]
            v0, v1 = values[i - 1], values[i]
            alpha = (step - t0) / (t1 - t0)
            return v0 + alpha * (v1 - v0)

    # After last milestone → final value
    return values[-1]
