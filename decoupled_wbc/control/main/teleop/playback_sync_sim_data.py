"""
A convenience script to playback random demonstrations using the decoupled_wbc controller from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --dataset (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --use-wbc-goals (optional): If set, will use the stored WBC goals to control the robot,
        otherwise will use the actions directly. Only relevant if --use-actions is set.
    --use-teleop-cmd (optional): If set, will use teleop IK directly with WBC timing
        for action generation. Only relevant if --use-actions is set.
    --visualize-gripper (optional): If set, will visualize the gripper site
    --save-video (optional): If set, will save video of the playback using offscreen rendering
    --video-path (optional): Path to save the output video. If not specified, will use the nearest
        folder to dataset and save as playback_video.mp4
    --num-episodes (optional): Number of episodes to playback/record (if None, plays random episodes)

Example:
    $ python decoupled_wbc/control/main/teleop/playback_sync_sim_data.py --dataset output/robocasa_datasets/
        --use-actions --use-wbc-goals

    $ python decoupled_wbc/control/main/teleop/playback_sync_sim_data.py --dataset output/robocasa_datasets/
        --use-actions --use-teleop-cmd

    # Record video of the first 5 episodes using WBC goals
    $ python decoupled_wbc/control/main/teleop/playback_sync_sim_data.py --dataset output/robocasa_datasets/
        --use-actions --use-wbc-goals --save-video --num-episodes 5
"""

import json
import os
from pathlib import Path
import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from robosuite.environments.robot_env import RobotEnv
from tqdm import tqdm
import tyro

from decoupled_wbc.control.main.teleop.configs.configs import SyncSimPlaybackConfig
from decoupled_wbc.control.robot_model.instantiation import get_robot_type_and_model
from decoupled_wbc.control.utils.sync_sim_utils import (
    generate_frame,
    get_data_exporter,
    get_env,
    get_policies,
)
from decoupled_wbc.data.constants import RS_VIEW_CAMERA_HEIGHT, RS_VIEW_CAMERA_WIDTH
from decoupled_wbc.data.exporter import TypedLeRobotDataset

CONTROL_NODE_NAME = "playback_node"
GREEN_BOLD = "\033[1;32m"
RED_BOLD = "\033[1;31m"
RESET = "\033[0m"


def load_lerobot_dataset(root_path, max_episodes=None):
    task_name = None
    episodes = []
    start_index = 0
    with open(Path(root_path) / "meta/episodes.jsonl", "r") as f:
        for line in f:
            episode = json.loads(line)
            episode["start_index"] = start_index
            start_index += episode["length"]
            assert (
                task_name is None or task_name == episode["tasks"][0]
            ), "All episodes should have the same task name"
            task_name = episode["tasks"][0]
            episodes.append(episode)

    dataset = TypedLeRobotDataset(
        repo_id="tmp/test",
        root=root_path,
        load_video=False,
    )

    script_config = dataset.meta.info["script_config"]

    assert len(dataset) == start_index, "Dataset length does not match expected length"

    # Limit episodes if specified
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
        print(
            f"Loading only first {len(episodes)} episodes (limited by max_episodes={max_episodes})"
        )

    f = {}
    seeds = []
    for ep in tqdm(range(len(episodes))):
        seed = None
        f[f"data/demo_{ep + 1}/states"] = []
        f[f"data/demo_{ep + 1}/actions"] = []
        f[f"data/demo_{ep + 1}/teleop_cmd"] = []
        f[f"data/demo_{ep + 1}/wbc_goal"] = []
        start_index = episodes[ep]["start_index"]
        end_index = start_index + episodes[ep]["length"]
        for i in tqdm(range(start_index, end_index)):
            frame = dataset[i]
            # load the seed
            assert (
                seed is None or seed == np.array(frame["observation.sim.seed"]).item()
            ), "All observations in an episode should have the same seed"
            seed = np.array(frame["observation.sim.seed"]).item()
            # load the state
            mujoco_state_len = frame["observation.sim.mujoco_state_len"]
            mujoco_state = frame["observation.sim.mujoco_state"]
            f[f"data/demo_{ep + 1}/states"].append(np.array(mujoco_state[:mujoco_state_len]))
            # load the action
            action = frame["action"]
            f[f"data/demo_{ep + 1}/actions"].append(np.array(action))

            # load the teleop command
            teleop_cmd = {
                "left_wrist": np.array(frame["observation.sim.left_wrist"].reshape(4, 4)),
                "right_wrist": np.array(frame["observation.sim.right_wrist"].reshape(4, 4)),
                "left_fingers": {
                    "position": np.array(frame["observation.sim.left_fingers"].reshape(25, 4, 4)),
                },
                "right_fingers": {
                    "position": np.array(frame["observation.sim.right_fingers"].reshape(25, 4, 4)),
                },
                "target_upper_body_pose": np.array(frame["observation.sim.target_upper_body_pose"]),
                "base_height_command": np.array(frame["teleop.base_height_command"]),
                "navigate_cmd": np.array(frame["teleop.navigate_command"]),
            }
            f[f"data/demo_{ep + 1}/teleop_cmd"].append(teleop_cmd)
            # load the WBC goal
            wbc_goal = {
                "wrist_pose": np.array(frame["action.eef"]),
                "target_upper_body_pose": np.array(frame["observation.sim.target_upper_body_pose"]),
                "navigate_cmd": np.array(frame["teleop.navigate_command"]),
                "base_height_command": np.array(frame["teleop.base_height_command"]),
            }
            f[f"data/demo_{ep + 1}/wbc_goal"].append(wbc_goal)

        seeds.append(seed)

    return seeds, f, script_config


def validate_state(recorded_state, playback_state, ep, step, tolerance=1e-5):
    """Validate that playback state matches recorded state within tolerance."""
    if not np.allclose(recorded_state, playback_state, atol=tolerance):
        err = np.linalg.norm(recorded_state - playback_state)
        print(f"[warning] state diverged by {err:.12f} for ep {ep} at step {step}")
        return False
    return True


def generate_and_save_frame(
    config, sync_env, obs, wbc_action, seed, teleop_cmd, wbc_goal, gr00t_exporter
):
    """Generate and save a frame to LeRobot dataset if enabled."""
    if config.save_lerobot:
        max_mujoco_state_len, mujoco_state_len, mujoco_state = sync_env.get_mujoco_state_info()
        frame = generate_frame(
            obs,
            wbc_action,
            seed,
            mujoco_state,
            mujoco_state_len,
            max_mujoco_state_len,
            teleop_cmd,
            wbc_goal,
            config.save_img_obs,
        )
        gr00t_exporter.add_frame(frame)


def playback_wbc_goals(
    sync_env,
    wbc_policy,
    wbc_goals,
    teleop_cmds,
    states,
    env,
    onscreen,
    config,
    video_writer,
    ep,
    seed,
    gr00t_exporter,
    end_steps,
):
    """Playback using WBC goals to control the robot."""
    ret = True
    num_wbc_goals = len(wbc_goals) if end_steps == -1 else min(end_steps, len(wbc_goals))

    for jj in range(num_wbc_goals):
        wbc_goal = wbc_goals[jj]
        obs = sync_env.observe()
        wbc_policy.set_observation(obs)
        wbc_policy.set_goal(wbc_goal)
        wbc_action = wbc_policy.get_action()
        sync_env.queue_action(wbc_action)

        # Save frame if needed
        if config.save_lerobot:
            teleop_cmd = teleop_cmds[jj]
            generate_and_save_frame(
                config, sync_env, obs, wbc_action, seed, teleop_cmd, wbc_goal, gr00t_exporter
            )

        capture_or_render_frame(env, onscreen, config, video_writer)

        if jj < len(states) - 1:
            state_playback = env.sim.get_state().flatten()
            if not validate_state(states[jj + 1], state_playback, ep, jj):
                ret = False

    return ret


def playback_teleop_cmd(
    sync_env,
    wbc_policy,
    teleop_policy,
    wbc_goals,
    teleop_cmds,
    states,
    env,
    onscreen,
    config,
    video_writer,
    ep,
    seed,
    gr00t_exporter,
    end_steps,
):
    """Playback using teleop commands to control the robot."""
    ret = True
    num_steps = len(wbc_goals) if end_steps == -1 else min(end_steps, len(wbc_goals))

    for jj in range(num_steps):
        wbc_goal = wbc_goals[jj]
        teleop_cmd = teleop_cmds[jj]

        # Set IK goal from teleop command
        ik_data = {
            "body_data": {
                teleop_policy.retargeting_ik.body.supplemental_info.hand_frame_names[
                    "left"
                ]: teleop_cmd["left_wrist"],
                teleop_policy.retargeting_ik.body.supplemental_info.hand_frame_names[
                    "right"
                ]: teleop_cmd["right_wrist"],
            },
            "left_hand_data": teleop_cmd["left_fingers"],
            "right_hand_data": teleop_cmd["right_fingers"],
        }
        teleop_policy.retargeting_ik.set_goal(ik_data)

        # Store original and get new upper body pose
        target_upper_body_pose = wbc_goal["target_upper_body_pose"].copy()
        wbc_goal["target_upper_body_pose"] = teleop_policy.retargeting_ik.get_action()

        # Execute WBC policy
        obs = sync_env.observe()
        wbc_policy.set_observation(obs)
        wbc_policy.set_goal(wbc_goal)
        wbc_action = wbc_policy.get_action()
        sync_env.queue_action(wbc_action)

        # Save frame if needed
        generate_and_save_frame(
            config, sync_env, obs, wbc_action, seed, teleop_cmd, wbc_goal, gr00t_exporter
        )

        # Render or capture frame
        capture_or_render_frame(env, onscreen, config, video_writer)

        # Validate states
        if jj < len(states) - 1:
            if not np.allclose(
                target_upper_body_pose, wbc_goal["target_upper_body_pose"], atol=1e-5
            ):
                err = np.linalg.norm(target_upper_body_pose - wbc_goal["target_upper_body_pose"])
                print(
                    f"[warning] target_upper_body_pose diverged by {err:.12f} for ep {ep} at step {jj}"
                )
                ret = False

            state_playback = env.sim.get_state().flatten()
            if not validate_state(states[jj + 1], state_playback, ep, jj):
                ret = False

    return ret


def playback_actions(
    sync_env,
    actions,
    teleop_cmds,
    wbc_goals,
    states,
    env,
    onscreen,
    config,
    video_writer,
    ep,
    seed,
    gr00t_exporter,
    end_steps,
):
    """Playback using actions directly."""
    ret = True
    num_actions = len(actions) if end_steps == -1 else min(end_steps, len(actions))

    for j in range(num_actions):
        sync_env.queue_action({"q": actions[j]})

        # Save frame if needed
        if config.save_lerobot:
            obs = sync_env.observe()
            teleop_cmd = teleop_cmds[j]
            wbc_goal = wbc_goals[j]
            wbc_action = {"q": actions[j]}
            generate_and_save_frame(
                config, sync_env, obs, wbc_action, seed, teleop_cmd, wbc_goal, gr00t_exporter
            )

        capture_or_render_frame(env, onscreen, config, video_writer)

        if j < len(states) - 1:
            state_playback = env.sim.get_state().flatten()
            if not validate_state(states[j + 1], state_playback, ep, j):
                ret = False

    return ret


def playback_states(
    sync_env,
    states,
    actions,
    teleop_cmds,
    wbc_goals,
    env,
    onscreen,
    config,
    video_writer,
    seed,
    gr00t_exporter,
    end_steps,
    ep,
):
    """Playback by forcing mujoco states directly."""
    ret = True
    num_states = len(states) if end_steps == -1 else min(end_steps, len(states))

    for i in range(num_states):
        sync_env.reset_to({"states": states[i]})
        sync_env.render()

        # Validate that the state was set correctly
        if i < len(states):
            state_playback = env.sim.get_state().flatten()
            if not validate_state(states[i], state_playback, ep, i):
                ret = False

        # Save frame if needed
        if config.save_lerobot:
            obs = sync_env.observe()
            teleop_cmd = teleop_cmds[i]
            wbc_goal = wbc_goals[i]
            wbc_action = {"q": actions[i]}
            generate_and_save_frame(
                config, sync_env, obs, wbc_action, seed, teleop_cmd, wbc_goal, gr00t_exporter
            )

        capture_or_render_frame(env, onscreen, config, video_writer)

    return ret


def main(config: SyncSimPlaybackConfig):
    ret = True
    start_time = time.time()

    np.set_printoptions(precision=5, suppress=True, linewidth=120)

    assert config.dataset is not None, "Folder must be specified for playback"

    seeds, f, script_config = load_lerobot_dataset(config.dataset)

    config.update(
        script_config,
        allowed_keys=[
            "wbc_version",
            "wbc_model_path",
            "wbc_policy_class",
            "control_frequency",
            "enable_waist",
            "with_hands",
            "env_name",
            "robot",
            "task_name",
            "teleop_frequency",
            "data_collection_frequency",
            "enable_gravity_compensation",
            "gravity_compensation_joints",
        ],
    )
    config.validate_args()

    robot_type, robot_model = get_robot_type_and_model(config.robot, config.enable_waist)

    # Setup rendering
    if config.save_video or config.save_img_obs:
        onscreen = False
        offscreen = True
    else:
        onscreen = True
        offscreen = False

    # Set default video path if not specified
    if config.save_video and config.video_path is None:
        if os.path.isfile(config.dataset):
            video_folder = Path(config.dataset).parent
        else:
            video_folder = Path(config.dataset)
        video_folder.mkdir(parents=True, exist_ok=True)
        config.video_path = str(video_folder / "playback_video.mp4")
        print(f"Video recording enabled. Output: {config.video_path}")

    sync_env = get_env(config, onscreen=onscreen, offscreen=offscreen)

    gr00t_exporter = None
    if config.save_lerobot:
        obs = sync_env.observe()
        gr00t_exporter = get_data_exporter(config, obs, robot_model)

    # Initialize policies
    wbc_policy, teleop_policy = get_policies(
        config, robot_type, robot_model, activate_keyboard_listener=False
    )

    # List of all demonstrations episodes
    demos = [f"demo_{i + 1}" for i in range(len(seeds))]
    print(f"Loaded and will playback {len(demos)} episodes")
    env = sync_env.base_env

    # Setup video writer
    video_writer = None
    fourcc = None
    if config.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            config.video_path, fourcc, 20, (RS_VIEW_CAMERA_WIDTH, RS_VIEW_CAMERA_HEIGHT)
        )

    print("Loaded {} episodes from {}".format(len(demos), config.dataset))
    print("seeds:", seeds)
    print("demos:", demos, "\n\n")

    # Handle episode selection - either limited number or infinite random
    max_episodes = len(demos)
    episode_count = 0
    while True:
        if episode_count >= max_episodes:
            break
        ep = demos[episode_count]
        print(f"Playing back episode: {ep}")
        episode_count += 1

        # read the model xml, using the metadata stored in the attribute for this episode
        seed = seeds[int(ep.split("_")[-1]) - 1]
        sync_env.reset(seed=seed)

        # load the actions and states
        states = f["data/{}/states".format(ep)]
        actions = f["data/{}/actions".format(ep)]
        teleop_cmds = f["data/{}/teleop_cmd".format(ep)]
        wbc_goals = f["data/{}/wbc_goal".format(ep)]

        # reset the policies
        wbc_policy, teleop_policy, _ = get_policies(
            config, robot_type, robot_model, activate_keyboard_listener=False
        )
        end_steps = 20 if config.ci_test else -1

        if config.use_actions:
            # load the initial state
            sync_env.reset_to({"states": states[0]})
            # load the actions and play them back open-loop
            if config.use_wbc_goals:
                # use the wbc_goals to control the robot
                episode_ret = playback_wbc_goals(
                    sync_env,
                    wbc_policy,
                    wbc_goals,
                    teleop_cmds,
                    states,
                    env,
                    onscreen,
                    config,
                    video_writer,
                    ep,
                    seed,
                    gr00t_exporter,
                    end_steps,
                )
                ret = ret and episode_ret
            elif config.use_teleop_cmd:
                # use the teleop commands to control the robot
                episode_ret = playback_teleop_cmd(
                    sync_env,
                    wbc_policy,
                    teleop_policy,
                    wbc_goals,
                    teleop_cmds,
                    states,
                    env,
                    onscreen,
                    config,
                    video_writer,
                    ep,
                    seed,
                    gr00t_exporter,
                    end_steps,
                )
                ret = ret and episode_ret
            else:
                episode_ret = playback_actions(
                    sync_env,
                    actions,
                    teleop_cmds,
                    wbc_goals,
                    states,
                    env,
                    onscreen,
                    config,
                    video_writer,
                    ep,
                    seed,
                    gr00t_exporter,
                    end_steps,
                )
                ret = ret and episode_ret
        else:
            # force the sequence of internal mujoco states one by one
            episode_ret = playback_states(
                sync_env,
                states,
                actions,
                teleop_cmds,
                wbc_goals,
                env,
                onscreen,
                config,
                video_writer,
                seed,
                gr00t_exporter,
                end_steps,
                ep,
            )
            ret = ret and episode_ret

        if config.save_lerobot:
            gr00t_exporter.save_episode()

        print(f"Episode {ep} playback finished.\n\n")

    # close the env
    sync_env.close()

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved to: {config.video_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    if config.save_lerobot:
        print(f"LeRobot dataset saved to: {gr00t_exporter.root}")

    print(
        f"{GREEN_BOLD}Playback with WBC version: {config.wbc_version}, {config.wbc_model_path}, "
        f"{config.wbc_policy_class}, use_actions: {config.use_actions}, use_wbc_goals: {config.use_wbc_goals}, "
        f"use_teleop_cmd: {config.use_teleop_cmd}{RESET}"
    )
    if ret:
        print(f"{GREEN_BOLD}Playback completed successfully in {elapsed_time:.2f} seconds!{RESET}")
    else:
        print(f"{RED_BOLD}Playback encountered an error in {elapsed_time:.2f} seconds!{RESET}")

    return ret


def capture_or_render_frame(
    env: RobotEnv,
    onscreen: bool,
    config: SyncSimPlaybackConfig,
    video_writer: Optional[cv2.VideoWriter],
):
    """Capture frame for video recording if enabled, or render the environment."""
    if config.save_video:
        if hasattr(env, "sim") and hasattr(env.sim, "render"):
            img = env.sim.render(
                width=RS_VIEW_CAMERA_WIDTH,
                height=RS_VIEW_CAMERA_HEIGHT,
                camera_name=env.render_camera[0],
            )
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_bgr = np.flipud(img_bgr)
            video_writer.write(img_bgr)
    elif onscreen:
        env.render()


if __name__ == "__main__":
    config = tyro.cli(SyncSimPlaybackConfig)

    rclpy.init(args=None)
    node = rclpy.create_node("playback_decoupled_wbc_control")

    main(config)

    rclpy.shutdown()
