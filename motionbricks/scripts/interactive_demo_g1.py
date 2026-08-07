import argparse
import torch as t
import time
import platform

import mujoco
import mujoco.viewer
import numpy as np
from motionbricks.motion_backbone.demo.utils import navigation_demo


def _disable_mujoco_keyboard_shortcuts(controller_keys='wasdrtfgeqzxcvb'):
    """Prevent MuJoCo's viewer from processing keyboard shortcuts that
    conflict with the WASD motion controller.

    On Linux/X11: uses passive key grabs to intercept keys at the X server
    level before GLFW sees them.  pynput still captures keys via XRecord.

    On macOS/Windows: not yet supported — MuJoCo shortcuts may interfere.
    """
    if platform.system() != 'Linux':
        return
    try:
        from Xlib import display as xdisplay, X
        _xdpy = xdisplay.Display()
        _root = _xdpy.screen().root

        def _find_window_by_name(win, name_substr):
            try:
                name = win.get_wm_name()
                if name and name_substr in name:
                    return win
            except Exception:
                pass
            for child in win.query_tree().children:
                r = _find_window_by_name(child, name_substr)
                if r:
                    return r
            return None

        time.sleep(0.5)
        mj_win = _find_window_by_name(_root, 'MuJoCo')
        if mj_win:
            for ch in controller_keys:
                keycode = _xdpy.keysym_to_keycode(ord(ch) - 32)
                mj_win.grab_key(keycode, X.AnyModifier,
                                False, X.GrabModeAsync, X.GrabModeAsync)
            _xdpy.sync()
    except Exception as e:
        print(f"Note: could not disable MuJoCo keyboard shortcuts: {e}")


def main(args) -> None:
    demo_agent = navigation_demo(args)

    num_runs = 0
    while num_runs < args.num_runs:
        num_runs += 1
        print(f"Running iteration {num_runs}... / {args.num_runs}")
        random_seed = args.random_seed * (num_runs + 2333) * 2333 % (2 ** 32 - 1)
        np.random.seed(random_seed)
        t.manual_seed(random_seed)
        demo_agent.full_agent.reset()

        steps = 0

        if args.has_viewer:
            with mujoco.viewer.launch_passive(demo_agent.mj_model, demo_agent.mj_data) as viewer:
                _disable_mujoco_keyboard_shortcuts()

                while viewer.is_running() and steps < args.max_steps:
                    force_idle = steps + 100 > args.max_steps
                    steps += 1
                    viewer.user_scn.ngeom = 0
                    step_start = time.time()
                    qpos = demo_agent.full_agent.get_next_frame()
                    context_motion_features = demo_agent.full_agent.get_context_motion_features()
                    context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
                    demo_agent.mj_data.qpos[:] = qpos

                    control_signals = demo_agent.controller.generate_control_signals(
                        viewer, demo_agent.mj_model, demo_agent.mj_data, visualize=True,
                        control_info={"force_idle": force_idle,
                                      'allowed_mode': getattr(args, 'allowed_mode', None)}
                    )

                    if args.use_qpos:
                        control_signals['context_mujoco_qpos'] = context_mujoco_qpos
                    else:
                        control_signals['context_motion_features'] = context_motion_features

                    with t.no_grad():
                        demo_agent.full_agent.generate_new_frames(
                            control_signals,
                            demo_agent.controller.get_controller_dt() * args.generate_dt
                        )

                    mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)
                    viewer.cam.lookat[:] = demo_agent.controller.get_prev_qpos()[:, :3].mean(axis=0)
                    viewer.sync()
                    time_until_next_step = demo_agent.mj_model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        else:
            while steps < args.max_steps:
                steps += 1
                force_idle = steps + 100 > args.max_steps
                qpos = demo_agent.full_agent.get_next_frame()
                context_motion_features = demo_agent.full_agent.get_context_motion_features()
                context_mujoco_qpos = demo_agent.full_agent.get_context_mujoco_qpos()
                demo_agent.mj_data.qpos[:] = qpos

                control_signals = demo_agent.controller.generate_control_signals(
                    None, demo_agent.mj_model, demo_agent.mj_data, visualize=False,
                    control_info={"force_idle": force_idle, 'allowed_mode': getattr(args, 'allowed_mode', None)}
                )
                if args.use_qpos:
                    control_signals['context_mujoco_qpos'] = context_mujoco_qpos
                else:
                    control_signals['context_motion_features'] = context_motion_features

                with t.no_grad():
                    demo_agent.full_agent.generate_new_frames(
                        control_signals, demo_agent.controller.get_controller_dt() * args.generate_dt
                    )

                mujoco.mj_forward(demo_agent.mj_model, demo_agent.mj_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive demo for the G1 humanoid")

    # path configs
    parser.add_argument("--humanoid_xml", type=str, default="assets/skeletons/g1/scene_29dof.xml")
    parser.add_argument("--result_dir", type=str, default="./out")
    parser.add_argument("--data_root", type=str, default="./datasets")
    parser.add_argument("--explicit_dataset_folder", type=str, default=None)
    parser.add_argument("--reprocess_clips", type=int, default=0)

    # controller config
    parser.add_argument("--controller", type=str, default="wasd",
                        choices=["wasd", "random"])
    parser.add_argument("--lookat_movement_direction", type=int, default=0)
    parser.add_argument("--has_viewer", type=int, default=1)
    parser.add_argument("--pre_filter_qpos", type=int, default=1)
    parser.add_argument("--source_root_realignment", type=int, default=1)
    parser.add_argument("--target_root_realignment", type=int, default=1)
    parser.add_argument("--force_canonicalization", type=int, default=1)
    parser.add_argument("--skip_ending_target_cond", type=int, default=0)
    parser.add_argument("--random_speed_scale", type=int, default=0)
    parser.add_argument("--speed_scale", type=str, default="0.8,1.2")
    parser.add_argument("--generate_dt", type=float, default=2.0)

    # run configs
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--num_runs", type=int, default=1)

    # model configurations
    parser.add_argument("--use_qpos", type=int, default=1)
    parser.add_argument("--planner", type=str, default="default")
    parser.add_argument("--allowed_mode", type=str, default=None)
    parser.add_argument("--clips", type=str, default="G1")

    args = parser.parse_args()

    args.return_model_configs = True
    args.return_dataloader = True
    args.recording_dir = None
    args.EXP = args.planner
    args.speed_scale = [float(i) for i in args.speed_scale.split(",")]

    main(args)
