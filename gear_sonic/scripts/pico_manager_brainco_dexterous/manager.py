"""Interactive manager state machine and resource orchestration."""

import time

import numpy as np
import zmq

from .brainco import (
    BraincoHandCommandPublisher, compute_brainco_hand_targets_from_inputs,
    _normalize_trigger_value,
)
from .constants import (
    BRAINCO_DEFAULT_EXCLUDED_FINGERS, BRAINCO_TRIGGER_THRESHOLD,
    LocomotionMode, StreamMode,
)
from .controller_input import (
    _init_input_source, get_abxy_buttons, get_axis_clicks, get_controller_inputs,
)
from .g1_state import _create_g1_state_subscriber
from .planner import PlannerStreamer
from .pose_streamer import PoseStreamer
from .runtime import build_command_message, pack_pose_message
from .tracking import ThreePointPose

def run_pico_manager(
    port: int = 5556,
    buffer_size: int = 15,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5557,
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    input_source: str = "xrt",
    enable_brainco_hand: bool = True,
    brainco_dds_domain_id: int = 0,
    brainco_network_interface: str | None = None,
    brainco_trigger_threshold: float = BRAINCO_TRIGGER_THRESHOLD,
    brainco_trigger_range: str = "auto",
    brainco_target_q: np.ndarray | None = None,
    brainco_tau_thresholds: np.ndarray | None = None,
    brainco_excluded_fingers: tuple[str, ...] | list[str] = BRAINCO_DEFAULT_EXCLUDED_FINGERS,
):
    """
    Manager: creates shared PUB socket and runs pose/planner streamers based on current mode.
    Controller input:
      A+X: Toggle between planner and pose mode
      A+B+X+Y: Toggle policy start/stop
    """
    reader = _init_input_source(input_source, buffer_size)
    brainco_hand = (
        BraincoHandCommandPublisher(
            dds_domain_id=brainco_dds_domain_id,
            dds_network_interface=brainco_network_interface,
            trigger_threshold=brainco_trigger_threshold,
            trigger_range=brainco_trigger_range,
            target_q=brainco_target_q,
            tau_thresholds=brainco_tau_thresholds,
            excluded_fingers=brainco_excluded_fingers,
        )
        if enable_brainco_hand
        else None
    )
    g1_state = _create_g1_state_subscriber(
        brainco_dds_domain_id,
        brainco_network_interface,
        dds_already_initialized=brainco_hand is not None,
    )

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[Manager] ZMQ socket bound to port {port}")

    # Print available locomotion modes
    try:
        print("[Manager] Available modes:")
        for mode in LocomotionMode:
            print(f"  {mode.value}: {mode.name}")
    except Exception:
        pass

    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix="PoseLoop",
    )

    pose_streamer = PoseStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        num_frames_to_send=num_frames_to_send,
        target_fps=target_fps,
        use_cuda=use_cuda,
        record_dir=record_dir,
        record_format=record_format,
        brainco_hand=brainco_hand,
        g1_state=g1_state,
        brainco_trigger_threshold=brainco_trigger_threshold,
        brainco_trigger_range=brainco_trigger_range,
        log_prefix="PoseLoop",
    )
    planner_streamer = PlannerStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        poll_hz=20,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
    )

    # State machine diagram:
    #
    #   Chain 1 (by_pressed enters/exits, left_axis_click toggles sub-mode):
    #     POSE <--(by)--> PLANNER_FROZEN_UPPER_BODY <--(left_axis_click)--> PLANNER_VR_3PT
    #                                                                         |
    #                                                                    (by)--> POSE
    #
    #   Chain 2 (ax_pressed enters/exits, left_axis_click toggles sub-mode):
    #     POSE <--(ax)--> PLANNER <--(left_axis_click)--> PLANNER_VR_3PT
    #                                                        |
    #                                                   (ax)--> POSE
    #
    #   Emergency stop from any mode: A+B+X+Y (start_combo) --> OFF
    #   POSE_PAUSE: left_menu_button held --> POSE_PAUSE, released --> POSE
    #
    print("Manager controls: A+X=toggle mode, A+B+X+Y=start/stop policy")
    current_mode = StreamMode.OFF
    # Track which mode VR_3PT was entered from, so left_axis_click returns to it.
    # Will be either PLANNER or PLANNER_FROZEN_UPPER_BODY.
    vr3pt_parent_mode = StreamMode.PLANNER
    prev_toggle_dc = False
    prev_toggle_da = False
    prev_right_grip_pressed = False
    prev_left_trigger_pressed = False
    prev_right_trigger_pressed = False
    trigger_events_legacy_range = brainco_trigger_range == "legacy_10_to_0"
    try:
        prev_ax_pressed = False
        prev_by_pressed = False
        prev_start_combo = False
        prev_left_axis_click = False
        while True:
            # Poll Pico controller for buttons/axes
            a_pressed, b_pressed, x_pressed, y_pressed = get_abxy_buttons(reader)

            (
                left_menu_button,
                left_trigger_mgr,
                right_trigger_mgr,
                left_grip_mgr,
                right_grip_mgr,
            ) = get_controller_inputs(reader)
            if brainco_hand is not None:
                left_brainco_q, right_brainco_q = brainco_hand.publish_from_controller_inputs(
                    left_trigger_mgr,
                    right_trigger_mgr,
                    left_grip_mgr,
                    right_grip_mgr,
                )
            else:
                left_brainco_q, right_brainco_q = compute_brainco_hand_targets_from_inputs(
                    left_trigger_mgr,
                    right_trigger_mgr,
                    brainco_trigger_threshold,
                    brainco_trigger_range,
                    left_grip_mgr,
                    right_grip_mgr,
                )

            left_axis_click, _ = get_axis_clicks(reader)

            # Rising edge: A+X pressed together -> toggle POSE/PLANNER mode
            ax_pressed = (a_pressed) and (x_pressed)

            # Rising edge: B+Y pressed together -> toggle POSE/PLANNER_FROZEN_UPPER_BODY mode
            by_pressed = (b_pressed) and (y_pressed)

            # Rising edge: A+B+X+Y pressed together -> toggle policy start/stop (planner=True)
            start_combo = (a_pressed) and (b_pressed) and (x_pressed) and (y_pressed)

            new_mode = current_mode
            if current_mode == StreamMode.OFF:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.PLANNER
                    # Calibrate VR 3pt tracking NOW: operator should be in zero-ref pose.
                    # Uses the current Pico SMPL frame + FK of all-zero body joints.
                    sample = reader.get_latest()
                    if sample is not None:
                        three_point.calibrate_now(sample["body_poses_np"])
                    else:
                        print("[Manager] WARNING: No SMPL data available for calibration")

            elif current_mode == StreamMode.PLANNER:
                # Chain 2: POSE <--(ax)--> PLANNER <--(left_axis_click)--> VR_3PT
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.POSE
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.POSE:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.PLANNER  # Enter chain 2
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY  # Enter chain 1
                elif left_menu_button:
                    new_mode = StreamMode.POSE_PAUSE

            elif current_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                # Chain 1: POSE <--(by)--> FROZEN <--(left_axis_click)--> VR_3PT
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.POSE
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.POSE_PAUSE:
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif not left_menu_button:
                    new_mode = StreamMode.POSE

            elif current_mode == StreamMode.PLANNER_VR_3PT:
                # VR_3PT is reachable from both chains:
                #   left_axis_click → return to parent (PLANNER or FROZEN)
                #   ax_pressed      → POSE (chain 2 exit)
                #   by_pressed      → POSE (chain 1 exit)
                if start_combo and not prev_start_combo:
                    new_mode = StreamMode.OFF
                elif left_axis_click and not prev_left_axis_click:
                    new_mode = vr3pt_parent_mode  # Return to parent mode
                elif ax_pressed and not prev_ax_pressed:
                    new_mode = StreamMode.POSE
                elif by_pressed and not prev_by_pressed:
                    new_mode = StreamMode.POSE

            # Handle mode transitions before running loop
            if new_mode != current_mode:
                if current_mode == StreamMode.POSE:
                    pose_streamer.on_mode_exit()

                # Track parent when entering VR_3PT
                if new_mode == StreamMode.PLANNER_VR_3PT:
                    vr3pt_parent_mode = current_mode
                    print(f"[Manager] VR_3PT parent: {vr3pt_parent_mode.name}")

                if new_mode == StreamMode.POSE:
                    pose_streamer.reset_yaw()
                elif new_mode == StreamMode.PLANNER and current_mode != StreamMode.PLANNER_VR_3PT:
                    # Only reset yaw when freshly entering PLANNER from POSE,
                    # not when returning from VR_3PT sub-mode
                    planner_streamer.reset_yaw()
                elif new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                    if current_mode != StreamMode.PLANNER_VR_3PT:
                        # Freshly entering from POSE: reset yaw and grab initial targets
                        planner_streamer.reset_yaw()
                    # Always re-grab the latest robot state as frozen targets,
                    # whether entering from POSE or returning from VR_3PT
                    # (the old targets are stale after VR_3PT moved the arms)
                    planner_streamer.save_upper_body_position_target()
                elif new_mode == StreamMode.PLANNER_VR_3PT:
                    # Recalibrate VR tracking against the robot's actual current pose
                    # (read via g1_debug feedback + FK) to prevent sudden jumps
                    planner_streamer.recalibrate_for_vr3pt()

            # Run one iteration of the new mode
            if new_mode == StreamMode.POSE:
                pose_streamer.run_once()
            elif (
                new_mode == StreamMode.PLANNER
                or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                or new_mode == StreamMode.PLANNER_VR_3PT
            ):
                planner_streamer.run_once(new_mode)

            # Make sure to send command messages after loop iteration to ensure data arrives before mode switch
            if new_mode != current_mode:
                if new_mode == StreamMode.OFF:
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    exit()
                elif (
                    new_mode == StreamMode.PLANNER
                    or new_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY
                    or new_mode == StreamMode.PLANNER_VR_3PT
                ):
                    socket.send(build_command_message(start=True, stop=False, planner=True))
                elif new_mode == StreamMode.POSE:
                    socket.send(build_command_message(start=True, stop=False, planner=False))

                print(f"[Manager] StreamMode switch: {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode

            # Mode-independent: send manager_state for data exporter
            toggle_dc_tmp = bool(a_pressed) and left_grip_mgr > 0.5
            toggle_da_tmp = bool(b_pressed) and left_grip_mgr > 0.5
            toggle_dc = toggle_dc_tmp and not prev_toggle_dc
            toggle_da = toggle_da_tmp and not prev_toggle_da
            prev_toggle_dc = toggle_dc_tmp
            prev_toggle_da = toggle_da_tmp

            # A short pulse on the rising edge of the right grip. Keeping this
            # in manager_state makes markers available in every stream mode and
            # prevents a held grip from producing one marker per frame. The left
            # grip remains reserved for episode start/stop/abort combinations.
            right_grip_pressed = right_grip_mgr > 0.5
            grip_marker = right_grip_pressed and not prev_right_grip_pressed
            prev_right_grip_pressed = right_grip_pressed

            # Persist auto-detection because a pressed legacy trigger can have a
            # raw value near zero and cannot be classified from that sample alone.
            if (
                brainco_trigger_range == "auto"
                and (float(left_trigger_mgr) > 1.0 or float(right_trigger_mgr) > 1.0)
            ):
                trigger_events_legacy_range = True
            trigger_event_range = (
                "legacy_10_to_0"
                if trigger_events_legacy_range
                else "normal_0_to_1"
            )
            left_trigger_pressed = (
                _normalize_trigger_value(left_trigger_mgr, trigger_event_range)
                >= brainco_trigger_threshold
            )
            right_trigger_pressed = (
                _normalize_trigger_value(right_trigger_mgr, trigger_event_range)
                >= brainco_trigger_threshold
            )
            left_trigger_press = left_trigger_pressed and not prev_left_trigger_pressed
            left_trigger_release = not left_trigger_pressed and prev_left_trigger_pressed
            right_trigger_press = right_trigger_pressed and not prev_right_trigger_pressed
            right_trigger_release = not right_trigger_pressed and prev_right_trigger_pressed
            prev_left_trigger_pressed = left_trigger_pressed
            prev_right_trigger_pressed = right_trigger_pressed
            manager_state_data = {
                "stream_mode": np.array([current_mode.value], dtype=np.int32),
                "toggle_data_collection": np.array([toggle_dc], dtype=bool),
                "toggle_data_abort": np.array([toggle_da], dtype=bool),
                "grip_marker": np.array([grip_marker], dtype=bool),
                "left_trigger_press": np.array([left_trigger_press], dtype=bool),
                "left_trigger_release": np.array([left_trigger_release], dtype=bool),
                "right_trigger_press": np.array([right_trigger_press], dtype=bool),
                "right_trigger_release": np.array([right_trigger_release], dtype=bool),
                "left_brainco_q": left_brainco_q.astype(np.float32),
                "right_brainco_q": right_brainco_q.astype(np.float32),
            }
            manager_state_data.update(g1_state.snapshot_fields())
            socket.send(
                pack_pose_message(
                    manager_state_data,
                    topic="manager_state",
                )
            )

            prev_ax_pressed = ax_pressed
            prev_by_pressed = by_pressed
            prev_start_combo = start_combo
            prev_left_axis_click = left_axis_click

    except KeyboardInterrupt:
        print("\nStopping manager...")
    finally:
        # Cleanup resources
        g1_state.close()
        if brainco_hand is not None:
            brainco_hand.close()
        reader.stop()
        three_point.close()
        socket.close()
        context.term()
        print("[Manager] Shutdown complete")
