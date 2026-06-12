"""Keyboard transition planning for SONIC VLA inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CppMode = Literal["OFF", "PLANNER", "POSE"]


@dataclass(frozen=True)
class InferenceControlState:
    """Small state model for VLA inference keyboard transitions."""

    pause_loop: bool = True
    cpp_loop_running: bool = False
    cpp_mode: CppMode = "OFF"
    initial_pose_ready: bool = False


@dataclass(frozen=True)
class KeyboardTransition:
    """Actions and resulting state for one keyboard transition."""

    next_state: InferenceControlState
    start_planner: bool = False
    start_pose: bool = False
    stop_control: bool = False
    publish_calib_full: bool = False
    publish_standing: bool = False
    publish_latent_initial: bool = False
    start_pose_on_next_action: bool = False
    clear_action_cache: bool = False
    reset_frame_counter: bool = False
    blocked_reason: str = ""


def plan_i_transition(state: InferenceControlState) -> KeyboardTransition:
    """Prepare the robot at CALIB_FULL and hold planner mode."""

    return KeyboardTransition(
        next_state=InferenceControlState(
            pause_loop=True,
            cpp_loop_running=True,
            cpp_mode="PLANNER",
            initial_pose_ready=True,
        ),
        start_planner=not state.cpp_loop_running or state.cpp_mode != "PLANNER",
        publish_calib_full=True,
        clear_action_cache=True,
        reset_frame_counter=True,
    )


def plan_p_transition(state: InferenceControlState) -> KeyboardTransition:
    """Toggle policy streaming with CALIB_FULL return on pause."""

    if state.pause_loop:
        if not state.cpp_loop_running:
            return KeyboardTransition(
                next_state=state,
                blocked_reason="C++ control loop is not running; press 'i' first.",
            )
        if not state.initial_pose_ready:
            return KeyboardTransition(
                next_state=state,
                blocked_reason="Initial CALIB_FULL pose is not ready; press 'i' first.",
            )
        return KeyboardTransition(
            next_state=InferenceControlState(
                pause_loop=False,
                cpp_loop_running=True,
                cpp_mode=state.cpp_mode,
                initial_pose_ready=True,
            ),
            start_pose=state.cpp_mode == "OFF",
            start_pose_on_next_action=state.cpp_mode == "PLANNER",
            clear_action_cache=state.cpp_mode != "POSE",
        )

    return KeyboardTransition(
        next_state=InferenceControlState(
            pause_loop=True,
            cpp_loop_running=state.cpp_loop_running,
            cpp_mode="PLANNER" if state.cpp_loop_running else "OFF",
            initial_pose_ready=state.cpp_loop_running,
        ),
        start_planner=state.cpp_loop_running and state.cpp_mode != "PLANNER",
        publish_calib_full=state.cpp_loop_running,
        clear_action_cache=True,
    )


def plan_k_transition(state: InferenceControlState) -> KeyboardTransition:
    """Start planner from OFF or ramp back to standing before stopping."""

    if not state.cpp_loop_running:
        return KeyboardTransition(
            next_state=InferenceControlState(
                pause_loop=True,
                cpp_loop_running=True,
                cpp_mode="PLANNER",
                initial_pose_ready=False,
            ),
            start_planner=True,
        )

    return KeyboardTransition(
        next_state=InferenceControlState(
            pause_loop=True,
            cpp_loop_running=False,
            cpp_mode="OFF",
            initial_pose_ready=False,
        ),
        start_planner=state.cpp_mode != "PLANNER",
        publish_standing=True,
        stop_control=True,
        clear_action_cache=True,
    )
