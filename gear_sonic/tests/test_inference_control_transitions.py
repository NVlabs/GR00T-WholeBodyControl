from gear_sonic.utils.inference.control_transitions import (
    InferenceControlState,
    plan_i_transition,
    plan_k_transition,
    plan_p_transition,
)


def test_i_from_standing_off_starts_planner_ramps_calib_and_holds_planner_mode() -> None:
    transition = plan_i_transition(InferenceControlState())

    assert transition.start_planner is True
    assert transition.publish_calib_full is True
    assert transition.start_pose is False
    assert transition.publish_latent_initial is False
    assert transition.clear_action_cache is True
    assert transition.reset_frame_counter is True
    assert transition.next_state.pause_loop is True
    assert transition.next_state.cpp_loop_running is True
    assert transition.next_state.cpp_mode == "PLANNER"
    assert transition.next_state.initial_pose_ready is True


def test_p_from_prepared_planner_starts_policy_and_defers_pose_until_action() -> None:
    state = InferenceControlState(
        pause_loop=True,
        cpp_loop_running=True,
        cpp_mode="PLANNER",
        initial_pose_ready=True,
    )

    transition = plan_p_transition(state)

    assert transition.publish_calib_full is False
    assert transition.start_pose is False
    assert transition.publish_latent_initial is False
    assert transition.start_pose_on_next_action is True
    assert transition.clear_action_cache is True
    assert transition.next_state.pause_loop is False
    assert transition.next_state.cpp_mode == "PLANNER"
    assert transition.next_state.initial_pose_ready is True


def test_p_from_prepared_pose_starts_policy_without_extra_ramp() -> None:
    state = InferenceControlState(
        pause_loop=True,
        cpp_loop_running=True,
        cpp_mode="POSE",
        initial_pose_ready=True,
    )

    transition = plan_p_transition(state)

    assert transition.publish_calib_full is False
    assert transition.start_pose is False
    assert transition.next_state.pause_loop is False
    assert transition.next_state.cpp_mode == "POSE"
    assert transition.next_state.initial_pose_ready is True


def test_p_while_running_pauses_and_returns_to_calib_full_planner_mode() -> None:
    state = InferenceControlState(
        pause_loop=False,
        cpp_loop_running=True,
        cpp_mode="POSE",
        initial_pose_ready=True,
    )

    transition = plan_p_transition(state)

    assert transition.start_planner is True
    assert transition.publish_calib_full is True
    assert transition.clear_action_cache is True
    assert transition.next_state.pause_loop is True
    assert transition.next_state.cpp_loop_running is True
    assert transition.next_state.cpp_mode == "PLANNER"
    assert transition.next_state.initial_pose_ready is True


def test_k_from_paused_calib_full_ramps_to_standing_before_stop() -> None:
    state = InferenceControlState(
        pause_loop=True,
        cpp_loop_running=True,
        cpp_mode="PLANNER",
        initial_pose_ready=True,
    )

    transition = plan_k_transition(state)

    assert transition.publish_standing is True
    assert transition.stop_control is True
    assert transition.clear_action_cache is True
    assert transition.next_state.pause_loop is True
    assert transition.next_state.cpp_loop_running is False
    assert transition.next_state.cpp_mode == "OFF"
    assert transition.next_state.initial_pose_ready is False
