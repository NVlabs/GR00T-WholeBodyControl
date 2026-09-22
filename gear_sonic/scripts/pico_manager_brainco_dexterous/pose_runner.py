"""Standalone pose-stream runtime."""

import threading
import time

import numpy as np
import zmq

from .brainco import BraincoHandCommandPublisher
from .constants import (
    BRAINCO_DEFAULT_EXCLUDED_FINGERS, BRAINCO_TRIGGER_THRESHOLD,
)
from .controller_input import PicoReader, _init_input_source
from .g1_state import G1UpperBodyStateSubscriber, _create_g1_state_subscriber
from .pose_streamer import PoseStreamer
from .runtime import build_command_message, build_planner_message, xrt
from .tracking import ThreePointPose

def _pose_stream_common(
    socket,
    buffer_size: int,
    num_frames_to_send: int,
    target_fps: int,
    use_cuda: bool,
    record_dir: str,
    record_format: str,
    stop_event: threading.Event | None = None,
    log_prefix: str = "PoseLoop",
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
    enable_smpl_vis: bool = False,
    reader=None,
    brainco_hand: BraincoHandCommandPublisher | None = None,
    g1_state: G1UpperBodyStateSubscriber | None = None,
    brainco_trigger_threshold: float = BRAINCO_TRIGGER_THRESHOLD,
    brainco_trigger_range: str = "auto",
):
    """Shared pose streaming loop used by run_pico."""
    if reader is None:
        if xrt is None:
            raise ImportError(
                "XRoboToolkit SDK not available. Install xrobotoolkit_sdk to run pose streaming."
            )

        # Create reader and start it
        reader = PicoReader(max_queue_size=buffer_size)
        reader.start()

    # Create 3-point pose processor with visualization settings
    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=enable_smpl_vis,
        log_prefix=log_prefix,
    )

    streamer = PoseStreamer(
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
        log_prefix=log_prefix,
    )

    if stop_event is None:
        stop_event = threading.Event()

    try:
        while not stop_event.is_set():
            streamer.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup resources
        reader.stop()
        three_point.close()
def run_pico(
    buffer_size: int = 15,
    port: int = 5556,
    num_frames_to_send: int = 5,
    target_fps: int = 50,
    use_cuda: bool = False,
    record_dir: str = "",
    record_format: str = "npz",
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
    """Run body tracking with real-time visualization and ZMQ streaming."""
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
    print(f"ZMQ socket bound to port {port}")
    if build_command_message is not None and build_planner_message is not None:
        try:
            socket.send(build_command_message(start=False, stop=False, planner=False))
            socket.send(build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0))
        except Exception as e:
            print(f"Warning: failed to send initial command/planner messages: {e}")
    try:
        _pose_stream_common(
            socket=socket,
            buffer_size=buffer_size,
            num_frames_to_send=num_frames_to_send,
            target_fps=target_fps,
            use_cuda=use_cuda,
            record_dir=record_dir,
            record_format=record_format,
            stop_event=None,
            log_prefix="Main",
            enable_vis_vr3pt=enable_vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=enable_waist_tracking,
            enable_smpl_vis=enable_smpl_vis,
            reader=reader,
            brainco_hand=brainco_hand,
            g1_state=g1_state,
            brainco_trigger_threshold=brainco_trigger_threshold,
            brainco_trigger_range=brainco_trigger_range,
        )
    finally:
        g1_state.close()
        if brainco_hand is not None:
            brainco_hand.close()
        socket.close()
        context.term()
        print("Threads stopped, ZMQ socket closed")
