"""Command-line interface for the Pico/BrainCo manager."""

import argparse

import numpy as np

from .constants import (
    BRAINCO_CLOSED_Q, BRAINCO_DEFAULT_EXCLUDED_FINGERS,
    BRAINCO_EXCLUDED_FINGER_CHOICES, BRAINCO_MOTOR_NAMES, BRAINCO_TAU_THRESHOLDS,
    BRAINCO_TRIGGER_RANGE_CHOICES, BRAINCO_TRIGGER_THRESHOLD,
)
from .manager import run_pico_manager
from .pose_runner import run_pico
from .visualization import (
    run_vr3pt_live_visualizer, run_vr3pt_realtime_visualizer,
    run_vr3pt_visualizer_test,
)

def main(argv=None) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer_size", type=int, default=15, help="Sliding window buffer size")
    parser.add_argument("--port", type=int, default=5556, help="ZMQ server port (default: 5556)")
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Number of frames to send (default: 200)"
    )
    parser.add_argument("--target_fps", type=int, default=50, help="Target loop FPS (default: 50)")
    parser.add_argument(
        "--cuda", action="store_true", help="Use CUDA for tensors and model (default: CPU)"
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default="",
        help="Directory to save sent batches (default: disabled)",
    )
    parser.add_argument(
        "--record_format",
        type=str,
        default="npz",
        help="Recording format: 'npz' or 'bin' (default: npz)",
    )
    parser.add_argument(
        "--manager",
        action="store_true",
        help="Run manager with planner and pose threads (interactive)",
    )
    parser.add_argument(
        "--zmq_feedback_host",
        type=str,
        default="localhost",
        help="ZMQ feedback host (default: localhost)",
    )
    parser.add_argument(
        "--zmq_feedback_port",
        type=int,
        default=5557,
        help="ZMQ feedback port (default: 5557)",
    )
    parser.add_argument(
        "--vr3pt_test",
        action="store_true",
        help="Run VR 3-point pose visualizer test (reference frames only)",
    )
    parser.add_argument(
        "--vr3pt_live",
        action="store_true",
        help="Capture one frame of VR 3-point pose and visualize with reference frames",
    )
    parser.add_argument(
        "--vr3pt_realtime",
        action="store_true",
        help="Run standalone real-time VR 3-point pose visualizer",
    )
    parser.add_argument(
        "--vis_vr3pt",
        action="store_true",
        help="Enable inline VR 3-point pose visualization in pose streaming mode",
    )
    parser.add_argument(
        "--vr3pt_hz",
        type=int,
        default=10,
        help="Update rate for real-time VR visualization in Hz (default: 10)",
    )
    parser.add_argument(
        "--no_g1",
        action="store_true",
        help="Disable G1 robot visualization in VR 3pt pose view (G1 is shown by default)",
    )
    parser.add_argument(
        "--waist_tracking",
        action="store_true",
        help="Enable G1 robot waist to follow VR head orientation (disabled by default for performance)",
    )
    parser.add_argument(
        "--vis_smpl",
        action="store_true",
        help="Enable SMPL body joint visualization (24 joint spheres) in the VR3pt viewer",
    )
    parser.add_argument(
        "--input-source",
        type=str,
        default="xrt",
        choices=["xrt", "isaac-teleop"],
        help=(
            "Input source: 'xrt' for XRoboToolkit SDK (default), "
            "'isaac-teleop' for in-process IsaacTeleop / CloudXR DeviceIO"
        ),
    )
    parser.add_argument(
        "--disable_brainco_hand",
        action="store_true",
        help="Disable direct BrainCo hand DDS control",
    )
    parser.add_argument(
        "--brainco_dds_domain",
        type=int,
        default=0,
        help="DDS domain id for BrainCo hand topics (default: 0)",
    )
    parser.add_argument(
        "--brainco_network_interface",
        type=str,
        default="enP8p1s0",
        help="Network interface for BrainCo DDS, e.g. eth0 (default: auto)",
    )
    parser.add_argument(
        "--brainco_trigger_threshold",
        type=float,
        default=BRAINCO_TRIGGER_THRESHOLD,
        help="Normalized trigger threshold for closing BrainCo hands (default: 0.5)",
    )
    parser.add_argument(
        "--brainco_trigger_range",
        type=str,
        default="auto",
        choices=BRAINCO_TRIGGER_RANGE_CHOICES,
        help=(
            "Trigger range for BrainCo hand control: auto, normal_0_to_1, "
            "or legacy_10_to_0 (default: auto)"
        ),
    )
    parser.add_argument(
        "--brainco_excluded_fingers",
        nargs="*",
        choices=BRAINCO_EXCLUDED_FINGER_CHOICES,
        default=list(BRAINCO_DEFAULT_EXCLUDED_FINGERS),
        metavar="FINGER",
        help=(
            "BrainCo fingers that always receive q=0. Motor names are "
            "{left,right}_{thumb,thumb_aux,index,middle,ring,pinky}. "
            "Default: right_thumb right_thumb_aux. Pass the option without "
            "values to control every finger."
        ),
    )
    for motor_index, motor_name in enumerate(BRAINCO_MOTOR_NAMES):
        parser.add_argument(
            f"--brainco_{motor_name}_target",
            type=float,
            default=float(BRAINCO_CLOSED_Q[motor_index]),
            help=f"Closing q target for {motor_name} (range: 0..1)",
        )
        parser.add_argument(
            f"--brainco_{motor_name}_tau_threshold",
            type=float,
            default=float(BRAINCO_TAU_THRESHOLDS[motor_index]),
            help=f"Raw tau_est contact threshold for {motor_name}",
        )
    args = parser.parse_args(argv)

    brainco_target_q = np.array(
        [getattr(args, f"brainco_{name}_target") for name in BRAINCO_MOTOR_NAMES],
        dtype=np.float32,
    )
    brainco_tau_thresholds = np.array(
        [
            getattr(args, f"brainco_{name}_tau_threshold")
            for name in BRAINCO_MOTOR_NAMES
        ],
        dtype=np.float32,
    )
    if np.any((brainco_target_q < 0.0) | (brainco_target_q > 1.0)):
        parser.error("BrainCo finger targets must be between 0 and 1")
    if np.any(brainco_tau_thresholds < 0.0):
        parser.error("BrainCo tau thresholds must be non-negative")

    # Standalone VR3Pt test modes (exit after finishing)
    if args.vr3pt_test:
        print("Running VR 3-point pose visualizer test...")
        run_vr3pt_visualizer_test()
        print("VR 3-point pose visualizer test completed")
        exit(0)

    if args.vr3pt_live:
        print("Running VR 3-point pose live capture...")
        run_vr3pt_live_visualizer()
        print("VR 3-point pose live visualizer completed")
        exit(0)

    if args.vr3pt_realtime:
        print("Running VR 3-point pose real-time visualizer...")
        run_vr3pt_realtime_visualizer(update_hz=args.vr3pt_hz)
        print("VR 3-point pose real-time visualizer completed")
        exit(0)

    # Main execution modes
    # G1 robot visualization is enabled by default when vis_vr3pt is used
    with_g1_robot = not args.no_g1

    if args.manager:
        run_pico_manager(
            port=args.port,
            buffer_size=args.buffer_size,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            zmq_feedback_host=args.zmq_feedback_host,
            zmq_feedback_port=args.zmq_feedback_port,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            input_source=args.input_source,
            enable_brainco_hand=not args.disable_brainco_hand,
            brainco_dds_domain_id=args.brainco_dds_domain,
            brainco_network_interface=args.brainco_network_interface,
            brainco_trigger_threshold=args.brainco_trigger_threshold,
            brainco_trigger_range=args.brainco_trigger_range,
            brainco_target_q=brainco_target_q,
            brainco_tau_thresholds=brainco_tau_thresholds,
            brainco_excluded_fingers=args.brainco_excluded_fingers,
        )
    else:
        # Run legacy single-thread pose streaming
        run_pico(
            buffer_size=args.buffer_size,
            port=args.port,
            num_frames_to_send=args.num_frames_to_send,
            target_fps=args.target_fps,
            use_cuda=args.cuda,
            record_dir=args.record_dir,
            record_format=args.record_format,
            enable_vis_vr3pt=args.vis_vr3pt,
            with_g1_robot=with_g1_robot,
            enable_waist_tracking=args.waist_tracking,
            enable_smpl_vis=args.vis_smpl,
            input_source=args.input_source,
            enable_brainco_hand=not args.disable_brainco_hand,
            brainco_dds_domain_id=args.brainco_dds_domain,
            brainco_network_interface=args.brainco_network_interface,
            brainco_trigger_threshold=args.brainco_trigger_threshold,
            brainco_trigger_range=args.brainco_trigger_range,
            brainco_target_q=brainco_target_q,
            brainco_tau_thresholds=brainco_tau_thresholds,
            brainco_excluded_fingers=args.brainco_excluded_fingers,
        )
