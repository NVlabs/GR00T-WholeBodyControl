"""
Validate Quest raw tracking data — visualize with or without calibration.

Reads controller poses from meta_quest_teleop and displays them in a 3D PyVista window.
Use this to verify Quest tracking and test the G1-based calibration flow.

Modes:
  - Raw (default): No calibration. Verifies that Quest tracking follows correctly.
  - Calibrated (--calibrated): Use G1 FK default as reference. Press right trigger
    to calibrate; then output = G1_default + delta from calibration pose.

Usage:
  # USB connection, raw mode
  python -m gear_sonic.utils.teleop.readers.validate_quest_raw

  # WiFi (Quest IP)
  python -m gear_sonic.utils.teleop.readers.validate_quest_raw --ip-address 192.168.x.x

  # Calibrated mode: adopt G1 default pose, press right trigger, then move
  python -m gear_sonic.utils.teleop.readers.validate_quest_raw --calibrated

  # Run for 30 seconds then exit
  python -m gear_sonic.utils.teleop.readers.validate_quest_raw --duration 30

  # Update rate (default 50 Hz)
  python -m gear_sonic.utils.teleop.readers.validate_quest_raw --hz 50

Press Ctrl+C or close the window to exit.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root for imports
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Validate Quest tracking — raw or calibrated (G1 FK reference)"
    )
    parser.add_argument(
        "--ip-address",
        type=str,
        default=None,
        help="Quest IP for WiFi (omit for USB)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=50.0,
        help="Update rate in Hz (default 50)",
    )
    parser.add_argument(
        "--no-g1",
        action="store_true",
        help="Hide G1 robot mesh (show only VR pose markers)",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Use G1 FK default as calibration reference. Press right trigger to calibrate.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run for N seconds then exit (0 = run until Ctrl+C)",
    )
    args = parser.parse_args()

    # Import meta_quest_teleop
    try:
        from meta_quest_teleop.reader import MetaQuestReader  # pyright: ignore[reportMissingImports]
    except ImportError:
        print("[ERROR] meta_quest_teleop no instalado. Ejecuta:")
        print("  bash install_scripts/install_pico.sh")
        print("  o: pip install 'meta_quest_teleop @ git+https://github.com/BrikHMP18/meta_quest_teleop.git@07bc15437f767c3517138367b6b3e3910b388c76'")
        sys.exit(1)

    from gear_sonic.utils.teleop.readers.quest_reader import (
        quest_controllers_to_vr_3pt_pose,
        _g1_pose_to_4x4,
        _apply_quest_yz_swap,
    )

    try:
        from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import (
            VR3PtPoseVisualizer,
            get_g1_key_frame_poses,
        )
    except ImportError as e:
        print(f"[ERROR] VR3PtPoseVisualizer no disponible: {e}")
        print("  Instala pyvista: pip install pyvista")
        sys.exit(1)

    # Load G1 model for calibrated mode
    home_left_4x4 = None
    home_right_4x4 = None
    robot_model = None
    if args.calibrated:
        try:
            from gear_sonic.data.robot_model.instantiation.g1 import (
                instantiate_g1_robot_model,
            )

            robot_model = instantiate_g1_robot_model()
            g1_poses = get_g1_key_frame_poses(robot_model)
            home_left_4x4 = _g1_pose_to_4x4(
                g1_poses["left_wrist"]["position"],
                g1_poses["left_wrist"]["orientation_wxyz"],
            )
            home_right_4x4 = _g1_pose_to_4x4(
                g1_poses["right_wrist"]["position"],
                g1_poses["right_wrist"]["orientation_wxyz"],
            )
        except Exception as e:
            print(f"[ERROR] No se pudo cargar G1 para modo calibrado: {e}")
            sys.exit(1)

    print("=" * 60)
    print("Quest Tracking Validation")
    print("=" * 60)
    print("Conectando a Quest...")
    if args.calibrated:
        print("  Modo: CALIBRADO (G1 FK como referencia)")
        print("  - Adopta la pose por defecto del G1 (muñecas como el robot)")
        print("  - Pulsa trigger derecho para calibrar")
        print("  - Luego mueve: output = G1_default + delta")
    else:
        print("  Modo: RAW (sin calibración)")
        print("  - Datos crudos del Quest")
        print("  - Si sigue bien → problema en calibración/mapeo")
        print("  - Si no sigue → problema en meta_quest_teleop o Quest")
    if args.duration > 0:
        print(f"  Duración: {args.duration}s")
    print("=" * 60)

    reader = MetaQuestReader(ip_address=args.ip_address, run=True)
    dt = 1.0 / max(1.0, args.hz)

    visualizer = VR3PtPoseVisualizer(
        axis_length=0.08,
        ball_radius=0.015,
        with_g1_robot=not args.no_g1,
        robot_model=robot_model,
    )
    visualizer.create_realtime_plotter(interactive=True)

    # Calibration state (for --calibrated mode)
    left_offset = None
    right_offset = None
    is_calibrated = False
    prev_trigger_pressed = False
    calibration_threshold = 0.5

    last_status = 0.0
    frame_count = 0
    fps_ema = 0.0
    last_stamp = time.monotonic()
    start_time = time.time()

    try:
        while visualizer.is_open:
            t_now = time.monotonic()
            device_dt = t_now - last_stamp
            if device_dt > 0:
                inst_fps = 1.0 / device_dt
                fps_ema = inst_fps if fps_ema == 0.0 else (0.9 * fps_ema + 0.1 * inst_fps)
            last_stamp = t_now

            left_raw = reader.get_hand_controller_transform_ros("left")
            right_raw = reader.get_hand_controller_transform_ros("right")
            # Correct Y/Z swap: Quest APK maps up to Y; we need Z up (ROS convention)
            if left_raw is not None:
                left_raw = _apply_quest_yz_swap(left_raw)
            if right_raw is not None:
                right_raw = _apply_quest_yz_swap(right_raw)
            trigger = float(reader.get_trigger_value("right"))
            trigger_pressed = trigger >= calibration_threshold

            # Calibrate on rising edge (calibrated mode only)
            if args.calibrated and trigger_pressed and not prev_trigger_pressed:
                if (
                    left_raw is not None
                    and right_raw is not None
                    and home_left_4x4 is not None
                    and home_right_4x4 is not None
                ):
                    try:
                        left_offset = home_left_4x4 @ np.linalg.inv(left_raw)
                        right_offset = home_right_4x4 @ np.linalg.inv(right_raw)
                        is_calibrated = True
                        print("\n[validate] Calibración completada (trigger derecho). Ref=G1 FK default.")
                    except np.linalg.LinAlgError:
                        pass
            prev_trigger_pressed = trigger_pressed

            # Build vr_3pt_pose
            if (
                args.calibrated
                and is_calibrated
                and left_offset is not None
                and right_offset is not None
                and left_raw is not None
                and right_raw is not None
            ):
                left_pose = left_offset @ left_raw
                right_pose = right_offset @ right_raw
                vr_3pt_pose = quest_controllers_to_vr_3pt_pose(left_pose, right_pose)
            else:
                vr_3pt_pose = quest_controllers_to_vr_3pt_pose(left_raw, right_raw)

            visualizer.update_vr_poses(vr_3pt_pose)
            visualizer.render()

            frame_count += 1
            now = time.time()
            if now - last_status >= 2.0:
                status = "L:OK" if left_raw is not None else "L:--"
                status += " R:OK" if right_raw is not None else " R:--"
                if left_raw is not None:
                    pos = left_raw[:3, 3]
                    status += f" | L=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
                if right_raw is not None:
                    pos = right_raw[:3, 3]
                    status += f" R=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
                calib_str = " [CALIB]" if (args.calibrated and is_calibrated) else ""
                status += f" | {fps_ema:.1f} Hz | frames={frame_count}{calib_str}"
                print(f"\r[validate] {status}   ", end="", flush=True)
                last_status = now

            # Duration limit
            if args.duration > 0 and (now - start_time) >= args.duration:
                print(f"\n[validate] Duración {args.duration}s alcanzada.")
                break

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[validate] Interrumpido por usuario.")
    finally:
        reader.stop()
        visualizer.close()
        print("[validate] Listo.")


if __name__ == "__main__":
    main()
