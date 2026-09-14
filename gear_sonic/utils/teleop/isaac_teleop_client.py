"""Isaac Teleop client wrapping CloudXR + DeviceIO + OpenXR for the host.

Launches the CloudXR runtime in-process (via ``CloudXRLauncher``), opens an
OpenXR session, and starts the DeviceIO trackers (head, hands, controllers,
full body). Provides synchronous getters that the gear_sonic teleop
readers poll on a background thread.

This replaces the legacy multi-container path (``run_cloudxr_via_docker.sh``
plus the ROS2 ``teleop_ros2_ref`` publisher) — see
``docs/source/tutorials/isaac_teleop_publisher_setup.md`` for the in-process
setup. Requires ``isaacteleop[cloudxr]`` from ``pypi.nvidia.com`` (installed
by ``install_scripts/install_pico.sh``).
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

import isaacteleop.deviceio as deviceio
import isaacteleop.oxr as oxr
from isaacteleop.cloudxr import CloudXRLauncher


def _default_pose_vec() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _controller_pose_vec(controller_data: Any) -> np.ndarray:
    """``[x, y, z, qx, qy, qz, qw]`` from ``ControllerSnapshot``; aim preferred, then grip."""
    if controller_data is None:
        return _default_pose_vec()
    controller_pose = None
    aim_pose = controller_data.aim_pose
    if aim_pose is not None and aim_pose.is_valid:
        controller_pose = aim_pose
    else:
        grip_pose = controller_data.grip_pose
        if grip_pose is not None and grip_pose.is_valid:
            controller_pose = grip_pose
    if controller_pose is None:
        return _default_pose_vec()
    p = controller_pose.pose.position
    o = controller_pose.pose.orientation
    pos = np.array([p.x, p.y, p.z], dtype=np.float64)
    quat = np.array([o.x, o.y, o.z, o.w], dtype=np.float64)
    return np.concatenate([pos, quat])


def _pose_vec_from_head_data(head_pose: Any) -> np.ndarray:
    if head_pose is None or not head_pose.is_valid:
        return _default_pose_vec()
    p = head_pose.pose.position
    o = head_pose.pose.orientation
    return np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float64)


def _controller_inputs(snapshot: Any) -> Any | None:
    if snapshot is None:
        return None
    return snapshot.inputs


class IsaacTeleopClient:
    """Single-process CloudXR + DeviceIO + OpenXR session.

    Args:
        app_name: OpenXR application name (shows up in CloudXR runtime logs).
        use_adb: If True, route signalling/WebRTC over a USB ADB tunnel
            (``setup_oob`` + ``usb_local`` on the launcher) so the headset
            reaches the host on loopback without needing shared Wi-Fi.
            Requires ``adb`` and ``coturn`` on PATH.
        cloudxr_install_dir: Where to install/find the CloudXR runtime.
            Defaults to ``~/.cloudxr``.
        cloudxr_env_config: Path to ``cloudxr.env`` (selects ``NV_DEVICE_PROFILE``).
            Defaults to ``~/cloudxr.env``; created by ``install_pico.sh``.
    """

    def __init__(
        self,
        app_name: str = "GearSonicIsaacTeleopClient",
        use_adb: bool = False,
        cloudxr_install_dir: str | Path | None = None,
        cloudxr_env_config: str | Path | None = None,
    ) -> None:
        self._app_name = app_name
        self._use_adb = bool(use_adb)
        self._cloudxr_install_dir = str(
            cloudxr_install_dir if cloudxr_install_dir is not None else Path.home() / ".cloudxr"
        )
        self._cloudxr_env_config = str(
            cloudxr_env_config if cloudxr_env_config is not None else Path.home() / "cloudxr.env"
        )

        self._exit_stack: ExitStack | None = None
        self._deviceio_session: Any = None
        self._head_tracker: Any = None
        self._hand_tracker: Any = None
        self._controller_tracker: Any = None
        self._body_tracker: Any = None
        self._cloudxr_launcher: CloudXRLauncher | None = None

    def _clear_trackers_and_session_ref(self) -> None:
        """Clear held refs after ``ExitStack.close()`` or a failed connect."""
        if self._cloudxr_launcher is not None:
            try:
                self._cloudxr_launcher.stop()
            except Exception:
                pass
            self._cloudxr_launcher = None
        self._deviceio_session = None
        self._head_tracker = None
        self._hand_tracker = None
        self._controller_tracker = None
        self._body_tracker = None

    def start_streaming(self) -> None:
        """Launch CloudXR + open OpenXR session + start DeviceIO trackers."""
        stack = ExitStack()
        try:
            # OOB hub + USB-local (when use_adb=True): route signalling and
            # WebRTC media over the USB cable via `adb reverse`, so the headset
            # reaches the host on loopback without needing shared Wi-Fi. Needs
            # `coturn` and `adb` on PATH.
            self._cloudxr_launcher = CloudXRLauncher(
                install_dir=self._cloudxr_install_dir,
                env_config=self._cloudxr_env_config,
                accept_eula=True,
                setup_oob=self._use_adb,
                usb_local=self._use_adb,
            )

            self._head_tracker = deviceio.HeadTracker()
            self._hand_tracker = deviceio.HandTracker()
            self._controller_tracker = deviceio.ControllerTracker()
            self._body_tracker = deviceio.FullBodyTracker()
            trackers = [
                self._head_tracker,
                self._hand_tracker,
                self._controller_tracker,
                self._body_tracker,
            ]
            required_extensions = deviceio.DeviceIOSession.get_required_extensions(trackers)
            oxr_session = stack.enter_context(
                oxr.OpenXRSession(self._app_name, required_extensions)
            )
            handles = oxr_session.get_handles()
            self._deviceio_session = stack.enter_context(
                deviceio.DeviceIOSession.run(trackers, handles)
            )
            self._exit_stack = stack
            print("Isaac Teleop session initialized.")

        except RuntimeError as e:
            stack.close()
            self._exit_stack = None
            self._clear_trackers_and_session_ref()
            if "Failed to get OpenXR system" in str(e) or "OpenXR" in str(e):
                print(f"IsaacTeleopClient: no XR session yet ({e}).")
            else:
                raise
        except Exception as e:
            stack.close()
            self._exit_stack = None
            self._clear_trackers_and_session_ref()
            print(f"IsaacTeleopClient: failed to start sessions ({e}).")

    def _get_tracker_data(self) -> dict[str, Any] | None:
        """Poll current tracking data and return it as a dictionary.

        Returns:
            Dict with keys ``left_controller``, ``right_controller``, ``head``,
            ``left_hand``, ``right_hand``, ``full_body``. Each value is the
            corresponding tracker's ``.data`` payload (raw DeviceIO type).
        """
        if self._deviceio_session is None:
            return None
        try:
            self._deviceio_session.update()
        except RuntimeError as e:
            print(f"IsaacTeleopClient: DeviceIO update failed ({e}); closing.")
            self.close()
            return None

        session = self._deviceio_session
        return {
            "left_controller": self._controller_tracker.get_left_controller(session).data,
            "right_controller": self._controller_tracker.get_right_controller(session).data,
            "head": self._head_tracker.get_head(session).data,
            "left_hand": self._hand_tracker.get_left_hand(session).data,
            "right_hand": self._hand_tracker.get_right_hand(session).data,
            "full_body": self._body_tracker.get_body_pose(session).data,
        }

    def get_pose_by_name(self, name: str) -> np.ndarray:
        """Return ``[x, y, z, qx, qy, qz, qw]`` for ``name`` ∈ {left_controller, right_controller, headset}."""
        raw = self._get_tracker_data()
        if raw is None:
            return _default_pose_vec()

        if name == "left_controller":
            return _controller_pose_vec(raw.get("left_controller"))
        if name == "right_controller":
            return _controller_pose_vec(raw.get("right_controller"))
        if name == "headset":
            return _pose_vec_from_head_data(raw.get("head"))
        raise ValueError(
            f"Invalid name: {name}. Valid names: 'left_controller', 'right_controller', 'headset'."
        )

    def _snapshot_side(self, raw: dict[str, Any] | None, side: str) -> Any:
        if raw is None:
            return None
        key = "left_controller" if side == "left" else "right_controller"
        return raw.get(key)

    def get_key_value_by_name(self, name: str) -> float:
        """Return trigger/grip value for ``name`` ∈ {left,right}_{trigger,grip}."""
        raw = self._get_tracker_data()
        if name == "left_trigger":
            inp = _controller_inputs(self._snapshot_side(raw, "left"))
            return float(inp.trigger_value) if inp is not None else 0.0
        if name == "right_trigger":
            inp = _controller_inputs(self._snapshot_side(raw, "right"))
            return float(inp.trigger_value) if inp is not None else 0.0
        if name == "left_grip":
            inp = _controller_inputs(self._snapshot_side(raw, "left"))
            return float(inp.squeeze_value) if inp is not None else 0.0
        if name == "right_grip":
            inp = _controller_inputs(self._snapshot_side(raw, "right"))
            return float(inp.squeeze_value) if inp is not None else 0.0
        raise ValueError(
            f"Invalid name: {name}. Valid names: "
            "'left_trigger', 'right_trigger', 'left_grip', 'right_grip'."
        )

    def get_button_state_by_name(self, name: str) -> bool:
        """Return True/False for face buttons and stick clicks.

        Valid names: ``A``, ``B``, ``X``, ``Y``,
        ``left_menu_button``, ``right_menu_button``,
        ``left_axis_click``, ``right_axis_click``.
        """
        raw = self._get_tracker_data()
        left = _controller_inputs(self._snapshot_side(raw, "left"))
        right = _controller_inputs(self._snapshot_side(raw, "right"))

        if name == "A":
            return right is not None and float(right.primary_click) > 0.5
        if name == "B":
            return right is not None and float(right.secondary_click) > 0.5
        if name == "X":
            return left is not None and float(left.primary_click) > 0.5
        if name == "Y":
            return left is not None and float(left.secondary_click) > 0.5
        if name in ("left_menu_button", "right_menu_button"):
            # Pico-specific menu button is not exposed via standard OpenXR
            # input bindings — DeviceIO doesn't surface it for the supported
            # controllers, so report not-pressed.
            return False
        if name == "left_axis_click":
            return left is not None and float(left.thumbstick_click) > 0.5
        if name == "right_axis_click":
            return right is not None and float(right.thumbstick_click) > 0.5
        raise ValueError(
            f"Invalid name: {name}. Valid names: 'A', 'B', 'X', 'Y', "
            "'left_menu_button', 'right_menu_button', 'left_axis_click', 'right_axis_click'."
        )

    def get_joystick_state(self, controller: str) -> list[float]:
        """Return ``[x, y]`` joystick state for ``controller`` ∈ {left, right}."""
        side = controller.lower()
        if side not in ("left", "right"):
            raise ValueError(
                f"Invalid controller: {controller}. Valid controllers: 'left', 'right'."
            )

        raw = self._get_tracker_data()
        inp = _controller_inputs(self._snapshot_side(raw, side))
        if inp is None:
            return [0.0, 0.0]
        return [float(inp.thumbstick_x), float(inp.thumbstick_y)]

    def get_full_body_data(self) -> Any | None:
        """Return the raw DeviceIO ``FullBodyTracker`` data payload (or None).

        Body-joint extraction (24×7 pose array) lives in
        ``input_readers.IsaacTeleopReader``, which knows the gear_sonic schema.
        """
        raw = self._get_tracker_data()
        if raw is None:
            return None
        return raw.get("full_body")

    def get_timestamp_ns(self) -> int:
        """Return the host monotonic timestamp in nanoseconds."""
        return int(time.monotonic_ns())

    def close(self) -> None:
        """Close OpenXR session, stop DeviceIO + CloudXR runtime."""
        if self._exit_stack is not None:
            try:
                self._exit_stack.close()
            except Exception:
                pass
            self._exit_stack = None
        self._clear_trackers_and_session_ref()


def main() -> None:
    """Poll ``IsaacTeleopClient`` getters periodically and print (Ctrl+C to stop).

    Usage::

        source .venv_teleop/bin/activate

        # Setup verification: bring CloudXR up, populate ~/.cloudxr/, exit clean
        python -m gear_sonic.utils.teleop.isaac_teleop_client --init-only

        # Live print (default): poll getters and print until Ctrl+C
        python -m gear_sonic.utils.teleop.isaac_teleop_client --hz 5
    """
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Print IsaacTeleopClient getter outputs at a fixed rate."
    )
    parser.add_argument("--hz", type=float, default=5.0, help="Print rate in Hz (default: 5)")
    parser.add_argument(
        "--use-adb", action="store_true", help="Route CloudXR over USB ADB (OOB / usb-local)."
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help=(
            "Bring CloudXR up to populate ~/.cloudxr/ ownership + env file, "
            "then exit. Use as a setup verification step."
        ),
    )
    args = parser.parse_args()

    client = IsaacTeleopClient(use_adb=args.use_adb)
    client.start_streaming()

    if args.init_only:
        # Confirm the runtime actually populated the install dir before we tear down.
        run_env = Path.home() / ".cloudxr" / "run" / "cloudxr.env"
        if run_env.exists():
            print(f"[OK] CloudXR runtime initialized; {run_env} written.")
            client.close()
            sys.exit(0)
        print(
            f"[ERROR] CloudXR runtime did not populate {run_env} — check the "
            "earlier log for IsaacTeleopClient errors.",
            file=sys.stderr,
        )
        client.close()
        sys.exit(1)

    period = 1.0 / max(0.1, float(args.hz))
    pose_names = ("left_controller", "right_controller", "headset")
    key_names = ("left_trigger", "right_trigger", "left_grip", "right_grip")
    button_names = (
        "A",
        "B",
        "X",
        "Y",
        "left_menu_button",
        "right_menu_button",
        "left_axis_click",
        "right_axis_click",
    )
    joy_sides = ("left", "right")

    try:
        while True:
            t0 = time.time()
            print("=" * 72, flush=True)
            print(
                f"time={time.strftime('%H:%M:%S')}  get_timestamp_ns={client.get_timestamp_ns()}",
                flush=True,
            )

            for name in pose_names:
                v = client.get_pose_by_name(name)
                print(
                    f"  get_pose_by_name({name!r}): "
                    f"{np.array2string(v, precision=4, suppress_small=True)}",
                    flush=True,
                )

            for name in key_names:
                print(
                    f"  get_key_value_by_name({name!r}): {client.get_key_value_by_name(name):.4f}",
                    flush=True,
                )

            for name in button_names:
                print(
                    f"  get_button_state_by_name({name!r}): {client.get_button_state_by_name(name)}",
                    flush=True,
                )

            for side in joy_sides:
                j = client.get_joystick_state(side)
                print(f"  get_joystick_state({side!r}): {j}", flush=True)

            dt = time.time() - t0
            time.sleep(max(0.0, period - dt))
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
