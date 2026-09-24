"""Isaac Teleop client wrapping CloudXR + DeviceIO + OpenXR for the host.

Launches the CloudXR runtime in-process (via ``CloudXRLauncher``), opens an
OpenXR session, and starts the DeviceIO trackers (head, hands, controllers,
full-body Pico). Provides synchronous getters that the gear_sonic teleop
readers poll on a background thread.

This replaces the legacy multi-container path (``run_cloudxr_via_docker.sh``
plus the ROS2 ``teleop_ros2_ref`` publisher) — see
``docs/source/tutorials/isaac_teleop_publisher_setup.md`` for the in-process
setup. Requires ``isaacteleop[cloudxr]`` from ``pypi.nvidia.com`` (installed
by ``install_scripts/install_pico.sh``).
"""

from __future__ import annotations

import os
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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


_WEBXR_BODY_VENDOR = "body.quest-cloudxr"
_BD_BODY_VENDOR = "body.pico-xr"


def _webxr_body_vendor_available() -> bool:
    """Whether this isaacteleop registers the WebXR/Quest full-body vendor.

    The WebXR full-body tracker arrived in IsaacTeleop #921. IsaacTeleop ships on
    a versioned cadence, so a deployment can easily be running a release that
    predates it, and asking for an unregistered vendor is not a soft failure:
    ``get_required_extensions`` raises ``ValueError`` before any session exists,
    which used to be swallowed by the broad ``except Exception`` in connect().
    The visible symptom was a single "failed to start sessions" line followed by
    waiting forever for body data -- indistinguishable from a network or headset
    problem. Probing instead lets BD/Pico body tracking (and the degeneracy
    guards built on it) work on any isaacteleop, with WebXR simply dormant.

    isaacteleop exposes no way to enumerate vendors, so this asks the factory to
    resolve the vendor for a throwaway tracker. ``get_required_extensions`` only
    inspects the tracker/vendor pairing -- it creates no session and has no side
    effects. Only the factory's "unknown vendor id" ValueError is treated as
    absence; anything else is a real fault and propagates.
    """
    probe = deviceio.FullBodyTracker()
    try:
        deviceio.DeviceIOSession.get_required_extensions(
            [probe],
            deviceio.VendorConfig([(probe, deviceio.TrackerVendor(_WEBXR_BODY_VENDOR))]),
        )
    except ValueError as exc:
        if "unknown vendor id" in str(exc):
            return False
        raise
    return True


@dataclass(frozen=True)
class TrackerSnapshot:
    """One poll of the DeviceIO trackers.

    A structure rather than a dict so the field names are declared in one place
    and a typo is an AttributeError here instead of a silent ``None`` at the far
    end of the reader.

    Every field is the raw isaacteleop payload for that tracker, or ``None``
    when it produced no data this frame. isaacteleop deliberately spells absence
    as ``None`` rather than an empty handle, so an inactive device cannot answer
    field reads with defaults indistinguishable from real zeroes; the consumers
    all treat ``None`` as "not present".
    """

    left_controller: Any = None
    right_controller: Any = None
    head: Any = None
    left_hand: Any = None
    right_hand: Any = None
    full_body: Any = None
    #: Which vendor extension delivered :attr:`full_body` -- ``"pico"``,
    #: ``"quest"``, or ``None`` when no body data arrived. This is the OpenXR
    #: Extension Method result: a fact the runtime states, not an inference.
    full_body_source: Optional[str] = None


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
        self._body_tracker_bd: Any = None
        self._body_tracker_meta: Any = None
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
        self._body_tracker_bd = None
        self._body_tracker_meta = None

    def start_streaming(self) -> None:
        """Launch CloudXR + open OpenXR session + start DeviceIO trackers."""
        stack = ExitStack()
        try:
            # OOB hub + USB-local (when use_adb=True): route signalling and
            # WebRTC media over the USB cable via `adb reverse`, so the headset
            # reaches the host on loopback without needing shared Wi-Fi. Needs
            # `coturn` and `adb` on PATH.
            # Opt in to CloudXR's WebXR full-body skeleton before the runtime is
            # launched. The check lives in the CloudXR server, i.e. the runtime
            # process, and that process inherits its environment when it is
            # started -- which CloudXRLauncher does below. Setting it after this
            # point (for instance next to the OpenXR session) is too late: the
            # runtime is already up with the variable absent, and it silently
            # ignores a client offering the full-body skeleton.
            #
            # isaacteleop sets the same default on the runtime's environment from
            # 1.5 on. setdefault means the two agree, this still works against an
            # externally started runtime, and an operator who exports the variable
            # -- including to "0" -- still wins.
            os.environ.setdefault("NV_CXR_ENABLE_NON_CONFORMANT_WEBXR_BODY_TRACKING", "1")

            self._cloudxr_launcher = CloudXRLauncher(
                install_dir=self._cloudxr_install_dir,
                env_config=self._cloudxr_env_config,
                accept_eula=True,
                setup_oob=self._use_adb,
                usb_local=self._use_adb,
                host_client=True,
            )
            try:
                import socket as _socket
                import subprocess as _sp
                from isaacteleop.cloudxr.oob_teleop_env import wss_proxy_port
                _port = wss_proxy_port()
                _all_ips = _sp.check_output(["hostname", "-I"], text=True).split()
                _ips = [ip for ip in _all_ips if not ip.startswith("127.")]
                print("[IsaacTeleopClient] Web client now served locally.")
                print("[IsaacTeleopClient] Navigate your Quest browser to ONE of these URLs:")
                for _ip in _ips:
                    print(f"[IsaacTeleopClient]   https://{_ip}:{_port}/client/")
                print("[IsaacTeleopClient] (use whichever IP is on the same subnet as your Quest)")
            except Exception:
                print("[IsaacTeleopClient] Web client served at https://<server-ip>:48322/client/")

            self._head_tracker = deviceio.HeadTracker()
            self._hand_tracker = deviceio.HandTracker()
            self._controller_tracker = deviceio.ControllerTracker()
            # Both full-body vendors are requested up front when isaacteleop has
            # them: which headset connects (Pico vs Quest) is not known until the
            # browser session negotiates, so both trackers are created and polled
            # every frame. The server keeps BD
            # and Meta full-body data mutually exclusive (only the vendor a client is
            # actually sending reports active), so exactly one of these is ever active
            # at a time; _get_tracker_data() below prefers Meta when both are present.
            # The Meta tracker impl reduces its native 84-joint skeleton down to the
            # same 24-joint BD-shaped FullBodyPoseT as the Pico tracker (the joint
            # selection CloudXR.js used to do in the browser), so both come back in
            # an identical layout and no downstream code needs to know which fired.
            self._body_tracker_bd = deviceio.FullBodyTracker()
            vendor_pairs = [(self._body_tracker_bd, deviceio.TrackerVendor(_BD_BODY_VENDOR))]
            trackers = [
                self._head_tracker,
                self._hand_tracker,
                self._controller_tracker,
                self._body_tracker_bd,
            ]
            # Only request the WebXR vendor from an isaacteleop that has it;
            # see _webxr_body_vendor_available(). Without it the Quest path is
            # dormant and BD/Pico is untouched, rather than the whole session
            # failing to start.
            if _webxr_body_vendor_available():
                self._body_tracker_meta = deviceio.FullBodyTracker()
                vendor_pairs.append(
                    (self._body_tracker_meta, deviceio.TrackerVendor(_WEBXR_BODY_VENDOR))
                )
                trackers.append(self._body_tracker_meta)
            else:
                self._body_tracker_meta = None
                print(
                    f"[IsaacTeleopClient] this isaacteleop does not provide the "
                    f"{_WEBXR_BODY_VENDOR!r} full-body vendor, so Quest/WebXR body "
                    f"tracking is unavailable; Pico/BD body tracking is unaffected. "
                    f"Upgrade isaacteleop to enable it."
                )
            vendor_config = deviceio.VendorConfig(vendor_pairs)
            required_extensions = deviceio.DeviceIOSession.get_required_extensions(
                trackers, vendor_config
            )

            oxr_session = stack.enter_context(
                oxr.OpenXRSession(self._app_name, required_extensions)
            )
            handles = oxr_session.get_handles()
            self._deviceio_session = stack.enter_context(
                deviceio.DeviceIOSession.run(trackers, handles, vendor_config=vendor_config)
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

    def _get_tracker_data(self) -> "TrackerSnapshot | None":
        """Poll current tracking data and return it as a dictionary.

        Returns:
            Dict with keys ``left_controller``, ``right_controller``, ``head``,
            ``left_hand``, ``right_hand``, ``full_body``. Each value is the
            corresponding tracker's payload (raw DeviceIO type), or None.
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
        # This is the "OpenXR Extension Method" of headset identification: which
        # full-body vendor extension actually delivered active data this frame is
        # a fact stated by the runtime, not an inference. Meta reports active only
        # when the connected client is actually sending the Meta full-body
        # skeleton (server-side vendor arbitration keeps BD and Meta mutually
        # exclusive), so exactly one of these is ever active; prefer it, and fall
        # back to BD for a Pico client. Re-derived every frame (not cached) so a
        # headset swap mid-session is picked up on the next poll.
        # Each accessor returns the payload itself, or None when that tracker has
        # produced no data -- isaacteleop deliberately spells absence as None
        # rather than an empty handle, so that an inactive device cannot answer
        # field reads with defaults indistinguishable from real zeroes. Every
        # consumer below already treats None as "not present".
        # _body_tracker_meta is None when this isaacteleop has no WebXR vendor,
        # in which case every frame resolves to BD/Pico.
        full_body = (
            self._body_tracker_meta.get_body_pose(session)
            if self._body_tracker_meta is not None
            else None
        )
        if full_body is not None:
            full_body_source = "quest"
        else:
            full_body = self._body_tracker_bd.get_body_pose(session)
            full_body_source = "pico" if full_body is not None else None
        return TrackerSnapshot(
            left_controller=self._controller_tracker.get_left_controller(session),
            right_controller=self._controller_tracker.get_right_controller(session),
            head=self._head_tracker.get_head(session),
            left_hand=self._hand_tracker.get_left_hand(session),
            right_hand=self._hand_tracker.get_right_hand(session),
            full_body=full_body,
            full_body_source=full_body_source,
        )

    def get_pose_by_name(self, name: str) -> np.ndarray:
        """Return ``[x, y, z, qx, qy, qz, qw]`` for ``name`` ∈ {left_controller, right_controller, headset}."""
        raw = self._get_tracker_data()
        if raw is None:
            return _default_pose_vec()

        if name == "left_controller":
            return _controller_pose_vec(raw.left_controller)
        if name == "right_controller":
            return _controller_pose_vec(raw.right_controller)
        if name == "headset":
            return _pose_vec_from_head_data(raw.head)
        raise ValueError(
            f"Invalid name: {name}. Valid names: 'left_controller', 'right_controller', 'headset'."
        )

    def _snapshot_side(self, raw: "TrackerSnapshot | None", side: str) -> Any:
        if raw is None:
            return None
        return raw.left_controller if side == "left" else raw.right_controller

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
        return raw.full_body

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
