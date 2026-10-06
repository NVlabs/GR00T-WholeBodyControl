"""MuJoCo backend for Inspire hand control.

The body/base is held kinematically at its initial pose. Finger dynamics and
couplings remain active; this is a hand bench, not whole-body SONIC simulation.
"""

import threading
import time

import numpy as np

from gear_sonic.data.robot_model.instantiation.g1_inspire_assets import load_asset_paths
from gear_sonic.utils.hand_control.inspire.config import load_mapping
from gear_sonic.utils.hand_control.inspire.contract import (
    ContractError,
    HandStatus,
    InspireHandCommand,
)
from gear_sonic.utils.hand_control.interface import HandBackend, HandDescription, HandState
from gear_sonic.utils.mujoco_sim.inspire.controller import InspireSimController


class InspireSimBackend(HandBackend):
    def __init__(self):
        self.mapping = load_mapping()
        self._lock = threading.Lock()
        self._exit = threading.Event()
        self._thread = None
        self._connected = False
        self._stopped = False
        self._error = None
        self._sequence = 0
        self.model = self.data = self.controller = None

    @property
    def description(self):
        return HandDescription(
            "inspire_rh56dfx",
            tuple(j.urdf_joint for j in self.mapping.left),
            tuple(j.urdf_joint for j in self.mapping.right),
        )

    def connect(self):
        if self._connected:
            if self._stopped:
                raise RuntimeError("close before reconnecting a stopped backend")
            return
        import mujoco

        try:
            self.model = mujoco.MjModel.from_xml_path(str(load_asset_paths().scene))
            self.data = mujoco.MjData(self.model)
            mujoco.mj_forward(self.model, self.data)
            self.controller = InspireSimController(self.model, self.data, mapping=self.mapping)
            # Fix only body joints and floating base; keep all passive finger joints free.
            body_q, body_v = [], []
            for jid in range(self.model.njnt):
                if "_inspire_" in (self.model.joint(jid).name or ""):
                    continue
                free = self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE
                qadr, vadr = self.model.jnt_qposadr[jid], self.model.jnt_dofadr[jid]
                body_q.extend(range(qadr, qadr + (7 if free else 1)))
                body_v.extend(range(vadr, vadr + (6 if free else 1)))
            self._body_q = np.asarray(body_q)
            self._body_v = np.asarray(body_v)
            self._body_pose = self.data.qpos[self._body_q].copy()
            self.controller.arm()
            self._sequence = 0
            self._connected, self._stopped, self._error = True, False, None
            self._exit.clear()
            self._thread = threading.Thread(target=self._run, name="inspire-hand-sim", daemon=True)
            self._thread.start()
        except BaseException:
            self.close()
            raise

    def _run(self):
        import mujoco

        deadline = time.monotonic()
        try:
            while not self._exit.is_set():
                with self._lock:
                    self.controller.step(now_ns=time.monotonic_ns())
                    self.data.qpos[self._body_q] = self._body_pose
                    self.data.qvel[self._body_v] = 0
                    mujoco.mj_step(self.model, self.data)
                    self.data.qpos[self._body_q] = self._body_pose
                    self.data.qvel[self._body_v] = 0
                    mujoco.mj_forward(self.model, self.data)
                    if (
                        not np.isfinite(self.data.qpos).all()
                        or not np.isfinite(self.data.qvel).all()
                    ):
                        raise FloatingPointError("nonfinite MuJoCo state")
                deadline += self.model.opt.timestep
                now = time.monotonic()
                if deadline < now:
                    deadline = now + self.model.opt.timestep
                self._exit.wait(max(0, deadline - now))
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
                self._stopped = True
                self._zero_hand_controls()

    def _require_connected(self):
        if not self._connected:
            raise RuntimeError("hand backend is not connected")

    def set_target(self, left, right):
        with self._lock:
            self._require_connected()
            if self._stopped or self._error:
                raise RuntimeError(self._error or "hand backend is stopped")
            try:
                left = self.mapping.validate_limits(left, field_name="left")
                right = self.mapping.validate_limits(right, field_name="right", side="right")
                if self._sequence == 0:
                    measured_left, measured_right, _, _ = self.controller.layout.read(self.data)
                    tolerance = np.array([0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
                    if np.any(abs(left - measured_left) > tolerance) or np.any(
                        abs(right - measured_right) > tolerance
                    ):
                        raise ContractError(
                            "first target must be near the measured position; ramp from feedback"
                        )
                now = time.monotonic_ns()
                command = InspireHandCommand(self._sequence + 1, now, left, right)
                self.controller.submit(command, now_ns=now)
                self._sequence += 1
            except (ValueError, RuntimeError) as exc:
                self._stopped, self._error = True, str(exc)
                self.controller.guard.disarm()
                self._zero_hand_controls()
                raise

    def read_state(self):
        with self._lock:
            self._require_connected()
            left, right, _, _ = self.controller.layout.read(self.data)
            status = self.controller.guard.status
            error = self._error
            if status in (HandStatus.STALE, HandStatus.FAULT):
                error = self.controller.guard.last_error or status.value
            elif self._stopped:
                error = error or "STOPPED"
            return HandState(
                tuple(map(float, left)), tuple(map(float, right)), time.monotonic_ns(), error
            )

    def _zero_hand_controls(self):
        if self.controller is not None:
            for side in (self.controller.layout.left, self.controller.layout.right):
                self.data.ctrl[side.actuator_ids] = 0

    def stop(self):
        with self._lock:
            self._stopped = True
            if self.controller is not None:
                self.controller.guard.disarm()
                self.controller.guard.last_command = None
                self._zero_hand_controls()

    def close(self):
        self.stop()
        self._exit.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            if self._thread.is_alive():
                raise RuntimeError("MuJoCo worker did not stop")
        self._thread = None
        self._connected = False
        self.model = self.data = self.controller = None


def create_backend() -> HandBackend:
    return InspireSimBackend()


def create_remote_backend(config_path=None) -> HandBackend:
    """Connect to run_sim_loop --hand inspire, without creating another world."""
    from gear_sonic.utils.hand_control.inspire.client import InspireGatewayBackend
    from gear_sonic.utils.hand_control.inspire.config import DEFAULT_MAPPING_PATH

    return InspireGatewayBackend(config_path or DEFAULT_MAPPING_PATH, simulation=True)
