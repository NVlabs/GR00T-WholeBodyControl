"""Position-only native motor PD with held position targets and mutable gains.

Actuators are declared as native PID in MJCF. Only stateless, unit-gear PD
actuators on scalar joints with one position input are supported. Their force is
``kp * (q_target - q) - kd * dq``; desired velocity and feedforward must be zero.
MuJoCo >=3.12 is required. SDK commands supply per-joint gains; zero MJCF gains
keep startup inactive. Gains are mutable model parameters, so independently
controlled environments must use separate models. Effort limits apply after
summing position feedback and velocity damping.
"""

from numbers import Integral

import mujoco
import numpy as np


def _require_native_pid():
    if (
        mujoco.mj_version() < 3012000
        or not hasattr(mujoco.mjtGain, "mjGAIN_PID")
    ):
        raise RuntimeError("Native motor PD requires MuJoCo >=3.12.0 with the native PID API")


def _joint_actuator_ids(model, joint_names):
    names = tuple(joint_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("joint_names must be nonempty and unique")
    joints, actuators = [], []
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0 or model.jnt_type[joint] not in (
            int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)
        ):
            raise ValueError(f"Expected a named scalar joint: {name}")
        matches = np.flatnonzero(
            (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
            & (model.actuator_trnid[:, 0] == joint)
        )
        if len(matches) != 1:
            raise ValueError(f"Expected one direct joint actuator for {name}")
        joints.append(joint)
        actuators.append(matches[0])
    if len(model.actuator_trntype) != len(names):
        raise ValueError("joint_names must cover every actuator exactly once")
    return np.asarray(joints), np.asarray(actuators)


def _native_pid_ids(model, joint_names):
    joints, ids = _joint_actuator_ids(model, joint_names)
    if (
        np.any(model.actuator_gaintype[ids] != mujoco.mjtGain.mjGAIN_PID)
        or np.any(model.actuator_biastype[ids] != mujoco.mjtBias.mjBIAS_AFFINE)
        or np.any(model.actuator_dyntype[ids] != mujoco.mjtDyn.mjDYN_NONE)
        or np.any(model.actuator_gainprm[ids, 0] != 0)
        or np.any(model.actuator_biasprm[ids, 0] != 0)
        or np.any(model.actuator_ctrlspec[ids] != 1)
        or np.any(model.actuator_ctrlnum[ids] != 1)
        or np.any(model.actuator_outnum[ids] != 1)
        or np.any(model.actuator_gear[ids] != [1, 0, 0, 0, 0, 0])
        or any(not mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(i)) for i in ids)
    ):
        raise ValueError("Expected named, stateless, unit-gear native PID with one position input")
    if np.any(model.actuator_delay[ids] != 0) or np.any(model.actuator_history[ids, 0] != 0):
        raise ValueError("Expected immediate motor controls without delay or sample history")
    controls = model.actuator_ctrladr[ids]
    if np.any(model.actuator_ctrllimited[controls]):
        raise ValueError("Native PID position targets must not be input-clamped")
    return joints, ids


def configure_pd_model(model, joint_names, effort_limits):
    """Validate native MJCF PID and intersect its output caps with configuration.

    ``effort_limits`` are positive symmetric joint-effort caps in named joint
    order, independent of actuator declaration order. MJCF actuator and joint
    force limits remain in force; position targets are not clipped as efforts.
    All caps are validated before changing the model.
    """
    _require_native_pid()
    names = tuple(joint_names)
    limits = np.asarray(effort_limits, dtype=float)
    if limits.shape != (len(names),) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError("effort_limits must contain one finite positive cap per joint")
    joints, actuators = _native_pid_ids(model, names)
    bounds = np.column_stack((-limits, limits))
    for i, (joint, actuator) in enumerate(zip(joints, actuators)):
        for enabled, interval in (
            (model.actuator_forcelimited[actuator], model.actuator_forcerange[actuator]),
            (model.jnt_actfrclimited[joint], model.jnt_actfrcrange[joint]),
        ):
            if enabled:
                bounds[i, 0] = np.maximum(bounds[i, 0], interval[0])
                bounds[i, 1] = np.minimum(bounds[i, 1], interval[1])
    if (not np.isfinite(bounds).all() or np.any(bounds[:, 0] > 0)
            or np.any(bounds[:, 1] < 0) or np.any(bounds[:, 0] >= bounds[:, 1])):
        raise ValueError("Effort-cap intersection must be finite and contain zero")
    model.actuator_forcerange[actuators] = bounds
    model.actuator_forcelimited[actuators] = True


class MotorPD:
    """Latch ``[q, dq, kp, kd, tau]`` commands in named joint order.

    Native PD evaluates position feedback and velocity damping every physics
    step. The SDK command layout is retained, but nonzero desired velocity
    (``dq``) or feedforward torque (``tau``) is rejected before any mutation.
    The caller serializes model/data access and owns physical state reset.
    """

    def __init__(self, model, joint_names):
        _require_native_pid()
        self.model = model
        names = tuple(joint_names)
        self.joint_ids, self.actuator_ids = _native_pid_ids(model, names)
        ids = self.actuator_ids
        self.topology = "pid"
        self.control_indices = model.actuator_ctrladr[ids].copy()
        self.effort_limits = model.actuator_forcerange[ids].copy()
        self._check_timestep()
        self.qpos_indices = model.jnt_qposadr[self.joint_ids].copy()
        self.dof_indices = model.jnt_dofadr[self.joint_ids].copy()
        self._commands = np.zeros((len(ids), 5))

    def _check_timestep(self):
        if self.model.opt.integrator not in (
            mujoco.mjtIntegrator.mjINT_EULER,
            mujoco.mjtIntegrator.mjINT_IMPLICIT,
            mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
        ):
            raise ValueError(
                "MotorPD force timestamps support Euler/implicit/implicitfast, not RK4"
            )
        if not np.isfinite(self.model.opt.timestep) or self.model.opt.timestep <= 0:
            raise ValueError("Physics timestep must be finite and positive")

    def _effort(self, data, commands):
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Entire physical state must be finite, including the free base")
        q, dq = data.qpos[self.qpos_indices], data.qvel[self.dof_indices]
        with np.errstate(over="ignore", invalid="ignore"):
            effort = (
                commands[:, 2] * (commands[:, 0] - q)
                - commands[:, 3] * dq
            )
        if not np.isfinite(effort).all():
            raise ValueError("Computed motor effort must be finite")
        # MuJoCo evaluates setpoint terms and affine state bias separately.
        # Equal targets/states can cancel algebraically while those native
        # intermediates overflow before cancellation (and before clipping).
        with np.errstate(over="ignore", invalid="ignore"):
            inputs = commands[:, 2] * commands[:, 0]
            bias = -commands[:, 2] * q - commands[:, 3] * dq
            finite = np.isfinite([inputs, bias, inputs + bias]).all()
        if not finite:
            raise ValueError("Native PD intermediate effort must be finite")
        return effort

    def latch(self, data, commands):
        """Validate the whole command before changing any gains or controls."""
        values = np.array(commands, dtype=float, copy=True)
        if values.shape != self._commands.shape or not np.isfinite(values).all():
            raise ValueError(f"Motor commands must be finite with shape {self._commands.shape}")
        if np.any(values[:, 2:4] < 0):
            raise ValueError("Motor gains must be nonnegative")
        if np.any(values[:, [1, 4]] != 0):
            raise ValueError(
                "Position-only native PID requires zero desired velocity (dq) "
                "and feedforward torque (tau)"
            )
        self._effort(data, values)
        self._commands = values
        self._set_gains(values[:, 2:4])
        data.ctrl[self.control_indices] = values[:, 0]

    def _set_gains(self, gains):
        self.model.actuator_biasprm[self.actuator_ids, 1:3] = -gains

    def step(self, data, nstep=1):
        """Advance held commands; return the last cached force-evaluation time.

        ``data.time`` after stepping is the integrated state time. Cached
        actuator force is from one timestep earlier for the admitted integrators.
        Unexpected MuJoCo automatic resets raise instead of hiding lost progress.
        """
        if isinstance(nstep, bool) or not isinstance(nstep, Integral) or nstep < 1:
            raise ValueError("nstep must be a positive integer")
        self._check_timestep()
        start, dt = float(data.time), self.model.opt.timestep
        if not np.isfinite(start):
            raise ValueError("Simulation time must be finite")
        self._effort(data, self._commands)
        if not all(
            np.isfinite(values).all()
            for values in (
                data.ctrl,
                data.qfrc_applied,
                data.xfrc_applied,
                self.model.actuator_biasprm[self.actuator_ids],
                self.model.actuator_gainprm[self.actuator_ids],
            )
        ):
            raise ValueError("Controls, applied forces and motor parameters must be finite")
        bad_warnings = [int(mujoco.mjtWarning.mjWARN_BADQPOS),
                        int(mujoco.mjtWarning.mjWARN_BADQVEL),
                        int(mujoco.mjtWarning.mjWARN_BADQACC),
                        int(mujoco.mjtWarning.mjWARN_BADCTRL)]
        warnings_before = data.warning.number[bad_warnings].copy()
        mujoco.mj_step(self.model, data, nstep=int(nstep))
        if (
            not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isfinite(data.qacc).all()
            or not np.array_equal(data.warning.number[bad_warnings], warnings_before)
            or not np.isclose(data.time, start + nstep * dt, rtol=1e-12, atol=1e-12)
        ):
            raise RuntimeError("MuJoCo did not advance the requested finite trajectory")
        return float(data.time - dt)

    def reset(self, data):
        """Clear held motor commands without changing physical state or time."""
        self._commands.fill(0)
        self._set_gains(np.zeros((len(self.joint_ids), 2)))
        data.ctrl[self.control_indices] = 0
