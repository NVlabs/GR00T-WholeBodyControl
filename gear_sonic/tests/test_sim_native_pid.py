"""Native PID integration in the upstream single-step loop; SDK transport is fake."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import yaml


SIM_DIR = Path(__file__).parents[1] / "utils/mujoco_sim"
ROOT = Path(__file__).parents[2]


class FakeBridge:
    def __init__(self, env):
        self.joystick = None
        self.published = []
        self.after_publish = lambda: None
        self.cmd_received = True
        self.set_commands(env, np.zeros((43, 5)))

    def set_commands(self, env, commands):
        for field, indices in (
            ("low_cmd", env.body_motor_index),
            ("left_hand_cmd", env.left_hand_motor_index),
            ("right_hand_cmd", env.right_hand_motor_index),
        ):
            motors = [SimpleNamespace(**dict(zip(("q", "dq", "kp", "kd", "tau"), row)))
                      for row in commands[indices]]
            setattr(self, field, SimpleNamespace(motor_cmd=motors))

    def PublishLowState(self, obs):
        self.published.append({key: value.copy() if isinstance(value, np.ndarray) else value
                               for key, value in obs.items()})
        self.after_publish()


@pytest.fixture
def env(monkeypatch):
    channel = ModuleType("unitree_sdk2py.core.channel")
    channel.ChannelFactoryInitialize = lambda *args: None
    bridge = ModuleType("gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge")
    bridge.UnitreeSdk2Bridge = FakeBridge
    bridge.ElasticBand = None
    monkeypatch.setitem(sys.modules, channel.__name__, channel)
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)
    spec = importlib.util.spec_from_file_location("native_pid_simulator", SIM_DIR / "base_sim.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = yaml.safe_load((SIM_DIR / "wbc_configs/g1_29dof_sonic_model12.yaml").read_text())
    config.update(ENABLE_ELASTIC_BAND=False, enable_waist=True)
    instance = module.DefaultEnv(config)
    # Upstream only creates this attribute when the optional band is enabled.
    instance.elastic_band = None
    instance.set_unitree_bridge(FakeBridge(instance))
    return instance


def torque_reference_model(env, tmp_path):
    """Reconstruct upstream torque inputs only for the explicit-PD reference."""
    scene_path = ROOT / env.config["ROBOT_SCENE"]
    scene = ET.parse(scene_path)
    include = scene.getroot().find("include")
    source = scene_path.parent / include.get("file")
    xml = ET.parse(source)
    defaults = xml.getroot().find("default")
    defaults.remove(defaults.find("pid"))
    for actuator in xml.getroot().find("actuator"):
        assert actuator.tag == "pid"
        actuator.tag = "motor"
        actuator.set("ctrlrange", actuator.attrib.pop("forcerange"))
    xml.getroot().find("compiler").set("meshdir", str(source.parent / "meshes"))
    reference_model = tmp_path / "torque_model.xml"
    xml.write(reference_model)
    include.set("file", str(reference_model))
    reference_scene = tmp_path / "torque_scene.xml"
    scene.write(reference_scene)
    return mujoco.MjModel.from_xml_path(str(reference_scene))


def test_stock_euler_trajectory_matches_original_torque_motors(env, tmp_path):
    original = torque_reference_model(env, tmp_path)
    assert original.nu == original.nactuator == 43
    np.testing.assert_array_equal(original.actuator_gaintype, mujoco.mjtGain.mjGAIN_FIXED)
    original.opt.timestep = env.sim_dt
    reference = mujoco.MjData(original)
    assert env.mj_model.opt.integrator == original.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    assert env.sim_dt == env.mj_model.opt.timestep == 0.005
    assert env.config["FREE_BASE"] is False and env.use_floating_root_link
    assert env.mj_model.nactuator == env.mj_model.nu == 43
    np.testing.assert_array_equal(env.mj_model.dof_damping, original.dof_damping)
    np.testing.assert_array_equal(env.mj_model.dof_armature, original.dof_armature)
    np.testing.assert_array_equal(env.mj_model.dof_frictionloss, original.dof_frictionloss)
    np.testing.assert_array_equal(env.mj_model.body_mass, original.body_mass)
    env.mj_data.qvel[:] = reference.qvel[:] = np.linspace(-0.05, 0.05, original.nv)
    commands = np.zeros((43, 5))
    commands[:, 0] = env.mj_data.qpos[env.motor_pd.qpos_indices] + np.linspace(-0.1, 0.1, 43)
    commands[:, 2] = np.linspace(20, 80, 43)
    commands[:, 3] = np.linspace(0.2, 3, 43)
    commands[::9, 0] += 100  # Exercise the same total-effort saturation as upstream.
    env.unitree_bridge.set_commands(env, commands)
    original_actuators = [np.flatnonzero(original.actuator_trnid[:, 0] == original.joint(name).id)[0]
                          for name in env.joint_names]
    for step in range(5):
        reference.ctrl[original_actuators] = np.clip(
            commands[:, 2] * (commands[:, 0] - reference.qpos[env.motor_pd.qpos_indices])
            - commands[:, 3] * reference.qvel[env.motor_pd.dof_indices],
            -env.torque_limit, env.torque_limit,
        )
        env.sim_step()
        mujoco.mj_step(original, reference)
        assert env.unitree_bridge.published[-1]["time"] == pytest.approx(step * env.sim_dt)
        assert env.mj_data.time == pytest.approx((step + 1) * env.sim_dt)
        np.testing.assert_allclose(env.mj_data.qfrc_actuator, reference.qfrc_actuator,
                                   rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(env.mj_data.qpos, reference.qpos, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(env.mj_data.qvel, reference.qvel, rtol=1e-12, atol=1e-10)


def test_commands_are_read_after_publication_and_all_sdk_groups_reach_native_inputs(env):
    commands = np.zeros((43, 5))
    commands[:, 0] = np.linspace(-0.1, 0.1, 43)
    commands[:, 2:4] = [2, 0.3]
    env.unitree_bridge.after_publish = lambda: env.unitree_bridge.set_commands(env, commands)
    env.sim_step()
    assert len(env.unitree_bridge.published) == 1
    assert env.unitree_bridge.published[0]["time"] == 0
    np.testing.assert_array_equal(env.mj_data.ctrl[env.motor_pd.control_indices],
                                  commands[:, 0])
    np.testing.assert_array_equal(env.mj_model.actuator_biasprm[env.motor_pd.actuator_ids, 1:3],
                                  -commands[:, 2:4])
    assert env.joint_names[22].startswith("left_hand")
    assert env.joint_names[29].startswith("right_shoulder")


def test_reset_clears_native_gains_but_preserves_upstream_command_reuse(env):
    commands = np.ones((43, 5)) * [0.1, 0, 2, 0.2, 0]
    bridge = env.unitree_bridge
    bridge.set_commands(env, commands)
    messages = (bridge.low_cmd, bridge.left_hand_cmd, bridge.right_hand_cmd)
    env.sim_step()
    assert env.mj_model.actuator_biasprm[:, 1:3].any()
    env.reset()
    assert env.mj_data.time == 0
    np.testing.assert_array_equal(env.mj_data.ctrl, 0)
    np.testing.assert_array_equal(env.mj_model.actuator_biasprm[:, 1:3], 0)
    assert env.unitree_bridge is bridge and bridge.cmd_received
    assert all(before is after for before, after in
               zip(messages, (bridge.low_cmd, bridge.left_hand_cmd, bridge.right_hand_cmd)))
    env.sim_step()
    np.testing.assert_array_equal(env.mj_data.ctrl[env.motor_pd.control_indices],
                                  commands[:, 0])
    np.testing.assert_array_equal(env.mj_model.actuator_biasprm[env.motor_pd.actuator_ids, 1:3],
                                  -commands[:, 2:4])


@pytest.mark.parametrize("group,field", [("low_cmd", "dq"), ("left_hand_cmd", "tau"),
                                       ("right_hand_cmd", "dq")])
def test_unsupported_sdk_command_is_rejected_after_publication_before_mutation(env, group, field):
    commands = np.ones((43, 5)) * [0.1, 0, 2, 0.2, 0]
    bridge = env.unitree_bridge
    bridge.set_commands(env, commands)
    env.sim_step()
    before = [env.mj_data.ctrl.copy(), env.mj_model.actuator_biasprm.copy(),
              env.motor_pd._commands.copy(), env.mj_data.qpos.copy(), env.mj_data.qvel.copy()]
    time_before, publications_before = env.mj_data.time, len(bridge.published)
    motor = getattr(bridge, group).motor_cmd[0]
    motor.q, motor.kp, motor.kd = 10, 100, 20
    setattr(motor, field, 1)
    with pytest.raises(ValueError, match="Position-only native PID requires zero"):
        env.sim_step()
    assert len(bridge.published) == publications_before + 1
    assert bridge.published[-1]["time"] == time_before
    assert env.mj_data.time == time_before
    for expected, actual in zip(before, [env.mj_data.ctrl, env.mj_model.actuator_biasprm,
                                        env.motor_pd._commands, env.mj_data.qpos, env.mj_data.qvel]):
        np.testing.assert_array_equal(actual, expected)
