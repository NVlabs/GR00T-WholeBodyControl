"""Native MuJoCo motor checks; scalar fixtures do not qualify G1 locomotion."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from gear_sonic.utils.mujoco_sim.actuation import MotorPD, configure_pd_model


def motor_scene(tmp_path, *, reverse=False, integrator="Euler", actuator="pid", gear=1,
                joint_type="slide", free_body=False, limited=False):
    free = ('<body><freejoint name="root"/><geom size=".1" mass="1"/></body>'
            if free_body else "")
    joint_cap = 'actuatorfrcrange="-5 4"' if limited else ""
    motors = [f'<{actuator} name="m_{name}" joint="{name}" gear="{gear}"/>'
              for name in ("a", "b")]
    if reverse:
        motors.reverse()
    defaults = ('pid kp="0" kv="0" ki="0" input="pos"'
                if actuator == "pid" else actuator)
    xml = f'''<mujoco>
      <option gravity="0 0 0" timestep=".001" integrator="{integrator}"/>
      <default><{defaults} forcerange="-7 6"/></default>
      <worldbody>{free}
        <body pos="0 0 1"><joint name="a" type="{joint_type}" {joint_cap}/>
          <geom size=".1" mass="1" contype="0" conaffinity="0"/></body>
        <body pos="1 0 1"><joint name="b" type="{joint_type}" {joint_cap}/>
          <geom size=".1" mass="1" contype="0" conaffinity="0"/></body>
      </worldbody><actuator>{''.join(motors)}</actuator>
    </mujoco>'''
    path = tmp_path / "motors.xml"
    path.write_text(xml)
    return path


def load_pd_model(path, names, caps):
    model = mujoco.MjModel.from_xml_path(str(path))
    configure_pd_model(model, names, caps)
    return model


def make_controller(tmp_path, **kwargs):
    path = motor_scene(tmp_path, **kwargs)
    model = load_pd_model(path, ["a", "b"], [10, 10])
    data = mujoco.MjData(model)
    return model, data, MotorPD(model, ["a", "b"])


def test_native_pid_has_one_position_input_and_no_integral_state(tmp_path):
    model, _, controller = make_controller(tmp_path)
    assert controller.topology == "pid"
    assert controller.control_indices.shape == (2,)
    assert len(np.unique(controller.control_indices)) == model.nu
    assert model.nactuator == 2 and model.na == 0
    np.testing.assert_array_equal(model.actuator_ctrlspec, 1)
    np.testing.assert_array_equal(model.actuator_ctrlnum, 1)
    np.testing.assert_array_equal(model.actuator_gaintype, mujoco.mjtGain.mjGAIN_PID)
    np.testing.assert_array_equal(model.actuator_gainprm[:, 0], 0)


def test_native_loading_does_not_require_model_editing_api(tmp_path, monkeypatch):
    path = motor_scene(tmp_path)
    monkeypatch.delattr(mujoco, "MjSpec")
    monkeypatch.delattr(mujoco, "MjsActuator")
    model = load_pd_model(path, ["a", "b"], [10, 10])
    MotorPD(model, ["a", "b"])
    assert model.nactuator == model.nu == 2


def test_equivalent_general_pid_declarations_are_admitted(tmp_path):
    path = motor_scene(tmp_path)
    path.write_text(path.read_text().replace("<pid ", "<general ").replace(
        'kp="0" kv="0" ki="0"',
        'gaintype="pid" biastype="affine" gainprm="0" biasprm="0"',
    ))
    model = load_pd_model(path, ["a", "b"], [10, 10])
    data = mujoco.MjData(model)
    controller = MotorPD(model, ["a", "b"])
    data.qvel[:] = [2, -1]
    controller.latch(data, [[1, 0, 3, 1, 0], [2, 0, 1, 2, 0]])
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator, [1, 4])


@pytest.mark.parametrize("change", ['ki="1"', 'slewmax="1"', 'input="pos vel"',
                                   'input="pos vel ff"', 'posrange="-1 1"',
                                   'nsample="10"'])
def test_rejects_source_features_outside_command_contract(tmp_path, change):
    path = motor_scene(tmp_path)
    path.write_text(path.read_text().replace("<pid name=", f"<pid {change} name="))
    with pytest.raises(ValueError, match="native PID|input-clamped|sample history"):
        load_pd_model(path, ["a", "b"], [10, 10])


@pytest.mark.parametrize("interval", [[1, 2], [np.nan, 2]])
def test_invalid_later_cap_does_not_partially_configure_model(tmp_path, interval):
    model = mujoco.MjModel.from_xml_path(str(motor_scene(tmp_path)))
    model.jnt_actfrclimited[1] = True
    model.jnt_actfrcrange[1] = interval
    before = model.actuator_forcerange.copy(), model.actuator_forcelimited.copy()
    with pytest.raises(ValueError, match="Effort-cap intersection"):
        configure_pd_model(model, ["a", "b"], [3, 10])
    np.testing.assert_array_equal(model.actuator_forcerange, before[0])
    np.testing.assert_array_equal(model.actuator_forcelimited, before[1])


@pytest.mark.parametrize("entry", ["load", "controller"])
@pytest.mark.parametrize("unsupported", ["version", "pid_api"])
def test_unsupported_native_pid_runtime_fails_clearly(tmp_path, monkeypatch, entry, unsupported):
    path = motor_scene(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    if unsupported == "version":
        monkeypatch.setattr(mujoco, "mj_version", lambda: 3011000)
    else:
        monkeypatch.setattr(mujoco, "mjtGain", object())
    with pytest.raises(RuntimeError, match=r"MuJoCo >=3\.12\.0.*native PID API"):
        if entry == "load":
            load_pd_model(path, ["a", "b"], [10, 10])
        else:
            MotorPD(model, ["a", "b"])


def test_native_xml_preserves_explicit_unit_gear_over_defaults(tmp_path):
    path = motor_scene(tmp_path)
    path.write_text(path.read_text().replace("<default>", '<default><general gear="2"/>'))
    model = load_pd_model(path, ["a", "b"], [10, 10])
    controller = MotorPD(model, ["a", "b"])
    data = mujoco.MjData(model)
    data.qvel[controller.dof_indices] = [1, -2]
    controller.latch(data, [[1, 0, 3, 4, 0], [-1, 0, 2, 1, 0]])
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[controller.dof_indices], [-1, 0])
    np.testing.assert_array_equal(model.actuator_gear[:, 0], 1)


def test_configuration_preserves_native_physical_parameters(tmp_path):
    path = tmp_path / "motor_defaults.xml"
    path.write_text("""<mujoco>
      <option gravity="0 0 0"/>
      <default><general damping=".3" armature=".2" delay=".01" nsample="100"/></default>
      <worldbody><body><joint name="j" type="slide"/><geom size=".1" mass="1"/></body></worldbody>
      <actuator><pid name="m" joint="j" input="pos" kp="0" kv="0" ki="0"
        delay="0" nsample="0"/></actuator>
    </mujoco>""")
    original = mujoco.MjModel.from_xml_path(str(path))
    model = load_pd_model(path, ["j"], [10])
    controller = MotorPD(model, ["j"])
    original_data, data = mujoco.MjData(original), mujoco.MjData(model)
    original_data.qvel[0] = data.qvel[0] = 1
    original.actuator_biasprm[0, 1] = -1
    original_data.ctrl[0] = 1
    controller.latch(data, [[1, 0, 1, 0, 0]])
    for m, d in ((original, original_data), (model, data)):
        mujoco.mj_forward(m, d)
    np.testing.assert_allclose(data.qacc, original_data.qacc)
    np.testing.assert_allclose(data.qfrc_passive, original_data.qfrc_passive)
    assert model.actuator_armature.sum() == pytest.approx(0.2)
    assert model.actuator_damping.sum() == pytest.approx(0.3)
    assert not model.actuator_delay.any()


def test_delayed_source_pid_is_rejected(tmp_path):
    path = motor_scene(tmp_path)
    path.write_text(
        path.read_text().replace("<pid name=", '<pid delay=".01" nsample="100" name=')
    )
    with pytest.raises(ValueError, match="without delay or sample history"):
        load_pd_model(path, ["a", "b"], [10, 10])


def test_configuration_intersects_limits_and_preserves_asset_bytes(tmp_path):
    path = motor_scene(tmp_path, reverse=True, limited=True, free_body=True)
    original_bytes = path.read_bytes()
    original = mujoco.MjModel.from_xml_path(str(path))
    model = load_pd_model(path, ["b", "a"], [3, 10])
    controller = MotorPD(model, ["b", "a"])
    np.testing.assert_array_equal(controller.joint_ids, [2, 1])
    np.testing.assert_array_equal(controller.qpos_indices, [8, 7])
    np.testing.assert_array_equal(controller.dof_indices, [7, 6])
    np.testing.assert_array_equal(controller.effort_limits, [[-3, 3], [-5, 4]])
    assert model.nu == 2 and model.na == 0
    assert model.nactuator == 2
    assert model.actuator_forcelimited.all()
    assert not model.actuator_ctrllimited.any()
    np.testing.assert_array_equal(model.body_mass, original.body_mass)
    np.testing.assert_array_equal(model.dof_damping, original.dof_damping)
    assert model.opt.integrator == original.opt.integrator
    assert model.opt.timestep == original.opt.timestep
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("changes", [{"gear": 2}, {"actuator": "position"},
                                    {"actuator": "motor"}, {"joint_type": "ball"}])
def test_rejects_unsupported_transmissions_and_feedback(tmp_path, changes):
    with pytest.raises(ValueError, match="scalar|native PID"):
        load_pd_model(motor_scene(tmp_path, **changes), ["a", "b"], [10, 10])


@pytest.mark.parametrize("names,limits", [(["a", "a"], [1, 1]), (["missing", "b"], [1, 1]),
                                          (["a"], [1]), (["a", "b"], [1]),
                                          (["a", "b"], [1, np.nan]), (["a", "b"], [1, 0])])
def test_rejects_bad_layout_or_config_caps(tmp_path, names, limits):
    with pytest.raises(ValueError):
        load_pd_model(motor_scene(tmp_path), names, limits)


def test_total_pd_force_nonzero_velocity_and_cancellation(tmp_path):
    model, data, controller = make_controller(tmp_path, reverse=True)
    # Individual P/D terms exceed the motor cap but cancel to a valid total.
    data.qvel[controller.dof_indices] = [10, -2.5]
    commands = np.array([[10.1, 0, 10, 10, 0], [0, 0, 0, 2, 0]], dtype=float)
    controller.latch(data, commands)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[controller.dof_indices], [1, 5])
    np.testing.assert_allclose(data.actuator_force[model.actuator_outadr[controller.actuator_ids]], [1, 5])
    # Changing the caller's array must not change a latched command.
    commands[:] = np.nan
    controller.step(data)
    assert np.isfinite(data.qpos).all()


def test_saturation_and_mutable_gains(tmp_path):
    model, data, controller = make_controller(tmp_path)
    controller.latch(data, [[100, 0, 20, 0, 0], [-100, 0, 20, 0, 0]])
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.qfrc_actuator, [6, -7])
    data.qvel[:] = [1, -2]
    controller.latch(data, [[100, 0, 0, 0, 0], [-100, 0, 0, 0, 0]])
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(data.qfrc_actuator, 0)
    np.testing.assert_array_equal(model.actuator_biasprm[:, 1:3], 0)


@pytest.mark.parametrize("bad", [np.full((2, 5), np.nan), np.zeros((1, 5)),
                                [[0, 0, -1, 0, 0], [0, 0, 0, 0, 0]],
                                [[1e308, 0, 1e308, 0, 0], [0, 0, 0, 0, 0]]])
def test_command_rejection_does_not_partially_mutate(tmp_path, bad):
    model, data, controller = make_controller(tmp_path)
    controller.latch(data, [[1, 0, 3, 4, 0], [5, 0, 3, 2, 0]])
    before = [data.ctrl.copy(), model.actuator_biasprm.copy(), controller._commands.copy(),
              data.qpos.copy(), data.qvel.copy()]
    with pytest.raises(ValueError):
        controller.latch(data, bad)
    for expected, actual in zip(before, [data.ctrl, model.actuator_biasprm, controller._commands,
                                        data.qpos, data.qvel]):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("channel", [1, 4], ids=["desired_velocity", "feedforward_torque"])
@pytest.mark.parametrize("value", [-1e-12, 1e-12])
def test_nonzero_unsupported_command_fields_are_rejected_atomically(tmp_path, channel, value):
    model, data, controller = make_controller(tmp_path)
    controller.latch(data, [[.1, 0, 3, 4, 0], [.2, 0, 4, 5, 0]])
    command = np.array([[1, 0, 10, 20, 0], [2, 0, 30, 40, 0]], dtype=float)
    command[1, channel] = value
    before = [data.ctrl.copy(), model.actuator_biasprm.copy(), controller._commands.copy(),
              data.qpos.copy(), data.qvel.copy()]
    with pytest.raises(ValueError, match="Position-only native PID requires zero"):
        controller.latch(data, command)
    for expected, actual in zip(before, [data.ctrl, model.actuator_biasprm, controller._commands,
                                        data.qpos, data.qvel]):
        np.testing.assert_array_equal(actual, expected)
    assert data.time == 0


@pytest.mark.parametrize("target,position,velocity", [(2, 2, 0), (1, 2, -1)])
def test_native_intermediate_overflow_rejected_before_mutation(tmp_path, target, position, velocity):
    model, data, controller = make_controller(tmp_path)
    controller.latch(data, [[.1, 0, 3, 4, 0], [.2, 0, 4, 5, 0]])
    data.qpos[0], data.qvel[0] = position, velocity
    command = np.array([[target, 0, 1e308, 1e308, 0], [0, 0, 0, 0, 0]])
    # The difference-form total is zero, but native setpoint and bias terms
    # overflow independently before MuJoCo adds them and applies output limits.
    before = [data.ctrl.copy(), model.actuator_biasprm.copy(), controller._commands.copy(),
              data.qpos.copy(), data.qvel.copy()]
    with pytest.raises(ValueError, match="Native PD intermediate"):
        controller.latch(data, command)
    for expected, actual in zip(before, [data.ctrl, model.actuator_biasprm, controller._commands,
                                        data.qpos, data.qvel]):
        np.testing.assert_array_equal(actual, expected)
    assert data.time == 0


def test_unlimited_scalar_hinge_uses_unwrapped_error_beyond_two_pi(tmp_path):
    model, data, controller = make_controller(tmp_path, joint_type="hinge")
    assert not model.jnt_limited.any()
    target = 2 * np.pi + .25
    command = np.array([[target, 0, .1, 0, 0], [-target, 0, .1, 0, 0]])
    controller.latch(data, command)
    mujoco.mj_forward(model, data)
    # Native PID uses the full scalar joint angle;
    # MuJoCo's rotational wrapping applies to ball/refsite transmissions only.
    np.testing.assert_allclose(data.qfrc_actuator, [.1 * target, -.1 * target])
    assert data.qfrc_actuator[0] > .6


@pytest.mark.parametrize("integrator", ["Euler", "implicit", "implicitfast"])
def test_native_batch_equals_five_single_steps_and_reports_force_time(tmp_path, integrator):
    model, batch, controller = make_controller(tmp_path, integrator=integrator)
    single = mujoco.MjData(model)
    command = np.array([[.2, 0, 12, 2, 0], [-.1, 0, 8, 1, 0]])
    controller.latch(batch, command)
    controller.latch(single, command)
    force_time = controller.step(batch, 5)
    for _ in range(5):
        pre_step_force = command[:, 2] * (command[:, 0] - single.qpos)
        pre_step_force -= command[:, 3] * single.qvel
        mujoco.mj_step(model, single)
    np.testing.assert_array_equal(batch.qpos, single.qpos)
    np.testing.assert_array_equal(batch.qvel, single.qvel)
    np.testing.assert_array_equal(batch.qfrc_actuator, single.qfrc_actuator)
    np.testing.assert_allclose(batch.qfrc_actuator, pre_step_force)
    final_state_force = command[:, 2] * (command[:, 0] - batch.qpos)
    final_state_force -= command[:, 3] * batch.qvel
    assert not np.allclose(batch.qfrc_actuator, final_state_force)
    assert batch.time == pytest.approx(.005)
    assert force_time == pytest.approx(.004)


def test_saturated_native_pid_suppresses_implicit_velocity_derivative(tmp_path):
    path = tmp_path / "saturated.xml"
    path.write_text("""<mujoco>
      <option gravity="0 0 0" timestep=".005" integrator="implicitfast"/>
      <worldbody><body><joint name="j" type="slide"/>
        <geom size=".1" mass=".01"/></body></worldbody>
      <actuator><pid name="m" joint="j" input="pos" kp="0" kv="0" ki="0"/></actuator>
    </mujoco>""")
    model = load_pd_model(path, ["j"], [2])
    data = mujoco.MjData(model)
    controller = MotorPD(model, ["j"])
    controller.latch(data, [[0.1, 0, 160, 8, 0]])
    controller.step(data)
    assert data.qfrc_actuator[0] == pytest.approx(2)
    # At the PID output cap, the implicit solve suppresses the damping
    # derivative, so the velocity increment is dt * capped_force / mass.
    assert data.qvel[0] == pytest.approx(1)


def test_reset_clears_model_gains_and_held_commands(tmp_path):
    model, data, controller = make_controller(tmp_path)
    controller.latch(data, [[1, 0, 3, 4, 0], [5, 0, 3, 2, 0]])
    controller.step(data, 5)
    physical = (data.qpos.copy(), data.qvel.copy(), data.time)
    controller.reset(data)
    np.testing.assert_array_equal(data.ctrl, 0)
    np.testing.assert_array_equal(model.actuator_biasprm[:, 1:3], 0)
    np.testing.assert_array_equal(model.actuator_gainprm[controller.actuator_ids, 0], 0)
    np.testing.assert_array_equal(data.qpos, physical[0])
    np.testing.assert_array_equal(data.qvel, physical[1])
    assert data.time == physical[2]
    controller.step(data)
    np.testing.assert_array_equal(data.qfrc_actuator, 0)


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_bad_step_count_leaves_time_unchanged(tmp_path, count):
    _, data, controller = make_controller(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        controller.step(data, count)
    assert data.time == 0


def test_rejects_rk4_force_timestamp_claim(tmp_path):
    path = motor_scene(tmp_path, integrator="RK4")
    model = load_pd_model(path, ["a", "b"], [10, 10])
    with pytest.raises(ValueError, match="not RK4"):
        MotorPD(model, ["a", "b"])


def test_nonfinite_state_rejected_before_control_mutation(tmp_path):
    model, data, controller = make_controller(tmp_path)
    data.qvel[0] = np.nan
    with pytest.raises(ValueError, match="state must be finite"):
        controller.latch(data, np.zeros((2, 5)))
    np.testing.assert_array_equal(data.ctrl, 0)
    np.testing.assert_array_equal(model.actuator_biasprm, 0)


@pytest.mark.parametrize("field", ["qpos", "qvel"])
def test_nonfinite_uncontrolled_free_base_rejected_before_step(tmp_path, field):
    _, data, controller = make_controller(tmp_path, free_body=True)
    controller.latch(data, np.zeros((2, 5)))
    getattr(data, field)[0] = np.nan
    before_ctrl = data.ctrl.copy()
    with pytest.raises(ValueError, match="free base"):
        controller.step(data, 5)
    assert data.time == 0
    assert np.isnan(getattr(data, field)[0])  # No silent reset to the initial pose.
    np.testing.assert_array_equal(data.ctrl, before_ctrl)
    np.testing.assert_array_equal(data.warning.number, 0)


def test_engine_reset_at_time_zero_is_reported_even_if_elapsed_time_matches(tmp_path):
    _, data, controller = make_controller(tmp_path, free_body=True)
    # Finite values still exceeding MuJoCo's validity threshold trigger its
    # automatic reset. The remaining batch can finish at the expected time.
    data.qvel[0] = 1e20
    with pytest.raises(RuntimeError, match="finite trajectory"):
        controller.step(data, 5)
    assert data.warning.number[int(mujoco.mjtWarning.mjWARN_BADQVEL)] > 0


@pytest.mark.parametrize("field", ["ctrl", "qfrc_applied", "xfrc_applied"])
def test_nonfinite_inputs_rejected_before_engine_step(tmp_path, field):
    _, data, controller = make_controller(tmp_path)
    getattr(data, field).flat[0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        controller.step(data, 5)
    assert data.time == 0


def test_stock_g1_names_caps_and_reordered_actuators(tmp_path):
    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "gear_sonic/utils/mujoco_sim/wbc_configs/"
                            "g1_29dof_sonic_model12.yaml").read_text())
    path = root / config["ROBOT_SCENE"]
    original = mujoco.MjModel.from_xml_path(str(path))
    assert original.nactuator == original.nu == 43 and original.na == 0
    np.testing.assert_array_equal(original.actuator_gaintype, mujoco.mjtGain.mjGAIN_PID)
    np.testing.assert_array_equal(original.actuator_ctrlspec, 1)
    np.testing.assert_array_equal(original.actuator_gainprm[:, 0], 0)
    np.testing.assert_array_equal(original.actuator_biasprm[:, :3], 0)
    assert not original.actuator_ctrllimited.any()
    assert original.actuator_forcelimited.all()
    names = [original.joint(i).name for i in range(original.njnt)
             if original.jnt_type[i] in (int(mujoco.mjtJoint.mjJNT_HINGE),
                                        int(mujoco.mjtJoint.mjJNT_SLIDE))]
    caps = np.asarray(config["motor_effort_limit_list"])
    assert len(names) == 43
    model = load_pd_model(path, names, caps)
    controller = MotorPD(model, names)
    assert model.nu == 43
    assert model.nactuator == 43
    np.testing.assert_array_equal(controller.effort_limits, np.column_stack([-caps, caps]))
    assert names[22].startswith("left_hand") and names[29].startswith("right_shoulder")
    # Source motor capability is retained; the application config is tighter.
    np.testing.assert_array_equal(original.actuator_forcerange[23], [-1.4, 1.4])
    np.testing.assert_array_equal(controller.effort_limits[23], [-0.7, 0.7])
    data = mujoco.MjData(model)
    command = np.zeros((43, 5))
    command[:, 0] = np.arange(1, 44) / 100
    command[:, 2] = 1
    data.qpos[controller.qpos_indices] = 0
    controller.latch(data, command)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[controller.dof_indices], command[:, 0])

    # Reorder actuator XML without changing the 43-joint command or cap order.
    # Temporary XML references the original meshes; no assets are duplicated.
    import xml.etree.ElementTree as ET

    source = path.parent / "g1_29dof_with_hand.xml"
    xml = ET.parse(source)
    elements = xml.getroot().find("actuator")
    elements[:] = list(reversed(elements[:]))
    xml.getroot().find("compiler").set("meshdir", str(source.parent / "meshes"))
    reordered_path = tmp_path / "reordered.xml"
    xml.write(reordered_path)
    reordered = load_pd_model(reordered_path, names, caps)
    reordered_pd = MotorPD(reordered, names)
    np.testing.assert_array_equal(reordered_pd.actuator_ids, np.arange(42, -1, -1))
    np.testing.assert_array_equal(reordered_pd.effort_limits, controller.effort_limits)
    reordered_data = mujoco.MjData(reordered)
    reordered_data.qpos[reordered_pd.qpos_indices] = 0
    reordered_pd.latch(reordered_data, command)
    mujoco.mj_forward(reordered, reordered_data)
    np.testing.assert_allclose(reordered_data.qfrc_actuator[reordered_pd.dof_indices],
                               command[:, 0])
