# Unitree G1 upper-body torque monitor

Монитор подписывается через `unitree_sdk2py` на:

- `rt/lowstate` (`LowState_`) — `motor_state[i].tau_est`;
- `rt/lowcmd` (`LowCmd_`) — `motor_cmd[i].tau`.

Для 17 upper-body приводов строится разница
`delta_tau = tau_est - tau_cmd` в Н·м. В одном OpenCV-окне одновременно
показываются torso, left arm и right arm.

Запуск из корня репозитория:

```bash
source /home/unitree/GR00T-WholeBodyControl/.venv_teleop/bin/activate
cd /home/unitree/teleop-ws/Teleop-Data-Collection

python -m gear_sonic.g1_tau_monitor \
  --network-interface wlp128s20f3 \
  --window-seconds 15
```

Имя интерфейса можно проверить командой `ip link`. Значения `auto` и `none`
включают автоматический выбор интерфейса Unitree SDK.

По умолчанию `tau_cmd` — непосредственно поле `MotorCmd.tau`, без PD-составляющей.
Чтобы учитывать позиционный и скоростной регуляторы, добавьте флаг:

```bash
python -m g1_tau_monitor --window-seconds 15 --include-pd
```

В этом режиме используется:

```text
tau_cmd = tau + kp * (q_cmd - q) + kd * (dq_cmd - dq)
```

Чтобы дополнительно вывести под графиками момента `q` и `q_cmd`:

```bash
python -m g1_tau_monitor --window-seconds 15 --show-q
```

На позиционных графиках `q` показан сплошной линией, `q_cmd` — пунктиром.
Флаги можно использовать одновременно: `--show-q --include-pd`.

Клавиши: `Space` — пауза, `C` — очистить историю, `Q`/`Esc` — выход.
