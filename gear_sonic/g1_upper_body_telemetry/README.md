# G1 upper-body telemetry

The robot process reads `rt/lowstate` and `rt/lowcmd`, samples them at a
configurable target rate, derives `ddq_cmd` from consecutive `dq_cmd` samples,
computes PD-aware torque and all residuals as `estimated - commanded`, and
publishes the result over ZMQ.

Robot (the folder can also be copied next to another `gear_sonic` checkout):

```bash
python -m gear_sonic.g1_upper_body_telemetry.robot_publisher \
  --publish-hz 50 --port 5560
```

If only this directory was copied to `/home/unitree/nikita/`:

```bash
cd /home/unitree/nikita
python -m g1_upper_body_telemetry.robot_publisher --publish-hz 50 --port 5560
```

Host connectivity test:

```bash
python -m gear_sonic.g1_upper_body_telemetry.host_receiver \
  --host 192.168.50.132 --port 5560
```

Direct SDK check on the robot (or any computer that receives G1 DDS):

```bash
python -m gear_sonic.g1_upper_body_telemetry.inspect_upper_body_state \
  --network-interface enp3s0 \
  --print-hz 2
```

It subscribes to both `rt/lowstate` and `rt/lowcmd`, printing upper-body
`q`, `dq`, `ddq`, `tau_est`, and command `q`, `dq`, `tau`, `kp`, `kd` for
motor indices 12–28. Use `--once` for one paired sample.

The BrainCo exporter uses the same receiver and saves every packet collected
during an episode to `raw-telemetry/chunk-NNN/episode_NNNNNN.npz`. New datasets
persist only `q_cmd`, `kp`, `kd`, `q_est`, `dq_est`, `tau_est`, and
`q_residual`, plus packet timing/sequence metadata. The richer ZMQ wire
payload is retained for compatibility and diagnostics but its other computed
fields are not written to either Parquet or raw telemetry.
