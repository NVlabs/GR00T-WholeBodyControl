# Inspire simulation

The Inspire adapter for the existing MuJoCo simulation. Assets remain in
[data/robots/g1/inspire](../../../data/robots/g1/inspire/ASSETS.md);
communication definitions remain in [hand_control/inspire](../../hand_control/inspire/README.md).

| File | Purpose |
| --- | --- |
| `backend.py` | `HandBackend` for an isolated hand scene, and a client factory for a running shared simulation. |
| `environment.py` | Configure the G1 + Inspire model in the existing simulation. |
| `layout.py` | Resolve configured joint/actuator names to MuJoCo indices. |
| `controller.py` | Apply hand targets and read simulated joint positions. |
| `gateway.py` | Handle hand commands and feedback inside the body-and-hands simulation. |

[run_sim_loop.py](../../../scripts/run_sim_loop.py) selects this adapter with
`--hand inspire`. The body and hands then run in one world. The isolated backend
is for hand checks; it does not run SONIC body control.

See [adding a hand](../../hand_control/README.md) for the extension layout and the
[execution guide](../../../../docs/source/tutorials/inspire_hands.md) for commands.
