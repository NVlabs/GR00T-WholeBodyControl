# Inspire communication

All Python communication helpers for Inspire are kept here. Its hardware and
simulation adapters share `HandBackend`; other hands can choose their own
implementation. See [adding a hand](../README.md) for the integration locations.

| File | Purpose |
| --- | --- |
| `client.py` | `InspireGatewayBackend`: targets and measured feedback through the gateway. |
| `config.py` | Load and validate joint/hardware settings; build native driver arguments. |
| `contract.py` | Six-channel order, units and command/feedback schema. |
| `session.py` | Operator session and gateway communication lifecycle. |
| `safety.py` | Existing target limit and rate checks. |

`client.create_backend()` constructs the client without connecting. Its owner
calls `connect()`, submits targets and reads feedback, then calls `stop()` and
`close()` during cleanup. The deployment launcher manages its own session.

Configuration is in [hand_config.yaml](../../../data/robots/g1/inspire/hand_config.yaml).
The [simulation adapter](../../mujoco_sim/inspire/README.md) uses the same channel
contract. Build, launch and logging commands are in the single
[execution guide](../../../../docs/source/tutorials/inspire_hands.md).
