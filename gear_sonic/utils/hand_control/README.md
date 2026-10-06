# Adding a hand

This guide shows where to integrate another hand in the existing project,
using Dex3 and Inspire as examples. Inspire support is the implementation
provided here; there is no automatic plugin registration or required base class
for other hands. The existing Dex3 implementation and default deployment remain
on their original path.

## Integration workflow

Follow these four steps for a new hand. Implement the modes that the hand supports;
the Inspire links below are working examples, not mandatory dependencies.

1. **Add assets and joint definitions.** Put the model, meshes or mesh references,
   configuration and source/license notices under `data/robots/g1/<hand>/`.
   Define the controllable channel order, units, limits and coupled joints. Add
   robot instantiation and supplemental metadata when using the FK model.
2. **Implement commands and measured feedback.** Keep device-specific conversion
   and transport together under `utils/hand_control/<hand>/` when using Python.
   Define connection, target submission, feedback and shutdown behavior; the
   optional `HandBackend` below is one way to expose these calls. Use the hand's
   own SDK or protocol. Add a separate native bridge only when required.
3. **Connect simulation and deployment.** For MuJoCo, select the model and map
   named joints/actuators to the hand controller. The existing
   `hand_controller_factory` hook passes a controller into the shared simulation;
   inspect [base_sim.py](../mujoco_sim/base_sim.py) for its lifecycle calls.
   Extend the explicit `--hand` selections listed below and connect startup and
   cleanup to the existing deployment entry point. Preserve the Dex3 default.
4. **Provide a runnable example and verification.** Exercise targets and measured
   feedback, channel order/units, invalid inputs and cleanup. Test each supported
   mode, including body-and-hand operation when claimed. Document commands,
   dependencies, stopping behavior and the actual validation scope in one guide.

Completing these steps integrates the new hand into the supported entry points.
Copying the Inspire directory or changing its model path alone is insufficient:
its calibration, six-channel protocol and session handling belong to Inspire.
This workflow covers hand integration; it does not provide automatic motion
retargeting, PICO/VLA inputs or policy adaptation for different hand dynamics.

## Existing integration points

Paths below are relative to the repository root. Some responsibilities span
several files; the two implementations do not have a file-for-file correspondence.

| Responsibility | Dex3 | Inspire |
| --- | --- | --- |
| Model assets used by the default entry points | `gear_sonic/data/robot_model/model_data/g1/`: `g1_29dof_with_hand.urdf`, `g1_29dof_with_hand.xml`, `scene_43dof.xml` | [data/robots/g1/inspire](../../data/robots/g1/inspire/ASSETS.md): URDF, XML, scene and `hand_config.yaml` |
| Robot instantiation | [instantiation/g1.py](../../data/robot_model/instantiation/g1.py) | [instantiation/g1_inspire.py](../../data/robot_model/instantiation/g1_inspire.py) |
| Joint groups, limits and frames | [g1_supplemental_info.py](../../data/robot_model/supplemental_info/g1/g1_supplemental_info.py) | [g1_inspire_supplemental_info.py](../../data/robot_model/supplemental_info/g1/g1_inspire_supplemental_info.py) |
| Simulation command/feedback transport | Hand DDS channels in [unitree_sdk2py_bridge.py](../mujoco_sim/unitree_sdk2py_bridge.py) | [inspire/gateway.py](../mujoco_sim/inspire/gateway.py) |
| Simulated hand control | `compute_hand_torques()` in [base_sim.py](../mujoco_sim/base_sim.py) | [inspire/controller.py](../mujoco_sim/inspire/controller.py) |
| Hardware commands and feedback | [dex3_hands.hpp](../../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/dex3_hands.hpp), using Unitree DDS | [inspire/client.py](inspire/client.py) plus the [serial bridge](../../../gear_sonic_deploy/src/inspire_hand_bridge/README.md) |

Dex3's hand driver runs inside the body deployment process. Inspire uses a
client and a separate bridge on the machine connected to its serial ports.
A new hand can use its own SDK or transport without adopting Inspire's ZMQ,
native6 messages, six-channel order or serial conversion.

## Shared calls used by Inspire

[HandBackend](interface.py) lets the Inspire example use the simulation and
hardware adapters through the same calls. Reuse it when useful; other hand
integrations do not have to inherit it, and Dex3 does not use it.

| Member | Responsibility |
| --- | --- |
| `description` | Model, ordered left/right control channels and units. |
| `connect()` | Acquire resources without sending a motion target. |
| `set_target(left, right)` | Validate and submit targets; submission does not imply arrival. |
| `read_state()` | Return measured feedback; use `None` for unavailable measurements. |
| `stop()` | Reject further targets and cancel queued work; document the device's resulting behavior. |
| `close()` | Release owned resources; safe to repeat after errors. |

Joint count, calibration, limits and communication belong to each implementation.
Inspire's six channels, native6 messages and serial conversion are specific to
Inspire, not requirements for other hands. If reusing this API, see its docstrings for
timestamp, validation and lifecycle semantics.

## Where to add files

Use the existing project subsystems and keep each hand's helpers together.
Add only the files needed for the supported modes, with a short directory overview.

| Location | Add when needed | Inspire example |
| --- | --- | --- |
| `gear_sonic/utils/hand_control/<hand>/` | Python communication/configuration helpers, if needed; `HandBackend` reuse is optional. | [inspire/](inspire/README.md) |
| `gear_sonic/data/robots/g1/<hand>/` | Model assets, hand configuration and source/license notices. | [Inspire assets](../../data/robots/g1/inspire/ASSETS.md) |
| `gear_sonic/data/robot_model/` | Instantiation and supplemental joint/frame metadata for FK users. | [g1_inspire.py](../../data/robot_model/instantiation/g1_inspire.py) |
| `gear_sonic/utils/mujoco_sim/<hand>/` | MuJoCo layout, controller and backend for simulation support. | [Inspire simulation](../mujoco_sim/inspire/README.md) |
| `gear_sonic_deploy/src/<hand>_bridge/` | An independently buildable driver, if hardware requires one. | [Inspire bridge](../../../gear_sonic_deploy/src/inspire_hand_bridge/README.md) |

## Connect the entry points

Adding files does not automatically register a new hand. Currently the launchers
accept `dex3` and `inspire`. To expose another hand, extend the explicit selection
in [SimLoopConfig](../mujoco_sim/configs.py),
[run_sim_loop.py](../../scripts/run_sim_loop.py),
[deploy.sh](../../../gear_sonic_deploy/deploy.sh) and the
[body launcher's hand selection](../../../gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp).
Add hand-specific startup/cleanup under `gear_sonic_deploy/scripts/` only if needed;
[inspire_hand.py](../../../gear_sonic_deploy/scripts/inspire_hand.py) shows the existing integration.
Leave the default Dex3 selection and its implementation unchanged.

Verify command ordering/units, measured feedback, failure cleanup, and the
supported deployment modes. The [Inspire tests](../../tests/test_inspire_backends.py)
and [execution guide](../../../docs/source/tutorials/inspire_hands.md) provide an example.
