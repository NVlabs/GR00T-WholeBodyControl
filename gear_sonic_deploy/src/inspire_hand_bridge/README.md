# Inspire serial bridge

An independent C++ bridge between hand commands and the Inspire serial devices.
It is needed on the machine connected to the hand serial ports. SONIC body
inference is launched separately by the common deployment entry point.
The default Dex3 build does not require this bridge.

| Location | Purpose |
| --- | --- |
| `include/inspire/` | Protocol, channel conversion, serial I/O and gateway helpers. |
| `src/inspire_direct_serial_hand_gateway.cpp` | Bridge executable. |
| `CMakeLists.txt` | Independent build and offline test targets. |
| `tests/` | Protocol and gateway tests. |
| `bridge/package.py` | Package sources and license notices for a separate machine. |
| `bridge/run.sh`, `bridge/bridge.env.example` | Standalone bridge launcher and example settings. |

Build, packaging and computer/robot launch commands are in the
[execution guide](../../../docs/source/tutorials/inspire_hands.md).
The Python client is in [hand_control/inspire](../../../gear_sonic/utils/hand_control/inspire/README.md).
For another hand, follow the [extension layout](../../../gear_sonic/utils/hand_control/README.md)
and add a separate driver only if that hand needs one.
