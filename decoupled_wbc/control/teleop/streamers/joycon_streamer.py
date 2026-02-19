import time
from typing import Any, Dict

from evdev import InputDevice, ecodes, list_devices
import numpy as np

from decoupled_wbc.control.teleop.streamers.base_streamer import BaseStreamer, StreamerOutput


class JoyConDevice:
    """
    A class to handle a single Joy-Con controller using evdev.
    Supports both individual Joy-Cons and their IMU sensors.
    """

    def __init__(
        self, device_path: str = None, controller_type: str = "auto", silent: bool = False
    ):
        """
        Initialize the Joy-Con device.

        Args:
            device_path: Path to the Joy-Con event device (e.g., /dev/input/event25)
            controller_type: "left", "right", or "auto"
            silent: Whether to suppress output messages
        """
        self._silent = silent
        self._device_path = device_path
        self._controller_type = controller_type
        self._device = None
        self._prev_button_states = {}

        # Joy-Con button mappings (evdev key codes)
        self._setup_mappings()

        # Current analog stick values
        self._stick_x = 0.0
        self._stick_y = 0.0

        # Axis codes (will be set in start() based on controller type)
        self._x_axis_code = 0  # Default for left Joy-Con
        self._y_axis_code = 1  # Default for left Joy-Con

    def _setup_mappings(self):
        """Set up button and axis mappings for Joy-Con event codes."""
        # Left Joy-Con button mappings
        self._left_button_mappings = {
            # Left Joy-Con specific buttons
            544: "dpad_up",
            545: "dpad_down",
            546: "dpad_left",
            547: "dpad_right",
            309: "capture",
            314: "minus",
            # Left Joy-Con shoulder/trigger buttons
            310: "L",  # L button
            312: "ZL",  # ZL trigger
            311: "SL",  # SL side button
            313: "SR",  # SR side button
        }

        # Right Joy-Con button mappings
        self._right_button_mappings = {
            # Right Joy-Con specific buttons
            304: "a",
            305: "b",
            307: "x",
            308: "y",
            315: "plus",
            316: "home",
            # Right Joy-Con shoulder/trigger buttons
            311: "R",  # SL side button
            313: "ZR",  # SR side button
            310: "SL",  # R button
            312: "SR",  # ZR trigger
        }

        # Will be set in start() based on controller type
        self._button_mappings = {}
        self._prev_button_states = {}

        # Axis codes (set dynamically in start())
        self._x_axis_code = 0
        self._y_axis_code = 1

    def _find_joycon_device(self):
        """Find and return a Joy-Con device, waiting until one is found."""
        while True:
            devices = [InputDevice(path) for path in list_devices()]
            joycon_devices = []
            for device in devices:
                if "Joy-Con" in device.name and "IMU" not in device.name:
                    joycon_devices.append(device)

            if not joycon_devices:
                if not self._silent:
                    print("Warning: No Joy-Con devices found. Waiting for Joy-Con connection...")
                time.sleep(1.0)  # Wait 1 second before checking again
                continue

            # Joy-Con devices found, proceed with selection
            # If specific type requested, find matching device
            if self._controller_type == "left":
                for device in joycon_devices:
                    if "Joy-Con (L)" in device.name:
                        if not self._silent:
                            print(f"Found requested left Joy-Con: {device.name}")
                        return device
                # If left Joy-Con not found but other Joy-Cons exist
                if not self._silent:
                    print("Warning: Left Joy-Con requested but not found. Waiting...")
                time.sleep(1.0)
                continue

            elif self._controller_type == "right":
                for device in joycon_devices:
                    if "Joy-Con (R)" in device.name:
                        if not self._silent:
                            print(f"Found requested right Joy-Con: {device.name}")
                        return device
                # If right Joy-Con not found but other Joy-Cons exist
                if not self._silent:
                    print("Warning: Right Joy-Con requested but not found. Waiting...")
                time.sleep(1.0)
                continue

            # Auto mode or fallback - return first available
            if not self._silent:
                print(f"Found Joy-Con in auto mode: {joycon_devices[0].name}")
            return joycon_devices[0]

    def start(self):
        """Start the Joy-Con device."""
        try:
            if self._device_path:
                # Use specific device path
                self._device = InputDevice(self._device_path)
            else:
                # Auto-find Joy-Con device
                self._device = self._find_joycon_device()

            # Determine actual controller type and axis codes from device name
            if "Joy-Con (L)" in self._device.name:
                self._controller_type = "left"
                self._x_axis_code = 0
                self._y_axis_code = 1
                # Set up left Joy-Con specific button mappings
                self._button_mappings = self._left_button_mappings.copy()
            elif "Joy-Con (R)" in self._device.name:
                self._controller_type = "right"
                self._x_axis_code = 3  # Right Joy-Con uses ABS_RX (3)
                self._y_axis_code = 4  # Right Joy-Con uses ABS_RY (4)
                # Set up right Joy-Con specific button mappings
                self._button_mappings = self._right_button_mappings.copy()
            else:
                # Unknown device, keep defaults
                if not self._silent:
                    print(f"Warning: Unknown Joy-Con type: {self._device.name}")
                self._button_mappings = self._left_button_mappings.copy()

            # Initialize button states
            self._prev_button_states = {name: False for name in self._button_mappings.values()}

            if not self._silent:
                print(f"Joy-Con connected: {self._device.name}")
                print(f"Device path: {self._device.path}")
                print(f"Controller type: {self._controller_type}")
                print(f"Axis codes: X={self._x_axis_code}, Y={self._y_axis_code}")
                print(f"Button mappings: {list(self._button_mappings.values())}")
                print(f"Capabilities: {list(self._device.capabilities(verbose=True).keys())}")

        except Exception as e:
            if not self._silent:
                print(f"Failed to initialize Joy-Con: {e}")
            raise e

    def stop(self):
        """Stop the Joy-Con device."""
        if self._device:
            try:
                self._device.close()
            except (OSError, AttributeError):
                pass  # Ignore errors during cleanup
        if not self._silent:
            print("Joy-Con disconnected.")

    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the Joy-Con by reading all available events."""
        if not self._device:
            return {}

        try:
            # Read all available events (non-blocking)
            import select

            r, w, x = select.select([self._device], [], [], 0)  # Non-blocking

            if r:
                # Read ALL events available right now
                events = self._device.read()
                for event in events:
                    self._process_event(event)

            # Return current state
            return self._build_state()

        except Exception as e:
            if not self._silent:
                print(f"Error reading Joy-Con state: {e}")
            return {}

    def _process_event(self, event):
        """Process a single input event from the Joy-Con."""
        if event.type == ecodes.EV_KEY:
            # Button event
            button_pressed = event.value == 1
            button_name = self._button_mappings.get(event.code)
            if button_name:
                self._prev_button_states[button_name] = button_pressed

        elif event.type == ecodes.EV_ABS:
            # Analog stick event - use dynamic axis codes
            if event.code == self._x_axis_code:
                self._stick_x = event.value / 32767.0
            elif event.code == self._y_axis_code:
                self._stick_y = -event.value / 32767.0

    def _build_state(self):
        """Build the current state dictionary."""
        if not self._device:
            return {}

        # Get current button states and detect press events
        buttons = {}
        button_events = {}

        for button_name in self._button_mappings.values():
            current_state = self._prev_button_states.get(button_name, False)
            buttons[button_name] = current_state

            # Detect button press events (transition from False to True)
            prev_state = getattr(self, f"_prev_press_{button_name}", False)
            button_events[f"{button_name}_pressed"] = current_state and not prev_state
            setattr(self, f"_prev_press_{button_name}", current_state)

        # Package state
        return {
            "buttons": buttons,
            "button_events": button_events,
            "axes": {"stick_x": self._stick_x, "stick_y": self._stick_y},
            "timestamp": time.time(),
        }


class JoyconStreamer(BaseStreamer):
    """
    A streamer for Nintendo Joy-Con controllers following iPhone streamer pattern.
    Supports two modes: locomotion and manipulation.
    """

    def __init__(
        self, left_device_path: str = None, right_device_path: str = None, silent: bool = False
    ):
        """Initialize the Joy-Con streamer."""
        super().__init__()

        self._silent = silent
        self.latest_data = {}

        # Mode management
        self.reset_status()

        # Initialize devices
        self.left_device = JoyConDevice(left_device_path, "left", silent)
        self.right_device = JoyConDevice(right_device_path, "right", silent)

    def reset_status(self):
        """Reset the cache of the streamer."""
        self.current_base_height = 0.74  # Initial base height, 0.74m (standing height)
        self.stand_toggle_cooldown = 0.5  # prevent rapid stand toggling
        self.last_stand_toggle_time = 0

    def start_streaming(self):
        """Start streaming from Joy-Con devices."""
        try:
            if not self._silent:
                print("Starting Joy-Con devices...")

            # Start devices and wait until both are connected
            self.left_device.start()
            self.right_device.start()

            # Wait until both devices are ready
            max_wait_time = 10  # seconds
            start_time = time.time()
            while (time.time() - start_time) < max_wait_time:
                left_state = self.left_device.get_state()
                right_state = self.right_device.get_state()

                if left_state and right_state:
                    if not self._silent:
                        print("Both Joy-Con devices connected successfully!")
                    break

                if not self._silent:
                    print("Waiting for both Joy-Con devices to connect...")
                time.sleep(0.5)
            else:
                print("Warning: Timeout waiting for both Joy-Con devices")

            if not self._silent:
                print("Joy-Con streamer started in unified mode")
                print("Controls:")
                print("  ZL+ZR: Toggle stand command")
                print("  D-pad Up/Down: Increase/Decrease base height")
                print("  Capture (Left): Toggle locomotion policy (e-stop for lower body)")
                print("  L/R shoulders: Left/Right finger open/close")
                print("  Home button: Toggle activation")
                print("  Left stick: Forward/backward and strafe movement")
                print("  Right stick: Yaw rotation")
                print("  A button: Toggle data collection")
                print("  B button: Abort current episode")

        except Exception as e:
            if not self._silent:
                print(f"Failed to start Joy-Con streaming: {e}")
            raise e

    def stop_streaming(self):
        """Stop streaming from Joy-Con devices."""
        if self.left_device:
            self.left_device.stop()
        if self.right_device:
            self.right_device.stop()

        if not self._silent:
            print("Joy-Con streaming stopped")

    def _get_joycon_state(self):
        """
        Get combined Joy-Con state with error handling.
        DDA: Warning and wait until we got all left and right device data.
        """
        left_state = self.left_device.get_state() if self.left_device else {}
        right_state = self.right_device.get_state() if self.right_device else {}

        # Check if we have valid data from both devices
        if not left_state or not right_state:
            if not self._silent:
                missing = []
                if not left_state:
                    missing.append("left")
                if not right_state:
                    missing.append("right")
                print(f"Warning: Missing Joy-Con data from {', '.join(missing)} device(s)")

            # Return empty data structure if either device is missing
            return {
                "left_button_states": {},
                "right_button_states": {},
                "left_stick_inputs": {},
                "right_stick_inputs": {},
            }

        # Combine states
        combined_data = {
            "left_button_states": left_state.get("buttons", {}),
            "right_button_states": right_state.get("buttons", {}),
            "left_stick_inputs": left_state.get("axes", {}),
            "right_stick_inputs": right_state.get("axes", {}),
            "left_button_events": left_state.get("button_events", {}),
            "right_button_events": right_state.get("button_events", {}),
        }
        return combined_data

    def _handle_stand_toggle(self, joycon_data):
        """Handle stand toggle command via ZL+ZR buttons."""
        current_time = time.time()
        if (current_time - self.last_stand_toggle_time) > self.stand_toggle_cooldown:
            # Use button states to check if both triggers are currently pressed
            left_buttons = joycon_data.get("left_button_states", {})
            right_buttons = joycon_data.get("right_button_states", {})

            # Use button events to detect if at least one trigger was just pressed
            left_button_events = joycon_data.get("left_button_events", {})
            right_button_events = joycon_data.get("right_button_events", {})

            # Check if both triggers are held AND at least one was just pressed
            both_triggers_held = left_buttons.get("ZL", False) and right_buttons.get("ZR", False)
            at_least_one_just_pressed = left_button_events.get(
                "ZL_pressed", False
            ) or right_button_events.get("ZR_pressed", False)

            if both_triggers_held and at_least_one_just_pressed:
                self.last_stand_toggle_time = current_time
                if not self._silent:
                    print("Stand toggle activated")
                return True
        return False

    def _apply_dead_zone(self, value, dead_zone):
        """Apply dead zone and normalize."""
        if abs(value) < dead_zone:
            return 0.0
        sign = 1 if value > 0 else -1
        # Normalize the output to be between -1 and 1 after dead zone
        return sign * (abs(value) - dead_zone) / (1.0 - dead_zone)

    def _handle_height_adjustment(self, joycon_data):
        """Handle base height adjustment via d-pad up/down."""
        left_button_states = joycon_data.get("left_button_states", {})

        # Check d-pad button states
        dpad_up = left_button_states.get("dpad_up", False)
        dpad_down = left_button_states.get("dpad_down", False)

        # Incremental height adjustment
        height_increment = 0.005  # Small step per call when button is pressed

        if dpad_up:
            self.current_base_height += height_increment
        elif dpad_down:
            self.current_base_height -= height_increment

        # Clamp to bounds
        self.current_base_height = np.clip(self.current_base_height, 0.2, 0.74)

    def _detect_stand_toggle(self, joycon_data):
        """Detect stand/walk toggle command - triggered by ZL+ZR."""
        return self._handle_stand_toggle(joycon_data)

    def _detect_locomotion_policy_toggle(self, joycon_data):
        """Detect locomotion policy toggle command using left Joy-Con capture button."""
        left_button_events = joycon_data.get("left_button_events", {})
        return left_button_events.get("capture_pressed", False)

    def _detect_emergency_stop(self, joycon_data):
        """Detect emergency stop command (Placeholder - not implemented yet)."""
        return False  # Not implemented yet

    def _detect_data_collection_toggle(self, joycon_data):
        """Detect data collection toggle command using right Joy-Con A button."""
        right_button_events = joycon_data.get("right_button_events", {})
        return right_button_events.get("a_pressed", False)

    def _detect_abort_toggle(self, joycon_data):
        """Detect abort toggle command using right Joy-Con B button."""
        right_button_events = joycon_data.get("right_button_events", {})
        return right_button_events.get("b_pressed", False)

    def _generate_finger_data(self, shoulder_button_pressed):
        """Generate finger position data similar to iPhone streamer."""
        fingertips = np.zeros([25, 4, 4])

        # Set identity matrices for all finger joints
        for i in range(25):
            fingertips[i] = np.eye(4)

        # Control thumb based on shoulder button state (index 4 is thumb tip)
        if shoulder_button_pressed:
            fingertips[4, 0, 3] = 0.0  # closed
        else:
            fingertips[4, 0, 3] = 1.0  # open

        return fingertips

    def _generate_unified_raw_data(self, joycon_data):
        """Generate unified raw_data combining navigation, finger control, and height adjustment."""

        # Extract stick inputs
        left_stick = joycon_data.get("left_stick_inputs", {})
        right_stick = joycon_data.get("right_stick_inputs", {})

        # Extract button/trigger states for finger control
        left_buttons = joycon_data.get("left_button_states", {})
        right_buttons = joycon_data.get("right_button_states", {})

        # Handle d-pad height adjustment
        self._handle_height_adjustment(joycon_data)

        # Map to velocity commands with dead zones and scaling
        DEAD_ZONE = 0.1
        MAX_LINEAR_VEL = 0.2  # m/s
        MAX_ANGULAR_VEL = 0.5  # rad/s

        # Left stick Y for forward/backward (lin_vel_x), X for strafe (lin_vel_y).
        # Right stick X for yaw (ang_vel_z). Verify command signs.
        fwd_bwd_input = left_stick.get("stick_y", 0.0)
        strafe_input = -left_stick.get("stick_x", 0.0)  # Flip sign for intuitive left/right
        yaw_input = -right_stick.get("stick_x", 0.0)

        lin_vel_x = self._apply_dead_zone(fwd_bwd_input, DEAD_ZONE) * MAX_LINEAR_VEL
        lin_vel_y = self._apply_dead_zone(strafe_input, DEAD_ZONE) * MAX_LINEAR_VEL  # Strafe
        ang_vel_z = self._apply_dead_zone(yaw_input, DEAD_ZONE) * MAX_ANGULAR_VEL

        # Extract home button press event for toggle activation
        right_button_events = joycon_data.get("right_button_events", {})
        home_button_pressed = right_button_events.get("home_pressed", False)

        # Map L/R (shoulder buttons) to finger control
        left_shoulder_pressed = left_buttons.get("L", False)
        right_shoulder_pressed = right_buttons.get("R", False)

        # Generate finger data based on shoulder button states
        left_fingers = self._generate_finger_data(left_shoulder_pressed)
        right_fingers = self._generate_finger_data(right_shoulder_pressed)

        return StreamerOutput(
            ik_data={
                "left_fingers": {"position": left_fingers},
                "right_fingers": {"position": right_fingers},
            },
            control_data={
                "base_height_command": self.current_base_height,
                "navigate_cmd": [lin_vel_x, lin_vel_y, ang_vel_z],
                "toggle_stand_command": self._detect_stand_toggle(joycon_data),
                "toggle_policy_action": self._detect_locomotion_policy_toggle(joycon_data),
            },
            teleop_data={
                "toggle_activation": home_button_pressed,
            },
            data_collection_data={
                "toggle_data_collection": self._detect_data_collection_toggle(joycon_data),
                "toggle_data_abort": self._detect_abort_toggle(joycon_data),
            },
            source="joycon",
        )

    def get(self) -> StreamerOutput:
        """
        Return StreamerOutput with unified control data.
        """
        # Get current Joy-Con state
        joycon_data = self._get_joycon_state()

        # Generate unified structured output
        raw_data = self._generate_unified_raw_data(joycon_data)
        return raw_data

    def __del__(self):
        """Cleanup when the streamer is destroyed."""
        self.stop_streaming()


def debug_joycons(controller_type="both", duration=30):
    """Debug function to show Joy-Con devices and capture real-time events."""
    print(f"\n=== JOY-CON DEBUG ({controller_type.upper()}) ===")

    # Show available devices
    devices = [InputDevice(path) for path in list_devices()]
    joycon_devices = [d for d in devices if "Joy-Con" in d.name and "IMU" not in d.name]
    print(f"Found {len(joycon_devices)} Joy-Con devices:")
    for device in joycon_devices:
        print(f"  {device.name} at {device.path}")

    if not joycon_devices:
        print("No Joy-Con devices found!")
        return

    # Select devices to monitor
    devices_to_monitor = []
    if controller_type in ["left", "both"]:
        devices_to_monitor.extend([d for d in joycon_devices if "Joy-Con (L)" in d.name])
    if controller_type in ["right", "both"]:
        devices_to_monitor.extend([d for d in joycon_devices if "Joy-Con (R)" in d.name])

    if not devices_to_monitor:
        print(f"No {controller_type} Joy-Con found!")
        return

    print(f"\nMonitoring events for {duration}s (Press Ctrl+C to stop)...")

    import select

    start_time = time.time()
    try:
        while (time.time() - start_time) < duration:
            r, w, x = select.select(devices_to_monitor, [], [], 0.1)
            for device in r:
                events = device.read()
                for event in events:
                    if event.type in [ecodes.EV_KEY, ecodes.EV_ABS]:
                        event_type = "KEY" if event.type == ecodes.EV_KEY else "ABS"
                        device_name = "L" if "Joy-Con (L)" in device.name else "R"
                        print(f"[{device_name}] {event_type}:{event.code}:{event.value}")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        for device in devices_to_monitor:
            device.close()

    print("=== END DEBUG ===\n")


def test_unified_controls():
    """Test unified control system with all functionality available."""
    print("=== Testing Unified Controls ===")
    print("Controls:")
    print("  ZL+ZR: Toggle stand command")
    print("  D-pad Up/Down: Increase/Decrease base height")
    print("  Capture (Left): Toggle locomotion policy (e-stop for lower body)")
    print("  Left stick: Forward/backward and strafe movement")
    print("  Right stick: Yaw rotation")
    print("  L/R shoulders: Left/Right finger open/close")
    print("  Home button: Toggle activation")
    print("  A button: Toggle data collection")
    print("  B button: Abort current episode")
    print("Press Ctrl+C to stop\n")

    streamer = JoyconStreamer(silent=False)

    try:
        streamer.start_streaming()

        while True:
            data = streamer.get()

            # Extract all control data
            nav_cmd = data.control_data.get("navigate_cmd", [0, 0, 0])
            height_cmd = data.control_data.get("base_height_command", 0.78)
            stand_toggle = data.control_data.get("toggle_stand_command", False)
            policy_action = data.control_data.get("toggle_policy_action", False)
            toggle_activation = data.teleop_data.get("toggle_activation", False)
            toggle_data_collection = data.data_collection_data.get("toggle_data_collection", False)
            toggle_data_abort = data.data_collection_data.get("toggle_data_abort", False)

            # Get finger states
            left_fingers = data.ik_data.get("left_fingers", {}).get(
                "position", np.zeros([25, 4, 4])
            )
            right_fingers = data.ik_data.get("right_fingers", {}).get(
                "position", np.zeros([25, 4, 4])
            )
            left_closed = (
                left_fingers[4, 0, 3] == 0.0 if left_fingers.shape == (25, 4, 4) else False
            )
            right_closed = (
                right_fingers[4, 0, 3] == 0.0 if right_fingers.shape == (25, 4, 4) else False
            )

            print(
                f"UNIFIED - Nav:[{nav_cmd[0]:.2f},{nav_cmd[1]:.2f},{nav_cmd[2]:.2f}] "
                f"Height:{height_cmd:.2f} L:{left_closed} R:{right_closed} "
                f"StandToggle:{stand_toggle} PolicyAction:{policy_action} "
                f"ToggleActivation:{toggle_activation} "
                f"ToggleDataCollection:{toggle_data_collection} ToggleDataAbort:{toggle_data_abort}"
            )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nStopping unified controls test...")
    finally:
        streamer.stop_streaming()


if __name__ == "__main__":
    import sys

    # Check for debug mode
    if len(sys.argv) > 1 and sys.argv[1] == "debug_both":
        debug_joycons(controller_type="both", duration=60)
        sys.exit(0)

    # Check for test mode
    if len(sys.argv) > 1 and sys.argv[1] == "test_unified":
        test_unified_controls()
        sys.exit(0)

    # Default: show usage and run unified controls test
    print("Joy-Con Teleop Streamer - Unified Mode")
    print("Usage:")
    print("  python joycon_streamer.py debug_both     - Debug both Joy-Con devices")
    print("  python joycon_streamer.py test_unified   - Test unified controls")
    print()
    print("Unified Button Mapping:")
    print("| Input | Function |")
    print("|-------|----------|")
    print("| Left Stick Y | Forward/Backward movement |")
    print("| Left Stick X | Strafe Left/Right |")
    print("| Right Stick X | Yaw rotation |")
    print("| D-pad Up | Increase base height |")
    print("| D-pad Down | Decrease base height |")
    print("| L (Left Shoulder) | Left finger open/close |")
    print("| R (Right Shoulder) | Right finger open/close |")
    print("| Home (Right Joy-Con) | Toggle activation |")
    print("| Capture (Left Joy-Con) | Toggle locomotion policy (e-stop) |")
    print("| ZL+ZR (Triggers) | Toggle stand command |")
    print("| A (Right Joy-Con) | Start/Stop Data Collection |")
    print("| B (Right Joy-Con) | Abort Current Episode |")
    print()
    print("Running unified controls test by default...")
    print()

    test_unified_controls()
