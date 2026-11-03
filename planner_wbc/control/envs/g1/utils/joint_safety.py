"""Joint safety monitor for G1 robot.

This module implements safety monitoring for arm and finger joint velocities using
joint groups defined in the robot model's supplemental info. Leg joints are not monitored.
"""

from datetime import datetime
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from decoupled_wbc.data.viz.rerun_viz import RerunViz


class JointSafetyMonitor:
    """Monitor joint velocities for G1 robot arms and hands."""

    # Velocity limits in rad/s
    ARM_VELOCITY_LIMIT = 6.0  # rad/s for arm joints
    HAND_VELOCITY_LIMIT = 50.0  # rad/s for finger joints

    def __init__(self, robot_model, enable_viz: bool = False, env_type: str = "real"):
        """Initialize joint safety monitor.

        Args:
            robot_model: The robot model containing joint information
            enable_viz: If True, enable rerun visualization (default False)
            env_type: Environment type - "sim" or "real" (default "real")
        """
        self.robot_model = robot_model
        self.safety_margin = 1.0  # Hardcoded safety margin
        self.enable_viz = enable_viz
        self.env_type = env_type

        # Startup ramping parameters
        self.control_frequency = 50  # Hz, hardcoded from run_g1_control_loop.py
        self.ramp_duration_steps = int(2.0 * self.control_frequency)  # 2 seconds * 50Hz = 100 steps
        self.startup_counter = 0
        self.initial_positions = None
        self.startup_complete = False

        # Initialize velocity and position limits for monitored joints
        self.velocity_limits = {}
        self.position_limits = {}
        self._initialize_limits()

        # Track violations for reporting
        self.violations = []

        # Initialize visualization
        self.right_arm_indices = None
        self.right_arm_joint_names = []
        self.left_arm_indices = None
        self.left_arm_joint_names = []
        self.right_hand_indices = None
        self.right_hand_joint_names = []
        self.left_hand_indices = None
        self.left_hand_joint_names = []
        try:
            arm_indices = self.robot_model.get_joint_group_indices("arms")
            all_joint_names = [self.robot_model.joint_names[i] for i in arm_indices]
            # Filter for right and left arm joints
            self.right_arm_joint_names = [
                name for name in all_joint_names if name.startswith("right_")
            ]
            self.right_arm_indices = [
                self.robot_model.joint_to_dof_index[name] for name in self.right_arm_joint_names
            ]
            self.left_arm_joint_names = [
                name for name in all_joint_names if name.startswith("left_")
            ]
            self.left_arm_indices = [
                self.robot_model.joint_to_dof_index[name] for name in self.left_arm_joint_names
            ]
            # Hand joints
            hand_indices = self.robot_model.get_joint_group_indices("hands")
            all_hand_names = [self.robot_model.joint_names[i] for i in hand_indices]
            self.right_hand_joint_names = [
                name for name in all_hand_names if name.startswith("right_")
            ]
            self.right_hand_indices = [
                self.robot_model.joint_to_dof_index[name] for name in self.right_hand_joint_names
            ]
            self.left_hand_joint_names = [
                name for name in all_hand_names if name.startswith("left_")
            ]
            self.left_hand_indices = [
                self.robot_model.joint_to_dof_index[name] for name in self.left_hand_joint_names
            ]
        except ValueError as e:
            print(f"[JointSafetyMonitor] Warning: Could not initialize arm/hand visualization: {e}")
        except Exception:
            pass

        # Use single tensor_key for each plot
        self.right_arm_pos_key = "right_arm_qpos"
        self.left_arm_pos_key = "left_arm_qpos"
        self.right_arm_vel_key = "right_arm_dq"
        self.left_arm_vel_key = "left_arm_dq"
        self.right_hand_pos_key = "right_hand_qpos"
        self.left_hand_pos_key = "left_hand_qpos"
        self.right_hand_vel_key = "right_hand_dq"
        self.left_hand_vel_key = "left_hand_dq"

        # Define a consistent color palette for up to 8 joints (tab10 + extra)
        self.joint_colors = [
            [31, 119, 180],  # blue
            [255, 127, 14],  # orange
            [44, 160, 44],  # green
            [214, 39, 40],  # red
            [148, 103, 189],  # purple
            [140, 86, 75],  # brown
            [227, 119, 194],  # pink
            [127, 127, 127],  # gray (for 8th joint if needed)
        ]

        # Initialize Rerun visualization only if enabled
        self.viz = None
        if self.enable_viz:
            try:
                self.viz = RerunViz(
                    image_keys=[],
                    tensor_keys=[
                        self.right_arm_pos_key,
                        self.left_arm_pos_key,
                        self.right_arm_vel_key,
                        self.left_arm_vel_key,
                        self.right_hand_pos_key,
                        self.left_hand_pos_key,
                        self.right_hand_vel_key,
                        self.left_hand_vel_key,
                    ],
                    window_size=10.0,
                    app_name="joint_safety_monitor",
                )
            except Exception:
                self.viz = None

    def _initialize_limits(self):
        """Initialize velocity and position limits for arm and hand joints using robot model joint groups."""
        if self.robot_model.supplemental_info is None:
            raise ValueError("Robot model must have supplemental_info to use joint groups")

        # Get arm joint indices from robot model joint groups
        try:
            arm_indices = self.robot_model.get_joint_group_indices("arms")
            arm_joint_names = [self.robot_model.joint_names[i] for i in arm_indices]

            for joint_name in arm_joint_names:
                # Set velocity limits
                vel_limit = self.ARM_VELOCITY_LIMIT * self.safety_margin
                self.velocity_limits[joint_name] = {"min": -vel_limit, "max": vel_limit}

                # Set position limits from robot model
                if joint_name in self.robot_model.joint_to_dof_index:
                    joint_idx = self.robot_model.joint_to_dof_index[joint_name]
                    # Adjust index for floating base if present
                    limit_idx = joint_idx - (7 if self.robot_model.is_floating_base_model else 0)

                    if 0 <= limit_idx < len(self.robot_model.lower_joint_limits):
                        pos_min = self.robot_model.lower_joint_limits[limit_idx]
                        pos_max = self.robot_model.upper_joint_limits[limit_idx]

                        # Apply safety margin to position limits
                        pos_range = pos_max - pos_min
                        margin = pos_range * (1.0 - self.safety_margin) / 2.0

                        self.position_limits[joint_name] = {
                            "min": pos_min + margin,
                            "max": pos_max - margin,
                        }
        except ValueError as e:
            print(f"[JointSafetyMonitor] Warning: Could not find 'arms' joint group: {e}")

        # Get hand joint indices from robot model joint groups
        try:
            hand_indices = self.robot_model.get_joint_group_indices("hands")
            hand_joint_names = [self.robot_model.joint_names[i] for i in hand_indices]

            for joint_name in hand_joint_names:
                # Set velocity limits only for hands (no position limits for now)
                vel_limit = self.HAND_VELOCITY_LIMIT * self.safety_margin
                self.velocity_limits[joint_name] = {"min": -vel_limit, "max": vel_limit}
        except ValueError as e:
            print(f"[JointSafetyMonitor] Warning: Could not find 'hands' joint group: {e}")

    def check_safety(self, obs: Dict, action: Dict) -> Tuple[bool, List[Dict]]:
        """Check if current velocities and positions are within safe bounds.

        Args:
            obs: Observation dictionary containing joint positions and velocities
            action: Action dictionary containing target positions

        Returns:
            (is_safe, violations): Tuple of safety status and list of violations
            Note: is_safe=False only for velocity violations (triggers shutdown)
                  Position violations are warnings only (don't affect is_safe)
        """
        self.violations = []
        is_safe = True
        joint_names = self.robot_model.joint_names

        # Check current joint velocities (critical - triggers shutdown)
        if "dq" in obs:
            joint_velocities = obs["dq"]

            for i, joint_name in enumerate(joint_names):
                # Only check monitored joints
                if joint_name not in self.velocity_limits:
                    continue

                if i < len(joint_velocities):
                    velocity = joint_velocities[i]
                    limits = self.velocity_limits[joint_name]

                    if velocity < limits["min"] or velocity > limits["max"]:
                        violation = {
                            "joint": joint_name,
                            "type": "velocity",
                            "value": velocity,
                            "limit_min": limits["min"],
                            "limit_max": limits["max"],
                            "exceeded_by": self._calculate_exceeded_percentage(
                                velocity, limits["min"], limits["max"]
                            ),
                            "critical": True,  # Velocity violations are critical
                        }
                        self.violations.append(violation)
                        is_safe = False

        # Check current joint positions (warning only - no shutdown)
        if "q" in obs:
            joint_positions = obs["q"]

            for i, joint_name in enumerate(joint_names):
                # Only check joints with position limits (arms)
                if joint_name not in self.position_limits:
                    continue

                if i < len(joint_positions):
                    position = joint_positions[i]
                    limits = self.position_limits[joint_name]

                    if position < limits["min"] or position > limits["max"]:
                        violation = {
                            "joint": joint_name,
                            "type": "position",
                            "value": position,
                            "limit_min": limits["min"],
                            "limit_max": limits["max"],
                            "exceeded_by": self._calculate_exceeded_percentage(
                                position, limits["min"], limits["max"]
                            ),
                            "critical": False,  # Position violations are warnings only
                        }
                        self.violations.append(violation)
                        # Don't set is_safe = False for position violations

        return is_safe, self.violations

    def _calculate_exceeded_percentage(
        self, value: float, limit_min: float, limit_max: float
    ) -> float:
        """Calculate by how much percentage a value exceeds the limits."""
        if value < limit_min:
            return abs((value - limit_min) / limit_min) * 100
        elif value > limit_max:
            return abs((value - limit_max) / limit_max) * 100
        return 0.0

    def get_safe_action(self, obs: Dict, original_action: Dict) -> Dict:
        """Generate a safe action with startup ramping for smooth initialization.

        Args:
            obs: Observation dictionary containing current joint positions
            original_action: The original action that may cause violations

        Returns:
            Safe action with startup ramping applied if within ramp duration
        """
        safe_action = original_action.copy()

        # Handle startup ramping for arm joints
        if not self.startup_complete:
            if self.initial_positions is None and "q" in obs:
                # Store initial positions from first observation
                self.initial_positions = obs["q"].copy()

            if (
                self.startup_counter < self.ramp_duration_steps
                and self.initial_positions is not None
                and "q" in safe_action
            ):
                # Ramp factor: 0.0 at start → 1.0 at end
                ramp_factor = self.startup_counter / self.ramp_duration_steps

                # Apply ramping only to monitored arm joints
                for joint_name in self.velocity_limits:  # Only monitored arm joints
                    if joint_name in self.robot_model.joint_to_dof_index:
                        joint_idx = self.robot_model.joint_to_dof_index[joint_name]
                        if joint_idx < len(safe_action["q"]) and joint_idx < len(
                            self.initial_positions
                        ):
                            initial_pos = self.initial_positions[joint_idx]
                            target_pos = original_action["q"][joint_idx]
                            # Linear interpolation: initial + ramp_factor * (target - initial)
                            safe_action["q"][joint_idx] = initial_pos + ramp_factor * (
                                target_pos - initial_pos
                            )

                # Increment counter for next iteration
                self.startup_counter += 1
            else:
                # Ramping complete - use original actions
                self.startup_complete = True

        return safe_action

    def get_violation_report(self, violations: Optional[List[Dict]] = None) -> str:
        """Generate a formatted error report for violations.

        Args:
            violations: List of violations to report (uses self.violations if None)

        Returns:
            Formatted error message string
        """
        if violations is None:
            violations = self.violations

        if not violations:
            return "No violations detected."

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Check if these are critical violations or warnings
        critical_violations = [v for v in violations if v.get("critical", True)]
        warning_violations = [v for v in violations if not v.get("critical", True)]

        if critical_violations and warning_violations:
            report = f"Joint safety bounds exceeded!\nTimestamp: {timestamp}\nViolations:\n"
        elif critical_violations:
            report = f"Joint safety bounds exceeded!\nTimestamp: {timestamp}\nViolations:\n"
        else:
            report = f"Joint position warnings!\nTimestamp: {timestamp}\nWarnings:\n"

        for violation in violations:
            joint = violation["joint"]
            vtype = violation["type"]
            value = violation["value"]
            exceeded = violation["exceeded_by"]
            limit_min = violation["limit_min"]
            limit_max = violation["limit_max"]

            if vtype == "velocity":
                report += f"  - {joint}: {vtype}={value:.3f} rad/s "
                report += f"(limit: ±{limit_max:.3f} rad/s) - "
                report += f"EXCEEDED by {exceeded:.1f}%\n"
            elif vtype == "position":
                report += f"  - {joint}: {vtype}={value:.3f} rad "
                report += f"(limits: [{limit_min:.3f}, {limit_max:.3f}] rad) - "
                report += f"EXCEEDED by {exceeded:.1f}%\n"

        # Add appropriate action message
        if critical_violations:
            report += "Action: Safe mode engaged (kp=0, tau=0). System shutdown initiated.\n"
            report += "Please restart Docker container to resume operation."
        else:
            report += "Action: Position warning only. Robot continues operation."

        return report

    def handle_violations(self, obs: Dict, action: Dict) -> Dict:
        """Check safety and handle violations appropriately.

        Args:
            obs: Observation dictionary
            action: Action dictionary

        Returns:
            Dict with keys:
            - 'safe_to_continue': bool - whether robot should continue operation
            - 'action': Dict - potentially modified safe action
            - 'shutdown_required': bool - whether system shutdown is needed
        """
        is_safe, violations = self.check_safety(obs, action)

        # Apply startup ramping (always, regardless of violations)
        safe_action = self.get_safe_action(obs, action)

        # Visualize arm and hand joint positions and velocities if enabled
        if self.enable_viz:
            if (
                self.right_arm_indices is not None
                and self.left_arm_indices is not None
                and self.right_hand_indices is not None
                and self.left_hand_indices is not None
                and "q" in obs
                and "dq" in obs
                and self.viz is not None
            ):
                try:
                    right_arm_positions = obs["q"][self.right_arm_indices]
                    left_arm_positions = obs["q"][self.left_arm_indices]
                    right_arm_velocities = obs["dq"][self.right_arm_indices]
                    left_arm_velocities = obs["dq"][self.left_arm_indices]
                    right_hand_positions = obs["q"][self.right_hand_indices]
                    left_hand_positions = obs["q"][self.left_hand_indices]
                    right_hand_velocities = obs["dq"][self.right_hand_indices]
                    left_hand_velocities = obs["dq"][self.left_hand_indices]
                    tensor_dict = {
                        self.right_arm_pos_key: right_arm_positions,
                        self.left_arm_pos_key: left_arm_positions,
                        self.right_arm_vel_key: right_arm_velocities,
                        self.left_arm_vel_key: left_arm_velocities,
                        self.right_hand_pos_key: right_hand_positions,
                        self.left_hand_pos_key: left_hand_positions,
                        self.right_hand_vel_key: right_hand_velocities,
                        self.left_hand_vel_key: left_hand_velocities,
                    }
                    self.viz.plot_tensors(tensor_dict, time.time())
                except Exception:
                    pass

        if not violations:
            return {"safe_to_continue": True, "action": safe_action, "shutdown_required": False}

        # Separate critical (velocity) and warning (position) violations
        critical_violations = [v for v in violations if v.get("critical", True)]
        # warning_violations = [v for v in violations if not v.get('critical', True)]

        # Print warnings for position violations
        # if warning_violations:
        # warning_msg = self.get_violation_report(warning_violations)
        # print(f"[SAFETY WARNING] {warning_msg}")

        # Handle critical violations (velocity) - trigger shutdown
        if not is_safe and critical_violations:
            error_msg = self.get_violation_report(critical_violations)
            if self.env_type == "real":
                print(f"[SAFETY VIOLATION] {error_msg}")
                self.trigger_system_shutdown()

            return {"safe_to_continue": False, "action": safe_action, "shutdown_required": True}

        # Only position violations - continue with safe action
        return {"safe_to_continue": True, "action": safe_action, "shutdown_required": False}

    def trigger_system_shutdown(self):
        """Trigger system shutdown after safety violation."""
        print("\n[SAFETY] Initiating system shutdown due to safety violation...")
        sys.exit(1)


def main():
    """Test the joint safety monitor with joint groups."""
    print("Testing joint safety monitor with joint groups...")

    try:
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

        # Instantiate robot model
        robot_model = instantiate_g1_robot_model()
        print(f"Robot model created with {len(robot_model.joint_names)} joints")

        # Create safety monitor
        safety_monitor = JointSafetyMonitor(robot_model)
        print("Safety monitor created successfully!")
        print(f"Monitoring {len(safety_monitor.velocity_limits)} joints")

        # Print monitored joints
        print("\nVelocity limits:")
        for joint_name, limits in safety_monitor.velocity_limits.items():
            print(f"  - {joint_name}: ±{limits['max']:.2f} rad/s")

        print(f"\nPosition limits (arms only): {len(safety_monitor.position_limits)} joints")
        for joint_name, limits in safety_monitor.position_limits.items():
            print(f"  - {joint_name}: [{limits['min']:.3f}, {limits['max']:.3f}] rad")

        # Test safety checking with safe values
        print("\n--- Testing Safety Checking ---")

        # Create mock observation with safe values
        safe_obs = {
            "q": np.zeros(robot_model.num_dofs),  # All joints at zero position
            "dq": np.zeros(robot_model.num_dofs),  # All joints at zero velocity
        }
        safe_action = {"q": np.zeros(robot_model.num_dofs)}

        # Test handle_violations method
        result = safety_monitor.handle_violations(safe_obs, safe_action)
        print(
            f"Safe values test: safe_to_continue={result['safe_to_continue']}, "
            f"shutdown_required={result['shutdown_required']}"
        )

        # Test with unsafe velocity
        unsafe_obs = safe_obs.copy()
        unsafe_obs["dq"] = np.zeros(robot_model.num_dofs)
        # Set left shoulder pitch velocity to exceed limit
        left_shoulder_idx = robot_model.dof_index("left_shoulder_pitch_joint")
        unsafe_obs["dq"][left_shoulder_idx] = 6.0  # Exceeds 5.0 rad/s limit

        print("\nUnsafe velocity test:")
        result = safety_monitor.handle_violations(unsafe_obs, safe_action)
        print(
            f"  safe_to_continue={result['safe_to_continue']}, shutdown_required={result['shutdown_required']}"
        )

        # Test with unsafe position only
        unsafe_pos_obs = safe_obs.copy()
        unsafe_pos_obs["q"] = np.zeros(robot_model.num_dofs)
        # Set left shoulder pitch position to exceed limit
        unsafe_pos_obs["q"][left_shoulder_idx] = -4.0  # Exceeds lower limit of -3.089

        print("\nUnsafe position test:")
        result = safety_monitor.handle_violations(unsafe_pos_obs, safe_action)
        print(
            f"  safe_to_continue={result['safe_to_continue']}, shutdown_required={result['shutdown_required']}"
        )

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
