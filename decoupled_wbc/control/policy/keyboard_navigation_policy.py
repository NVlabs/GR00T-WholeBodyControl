from typing import Any, Dict, Optional

import numpy as np

from decoupled_wbc.control.base.policy import Policy


class KeyboardNavigationPolicy(Policy):
    def __init__(
        self,
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 0.5,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize the navigation policy.

        Args:
            max_linear_velocity: Maximum linear velocity in m/s (for x and y components)
            max_angular_velocity: Maximum angular velocity in rad/s (for yaw component)
            **kwargs: Additional arguments passed to the base Policy class
        """
        super().__init__(**kwargs)
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.verbose = verbose

        # Initialize velocity commands
        self.lin_vel_command = np.zeros(2, dtype=np.float32)  # [vx, vy]
        self.ang_vel_command = np.zeros(1, dtype=np.float32)  # [wz]

    def get_action(self, time: Optional[float] = None) -> Dict[str, Any]:
        """
        Get the action to execute based on current state.

        Args:
            time: Current time (optional)

        Returns:
            Dict containing the action to execute with:
                - navigate_cmd: np.array([vx, vy, wz]) where:
                    - vx: linear velocity in x direction (m/s)
                    - vy: linear velocity in y direction (m/s)
                    - wz: angular velocity around z axis (rad/s)
        """
        # Combine linear and angular velocities into a single command
        # Ensure velocities are within limits
        vx = np.clip(self.lin_vel_command[0], -self.max_linear_velocity, self.max_linear_velocity)
        vy = np.clip(self.lin_vel_command[1], -self.max_linear_velocity, self.max_linear_velocity)
        wz = np.clip(self.ang_vel_command[0], -self.max_angular_velocity, self.max_angular_velocity)

        navigate_cmd = np.array([vx, vy, wz], dtype=np.float32)

        action = {"navigate_cmd": navigate_cmd}
        return action

    def handle_keyboard_button(self, keycode: str):
        """
        Handle keyboard inputs for navigation control.

        Args:
            keycode: The key that was pressed
        """
        if keycode == "w":
            self.lin_vel_command[0] += 0.1  # Increase forward velocity
        elif keycode == "s":
            self.lin_vel_command[0] -= 0.1  # Increase backward velocity
        elif keycode == "a":
            self.lin_vel_command[1] += 0.1  # Increase left velocity
        elif keycode == "d":
            self.lin_vel_command[1] -= 0.1  # Increase right velocity
        elif keycode == "q":
            self.ang_vel_command[0] += 0.1  # Increase counter-clockwise rotation
        elif keycode == "e":
            self.ang_vel_command[0] -= 0.1  # Increase clockwise rotation
        elif keycode == "z":
            # Reset all velocities
            self.lin_vel_command[:] = 0.0
            self.ang_vel_command[:] = 0.0
            if self.verbose:
                print("Navigation policy: Reset all velocity commands to zero")

        # Print current velocities after any keyboard input
        if self.verbose:
            print(f"Nav lin vel: ({self.lin_vel_command[0]:.2f}, {self.lin_vel_command[1]:.2f})")
            print(f"Nav ang vel: {self.ang_vel_command[0]:.2f}")
