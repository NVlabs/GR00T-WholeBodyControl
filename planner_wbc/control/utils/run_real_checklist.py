#!/usr/bin/env python3

import sys


def check_real_deployment(extra_args):
    """Check if this is a real robot deployment."""
    is_real_deployment = False

    # Check if interface argument is provided and not 'lo' or 'lo0'
    for i, arg in enumerate(extra_args):
        if arg == "--interface":
            # Get the next argument (interface value)
            if i + 1 < len(extra_args):
                interface_value = extra_args[i + 1]
                if interface_value not in ["lo", "lo0"]:
                    is_real_deployment = True
                    print(f"Real deployment detected: interface = {interface_value}")
                    break
                else:
                    print(f"Simulation deployment detected: interface = {interface_value}")

    # If no interface specified, assume simulation (default is 'lo' in deploy_g1.py)
    if not is_real_deployment:
        print("No interface specified - assuming simulation (default interface = lo)")

    return is_real_deployment


def show_deployment_checklist():
    """Show deployment checklist and get confirmation."""
    checklist_content = """═══════════════════════════════════════════════════════════════════════════════
                          G1 ROBOT DEPLOYMENT CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

⚠️  SAFETY VERIFICATION - Complete ALL checks before deployment

PRE-DEPLOYMENT CHECKLIST:

□  Sim2Sim Verification
   Test in simulation first with interface set to 'sim' before real deployment

□  Camera System Check
   Test real camera with simulation environment before full deployment

□  State Reading Validation
   • Disable action queue
   • Verify sensor readings (IMU, joints, fingers)
   • Use rerun for visualization
   • Contact: Dennis Da (xda@nvidia.com) for assistance

□  Low Gain Test
   • Start with low kp values (2-5x lower than normal)
   • Keep kd values unchanged

□  Clear Workspace
   • Remove obstacles and avoid tables
   • Ensure adequate clearance in all directions

□  Emergency Stop Ready
   Ensure access to at least one:
   • Keyboard e-stop
   • Joycon controller
   • External power cutoff

═══════════════════════════════════════════════════════════════════════════════
🚨 EMERGENCY: Press ` at any time to stop all processes
📹 RECORDING: Connect a webcam to your computer to record the experiment
═══════════════════════════════════════════════════════════════════════════════

Usages:

- hit `      to stop all processes
- hit Ctrl+C to stop single process
- hit Ctrl+\ to quit the tmux
"""

    print("")
    print("🚨 REAL ROBOT DEPLOYMENT DETECTED 🚨")
    print("")
    print(checklist_content)
    print("")

    # Get user confirmation
    while True:
        user_input = input("Continue with deployment? [Y/n]: ").strip()

        # Default to Y if empty input
        if not user_input:
            user_input = "Y"

        user_input_upper = user_input.upper()

        if user_input_upper in ["Y", "YES"]:
            print("")
            print("✅ Deployment confirmed. Proceeding with robot deployment...")
            print("")
            return True
        elif user_input_upper in ["N", "NO"]:
            print("")
            print("❌ Deployment aborted by user.")
            print("")
            return False
        else:
            print(
                "❌ Invalid input. Please enter 'Y' for yes, 'N' for no, or press Enter for default (Y)."
            )


def main():
    """Main function."""
    # Always show the checklist
    if not show_deployment_checklist():
        print("Deployment cancelled.")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    main()
