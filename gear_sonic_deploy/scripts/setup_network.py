#!/usr/bin/env python3
"""
Setup a network interface for communication with the Unitree G1 robot.

Assigns a static IP to the selected ethernet interface so the host machine
can communicate with the robot over the 192.168.123.x subnet.

Requirements: Linux (Ubuntu 20.04/22.04/24.04), iproute2, sudo privileges.
ufw is optional — if present, a scoped allow rule is added for the robot
interface rather than disabling the firewall globally.

Usage:
    python3 setup_network.py                  # interactive robot + interface selection
    python3 setup_network.py -i enp5s0        # skip interface selection
"""

import argparse
import platform
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Robot configurations — host IP is what gets assigned to the workstation NIC;
# robot_ip_address is used only for the post-setup connectivity ping.
# ---------------------------------------------------------------------------
ROBOT_CONFIGURATIONS = {
    "Unitree G1": {
        "ip_address": "192.168.123.222",
        "subnet_mask": "255.255.255.0",
        "gateway": "",
        "connection_name": "Unitree G1",
        "robot_ip_address": "192.168.123.164",
    },
}


class Colors:
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    NC = "\033[0m"


def print_info(msg: str) -> None:
    print(f"{Colors.GREEN}[INFO]{Colors.NC} {msg}")


def print_warning(msg: str) -> None:
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")


def print_error(msg: str) -> None:
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def check_linux() -> None:
    if platform.system() != "Linux":
        print_error(
            f"Unsupported OS: {platform.system()}. "
            "This script requires Linux (Ubuntu 20.04/22.04/24.04)."
        )
        sys.exit(1)


def check_sudo() -> None:
    if not shutil.which("sudo"):
        print_error("sudo not found. Please install sudo or run as root.")
        sys.exit(1)
    result = subprocess.run(["sudo", "-n", "true"], capture_output=True)
    if result.returncode != 0:
        print_info("Some commands require sudo. You may be prompted for your password.")


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# Robot selection
# ---------------------------------------------------------------------------

def select_robot_type() -> dict[str, str]:
    print_info("Select robot type:")
    print()
    robot_names = list(ROBOT_CONFIGURATIONS.keys())
    for i, name in enumerate(robot_names, 1):
        print(f"  {i}) {name}")
    print()

    while True:
        try:
            choice = int(input(f"Select robot type (1-{len(robot_names)}): "))
            if 1 <= choice <= len(robot_names):
                name = robot_names[choice - 1]
                config = ROBOT_CONFIGURATIONS[name].copy()
                config["robot_type"] = name
                print_info(f"Selected: {name}")
                print()
                print_info(f"Host IP to assign : {config['ip_address']}")
                print_info(f"Subnet mask       : {config['subnet_mask']}")
                print_info(f"Robot IP (ping)   : {config['robot_ip_address']}")
                print()
                return config
            print_error(f"Enter a number between 1 and {len(robot_names)}.")
        except ValueError:
            print_error("Invalid input. Enter a number.")


# ---------------------------------------------------------------------------
# Interface enumeration
# ---------------------------------------------------------------------------

def get_available_interfaces() -> list[tuple[str, str, str]]:
    """Return [(name, state, ip_or_'No IP assigned'), ...] for physical NICs."""
    result = run_command(["sudo", "ip", "link", "show"])
    interfaces = []

    for line in result.stdout.splitlines():
        parts = line.split(":")
        if not (parts[0].strip().isdigit() and len(parts) >= 2):
            continue

        iface = parts[1].strip()
        # Skip loopback and well-known virtual prefixes.
        if iface == "lo" or any(
            iface.startswith(p) for p in ["docker", "veth", "br-", "virbr", "vmnet"]
        ):
            continue

        # State
        state_result = run_command(["sudo", "ip", "link", "show", iface], check=False)
        state = "UNKNOWN"
        for sl in state_result.stdout.splitlines():
            if "state" in sl:
                state = sl.split("state")[1].strip().split()[0]
                break

        # Current IP
        ip_result = run_command(["sudo", "ip", "addr", "show", iface], check=False)
        ip_info = "No IP assigned"
        for il in ip_result.stdout.splitlines():
            if "inet " in il and "127.0.0.1" not in il:
                ip_info = il.split("inet ")[1].split()[0]
                break

        interfaces.append((iface, state, ip_info))

    return interfaces


def select_interface() -> str:
    print_info("Scanning for available network interfaces...")
    print()
    interfaces = get_available_interfaces()

    if not interfaces:
        print_error("No network interfaces found.")
        sys.exit(1)

    print_info(f"Found {len(interfaces)} network interface(s):")
    print()

    for i, (iface, state, ip_info) in enumerate(interfaces, 1):
        if state == "UP":
            state_str = f"{Colors.GREEN}UP{Colors.NC}"
        elif state == "DOWN":
            state_str = f"{Colors.RED}DOWN{Colors.NC}"
        else:
            state_str = f"{Colors.YELLOW}{state}{Colors.NC}"
        print(f"  {i}) {iface} ({state_str}) - {ip_info}")

    print()
    while True:
        try:
            choice = int(input(f"Select network interface (1-{len(interfaces)}): "))
            if 1 <= choice <= len(interfaces):
                iface = interfaces[choice - 1][0]
                print_info(f"Selected interface: {iface}")
                return iface
            print_error(f"Enter a number between 1 and {len(interfaces)}.")
        except ValueError:
            print_error("Invalid input. Enter a number.")


# ---------------------------------------------------------------------------
# Backup
# ---------------------------------------------------------------------------

def backup_config(config: dict[str, str], interface: str) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_file = f"/tmp/network_backup_{timestamp}.txt"
    print_info(f"Backing up current interface config to {backup_file}")

    with open(backup_file, "w", encoding="utf-8") as f:
        f.write(f"# Network configuration backup - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Robot type : {config['robot_type']}\n")
        f.write(f"# Interface  : {interface}\n")
        f.write(f"# Host IP    : {config['ip_address']}\n")
        f.write(f"# Subnet mask: {config['subnet_mask']}\n\n")

        f.write("# Current IP addresses:\n")
        r = run_command(["sudo", "ip", "addr", "show", interface], check=False)
        for line in r.stdout.splitlines():
            if "inet" in line:
                f.write(line + "\n")

        f.write("\n# Current routes for this interface:\n")
        r = run_command(["sudo", "ip", "route", "show"], check=False)
        for line in r.stdout.splitlines():
            if interface in line:
                f.write(line + "\n")

    print_info(f"Backup saved: {backup_file}")
    return backup_file


# ---------------------------------------------------------------------------
# Firewall — scoped rule, not global disable
# ---------------------------------------------------------------------------

def configure_firewall(interface: str) -> None:
    if not shutil.which("ufw"):
        print_warning("ufw not found — skipping firewall configuration.")
        return

    status_result = run_command(["sudo", "ufw", "status"], check=False)
    if "inactive" in status_result.stdout.lower():
        print_info("ufw is inactive — no firewall rules needed.")
        return

    print_info(f"Adding scoped ufw rules for interface {interface} (robot subnet only)...")
    in_result = run_command(["sudo", "ufw", "allow", "in", "on", interface], check=False)
    out_result = run_command(["sudo", "ufw", "allow", "out", "on", interface], check=False)

    if in_result.returncode == 0 and out_result.returncode == 0:
        print_info(f"ufw: allowed all traffic in/out on {interface}.")
    else:
        print_warning("Could not add ufw rules — robot communication may be blocked.")
        print_warning("Run manually: sudo ufw allow in on <interface> && sudo ufw allow out on <interface>")


# ---------------------------------------------------------------------------
# Interface configuration
# ---------------------------------------------------------------------------

def configure_interface(config: dict[str, str], interface: str) -> None:
    print_info(f"Configuring {interface} for {config['connection_name']}...")

    configure_firewall(interface)

    # Flush existing addresses and bring the interface up.
    run_command(["sudo", "ip", "addr", "flush", "dev", interface], check=False)
    run_command(["sudo", "ip", "link", "set", interface, "up"])
    time.sleep(1)

    # Assign static IP.
    cidr = sum(bin(int(x)).count("1") for x in config["subnet_mask"].split("."))
    run_command(["sudo", "ip", "addr", "add", f"{config['ip_address']}/{cidr}", "dev", interface])

    # Add default route only when a gateway is specified.
    if config["gateway"]:
        run_command(
            ["sudo", "ip", "route", "add", "default", "via", config["gateway"], "dev", interface],
            check=False,
        )

    # Verify.
    result = run_command(["sudo", "ip", "addr", "show", interface], check=False)
    if config["ip_address"] in result.stdout:
        print_info("Interface configured successfully.")
        print_info(f"  Interface : {interface}")
        print_info(f"  Host IP   : {config['ip_address']}/{cidr}")
        if config["gateway"]:
            print_info(f"  Gateway   : {config['gateway']}")
    else:
        print_error("Failed to assign IP address to interface.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Connectivity checks
# ---------------------------------------------------------------------------

def check_interface_state(interface: str) -> None:
    result = run_command(["sudo", "ip", "link", "show", interface], check=False)
    if "state UP" in result.stdout:
        print_info(f"Interface {interface} is UP.")
    else:
        print_warning(f"Interface {interface} is not UP — check the ethernet cable.")


def ping_robot(robot_ip: str) -> bool:
    print_info(f"Pinging robot at {robot_ip}...")
    result = run_command(["ping", "-c", "1", "-W", "3", robot_ip], check=False)
    if result.returncode == 0:
        print_info(f"Robot at {robot_ip} is reachable.")
        return True
    print_warning(f"Robot at {robot_ip} is not reachable.")
    print_warning("This is expected if the robot is powered off or not yet connected.")
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configure a network interface for Unitree G1 robot communication."
    )
    parser.add_argument("-i", "--interface", help="Network interface to configure (e.g. enp5s0)")
    args = parser.parse_args()

    print_info("Starting network setup for robot communication.")

    check_linux()
    check_sudo()

    config = select_robot_type()

    if args.interface:
        result = run_command(["sudo", "ip", "link", "show", args.interface], check=False)
        if result.returncode != 0:
            print_error(f"Interface '{args.interface}' does not exist.")
            sys.exit(1)
        interface = args.interface
        print_info(f"Using specified interface: {interface}")
    else:
        interface = select_interface()

    backup_config(config, interface)
    configure_interface(config, interface)
    check_interface_state(interface)
    ping_robot(config["robot_ip_address"])

    print_info("Network setup complete.")
    print_info(f"Interface {interface} is ready for {config['connection_name']} communication.")
    print_warning("Note: this configuration is not persistent and will reset on reboot.")


if __name__ == "__main__":
    main()

