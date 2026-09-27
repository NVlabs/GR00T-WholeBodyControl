#!/usr/bin/env python3
"""
Assign a static IP to the wired ethernet port connected to the Unitree G1 robot.

The G1 communicates exclusively over a dedicated wired ethernet link on the
192.168.123.0/24 subnet (robot at .164, host at .222). This connection carries
both DDS motor control traffic (500 Hz, latency-critical) and the camera stream.
WiFi cannot substitute for this link.

This script assigns the static IP via NetworkManager so the assignment is
NM-aware and persists for the session without being reverted by the NM daemon.

Requirements: Ubuntu 20.04 / 22.04 / 24.04, NetworkManager (nmcli), sudo privileges.
ufw is optional — if present, a scoped allow rule is added for the robot
subnet (192.168.123.0/24) rather than the entire interface.

Usage:
    python3 setup_network.py                  # interactive robot + interface selection
    python3 setup_network.py -i enp5s0        # manual interface selection
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
        "prefix_length": "24",
        "gateway": "",
        "connection_name": "unitree-g1-robot",
        "robot_ip_address": "192.168.123.164",
        "robot_subnet": "192.168.123.0/24",
    },
}

SUPPORTED_UBUNTU_VERSIONS = {"20.04", "22.04", "24.04"}


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


def check_ubuntu() -> None:
    if platform.system() != "Linux":
        print_error(
            f"Unsupported OS: {platform.system()}. "
            "This script requires Ubuntu 20.04 / 22.04 / 24.04."
        )
        sys.exit(1)

    # Read /etc/os-release directly — works without lsb_release installed.
    os_info: dict[str, str] = {}
    try:
        with open("/etc/os-release", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, _, v = line.partition("=")
                    os_info[k] = v.strip('"')
    except FileNotFoundError:
        pass

    distro_id = os_info.get("ID", "").lower()
    version = os_info.get("VERSION_ID", "")

    if distro_id != "ubuntu":
        print_warning(
            f"Detected distro: '{distro_id}' (expected Ubuntu). "
            "This script is tested on Ubuntu only — proceed with caution."
        )
    elif version not in SUPPORTED_UBUNTU_VERSIONS:
        print_warning(
            f"Ubuntu {version} is not in the tested set "
            f"({', '.join(sorted(SUPPORTED_UBUNTU_VERSIONS))}). "
            "Proceeding anyway."
        )
    else:
        print_info(f"Ubuntu {version} detected.")


def check_nmcli() -> None:
    if not shutil.which("nmcli"):
        print_error(
            "nmcli not found. NetworkManager is required to configure the interface "
            "in a way that survives NM management.\n"
            "Install with: sudo apt-get install network-manager"
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
                print_info(f"Host IP to assign : {config['ip_address']}/{config['prefix_length']}")
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
            iface.startswith(p) for p in ["docker", "veth", "br-", "virbr", "vmnet", "wl"]
        ):
            continue

        state_result = run_command(["sudo", "ip", "link", "show", iface], check=False)
        state = "UNKNOWN"
        for sl in state_result.stdout.splitlines():
            if "state" in sl:
                state = sl.split("state")[1].strip().split()[0]
                break

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
# Firewall — scoped to robot subnet, not the whole interface
# ---------------------------------------------------------------------------

def configure_firewall(config: dict[str, str]) -> None:
    if not shutil.which("ufw"):
        print_warning("ufw not found — skipping firewall configuration.")
        return

    status_result = run_command(["sudo", "ufw", "status"], check=False)
    if "inactive" in status_result.stdout.lower():
        print_info("ufw is inactive — no firewall rules needed.")
        return

    robot_subnet = config["robot_subnet"]
    print_info(f"Adding scoped ufw rules for robot subnet {robot_subnet}...")

    in_result = run_command(
        ["sudo", "ufw", "allow", "from", robot_subnet], check=False
    )
    out_result = run_command(
        ["sudo", "ufw", "allow", "to", robot_subnet], check=False
    )

    if in_result.returncode == 0 and out_result.returncode == 0:
        print_info(f"ufw: allowed traffic to/from {robot_subnet}.")
    else:
        print_warning("Could not add ufw rules — robot communication may be blocked.")
        print_warning(
            f"Run manually: sudo ufw allow from {robot_subnet} && sudo ufw allow to {robot_subnet}"
        )


# ---------------------------------------------------------------------------
# Interface configuration via NetworkManager (nmcli)
# Using nmcli ensures the static IP is NM-aware and won't be reverted by the
# NetworkManager daemon, which manages wired interfaces by default on Ubuntu.
# ---------------------------------------------------------------------------

def configure_interface(config: dict[str, str], interface: str) -> None:
    conn_name = config["connection_name"]
    ip_cidr = f"{config['ip_address']}/{config['prefix_length']}"

    print_info(f"Configuring {interface} for {config['robot_type']} via NetworkManager...")

    configure_firewall(config)

    # Remove any existing profile with this connection name to avoid conflicts.
    existing = run_command(
        ["sudo", "nmcli", "connection", "show", conn_name], check=False
    )
    if existing.returncode == 0:
        print_info(f"Removing existing NM profile '{conn_name}'...")
        run_command(["sudo", "nmcli", "connection", "delete", conn_name], check=False)

    # Create a new wired connection profile with a static IP.
    add_cmd = [
        "sudo", "nmcli", "connection", "add",
        "type", "ethernet",
        "con-name", conn_name,
        "ifname", interface,
        "ipv4.method", "manual",
        "ipv4.addresses", ip_cidr,
        "ipv6.method", "disabled",
    ]
    if config["gateway"]:
        add_cmd += ["ipv4.gateway", config["gateway"]]

    result = run_command(add_cmd, check=False)
    if result.returncode != 0:
        print_error(f"Failed to create NM connection profile:\n{result.stderr}")
        sys.exit(1)

    # Bring it up.
    up_result = run_command(
        ["sudo", "nmcli", "connection", "up", conn_name], check=False
    )
    if up_result.returncode != 0:
        print_error(f"Failed to bring up connection '{conn_name}':\n{up_result.stderr}")
        sys.exit(1)

    # Verify.
    time.sleep(1)
    addr_result = run_command(["sudo", "ip", "addr", "show", interface], check=False)
    if config["ip_address"] in addr_result.stdout:
        print_info("Interface configured successfully.")
        print_info(f"  Interface : {interface}")
        print_info(f"  Host IP   : {ip_cidr}")
        if config["gateway"]:
            print_info(f"  Gateway   : {config['gateway']}")
    else:
        print_error("IP address was not assigned — check nmcli output above.")
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

    check_ubuntu()
    check_nmcli()
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

    configure_interface(config, interface)
    check_interface_state(interface)
    ping_robot(config["robot_ip_address"])

    print_info("Network setup complete.")
    print_info(
        f"Interface {interface} is configured for {config['robot_type']} communication.\n"
        f"  Connection profile '{config['connection_name']}' is managed by NetworkManager\n"
        f"  and will persist until the profile is deleted or the interface is reconfigured."
    )
    print_warning(
        "Note: this assigns a static IP for the duration of the NM connection profile. "
        "To remove it: sudo nmcli connection delete " + config["connection_name"]
    )


if __name__ == "__main__":
    main()
