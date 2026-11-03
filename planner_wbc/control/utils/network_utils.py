#!/usr/bin/env python3
"""
Simple network interface utilities
"""

import platform
import re
import subprocess


def get_network_interfaces():
    """Get network interfaces with their IP addresses"""
    try:
        result = subprocess.run(
            ["/sbin/ip", "addr", "show"], capture_output=True, text=True, check=True
        )
        return _parse_ip_output(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            result = subprocess.run(["ifconfig"], capture_output=True, text=True, check=True)
            return _parse_ifconfig_output(result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {}


def _parse_ip_output(output):
    """Parse 'ip addr' command output"""
    interfaces = {}
    current_interface = None

    for line in output.split("\n"):
        interface_match = re.match(r"^\d+:\s+(\w+):", line)
        if interface_match:
            current_interface = interface_match.group(1)
            interfaces[current_interface] = []

        ip_match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", line)
        if ip_match and current_interface:
            interfaces[current_interface].append(ip_match.group(1))

    return interfaces


def _parse_ifconfig_output(output):
    """Parse 'ifconfig' command output"""
    interfaces = {}
    current_interface = None

    for line in output.split("\n"):
        interface_match = re.match(r"^(\w+):", line)
        if interface_match:
            current_interface = interface_match.group(1)
            interfaces[current_interface] = []

        ip_match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", line)
        if ip_match and current_interface:
            interfaces[current_interface].append(ip_match.group(1))

    return interfaces


def find_interface_by_ip(target_ip):
    """Find interface name for given IP address"""
    interfaces = get_network_interfaces()
    for interface, ip_list in interfaces.items():
        if target_ip in ip_list:
            return interface
    return None


def resolve_interface(interface: str) -> tuple[str, str]:
    """
    Resolve interface parameter to actual network interface name and environment type

    Args:
        interface: "sim", "real", or direct interface name or IP address

    Returns:
        tuple: (interface_name, env_type) where env_type is "sim" or "real"
    """
    # Check if interface is an IP address
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", interface):
        if interface == "127.0.0.1":
            return interface, "sim"
        else:
            return interface, "real"

    if interface == "sim":
        lo_interface = find_interface_by_ip("127.0.0.1")
        if lo_interface:
            # macOS uses lo0 instead of lo
            if platform.system() == "Darwin" and lo_interface == "lo":
                return "lo0", "sim"
            return lo_interface, "sim"
        return ("lo0" if platform.system() == "Darwin" else "lo"), "sim"

    elif interface == "real":
        interfaces = get_network_interfaces()
        for iface, ip_list in interfaces.items():
            for ip in ip_list:
                if ip.startswith("192.168.123."):
                    return iface, "real"
        return interface, "real"  # fallback

    else:
        # Direct interface name - check if it has 127.0.0.1 to determine env_type
        interfaces = get_network_interfaces()
        if interface in interfaces:
            for ip in interfaces[interface]:
                if ip == "127.0.0.1":
                    return interface, "sim"

        # macOS lo interface handling
        if platform.system() == "Darwin" and interface == "lo":
            return "lo0", "sim"

        # Default to real for unknown interfaces
        return interface, "real"


if __name__ == "__main__":
    interfaces = get_network_interfaces()

    if not interfaces:
        print("No network interfaces found")
        exit(1)

    # Show all interfaces
    print("Network interfaces:")
    for interface, ip_list in interfaces.items():
        print(f"  {interface}: {', '.join(ip_list)}")

    # Test resolve_interface function
    print("\nTesting resolve_interface:")
    for test_interface in ["sim", "real", "lo", "127.0.0.1"]:
        interface_name, env_type = resolve_interface(test_interface)
        print(f"  {test_interface} -> {interface_name} ({env_type})")
