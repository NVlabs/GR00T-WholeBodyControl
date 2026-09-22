"""CLI configuration for the G1 torque monitor."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class MonitorConfig:
    dds_domain_id: int = 0
    network_interface: Optional[str] = None
    state_topic: str = "rt/lowstate"
    command_topic: str = "rt/lowcmd"
    include_pd: bool = False
    show_q: bool = False
    window_seconds: float = 15.0
    refresh_hz: float = 30.0
    stale_seconds: float = 0.5
    y_limit: float = 0.0
    width: int = 1800
    height: int = 720
    window_name: str = "Unitree G1 upper-body torque residuals"


def _optional_string(value: str) -> Optional[str]:
    return None if value.strip().lower() in {"", "auto", "none"} else value


def parse_args(argv: Optional[Sequence[str]] = None) -> MonitorConfig:
    defaults = MonitorConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Plot Unitree G1 upper-body tau_est - tau_cmd from DDS LowState/LowCmd."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dds-domain-id", type=int, default=defaults.dds_domain_id)
    parser.add_argument(
        "--network-interface",
        type=_optional_string,
        default=defaults.network_interface,
        help="DDS interface; use 'auto' or 'none' for SDK auto-selection",
    )
    parser.add_argument("--state-topic", default=defaults.state_topic)
    parser.add_argument("--command-topic", default=defaults.command_topic)
    parser.add_argument(
        "--include-pd",
        action="store_true",
        default=defaults.include_pd,
        help="Include position/velocity PD terms in tau_cmd",
    )
    parser.add_argument(
        "--show-q",
        action="store_true",
        default=defaults.show_q,
        help="Add q and q_cmd plots below the torque residual plots",
    )
    parser.add_argument(
        "--window-seconds", type=float, default=defaults.window_seconds
    )
    parser.add_argument("--refresh-hz", type=float, default=defaults.refresh_hz)
    parser.add_argument(
        "--stale-seconds", type=float, default=defaults.stale_seconds
    )
    parser.add_argument(
        "--y-limit",
        type=float,
        default=defaults.y_limit,
        help="Fixed symmetric Y limit in Nm; 0 enables per-panel autoscaling",
    )
    parser.add_argument("--width", type=int, default=defaults.width)
    parser.add_argument("--height", type=int, default=defaults.height)
    parser.add_argument("--window-name", default=defaults.window_name)
    args = parser.parse_args(argv)

    config = MonitorConfig(
        dds_domain_id=args.dds_domain_id,
        network_interface=args.network_interface,
        state_topic=args.state_topic,
        command_topic=args.command_topic,
        include_pd=args.include_pd,
        show_q=args.show_q,
        window_seconds=args.window_seconds,
        refresh_hz=args.refresh_hz,
        stale_seconds=args.stale_seconds,
        y_limit=args.y_limit,
        width=args.width,
        height=args.height,
        window_name=args.window_name,
    )
    try:
        validate_config(config)
    except ValueError as exc:
        parser.error(str(exc))
    return config


def validate_config(config: MonitorConfig) -> None:
    if config.dds_domain_id < 0:
        raise ValueError("dds_domain_id must be non-negative")
    if not config.state_topic or not config.command_topic:
        raise ValueError("DDS topics must not be empty")
    if config.window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if config.refresh_hz <= 0:
        raise ValueError("refresh_hz must be positive")
    if config.stale_seconds <= 0:
        raise ValueError("stale_seconds must be positive")
    if config.y_limit < 0:
        raise ValueError("y_limit cannot be negative")
    if config.width < 900 or config.height < 400:
        raise ValueError("window dimensions must be at least 900x400")
