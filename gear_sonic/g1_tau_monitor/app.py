"""Application loop for the G1 upper-body torque monitor."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2

from .config import MonitorConfig, parse_args
from .dds import G1TorqueSubscriber
from .history import TorqueHistory
from .renderer import render


def run(config: MonitorConfig) -> None:
    subscriber = G1TorqueSubscriber(
        domain_id=config.dds_domain_id,
        network_interface=config.network_interface,
        state_topic=config.state_topic,
        command_topic=config.command_topic,
        include_pd=config.include_pd,
        show_q=config.show_q,
        stale_seconds=config.stale_seconds,
    )
    history = TorqueHistory(config.window_seconds)
    paused = False
    delay_ms = max(1, int(round(1000.0 / config.refresh_hz)))

    try:
        cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.window_name, config.width, config.height)
        print("[G1TauMonitor] Q/Esc: quit | Space: pause | C: clear")
        while True:
            sample = subscriber.read()
            if sample is not None and not paused:
                history.append(sample)
            frame = render(
                history,
                subscriber.status(),
                width=config.width,
                height=config.height,
                stale_seconds=config.stale_seconds,
                include_pd=config.include_pd,
                show_q=config.show_q,
                y_limit=config.y_limit,
                paused=paused,
            )
            cv2.imshow(config.window_name, frame)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("c"):
                history.clear()
            try:
                if cv2.getWindowProperty(config.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close()
        cv2.destroyAllWindows()
        print("[G1TauMonitor] stopped")


def main(argv: Optional[Sequence[str]] = None) -> None:
    run(parse_args(argv))
