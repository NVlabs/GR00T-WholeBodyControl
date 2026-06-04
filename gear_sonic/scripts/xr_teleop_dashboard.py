from __future__ import annotations

import argparse
import threading

from gear_sonic.utils.teleop.xr_dashboard import (
    DashboardState,
    run_bridge_receiver,
    run_camera_receiver,
    run_xr_receiver,
    serve_dashboard,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only XR + GEAR-SONIC teleop dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--xr-source-host", default="localhost")
    parser.add_argument("--xr-source-port", type=int, default=5560)
    parser.add_argument("--xr-source-topic", default="xr_teleop")
    parser.add_argument("--bridge-host", default="localhost")
    parser.add_argument("--bridge-port", type=int, default=5556)
    parser.add_argument("--camera-host", default="localhost")
    parser.add_argument("--camera-port", type=int, default=5555)
    parser.add_argument("--no-camera", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    state = DashboardState()
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=run_xr_receiver,
            args=(
                state,
                args.xr_source_host,
                args.xr_source_port,
                args.xr_source_topic,
                stop_event,
            ),
            daemon=True,
        ),
        threading.Thread(
            target=run_bridge_receiver,
            args=(state, args.bridge_host, args.bridge_port, stop_event),
            daemon=True,
        ),
    ]
    if not args.no_camera:
        threads.append(
            threading.Thread(
                target=run_camera_receiver,
                args=(state, args.camera_host, args.camera_port, stop_event),
                daemon=True,
            )
        )
    for thread in threads:
        thread.start()
    server = serve_dashboard(state, args.host, args.port)
    print(f"XR teleop dashboard: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
