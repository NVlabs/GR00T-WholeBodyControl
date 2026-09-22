"""Robot-side fixed-rate DDS-to-ZMQ publisher."""

from __future__ import annotations

import argparse
import time

import zmq

from .dds import G1DDSReader
from .protocol import TOPIC, encode_sample
from .sampler import TelemetrySampler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Publish Unitree G1 upper-body LowState/LowCmd telemetry over ZMQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--publish-hz", type=float, default=50.0)
    parser.add_argument("--bind-host", default="*")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--dds-domain-id", type=int, default=0)
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--state-topic", default="rt/lowstate")
    parser.add_argument("--command-topic", default="rt/lowcmd")
    parser.add_argument("--stale-seconds", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.publish_hz <= 0 or args.stale_seconds <= 0:
        parser.error("publish-hz and stale-seconds must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in range 1..65535")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    reader = G1DDSReader(
        domain_id=args.dds_domain_id,
        network_interface=args.network_interface,
        state_topic=args.state_topic,
        command_topic=args.command_topic,
    )
    sampler = TelemetrySampler(args.publish_hz)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.SNDHWM, max(100, int(args.publish_hz * 4)))
    socket.setsockopt(zmq.LINGER, 0)
    endpoint = f"tcp://{args.bind_host}:{args.port}"
    socket.bind(endpoint)
    print(f"[G1Telemetry] publishing {args.publish_hz:g} Hz on {endpoint}")

    period_ns = int(1e9 / args.publish_hz)
    deadline_ns = time.monotonic_ns()
    last_wait_log = 0.0
    try:
        while True:
            now_ns = time.monotonic_ns()
            if now_ns < deadline_ns:
                time.sleep((deadline_ns - now_ns) * 1e-9)
                now_ns = time.monotonic_ns()
            deadline_ns += period_ns
            if deadline_ns <= now_ns:
                deadline_ns = now_ns + period_ns

            snapshot = reader.snapshot()
            if snapshot is None:
                now = time.monotonic()
                if now - last_wait_log >= 1.0:
                    print(f"[G1Telemetry] waiting for DDS: {reader.status()}")
                    last_wait_log = now
                continue
            state_age = (now_ns - snapshot["state_time_ns"]) * 1e-9
            command_age = (now_ns - snapshot["command_time_ns"]) * 1e-9
            if state_age > args.stale_seconds or command_age > args.stale_seconds:
                now = time.monotonic()
                if now - last_wait_log >= 1.0:
                    print(
                        "[G1Telemetry] waiting for fresh DDS: "
                        f"state={state_age * 1000:.1f}ms cmd={command_age * 1000:.1f}ms"
                    )
                    last_wait_log = now
                continue
            sample = sampler.build(snapshot, now_ns)
            socket.send_multipart([TOPIC, encode_sample(sample)])
    except KeyboardInterrupt:
        print("[G1Telemetry] stopped")
    finally:
        reader.close()
        socket.close()
        context.term()


if __name__ == "__main__":
    main()
