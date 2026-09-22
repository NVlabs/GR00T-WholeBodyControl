"""Small host-side diagnostic receiver."""

import argparse
import time

from .receiver import TelemetryReceiver


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--host", default="192.168.50.132")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    receiver = TelemetryReceiver(args.host, args.port)
    try:
        receiver.wait_until_ready(args.timeout)
        last_sequence = None
        last_time = time.monotonic()
        while True:
            receiver.poll()
            sample = receiver.latest(1.0)
            now = time.monotonic()
            if sample is not None and now - last_time >= 1.0:
                delta = 0 if last_sequence is None else sample["sequence_id"] - last_sequence
                print(
                    f"seq={sample['sequence_id']} received={delta}/s "
                    f"dds_age=({sample['state_age_sec'] * 1000:.1f}, "
                    f"{sample['command_age_sec'] * 1000:.1f})ms"
                )
                last_sequence = sample["sequence_id"]
                last_time = now
            receiver.drain_raw()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
