#!/usr/bin/env python3

import select
import struct
import sys

import cv2
import numpy as np
import zmq


def main():
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://192.168.123.164:5555")
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)

    print("Receiving video from tcp://192.168.123.164:5555")
    print("Type q and press Enter, or press q in an image window, to quit.")

    try:
        while True:
            terminal_ready, _, _ = select.select(
                [sys.stdin], [], [], 0
            )
            if terminal_ready:
                command = sys.stdin.readline().strip().lower()
                if command == "q":
                    break

            events = dict(poller.poll(30))
            if socket not in events:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            message = socket.recv_multipart()

            if len(message) != 3:
                print(
                    f"Expected metadata, raw color, and raw depth, "
                    f"received {len(message)} message part(s)."
                )
                continue

            frame_metadata, color_bytes, depth_bytes = message
            (
                color_height,
                color_width,
                color_channels,
                depth_height,
                depth_width,
            ) = struct.unpack(
                "!IIIII", frame_metadata
            )
            color_frame = np.frombuffer(
                color_bytes, dtype=np.uint8
            ).reshape(
                color_height, color_width, color_channels
            )
            depth_raw = np.frombuffer(
                depth_bytes, dtype=np.uint16
            ).reshape(depth_height, depth_width)

            cv2.imshow("Camera Color", color_frame)
            cv2.imshow("Camera Raw Depth", depth_raw)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        socket.close()
        context.term()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
