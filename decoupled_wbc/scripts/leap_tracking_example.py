"""Prints the palm position of each hand, every frame. When a device is
connected we set the tracking mode to desktop and then generate logs for
every tracking frame received. The events of creating a connection to the
server and a device being plugged in also generate logs.
"""

import time

import leap


class MyListener(leap.Listener):
    def on_connection_event(self, event):
        print("Connected")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        print(f"Frame {event.tracking_frame_id} with {len(event.hands)} hands.")
        for hand in event.hands:
            hand_type = "left" if str(hand.type) == "HandType.Left" else "right"
            print(
                f"Hand id {hand.id} is a {hand_type} hand with position ({hand.palm.position.x}, "
                f"{hand.palm.position.y}, {hand.palm.position.z})."
            )


def main():
    my_listener = MyListener()

    connection = leap.Connection()
    connection.add_listener(my_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        while running:
            time.sleep(1)


if __name__ == "__main__":
    main()
