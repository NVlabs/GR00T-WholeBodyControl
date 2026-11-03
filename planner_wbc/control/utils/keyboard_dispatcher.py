import os
import subprocess
import sys
import threading

import rclpy
from sshkeyboard import listen_keyboard, stop_listening
from std_msgs.msg import String as RosStringMsg

from decoupled_wbc.control.main.constants import KEYBOARD_INPUT_TOPIC

# Global variable to store original terminal attributes
_original_terminal_attrs = None


def save_terminal_state():
    """Save the current terminal state."""
    global _original_terminal_attrs
    try:
        import termios

        fd = sys.stdin.fileno()
        _original_terminal_attrs = termios.tcgetattr(fd)
    except (ImportError, OSError, termios.error):
        _original_terminal_attrs = None


def restore_terminal():
    """Restore terminal to original state."""
    global _original_terminal_attrs
    try:
        import termios

        if _original_terminal_attrs is not None:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSANOW, _original_terminal_attrs)
            return
    except (ImportError, OSError, termios.error):
        pass

    # Fallback for non-Unix systems or if termios fails
    try:
        if os.name == "posix":
            os.system("stty sane")
    except OSError:
        pass


class ROSKeyboardDispatcher:
    """ROS-based keyboard dispatcher that receives keyboard events via ROS topics."""

    def __init__(self):
        self.listeners = []
        self._active = False
        assert rclpy.ok(), "Expected ROS2 to be initialized in this process..."
        executor = rclpy.get_global_executor()
        self.node = executor.get_nodes()[0]
        print("creating keyboard input subscriber...")
        self.subscription = self.node.create_subscription(
            RosStringMsg, KEYBOARD_INPUT_TOPIC, self._callback, 10
        )

    def register(self, listener):
        if not hasattr(listener, "handle_keyboard_button"):
            raise NotImplementedError("handle_keyboard_button is not implemented")
        self.listeners.append(listener)

    def start(self):
        """Start the ROS keyboard dispatcher."""
        self._active = True
        print("ROS keyboard dispatcher started")

    def stop(self):
        """Stop the ROS keyboard dispatcher and cleanup."""
        if self._active:
            self._active = False
            # Clean up subscription
            if hasattr(self, "subscription"):
                self.node.destroy_subscription(self.subscription)
            print("ROS keyboard dispatcher stopped")

    def _callback(self, msg: RosStringMsg):
        if self._active:
            for listener in self.listeners:
                listener.handle_keyboard_button(msg.data)

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


class KeyboardDispatcher:
    def __init__(self):
        self.listeners = []
        self._listening_thread = None
        self._stop_event = threading.Event()
        self._key = None

    def register(self, listener):
        # raise if handle_keyboard_button is not implemented
        # TODO(YL): let listener be a Callable instead of a class
        if not hasattr(listener, "handle_keyboard_button"):
            raise NotImplementedError("handle_keyboard_button is not implemented")
        self.listeners.append(listener)

    def handle_key(self, key):
        # Check if we should stop
        if self._stop_event.is_set():
            stop_listening()
            return

        for listener in self.listeners:
            listener.handle_keyboard_button(key)

    def start_listening(self):
        try:
            save_terminal_state()  # Save original terminal state before listening
            listen_keyboard(
                on_press=self.handle_key,
                delay_second_char=0.1,
                delay_other_chars=0.05,
                sleep=0.01,
            )
        except Exception as e:
            print(f"Keyboard listener stopped: {e}")
        finally:
            # Ensure terminal is restored even if an exception occurs
            self._restore_terminal()

    def start(self):
        self._listening_thread = threading.Thread(target=self.start_listening, daemon=True)
        self._listening_thread.start()

    def stop(self):
        """Stop the keyboard listener and restore terminal settings."""
        if self._listening_thread and self._listening_thread.is_alive():
            self._stop_event.set()
            # Force stop_listening to be called
            try:
                stop_listening()
            except Exception:
                pass
            # Wait a bit for the thread to finish
            self._listening_thread.join(timeout=0.5)
            # Restore terminal settings
            self._restore_terminal()

    def _restore_terminal(self):
        """Restore terminal to a sane state."""
        restore_terminal()

    def __del__(self):
        """Cleanup when object is destroyed."""
        self.stop()


KEYBOARD_LISTENER_TOPIC_NAME = "/Gr00tKeyboardListener"


class KeyboardListener:
    def __init__(self):
        self.key = None

    def handle_keyboard_button(self, key):
        self.key = key

    def pop_key(self):
        key = self.key
        self.key = None
        return key


class KeyboardListenerPublisher:
    def __init__(self, topic_name: str = KEYBOARD_LISTENER_TOPIC_NAME):
        """
        Initialize keyboard listener for remote teleop with simplified interface.

        Args:
            remote_system: RemoteSystem instance
            control_channel_name: Name of the control channel
        """
        assert rclpy.ok(), "Expected ROS2 to be initialized in this process..."
        executor = rclpy.get_global_executor()
        self.node = executor.get_nodes()[0]
        self.publisher = self.node.create_publisher(RosStringMsg, topic_name, 1)

    def handle_keyboard_button(self, key):
        self.publisher.publish(RosStringMsg(data=key))


class KeyboardListenerSubscriber:
    def __init__(
        self,
        topic_name: str = KEYBOARD_LISTENER_TOPIC_NAME,
        node_name: str = "keyboard_listener_subscriber",
    ):
        assert rclpy.ok(), "Expected ROS2 to be initialized in this process..."
        executor = rclpy.get_global_executor()
        nodes = executor.get_nodes()
        if nodes:
            self.node = nodes[0]
            self._create_node = False
        else:
            self.node = rclpy.create_node("KeyboardListenerSubscriber")
            executor.add_node(self.node)
            self._create_node = True
        self.subscriber = self.node.create_subscription(RosStringMsg, topic_name, self._callback, 1)
        self._data = None

    def _callback(self, msg: RosStringMsg):
        self._data = msg.data

    def read_msg(self):
        data = self._data
        self._data = None
        return data


class KeyboardEStop:
    def __init__(self):
        """Initialize KeyboardEStop with automatic tmux cleanup detection."""
        # Automatically create tmux cleanup if in deployment mode
        self.cleanup_callback = self._create_tmux_cleanup_callback()

    def _create_tmux_cleanup_callback(self):
        """Create a cleanup callback that kills the tmux session if running in deployment mode."""
        tmux_session = os.environ.get("DECOUPLED_WBC_TMUX_SESSION")

        def cleanup_callback():
            if tmux_session:
                print(f"Emergency stop: Killing tmux session '{tmux_session}'...")
                try:
                    subprocess.run(["tmux", "kill-session", "-t", tmux_session], timeout=5)
                    print("Tmux session terminated successfully.")
                except subprocess.TimeoutExpired:
                    print("Warning: Tmux session termination timed out, forcing kill...")
                    try:
                        subprocess.run(["tmux", "kill-session", "-t", tmux_session, "-9"])
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Warning: Error during tmux cleanup: {e}")
                    # If tmux cleanup fails, fallback to immediate exit
                    restore_terminal()
                    os._exit(1)
            else:
                print("Emergency stop: No tmux session, exiting normally...")
                sys.exit(1)

        return cleanup_callback

    def handle_keyboard_button(self, key):
        if key == "`":
            print("Emergency stop triggered - running cleanup...")
            self.cleanup_callback()
