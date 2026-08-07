import argparse
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np
import pytest
import rclpy
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String as RosStringMsg

from decoupled_wbc.control.main.constants import (
    CONTROL_GOAL_TOPIC,
    KEYBOARD_INPUT_TOPIC,
    STATE_TOPIC_NAME,
)
from decoupled_wbc.control.utils.ros_utils import ROSMsgPublisher, ROSMsgSubscriber
from decoupled_wbc.control.utils.term_color_constants import GREEN_BOLD, RESET, YELLOW_BOLD
from decoupled_wbc.data.viz.rerun_viz import RerunViz


class KeyboardPublisher:
    def __init__(self, topic_name: str = KEYBOARD_INPUT_TOPIC):
        assert rclpy.ok(), "Expected ROS2 to be initialized in this process..."
        executor = rclpy.get_global_executor()
        self.node = executor.get_nodes()[0]
        self.publisher = self.node.create_publisher(RosStringMsg, topic_name, 1)

    def publish(self, key: str):
        msg = RosStringMsg()
        msg.data = key
        self.publisher.publish(msg)


def is_robot_fallen_from_quat(mujoco_quat):
    # Convert MuJoCo [w, x, y, z] → SciPy [x, y, z, w]
    w, x, y, z = mujoco_quat
    scipy_quat = [x, y, z, w]

    r = R.from_quat(scipy_quat)
    roll, pitch, _ = r.as_euler("xyz", degrees=False)

    MAX_ROLL_PITCH = np.radians(60)
    print(f"[Fall Check] roll={roll:.3f} rad, pitch={pitch:.3f} rad")
    return abs(roll) > MAX_ROLL_PITCH or abs(pitch) > MAX_ROLL_PITCH


class LocomotionRunner:
    def __init__(self, test_mode: str = "squat"):
        self.test_mode = test_mode
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node(f"EvalDriver_{test_mode}_{int(time.time())}")

        # gracefully shutdown the spin thread when the test is done
        self._stop_event = threading.Event()

        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=False)
        self.spin_thread.start()

        self.keyboard_event_publisher = KeyboardPublisher(KEYBOARD_INPUT_TOPIC)
        self.control_publisher = ROSMsgPublisher(CONTROL_GOAL_TOPIC)
        self.state_subscriber = ROSMsgSubscriber(STATE_TOPIC_NAME)
        print(f"{test_mode} test initialized...")

    def _spin_loop(self):
        try:
            while rclpy.ok() and not self._stop_event.is_set():
                rclpy.spin_once(self.node)
        except rclpy.executors.ExternalShutdownException:
            print("[INFO] Spin thread exiting due to shutdown.")
        finally:
            print("spin loop stopped...")

    def warm_up(self):
        """Stabilize and release the robot."""
        print("waiting for 2 seconds...")
        time.sleep(2)
        print(f"running {self.test_mode} test...")
        self.activate()
        print("activated...")
        time.sleep(1)
        self.release()
        print("released...")
        time.sleep(5)

    def _run_walk_test(self):
        self.walk_forward()  # speed up to 0.2 m/s
        time.sleep(1)
        self.walk_forward()  # speed up to 0.4 m/s

        rate = self.node.create_rate(0.5)
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < 10.0:
            obs = self.state_subscriber.get_msg()

            if is_robot_fallen_from_quat(obs["torso_quat"]):
                print("robot fallen...")
                return 0
            elif self._check_success_condition(obs):
                print(f"robot reaching target ({self.test_mode})...")
                return 1, {}
            else:
                rate.sleep()

        print("test timed out after 10 seconds...")
        return 0, {}

    def _run_squat_test(self):
        rate = self.node.create_rate(0.5)
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < 10.0:
            obs = self.state_subscriber.get_msg()

            if is_robot_fallen_from_quat(obs["torso_quat"]):
                print("robot fallen...")
                return 0, {}
            elif self._check_success_condition(obs):
                print(f"robot reaching target ({self.test_mode})...")
                return 1, {}
            else:
                self.go_down()
                rate.sleep()

        print("test timed out after 10 seconds...")
        return 0, {}

    def cmd_to_velocity(self, cmd_list):
        cmd_to_velocity = {
            "w": np.array([0.2, 0.0, 0.0]),
            "s": np.array([-0.2, 0.0, 0.0]),
            "q": np.array([0.0, 0.2, 0.0]),
            "e": np.array([0.0, -0.2, 0.0]),
            "z": np.array([0.0, 0.0, 0.0]),
        }

        accumulated_velocity = np.array([0.0, 0.0, 0.0])
        velocity_list = []
        for cmd in cmd_list:
            if cmd == "z":
                accumulated_velocity = [0.0, 0.0, 0.0]
            elif cmd in ["CHECK", "SKIP"]:
                accumulated_velocity = velocity_list[-1]
            else:
                accumulated_velocity += cmd_to_velocity[cmd]
            velocity_list.append(accumulated_velocity.copy())

        return velocity_list

    def _run_stop_test(self):
        base_vel_thres = 0.25

        cmd_list = (
            ["w", "w", "w", "w", "s", "s", "s", "z", "SKIP", "CHECK"]
            + ["s", "s", "q", "w", "w", "w", "e", "s", "s", "z", "SKIP", "CHECK"]
            + ["q", "q", "w", "q", "e", "s", "s", "e", "w", "z", "SKIP", "CHECK"]
            + ["w", "w", "w", "w", "w", "s", "s", "s", "s", "z", "SKIP", "CHECK"]
        )

        success_flag = 1

        statistics = {
            "floating_base_pose": {"state": []},
            "floating_base_vel": {"state": [], "cmd": []},
            "timestamp": [],
        }
        for cmd in cmd_list:
            self.keyboard_event_publisher.publish(cmd)
            time.sleep(0.5)
            obs = self.state_subscriber.get_msg()
            statistics["floating_base_pose"]["state"].append(
                np.linalg.norm(obs["floating_base_pose"])
            )
            statistics["floating_base_vel"]["state"].append(
                np.linalg.norm(obs["floating_base_vel"])
            )
            statistics["timestamp"].append(time.time())

            if cmd == "CHECK" and np.linalg.norm(obs["floating_base_vel"]) > base_vel_thres:
                print(
                    f" [{YELLOW_BOLD}WARNING{RESET}] robot is not stopped fully. "
                    f"Current base velocity: {np.linalg.norm(obs['floating_base_vel']):.3f} > {base_vel_thres:.3f}"
                )
                # success_flag = 0  # robot is not stopped

            time.sleep(0.5)

        vel_cmd = self.cmd_to_velocity(cmd_list)
        vel_cmd = [np.linalg.norm(v) for v in vel_cmd]
        statistics["floating_base_vel"]["cmd"] = vel_cmd
        return success_flag, statistics

    def _run_eef_track_test(self):
        from decoupled_wbc.control.policy.lerobot_replay_policy import LerobotReplayPolicy

        parquet_path = (
            Path(__file__).parent.parent.parent.parent / "replay_data" / "g1_pnpbottle.parquet"
        )
        replay_policy = LerobotReplayPolicy(parquet_path=str(parquet_path))

        freq = 50
        rate = self.node.create_rate(freq)

        statistics = {
            # "floating_base_pose": {"state": [], "cmd": []},
            "eef_base_pose": {"state": [], "cmd": []},
            "timestamp": [],
        }

        for ii in range(500):
            action = replay_policy.get_action()
            action = replay_policy.action_to_cmd(action)
            action["timestamp"] = time.monotonic()
            action["target_time"] = time.monotonic() + ii / freq
            self.control_publisher.publish(action)
            obs = self.state_subscriber.get_msg()
            if obs is None:
                print("no obs...")
                continue
            gt_obs = replay_policy.get_observation()

            # statistics["floating_base_pose"]["state"].append(obs["floating_base_pose"])
            # statistics["floating_base_pose"]["cmd"].append(np.zeros_like(obs["floating_base_pose"]))
            statistics["eef_base_pose"]["state"].append(obs["wrist_pose"])
            statistics["eef_base_pose"]["cmd"].append(gt_obs["wrist_pose"])
            statistics["timestamp"].append(time.time())

            pos_err = np.linalg.norm(obs["wrist_pose"][:3] - gt_obs["wrist_pose"][:3])
            if pos_err > 1e-1:
                print(
                    f" [{YELLOW_BOLD}WARNING{RESET}] robot failed to track the eef, "
                    f"error: {pos_err:.3f} ({self.test_mode})..."
                )
                return 0, statistics

            if is_robot_fallen_from_quat(obs["torso_quat"]):
                print("robot fallen...")
                return 0, statistics
            else:
                rate.sleep()

        return 1, statistics

    def run(self):
        self.warm_up()

        test_mode_to_func = {
            "squat": self._run_squat_test,
            "walk": self._run_walk_test,
            "stop": self._run_stop_test,
            "eef_track": self._run_eef_track_test,
        }

        result, statistics = test_mode_to_func[self.test_mode]()

        self.post_process(statistics)
        return result

    def _check_success_condition(self, obs):
        if self.test_mode == "squat":
            return obs["floating_base_pose"][2] < 0.4
        elif self.test_mode == "walk":
            return np.linalg.norm(obs["floating_base_pose"][0:2]) > 1.0
        return False

    def activate(self):
        self.keyboard_event_publisher.publish("]")

    def release(self):
        self.keyboard_event_publisher.publish("9")

    def go_down(self):
        self.keyboard_event_publisher.publish("2")

    def walk_forward(self):
        self.keyboard_event_publisher.publish("w")

    def walk_stop(self):
        self.keyboard_event_publisher.publish("z")

    def post_process(self, statistics):
        if len(statistics) == 0:
            return

        # plot the statistics
        plot_keys = [key for key in statistics.keys() if key != "timestamp"]
        viz = RerunViz(
            image_keys=[],
            tensor_keys=plot_keys,
            window_size=10.0,
            app_name=f"{self.test_mode}_test",
        )

        for ii in range(len(statistics[plot_keys[0]]["state"])):
            tensor_data = {}
            for k in plot_keys:
                if "state" in statistics[k] and "cmd" in statistics[k]:
                    tensor_data[k] = np.array(
                        (statistics[k]["state"][ii], statistics[k]["cmd"][ii])
                    ).reshape(2, -1)
                else:
                    tensor_data[k] = np.asarray(statistics[k]["state"][ii]).reshape(1, -1)
            viz.plot_tensors(
                tensor_data,
                statistics["timestamp"][ii],
            )

        if self.test_mode == "stop":
            base_velocity = statistics["floating_base_vel"]["state"]
            base_velocity_cmd = statistics["floating_base_vel"]["cmd"]

            base_velocity_tracking_err = []
            for v_cmd, v in zip(base_velocity_cmd, base_velocity):  # TODO: check if this is correct
                if v_cmd.max() < 1e-4:
                    base_velocity_tracking_err.append(v)
            print(
                f" [{GREEN_BOLD}INFO{RESET}] Base velocity tracking when stopped: "
                f"{np.mean(base_velocity_tracking_err):.3f}"
            )

        if self.test_mode == "eef_track":
            eef_pose = statistics["eef_base_pose"]["state"]
            eef_pose_cmd = statistics["eef_base_pose"]["cmd"]
            eef_pose_tracking_err = []
            for p_cmd, p in zip(eef_pose_cmd, eef_pose):
                eef_pose_tracking_err.append(np.linalg.norm(p - p_cmd))
            print(
                f" [{GREEN_BOLD}INFO{RESET}] Eef pose tracking error: {np.mean(eef_pose_tracking_err):.3f}"
            )

    def shutdown(self):
        self._stop_event.set()
        self.spin_thread.join()
        del self.state_subscriber
        del self.keyboard_event_publisher
        # Don't shutdown ROS between tests - let pytest handle it


def start_g1_control_loop():
    proc = subprocess.Popen(
        [
            "python3",
            "decoupled_wbc/control/main/teleop/run_g1_control_loop.py",
            "--keyboard_dispatcher_type",
            "ros",
            "--enable-offscreen",
        ],
        preexec_fn=os.setsid,
    )
    time.sleep(10)
    return proc


def run_test(test_mode: str):
    """Run a single test with the specified mode."""
    proc = start_g1_control_loop()
    print(f"G1 control loop started for {test_mode} test...")

    test = LocomotionRunner(test_mode)
    result = test.run()

    print("Shutting down...")
    test.shutdown()
    proc.send_signal(signal.SIGKILL)
    proc.wait()

    return result


def test_squat():
    """Pytest function for squat test."""
    result = run_test("squat")
    assert result == 1, "Squat test failed - robot either fell or didn't reach target height"


def test_walk():
    """Pytest function for walk test."""
    result = run_test("walk")
    assert result == 1, "Walk test failed - robot either fell or didn't reach target distance"


@pytest.mark.skip(reason="skipping test for now, cicd test always gets killed")
def test_stop():
    """Pytest function for walking to a nearby position and stop test."""
    result = run_test("stop")
    assert result == 1, "Stop test failed - robot either fell or didn't reach target distance"


@pytest.mark.skip(reason="skipping test for now, cicd test always gets killed")
def test_eef_track():
    """Pytest function for eef track test."""
    result = run_test("eef_track")
    assert result == 1, "Eef track test failed - robot either fell or didn't reach target distance"


def main():
    parser = argparse.ArgumentParser(description="Run locomotion tests")
    parser.add_argument("--squat", action="store_true", help="Run squat test only")
    parser.add_argument("--walk", action="store_true", help="Run walk test only")
    parser.add_argument("--stop", action="store_true", help="Run stop test only")
    parser.add_argument("--eef_track", action="store_true", help="Run eef track test only")

    args = parser.parse_args()

    if args.squat and args.walk:
        print("Error: Cannot specify both --squat and --walk")
        return 1

    if args.squat:
        print("Running squat test only...")
        result = run_test("squat")
        if result == 1:
            print("✓ Squat test PASSED")
            return 0
        else:
            print("✗ Squat test FAILED")
            return 1

    elif args.walk:
        print("Running walk test only...")
        result = run_test("walk")
        if result == 1:
            print("✓ Walk test PASSED")
            return 0
        else:
            print("✗ Walk test FAILED")
            return 1

    elif args.stop:
        print("Running stop test only...")
        result = run_test("stop")
        if result == 1:
            print("✓ Stop test PASSED")
            return 0
        else:
            print("✗ Stop test FAILED")
            return 1

    elif args.eef_track:
        print("Running eef track test only...")
        result = run_test("eef_track")
        if result == 1:
            print("✓ Eef track test PASSED")
            return 0
        else:
            print("✗ Eef track test FAILED")
            return 1

    else:
        print("Running both tests...")
        squat_result = run_test("squat")
        walk_result = run_test("walk")

        if squat_result == 1 and walk_result == 1:
            print("✓ All tests PASSED")
            return 0
        else:
            print(
                f"✗ Test results: squat={'PASSED' if squat_result == 1 else 'FAILED'}, "
                f"walk={'PASSED' if walk_result == 1 else 'FAILED'}"
            )
            return 1


if __name__ == "__main__":
    exit(main())
