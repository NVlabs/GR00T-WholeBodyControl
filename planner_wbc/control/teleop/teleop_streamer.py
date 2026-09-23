from math import floor
import pickle
from typing import Optional

from decoupled_wbc.control.robot_model.robot_model import RobotModel
from decoupled_wbc.control.teleop.pre_processor.fingers.fingers import FingersPreProcessor
from decoupled_wbc.control.teleop.pre_processor.wrists.wrists import WristsPreProcessor
from decoupled_wbc.control.teleop.streamers.base_streamer import StreamerOutput


class TeleopStreamer:
    def __init__(
        self,
        robot_model: RobotModel,
        body_control_device: Optional[str] = None,
        hand_control_device: Optional[str] = None,
        enable_real_device=True,
        body_streamer_ip="",
        body_streamer_keyword="",
        replay_data_path: Optional[str] = None,
        replay_speed: float = 1.0,
    ):
        # initialize the body
        self.body = robot_model

        self.body_control_device = body_control_device
        self.hand_control_device = hand_control_device
        self.body_streamer_ip = body_streamer_ip
        self.body_streamer_keyword = body_streamer_keyword
        self.replay_speed = replay_speed

        # enable real robot and devices
        self.enable_real_device = enable_real_device
        if self.enable_real_device:
            if body_control_device == "vive":
                from decoupled_wbc.control.teleop.streamers.vive_streamer import ViveStreamer

                self.body_streamer = ViveStreamer(
                    ip=self.body_streamer_ip, keyword=self.body_streamer_keyword
                )
                self.body_streamer.start_streaming()
            elif body_control_device == "iphone":
                from decoupled_wbc.control.teleop.streamers.iphone_streamer import IphoneStreamer

                self.body_streamer = IphoneStreamer()
                self.body_streamer.start_streaming()
            elif body_control_device == "leapmotion":
                from decoupled_wbc.control.teleop.streamers.leapmotion_streamer import (
                    LeapMotionStreamer,
                )

                self.body_streamer = LeapMotionStreamer()
                self.body_streamer.start_streaming()
            elif body_control_device == "joycon":
                from decoupled_wbc.control.teleop.streamers.joycon_streamer import JoyconStreamer

                self.body_streamer = JoyconStreamer()
                self.body_streamer.start_streaming()

            elif body_control_device == "pico":
                from decoupled_wbc.control.teleop.streamers.pico_streamer import PicoStreamer

                self.body_streamer = PicoStreamer()
                self.body_streamer.start_streaming()
            elif body_control_device == "dummy":
                from decoupled_wbc.control.teleop.streamers.dummy_streamer import DummyStreamer

                self.body_streamer = DummyStreamer()
                self.body_streamer.start_streaming()
            else:
                self.body_streamer = None

            if hand_control_device and hand_control_device != body_control_device:
                if hand_control_device == "manus":
                    from decoupled_wbc.control.teleop.streamers.manus_streamer import ManusStreamer

                    self.hand_streamer = ManusStreamer()
                    self.hand_streamer.start_streaming()
                elif hand_control_device == "joycon":
                    from decoupled_wbc.control.teleop.streamers.joycon_streamer import JoyconStreamer

                    self.hand_streamer = JoyconStreamer()
                    self.hand_streamer.start_streaming()
                elif hand_control_device == "iphone":
                    from decoupled_wbc.control.teleop.streamers.iphone_streamer import IphoneStreamer

                    self.hand_streamer = IphoneStreamer()
                    self.hand_streamer.start_streaming()
                elif hand_control_device == "pico":
                    from decoupled_wbc.control.teleop.streamers.pico_streamer import PicoStreamer

                    self.hand_streamer = PicoStreamer()
                    self.hand_streamer.start_streaming()
                else:
                    self.hand_streamer = None
            else:
                self.hand_streamer = None
        else:
            self.body_streamer = None
            self.hand_streamer = None

        self.raw_replay_data = None
        self.replay_calibration_data = None
        self.replay_mode = False
        if replay_data_path and not self.enable_real_device:
            with open(replay_data_path, "rb") as f:
                data_ = pickle.load(f)
            self.raw_replay_data = data_["replay_data"]
            self.replay_calibration_data = data_["calibration_data"]
            print("Found teleop replay data in file: ", replay_data_path)
            self.replay_idx = 0
            self.replay_mode = True

        # initialize pre_processors
        self.body_control_device = body_control_device
        if body_control_device or self.replay_mode:
            self.body_pre_processor = WristsPreProcessor(
                motion_scale=robot_model.supplemental_info.teleop_upper_body_motion_scale
            )
            self.body_pre_processor.register(self.body)
        else:
            self.body_pre_processor = None

        # initialize hand pre-processors and post-processors
        self.hand_control_device = hand_control_device
        if hand_control_device or self.replay_mode:
            self.left_hand_pre_processor = FingersPreProcessor(side="left")
            self.right_hand_pre_processor = FingersPreProcessor(side="right")

        else:
            self.left_hand_pre_processor = None
            self.right_hand_pre_processor = None

        self.is_calibrated = False

    def _get_replay_data(self) -> StreamerOutput:
        streamer_data = StreamerOutput()

        if self.replay_idx < len(self.raw_replay_data):
            streamer_data.ik_data.update(
                self.raw_replay_data[floor(self.replay_idx / self.replay_speed)]
            )
            self.replay_idx += 1

        return streamer_data

    def _get_live_data(self) -> StreamerOutput:
        """Get structured data instead of raw dict"""
        if self.body_streamer:
            streamer_data = self.body_streamer.get()
        else:
            streamer_data = StreamerOutput()

        if self.hand_streamer and self.hand_streamer != self.body_streamer:
            hand_data = self.hand_streamer.get()

            # Merge hand data into body data (hand data takes precedence)
            streamer_data.ik_data.update(hand_data.ik_data)
            streamer_data.control_data.update(hand_data.control_data)
            streamer_data.teleop_data.update(hand_data.teleop_data)
            streamer_data.data_collection_data.update(hand_data.data_collection_data)

        return streamer_data

    def get_streamer_data(self) -> StreamerOutput:
        if self.enable_real_device:
            streamer_data = self._get_live_data()
        elif self.replay_mode:
            streamer_data = self._get_replay_data()
        else:
            streamer_data = StreamerOutput()

        if self.is_calibrated and streamer_data.ik_data:
            body_data, left_hand_data, right_hand_data = self.pre_process(streamer_data.ik_data)
            streamer_data.ik_data = {
                "body_data": body_data,
                "left_hand_data": left_hand_data,
                "right_hand_data": right_hand_data,
            }
        elif not self.is_calibrated:
            streamer_data.ik_data = {}

        return streamer_data

    def calibrate(self):
        """Calibrate the pre-processors using only IK data."""
        if self.replay_mode:
            ik_data = self.replay_calibration_data
        else:
            streamer_data = self._get_live_data()
            ik_data = streamer_data.ik_data

        if self.body_pre_processor:
            self.body_pre_processor.calibrate(ik_data, self.body_control_device)
        if self.left_hand_pre_processor:
            self.left_hand_pre_processor.calibrate(ik_data, self.hand_control_device)
        if self.right_hand_pre_processor:
            self.right_hand_pre_processor.calibrate(ik_data, self.hand_control_device)

        self.is_calibrated = True

    def pre_process(self, raw_data):
        """Pre-process the raw data."""
        assert (
            self.body_pre_processor or self.left_hand_pre_processor or self.right_hand_pre_processor
        ), "Pre-processors are not initialized."

        # Check if finger data is present in raw_data
        has_finger_data = "left_fingers" in raw_data and "right_fingers" in raw_data

        if self.body_pre_processor:
            body_data = self.body_pre_processor(raw_data)
            # Only process hand data if finger keys are present and preprocessors are available
            if has_finger_data and self.left_hand_pre_processor and self.right_hand_pre_processor:
                left_hand_data = self.left_hand_pre_processor(raw_data)
                right_hand_data = self.right_hand_pre_processor(raw_data)
                return body_data, left_hand_data, right_hand_data
            else:
                return body_data, None, None
        else:  # only hands
            if has_finger_data and self.left_hand_pre_processor and self.right_hand_pre_processor:
                left_hand_data = self.left_hand_pre_processor(raw_data)
                right_hand_data = self.right_hand_pre_processor(raw_data)
                return None, left_hand_data, right_hand_data
            else:
                # No finger data available, return None for hand data
                return None, None, None

    def reset(self):
        if self.body_streamer is not None:
            self.body_streamer.reset_status()
        if self.hand_streamer is not None:
            self.hand_streamer.reset_status()

    def stop_streaming(self):
        if self.body_streamer:
            self.body_streamer.stop_streaming()
        # Only stop hand_streamer if it's a different instance than body_streamer
        if self.hand_streamer and self.hand_streamer is not self.body_streamer:
            self.hand_streamer.stop_streaming()
