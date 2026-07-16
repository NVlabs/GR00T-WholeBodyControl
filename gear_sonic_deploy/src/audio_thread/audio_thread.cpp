#include "audio_thread.hpp"
#include <chrono>

static const std::string PLANNER_MODE = "Planner mode";
static const std::string POSE_MODE = "Pose mode";
static const std::string WARNING_STREAMING_DATA_ABSENT = "Streaming data absent";
static const std::string WARNING_MOTOR_ERROR = "Motor error detected";
static const std::string WARNING_LOW_STATE_LATE = "ROBOT DATA LATE";

AudioThread::AudioThread():
  client_() {
  client_.Init();
  client_.SetTimeout(10.0f);
  client_.SetVolume(100);
  thread_ = std::jthread([this](std::stop_token st) { loop(st); });
}

void AudioThread::SetCommand(const AudioCommand& command) {
  std::lock_guard<std::mutex> lock(command_mutex_);
  std::string prev_tts = std::move(command_.tts_message);
  command_ = command;

  if (command_.tts_message.empty()) {
    // No new tts, keep pending
    command_.tts_message = std::move(prev_tts);
  } else if (!prev_tts.empty()) {
    // Both have tts, concatenate
    command_.tts_message = prev_tts + ". " + command_.tts_message;
  }
}

void AudioThread::loop(std::stop_token st) {
  while (!st.stop_requested()) {
    AudioCommand command;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      command = command_;
      // Clear one-shot TTS so it's only spoken once
      command_.tts_message.clear();
    }
    if (command.streaming_data_absent && !command_last_.streaming_data_absent) {
      client_.TtsMaker(WARNING_STREAMING_DATA_ABSENT, 1);
    }
    if (command.motor_error && !command_last_.motor_error) {
      client_.TtsMaker(WARNING_MOTOR_ERROR, 1);
    }
    if (!command.tts_message.empty()) {
      client_.TtsMaker(command.tts_message, 1);
    }
    if (command.high_temperature && !command.high_temperature_message.empty()) {
      auto now = std::chrono::steady_clock::now();
      if (now - last_high_temp_tts_ >= HIGH_TEMP_TTS_INTERVAL) {
        client_.TtsMaker(command.high_temperature_message, 1);
        last_high_temp_tts_ = now;
      }
    }
    if (command.low_state_late && !command_last_.low_state_late) {
      client_.TtsMaker(WARNING_LOW_STATE_LATE, 1);
    }

    command_last_ = command;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
