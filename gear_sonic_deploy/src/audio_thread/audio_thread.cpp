#include "audio_thread.hpp"
#include <chrono>

static const std::string PLANNER_MODE = "Planner mode";
static const std::string POSE_MODE = "Pose mode";
static const std::string WARNING_STREAMING_DATA_ABSENT = "Streaming data absent";

AudioThread::AudioThread():
  client_() {
  client_.Init();
  client_.SetTimeout(10.0f);
  client_.SetVolume(100);
  thread_ = std::jthread([this](std::stop_token st) { loop(st); });
}

void AudioThread::SetCommand(const AudioCommand& command) {
  std::lock_guard<std::mutex> lock(command_mutex_);
  command_ = command;
}

void AudioThread::loop(std::stop_token st) {
  while (!st.stop_requested()) {
    AudioCommand command;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      command = command_;
    }
    if (command.streaming_data_absent && !command_last_.streaming_data_absent) {
      client_.TtsMaker(WARNING_STREAMING_DATA_ABSENT, 1);
    }

    command_last_ = command;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}
