#pragma once

#include <memory>
#include <stop_token>
#include <thread>
#include <unitree/robot/g1/audio/g1_audio_client.hpp>

struct AudioCommand {
  bool streaming_data_absent = false;
};

class AudioThread {
 public:
  AudioThread();

  void SetCommand(const AudioCommand& command);

 private:
  void loop(std::stop_token st);

  unitree::robot::g1::AudioClient client_;
  std::jthread thread_;

  std::mutex command_mutex_;
  AudioCommand command_;
  AudioCommand command_last_;
};
