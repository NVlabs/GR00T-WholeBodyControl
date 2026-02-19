// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/robot/channel/channel_subscriber.hpp>
#include <mutex>
#include <thread>
#include <spdlog/spdlog.h>

namespace unitree
{
namespace robot
{
/**
 * @brief Base class for convenient subscription to topics.
 * This class provides an easy-to-use interface for subscribing to topics, 
 * but the continuous background updates may increase CPU usage.
 *
 * @tparam MessageType The type of message being subscribed to.
 */
template <typename MessageType>
class SubscriptionBase
{
public:
  using MsgType = MessageType;
  using SharedPtr = std::shared_ptr<SubscriptionBase<MsgType>>;

  SubscriptionBase(const std::string& topic, const std::function<void(const void*)>& handler = nullptr)
  {
    last_update_time_ = std::chrono::steady_clock::now() - std::chrono::milliseconds(timeout_ms_);
    sub_ = std::make_shared<unitree::robot::ChannelSubscriber<MessageType>>(topic);
    if (handler) {
      sub_->InitChannel(handler);
    } else {
      sub_->InitChannel([this](const void *msg){
        last_update_time_ = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mutex_);
        pre_communication();
        msg_ = *(const MessageType*)msg;
        post_communication();
      });
    }
  }

  void set_timeout_ms(uint32_t timeout_ms) { timeout_ms_ = timeout_ms; }

  bool isTimeout() {
    auto now = std::chrono::steady_clock::now();
    auto elasped_time = now - last_update_time_;
    return elasped_time > std::chrono::milliseconds(timeout_ms_);
  }

  void wait_for_connection() {
    auto t0 = std::chrono::steady_clock::now();
    bool warn_info = false;
    while(isTimeout()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (!warn_info && std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
        warn_info = true;
        spdlog::warn("Waiting for connection {}", sub_->GetChannelName());
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // wait for stable communicaiton
    if (warn_info) {
      spdlog::info("Connected {}", sub_->GetChannelName());
    }
  }

  MessageType msg_;
  std::mutex mutex_;

protected:
  virtual void pre_communication() {}  // something before receiving message
  virtual void post_communication() {} // something after receiving message

  uint32_t timeout_ms_{1000};
  unitree::robot::ChannelSubscriberPtr<MessageType> sub_;
  std::chrono::steady_clock::time_point last_update_time_;
};


}; // namespace robot
}; // namespace unitree