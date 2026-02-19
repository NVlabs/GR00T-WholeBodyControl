// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/robot/channel/channel_publisher.hpp>
#include <atomic>
#include <thread>
#include <memory>

namespace unitree
{
namespace robot
{

/**
 * @brief Base class for convenient publishing to topics.
 * This class provides an easy-to-use interface for publishing messages to topics.
 *
 * @tparam MessageType The type of message being published.
 */
template <typename MessageType>
class PublisherBase : public unitree::robot::ChannelPublisher<MessageType>
{
public:
  using MsgType = MessageType;
  using SharedPtr = std::shared_ptr<PublisherBase<MsgType>>;

  PublisherBase(std::string TOPIC_NAME)
  : unitree::robot::ChannelPublisher<MessageType>(TOPIC_NAME)
  {
    this->InitChannel();
  }
};

// For details: see https://github.com/ros-controls/realtime_tools
template <typename MessageType>
class RealTimePublisher
{
public:
  using MsgType = MessageType;
  using PublisherSharedPtr = typename unitree::robot::ChannelPublisherPtr<MessageType>;

  MessageType msg_{};

  explicit RealTimePublisher(PublisherSharedPtr publisher)
  : publisher_(publisher), is_running_(false), keep_running_(true), turn_(LOOP_NOT_STARTED)
  {
    thread_ = std::thread(&RealTimePublisher::publishingLoop, this);
  }

  explicit RealTimePublisher(std::string topic)
  : RealTimePublisher(std::make_shared<PublisherBase<MsgType>>(topic))
  {}

  ~RealTimePublisher()
  {
    stop();
    while (is_running()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if(thread_.joinable()) { thread_.join(); }
  }

  void stop()
  {
    keep_running_ = false;
  }

  /**
   * @brief Try to get the data lock from realtime
   * 
   * To publish data from the realtime loop, you need to run trylock to
   * attenot to get unique access to the msg_ variable. Teylock returns
   * true if the lock was aquired, and false otherwise.
   */
  bool trylock()
  {
    if(mutex_.try_lock())
    {
      if(turn_ == REALTIME) {
        return true;
      } else {
        mutex_.unlock();
        return false;
      }
    } else {
      return false;
    }
  }

  /**
   * @brief Unlock the msg_ variable and publish it.
   */
  void unlockAndPublish() 
  {
    turn_ = NON_REALTIME;
    mutex_.unlock();
  }

  /**
   * @brief Get the data lock from non-realtime.
   * 
   * To publish data from the realtime loop, you need to run trylock to
   * attenot to get unique access to the msg_ variable. Teylock returns
   * true if the lock was aquired, and false otherwise.
   */
  void lock()
  {
    // never actually lock on the lock
    while (!mutex_.try_lock()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  /**
   * @brief Unlocks the data without publishing anything.
   */
  void unlock() { mutex_.unlock(); }

protected:
  virtual void pre_communication() {}  // something before sending the message
  virtual void post_communication() {} // something after sending the message

private:
  // non-copyable
  RealTimePublisher(const RealTimePublisher&) = delete;
  RealTimePublisher& operator=(const RealTimePublisher&) = delete;


  bool is_running() const { return is_running_; }

  void publishingLoop()
  {
    is_running_ = true;
    turn_ = REALTIME;

    while (keep_running_)
    {
      MsgType outgoing;

      // Locks msg_ and copies it
      lock();
      while (turn_ != NON_REALTIME && keep_running_)
      {
        unlock();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        lock();
      }
      pre_communication();
      outgoing = msg_;
      turn_ = REALTIME;

      unlock();

      if(keep_running_) { 
        publisher_->Write(outgoing, 0); 
      }
      post_communication();
    }
    is_running_ = false;
  }

  PublisherSharedPtr publisher_;
  std::atomic_bool is_running_;
  std::atomic_bool keep_running_;

  std::mutex mutex_;

  std::thread thread_;

  enum { REALTIME, NON_REALTIME, LOOP_NOT_STARTED };
  std::atomic<int> turn_;
};

} // namespace robot
} // namespace unitree