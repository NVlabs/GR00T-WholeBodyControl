// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include "unitree/dds_wrapper/common/Subscription.h"
#include "unitree/dds_wrapper/common/unitree_joystick.hpp"

#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/SportModeCmd_.hpp>
#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/go2/LidarState_.hpp>
#include <unitree/idl/go2/UwbState_.hpp>
#include <unitree/idl/go2/HeightMap_.hpp>
#include <unitree/idl/ros2/Time_.hpp>
#include <unitree/idl/ros2/PointCloud2_.hpp>
#include <unitree/idl/go2/WirelessController_.hpp>
#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>


namespace unitree
{
namespace robot
{
namespace go2
{ 

namespace subscription
{

class LowState : public SubscriptionBase<unitree_go::msg::dds_::LowState_>
{
public:
  using SharedPtr = std::shared_ptr<LowState>;

  LowState(std::string topic = "rt/lowstate") : SubscriptionBase<MsgType>(topic) {}

  void update()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    // ********** Joystick ********** //
    // Check if all joystick values are zero to determine if the joystick is inactive
    if(std::all_of(msg_.wireless_remote().begin(), msg_.wireless_remote().end(), [](uint8_t i){return i == 0;}))
    {
      auto now = std::chrono::system_clock::now();
      auto elasped_time = now - last_joystick_time_;
      if(elasped_time > std::chrono::milliseconds(joystick_timeout_ms_))
      {
        isJoystickTimeout_ = true;
      }
    } else {
      last_joystick_time_ = std::chrono::system_clock::now();
      isJoystickTimeout_ = false;
    }

    // update joystick state
    unitree::common::REMOTE_DATA_RX key;
    memcpy(&key, &msg_.wireless_remote()[0], 40);
    joystick.extract(key);
  }

  bool isJoystickTimeout() const  { return isJoystickTimeout_; }

  unitree::common::UnitreeJoystick joystick;

private:
  uint32_t joystick_timeout_ms_ = 3000;
  bool isJoystickTimeout_ = false;
  std::chrono::time_point<std::chrono::system_clock> last_joystick_time_;
};

class LowCmd : public SubscriptionBase<unitree_go::msg::dds_::LowCmd_>
{
public:
  using SharedPtr = std::shared_ptr<LowCmd>;

  LowCmd(std::string topic = "rt/lowcmd") : SubscriptionBase<MsgType>(topic) {}
};

class SportModeState : public SubscriptionBase<unitree_go::msg::dds_::SportModeState_>
{
public:
  using SharedPtr = std::shared_ptr<SportModeState>;

  SportModeState(std::string topic = "rt/sportmodestate") : SubscriptionBase<MsgType>(topic) {}

  const uint32_t gaitType() const { return msg_.gait_type(); }
  
  const Eigen::Vector3f position() const {
    return Eigen::Map<const Eigen::Vector3f>(msg_.position().data());
  }
  const Eigen::Vector3f velocity() const{
    return Eigen::Map<const Eigen::Vector3f>(msg_.velocity().data());
  }
};

class MotorStates : public SubscriptionBase<unitree_go::msg::dds_::MotorStates_>
{
public:
  using SharedPtr = std::shared_ptr<unitree_go::msg::dds_::MotorStates_>;

  MotorStates(std::string topic, int num = 0) : SubscriptionBase<MsgType>(topic) 
  {
    if (num != 0) msg_.states().resize(num);
  }
};

class MotorCmds : public SubscriptionBase<unitree_go::msg::dds_::MotorCmds_>
{
public:
  using SharedPtr = std::shared_ptr<unitree_go::msg::dds_::MotorCmds_>;

  MotorCmds(std::string topic, int num = 0) : SubscriptionBase<MsgType>(topic) 
  {
    if (num != 0) msg_.cmds().resize(num);
  }
};


} // namespace subscriber
} // namespace go2
} // namespace robot
} // namespace unitree