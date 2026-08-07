#ifndef __UT_ROBOT_G1_LOCO_CLIENT_HPP__
#define __UT_ROBOT_G1_LOCO_CLIENT_HPP__

#include "g1_loco_api.hpp"
#include <limits>
#include <unitree/robot/client/client.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>

namespace unitree {
namespace robot {
namespace g1 {
class LocoClient : public Client {
 public:
  LocoClient() : Client(LOCO_SERVICE_NAME, false) {}
  ~LocoClient() {}

  /*Init*/
  void Init() {
    SetApiVersion(LOCO_API_VERSION);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_GET_FSM_ID);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_GET_FSM_MODE);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_GET_BALANCE_MODE);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_GET_SWING_HEIGHT);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_GET_STAND_HEIGHT);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_GET_PHASE);  // deprecated

    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_SET_FSM_ID);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_SET_BALANCE_MODE);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_SET_SWING_HEIGHT);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_SET_STAND_HEIGHT);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_SET_VELOCITY);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_SET_ARM_TASK);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_LOCO_SET_SPEED_MODE);
  };

  /*Low Level API Call*/
  int32_t GetFsmId(int& fsm_id) {
    std::string parameter, data;

    int32_t ret = Call(ROBOT_API_ID_LOCO_GET_FSM_ID, parameter, data);

    if (ret == 0) {
      go2::JsonizeDataInt json;
      common::FromJsonString(data, json);
      fsm_id = json.data;
    }

    return ret;
  }

  int32_t GetFsmMode(int& fsm_mode) {
    std::string parameter, data;

    int32_t ret = Call(ROBOT_API_ID_LOCO_GET_FSM_MODE, parameter, data);

    if (ret == 0) {
      go2::JsonizeDataInt json;
      common::FromJsonString(data, json);
      fsm_mode = json.data;
    }

    return ret;
  }

  int32_t GetBalanceMode(int& balance_mode) {
    std::string parameter, data;

    int32_t ret = Call(ROBOT_API_ID_LOCO_GET_BALANCE_MODE, parameter, data);

    if (ret == 0) {
      go2::JsonizeDataInt json;
      common::FromJsonString(data, json);
      balance_mode = json.data;
    }

    return ret;
  }

  int32_t GetSwingHeight(float& swing_height) {
    std::string parameter, data;

    int32_t ret = Call(ROBOT_API_ID_LOCO_GET_SWING_HEIGHT, parameter, data);

    if (ret == 0) {
      go2::JsonizeDataFloat json;
      common::FromJsonString(data, json);
      swing_height = json.data;
    }

    return ret;
  }

  int32_t GetStandHeight(float& stand_height) {
    std::string parameter, data;

    int32_t ret = Call(ROBOT_API_ID_LOCO_GET_STAND_HEIGHT, parameter, data);

    if (ret == 0) {
      go2::JsonizeDataFloat json;
      common::FromJsonString(data, json);
      stand_height = json.data;
    }

    return ret;
  }

  int32_t GetPhase(std::vector<float>& phase) {
    std::string parameter, data;

    int32_t ret = Call(ROBOT_API_ID_LOCO_GET_PHASE, parameter, data);

    if (ret == 0) {
      JsonizeDataVecFloat json;
      common::FromJsonString(data, json);
      phase = json.data;
    }

    return ret;
  }

  int32_t SetFsmId(int fsm_id) {
    std::string parameter, data;

    go2::JsonizeDataInt json;
    json.data = fsm_id;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_LOCO_SET_FSM_ID, parameter, data);
  }

  int32_t SetBalanceMode(int balance_mode) {
    std::string parameter, data;

    go2::JsonizeDataInt json;
    json.data = balance_mode;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_LOCO_SET_BALANCE_MODE, parameter, data);
  }

  int32_t SetSwingHeight(float swing_height) {
    std::string parameter, data;

    go2::JsonizeDataFloat json;
    json.data = swing_height;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_LOCO_SET_SWING_HEIGHT, parameter, data);
  }

  int32_t SetStandHeight(float stand_height) {
    std::string parameter, data;

    go2::JsonizeDataFloat json;
    json.data = stand_height;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_LOCO_SET_STAND_HEIGHT, parameter, data);
  }

  int32_t SetVelocity(float vx, float vy, float omega, float duration = 1.f) {
    std::string parameter, data;

    JsonizeVelocityCommand json;
    std::vector<float> velocity = {vx, vy, omega};
    json.velocity = velocity;
    json.duration = duration;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_LOCO_SET_VELOCITY, parameter, data);
  }

  int32_t SetTaskId(int task_id) {
    std::string parameter, data;

    go2::JsonizeDataInt json;
    json.data = task_id;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_LOCO_SET_ARM_TASK, parameter, data);
  }

  /*High Level API Call*/
  int32_t Damp() { return SetFsmId(1); }

  int32_t Start() { return SetFsmId(500); }

  int32_t Squat() { return SetFsmId(2); }

  int32_t Sit() { return SetFsmId(3); }

  int32_t StandUp() { return SetFsmId(4); }

  int32_t ZeroTorque() { return SetFsmId(0); }

  int32_t StopMove() { return SetVelocity(0.f, 0.f, 0.f); }

  int32_t HighStand() { return SetStandHeight(std::numeric_limits<uint32_t>::max()); }

  int32_t LowStand() { return SetStandHeight(std::numeric_limits<uint32_t>::min()); }

  int32_t Move(float vx, float vy, float vyaw, bool continous_move) {
    return SetVelocity(vx, vy, vyaw, continous_move ? 864000.f : 1.f);
  }

  int32_t Move(float vx, float vy, float vyaw) { return Move(vx, vy, vyaw, continous_move_); }

  int32_t BalanceStand() { return SetBalanceMode(0); }

  int32_t ContinuousGait(bool flag) { return SetBalanceMode(flag ? 1 : 0); }

  int32_t SwitchMoveMode(bool flag) {
    continous_move_ = flag;
    return 0;
  }

  int32_t WaveHand(bool turn_flag = false) { return SetTaskId(turn_flag ? 1 : 0); }

  int32_t ShakeHand(int stage = -1) {
    switch (stage) {
      case 0:
        first_shake_hand_stage_ = false;
        return SetTaskId(2);

      case 1:
        first_shake_hand_stage_ = true;
        return SetTaskId(3);

      default:
        first_shake_hand_stage_ = !first_shake_hand_stage_;
        return SetTaskId(first_shake_hand_stage_ ? 3 : 2);
    }
  }

  int32_t SetSpeedMode(int speed_mode) {
    std::string parameter, data;

    go2::JsonizeDataInt json;
    json.data = speed_mode;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_LOCO_SET_SPEED_MODE, parameter, data);
  }

private:
  bool continous_move_ = false;
  bool first_shake_hand_stage_ = true;
};
} // namespace g1

} // namespace robot
} // namespace unitree
#endif // __UT_ROBOT_G1_LOCO_CLIENT_HPP__
