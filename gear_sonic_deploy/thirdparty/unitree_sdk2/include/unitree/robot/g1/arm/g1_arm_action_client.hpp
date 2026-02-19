#pragma once

#include <unitree/robot/client/client.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>
#include "g1_arm_action_api.hpp"

namespace unitree {
namespace robot {
namespace g1 {

  /**
   * @brief Arm action client
   * 
   * The arm action server provides some upper body actions.
   * The controller is based on the `rt/arm_sdk` interface.
   */
class G1ArmActionClient : public Client {
  public:
    G1ArmActionClient() : Client(ARM_ACTION_SERVICE_NAME, false) {}
    ~G1ArmActionClient() {}
  
    /*Init*/
    void Init() {
      SetApiVersion(ARM_ACTION_API_VERSION);
      UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_ARM_ACTION_EXECUTE_ACTION);
      UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_ARM_ACTION_GET_ACTION_LIST);
  }
  
  /*API Call*/
  int32_t ExecuteAction(int32_t action_id) {
    std::string parameter, data;
    JsonizeArmActionCommand json;
  
    json.action_id = action_id;
    parameter = common::ToJsonString(json);
  
    return Call(ROBOT_API_ID_ARM_ACTION_EXECUTE_ACTION, parameter, data);
  }

  int32_t GetActionList(std::string &data) {
    std::string parameter;
    return Call(ROBOT_API_ID_ARM_ACTION_GET_ACTION_LIST, parameter, data);
  }

  /*Action List*/
  std::map<std::string, int32_t> action_map = {
    {"release arm", 99},
    {"two-hand kiss", 11},
    {"left kiss", 12},
    {"right kiss", 12},
    {"hands up", 15},
    {"clap", 17},
    {"high five", 18},
    {"hug", 19},
    {"heart", 20},
    {"right heart", 21},
    {"reject", 22},
    {"right hand up", 23},
    {"x-ray", 24},
    {"face wave", 25},
    {"high wave", 26},
    {"shake hand", 27},
  };
};


} // namespace g1
} // namespace robot
} // namespace unitree 