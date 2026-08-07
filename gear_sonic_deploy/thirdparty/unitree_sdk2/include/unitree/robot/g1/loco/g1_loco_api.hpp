#ifndef __UT_ROBOT_G1_LOCO_API_HPP__
#define __UT_ROBOT_G1_LOCO_API_HPP__

#include <unitree/common/json/jsonize.hpp>
#include <variant>

namespace unitree {
namespace robot {
namespace g1 {
/*service name*/
const std::string LOCO_SERVICE_NAME = "sport";

/*api version*/
const std::string LOCO_API_VERSION = "1.0.0.0";

/*api id*/
const int32_t ROBOT_API_ID_LOCO_GET_FSM_ID = 7001;
const int32_t ROBOT_API_ID_LOCO_GET_FSM_MODE = 7002;
const int32_t ROBOT_API_ID_LOCO_GET_BALANCE_MODE = 7003;
const int32_t ROBOT_API_ID_LOCO_GET_SWING_HEIGHT = 7004;
const int32_t ROBOT_API_ID_LOCO_GET_STAND_HEIGHT = 7005;
const int32_t ROBOT_API_ID_LOCO_GET_PHASE = 7006; // deprecated

const int32_t ROBOT_API_ID_LOCO_SET_FSM_ID = 7101;
const int32_t ROBOT_API_ID_LOCO_SET_BALANCE_MODE = 7102;
const int32_t ROBOT_API_ID_LOCO_SET_SWING_HEIGHT = 7103;
const int32_t ROBOT_API_ID_LOCO_SET_STAND_HEIGHT = 7104;
const int32_t ROBOT_API_ID_LOCO_SET_VELOCITY = 7105;
const int32_t ROBOT_API_ID_LOCO_SET_ARM_TASK = 7106;
const int32_t ROBOT_API_ID_LOCO_SET_SPEED_MODE = 7107;

using LocoCmd =
    std::map<std::string, std::variant<int, float, std::vector<float>>>;

class JsonizeDataVecFloat : public common::Jsonize {
public:
  JsonizeDataVecFloat() {}
  ~JsonizeDataVecFloat() {}

  void fromJson(common::JsonMap &json) { common::FromJson(json["data"], data); }

  void toJson(common::JsonMap &json) const {
    common::ToJson(data, json["data"]);
  }

  std::vector<float> data;
};

class JsonizeVelocityCommand : public common::Jsonize {
public:
  JsonizeVelocityCommand() {}
  ~JsonizeVelocityCommand() {}

  void fromJson(common::JsonMap &json) {
    common::FromJson(json["velocity"], velocity);
    common::FromJson(json["duration"], duration);
  }

  void toJson(common::JsonMap &json) const {
    common::ToJson(velocity, json["velocity"]);
    common::ToJson(duration, json["duration"]);
  }

  std::vector<float> velocity;
  float duration;
};

} // namespace g1
} // namespace robot
} // namespace unitree

#endif // __UT_ROBOT_G1_LOCO_API_HPP__
