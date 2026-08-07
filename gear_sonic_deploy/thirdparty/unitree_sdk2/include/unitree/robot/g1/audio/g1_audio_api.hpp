#ifndef __UT_ROBOT_G1_AUDIO_API_HPP__
#define __UT_ROBOT_G1_AUDIO_API_HPP__

#include <unitree/common/json/jsonize.hpp>
// #include <variant>

namespace unitree {
namespace robot {
namespace g1 {
/*service name*/
const std::string AUDIO_SERVICE_NAME = "voice";

/*api version*/
const std::string AUDIO_API_VERSION = "1.0.0.0";

/*api id*/
const int32_t ROBOT_API_ID_AUDIO_TTS = 1001;
const int32_t ROBOT_API_ID_AUDIO_ASR = 1002;
const int32_t ROBOT_API_ID_AUDIO_START_PLAY = 1003;
const int32_t ROBOT_API_ID_AUDIO_STOP_PLAY = 1004;
const int32_t ROBOT_API_ID_AUDIO_GET_VOLUME = 1005;
const int32_t ROBOT_API_ID_AUDIO_SET_VOLUME = 1006;
const int32_t ROBOT_API_ID_AUDIO_SET_RGB_LED = 1010;

class TtsMakerParameter : public common::Jsonize {
 public:
  TtsMakerParameter() {}
  ~TtsMakerParameter() {}

  void fromJson(common::JsonMap &json) {}

  void toJson(common::JsonMap &json) const {
    common::ToJson(index, json["index"]);
    common::ToJson(speaker_id, json["speaker_id"]);
    common::ToJson(text, json["text"]);
  }

  int32_t index = 0;
  uint16_t speaker_id = 0;
  std::string text;
};

class PlayStreamParameter : public common::Jsonize {
 public:
  PlayStreamParameter() {}
  ~PlayStreamParameter() {}

  void fromJson(common::JsonMap &json) {}

  void toJson(common::JsonMap &json) const {
    common::ToJson(app_name, json["app_name"]);
    common::ToJson(stream_id, json["stream_id"]);
  }

  std::string app_name;
  std::string stream_id;
};

class PlayStopParameter : public common::Jsonize {
 public:
  PlayStopParameter() {}
  ~PlayStopParameter() {}

  void fromJson(common::JsonMap &json) {}

  void toJson(common::JsonMap &json) const {
    common::ToJson(app_name, json["app_name"]);
  }

  std::string app_name;
};

class LedControlParameter : public common::Jsonize {
 public:
  LedControlParameter() {}
  ~LedControlParameter() {}

  void fromJson(common::JsonMap &json) {}

  void toJson(common::JsonMap &json) const {
    common::ToJson(R, json["R"]);
    common::ToJson(G, json["G"]);
    common::ToJson(B, json["B"]);
  }

  uint8_t R;
  uint8_t G;
  uint8_t B;
};
}  // namespace g1
}  // namespace robot
}  // namespace unitree

#endif  // __UT_ROBOT_G1_AUDIO_API_HPP__
