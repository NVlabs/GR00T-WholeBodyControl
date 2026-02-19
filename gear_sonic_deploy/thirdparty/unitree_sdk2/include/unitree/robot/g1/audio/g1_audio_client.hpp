#ifndef __UT_ROBOT_G1_AUDIO_CLIENT_HPP__
#define __UT_ROBOT_G1_AUDIO_CLIENT_HPP__

#include <limits>
#include <unitree/robot/client/client.hpp>
#include <unitree/robot/go2/public/jsonize_type.hpp>

#include "g1_audio_api.hpp"

namespace unitree {
namespace robot {
namespace g1 {
class AudioClient : public Client {
 public:
  AudioClient() : Client(AUDIO_SERVICE_NAME, false) {}
  ~AudioClient() {}

  /*Init*/
  void Init() {
    SetApiVersion(AUDIO_API_VERSION);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_AUDIO_TTS);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_AUDIO_ASR);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_AUDIO_START_PLAY);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_AUDIO_STOP_PLAY);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_AUDIO_GET_VOLUME);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_AUDIO_SET_VOLUME);
    UT_ROBOT_CLIENT_REG_API_NO_PROI(ROBOT_API_ID_AUDIO_SET_RGB_LED);
  };

  /*API Call*/
  int32_t TtsMaker(const std::string& text, int32_t speaker_id) {
    std::string parameter, data;
    TtsMakerParameter json;

    json.index = tts_index++;
    json.text = text;
    json.speaker_id = speaker_id;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_AUDIO_TTS, parameter, data);
  }

  int32_t GetVolume(uint8_t& volume) {
    std::string parameter, data;

    int32_t ret = Call(ROBOT_API_ID_AUDIO_GET_VOLUME, parameter, data);
    if (ret == 0) {
      unitree::robot::go2::JsonizeCommObjInt json;
      json.name = "volume";
      common::FromJsonString(data, json);
      volume = json.value;
    }

    return ret;
  }

  int32_t SetVolume(uint8_t volume) {
    std::string parameter, data;
    unitree::robot::go2::JsonizeCommObjInt json;

    json.value = volume;
    json.name = "volume";
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_AUDIO_SET_VOLUME, parameter, data);
  }

  int32_t PlayStream(std::string app_name, std::string stream_id,
                     std::vector<uint8_t> pcm_data) {
    std::string parameter;
    PlayStreamParameter json;

    json.app_name = app_name;
    json.stream_id = stream_id;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_AUDIO_START_PLAY, parameter, pcm_data);
  }

  int32_t PlayStop(std::string app_name) {
    std::string parameter, data;
    PlayStopParameter json;

    json.app_name = app_name;
    parameter = common::ToJsonString(json);

    Call(ROBOT_API_ID_AUDIO_STOP_PLAY, parameter, data);
    return 0;
  }

  int32_t LedControl(uint8_t R, uint8_t G, uint8_t B) {
    std::string parameter, data;
    LedControlParameter json;

    json.R = R;
    json.G = G;
    json.B = B;
    parameter = common::ToJsonString(json);

    return Call(ROBOT_API_ID_AUDIO_SET_RGB_LED, parameter, data);
  }

 private:
  uint32_t tts_index = 0;
};
}  // namespace g1

}  // namespace robot
}  // namespace unitree
#endif  // __UT_ROBOT_G1_AUDIO_CLIENT_HPP__
