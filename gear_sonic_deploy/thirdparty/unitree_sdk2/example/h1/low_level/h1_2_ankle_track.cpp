#include <cmath>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

using namespace unitree::common;
using namespace unitree::robot;

const int H1_NUM_MOTOR = 27;

enum PRorAB { PR = 0, AB = 1 };

inline uint32_t Crc32Core(uint32_t *ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t CRC32 = 0xFFFFFFFF;
  const uint32_t dwPolynomial = 0x04c11db7;
  for (uint32_t i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else
        CRC32 <<= 1;
      if (data & xbit) CRC32 ^= dwPolynomial;

      xbit >>= 1;
    }
  }
  return CRC32;
};

class H1Example {
 private:
  double time_;
  double control_dt_;  // [2ms]
  double duration_;    // [3 s]
  PRorAB mode_;

  unitree_hg::msg::dds_::LowState_ low_state_;

  ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;
  ThreadPtr command_writer_ptr_, control_thread_ptr_;

 public:
  H1Example(std::string networkInterface)
      : time_(0.0), control_dt_(0.002), duration_(3.0), mode_(PR) {
    ChannelFactory::Instance()->Init(0, networkInterface);

    // create publisher
    lowcmd_publisher_.reset(
        new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();

    // create subscriber
    lowstate_subscriber_.reset(
        new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
            HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(
        std::bind(&H1Example::LowStateHandler, this, std::placeholders::_1), 1);

    // create threads
    control_thread_ptr_ = CreateRecurrentThreadEx(
        "control", UT_CPU_ID_NONE, 2000, &H1Example::Control, this);
  }

  void LowStateHandler(const void *message) {
    low_state_ = *(const unitree_hg::msg::dds_::LowState_ *)message;

    if (low_state_.crc() !=
        Crc32Core((uint32_t *)&low_state_,
                  (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1)) {
      std::cout << "low_state CRC Error" << std::endl;
      return;
    }
  }

  void Control() {
    unitree_hg::msg::dds_::LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = mode_;
    dds_low_command.mode_machine() = low_state_.mode_machine();
    for (int i = 0; i < H1_NUM_MOTOR; ++i) {
      dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
      dds_low_command.motor_cmd().at(i).tau() = 0.0;
      dds_low_command.motor_cmd().at(i).q() = 0.0;
      dds_low_command.motor_cmd().at(i).dq() = 0.0;
      dds_low_command.motor_cmd().at(i).kp() = (i < 13) ? 100.0 : 50.0;
      dds_low_command.motor_cmd().at(i).kd() = 1.0;
    }

    time_ += control_dt_;
    if (time_ < duration_) {
      // [Stage 1]: set robot to zero posture
      for (int i = 0; i < H1_NUM_MOTOR; ++i) {
        double ratio = std::clamp(time_ / duration_, 0.0, 1.0);
        dds_low_command.motor_cmd().at(i).q() =
            (1. - ratio) * low_state_.motor_state().at(i).q();
      }
    } else {
      // [Stage 2]: swing ankle's PR
      mode_ = PR;  // Enable PR mode

      // generate sin/cos trajectory
      double max_P = 0.25;  // [rad]
      double max_R = 0.25;  // [rad]
      double t = time_ - duration_;
      double L_P_des = max_P * std::cos(2.0 * M_PI * t);
      double L_R_des = max_R * std::sin(2.0 * M_PI * t);
      double R_P_des = max_P * std::cos(2.0 * M_PI * t);
      double R_R_des = -max_R * std::sin(2.0 * M_PI * t);

      // update ankle joint position targets
      float Kp_Pitch = 80;
      float Kd_Pitch = 1;
      float Kp_Roll = 80;
      float Kd_Roll = 1;
      dds_low_command.motor_cmd().at(4).q() = L_P_des;  // 4: LeftAnklePitch
      dds_low_command.motor_cmd().at(4).dq() = 0;
      dds_low_command.motor_cmd().at(4).kp() = Kp_Pitch;
      dds_low_command.motor_cmd().at(4).kd() = Kd_Pitch;
      dds_low_command.motor_cmd().at(4).tau() = 0;
      dds_low_command.motor_cmd().at(5).q() = L_R_des;  // 5: LeftAnkleRoll
      dds_low_command.motor_cmd().at(5).dq() = 0;
      dds_low_command.motor_cmd().at(5).kp() = Kp_Roll;
      dds_low_command.motor_cmd().at(5).kd() = Kd_Roll;
      dds_low_command.motor_cmd().at(5).tau() = 0;
      dds_low_command.motor_cmd().at(10).q() = R_P_des;  // 10: RightAnklePitch
      dds_low_command.motor_cmd().at(10).dq() = 0;
      dds_low_command.motor_cmd().at(10).kp() = Kp_Pitch;
      dds_low_command.motor_cmd().at(10).kd() = Kd_Pitch;
      dds_low_command.motor_cmd().at(10).tau() = 0;
      dds_low_command.motor_cmd().at(11).q() = R_R_des;  // 11: RightAnkleRoll
      dds_low_command.motor_cmd().at(11).dq() = 0;
      dds_low_command.motor_cmd().at(11).kp() = Kp_Roll;
      dds_low_command.motor_cmd().at(11).kd() = Kd_Roll;
      dds_low_command.motor_cmd().at(11).tau() = 0;

      float L_P_m = low_state_.motor_state().at(4).q();
      float L_R_m = low_state_.motor_state().at(5).q();
      float R_P_m = low_state_.motor_state().at(10).q();
      float R_R_m = low_state_.motor_state().at(11).q();
      // clang-format off
        printf("%f,%f,%f,%f,%f,%f,%f,%f\n",
               L_P_des, L_P_m, L_R_des, L_R_m,
               R_P_des, R_P_m, R_R_des, R_R_m);
      // clang-format on
    }

    dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command,
                                      (sizeof(dds_low_command) >> 2) - 1);
    lowcmd_publisher_->Write(dds_low_command);
  }
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: h1_2_ankle_track network_interface_name" << std::endl;
    exit(0);
  }
  std::string networkInterface = argv[1];
  H1Example custom(networkInterface);
  while (true) sleep(10);
  return 0;
}
