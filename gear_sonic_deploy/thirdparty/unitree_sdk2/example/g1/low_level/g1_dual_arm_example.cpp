#include <yaml-cpp/yaml.h>

#include <cmath>
#include <memory>
#include <mutex>
#include <shared_mutex>

// DDS
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

// IDL
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

#include <unitree/robot/b2/motion_switcher/motion_switcher_client.hpp>
using namespace unitree::robot::b2;

static const std::string HG_CMD_TOPIC = "rt/lowcmd";
static const std::string HG_STATE_TOPIC = "rt/lowstate";

using namespace unitree::common;
using namespace unitree::robot;

const int G1_NUM_MOTOR = 29;

template <typename T>
class DataBuffer {
 public:
  void SetData(const T &newData) {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = std::make_shared<T>(newData);
  }

  std::shared_ptr<const T> GetData() {
    std::shared_lock<std::shared_mutex> lock(mutex);
    return data ? data : nullptr;
  }

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(mutex);
    data = nullptr;
  }

 private:
  std::shared_ptr<T> data;
  std::shared_mutex mutex;
};

struct ImuState {
  std::array<float, 3> rpy = {};
  std::array<float, 3> omega = {};
};

struct MotorCommand {
  std::array<float, G1_NUM_MOTOR> q_target = {};
  std::array<float, G1_NUM_MOTOR> dq_target = {};
  std::array<float, G1_NUM_MOTOR> kp = {};
  std::array<float, G1_NUM_MOTOR> kd = {};
  std::array<float, G1_NUM_MOTOR> tau_ff = {};
};

struct MotorState {
  std::array<float, G1_NUM_MOTOR> q = {};
  std::array<float, G1_NUM_MOTOR> dq = {};
};

enum MotorType { GearboxS = 0, GearboxM = 1, GearboxL = 2 };

std::array<MotorType, G1_NUM_MOTOR> G1MotorType{
    // clang-format off
    // legs
    GearboxM, GearboxM, GearboxM, GearboxL, GearboxS, GearboxS,
    GearboxM, GearboxM, GearboxM, GearboxL, GearboxS, GearboxS,
    // waist
    GearboxM, GearboxS, GearboxS,
    // arms
    GearboxS, GearboxS, GearboxS, GearboxS, GearboxS, GearboxS, GearboxS,
    GearboxS, GearboxS, GearboxS, GearboxS, GearboxS, GearboxS, GearboxS
    // clang-format on
};

enum PRorAB { PR = 0, AB = 1 };

enum G1JointValidIndex {
  LeftShoulderPitch = 15,
  LeftShoulderRoll = 16,
  LeftShoulderYaw = 17,
  LeftElbow = 18,
  LeftWristRoll = 19,
  LeftWristPitch = 20,
  LeftWristYaw = 21,
  RightShoulderPitch = 22,
  RightShoulderRoll = 23,
  RightShoulderYaw = 24,
  RightElbow = 25,
  RightWristRoll = 26,
  RightWristPitch = 27,
  RightWristYaw = 28
};

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

float GetMotorKp(MotorType type) {
  switch (type) {
    case GearboxS:
      return 40;
    case GearboxM:
      return 40;
    case GearboxL:
      return 100;
    default:
      return 0;
  }
}

float GetMotorKd(MotorType type) {
  switch (type) {
    case GearboxS:
      return 1;
    case GearboxM:
      return 1;
    case GearboxL:
      return 1;
    default:
      return 0;
  }
}

class G1Example {
 private:
  double time_;
  double control_dt_;  // [2ms]
  double duration_;    // [3 s]
  PRorAB mode_;
  uint8_t mode_machine_;
  std::vector<std::vector<double>> frames_data_;

  DataBuffer<MotorState> motor_state_buffer_;
  DataBuffer<MotorCommand> motor_command_buffer_;
  DataBuffer<ImuState> imu_state_buffer_;

  ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_> lowcmd_publisher_;
  ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> lowstate_subscriber_;
  ThreadPtr command_writer_ptr_, control_thread_ptr_;

  std::shared_ptr<MotionSwitcherClient> msc;

 public:
  G1Example(std::string networkInterface)
      : time_(0.0),
        control_dt_(0.002),
        duration_(3.0),
        mode_(PR),
        mode_machine_(0) {
    ChannelFactory::Instance()->Init(0, networkInterface);

    msc.reset(new MotionSwitcherClient());
    msc->SetTimeout(5.0F);
    msc->Init();

    /*Shut down  motion control-related service*/
    while(queryMotionStatus())
    {
        std::cout << "Try to deactivate the motion control-related service." << std::endl;
        int32_t ret = msc->ReleaseMode(); 
        if (ret == 0) {
            std::cout << "ReleaseMode succeeded." << std::endl;
        } else {
            std::cout << "ReleaseMode failed. Error code: " << ret << std::endl;
        }
        sleep(5);
    }

    loadBehaviorLibrary("motion");

    // create publisher
    lowcmd_publisher_.reset(
        new ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(HG_CMD_TOPIC));
    lowcmd_publisher_->InitChannel();

    // create subscriber
    lowstate_subscriber_.reset(
        new ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
            HG_STATE_TOPIC));
    lowstate_subscriber_->InitChannel(
        std::bind(&G1Example::LowStateHandler, this, std::placeholders::_1), 1);

    // create threads
    command_writer_ptr_ =
        CreateRecurrentThreadEx("command_writer", UT_CPU_ID_NONE, 2000,
                                &G1Example::LowCommandWriter, this);
    control_thread_ptr_ = CreateRecurrentThreadEx(
        "control", UT_CPU_ID_NONE, 2000, &G1Example::Control, this);
  }

  void loadBehaviorLibrary(std::string behavior_name) {
    std::string resource_dir = BLIB_DIR;
    YAML::Node motion = YAML::LoadFile(resource_dir + behavior_name + ".seq");

    std::string content = motion["components"][1]["content"].as<std::string>();
    int num_parts = motion["components"][1]["num_parts"].as<int>();
    std::cout << "BehaviorName: " << behavior_name + ".seq\n";
    std::cout << content << " with " << num_parts << "\n";

    auto frames = motion["components"][1]["frames"];

    for (const auto &frame : frames) {
      std::vector<double> frame_data;
      for (const auto &element : frame) {
        frame_data.push_back(element.as<double>());
      }
      frames_data_.push_back(frame_data);
    }

    std::cout << frames_data_.size() << " knots with " << frames_data_[0].size()
              << " DOF\n";
  }

  void ReportRPY() {
    const std::shared_ptr<const ImuState> imu_tmp_ptr =
        imu_state_buffer_.GetData();
    if (imu_tmp_ptr) {
      std::cout << "rpy: [" << imu_tmp_ptr->rpy.at(0) << ", "
                << imu_tmp_ptr->rpy.at(1) << ", " << imu_tmp_ptr->rpy.at(2)
                << "]" << std::endl;
    }
  }

  void LowStateHandler(const void *message) {
    unitree_hg::msg::dds_::LowState_ low_state =
        *(const unitree_hg::msg::dds_::LowState_ *)message;

    if (low_state.crc() !=
        Crc32Core((uint32_t *)&low_state,
                  (sizeof(unitree_hg::msg::dds_::LowState_) >> 2) - 1)) {
      std::cout << "low_state CRC Error" << std::endl;
      return;
    }

    // get motor state
    MotorState ms_tmp;
    for (int i = 0; i < G1_NUM_MOTOR; ++i) {
      ms_tmp.q.at(i) = low_state.motor_state()[i].q();
      ms_tmp.dq.at(i) = low_state.motor_state()[i].dq();
    }
    motor_state_buffer_.SetData(ms_tmp);

    // get imu state
    ImuState imu_tmp;
    imu_tmp.omega = low_state.imu_state().gyroscope();
    imu_tmp.rpy = low_state.imu_state().rpy();
    imu_state_buffer_.SetData(imu_tmp);

    // update mode machine
    if (mode_machine_ != low_state.mode_machine()) {
      if (mode_machine_ == 0)
        std::cout << "G1 type: " << unsigned(low_state.mode_machine())
                  << std::endl;
      mode_machine_ = low_state.mode_machine();
    }
  }

  void LowCommandWriter() {
    unitree_hg::msg::dds_::LowCmd_ dds_low_command;
    dds_low_command.mode_pr() = mode_;
    dds_low_command.mode_machine() = mode_machine_;

    const std::shared_ptr<const MotorCommand> mc =
        motor_command_buffer_.GetData();
    if (mc) {
      for (size_t i = 0; i < G1_NUM_MOTOR; i++) {
        dds_low_command.motor_cmd().at(i).mode() = 1;  // 1:Enable, 0:Disable
        dds_low_command.motor_cmd().at(i).tau() = mc->tau_ff.at(i);
        dds_low_command.motor_cmd().at(i).q() = mc->q_target.at(i);
        dds_low_command.motor_cmd().at(i).dq() = mc->dq_target.at(i);
        dds_low_command.motor_cmd().at(i).kp() = mc->kp.at(i);
        dds_low_command.motor_cmd().at(i).kd() = mc->kd.at(i);
      }

      dds_low_command.crc() = Crc32Core((uint32_t *)&dds_low_command,
                                        (sizeof(dds_low_command) >> 2) - 1);
      lowcmd_publisher_->Write(dds_low_command);
    }
  }

  void Control() {
    MotorCommand motor_command_tmp;
    const std::shared_ptr<const MotorState> ms = motor_state_buffer_.GetData();

    if (ms) {
      time_ += control_dt_;
      if (time_ < duration_) {
        // [Stage 1]: set robot to zero posture
        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          double ratio = std::clamp(time_ / duration_, 0.0, 1.0);

          double q_des = 0;
          motor_command_tmp.tau_ff.at(i) = 0.0;
          motor_command_tmp.q_target.at(i) =
              (q_des - ms->q.at(i)) * ratio + ms->q.at(i);
          motor_command_tmp.dq_target.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = GetMotorKp(G1MotorType[i]);
          motor_command_tmp.kd.at(i) = GetMotorKd(G1MotorType[i]);
        }
      } else {
        // [Stage 2]: tracking the offline trajectory
        size_t frame_index = (size_t)((time_ - duration_) / control_dt_);
        if (frame_index >= frames_data_.size()) {
          frame_index = frames_data_.size() - 1;
          time_ = 0.0;  // RESET
        }

        if (frame_index % 100 == 0)
          std::cout << "Frame Index: " << frame_index << std::endl;

        for (int i = 0; i < G1_NUM_MOTOR; ++i) {
          size_t index_in_frame = i - LeftShoulderPitch;
          motor_command_tmp.q_target.at(i) =
              (i >= LeftShoulderPitch)
                  ? frames_data_[frame_index][index_in_frame]
                  : 0.0;
          motor_command_tmp.dq_target.at(i) = 0.0;
          motor_command_tmp.tau_ff.at(i) = 0.0;
          motor_command_tmp.kp.at(i) = GetMotorKp(G1MotorType[i]);
          motor_command_tmp.kd.at(i) = GetMotorKd(G1MotorType[i]);
        }
      }

      motor_command_buffer_.SetData(motor_command_tmp);
    }
  }

  std::string queryServiceName(std::string form,std::string name)
  {
      if(form == "0")
      {
          if(name == "normal" ) return "sport_mode"; 
          if(name == "ai" ) return "ai_sport"; 
          if(name == "advanced" ) return "advanced_sport"; 
      }
      else
      {
          if(name == "ai-w" ) return "wheeled_sport(go2W)"; 
          if(name == "normal-w" ) return "wheeled_sport(b2W)";
      }
      return "";
  }

  int queryMotionStatus()
  {
      std::string robotForm,motionName;
      int motionStatus;
      int32_t ret = msc->CheckMode(robotForm,motionName);
      if (ret == 0) {
          std::cout << "CheckMode succeeded." << std::endl;
      } else {
          std::cout << "CheckMode failed. Error code: " << ret << std::endl;
      }
      if(motionName.empty())
      {
          std::cout << "The motion control-related service is deactivated." << std::endl;
          motionStatus = 0;
      }
      else
      {
          std::string serviceName = queryServiceName(robotForm,motionName);
          std::cout << "Service: "<< serviceName<< " is activate" << std::endl;
          motionStatus = 1;
      }
      return motionStatus;
  }
};

int main(int argc, char const *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: g1_dual_arm_example network_interface_name"
              << std::endl;
    exit(0);
  }
  std::string networkInterface = argv[1];
  G1Example custom(networkInterface);

  while (true) sleep(10);

  return 0;
}
