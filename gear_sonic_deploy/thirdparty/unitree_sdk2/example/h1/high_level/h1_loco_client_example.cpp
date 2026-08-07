#include <chrono>
#include <iostream>
#include <thread>

#include <unitree/robot/h1/loco/h1_loco_api.hpp>
#include <unitree/robot/h1/loco/h1_loco_client.hpp>

std::vector<float> stringToFloatVector(const std::string &str) {
  std::vector<float> result;
  std::stringstream ss(str);
  float num;
  while (ss >> num) {
    result.push_back(num);
    // ignore any trailing whitespace
    ss.ignore();
  }
  return result;
}

int main(int argc, char const *argv[]) {
  std::map<std::string, std::string> args = {{"network_interface", "lo"}};

  std::map<std::string, std::string> values;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) == "--") {
      size_t pos = arg.find("=");
      std::string key, value;
      if (pos != std::string::npos) {
        key = arg.substr(2, pos - 2);
        value = arg.substr(pos + 1);

        if (value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.length() - 2);
        }
      } else {
        key = arg.substr(2);
        value = "";
      }
      if (args.find(key) != args.end()) {
        args[key] = value;
      } else {
        args.insert({{key, value}});
      }
    }
  }

  unitree::robot::ChannelFactory::Instance()->Init(0, args["network_interface"]);

  unitree::robot::h1::LocoClient client;

  client.Init();
  client.SetTimeout(10.f);

  for (const auto &arg_pair : args) {
    std::cout << "Processing command: [" << arg_pair.first << "] with param: [" << arg_pair.second << "] ..."
              << std::endl;
    if (arg_pair.first == "network_interface") {
      continue;
    }

    if (arg_pair.first == "get_fsm_id") {
      int fsm_id;
      client.GetFsmId(fsm_id);
      std::cout << "current fsm_id: " << fsm_id << std::endl;
    }

    if (arg_pair.first == "get_fsm_mode") {
      int fsm_mode;
      client.GetFsmMode(fsm_mode);
      std::cout << "current fsm_mode: " << fsm_mode << std::endl;
    }

    if (arg_pair.first == "get_balance_mode") {
      int balance_mode;
      client.GetBalanceMode(balance_mode);
      std::cout << "current balance_mode: " << balance_mode << std::endl;
    }

    if (arg_pair.first == "get_swing_height") {
      float swing_height;
      client.GetSwingHeight(swing_height);
      std::cout << "current swing_height: " << swing_height << std::endl;
    }

    if (arg_pair.first == "get_stand_height") {
      float stand_height;
      client.GetStandHeight(stand_height);
      std::cout << "current stand_height: " << stand_height << std::endl;
    }

    if (arg_pair.first == "get_phase") {
      std::vector<float> phase;
      client.GetPhase(phase);
      std::cout << "current phase: (";
      for (const auto &p : phase) {
        std::cout << p << ", ";
      }
      std::cout << ")" << std::endl;
    }

    if (arg_pair.first == "enable_odom") {
      float x, y, yaw;
      client.EnableOdom();
      std::cout << "Send enable odom signal" << std::endl;
    }

    if (arg_pair.first == "disable_odom") {
      float x, y, yaw;
      client.DisableOdom();
      std::cout << "Send disable odom signal" << std::endl;
    }

    if (arg_pair.first == "get_odom") {
      float x, y, yaw;
      client.GetOdom(x, y, yaw);
      std::cout << "Get Odom: (" << x << ", " << y << ", " << yaw << ")" << std::endl;
    }

    if (arg_pair.first == "set_fsm_id") {
      int fsm_id = std::stoi(arg_pair.second);
      client.SetFsmId(fsm_id);
      std::cout << "set fsm_id to " << fsm_id << std::endl;
    }

    if (arg_pair.first == "set_balance_mode") {
      int balance_mode = std::stoi(arg_pair.second);
      client.SetBalanceMode(balance_mode);
      std::cout << "set balance_mode to " << balance_mode << std::endl;
    }

    if (arg_pair.first == "set_swing_height") {
      float swing_height = std::stof(arg_pair.second);
      client.SetSwingHeight(swing_height);
      std::cout << "set swing_height to " << swing_height << std::endl;
    }

    if (arg_pair.first == "set_stand_height") {
      float stand_height = std::stof(arg_pair.second);
      client.SetStandHeight(stand_height);
      std::cout << "set stand_height to " << stand_height << std::endl;
    }

    if (arg_pair.first == "set_velocity") {
      std::vector<float> param = stringToFloatVector(arg_pair.second);
      auto param_size = param.size();
      float vx, vy, omega, duration;
      if (param_size == 3) {
        vx = param.at(0);
        vy = param.at(1);
        omega = param.at(2);
        duration = 1.f;
      } else if (param_size == 4) {
        vx = param.at(0);
        vy = param.at(1);
        omega = param.at(2);
        duration = param.at(3);
      } else {
        std::cerr << "Invalid param size for method SetVelocity: " << param_size << std::endl;
        return 1;
      }

      client.SetVelocity(vx, vy, omega, duration);
      std::cout << "set velocity to " << arg_pair.second << std::endl;
    }

    if (arg_pair.first == "set_phase") {
      std::vector<float> param = stringToFloatVector(arg_pair.second);
      auto param_size = param.size();
      if (param_size == 2) {
        client.SetPhase(param);
        std::cout << "set phase to " << arg_pair.second << std::endl;
      } else {
        std::cerr << "Invalid param size for method SetPhase: " << param_size << std::endl;
        return 1;
      }
    }

    if (arg_pair.first == "set_target_pos") {
      std::vector<float> param = stringToFloatVector(arg_pair.second);
      client.SetTargetPos(param.at(0), param.at(1), param.at(2), false);
      std::cout << "set_target_pos: " << arg_pair.second << std::endl;
    }

    if (arg_pair.first == "set_target_pos_relative") {
      std::vector<float> param = stringToFloatVector(arg_pair.second);
      client.SetTargetPos(param.at(0), param.at(1), param.at(2));
      std::cout << "set_target_pos: " << arg_pair.second << std::endl;
    }

    if (arg_pair.first == "damp") {
      client.Damp();
    }

    if (arg_pair.first == "start") {
      client.Start();
    }

    if (arg_pair.first == "stand_up") {
      client.StandUp();
    }

    if (arg_pair.first == "zero_torque") {
      client.ZeroTorque();
    }

    if (arg_pair.first == "stop_move") {
      client.StopMove();
    }

    if (arg_pair.first == "high_stand") {
      client.HighStand();
    }

    if (arg_pair.first == "low_stand") {
      client.LowStand();
    }

    if (arg_pair.first == "balance_stand") {
      client.BalanceStand();
    }

    if (arg_pair.first == "continous_gait") {
      bool flag;
      if (arg_pair.second == "true") {
        flag = true;
      } else if (arg_pair.second == "false") {
        flag = false;
      } else {
        std::cerr << "invalid argument: " << arg_pair.second << std::endl;
        return 1;
      }
      client.ContinuousGait(flag);
    }

    if (arg_pair.first == "switch_move_mode") {
      bool flag;
      if (arg_pair.second == "true") {
        flag = true;
      } else if (arg_pair.second == "false") {
        flag = false;
      } else {
        std::cerr << "invalid argument: " << arg_pair.second << std::endl;
        return 1;
      }
      client.SwitchMoveMode(flag);
    }

    if (arg_pair.first == "move") {
      std::vector<float> param = stringToFloatVector(arg_pair.second);
      auto param_size = param.size();
      float vx, vy, omega;
      if (param_size == 3) {
        vx = param.at(0);
        vy = param.at(1);
        omega = param.at(2);
      } else {
        std::cerr << "Invalid param size for method SetVelocity: " << param_size << std::endl;
        return 1;
      }
      client.Move(vx, vy, omega);
    }

    if (arg_pair.first == "set_next_foot") {
      bool flag;
      if (arg_pair.second == "0") {
        flag = true;
      } else if (arg_pair.second == "1") {
        flag = false;
      } else {
        std::cerr << "invalid argument: " << arg_pair.second << std::endl;
        return 1;
      }
      client.SetNextFoot(flag);
    }

    if (arg_pair.first == "set_task_id") {
      int task_id = std::stoi(arg_pair.second);
      client.SetTaskId(task_id);
      std::cout << "set task_id to " << task_id << std::endl;
    }

    if (arg_pair.first == "shake_hand") {
      client.ShakeHand(0);
      std::cout << "Shake hand starts! Waiting for 10 s for ending" << std::endl;
      std::this_thread::sleep_for(std::chrono::seconds(10));
      std::cout << "Shake hand ends!" << std::endl;
      client.ShakeHand(1);
    }

    if (arg_pair.first == "wave_hand") {
      client.WaveHand();
      std::cout << "wave hand" << std::endl;
    }

    std::cout << "Done!" << std::endl;
  }

  return 0;
}