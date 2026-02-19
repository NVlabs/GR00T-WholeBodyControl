#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <optional>
#include <chrono>
#include <iomanip>

#include "robot_controller.hpp"
#include "user_controller.hpp"
#include "robot_interface.hpp"

using namespace unitree::common;
using namespace unitree::robot;

namespace fs = std::filesystem;

int main(int argc, char const *argv[])
{
    std::string param_folder;

    // parse command line params
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--param" && i + 1 < argc)
        {
            param_folder = argv[i + 1];
        }
    }
    fs::path param = fs::current_path() / param_folder;

    // log
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y_%m_%d_%H_%M_%S");

    fs::path log_folder = fs::current_path() / "logs" / ss.str();
    fs::create_directories(log_folder);

    std::ofstream cfg_file(log_folder / "cfg.txt");
    cfg_file << "param_folder: " << param << std::endl;
    cfg_file.close();

    fs::path log_file_name = log_folder / "log.txt";

    RobotController<ExampleUserController> robot_controller(log_file_name);
    robot_controller.LoadParam(param);

    robot_controller.InitDdsModel();

    robot_controller.StartControl();

    return 0;
}