#pragma once

#include <array>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>

#include "robot_interface.hpp"
#include "gamepad.hpp"
#include "cfg.hpp"

namespace fs = std::filesystem;

namespace unitree::common
{
    class BasicUserController
    {
    public:
        BasicUserController() {}

        virtual void LoadParam(fs::path &param_folder) = 0;

        virtual void Reset(RobotInterface &robot_interface, Gamepad &gamepad) = 0;

        virtual void GetInput(RobotInterface &robot_interface, Gamepad &gamepad) = 0;

        virtual void Calculate() = 0;

        virtual std::vector<float> GetLog() = 0;

        float dt, kp, kd;
        std::array<float, 12> init_pos;
        std::array<float, 12> jpos_des;
    };

    class ExampleUserController : public BasicUserController
    {
    public:
        ExampleUserController() {}

        void LoadParam(fs::path &param_folder)
        {
            // load param file
            std::ifstream cfg_file(param_folder / "params.json");
            std::cout << "Read params from: " << param_folder / "params.json" << std::endl;
            std::stringstream ss;
            ss << cfg_file.rdbuf();
            FromJsonString(ss.str(), cfg);

            // get data from json
            dt = cfg.dt;
            kp = cfg.kp;
            kd = cfg.kd;
            for (int i = 0; i < 12; ++i)
            {
                init_pos.at(i) = cfg.init_pos.at(i);
            }
        }

        void GetInput(RobotInterface &robot_interface, Gamepad &gamepad)
        {
            // save necessary data from input

            // record command
            cmd.at(0) = gamepad.ly;
            cmd.at(1) = -gamepad.lx;
            cmd.at(2) = -gamepad.rx;

            // record robot state
            for (int i = 0; i < 12; ++i)
            {
                jpos_processed.at(i) = robot_interface.jpos.at(i) - init_pos.at(i);
                jvel_processed.at(i) = robot_interface.jvel.at(i) / 2.0;
            }
        }

        void Reset(RobotInterface &robot_interface, Gamepad &gamepad)
        {
            GetInput(robot_interface, gamepad);
            Calculate();
        }

        void Calculate()
        {
            // calculate jpos_des
            jpos_des = init_pos;
        }

        std::vector<float> GetLog()
        {
            // record input, output and other info into a vector
            std::vector<float> log;
            for (int i = 0; i < 3; ++i)
            {
                log.push_back(cmd.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jpos_processed.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jvel_processed.at(i));
            }
            for (int i = 0; i < 12; ++i)
            {
                log.push_back(jpos_des.at(i));
            }
            
            return log;
        }

        // cfg
        ExampleCfg cfg;

        // state
        std::array<float, 3> cmd;
        std::array<float, 12> jpos_processed;
        std::array<float, 12> jvel_processed;
    };
} // namespace unitree::common
