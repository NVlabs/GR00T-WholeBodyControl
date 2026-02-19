#pragma once

#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <array>
#include <chrono>
#include <thread>
#include <filesystem>
#include <mutex>
#include <fstream>
#include <future>
#include <algorithm>

#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/idl/go2/LowCmd_.hpp"
#include "unitree/common/thread/thread.hpp"

#include "unitree/robot/channel/channel_publisher.hpp"
#include "unitree/robot/channel/channel_subscriber.hpp"
#include "unitree/common/time/time_tool.hpp"

#include "state_machine.hpp"
#include "gamepad.hpp"
#include "robot_interface.hpp"

using namespace unitree::common;
using namespace unitree::robot;
namespace fs = std::filesystem;

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

template <typename USER_CTRL>
class RobotController
{
public:
    RobotController() {}

    RobotController(fs::path &log_file_name)
    {
        // set log file
        log_file = std::ofstream(log_file_name, std::ios::binary);
    }

    void LoadParam(fs::path &param_folder)
    {
        ctrl.LoadParam(param_folder);
    }

    void InitDdsModel(const std::string &networkInterface = "")
    {
        // init dds
        ChannelFactory::Instance()->Init(0, networkInterface);
        lowcmd_publisher.reset(new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
        lowstate_subscriber.reset(new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));

        lowcmd_publisher->InitChannel();
        lowstate_subscriber->InitChannel(std::bind(&RobotController::LowStateMessageHandler, this, std::placeholders::_1), 1);
    }

    void StartControl()
    {
        // waiting for gamepad command to start the control thread
        std::chrono::milliseconds duration(100);
        // listen to gamepad command
        while (true)
        {
            std::cout << "Press R2 to start!" << std::endl;
            std::this_thread::sleep_for(duration);

            InteprateGamePad();
            if (gamepad.R2.on_press)
            {
                break;
            }
        }

        // prepare for start
        std::cout << "Start!" << std::endl;
        Damping();
        ctrl_dt_micro_sec = static_cast<uint64_t>(ctrl.dt * 1000000);

        // Start the control thread
        control_thread_ptr = CreateRecurrentThreadEx("ctrl", UT_CPU_ID_NONE, ctrl_dt_micro_sec, &RobotController::ControlStep, this);

        // Start the lowlevel command thread
        std::this_thread::sleep_for(duration);
        StartSendCmd();

        // keep the main thread alive
        while (true)
        {
            std::this_thread::sleep_for(duration);
        }
    }

protected:
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;
    ThreadPtr low_cmd_write_thread_ptr, control_thread_ptr;
    unitree_go::msg::dds_::LowCmd_ cmd;
    unitree_go::msg::dds_::LowState_ state;

    Gamepad gamepad;
    REMOTE_DATA_RX rx;

    SimpleStateMachine state_machine;
    USER_CTRL ctrl;
    RobotInterface robot_interface;

    std::mutex state_mutex, cmd_mutex;

    std::ofstream log_file;

    uint64_t ctrl_dt_micro_sec = 2000;

    std::vector<float> compute_time;

private:

    void LowStateMessageHandler(const void *message)
    {
        state = *(unitree_go::msg::dds_::LowState_ *)message;
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            robot_interface.GetState(state);
        }
    }

    void InteprateGamePad()
    {
        // update gamepad
        memcpy(rx.buff, &state.wireless_remote()[0], 40);
        gamepad.update(rx.RF_RX);
    }

    void LowCmdwriteHandler()
    {
        // write low-level command
        {
            std::lock_guard<std::mutex> lock(cmd_mutex);
            lowcmd_publisher->Write(cmd);
        }
    }

    void StartSendCmd()
    {
        low_cmd_write_thread_ptr = CreateRecurrentThreadEx("writebasiccmd", UT_CPU_ID_NONE, 2000, &RobotController::LowCmdwriteHandler, this);
    }

   void UpdateStateMachine()
    {
        // R2 -> Stand
        // A -> Ctrl
        // L2 + B -> Stop
        if (gamepad.R2.on_press)
        {
            if (state_machine.Stand())
            {
                StandCallback();
            }
        }
        if (gamepad.A.on_press)
        {
            if (state_machine.Ctrl())
            {
                CtrlCallback();
            }
        }
        if (gamepad.L2.pressed && gamepad.B.pressed)
        {
            state_machine.Stop();
        }
    }

    void ControlStep()
    {
        // main loop

        // update state
        InteprateGamePad();
        UpdateStateMachine();

        // select control modes according to the state machine
        auto start = std::chrono::high_resolution_clock::now();
        if (state_machine.state == STATES::STAND)
        {
            Standing();
        }
        if (state_machine.state == STATES::DAMPING)
        {
            Damping();
        }
        if (state_machine.state == STATES::CTRL)
        {
            UserControlStep(true);
            if (CheckTermination())
            {
                state_machine.Stop();
            }
        }

        // update low-level command
        {
            std::lock_guard<std::mutex> lock(cmd_mutex);
            robot_interface.SetCommand(cmd);
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        compute_time.push_back(duration.count() / 1000.);

        // write log
        WriteLog();

        if (compute_time.size() == 100)
        {
            float sum = 0;
            for (auto &t : compute_time)
            {
                sum += t;
            }
            std::cout << "Performance: mean: " << sum / 100
                      << " ms; max: " << *std::max_element(compute_time.begin(), compute_time.end())
                      << " ms; min: " << *std::min_element(compute_time.begin(), compute_time.end())
                      << "ms." << std::endl;

            compute_time.clear();

            std::cout << "Current State: " << static_cast<size_t>(state_machine.state) << std::endl;
        }
    }

    void WriteLog()
    {
        if (log_file.is_open())
        {
            auto log = ctrl.GetLog();
            for (const auto &v : log)
            {
                log_file << v << " ";
            }
            log_file << std::endl;
        }
    }

    void StandCallback()
    {
        robot_interface.jpos_des = ctrl.init_pos;
        robot_interface.jvel_des.fill(0.);
        robot_interface.kp.fill(ctrl.kp * state_machine.pd_ratio);
        robot_interface.kd.fill(ctrl.kd * state_machine.pd_ratio);
        robot_interface.tau_ff.fill(0.);
    }

    void CtrlCallback()
    {
        ctrl.Reset(robot_interface, gamepad);

        robot_interface.jpos_des = ctrl.init_pos;
        robot_interface.jvel_des.fill(0.);
        robot_interface.kp.fill(ctrl.kp);
        robot_interface.kd.fill(ctrl.kd);
        robot_interface.tau_ff.fill(0.);
    }

    void Damping(float kd = 2.0)
    {
        robot_interface.jpos_des = ctrl.init_pos;
        robot_interface.jvel_des.fill(0.);
        robot_interface.kp.fill(0.);
        robot_interface.kd.fill(kd);
        robot_interface.tau_ff.fill(0.);
    }

    void Standing(float kp = 40.0, float kd = 1.0)
    {
        if (gamepad.R2.pressed)
        {
            state_machine.Standing(true);
        }
        if (gamepad.R1.pressed)
        {
            state_machine.Standing(false);
        }

        UserControlStep(false);

        robot_interface.kp.fill(kp * state_machine.pd_ratio);
        robot_interface.kd.fill(kd * state_machine.pd_ratio);
    }

    void UserControlStep(bool send = true)
    {
        {
            std::lock_guard<std::mutex> lock(state_mutex);
            ctrl.GetInput(robot_interface, gamepad);
        }
        ctrl.Calculate();

        if (send)
        {
            robot_interface.jpos_des = ctrl.jpos_des;
        }
    }

    bool CheckTermination()
    {
        if (robot_interface.projected_gravity.at(2) > 0)
        {
            return true;
        }
        return false;
    }
};
