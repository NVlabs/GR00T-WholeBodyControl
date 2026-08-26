#pragma once

#include <array>
#include <iostream>

#include "comm.h"
#include "unitree/idl/go2/LowState_.hpp"
#include "unitree/idl/go2/LowCmd_.hpp"
#include "conversion.hpp"

namespace unitree::common
{

    constexpr double PosStopF = (2.146E+9f);
    constexpr double VelStopF = (16000.0f);

    class BasicRobotInterface
    {
    public:
        BasicRobotInterface()
        {
            jpos_des.fill(0.);
            jvel_des.fill(0.);
            kp.fill(0.);
            kd.fill(0.);
            tau_ff.fill(0.);
            projected_gravity.fill(0.);
            projected_gravity.at(2) = -1.0;
        }

        void GetState(unitree_go::msg::dds_::LowState_ &state)
        {
            // imu
            const unitree_go::msg::dds_::IMUState_ &imu = state.imu_state();
            quat = imu.quaternion();
            rpy = imu.rpy();
            gyro = imu.gyroscope();
            UpdateProjectedGravity();

            // motor
            const std::array<unitree_go::msg::dds_::MotorState_, 20> &motor = state.motor_state();
            for (size_t i = 0; i < 12; ++i)
            {
                const unitree_go::msg::dds_::MotorState_ &m = motor.at(i);
                jpos.at(i) = m.q();
                jvel.at(i) = m.dq();
                tau.at(i) = m.tau_est();
            }
        }

        virtual void SetCommand(unitree_go::msg::dds_::LowCmd_ &cmd) = 0;

        std::array<float, 12> jpos, jvel, tau;
        std::array<float, 4> quat;
        std::array<float, 3> rpy, gyro, projected_gravity;
        std::array<float, 12> jpos_des, jvel_des, kp, kd, tau_ff;

    private:
        inline void UpdateProjectedGravity()
        {
            // inverse quat
            float w = quat.at(0);
            float x = -quat.at(1);
            float y = -quat.at(2);
            float z = -quat.at(3);

            float x2 = x * x;
            float y2 = y * y;
            float z2 = z * z;
            float w2 = w * w;
            float xy = x * y;
            float xz = x * z;
            float yz = y * z;
            float wx = w * x;
            float wy = w * y;
            float wz = w * z;

            projected_gravity.at(0) = -2 * (xz + wy);
            projected_gravity.at(1) = -2 * (yz - wx);
            projected_gravity.at(2) = -(w2 - x2 - y2 + z2);
        }
    };

    class RobotInterface : public BasicRobotInterface
    {
    public:
        RobotInterface() : BasicRobotInterface()
        {
            InitLowCmd();
        }

        void SetCommand(unitree_go::msg::dds_::LowCmd_ &cmd)
        {

            for (int i = 0; i < 12; ++i)
            {
                low_cmd.motor_cmd()[i].q() = jpos_des.at(i);
                low_cmd.motor_cmd()[i].dq() = jvel_des.at(i);
                low_cmd.motor_cmd()[i].kp() = kp.at(i);
                low_cmd.motor_cmd()[i].kd() = kd.at(i);
                low_cmd.motor_cmd()[i].tau() = tau_ff.at(i);
            }

            low_cmd.crc() = crc32_core((uint32_t *)&low_cmd, (sizeof(unitree_go::msg::dds_::LowCmd_)>>2)-1);
            // lowCmd2Dds(low_cmd_raw, cmd);
            cmd = low_cmd;
        }

    private:
        void InitLowCmd()
        {
            low_cmd.head()[0] = 0xFE;
            low_cmd.head()[1] = 0xEF;
            low_cmd.level_flag() = 0xFF;
            low_cmd.gpio() = 0;

            for(int i=0; i<20; i++)
            {
                low_cmd.motor_cmd()[i].mode() = (0x01);   // motor switch to servo (PMSM) mode
                low_cmd.motor_cmd()[i].q() = (PosStopF);
                low_cmd.motor_cmd()[i].kp() = (0);
                low_cmd.motor_cmd()[i].dq() = (VelStopF);
                low_cmd.motor_cmd()[i].kd() = (0);
                low_cmd.motor_cmd()[i].tau() = (0);
            }
        }

        unitree_go::msg::dds_::LowCmd_ low_cmd;
    };
} // namespace unitree::common