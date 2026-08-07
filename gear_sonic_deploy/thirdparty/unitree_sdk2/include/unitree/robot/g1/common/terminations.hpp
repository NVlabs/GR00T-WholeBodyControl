#pragma once

/**
 * This file implements some functions used to protect the G1 during runtime.
 * When the function returns true, it is recommended to set the motor to passive mode in the lower-level control.
 */

#include <eigen3/Eigen/Dense>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/idl/hg/BmsState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace unitree {
namespace robot {
namespace g1 {  

// robot orientation is too far from the desired orientation limits.
inline bool bad_orientation(const unitree_hg::msg::dds_::LowState_ & lowstate, float limit_angle = 1.0)
{
    auto & imu = lowstate.imu_state();
    Eigen::Quaternionf quat(
        imu.quaternion()[0],
        imu.quaternion()[1],
        imu.quaternion()[2],
        imu.quaternion()[3]
    );
    Eigen::Vector3f projected_gravity_b = quat.conjugate() * Eigen::Vector3f(0, 0, -1);
    return std::fabs(std::acos(-projected_gravity_b[2])) > limit_angle;
}

// Joint velocities are outside of the soft joint limits
inline bool joint_vel_out_of_limit(const unitree_hg::msg::dds_::LowState_ & lowstate, float limit_vel = 10.0)
{
    auto & motors = lowstate.motor_state();
    return std::any_of(motors.begin(), motors.end(), [limit_vel](const auto & motor) {
        return std::fabs(motor.dq()) > limit_vel;
    });
}

// Agnular velocities are outside of the soft joint limits
inline bool ang_vel_out_of_limit(const unitree_hg::msg::dds_::LowState_ & lowstate, float limit_vel = 6.0)
{
    auto & gyroscope = lowstate.imu_state().gyroscope();
    return std::any_of(gyroscope.begin(), gyroscope.end(), [limit_vel](const auto & ang_vel) {
        return std::fabs(ang_vel) > limit_vel;
    });
}

// Motor winding temperature is above the limit
inline bool motor_winding_overheat(const unitree_hg::msg::dds_::LowState_ & lowstate, float limit_temp = 120.0)
{
    auto & motors = lowstate.motor_state();
    return std::any_of(motors.begin(), motors.end(), [limit_temp](const auto & motor) {
        return motor.temperature()[1] > limit_temp;
    });
}

// Motor casing temperature is above the limit
inline bool motor_casing_overheat(const unitree_hg::msg::dds_::LowState_ & lowstate, float limit_temp = 85.0)
{
    auto & motors = lowstate.motor_state();
    return std::any_of(motors.begin(), motors.end(), [limit_temp](const auto & motor) {
        return motor.temperature()[0] > limit_temp;
    });
}

// State of charge (SOC) is below the limit 
inline bool low_battery(const  unitree_hg::msg::dds_::BmsState_ & bms_state, float limit_soc = 20.0)
{
    return bms_state.soc() < limit_soc;
}

/**
 * @brief Lost connection to the robot
 * This function checks if the last data available time is older than the specified timeout.
 * If the timeout is reached, it indicates a lost connection.
 * 
 * When using a wired connection to the robot, a loose network cable may cause the connection to be interrupted.
 * If the program continues to run at this time, it will send a step signal to the motors, causing violent movement.
 */
inline bool lost_connection(unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_> & subscriber, int64_t timeout_ms = 1000)
{
    auto now = unitree::common::GetCurrentMonotonicTimeNanosecond();
    auto elasped_ms = (now - subscriber->GetLastDataAvailableTime()) / 1e6;
    return elasped_ms > timeout_ms;
}

}
}
}