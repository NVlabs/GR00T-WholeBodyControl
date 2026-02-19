/**
 * @file dex3_hands.hpp
 * @brief Driver for the Unitree Dex3 robotic hands (left + right).
 *
 * Dex3Hands manages two 7-DOF Dex3 hands via the Unitree DDS channel API.
 * It provides:
 *  - Thread-safe command and state buffers (via DataBuffer).
 *  - `writeOnce()` – called at the user's cadence to publish smoothed
 *    commands with delta-q clamping and close-ratio limiting.
 *  - Convenience helpers: `open()`, `close()`, `hold()`, `stop()`.
 *  - Per-joint or all-joint command setters.
 *
 * ## Close-Ratio Limiting
 *
 * A runtime-adjustable `max_close_ratio_` (range [0.2, 1.0]) limits how
 * far the fingers can close.  1.0 allows full closure; 0.2 keeps them
 * mostly open.  Controlled via `--max-close-ratio` CLI arg and X/C keys.
 *
 * ## Smoothing
 *
 * Each `writeOnce()` call clamps the per-joint position delta to
 * `MAX_DELTA_Q = 0.25` rad relative to the last known state, preventing
 * sudden jumps.
 *
 * ## DDS Topics
 *
 *   Direction | Left hand             | Right hand
 *   ----------|-----------------------|------------------------
 *   Command   | rt/dex3/left/cmd      | rt/dex3/right/cmd
 *   State     | rt/dex3/left/state    | rt/dex3/right/state
 */

#ifndef DEX3_HANDS_HPP
#define DEX3_HANDS_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <iostream>

#include <unitree/idl/hg/HandCmd_.hpp>
#include <unitree/idl/hg/HandState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "utils.hpp"

static constexpr int DEX3_MOTOR_MAX = 7;    ///< Number of motors per Dex3 hand.
static constexpr int DEX3_SENSOR_MAX = 9;   ///< Number of sensors per Dex3 hand.

/**
 * @class Dex3Hands
 * @brief Manages two Unitree Dex3 hands (left + right) over DDS channels.
 *
 * Does not run its own thread – the owning class calls `writeOnce()` at the
 * desired cadence from the command-writer thread.
 */
class Dex3Hands
{
public:
    Dex3Hands() = default;

    // Initializes channels for both hands. If networkInterface is empty, skips ChannelFactory init.
    void initialize(const std::string &networkInterface)
    {
        if (!networkInterface.empty())
        {
            unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface.c_str());
        }

        // Left hand namespaces
        const std::string leftPubNs = "rt/dex3/left";
        const std::string leftSubNs = "rt/dex3/left/state";
        // Right hand namespaces
        const std::string rightPubNs = "rt/dex3/right";
        const std::string rightSubNs = "rt/dex3/right/state";


        // initialize left hand cmd with default value
        unitree_hg::msg::dds_::HandCmd_ left_cmd;
        sizeCommand(left_cmd);
        left_.cmd_buffer.SetData(left_cmd);

        // initialize right hand cmd with default value
        unitree_hg::msg::dds_::HandCmd_ right_cmd;
        sizeCommand(right_cmd);
        right_.cmd_buffer.SetData(right_cmd);

        // Left hand
        left_.publisher.reset(new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>(leftPubNs + "/cmd"));
        left_.subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_>(leftSubNs));
        left_.publisher->InitChannel();
        left_.subscriber->InitChannel(
            [this](const void *message) { this->onState(true, message); }, 1);

        // Right hand
        right_.publisher.reset(new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::HandCmd_>(rightPubNs + "/cmd"));
        right_.subscriber.reset(new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::HandState_>(rightSubNs));
        right_.publisher->InitChannel();
        right_.subscriber->InitChannel(
            [this](const void *message) { this->onState(false, message); }, 1);
    }

    // Set max close ratio at runtime (bounded to [0.2, 1.0])
    void SetMaxCloseRatio(double ratio) {
        max_close_ratio_ = std::max(0.2, std::min(1.0, ratio));
    }
    
    // Get current max close ratio
    double GetMaxCloseRatio() const { return max_close_ratio_; }

    // Perform one publish tick; call this at your own cadence from the main class/thread.
    void writeOnce()
    {
        constexpr double MAX_DELTA_Q = 0.25;
        // Use runtime adjustable max_close_ratio_ instead of constexpr

        // Left hand publish with smoothing
        {
            const auto cmdPtr = left_.cmd_buffer.GetDataWithTime().data;
            const auto statePtr = left_.state_buffer.GetDataWithTime().data;
            
            if (left_.publisher && cmdPtr)
            {
                // Create a copy to apply clipping and smoothing
                unitree_hg::msg::dds_::HandCmd_ smoothedCmd = *cmdPtr;
                
                // Clip desired positions to prevent closing beyond max_close_ratio_
                // This applies to ALL joints including thumb (0-2)
                for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
                {
                    double desired_q = cmdPtr->motor_cmd()[i].q();
                    desired_q = clipToMaxOpen(desired_q, MAX_LIMITS_LEFT[i], MIN_LIMITS_LEFT[i], max_close_ratio_);
                    smoothedCmd.motor_cmd()[i].q(desired_q);
                }
                
                // If we have state feedback, clamp the delta
                if (statePtr && static_cast<int>(statePtr->motor_state().size()) == DEX3_MOTOR_MAX)
                {
                    for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
                    {
                        const double current_q = statePtr->motor_state()[i].q();
                        const double desired_q = smoothedCmd.motor_cmd()[i].q();
                        const double delta = desired_q - current_q;
                        
                        // Clamp delta to [-MAX_DELTA_Q, +MAX_DELTA_Q]
                        const double clamped_delta = std::max(-MAX_DELTA_Q, std::min(MAX_DELTA_Q, delta));
                        smoothedCmd.motor_cmd()[i].q(current_q + clamped_delta);
                    }
                }
                
                left_.publisher->Write(smoothedCmd);
            }
        }

        // Right hand publish with smoothing
        {
            const auto cmdPtr = right_.cmd_buffer.GetDataWithTime().data;
            const auto statePtr = right_.state_buffer.GetDataWithTime().data;
            
            if (right_.publisher && cmdPtr)
            {
                // Create a copy to apply clipping and smoothing
                unitree_hg::msg::dds_::HandCmd_ smoothedCmd = *cmdPtr;
                
                // Clip desired positions to prevent closing beyond max_close_ratio_
                // This applies to ALL joints including thumb (0-2)
                for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
                {
                    double desired_q = cmdPtr->motor_cmd()[i].q();
                    desired_q = clipToMaxOpen(desired_q, MAX_LIMITS_RIGHT[i], MIN_LIMITS_RIGHT[i], max_close_ratio_);
                    smoothedCmd.motor_cmd()[i].q(desired_q);
                }
                
                // If we have state feedback, clamp the delta
                if (statePtr && static_cast<int>(statePtr->motor_state().size()) == DEX3_MOTOR_MAX)
                {
                    for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
                    {
                        const double current_q = statePtr->motor_state()[i].q();
                        const double desired_q = smoothedCmd.motor_cmd()[i].q();
                        const double delta = desired_q - current_q;
                        
                        // Clamp delta to [-MAX_DELTA_Q, +MAX_DELTA_Q]
                        const double clamped_delta = std::max(-MAX_DELTA_Q, std::min(MAX_DELTA_Q, delta));
                        smoothedCmd.motor_cmd()[i].q(current_q + clamped_delta);
                    }
                }
                
                right_.publisher->Write(smoothedCmd);
            }
        }
    }

    // Returns a snapshot of the latest state for the requested hand.
    std::shared_ptr<const unitree_hg::msg::dds_::HandState_> getState(bool is_left) const
    {
        const HandCtx &ctx = is_left ? left_ : right_;
        return ctx.state_buffer.GetDataWithTime().data;
    }

    // Returns a snapshot of the current command buffer for the requested hand.
    std::shared_ptr<const unitree_hg::msg::dds_::HandCmd_> getCommand(bool is_left) const
    {
        const HandCtx &ctx = is_left ? left_ : right_;
        return ctx.cmd_buffer.GetDataWithTime().data;
    }

    // Optional: access DataBuffer results directly to detect missing data
    TimestampedData<unitree_hg::msg::dds_::HandState_> getStateWithTime(bool is_left) const
    {
        const HandCtx &ctx = is_left ? left_ : right_;
        return ctx.state_buffer.GetDataWithTime();
    }

    TimestampedData<unitree_hg::msg::dds_::HandCmd_> getCommandWithTime(bool is_left) const
    {
        const HandCtx &ctx = is_left ? left_ : right_;
        return ctx.cmd_buffer.GetDataWithTime();
    }

    bool hasState(bool is_left) const { return getStateWithTime(is_left).HasData(); }
    bool hasCommand(bool is_left) const { return getCommandWithTime(is_left).HasData(); }

    // Sets the command buffer for the requested hand. The writer will publish it in writeOnce().
    void setCommand(bool is_left, const unitree_hg::msg::dds_::HandCmd_ &cmd)
    {
        HandCtx &ctx = is_left ? left_ : right_;
        if (static_cast<int>(cmd.motor_cmd().size()) != DEX3_MOTOR_MAX)
        {
            std::cerr << "[Dex3Hands] setCommand: invalid motor_cmd size "
                      << cmd.motor_cmd().size() << ", expected " << DEX3_MOTOR_MAX << std::endl;
            return;
        }
        ctx.cmd_buffer.SetData(cmd);
    }

    // Convenience: set a single joint's command using simple fields.
    // Required: q. Optional: dq, kp, kd, tau. Mode is set internally following Unitree example (status=0x01, timeout=0).
    void setJointCommand(bool is_left, int joint_index,
                         double q,
                         std::optional<double> dq = std::nullopt,
                         std::optional<double> kp = std::nullopt,
                         std::optional<double> kd = std::nullopt,
                         std::optional<double> tau = std::nullopt)
    {
        if (joint_index < 0 || joint_index >= DEX3_MOTOR_MAX) { return; }
        HandCtx &ctx = is_left ? left_ : right_;
        const auto currentPtr = ctx.cmd_buffer.GetDataWithTime().data;
        unitree_hg::msg::dds_::HandCmd_ cmd = currentPtr ? *currentPtr : unitree_hg::msg::dds_::HandCmd_();
        if (!currentPtr) { sizeCommand(cmd); }
        auto &m = cmd.motor_cmd()[joint_index];
        m.mode(makeMode(static_cast<uint8_t>(joint_index), /*status*/ 0x01, /*timeout*/ 0));
        m.q(q);
        if (dq) { m.dq(*dq); }
        if (kp) { m.kp(*kp); }
        if (kd) { m.kd(*kd); }
        if (tau) { m.tau(*tau); }
        ctx.cmd_buffer.SetData(std::move(cmd));
    }

    // Convenience: set all joints' q, and optionally per-joint dq/kp/kd/tau arrays.
    void setAllJointsCommand(bool is_left,
                             const std::array<double, DEX3_MOTOR_MAX> &q,
                             std::optional<std::array<double, DEX3_MOTOR_MAX>> dq = std::nullopt,
                             std::optional<std::array<double, DEX3_MOTOR_MAX>> kp = std::nullopt,
                             std::optional<std::array<double, DEX3_MOTOR_MAX>> kd = std::nullopt,
                             std::optional<std::array<double, DEX3_MOTOR_MAX>> tau = std::nullopt)
    {
        HandCtx &ctx = is_left ? left_ : right_;
        const auto currentPtr = ctx.cmd_buffer.GetDataWithTime().data;
        unitree_hg::msg::dds_::HandCmd_ cmd = currentPtr ? *currentPtr : unitree_hg::msg::dds_::HandCmd_();
        if (!currentPtr) { sizeCommand(cmd); }
        for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
        {
            auto &m = cmd.motor_cmd()[i];
            m.mode(makeMode(static_cast<uint8_t>(i), /*status*/ 0x01, /*timeout*/ 0));
            m.q(q[i]);
            if (dq) { m.dq((*dq)[i]); }
            if (kp) { m.kp((*kp)[i]); }
            if (kd) { m.kd((*kd)[i]); }
            if (tau) { m.tau((*tau)[i]); }
        }
        ctx.cmd_buffer.SetData(std::move(cmd));
    }

    // Quick helper: STOP - relax all joints (timeout=1, zero gains/vel/torque/pos like example)
    void stop(bool is_left)
    {
        HandCtx &ctx = is_left ? left_ : right_;
        unitree_hg::msg::dds_::HandCmd_ cmd; sizeCommand(cmd);
        for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
        {
            auto &m = cmd.motor_cmd()[i];
            m.mode(makeMode(static_cast<uint8_t>(i), /*status*/ 0x01, /*timeout*/ 0x01));
            m.tau(0);
            m.dq(0);
            m.kp(0);
            m.kd(0);
            m.q(0);
        }
        ctx.cmd_buffer.SetData(std::move(cmd));
    }

    // Quick helper: HOLD - hold current pose with gains (default kp=1.5, kd=0.1, dq=0, tau=0)
    void hold(bool is_left, double kp = 1.5, double kd = 0.1)
    {
        // Copy current joint positions from state under lock
        std::array<double, DEX3_MOTOR_MAX> q_current {};
        {
            const HandCtx &ctx_ro = is_left ? left_ : right_;
            const auto data = ctx_ro.state_buffer.GetDataWithTime().data;
            if (!data) { return; }
            for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
            {
                q_current[i] = data->motor_state()[i].q();
            }
        }
        // Write hold command into command buffer
        HandCtx &ctx = is_left ? left_ : right_;
        unitree_hg::msg::dds_::HandCmd_ cmd; sizeCommand(cmd);
        for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
        {
            auto &m = cmd.motor_cmd()[i];
            m.mode(makeMode(static_cast<uint8_t>(i), /*status*/ 0x01, /*timeout*/ 0));
            m.q(q_current[i]);
            m.dq(0);
            m.kp(kp);
            m.kd(kd);
            m.tau(0);
        }
        ctx.cmd_buffer.SetData(std::move(cmd));
    }

    // Quick helper: CLOSE - move to mid-range pose (per Unitree example) with gains
    void close(bool is_left, double kp = 1.5, double kd = 0.1)
    {
        const auto &maxLims = is_left ? MAX_LIMITS_LEFT : MAX_LIMITS_RIGHT;
        const auto &minLims = is_left ? MIN_LIMITS_LEFT : MIN_LIMITS_RIGHT;
        HandCtx &ctx = is_left ? left_ : right_;
        unitree_hg::msg::dds_::HandCmd_ cmd; sizeCommand(cmd);
        for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
        {
            const double mid = (maxLims[i] + minLims[i]) / 2.0;
            auto &m = cmd.motor_cmd()[i];
            m.mode(makeMode(static_cast<uint8_t>(i), /*status*/ 0x01, /*timeout*/ 0));
            m.q(mid);
            m.dq(0);
            m.kp(kp);
            m.kd(kd);
            m.tau(0);
        }
        ctx.cmd_buffer.SetData(std::move(cmd));
    }

    // Quick helper: OPEN - move to open pose using per-joint 0 with gains
    void open(bool is_left, double kp = 1.5, double kd = 0.1)
    {
        HandCtx &ctx = is_left ? left_ : right_;
        unitree_hg::msg::dds_::HandCmd_ cmd; sizeCommand(cmd);
        for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
        {
            auto &m = cmd.motor_cmd()[i];
            m.mode(makeMode(static_cast<uint8_t>(i), /*status*/ 0x01, /*timeout*/ 0));
            m.q(0);
            m.dq(0);
            m.kp(kp);
            m.kd(kd);
            m.tau(0);
        }
        ctx.cmd_buffer.SetData(std::move(cmd));
    }

private:
    struct HandCtx
    {
        unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::HandCmd_> publisher;
        unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::HandState_> subscriber;

        DataBuffer<unitree_hg::msg::dds_::HandState_> state_buffer;
        DataBuffer<unitree_hg::msg::dds_::HandCmd_> cmd_buffer;
    };

    static void sizeCommand(unitree_hg::msg::dds_::HandCmd_ &cmd)
    {
        cmd.motor_cmd().resize(DEX3_MOTOR_MAX);
        // give default value to the command
        for (int i = 0; i < DEX3_MOTOR_MAX; ++i)
        {
            cmd.motor_cmd()[i].mode(makeMode(static_cast<uint8_t>(i), /*status*/ 0x01, /*timeout*/ 0));
            cmd.motor_cmd()[i].q(0);
            cmd.motor_cmd()[i].dq(0);
            cmd.motor_cmd()[i].kp(1.5);
            cmd.motor_cmd()[i].kd(0.1);
            cmd.motor_cmd()[i].tau(0);
        }
    }

    void onState(bool is_left, const void *message)
    {
        HandCtx &ctx = is_left ? left_ : right_;
        const auto *incoming = static_cast<const unitree_hg::msg::dds_::HandState_ *>(message);
        ctx.state_buffer.SetData(*incoming);
    }

    static uint8_t makeMode(uint8_t motor_id, uint8_t status, uint8_t timeout)
    {
        uint8_t mode = 0;
        mode |= (motor_id & 0x0F);
        mode |= (status & 0x07) << 4;
        mode |= (timeout & 0x01) << 7;
        return mode;
    }

    // Clip desired_q to prevent closing beyond MAX_CLOSE_RATIO (e.g., 0.05 = 95% open, 5% closed)
    // q=0 is fully open, so we clip to prevent q from being more closed than MAX_CLOSE_RATIO * limit
    // Allow range: fully open (q=0) to 95% open (MAX_CLOSE_RATIO * limit)
    static double clipToMaxOpen(double desired_q, double max_limit, double min_limit, double MAX_CLOSE_RATIO)
    {
        // Calculate the maximum allowed closed position (95% open = 5% closed)
        double q_max_open_pos = MAX_CLOSE_RATIO * max_limit;  // For positive direction
        double q_max_open_neg = MAX_CLOSE_RATIO * min_limit;  // For negative direction
        
        // Clip based on direction - allow q=0 (fully open) but prevent closing beyond 95% open
        if (desired_q > 0.0 && max_limit > 0.0)
        {
            // Positive direction: prevent closing beyond 95% open
            if (desired_q > q_max_open_pos)
            {
                return q_max_open_pos;
            }
        }
        else if (desired_q < 0.0 && min_limit < 0.0)
        {
            // Negative direction: prevent closing beyond 95% open
            if (desired_q < q_max_open_neg)
            {
                return q_max_open_neg;
            }
        }
        // If q=0 or both limits are zero, allow it (fully open is allowed)
        
        return desired_q;
    }

    // no internal thread; main class owns timing

    HandCtx left_;
    HandCtx right_;
    
    // Runtime adjustable max close ratio (default 1.0 = fully closed allowed)
    // Bounded to [0.2, 1.0] - higher values allow more closing
    // Use --max-close-ratio arg to set initial limit, X/C keys to adjust at runtime
    double max_close_ratio_ = 1.0;

    // Limits taken from Unitree example for test "close" pose
    static constexpr std::array<double, DEX3_MOTOR_MAX> MAX_LIMITS_LEFT  = { 1.05,  1.05,  1.75,  0.0,  0.0,  0.0,  0.0 };
    static constexpr std::array<double, DEX3_MOTOR_MAX> MIN_LIMITS_LEFT  = {-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75 };
    static constexpr std::array<double, DEX3_MOTOR_MAX> MAX_LIMITS_RIGHT = { 1.05,  0.742, 0.0,  1.57,  1.75,  1.57,  1.75 };
    static constexpr std::array<double, DEX3_MOTOR_MAX> MIN_LIMITS_RIGHT = {-1.05, -1.05, -1.75, 0.0,  0.0,  0.0,  0.0 };
};

#endif // DEX3_HANDS_HPP