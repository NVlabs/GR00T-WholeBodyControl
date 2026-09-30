/**
 * @file safe_stop.hpp
 * @brief "Safe stop": stop teleop / VLA and go to a soft ready stand.
 *
 * Independent of the arm-compliance layer.  Used with --input-type zmq_manager
 * (the deploy used for PICO teleop and the VLA).
 *
 * What happens on a stop request:
 *   1. ZMQManager (input thread) stops listening to the operator / VLA:
 *        - VR 3-point teleop: the operator's hands are replaced by a smooth
 *          minimum-jerk path of the VR hand targets from where they were to the
 *          rest pose (peak hand speed hand_speed, lower_min_s..lower_max_s); the
 *          POLICY follows it (it looks like an operator slowly lowering the
 *          hands) and keeps its balance. The hands then stay at the rest pose.
 *        - planner messages are ignored: locomotion forced to IDLE, upper-body
 *          and hand targets dropped;
 *        - streamed full-body motion -> switched to PLANNER mode (SONIC's own
 *          safety-reset path, the same one the PICO uses to leave teleop).
 *   2. The deploy (control thread) waits until the arms have come down and are
 *      still, then ramps the arm stiffness down slowly (min-jerk) to a soft level
 *      (default Kp x 0.5, Kd x sqrt(0.5) so the damping ratio is unchanged).
 *   LATCHED until a release request.  Release: arm stiffness ramps back to
 *   normal; the robot stays in planner idle.  Teleop / VLA only take over again
 *   after the operator re-engages (leave + re-enter teleop on the PICO), never
 *   automatically.
 *
 * Triggers (any thread):
 *   - keyboard in the deploy terminal: k = stop, u = release   (ZMQManager)
 *   - ZMQ command topic: optional bool fields "safe_stop" / "safe_release"
 *     (e.g. from a voice-command node)
 *   - code: safe_stop::Request("reason") / safe_stop::Release("reason")
 *
 * The Unitree remote and the 'O' key remain the whole-robot emergency stop.
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <string>

namespace safe_stop {

/// true while a safe stop is latched.
inline std::atomic<bool>& Active() {
  static std::atomic<bool> active{false};
  return active;
}

/// Incremented on every new stop (lets the control thread restart its gain sequence).
inline std::atomic<unsigned>& Epoch() {
  static std::atomic<unsigned> epoch{0};
  return epoch;
}

/// Request a safe stop (idempotent while active).
inline void Request(const std::string& source) {
  if (!Active().exchange(true)) {
    Epoch().fetch_add(1);
    std::cout << "\n[SafeStop] STOP (" << source << "): teleop/VLA ignored, going to ready stand; "
              << "arms soften once lowered. Press u (or send safe_release) to release." << std::endl;
  }
}

/// Release a latched safe stop.
inline void Release(const std::string& source) {
  if (Active().exchange(false)) {
    std::cout << "\n[SafeStop] RELEASED (" << source << "): arm stiffness back to normal; robot stays in "
              << "idle. Re-engage teleop / VLA to continue." << std::endl;
  } else {
    std::cout << "[SafeStop] (" << source << ") no safe stop active." << std::endl;
  }
}

/// Gain-softening settings (set from the command line; read by the control thread).
struct Config {
  double kp_scale = 0.5;       ///< Arm Kp x this once the arms are down.
  double soften_s = 2.0;       ///< Duration of the softening ramp (slow).
  double restore_s = 1.0;      ///< Ramp back to normal stiffness on release.
  double settle_speed = 0.15;  ///< Arms count as "still" below this joint speed (rad/s)...
  double settle_hold_s = 0.3;  ///< ...for this long,
  double settle_min_s = 1.0;   ///< but not earlier than this after the stop,
  double settle_max_s = 4.0;   ///< and at the latest after this (after the lowering path).
  // Lowering path of the VR hand targets (VR 3-point teleop)
  double hand_speed = 0.20;    ///< Peak hand speed of the path (m/s).
  double lower_min_s = 2.0;    ///< Path duration limits (s).
  double lower_max_s = 5.0;
};

/// Duration of the current lowering path (s), set by ZMQManager, read by ArmSoftener.
inline std::atomic<double>& LoweringDuration() {
  static std::atomic<double> d{0.0};
  return d;
}

inline Config& Settings() {
  static Config cfg;
  return cfg;
}

/**
 * @brief Control-thread helper: arm gain scale for the safe stop.
 *
 * IDLE -> (stop) WAIT_SETTLE: scale held while the policy lowers the arms ->
 * SOFTENING: min-jerk ramp to kp_scale over soften_s -> SOFT (latched) ->
 * (release) RESTORING: ramp back to 1 over restore_s -> IDLE.
 * Kd is scaled by sqrt(kp scale), which keeps the damping ratio.
 */
class ArmSoftener {
 public:
  enum class State { kIdle, kWaitSettle, kSoftening, kSoft, kRestoring };

  /// @param arm_dq measured arm joint speeds (any count). Returns the Kp scale.
  template <size_t N>
  double Update(const std::array<float, N>& arm_dq, double dt) {
    const Config& c = Settings();
    const bool active = Active().load();
    const unsigned epoch = Epoch().load();
    if (active && (state_ == State::kIdle || state_ == State::kRestoring || epoch != epoch_)) {
      epoch_ = epoch;
      state_ = State::kWaitSettle;
      t_ = 0.0;
      still_t_ = 0.0;
      peak_speed_ = 0.0;
    } else if (!active && state_ != State::kIdle && state_ != State::kRestoring) {
      Start(State::kRestoring, 1.0, c.restore_s);
      std::cout << "[SafeStop] Restoring arm stiffness over " << c.restore_s << " s" << std::endl;
    }

    t_ += dt;
    switch (state_) {
      case State::kIdle: scale_ = 1.0; break;
      case State::kWaitSettle: {
        float vmax = 0.0f;
        for (float v : arm_dq) vmax = std::max(vmax, std::fabs(v));
        peak_speed_ = std::max(peak_speed_, static_cast<double>(vmax));
        still_t_ = (vmax < c.settle_speed) ? still_t_ + dt : 0.0;
        // Never soften before the lowering path is done.
        const double path = LoweringDuration().load();
        const double min_t = std::max(c.settle_min_s, path + 0.2);
        const double max_t = std::max(c.settle_max_s, path + 2.0);
        const bool settled = t_ >= min_t && still_t_ >= c.settle_hold_s;
        if (settled || t_ >= max_t) {
          const double waited = t_;
          Start(State::kSoftening, std::clamp(c.kp_scale, 0.05, 1.0), c.soften_s);
          std::cout << "[SafeStop] Arms " << (settled ? "down and still" : "settle timeout")
                    << " after " << waited << " s (peak arm joint speed " << peak_speed_
                    << " rad/s) -> softening arms to Kp x" << to_ << " over " << c.soften_s << " s" << std::endl;
        }
        break;
      }
      case State::kSoftening:
      case State::kRestoring: {
        const double r = dur_ <= 0.0 ? 1.0 : std::clamp(t_ / dur_, 0.0, 1.0);
        const double s = r * r * r * (10.0 + r * (-15.0 + 6.0 * r));  // min-jerk
        scale_ = from_ + s * (to_ - from_);
        if (r >= 1.0) state_ = (state_ == State::kSoftening) ? State::kSoft : State::kIdle;
        break;
      }
      case State::kSoft: break;
    }
    return scale_;
  }

  State state() const { return state_; }
  double scale() const { return scale_; }

 private:
  void Start(State s, double to, double dur) {
    state_ = s; from_ = scale_; to_ = to; dur_ = dur; t_ = 0.0;
  }
  State state_ = State::kIdle;
  unsigned epoch_ = 0;
  double scale_ = 1.0, from_ = 1.0, to_ = 1.0, dur_ = 0.0, t_ = 0.0, still_t_ = 0.0, peak_speed_ = 0.0;
};

}  // namespace safe_stop
