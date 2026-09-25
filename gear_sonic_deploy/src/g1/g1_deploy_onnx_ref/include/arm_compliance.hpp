/**
 * @file arm_compliance.hpp
 * @brief Runtime-switchable joint impedance (Kp/Kd) for the G1 arms.
 *
 * The SONIC policy outputs joint position targets at 50 Hz, and the motor
 * drivers close the loop with  tau = Kp (q* - q) + Kd (dq* - dq).  The default
 * Kp/Kd come from `policy_parameters.hpp` (kps / kds) and are constant.
 *
 * This layer rescales Kp/Kd on the 14 arm motors (hardware indices 15-28:
 * shoulders, elbows, wrists) while the robot runs, without touching legs/waist.
 *
 *   - Profiles: P0 rigid, P1 soft, P2 compliant (per-joint Kp/Kd scale factors),
 *     or arbitrary per-joint scales sent in a command.
 *   - Smooth transitions: gains are ramped linearly over `slew_s` (default 0.3 s).
 *   - ESTOP: arms ramp to Kp = estop_kp, Kd = estop_kd over `estop_ramp_s`
 *     (defaults 0 / 8 / 0 s = immediate; e.g. 0 / 0 / 0.5 s for a smooth "go limp"),
 *     and the state is LATCHED.  Only a command with "release_estop": true leaves it,
 *     ramping to the requested profile over `estop_release_s` (default 1.0 s).
 *   - Watchdog: if commands stop arriving, the last gains are HELD (never snapped
 *     back to rigid) and a warning is printed once.
 *
 * Commands arrive as JSON over ZMQ (see ArmComplianceSubscriber below):
 *
 *   "compliance {\"profile\": \"P1\"}"
 *   "compliance {\"profile\": \"P2\", \"slew_s\": 0.5}"
 *   "compliance {\"kp_scale\": 0.4, \"kd_scale\": 0.6}"          (all 14 arm joints)
 *   "compliance {\"kp_scale\": [14 values], \"kd_scale\": [14 values]}"
 *   "compliance {\"estop\": true}"
 *   "compliance {\"release_estop\": true, \"profile\": \"P0\"}"
 *
 * The publisher should resend its current command periodically (e.g. 10 Hz) so
 * the watchdog can tell a live link from a dead one.  Re-sending an identical
 * command does not restart the ramp.
 *
 * Thread safety: SetCommand() may be called from any thread (ZMQ thread);
 * Apply() is called from the 50 Hz control thread.
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

namespace arm_compliance {

/// First hardware motor index of the arms (left_shoulder_pitch).
constexpr int kFirstArmMotor = 15;
/// Number of arm motors (7 per arm: 3 shoulder, 1 elbow, 3 wrist).
constexpr int kNumArmMotors = 14;

using ArmArray = std::array<float, kNumArmMotors>;

/// Joint names in the order used by all 14-element arrays (hardware 15..28).
inline const std::array<const char*, kNumArmMotors>& ArmJointNames() {
  static const std::array<const char*, kNumArmMotors> names = {
      "L_shoulder_pitch", "L_shoulder_roll", "L_shoulder_yaw", "L_elbow",
      "L_wrist_roll",     "L_wrist_pitch",   "L_wrist_yaw",
      "R_shoulder_pitch", "R_shoulder_roll", "R_shoulder_yaw", "R_elbow",
      "R_wrist_roll",     "R_wrist_pitch",   "R_wrist_yaw"};
  return names;
}

/// Build a 14-element array: `proximal` for shoulder+elbow, `wrist` for the 3 wrist joints.
inline ArmArray SplitArm(float proximal, float wrist) {
  ArmArray a{};
  for (int side = 0; side < 2; ++side) {
    const int o = side * 7;
    for (int j = 0; j < 4; ++j) a[o + j] = proximal;  // shoulder pitch/roll/yaw, elbow
    for (int j = 4; j < 7; ++j) a[o + j] = wrist;     // wrist roll/pitch/yaw
  }
  return a;
}

/// A named set of per-joint scale factors applied to the nominal kps/kds.
struct Profile {
  std::string name;
  ArmArray kp_scale;
  ArmArray kd_scale;
};

/**
 * Built-in profiles.  STARTING VALUES — tune them in simulation.
 *
 * Kd scale ~ sqrt(Kp scale) keeps the damping ratio of each joint roughly constant
 * (zeta = Kd / (2 sqrt(Kp J))).  Wrists keep more stiffness than shoulders/elbows
 * because uniform scaling made them sluggish in the first sim test.
 *
 * Note: tau_ff is 0 in this stack, so gravity is held only by Kp * (q* - q).
 * Low Kp on the shoulders/elbows means visible sag unless the policy compensates.
 */
inline const std::vector<Profile>& BuiltinProfiles() {
  static const std::vector<Profile> profiles = {
      {"P0", SplitArm(1.00f, 1.00f), SplitArm(1.00f, 1.00f)},  // rigid (SONIC default gains)
      {"P1", SplitArm(0.50f, 0.80f), SplitArm(0.71f, 0.90f)},  // soft
      {"P2", SplitArm(0.25f, 0.60f), SplitArm(0.50f, 0.77f)},  // compliant
  };
  return profiles;
}

inline const Profile* FindProfile(const std::string& name) {
  for (const auto& p : BuiltinProfiles()) {
    if (p.name == name) return &p;
  }
  return nullptr;
}

/// Static configuration (set once from the command line).
struct Config {
  bool enabled = false;               ///< Master switch; when false Apply() is a no-op.
  std::string host = "localhost";     ///< Host of the command publisher.
  int port = 5565;                    ///< Port of the command publisher.
  std::string topic = "compliance";   ///< ZMQ topic prefix.
  std::string initial_profile = "P0"; ///< Profile at start-up.
  double slew_s = 0.3;                ///< Default ramp time between profiles.
  double estop_ramp_s = 0.0;          ///< Ramp time when entering ESTOP (0 = immediate).
  double estop_release_s = 1.0;       ///< Ramp time when leaving ESTOP.
  float estop_kp = 0.0f;              ///< Arm Kp during ESTOP (absolute, Nm/rad).
  float estop_kd = 8.0f;              ///< Arm Kd during ESTOP (absolute, Nm*s/rad).
  double watchdog_s = 1.0;            ///< Warn (and hold) if no command for this long.
};

/// One parsed command.
struct Command {
  bool estop = false;          ///< Enter ESTOP.
  bool release_estop = false;  ///< Allowed to leave a latched ESTOP.
  std::string name;            ///< Profile name, or "custom".
  ArmArray kp_scale{};
  ArmArray kd_scale{};
  std::optional<double> slew_s;

  bool SameTargetAs(const Command& o) const {
    return estop == o.estop && name == o.name && kp_scale == o.kp_scale && kd_scale == o.kd_scale;
  }
};

/// Read either a scalar (applied to all 14 joints) or a 14-element array.
inline bool ReadScale(const nlohmann::json& j, ArmArray& out, std::string& err) {
  if (j.is_number()) {
    out.fill(j.get<float>());
  } else if (j.is_array() && j.size() == kNumArmMotors) {
    for (int i = 0; i < kNumArmMotors; ++i) {
      if (!j[i].is_number()) { err = "scale array must contain numbers"; return false; }
      out[i] = j[i].get<float>();
    }
  } else {
    err = "scale must be a number or an array of 14 numbers";
    return false;
  }
  for (float v : out) {
    if (!std::isfinite(v) || v < 0.0f || v > 1.5f) {
      err = "scale values must be finite and within [0, 1.5]";
      return false;
    }
  }
  return true;
}

/**
 * Parse a JSON command.  Returns false (and fills `err`) on malformed input;
 * malformed commands are ignored by the controller.
 */
inline bool ParseCommand(const std::string& text, Command& cmd, std::string& err) {
  nlohmann::json j;
  try {
    j = nlohmann::json::parse(text);
  } catch (const std::exception& e) {
    err = std::string("invalid JSON: ") + e.what();
    return false;
  }
  if (!j.is_object()) { err = "command must be a JSON object"; return false; }

  cmd = Command{};
  cmd.release_estop = j.value("release_estop", false);
  if (j.contains("slew_s")) {
    if (!j["slew_s"].is_number()) { err = "slew_s must be a number"; return false; }
    const double s = j["slew_s"].get<double>();
    if (!(s >= 0.0 && s <= 10.0)) { err = "slew_s must be within [0, 10] s"; return false; }
    cmd.slew_s = s;
  }

  const bool estop = j.value("estop", false);
  std::string profile = j.value("profile", std::string());
  if (estop || profile == "ESTOP") {
    cmd.estop = true;
    cmd.name = "ESTOP";
    return true;
  }

  if (!profile.empty()) {
    const Profile* p = FindProfile(profile);
    if (!p) { err = "unknown profile '" + profile + "'"; return false; }
    cmd.name = p->name;
    cmd.kp_scale = p->kp_scale;
    cmd.kd_scale = p->kd_scale;
    return true;
  }

  if (j.contains("kp_scale") || j.contains("kd_scale")) {
    if (!j.contains("kp_scale") || !j.contains("kd_scale")) {
      err = "custom command needs both kp_scale and kd_scale";
      return false;
    }
    if (!ReadScale(j["kp_scale"], cmd.kp_scale, err)) return false;
    if (!ReadScale(j["kd_scale"], cmd.kd_scale, err)) return false;
    cmd.name = "custom";
    return true;
  }

  err = "command needs 'profile', 'estop', or 'kp_scale'+'kd_scale'";
  return false;
}

/**
 * @class Controller
 * @brief Holds the target arm gains and ramps the applied gains toward them.
 */
class Controller {
  public:
    explicit Controller(const Config& cfg = Config{}) : cfg_(cfg) {
      const Profile* p = FindProfile(cfg_.initial_profile);
      if (!p) p = FindProfile("P0");
      target_.name = p->name;
      target_.kp_scale = p->kp_scale;
      target_.kd_scale = p->kd_scale;
      target_version_ = 1;
    }

    const Config& config() const { return cfg_; }
    bool enabled() const { return cfg_.enabled; }

    /**
     * @brief Submit a new command (thread-safe).
     * @return true if the command changed the target.
     */
    bool SetCommand(const Command& cmd) {
      std::lock_guard<std::mutex> lock(mutex_);
      last_command_time_ = std::chrono::steady_clock::now();
      has_received_ = true;

      if (estop_latched_ && !cmd.estop && !cmd.release_estop) {
        if (!warned_latched_) {
          std::cout << "[ArmCompliance] ESTOP is latched; ignoring '" << cmd.name
                    << "'. Send {\"release_estop\": true, \"profile\": ...} to leave it." << std::endl;
          warned_latched_ = true;
        }
        return false;
      }
      if (cmd.SameTargetAs(target_)) return false;

      target_ = cmd;
      ++target_version_;
      if (cmd.estop) {
        estop_latched_ = true;
        warned_latched_ = false;
      } else {
        estop_latched_ = false;
      }
      return true;
    }

    /**
     * @brief Overwrite Kp/Kd of the arm motors in-place (control thread, 50 Hz).
     * @param kp  29-element Kp array already filled with the nominal gains.
     * @param kd  29-element Kd array already filled with the nominal gains.
     * @param dt  Control period in seconds.
     */
    template <size_t N>
    void Apply(std::array<float, N>& kp, std::array<float, N>& kd, double dt) {
      static_assert(N >= kFirstArmMotor + kNumArmMotors, "gain array too small");
      if (!cfg_.enabled) return;

      Command target;
      uint64_t version;
      bool stale = false;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        target = target_;
        version = target_version_;
        if (has_received_ && cfg_.watchdog_s > 0.0) {
          const double age = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - last_command_time_).count();
          stale = age > cfg_.watchdog_s;
        }
      }

      // Absolute target gains for this tick.
      ArmArray target_kp{}, target_kd{};
      for (int j = 0; j < kNumArmMotors; ++j) {
        const int m = kFirstArmMotor + j;
        if (target.estop) {
          target_kp[j] = cfg_.estop_kp;
          target_kd[j] = cfg_.estop_kd;
        } else {
          target_kp[j] = kp[m] * target.kp_scale[j];
          target_kd[j] = kd[m] * target.kd_scale[j];
        }
      }

      if (!initialized_) {
        // First policy tick: start from the nominal gains the policy was using,
        // then ramp to whatever profile is requested (never jump).
        for (int j = 0; j < kNumArmMotors; ++j) {
          current_kp_[j] = kp[kFirstArmMotor + j];
          current_kd_[j] = kd[kFirstArmMotor + j];
        }
        applied_version_ = 0;  // forces the "new target" branch below
        prev_was_estop_ = false;
        initialized_ = true;
      }
      if (version != applied_version_) {
        // New target: start a ramp from wherever we are now.
        applied_version_ = version;
        start_kp_ = current_kp_;
        start_kd_ = current_kd_;
        ramp_elapsed_ = 0.0;
        if (target.estop) {
          ramp_duration_ = target.slew_s.value_or(cfg_.estop_ramp_s);
        } else if (prev_was_estop_) {
          ramp_duration_ = cfg_.estop_release_s;
        } else {
          ramp_duration_ = target.slew_s.value_or(cfg_.slew_s);
        }
        ramping_ = true;
        LogTransition(target, ramp_duration_);
      }
      prev_was_estop_ = target.estop;

      if (ramping_) {
        ramp_elapsed_ += dt;
        const double r = (ramp_duration_ <= 0.0) ? 1.0 : std::clamp(ramp_elapsed_ / ramp_duration_, 0.0, 1.0);
        for (int j = 0; j < kNumArmMotors; ++j) {
          current_kp_[j] = static_cast<float>(start_kp_[j] + r * (target_kp[j] - start_kp_[j]));
          current_kd_[j] = static_cast<float>(start_kd_[j] + r * (target_kd[j] - start_kd_[j]));
        }
        if (r >= 1.0) ramping_ = false;
      } else {
        current_kp_ = target_kp;
        current_kd_ = target_kd;
      }

      for (int j = 0; j < kNumArmMotors; ++j) {
        kp[kFirstArmMotor + j] = current_kp_[j];
        kd[kFirstArmMotor + j] = current_kd_[j];
      }

      // Watchdog: hold the current gains, only report.
      if (stale && !watchdog_warned_) {
        std::cout << "[ArmCompliance] WARNING: no compliance command for > " << cfg_.watchdog_s
                  << " s. Holding current arm gains (" << target.name << ")." << std::endl;
        watchdog_warned_ = true;
      } else if (!stale && watchdog_warned_) {
        std::cout << "[ArmCompliance] Compliance commands resumed." << std::endl;
        watchdog_warned_ = false;
      }
    }

    /// Name of the current target ("P0", "P1", "P2", "custom", "ESTOP").
    std::string CurrentTargetName() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return target_.name;
    }

    bool EstopLatched() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return estop_latched_;
    }

    /// Currently applied arm gains (valid after the first Apply()).
    ArmArray CurrentKp() const { return current_kp_; }
    ArmArray CurrentKd() const { return current_kd_; }

  private:
    void LogTransition(const Command& t, double ramp) const {
      std::ostringstream os;
      os << std::fixed << std::setprecision(2);
      os << "[ArmCompliance] -> " << t.name;
      if (t.estop) {
        os << " (arms Kp=" << cfg_.estop_kp << " Kd=" << cfg_.estop_kd << ", ramp " << ramp << " s, latched)";
      } else {
        os << " (Kp x" << t.kp_scale[0] << "/" << t.kp_scale[4]
           << ", Kd x" << t.kd_scale[0] << "/" << t.kd_scale[4]
           << " shoulder-elbow/wrist, ramp " << ramp << " s)";
      }
      std::cout << os.str() << std::endl;
    }

    Config cfg_;

    mutable std::mutex mutex_;
    Command target_;
    uint64_t target_version_ = 0;
    bool estop_latched_ = false;
    bool warned_latched_ = false;
    bool has_received_ = false;
    std::chrono::steady_clock::time_point last_command_time_{};

    // Control-thread state (only touched in Apply()).
    bool initialized_ = false;
    bool ramping_ = false;
    bool prev_was_estop_ = false;
    bool watchdog_warned_ = false;
    uint64_t applied_version_ = 0;
    double ramp_elapsed_ = 0.0;
    double ramp_duration_ = 0.0;
    ArmArray start_kp_{}, start_kd_{};
    ArmArray current_kp_{}, current_kd_{};
};

}  // namespace arm_compliance
