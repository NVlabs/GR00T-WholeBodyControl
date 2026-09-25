/**
 * @file arm_compliance.hpp
 * @brief Runtime-switchable joint impedance (Kp/Kd) for the G1 arms.
 *
 * The SONIC policy outputs joint position targets at 50 Hz, and the motor
 * drivers close the loop with  tau = Kp (q* - q) + Kd (dq* - dq).  The default
 * Kp/Kd come from `policy_parameters.hpp` (kps / kds: Kp = J w^2, Kd = 2 zeta J w,
 * zeta = 2) and are constant.
 *
 * This layer rescales Kp/Kd on the 14 arm motors (hardware indices 15-28) while
 * the robot runs, without touching legs/waist.
 *
 * PROFILES
 *   A profile gives, for each side (left/right) and joint group (shoulder = 3
 *   joints, elbow, wrist = 3 joints), a stiffness scale alpha (Kp x alpha) and a
 *   damping scale beta (Kd x beta).  beta can be given directly ("kd") or through
 *   the damping ratio ("zeta"):  beta = sqrt(alpha) * zeta / 2   (nominal zeta = 2).
 *   If neither is given, zeta = 2 (same damping ratio as SONIC's defaults).
 *
 *   Spec format (JSON; the same format is used for the built-ins and --compliance-profiles):
 *     "HANDSHAKE": { "default": {"kp": 1.0},
 *                    "elbow":   {"kp": 0.6, "kd": 0.83} }
 *   Keys, later ones override earlier ones for the joints they cover:
 *     "default", "shoulder" | "elbow" | "wrist",
 *     "left" | "right", "left_shoulder" | ... | "right_wrist"
 *
 * TRANSITIONS (per joint)
 *   - Minimum-jerk shape  s(r) = 10r^3 - 15r^4 + 6r^5  (no kink at start or end).
 *   - Stiffening (Kp goes up) takes `stiffen_s` (default 1.0 s), softening
 *     `soften_s` (default 0.3 s).  A command's "slew_s" overrides both.
 *   - Damping stays on the high side: when Kd goes UP it finishes in the first
 *     `lead_frac` (40%) of the ramp; when Kp goes DOWN it finishes in the first
 *     40% while Kd follows over the full ramp.  So the joint is never briefly
 *     stiff-but-underdamped.
 *
 * ESTOP (controlled stop, default)
 *   1. RETRACT: from the first tick the arms stop following the policy (teleop /
 *      VLA / planner). Their targets move on a minimum-jerk path from the measured
 *      pose to the safe pose (the policy's default arm pose), with soft gains
 *      (retract_kp_scale, never stiffer than before).  Duration from the largest
 *      joint distance and retract_speed, clamped to [retract_min_s, retract_max_s].
 *   2. LIMP: targets held at the safe pose; gains ramp to estop_kp / estop_kd
 *      (absolute) over estop_ramp_s.  The state is LATCHED.
 *   3. RELEASE: only {"release_estop": true, "profile": ...}.  Targets blend from
 *      the measured pose back to the policy's targets and gains ramp to the new
 *      profile, both over estop_release_s.
 *   estop_mode = limp: skip 1, just ramp the gains (the policy keeps the arm targets).
 *
 * WATCHDOG
 *   If commands stop arriving, the last gains are HELD (never snapped back to
 *   rigid) and a warning is printed once.
 *
 * Commands arrive as JSON over ZMQ (see arm_compliance_subscriber.hpp):
 *   {"profile": "HANDSHAKE"}            {"profile": "HUG", "slew_s": 0.5}
 *   {"kp_scale": 0.4, "kd_scale": 0.6}  {"kp_scale": [14], "kd_scale": [14]}
 *   {"estop": true}                     {"release_estop": true, "profile": "RIGID"}
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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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

/// Joint group of arm index j (0..13): "shoulder", "elbow" or "wrist".
inline const char* GroupOf(int j) {
  const int k = j % 7;
  return k < 3 ? "shoulder" : (k == 3 ? "elbow" : "wrist");
}
/// Side of arm index j (0..13): "left" or "right".
inline const char* SideOf(int j) { return j < 7 ? "left" : "right"; }

/// Nominal damping ratio of SONIC's default gains (policy_parameters.hpp).
constexpr double kNominalZeta = 2.0;

/// Validation limits for scale factors.
constexpr float kMaxKpScale = 1.5f;
constexpr float kMaxKdScale = 3.0f;

/// A named set of per-joint scale factors applied to the nominal kps/kds.
struct Profile {
  std::string name;
  ArmArray kp_scale;
  ArmArray kd_scale;
};

/// Resolve one {"kp", "kd" | "zeta"} entry into (alpha, beta).
inline bool ResolveGainEntry(const nlohmann::json& e, float& alpha, float& beta, std::string& err) {
  if (!e.is_object()) { err = "gain entry must be an object"; return false; }
  alpha = 1.0f;
  if (e.contains("kp")) {
    if (!e["kp"].is_number()) { err = "'kp' must be a number"; return false; }
    alpha = e["kp"].get<float>();
  }
  if (!std::isfinite(alpha) || alpha < 0.0f || alpha > kMaxKpScale) {
    err = "'kp' scale must be within [0, 1.5]";
    return false;
  }
  if (e.contains("kd") && e.contains("zeta")) { err = "give either 'kd' or 'zeta', not both"; return false; }
  if (e.contains("kd")) {
    if (!e["kd"].is_number()) { err = "'kd' must be a number"; return false; }
    beta = e["kd"].get<float>();
  } else {
    double zeta = kNominalZeta;
    if (e.contains("zeta")) {
      if (!e["zeta"].is_number()) { err = "'zeta' must be a number"; return false; }
      zeta = e["zeta"].get<double>();
    }
    beta = static_cast<float>(std::sqrt(static_cast<double>(alpha)) * zeta / kNominalZeta);
  }
  if (!std::isfinite(beta) || beta < 0.0f || beta > kMaxKdScale) {
    err = "'kd' scale must be within [0, 3]";
    return false;
  }
  return true;
}

/// Build a Profile from a spec object (see file header for the format).
inline bool BuildProfile(const std::string& name, const nlohmann::json& spec, Profile& out, std::string& err) {
  if (!spec.is_object()) { err = "profile '" + name + "' must be an object"; return false; }
  static const std::vector<std::string> kKnownKeys = {
      "default", "shoulder", "elbow", "wrist", "left", "right",
      "left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
      "description"};
  for (auto it = spec.begin(); it != spec.end(); ++it) {
    if (std::find(kKnownKeys.begin(), kKnownKeys.end(), it.key()) == kKnownKeys.end()) {
      err = "profile '" + name + "': unknown key '" + it.key() + "'";
      return false;
    }
  }
  out.name = name;
  out.kp_scale.fill(1.0f);
  out.kd_scale.fill(1.0f);
  for (int j = 0; j < kNumArmMotors; ++j) {
    const std::string group = GroupOf(j), side = SideOf(j);
    // Most generic first; later keys override.
    for (const std::string& key : {std::string("default"), group, side, side + "_" + group}) {
      if (!spec.contains(key)) continue;
      float a, b;
      std::string e;
      if (!ResolveGainEntry(spec[key], a, b, e)) {
        err = "profile '" + name + "', '" + key + "': " + e;
        return false;
      }
      out.kp_scale[j] = a;
      out.kd_scale[j] = b;
    }
  }
  return true;
}

/**
 * Built-in profiles (Step 0 tests on the real robot with sonic_v1_1).
 *   HANDSHAKE  soft elbows (0.6 / 0.83) — rated better than rigid.
 *   HUG        all arm joints 0.6 / 0.83 — "maybe", to be confirmed.
 *   FISTBUMP   rigid — only rigid felt natural (pilot, n = 2).
 *   FISTBUMP_SOFTWRIST  right wrist 0.5 / 0.7 — candidate for the study.
 *   SOFT       all arm joints 0.25 / 0.5 (zeta = 2).
 */
inline const char* BuiltinProfilesJson() {
  return R"JSON({
    "RIGID":              {"default": {"kp": 1.0, "kd": 1.0}},
    "HANDSHAKE":          {"default": {"kp": 1.0, "kd": 1.0}, "elbow": {"kp": 0.6, "kd": 0.83}},
    "HUG":                {"default": {"kp": 0.6, "kd": 0.83}},
    "FISTBUMP":           {"default": {"kp": 1.0, "kd": 1.0}},
    "FISTBUMP_SOFTWRIST": {"default": {"kp": 1.0, "kd": 1.0}, "right_wrist": {"kp": 0.5, "kd": 0.7}},
    "SOFT":               {"default": {"kp": 0.25, "kd": 0.5}}
  })JSON";
}

/// Profile names that are accepted as aliases (old names).
inline std::string CanonicalProfileName(const std::string& name) {
  if (name == "P0") return "RIGID";
  return name;
}

/**
 * @class ProfileRegistry
 * @brief Built-in profiles plus optional ones from a JSON file (file entries override).
 *        Immutable after construction, so it can be read from any thread.
 */
class ProfileRegistry {
  public:
    ProfileRegistry() {
      std::string err;
      if (!AddFromJson(nlohmann::json::parse(BuiltinProfilesJson()), "built-in", err)) {
        std::cerr << "[ArmCompliance] BUG in built-in profiles: " << err << std::endl;
      }
    }

    /// Load additional profiles.  File format: {"profiles": {NAME: spec, ...}} or {NAME: spec, ...}.
    bool LoadFile(const std::string& path, std::string& err) {
      std::ifstream f(path);
      if (!f) { err = "cannot open " + path; return false; }
      nlohmann::json j;
      try {
        f >> j;
      } catch (const std::exception& e) {
        err = path + ": invalid JSON: " + e.what();
        return false;
      }
      if (j.contains("profiles")) j = j["profiles"];
      return AddFromJson(j, path, err);
    }

    const Profile* Find(const std::string& name) const {
      auto it = profiles_.find(CanonicalProfileName(name));
      return it == profiles_.end() ? nullptr : &it->second;
    }

    std::vector<std::string> Names() const {
      std::vector<std::string> n;
      for (const auto& kv : profiles_) n.push_back(kv.first);
      return n;
    }

  private:
    bool AddFromJson(const nlohmann::json& j, const std::string& source, std::string& err) {
      if (!j.is_object()) { err = source + ": profiles must be a JSON object"; return false; }
      std::map<std::string, Profile> staged;
      for (auto it = j.begin(); it != j.end(); ++it) {
        if (it.key() == "ESTOP" || it.key() == "custom") {
          err = source + ": '" + it.key() + "' is a reserved name";
          return false;
        }
        Profile p;
        if (!BuildProfile(it.key(), it.value(), p, err)) { err = source + ": " + err; return false; }
        staged[it.key()] = p;
      }
      for (auto& kv : staged) profiles_[kv.first] = kv.second;
      return true;
    }

    std::map<std::string, Profile> profiles_;
};

/// What ESTOP does with the arms.
enum class EstopMode {
  kAuto,     ///< Hand-target handoff when the policy can bring the arms down itself, else retract.
  kRetract,  ///< Always override the arm targets: retract to the safe pose, then go limp.
  kLimp,     ///< Only ramp the arm gains down (arms keep the policy's targets).
};

/// Static configuration (set once from the command line).
struct Config {
  bool enabled = false;               ///< Master switch; when false Apply() is a no-op.
  std::string host = "localhost";     ///< Host of the command publisher.
  int port = 5565;                    ///< Port of the command publisher.
  std::string topic = "compliance";   ///< ZMQ topic prefix.
  std::string profiles_file;          ///< Optional JSON file with extra profiles.
  std::string initial_profile = "RIGID"; ///< Profile at start-up.
  double soften_s = 0.3;              ///< Ramp time when a joint gets softer (Kp down).
  double stiffen_s = 1.0;             ///< Ramp time when a joint gets stiffer (Kp up).
  double lead_frac = 0.4;             ///< Fraction of the ramp for the "leading" gain (see header).
  double watchdog_s = 1.0;            ///< Warn (and hold) if no command for this long.

  // --- ESTOP ---
  EstopMode estop_mode = EstopMode::kAuto;
  double handoff_s = 1.5;             ///< Hand-target blend time (operator -> idle) in handoff ESTOP.
  /// Arm pose the retract goes to (hardware order 15..28). Set from the policy's
  /// default_angles by the deploy binary; zeros here are only a placeholder.
  ArmArray safe_pose{};
  double retract_speed = 0.8;         ///< Peak joint speed of the retract (rad/s, ~45 deg/s).
  double retract_min_s = 0.8;         ///< Retract duration limits (s).
  double retract_max_s = 3.0;
  float retract_kp_scale = 0.6f;      ///< Stiffness while retracting (x nominal Kp); Kd x sqrt().
  double estop_ramp_s = 1.0;          ///< Ramp to the ESTOP gains (after the retract, or directly in limp mode).
  double estop_release_s = 1.0;       ///< Blend back to the policy after release.
  float estop_kp = 1.5f;              ///< Arm Kp during ESTOP (absolute, Nm/rad; nominal ~14.3).
  float estop_kd = 0.9f;              ///< Arm Kd during ESTOP (absolute, Nm*s/rad; nominal ~0.9).
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
inline bool ReadScale(const nlohmann::json& j, ArmArray& out, float max_value, std::string& err) {
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
    if (!std::isfinite(v) || v < 0.0f || v > max_value) {
      std::ostringstream os;
      os << "scale values must be finite and within [0, " << max_value << "]";
      err = os.str();
      return false;
    }
  }
  return true;
}

/**
 * Parse a JSON command.  Returns false (and fills `err`) on malformed input;
 * malformed commands are ignored by the controller.
 */
inline bool ParseCommand(const std::string& text, const ProfileRegistry& registry, Command& cmd, std::string& err) {
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
    const Profile* p = registry.Find(profile);
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
    if (!ReadScale(j["kp_scale"], cmd.kp_scale, kMaxKpScale, err)) return false;
    if (!ReadScale(j["kd_scale"], cmd.kd_scale, kMaxKdScale, err)) return false;
    cmd.name = "custom";
    return true;
  }

  err = "command needs 'profile', 'estop', or 'kp_scale'+'kd_scale'";
  return false;
}

/// Minimum-jerk time scaling: s(0)=0, s(1)=1, zero 1st and 2nd derivative at both ends.
inline double MinJerk(double r) {
  r = std::clamp(r, 0.0, 1.0);
  return r * r * r * (10.0 + r * (-15.0 + 6.0 * r));
}

/**
 * @class Controller
 * @brief Holds the target arm gains and ramps the applied gains toward them;
 *        during ESTOP it also owns the arm position targets (controlled stop).
 *
 * ESTOP, handoff kind (default when the policy has live hand targets or runs the planner):
 *   HANDOFF  the policy's hand targets (vr_3point observations) blend from the
 *            operator's hands to the reference/planner hands (arms down) over
 *            handoff_s — the POLICY brings the arms down and keeps balance; arm
 *            gains soften to the retract stiffness.  No target override.
 *   LIMP     hand targets = reference; gains ramp to estop_kp / estop_kd.
 *   RELEASE  hand targets blend back to the operator; gains ramp to the profile.
 *   Read the blend with HandTargetBlend() when building the observations.
 *
 * ESTOP, retract kind (fallback: reference motion clips, full-body POSE streaming):
 *   RETRACT  arm targets follow a minimum-jerk path from the MEASURED arm pose to
 *            cfg.safe_pose; gains go to the retract stiffness.  The policy's arm
 *            targets (teleop / VLA / planner) are ignored from the first tick.
 *   LIMP     arm targets held at the safe pose; gains ramp to estop_kp / estop_kd.
 *   RELEASE  after {"release_estop": true}: arm targets blend from the measured
 *            pose back to the policy's targets; gains ramp to the new profile.
 */
class Controller {
  public:
    enum class Phase { kNormal, kHandoff, kRetract, kLimp, kRelease };
    enum class EstopKind { kNone, kHandoff, kRetract, kLimp };

    explicit Controller(const Config& cfg = Config{}) : cfg_(cfg) {
      if (!cfg_.profiles_file.empty()) {
        std::string err;
        if (registry_.LoadFile(cfg_.profiles_file, err)) {
          std::cout << "[ArmCompliance] Loaded profiles from " << cfg_.profiles_file << std::endl;
        } else {
          std::cerr << "[ArmCompliance] ERROR loading profiles: " << err
                    << " — using built-in profiles only." << std::endl;
        }
      }
      const Profile* p = registry_.Find(cfg_.initial_profile);
      if (!p) {
        std::cerr << "[ArmCompliance] Unknown initial profile '" << cfg_.initial_profile
                  << "', using RIGID." << std::endl;
        p = registry_.Find("RIGID");
      }
      target_.name = p->name;
      target_.kp_scale = p->kp_scale;
      target_.kd_scale = p->kd_scale;
      target_version_ = 1;
    }

    const Config& config() const { return cfg_; }
    const ProfileRegistry& profiles() const { return registry_; }
    bool enabled() const { return cfg_.enabled; }

    /// Parse a JSON command against this controller's profiles.
    bool Parse(const std::string& text, Command& cmd, std::string& err) const {
      return ParseCommand(text, registry_, cmd, err);
    }

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
     * @brief Update arm targets and gains in-place (control thread, 50 Hz).
     * @param q_target 29-element position targets from the policy (overwritten on
     *                 the arms during ESTOP / release).
     * @param kp       29-element Kp array filled with the nominal gains.
     * @param kd       29-element Kd array filled with the nominal gains.
     * @param q_meas   Measured arm joint positions (hardware 15..28).
     * @param dt       Control period in seconds.
     * @param policy_can_retract  true if the policy's hand targets can be handed over to a
     *                 reference that has the arms down (live VR hand targets, or planner running).
     */
    template <size_t N>
    void Apply(std::array<float, N>& q_target, std::array<float, N>& kp, std::array<float, N>& kd,
               const ArmArray& q_meas, double dt, bool policy_can_retract = false) {
      static_assert(N >= kFirstArmMotor + kNumArmMotors, "array too small");
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

      // Nominal arm gains this tick (as filled in by the caller).
      ArmArray nom_kp{}, nom_kd{};
      for (int j = 0; j < kNumArmMotors; ++j) {
        nom_kp[j] = kp[kFirstArmMotor + j];
        nom_kd[j] = kd[kFirstArmMotor + j];
      }

      if (!initialized_) {
        // First policy tick: start from the nominal gains and ramp to the requested profile.
        current_kp_ = nom_kp;
        current_kd_ = nom_kd;
        applied_version_ = 0;
        initialized_ = true;
      }

      // ---- React to a new command ----
      if (version != applied_version_) {
        applied_version_ = version;
        if (target.estop) {
          if (cfg_.estop_mode == EstopMode::kAuto && policy_can_retract) {
            EnterHandoff(nom_kp, nom_kd);
          } else if (cfg_.estop_mode != EstopMode::kLimp) {
            EnterRetract(q_meas, nom_kp, nom_kd);
          } else {
            estop_kind_ = EstopKind::kLimp;
            phase_ = Phase::kLimp;
            StartGainRamp(EstopKp(), EstopKd(), target.slew_s.value_or(cfg_.estop_ramp_s), true);
            std::cout << "[ArmCompliance] -> ESTOP (limp mode: arms Kp=" << cfg_.estop_kp
                      << " Kd=" << cfg_.estop_kd << ", ramp " << ramp_duration_ << " s, latched)" << std::endl;
          }
        } else if (phase_ == Phase::kHandoff || phase_ == Phase::kRetract || phase_ == Phase::kLimp) {
          // Release: blend back to the policy / operator (from wherever the blend is now).
          blend_start_ = hand_blend_.load(std::memory_order_relaxed);
          phase_ = Phase::kRelease;
          phase_elapsed_ = 0.0;
          phase_duration_ = std::max(cfg_.estop_release_s, 1e-3);
          phase_start_q_ = q_meas;
          StartGainRamp(ProfileKp(target, nom_kp), ProfileKd(target, nom_kd), cfg_.estop_release_s, true);
          std::cout << "[ArmCompliance] ESTOP released -> " << target.name << " (blend back to policy over "
                    << std::fixed << std::setprecision(2) << phase_duration_ << " s)" << std::endl;
        } else {
          // Normal profile change (also while a release blend is still running).
          StartProfileRamp(target, ProfileKp(target, nom_kp), ProfileKd(target, nom_kd));
        }
      }

      // ---- Phase progression ----
      if (phase_ == Phase::kHandoff || phase_ == Phase::kRetract || phase_ == Phase::kRelease) phase_elapsed_ += dt;
      if ((phase_ == Phase::kHandoff || phase_ == Phase::kRetract) && phase_elapsed_ >= phase_duration_) {
        phase_ = Phase::kLimp;
        StartGainRamp(EstopKp(), EstopKd(), cfg_.estop_ramp_s, true);
        std::cout << "[ArmCompliance] ESTOP: arms down, ramping to Kp=" << cfg_.estop_kp
                  << " Kd=" << cfg_.estop_kd << " over " << cfg_.estop_ramp_s << " s (latched)" << std::endl;
      }
      if (phase_ == Phase::kRelease && phase_elapsed_ >= phase_duration_) {
        phase_ = Phase::kNormal;
        estop_kind_ = EstopKind::kNone;
      }

      // ---- Hand-target blend for the observations (handoff kind only) ----
      {
        float w = 0.0f;
        if (estop_kind_ == EstopKind::kHandoff) {
          const double r = phase_elapsed_ / std::max(phase_duration_, 1e-6);
          if (phase_ == Phase::kHandoff) w = static_cast<float>(blend_start_ + (1.0 - blend_start_) * MinJerk(r));
          else if (phase_ == Phase::kLimp) w = 1.0f;
          else if (phase_ == Phase::kRelease) w = static_cast<float>(blend_start_ * (1.0 - MinJerk(r)));
        }
        hand_blend_.store(w, std::memory_order_relaxed);
      }

      // ---- Gain target for this tick ----
      ArmArray goal_kp{}, goal_kd{};
      switch (phase_) {
        case Phase::kHandoff:
        case Phase::kRetract: goal_kp = RetractKp(nom_kp); goal_kd = RetractKd(nom_kd); break;
        case Phase::kLimp:    goal_kp = EstopKp();         goal_kd = EstopKd();         break;
        default:              goal_kp = ProfileKp(target, nom_kp); goal_kd = ProfileKd(target, nom_kd); break;
      }
      AdvanceGainRamp(goal_kp, goal_kd, dt);

      // ---- Arm position targets ----
      // (In limp mode the arms keep the policy's targets while in ESTOP; only the release blends.)
      const bool own_targets =
          estop_kind_ == EstopKind::kRetract &&
          (phase_ == Phase::kRetract || phase_ == Phase::kLimp || phase_ == Phase::kRelease);
      if (own_targets) {
        const double s = MinJerk(phase_elapsed_ / phase_duration_);
        for (int j = 0; j < kNumArmMotors; ++j) {
          float& q = q_target[kFirstArmMotor + j];
          if (phase_ == Phase::kRetract) {
            q = static_cast<float>(phase_start_q_[j] + s * (cfg_.safe_pose[j] - phase_start_q_[j]));
          } else if (phase_ == Phase::kLimp) {
            q = cfg_.safe_pose[j];
          } else {  // release: from measured pose at release to the policy's (moving) target
            q = static_cast<float>(phase_start_q_[j] + s * (q - phase_start_q_[j]));
          }
        }
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

    /// Name of the current target ("RIGID", "HUG", ..., "custom", "ESTOP").
    std::string CurrentTargetName() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return target_.name;
    }

    bool EstopLatched() const {
      std::lock_guard<std::mutex> lock(mutex_);
      return estop_latched_;
    }

    /// Control-thread-only views (valid after the first Apply()).
    ArmArray CurrentKp() const { return current_kp_; }
    ArmArray CurrentKd() const { return current_kd_; }
    Phase CurrentPhase() const { return phase_; }
    EstopKind CurrentEstopKind() const { return estop_kind_; }
    /// 0 = policy sees the operator's hand targets, 1 = the reference's (arms down).
    float HandTargetBlend() const { return hand_blend_.load(std::memory_order_relaxed); }
    double PhaseDuration() const { return phase_duration_; }

  private:
    // ----- gain targets -----
    static ArmArray ProfileKp(const Command& t, const ArmArray& nom) {
      ArmArray a{}; for (int j = 0; j < kNumArmMotors; ++j) a[j] = nom[j] * t.kp_scale[j]; return a;
    }
    static ArmArray ProfileKd(const Command& t, const ArmArray& nom) {
      ArmArray a{}; for (int j = 0; j < kNumArmMotors; ++j) a[j] = nom[j] * t.kd_scale[j]; return a;
    }
    ArmArray RetractKp(const ArmArray& nom) const {
      ArmArray a{}; for (int j = 0; j < kNumArmMotors; ++j) a[j] = nom[j] * cfg_.retract_kp_scale; return a;
    }
    ArmArray RetractKd(const ArmArray& nom) const {
      const float b = std::sqrt(std::max(cfg_.retract_kp_scale, 0.0f));  // keeps zeta
      ArmArray a{}; for (int j = 0; j < kNumArmMotors; ++j) a[j] = nom[j] * b; return a;
    }
    ArmArray EstopKp() const { ArmArray a{}; a.fill(cfg_.estop_kp); return a; }
    ArmArray EstopKd() const { ArmArray a{}; a.fill(cfg_.estop_kd); return a; }

    // ----- ESTOP -----
    void EnterHandoff(const ArmArray& nom_kp, const ArmArray& nom_kd) {
      // Continue from the current blend (e.g. ESTOP again during a release).
      blend_start_ = hand_blend_.load(std::memory_order_relaxed);
      estop_kind_ = EstopKind::kHandoff;
      phase_ = Phase::kHandoff;
      phase_elapsed_ = 0.0;
      phase_duration_ = std::max(cfg_.handoff_s, 1e-3);
      ArmArray rk = RetractKp(nom_kp);
      for (int j = 0; j < kNumArmMotors; ++j) rk[j] = std::min(rk[j], current_kp_[j]);
      StartGainRamp(rk, RetractKd(nom_kd), cfg_.soften_s, true);
      std::cout << "[ArmCompliance] -> ESTOP: handing the policy's hand targets to the idle pose over "
                << std::fixed << std::setprecision(2) << phase_duration_ << " s (soft arms), then Kp="
                << cfg_.estop_kp << " Kd=" << cfg_.estop_kd << " (latched)" << std::endl;
    }

    void EnterRetract(const ArmArray& q_meas, const ArmArray& nom_kp, const ArmArray& nom_kd) {
      estop_kind_ = EstopKind::kRetract;
      phase_ = Phase::kRetract;
      phase_elapsed_ = 0.0;
      phase_start_q_ = q_meas;
      double dmax = 0.0;
      for (int j = 0; j < kNumArmMotors; ++j) dmax = std::max(dmax, static_cast<double>(std::fabs(cfg_.safe_pose[j] - q_meas[j])));
      // Minimum-jerk peak speed = 1.875 * distance / duration.
      const double speed = std::max(cfg_.retract_speed, 1e-3);
      phase_duration_ = std::clamp(1.875 * dmax / speed, cfg_.retract_min_s, cfg_.retract_max_s);
      // Soften quickly toward the retract stiffness (never stiffer than now).
      ArmArray rk = RetractKp(nom_kp);
      const ArmArray rd = RetractKd(nom_kd);
      for (int j = 0; j < kNumArmMotors; ++j) rk[j] = std::min(rk[j], current_kp_[j]);
      StartGainRamp(rk, rd, cfg_.soften_s, true);
      std::cout << "[ArmCompliance] -> ESTOP: retracting arms to safe pose over " << std::fixed
                << std::setprecision(2) << phase_duration_ << " s (max joint travel " << dmax
                << " rad), then Kp=" << cfg_.estop_kp << " Kd=" << cfg_.estop_kd << " (latched)" << std::endl;
    }

    // ----- gain ramps -----
    /// Profile change: per-joint durations (stiffen/soften) with the lead rule.
    void StartProfileRamp(const Command& target, const ArmArray& goal_kp, const ArmArray& goal_kd) {
      start_kp_ = current_kp_;
      start_kd_ = current_kd_;
      ramp_elapsed_ = 0.0;
      const double lead = std::clamp(cfg_.lead_frac, 0.0, 1.0);
      double longest = 0.0;
      for (int j = 0; j < kNumArmMotors; ++j) {
        const bool kp_up = goal_kp[j] > start_kp_[j];
        const bool kd_up = goal_kd[j] > start_kd_[j];
        const double T = target.slew_s ? *target.slew_s : (kp_up ? cfg_.stiffen_s : cfg_.soften_s);
        // Keep damping on the high side: Kd leads when rising, Kp leads when falling.
        dur_kp_[j] = kp_up ? T : T * lead;
        dur_kd_[j] = kd_up ? T * lead : T;
        longest = std::max(longest, T);
      }
      ramp_duration_ = longest;
      ramping_ = true;
      LogProfile(target, longest);
    }

    /// Fixed-duration ramp of all joints (ESTOP phases).  `symmetric`: Kp and Kd over the full time.
    void StartGainRamp(const ArmArray& goal_kp, const ArmArray& goal_kd, double T, bool symmetric) {
      (void)goal_kp; (void)goal_kd;
      start_kp_ = current_kp_;
      start_kd_ = current_kd_;
      ramp_elapsed_ = 0.0;
      ramp_duration_ = std::max(T, 0.0);
      for (int j = 0; j < kNumArmMotors; ++j) {
        dur_kp_[j] = ramp_duration_;
        dur_kd_[j] = symmetric ? ramp_duration_ : ramp_duration_ * cfg_.lead_frac;
      }
      ramping_ = true;
      phase_goal_kp_ = goal_kp;
      phase_goal_kd_ = goal_kd;
    }

    void AdvanceGainRamp(const ArmArray& goal_kp, const ArmArray& goal_kd, double dt) {
      // During ESTOP phases, clamp the goal used by EnterRetract (never stiffer than at ESTOP time).
      const bool use_phase_goal = (phase_ == Phase::kRetract || phase_ == Phase::kHandoff);
      const ArmArray& gk = use_phase_goal ? phase_goal_kp_ : goal_kp;
      const ArmArray& gd = use_phase_goal ? phase_goal_kd_ : goal_kd;
      if (!ramping_) {
        current_kp_ = gk;
        current_kd_ = gd;
        return;
      }
      ramp_elapsed_ += dt;
      bool done = true;
      for (int j = 0; j < kNumArmMotors; ++j) {
        const double sp = dur_kp_[j] <= 0.0 ? 1.0 : MinJerk(ramp_elapsed_ / dur_kp_[j]);
        const double sd = dur_kd_[j] <= 0.0 ? 1.0 : MinJerk(ramp_elapsed_ / dur_kd_[j]);
        current_kp_[j] = static_cast<float>(start_kp_[j] + sp * (gk[j] - start_kp_[j]));
        current_kd_[j] = static_cast<float>(start_kd_[j] + sd * (gd[j] - start_kd_[j]));
        if (sp < 1.0 || sd < 1.0) done = false;
      }
      if (done) ramping_ = false;
    }

    void LogProfile(const Command& t, double ramp) const {
      auto side = [&](int o) {
        std::ostringstream s;
        s << std::fixed << std::setprecision(2) << "S " << t.kp_scale[o] << "/" << t.kd_scale[o]
          << " E " << t.kp_scale[o + 3] << "/" << t.kd_scale[o + 3]
          << " W " << t.kp_scale[o + 4] << "/" << t.kd_scale[o + 4];
        return s.str();
      };
      std::ostringstream os;
      os << std::fixed << std::setprecision(2) << "[ArmCompliance] -> " << t.name
         << " (Kp/Kd scale  L[" << side(0) << "]  R[" << side(7) << "], ramp " << ramp << " s)";
      std::cout << os.str() << std::endl;
    }

    Config cfg_;
    ProfileRegistry registry_;

    mutable std::mutex mutex_;
    Command target_;
    uint64_t target_version_ = 0;
    bool estop_latched_ = false;
    bool warned_latched_ = false;
    bool has_received_ = false;
    std::chrono::steady_clock::time_point last_command_time_{};

    // Control-thread state (only touched in Apply()).
    bool initialized_ = false;
    bool watchdog_warned_ = false;
    uint64_t applied_version_ = 0;
    Phase phase_ = Phase::kNormal;
    EstopKind estop_kind_ = EstopKind::kNone;
    std::atomic<float> hand_blend_{0.0f};
    double blend_start_ = 0.0;
    double phase_elapsed_ = 0.0;
    double phase_duration_ = 1.0;
    ArmArray phase_start_q_{};
    ArmArray phase_goal_kp_{}, phase_goal_kd_{};
    bool ramping_ = false;
    double ramp_elapsed_ = 0.0;
    double ramp_duration_ = 0.0;
    std::array<double, kNumArmMotors> dur_kp_{}, dur_kd_{};
    ArmArray start_kp_{}, start_kd_{};
    ArmArray current_kp_{}, current_kd_{};
};

}  // namespace arm_compliance
