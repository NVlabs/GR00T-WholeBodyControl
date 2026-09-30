/**
 * @file safe_stop.hpp
 * @brief "Safe stop": stop teleop / VLA and go to a soft ready stand.
 *
 * Independent of the arm-compliance layer (it can reuse its profiles).  Used with
 * --input-type zmq_manager (the deploy used for PICO teleop and the VLA).
 *
 * Sequence after a stop request:
 *   1. SOFTEN FIRST (control thread): the arm gains ramp (min-jerk, soften_s,
 *      default 0.5 s) to the safe-stop profile (--safe-stop-profile, default
 *      SOFT; any compliance profile, per joint).  Meanwhile the hand targets hold
 *      where the operator left them.
 *   2. LOWER (input thread, ZMQManager): the VR hand targets follow a smooth path
 *      to the rest pose; the POLICY follows it (like an operator slowly lowering
 *      the hands) and keeps its balance, now with soft arms.
 *        - both hands forward (a hug): the path first opens the hands outward
 *          (open_width) and then brings them down beside the body, so they do
 *          not sweep inward across the person's back;
 *        - otherwise (handshake, fist bump): straight to the rest pose.
 *      Peak hand speed hand_speed; duration lower_min_s..lower_max_s.
 *      Planner messages are ignored (walking forced to IDLE, no upper-body /
 *      hand targets); streamed full-body motion is switched to PLANNER first.
 *   3. HOLD soft at the rest pose.  LATCHED.
 *   RELEASE: gains ramp back (restore_s); the hands stay at the rest pose until
 *   the operator leaves teleop, then re-entering teleop works as usual.
 *
 * Triggers (any thread):
 *   - keyboard in the deploy terminal: k = stop, u = release   (ZMQManager)
 *   - ZMQ command topic: optional bool fields "safe_stop" / "safe_release"
 *   - voice node (gear_sonic/scripts/voice_safe_stop.py): its own PUB socket,
 *     topic "safety", same fields; deploy connects to <zmq-host>:<voice_port>
 *   - code: safe_stop::Request("reason") / safe_stop::Release("reason")
 *
 * The Unitree remote and the 'O' key remain the whole-robot emergency stop.
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <zmq.hpp>

namespace safe_stop {

constexpr int kNumArm = 14;  ///< Arm motors, hardware 15..28.

/// true while a safe stop is latched.
inline std::atomic<bool>& Active() {
  static std::atomic<bool> active{false};
  return active;
}

/// Incremented on every new stop.
inline std::atomic<unsigned>& Epoch() {
  static std::atomic<unsigned> epoch{0};
  return epoch;
}

/// Request a safe stop (idempotent while active).
inline void Request(const std::string& source) {
  if (!Active().exchange(true)) {
    Epoch().fetch_add(1);
    std::cout << "\n[SafeStop] STOP (" << source << "): teleop/VLA ignored; arms soften, then the hands "
              << "are lowered to the ready stand. Press u (or send safe_release) to release." << std::endl;
  }
}

/// Release a latched safe stop.
inline void Release(const std::string& source) {
  if (Active().exchange(false)) {
    std::cout << "\n[SafeStop] RELEASED (" << source << "): arm stiffness back to normal; robot stays in "
              << "idle. Leave and re-enter teleop / restart the VLA to continue." << std::endl;
  } else if (source != "voice") {  // the voice node sends a few copies of each command
    std::cout << "[SafeStop] (" << source << ") no safe stop active." << std::endl;
  }
}

/// Settings (from the command line).
struct Config {
  // Arm gains while stopped: per-joint scales of the default Kp / Kd (profile).
  std::string profile = "SOFT";
  std::array<float, kNumArm> kp_scale = [] { std::array<float, kNumArm> a{}; a.fill(0.25f); return a; }();
  std::array<float, kNumArm> kd_scale = [] { std::array<float, kNumArm> a{}; a.fill(0.5f); return a; }();
  double soften_s = 0.5;        ///< Soften first (hands hold) before the path starts.
  double restore_s = 1.0;       ///< Ramp back to normal stiffness on release.
  // Lowering path of the VR hand targets
  double hand_speed = 0.20;     ///< Peak hand speed (m/s).
  double lower_min_s = 2.0;     ///< Path duration limits (s).
  double lower_max_s = 6.0;
  double open_width = 0.25;     ///< Hug: extra outward opening of each hand (m); 0 = always straight.
  double forward_min = 0.12;    ///< Hug = both hands at least this far in front of the rest pose (m).
  std::string log_file;         ///< Optional CSV log of each stop (--safe-stop-log).
  int voice_port = 5570;        ///< Port of the voice node's PUB socket on --zmq-host (0 = off).
  int status_port = 5571;       ///< PUB of the stop state for the hand scripts (BrainCo); 0 = off.
};

inline Config& Settings() {
  static Config cfg;
  return cfg;
}

inline double MinJerk(double r) {
  r = std::clamp(r, 0.0, 1.0);
  return r * r * r * (10.0 + r * (-15.0 + 6.0 * r));
}

/// Rest pose of the VR 3-point targets = InputInterface defaults (arms down, idle).
/// Order: left wrist xyz, right wrist xyz, head xyz (root frame: x fwd, y left, z up).
inline constexpr std::array<double, 9> kRestVRPos = {0.0903, 0.1615, -0.2411, 0.1280, -0.1522, -0.2461,
                                                     0.0241, -0.0081, 0.4028};
inline constexpr std::array<double, 12> kRestVROrn = {0.7295, 0.3145, 0.5533, -0.2506, 0.7320, -0.2639,
                                                      0.5395, 0.3217, 0.9991, 0.011, 0.0402, -0.0002};

/**
 * @brief The lowering path of the three VR points (pure math, no threads).
 *
 * Wrists: cubic Bezier.  Hug: control points pushed outward (y) by open_width, so
 * the hands first move out and then come down beside the body.  Otherwise the
 * control points lie on the straight line (= straight path).  Head: straight.
 * Timing: hold for soften_s, then a minimum-jerk parameter over the duration.
 */
class HandPath {
 public:
  void Start(const std::array<double, 9>& pos, const std::array<double, 12>& orn, const Config& c) {
    p0_ = pos;
    q0_ = orn;
    hug_ = c.open_width > 0.0 && pos[0] > kRestVRPos[0] + c.forward_min && pos[3] > kRestVRPos[3] + c.forward_min;
    for (int h = 0; h < 2; ++h) {
      const double side = (h == 0) ? 1.0 : -1.0;  // left hand +y, right hand -y
      for (int k = 0; k < 3; ++k) {
        const double a = p0_[h * 3 + k], b = kRestVRPos[h * 3 + k];
        c1_[h * 3 + k] = a + (b - a) / 3.0;
        c2_[h * 3 + k] = a + 2.0 * (b - a) / 3.0;
      }
      if (hug_) {
        const double wide = std::max(std::fabs(p0_[h * 3 + 1]), std::fabs(kRestVRPos[h * 3 + 1])) + c.open_width;
        c1_[h * 3 + 0] = p0_[h * 3 + 0];  // first: straight out (not back / down)
        c1_[h * 3 + 1] = side * wide;
        c1_[h * 3 + 2] = p0_[h * 3 + 2];
        c2_[h * 3 + 0] = kRestVRPos[h * 3 + 0];  // then down from the outside
        c2_[h * 3 + 1] = side * wide;
        c2_[h * 3 + 2] = kRestVRPos[h * 3 + 2];
      }
    }
    // Duration from the longer wrist path: min-jerk peak speed = 1.875 * length / T.
    std::array<double, 2> len{0.0, 0.0};
    std::array<double, 9> prev = Point(0.0);
    for (int i = 1; i <= 40; ++i) {
      const std::array<double, 9> cur = Point(i / 40.0);
      for (int h = 0; h < 2; ++h) {
        double d = 0.0;
        for (int k = 0; k < 3; ++k) d += std::pow(cur[h * 3 + k] - prev[h * 3 + k], 2);
        len[h] += std::sqrt(d);
      }
      prev = cur;
    }
    length_ = std::max(len[0], len[1]);
    duration_ = std::clamp(1.875 * length_ / std::max(c.hand_speed, 0.01), c.lower_min_s, c.lower_max_s);
    delay_ = c.soften_s;
  }

  /// Targets at time t (s) since the stop.
  void Sample(double t, std::array<double, 9>& pos, std::array<double, 12>& orn) const {
    const double s = MinJerk((t - delay_) / std::max(duration_, 1e-3));
    pos = Point(s);
    for (int q = 0; q < 3; ++q) {  // slerp each quaternion
      const double* a = &q0_[q * 4];
      const double* b = &kRestVROrn[q * 4];
      double dot = 0.0;
      for (int k = 0; k < 4; ++k) dot += a[k] * b[k];
      const double sgn = dot < 0.0 ? -1.0 : 1.0;
      dot = std::min(1.0, std::fabs(dot));
      const double th = std::acos(dot);
      double wa = 1.0 - s, wb = s;
      if (th > 1e-4) { wa = std::sin((1.0 - s) * th) / std::sin(th); wb = std::sin(s * th) / std::sin(th); }
      double n = 0.0;
      for (int k = 0; k < 4; ++k) { orn[q * 4 + k] = wa * a[k] + wb * sgn * b[k]; n += orn[q * 4 + k] * orn[q * 4 + k]; }
      n = std::sqrt(n);
      for (int k = 0; k < 4; ++k) orn[q * 4 + k] /= (n > 1e-9 ? n : 1.0);
    }
  }

  bool hug() const { return hug_; }
  double duration() const { return duration_; }
  double delay() const { return delay_; }
  double length() const { return length_; }

 private:
  std::array<double, 9> Point(double s) const {
    std::array<double, 9> p{};
    const double u = 1.0 - s;
    for (int i = 0; i < 6; ++i)
      p[i] = u * u * u * p0_[i] + 3 * u * u * s * c1_[i] + 3 * u * s * s * c2_[i] + s * s * s * kRestVRPos[i];
    for (int i = 6; i < 9; ++i) p[i] = p0_[i] + s * (kRestVRPos[i] - p0_[i]);
    return p;
  }
  std::array<double, 9> p0_{}, c1_{}, c2_{};
  std::array<double, 12> q0_{};
  bool hug_ = false;
  double duration_ = 0.0, delay_ = 0.0, length_ = 0.0;
};

/**
 * @brief Control-thread helper: how far the arm gains are toward the safe-stop profile.
 *
 * Returns b in [0, 1]:  Kp_j = Kp_nom_j * (1 + b * (kp_scale_j - 1)), same for Kd.
 * Stop -> SOFTENING (soften_s) -> SOFT (latched) -> release -> RESTORING (restore_s).
 */
class ArmSoftener {
 public:
  enum class State { kIdle, kSoftening, kSoft, kRestoring };

  double Update(double dt) {
    const Config& c = Settings();
    const bool active = Active().load();
    const unsigned epoch = Epoch().load();
    if (active && (state_ == State::kIdle || state_ == State::kRestoring || epoch != epoch_)) {
      epoch_ = epoch;
      Start(State::kSoftening, 1.0, c.soften_s);
      std::cout << "[SafeStop] Arms -> " << c.profile << " over " << c.soften_s << " s" << std::endl;
    } else if (!active && (state_ == State::kSoftening || state_ == State::kSoft)) {
      Start(State::kRestoring, 0.0, c.restore_s);
      std::cout << "[SafeStop] Restoring arm stiffness over " << c.restore_s << " s" << std::endl;
    }
    t_ += dt;
    if (state_ == State::kSoftening || state_ == State::kRestoring) {
      const double r = dur_ <= 0.0 ? 1.0 : t_ / dur_;
      b_ = from_ + MinJerk(r) * (to_ - from_);
      if (r >= 1.0) state_ = (state_ == State::kSoftening) ? State::kSoft : State::kIdle;
    }
    return b_;
  }

  State state() const { return state_; }
  double blend() const { return b_; }

 private:
  void Start(State s, double to, double dur) { state_ = s; from_ = b_; to_ = to; dur_ = dur; t_ = 0.0; }
  State state_ = State::kIdle;
  unsigned epoch_ = 0;
  double b_ = 0.0, from_ = 0.0, to_ = 0.0, dur_ = 0.0, t_ = 0.0;
};

/// Optional CSV log (control thread): one row per tick while stopped and 3 s after.
class StopLogger {
 public:
  /// write_row(std::ostream&, int stop_id, double t_since_stop)
  template <typename Fn>
  void Tick(bool active, const std::string& header, const Fn& write_row) {
    const std::string& path = Settings().log_file;
    if (path.empty()) return;
    const auto now = std::chrono::steady_clock::now();
    if (active && !was_active_) { ++stop_id_; start_ = now; }
    if (active) last_active_ = now;
    was_active_ = active;
    if (!active && (stop_id_ == 0 || now - last_active_ > std::chrono::seconds(3))) return;
    if (!out_) {
      out_ = std::make_unique<std::ofstream>(path, std::ios::app);
      *out_ << header << "\n";
      std::cout << "[SafeStop] Logging this stop to " << path << std::endl;
    }
    write_row(*out_, stop_id_, std::chrono::duration<double>(now - start_).count());
  }

 private:
  std::unique_ptr<std::ofstream> out_;
  std::chrono::steady_clock::time_point start_{}, last_active_{};
  bool was_active_ = false;
  int stop_id_ = 0;
};

/// Publishes the stop state for other processes (the BrainCo hand senders open the
/// hands while it is active): PUB tcp://*:<status_port>, topic "safe_stop_state",
/// message `safe_stop_state {"active":0|1,"epoch":n}`, at 10 Hz and at once on change.
class StatusPublisher {
 public:
  explicit StatusPublisher(int port) {
    if (port <= 0) return;
    try {
      sock_ = std::make_unique<zmq::socket_t>(ctx_, ZMQ_PUB);
      sock_->set(zmq::sockopt::linger, 0);
      sock_->set(zmq::sockopt::sndhwm, 10);
      sock_->bind("tcp://*:" + std::to_string(port));
    } catch (const zmq::error_t& e) {
      std::cerr << "[SafeStop] Status publisher on port " << port << " failed: " << e.what()
                << " (BrainCo hands will not open on a safe stop)" << std::endl;
      sock_.reset();
      return;
    }
    std::cout << "  - Safe-stop state: PUB tcp://*:" << port << " topic 'safe_stop_state' (BrainCo hands)"
              << std::endl;
    thread_ = std::thread([this] { Loop(); });
  }
  ~StatusPublisher() {
    run_ = false;
    if (thread_.joinable()) thread_.join();
  }
  StatusPublisher(const StatusPublisher&) = delete;
  StatusPublisher& operator=(const StatusPublisher&) = delete;

 private:
  void Loop() {
    bool last_active = !Active().load();
    auto last_send = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    while (run_) {
      const bool active = Active().load();
      const auto now = std::chrono::steady_clock::now();
      if (active != last_active || now - last_send >= std::chrono::milliseconds(100)) {
        const std::string msg = std::string("safe_stop_state {\"active\":") + (active ? "1" : "0") +
                                ",\"epoch\":" + std::to_string(Epoch().load()) + "}";
        try {
          sock_->send(zmq::buffer(msg), zmq::send_flags::dontwait);
        } catch (const zmq::error_t&) {
        }
        last_active = active;
        last_send = now;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  zmq::context_t ctx_{1};
  std::unique_ptr<zmq::socket_t> sock_;
  std::atomic<bool> run_{true};
  std::thread thread_;
};

}  // namespace safe_stop
