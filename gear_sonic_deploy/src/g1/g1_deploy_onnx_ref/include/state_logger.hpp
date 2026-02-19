/**
 * @file state_logger.hpp
 * @brief Thread-safe state logger with an in-memory ring buffer and split CSV
 *        file persistence.
 *
 * StateLogger serves two purposes:
 *  1. **In-memory ring buffer** – Keeps the latest N entries accessible for
 *     real-time consumers (output interfaces, observation gathering).
 *  2. **Split CSV files** – Writes one CSV file per signal for offline analysis.
 *
 * ## Ring Buffer
 *
 * - Fixed-capacity, configured at construction.
 * - Each record gets a monotonically increasing index (never reused).
 * - Oldest entries are silently dropped when full.
 * - Thread-safe: all access is guarded by `ring_mutex_`.
 *
 * ## CSV Files
 *
 * Created in the provided directory (one file per signal):
 *
 *   File                | Content
 *   --------------------|--------
 *   base_quat.csv       | IMU base orientation (qw,qx,qy,qz)
 *   base_ang_vel.csv    | Base angular velocity (wx,wy,wz)
 *   torso_quat.csv      | Torso IMU orientation
 *   torso_ang_vel.csv   | Torso angular velocity
 *   q.csv               | Joint positions (hardware order, with default offsets)
 *   dq.csv              | Joint velocities (hardware order)
 *   action.csv          | Policy actions (hardware order, scaled + offset)
 *   left_hand_q/dq.csv  | Left Dex3 hand positions / velocities
 *   right_hand_q/dq.csv | Right Dex3 hand positions / velocities
 *   left/right_hand_action.csv | Hand actions
 *   token_state.csv     | Encoder token output
 *   encoder_mode.csv    | Encoder mode per tick
 *   motion_name.csv     | Active motion name per tick
 *   motion_playing.csv  | Play/pause state per tick
 *   metadata.json       | Robot config + logging parameters
 *
 * Times are normalised so the first record is 0.0 ms.
 *
 * ## Two-Phase Logging
 *
 * Each control tick calls:
 *  1. `LogFullState(...)` – records IMU, joints, velocities, last action.
 *  2. `LogPostState(...)` – appends encoder token state and motion metadata
 *     to the **same** entry (must be called after LogFullState).
 *
 * @note q.csv and dq.csv are transformed to raw hardware measurements
 *       (MuJoCo → hardware joint order, default_angles added back to q).
 */

#pragma once

#include "file_sink.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <cstdlib>
#include <cmath>

/**
 * @class StateLogger
 * @brief Thread-safe ring-buffer logger with optional CSV persistence.
 */
class StateLogger {
 public:
  struct Entry {
    uint64_t index = 0;
    std::chrono::system_clock::time_point timestamp; // wall-clock time (for CSV logging)
    std::chrono::steady_clock::time_point timestamp_monotonic; // monotonic time (for CSV logging)
    double ros_timestamp = 0.0;  // ROS time in seconds (for ROS message publishing)

    std::array<double, 4> base_quat {0, 0, 0, 0};         // qw, qx, qy, qz
    std::array<double, 3> base_ang_vel {0, 0, 0};         // wx, wy, wz
    std::array<double, 3> base_accel {0, 0, 0};           // ax, ay, az
    std::array<double, 4> body_torso_quat {0, 0, 0, 0};   // qw, qx, qy, qz
    std::array<double, 3> body_torso_ang_vel {0, 0, 0};   // wx, wy, wz
    std::array<double, 3> body_torso_accel {0, 0, 0};     // ax, ay, az

    // Optional dynamic fields (may be empty)
    std::vector<double> body_q;       // size = num_joints
    std::vector<double> body_dq;      // size = num_joints
    std::vector<double> last_action;  // size = num_actions

    // Dex3 hands (7 motors each)
    std::vector<double> left_hand_q;       // size = 7 (q positions)
    std::vector<double> left_hand_dq;      // size = 7 (dq velocities)
    std::vector<double> right_hand_q;      // size = 7 (q positions)
    std::vector<double> right_hand_dq;     // size = 7 (dq velocities)
    std::vector<double> last_left_hand_action;  // size = 7
    std::vector<double> last_right_hand_action; // size = 7

    // Post-state data (set after initial state logging via LogPostState)
    bool has_post_state_data = false;
    std::vector<double> token_state;  // Token/latent state from encoder
    int encoder_mode = -2;          // Encoder mode when token state was generated; -2: no token state, -1: need token but no encoder, 0,1,2,...: encoder mode.
    std::string motion_name = "";   // Name of the motion sequence being executed
    bool play = false;              // Operator play state (controls motion playback)
  };

  // Logging period (seconds). Informational only; logger does not enforce cadence.
  // Useful for consumers to understand intended logging frequency.
  double dt = 0.0;

  /**
   * Get robot configuration if it was provided during construction.
   * Returns std::nullopt if no configuration was provided.
   */
  std::optional<std::map<std::string, std::variant<std::string, int, double, bool>>> GetConfig() const;

  /**
   * Construct a logger.
   * @param csv_dir        Directory path for split CSV files (one file per signal)
   * @param ring_capacity  Max number of entries to keep in memory
   * @param num_joints     If >=0, include q/dq columns with this size
   * @param num_actions    If >=0, include action columns with this size
   * @param dt_seconds     Intended logging period in seconds (informational; default 0.0)
   * @param enable_csv     If false, disable CSV output entirely (ring buffer only; default: true)
   * @param robot_config   Optional robot configuration map (model paths, frequencies, etc.) - immutable after construction
   */
  StateLogger(std::string csv_dir, size_t ring_capacity, int num_joints = -1, int num_actions = -1, double dt_seconds = 0.0, bool enable_csv = true,
              std::map<std::string, std::variant<std::string, int, double, bool>> robot_config = {});

  // Non-copyable, movable
  StateLogger(const StateLogger&) = delete;
  StateLogger& operator=(const StateLogger&) = delete;
  StateLogger(StateLogger&&) = default;
  StateLogger& operator=(StateLogger&&) = default;

  /**
   * Log the full state including joints, last action, and hand states/actions.
   * Returns the assigned monotonic index.
   */
  uint64_t LogFullState(const std::array<double, 4>& base_quat,
                        const std::array<double, 3>& base_ang_vel,
                        const std::array<double, 3>& base_accel,
                        const std::array<double, 4>& body_torso_quat,
                        const std::array<double, 3>& body_torso_ang_vel,
                        const std::array<double, 3>& body_torso_accel,
                        const std::span<double>& body_q,
                        const std::span<double>& body_dq,
                        const std::span<double>& last_action,
                        const std::span<double>& left_hand_q,
                        const std::span<double>& left_hand_dq,
                        const std::span<double>& right_hand_q,
                        const std::span<double>& right_hand_dq,
                        const std::span<double>& last_left_hand_action,
                        const std::span<double>& last_right_hand_action,
                        double ros_timestamp = 0.0);

  /**
   * Log post-state data (e.g., token state from encoder) to the most recent entry.
   * This modifies the newest entry in the ring buffer without creating a new entry.
   * Must be called AFTER LogFullState. Returns false if no entry exists, entry already has post-state data,
   * or if the update fails.
   * @param token_state Token/latent state vector from encoder
   * @param encoder_mode Encoder mode value (-2: no token state, -1: need token but no encoder, 0+: encoder mode)
   * @param motion_name Name of the current motion sequence being executed
   * @param play Operator play state (controls motion playback)
   */
  bool LogPostState(const std::span<double>& token_state, int encoder_mode = -2, const std::string& motion_name = "", bool play = false);

  size_t capacity() const;
  size_t size() const;

  // Returns copies of the latest n entries (up to available size)
  // If newest_first is true (default), returns [newest, ..., oldest]; otherwise [oldest, ..., newest]
  std::vector<Entry> GetLatest(size_t n, bool newest_first = true) const;

  // Returns up to n entries sampled approximately every sample_dt_seconds going backward
  // from the most recent entry. If sample_dt_seconds <= 0, behaves like GetLatest(n, newest_first).
  // Optimized path: if dt > 0 and sample_dt_seconds is an integer multiple of dt, we select by fixed stride.
  // If newest_first is true (default), returns [newest, ..., oldest]; otherwise [oldest, ..., newest]
  std::vector<Entry> GetLatest(size_t n, double sample_dt_seconds, bool newest_first = true) const;

 private:
  // Time normalization helper
  double toMillisNormalized_(const std::chrono::system_clock::time_point& t);

  // ---------- Split-file support ----------
  void appendCsvLinesSplit_(const Entry& e);

  // Append token state to CSV (called by LogPostState)
  void appendTokenStateToCSV_(const Entry& e);

  // Ring buffer operations
  void pushToRing_(const Entry& e);

 private:
  // Configuration
  std::string csv_path_;
  size_t capacity_ = 1;
  int configured_num_joints_ = -1;
  int configured_num_actions_ = -1;
  bool enable_csv_ = true;

  // Ring buffer state
  mutable std::mutex ring_mutex_;
  std::vector<Entry> ring_;
  size_t start_ = 0; // index of the oldest element
  size_t size_ = 0;  // number of valid elements

  // Index counter
  std::atomic<uint64_t> next_index_ {0};

  // Split CSV file sinks
  FileSink sink_base_quat_;
  FileSink sink_base_ang_vel_;
  FileSink sink_base_accel_;
  FileSink sink_torso_quat_;
  FileSink sink_torso_ang_vel_;
  FileSink sink_torso_accel_;
  FileSink sink_q_;
  FileSink sink_dq_;
  FileSink sink_action_;
  FileSink sink_left_hand_q_;
  FileSink sink_left_hand_dq_;
  FileSink sink_right_hand_q_;
  FileSink sink_right_hand_dq_;
  FileSink sink_left_hand_action_;
  FileSink sink_right_hand_action_;
  FileSink sink_token_state_;      // Post-state data (token from encoder)
  FileSink sink_encoder_mode_;     // Post-state data (encoder mode)
  FileSink sink_motion_playing_;   // Post-state data (motion playing state)
  
  // Custom sink for motion_name (string data)
  std::ofstream motion_name_file_;
  bool motion_name_header_written_ = false;
  void writeMotionNameLine_(uint64_t index, double t_ms, double t_realtime_ms, double t_monotonic_ms, double ros_timestamp, const std::string& motion_name);

  // Time normalization state (first record as zero)
  std::once_flag start_once_;
  std::chrono::system_clock::time_point start_time_ {};

  // Robot configuration storage (set during construction, immutable)
  std::map<std::string, std::variant<std::string, int, double, bool>> robot_config_;

  // Build default logs directory logs/dd-mm-yy/hh-mm-ss and ensure it exists
  static std::string getLogsDir(bool enable_csv, std::string csv_path);
  static std::string buildDefaultLogsDir_();

  // Write robot configuration metadata to JSON file
  void writeConfigMetadata_();

  Entry makeZeroEntry_() const;
};
