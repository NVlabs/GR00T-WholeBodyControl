/**
 * @file state_logger.cpp
 * @brief Implementation of StateLogger – ring-buffer management, CSV persistence,
 *        and metadata output.
 *
 * Key implementation details:
 *  - Ring buffer uses modulo arithmetic with `start_` / `size_` indices.
 *  - CSV times are normalised so that the first `LogFullState` call is t = 0 ms
 *    (via `std::call_once` in `toMillisNormalized_`).
 *  - Joint data (q, dq) is transformed from IsaacLab order to hardware /
 *    MuJoCo order with `default_angles` offsets when writing to CSV, so the
 *    files contain raw measured robot values.
 *  - `LogPostState` modifies the newest ring-buffer entry in-place (guarded
 *    by `ring_mutex_`) and appends token state + metadata to their own CSV files.
 *  - `GetLatest(n, sample_dt)` supports both stride-based (exact) and
 *    timestamp-based (approximate) down-sampling.
 */

#include "state_logger.hpp"
#include "policy_parameters.hpp"

#include <iostream>
#include <algorithm>

using Entry = StateLogger::Entry;

std::optional<std::map<std::string, std::variant<std::string, int, double, bool>>> StateLogger::GetConfig() const {
  if (robot_config_.empty()) {
    return std::nullopt;
  }
  return robot_config_;
}

StateLogger::StateLogger(std::string csv_dir, size_t ring_capacity, int num_joints, int num_actions, double dt_seconds, bool enable_csv, std::map<std::string, std::variant<std::string, int, double, bool>> robot_config)
  : dt{dt_seconds},
  csv_path_(getLogsDir(enable_csv, csv_dir)),
  capacity_(ring_capacity > 0 ? ring_capacity : 1),
  ring_(capacity_),
  configured_num_joints_(num_joints),
  configured_num_actions_(num_actions),
  enable_csv_(enable_csv),

  sink_base_quat_(enable_csv, csv_path_, "base_quat", "base_q", FileSink::HeaderType::QUATERNION),
  sink_base_ang_vel_(enable_csv, csv_path_, "base_ang_vel", "base_w", FileSink::HeaderType::XYZ),
  sink_base_accel_(enable_csv, csv_path_, "base_accel", "base_a", FileSink::HeaderType::XYZ),
  sink_torso_quat_(enable_csv, csv_path_, "torso_quat", "torso_q", FileSink::HeaderType::QUATERNION),
  sink_torso_ang_vel_(enable_csv, csv_path_, "torso_ang_vel", "torso_w", FileSink::HeaderType::XYZ),
  sink_torso_accel_(enable_csv, csv_path_, "torso_accel", "torso_a", FileSink::HeaderType::XYZ),
  sink_q_(enable_csv, csv_path_, "q", "q", FileSink::HeaderType::VECTOR),
  sink_dq_(enable_csv, csv_path_, "dq", "dq", FileSink::HeaderType::VECTOR),
  sink_action_(enable_csv, csv_path_, "action", "act", FileSink::HeaderType::VECTOR),
  sink_left_hand_q_(enable_csv, csv_path_, "left_hand_q", "left_hand_q", FileSink::HeaderType::VECTOR),
  sink_left_hand_dq_(enable_csv, csv_path_, "left_hand_dq", "left_hand_dq", FileSink::HeaderType::VECTOR),
  sink_right_hand_q_(enable_csv, csv_path_, "right_hand_q", "right_hand_q", FileSink::HeaderType::VECTOR),
  sink_right_hand_dq_(enable_csv, csv_path_, "right_hand_dq", "right_hand_dq", FileSink::HeaderType::VECTOR),
  sink_left_hand_action_(enable_csv, csv_path_, "left_hand_action", "left_hand_act", FileSink::HeaderType::VECTOR),
  sink_right_hand_action_(enable_csv, csv_path_, "right_hand_action", "right_hand_act", FileSink::HeaderType::VECTOR),
  sink_token_state_(enable_csv, csv_path_, "token_state", "token", FileSink::HeaderType::VECTOR),
  sink_encoder_mode_(enable_csv, csv_path_, "encoder_mode", "encoder_mode", FileSink::HeaderType::VECTOR),
  sink_motion_playing_(enable_csv, csv_path_, "motion_playing", "playing", FileSink::HeaderType::VECTOR),

  robot_config_(std::move(robot_config)) {
  // Open motion_name CSV file
  if (enable_csv_) {
    std::string motion_name_path = csv_path_ + "/motion_name.csv";
    motion_name_file_.open(motion_name_path, std::ios::out | std::ios::app);
    if (motion_name_file_.good()) {
      // Check if file is empty to determine if we need header
      std::ifstream check_file(motion_name_path, std::ios::binary);
      if (check_file.good()) {
        check_file.seekg(0, std::ios::end);
        motion_name_header_written_ = check_file.tellg() > 0;
      }
      check_file.close();
    }
    
    // Write robot configuration metadata to file
    writeConfigMetadata_();
  }
}

uint64_t StateLogger::LogFullState(const std::array<double, 4>& base_quat,
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
                                   double ros_timestamp) {
  Entry e;
  e.index = next_index_.fetch_add(1, std::memory_order_relaxed);
  e.timestamp = std::chrono::system_clock::now();
  e.timestamp_monotonic = std::chrono::steady_clock::now();
  e.ros_timestamp = ros_timestamp;
  e.base_quat = base_quat;
  e.base_ang_vel = base_ang_vel;
  e.base_accel = base_accel;
  e.body_torso_quat = body_torso_quat;
  e.body_torso_ang_vel = body_torso_ang_vel;
  e.body_torso_accel = body_torso_accel;

  // Copy dynamic containers
  e.body_q.assign(std::begin(body_q), std::end(body_q));
  e.body_dq.assign(std::begin(body_dq), std::end(body_dq));
  e.last_action.assign(std::begin(last_action), std::end(last_action));

  // Copy hand state and action containers
  e.left_hand_q.assign(std::begin(left_hand_q), std::end(left_hand_q));
  e.left_hand_dq.assign(std::begin(left_hand_dq), std::end(left_hand_dq));
  e.right_hand_q.assign(std::begin(right_hand_q), std::end(right_hand_q));
  e.right_hand_dq.assign(std::begin(right_hand_dq), std::end(right_hand_dq));
  e.last_left_hand_action.assign(std::begin(last_left_hand_action), std::end(last_left_hand_action));
  e.last_right_hand_action.assign(std::begin(last_right_hand_action), std::end(last_right_hand_action));

  pushToRing_(e);
  if (enable_csv_) {
    appendCsvLinesSplit_(e);
  }
  return e.index;
}

bool StateLogger::LogPostState(const std::span<double>& token_state, int encoder_mode, const std::string& motion_name, bool play) {
  std::lock_guard<std::mutex> lock(ring_mutex_);

  // Check if we have any entries
  if (size_ == 0) {
    std::cerr << "[StateLogger ERROR] LogPostState called but no entries exist. Call LogFullState first." << std::endl;
    return false;
  }

  // Get the newest entry (last one in the ring)
  size_t newest_idx = (start_ + size_ - 1) % capacity_;
  Entry& newest = ring_[newest_idx];

  // Check if post-state data was already set
  if (newest.has_post_state_data) {
    std::cerr << "[StateLogger ERROR] LogPostState called but newest entry (index " << newest.index
      << ") already has post-state data. Each entry can only have post-state set once." << std::endl;
    return false;
  }

  // Update the entry with token state and metadata
  newest.token_state.assign(std::begin(token_state), std::end(token_state));
  newest.encoder_mode = encoder_mode;
  newest.motion_name = motion_name;
  newest.play = play;
  newest.has_post_state_data = true;

  // Write to CSV if enabled (token state gets its own file)
  if (enable_csv_) {
    appendTokenStateToCSV_(newest);
  }

  return true;
}

size_t StateLogger::capacity() const { return capacity_; }
size_t StateLogger::size() const {
  std::lock_guard<std::mutex> lock(ring_mutex_);
  return size_;
}

std::vector<Entry> StateLogger::GetLatest(size_t n, bool newest_first) const {
  std::lock_guard<std::mutex> lock(ring_mutex_);
  const size_t count = n < size_ ? n : size_;
  std::vector<Entry> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    size_t idx = (start_ + size_ - 1 - i + capacity_) % capacity_;
    out.push_back(ring_[idx]);
  }
  // Pad with zeros if requested more than available
  if (out.size() < n) {
    const size_t missing = n - out.size();
    for (size_t i = 0; i < missing; ++i) { out.push_back(makeZeroEntry_()); }
  }
  // Reverse if oldest_first requested
  if (!newest_first) { std::reverse(out.begin(), out.end()); }
  return out;
}

std::vector<Entry> StateLogger::GetLatest(size_t n, double sample_dt_seconds, bool newest_first) const {
  if (sample_dt_seconds <= 0.0) { return GetLatest(n, newest_first); }
  std::lock_guard<std::mutex> lock(ring_mutex_);
  std::vector<Entry> out;
  if (n == 0) return out;
  if (size_ == 0) {
    // Pad with zeros if requested more than available (consistent with first overload)
    out.reserve(n);
    for (size_t k = 0; k < n; ++k) { out.push_back(makeZeroEntry_()); }
    return out;
  }
  out.reserve(n);

  // If sample_dt aligns with dt, use stride-based selection for exact spacing
  if (dt > 0.0) {
    const double ratio = sample_dt_seconds / dt;
    const long stride = static_cast<long>(std::llround(ratio));
    const double err = std::fabs(ratio - static_cast<double>(stride));
    if (stride > 0 && err < 1e-6) {
      // newest at offset 0, then stride, 2*stride, ...
      const size_t newest_idx = (start_ + size_ - 1) % capacity_;
      for (size_t j = 0; j < n; ++j) {
        const size_t offset = static_cast<size_t>(j * stride);
        if (offset >= size_) break;
        const size_t idx = (start_ + size_ - 1 - offset + capacity_) % capacity_;
        out.push_back(ring_[idx]);
      }
      // Pad with zeros if requested more than available
      if (out.size() < n) {
        const size_t missing = n - out.size();
        for (size_t k = 0; k < missing; ++k) { out.push_back(makeZeroEntry_()); }
      }
      // Reverse if oldest_first requested
      if (!newest_first) { std::reverse(out.begin(), out.end()); }
      return out;
    }
  }

  // Fallback: timestamp-based approximate downsampling
  const size_t newest_idx = (start_ + size_ - 1) % capacity_;
  Entry newest = ring_[newest_idx];
  out.push_back(newest);

  auto step = std::chrono::duration_cast<std::chrono::system_clock::duration>(
    std::chrono::duration<double>(sample_dt_seconds));
  auto next_target_time = newest.timestamp - step;

  size_t scanned = 1; // we already took 1 entry
  size_t i = (newest_idx + capacity_ - 1) % capacity_;
  while (out.size() < n && scanned < size_) {
    const Entry& candidate = ring_[i];
    if (candidate.timestamp <= next_target_time) {
      out.push_back(candidate);
      next_target_time -= step;
    }
    i = (i + capacity_ - 1) % capacity_;
    scanned += 1;
  }
  // Pad with zeros if requested more than available
  if (out.size() < n) {
    const size_t missing = n - out.size();
    for (size_t k = 0; k < missing; ++k) { out.push_back(makeZeroEntry_()); }
  }
  // Reverse if oldest_first requested
  if (!newest_first) { std::reverse(out.begin(), out.end()); }
  return out;
}

double StateLogger::toMillisNormalized_(const std::chrono::system_clock::time_point& t) {
  std::call_once(start_once_, [&]{ start_time_ = t; });
  std::chrono::duration<double, std::milli> delta = t - start_time_;
  return delta.count();
}

void StateLogger::appendCsvLinesSplit_(const Entry& e) {
  if (!enable_csv_) return;
  // Common prefix
  uint64_t idx = e.index;
  const double t_ms = toMillisNormalized_(e.timestamp);
  const double t_ros = e.ros_timestamp;
  const double t_realtime_ms = std::chrono::duration<double, std::milli>(e.timestamp.time_since_epoch()).count();
  const double t_monotonic_ms = std::chrono::duration<double, std::milli>(e.timestamp_monotonic.time_since_epoch()).count();

  sink_base_quat_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.base_quat));
  sink_base_ang_vel_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.base_ang_vel));
  sink_base_accel_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.base_accel));
  sink_torso_quat_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.body_torso_quat));
  sink_torso_ang_vel_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.body_torso_ang_vel));
  sink_torso_accel_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.body_torso_accel));

  // Transform body_q and body_dq back to robot measured order for CSV logging
  // This reverses the transformation done when reading from robot: 
  // - body_q: add default_angles back and convert from IsaacLab order to robot hardware order
  // - body_dq: convert from IsaacLab order to robot hardware order
  std::vector<double> body_q_measured(e.body_q.size());
  std::vector<double> body_dq_measured(e.body_dq.size());
  for (size_t i = 0; i < e.body_q.size() && i < 29; i++) {
    body_q_measured[i] = e.body_q[isaaclab_to_mujoco[i]] + default_angles[i];
    body_dq_measured[i] = e.body_dq[isaaclab_to_mujoco[i]];
  }
  
  sink_q_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(body_q_measured));
  sink_dq_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(body_dq_measured));
  sink_action_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.last_action));
  sink_left_hand_q_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.left_hand_q));
  sink_left_hand_dq_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.left_hand_dq));
  sink_right_hand_q_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.right_hand_q));
  sink_right_hand_dq_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.right_hand_dq));
  sink_left_hand_action_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.last_left_hand_action));
  sink_right_hand_action_.writeLine(idx, t_ms, t_realtime_ms, t_monotonic_ms, t_ros, std::span(e.last_right_hand_action));
}

void StateLogger::appendTokenStateToCSV_(const Entry& e) {
  if (!enable_csv_) return;

  // Skip if post-state was never set
  if (!e.has_post_state_data) {
    return;
  }

  const double t_ms = toMillisNormalized_(e.timestamp);
  const double t_realtime_ms = std::chrono::duration<double, std::milli>(e.timestamp.time_since_epoch()).count();
  const double t_monotonic_ms = std::chrono::duration<double, std::milli>(e.timestamp_monotonic.time_since_epoch()).count();

  // Write token state using FileSink (only if not empty)
  if (!e.token_state.empty()) {
    sink_token_state_.writeLine(e.index, t_ms, t_realtime_ms, t_monotonic_ms, e.ros_timestamp, std::span(e.token_state));
  }
  
  // Always write metadata (encoder_mode, motion_name, play state) even if token is empty
  // This ensures we have metadata for every frame where LogPostState was called
  
  // Write encoder mode to separate file (convert int to double for FileSink)
  std::array<double, 1> encoder_mode_arr = {static_cast<double>(e.encoder_mode)};
  sink_encoder_mode_.writeLine(e.index, t_ms, t_realtime_ms, t_monotonic_ms, e.ros_timestamp, std::span(encoder_mode_arr));
  
  // Write motion playing state to separate file (convert bool to double for FileSink)
  std::array<double, 1> motion_playing_arr = {e.play ? 1.0 : 0.0};
  sink_motion_playing_.writeLine(e.index, t_ms, t_realtime_ms, t_monotonic_ms, e.ros_timestamp, std::span(motion_playing_arr));
  
  // Write motion name to separate file (custom string handling)
  writeMotionNameLine_(e.index, t_ms, t_realtime_ms, t_monotonic_ms, e.ros_timestamp, e.motion_name);
}

void StateLogger::writeMotionNameLine_(uint64_t index, double t_ms, double t_realtime_ms, double t_monotonic_ms, double ros_timestamp, const std::string& motion_name) {
  if (!motion_name_file_.good()) return;
  
  // Write header if needed
  if (!motion_name_header_written_) {
    motion_name_file_ << "index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp,motion_name" << std::endl;
    motion_name_header_written_ = true;
  }
  
  // Write data line
  motion_name_file_.setf(std::ios::fixed, std::ios::floatfield);
  motion_name_file_ << std::setprecision(3);
  motion_name_file_ << index << ',' << t_ms << ',';
  motion_name_file_ << t_realtime_ms << ',' << t_monotonic_ms << ',';
  motion_name_file_ << std::setprecision(9);
  motion_name_file_ << ros_timestamp << ",\"" << motion_name << "\"";
  motion_name_file_ << std::endl;
}

void StateLogger::pushToRing_(const Entry& e) {
  std::lock_guard<std::mutex> lock(ring_mutex_);
  if (size_ < capacity_) {
    size_t pos = (start_ + size_) % capacity_;
    ring_[pos] = e;
    size_ += 1;
  } else {
    // Overwrite oldest
    ring_[start_] = e;
    start_ = (start_ + 1) % capacity_;
  }
}

std::string StateLogger::getLogsDir(bool enable_csv, std::string csv_path) {
  if (enable_csv) { // resolve default logs directory if none provided
    if (csv_path.empty()) {
      csv_path = buildDefaultLogsDir_();
    }
    // Ensure directory exists
    std::ostringstream cmd;
    cmd << "mkdir -p " << csv_path;
    int rc = std::system(cmd.str().c_str());
    (void)rc;
  }
  return csv_path;
}

std::string StateLogger::buildDefaultLogsDir_() {
  auto now = std::chrono::system_clock::now();
  std::time_t tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&tt, &tm);
  std::ostringstream dd, MM, yy, HH, mm, ss;
  dd << std::put_time(&tm, "%d");
  MM << std::put_time(&tm, "%m");
  yy << std::put_time(&tm, "%y");
  HH << std::put_time(&tm, "%H");
  mm << std::put_time(&tm, "%M");
  ss << std::put_time(&tm, "%S");

  std::ostringstream path_builder;
  path_builder << "logs/" << dd.str() << '-' << MM.str() << '-' << yy.str()
    << '/' << HH.str() << '-' << mm.str() << '-' << ss.str();
  const std::string dir = path_builder.str();
  std::ostringstream cmd; cmd << "mkdir -p " << dir; int rc = std::system(cmd.str().c_str()); (void)rc;
  return dir;
}

void StateLogger::writeConfigMetadata_() {
  if (!enable_csv_ || robot_config_.empty()) {
    return;
  }
  
  std::string config_path = csv_path_ + "/metadata.json";
  std::ofstream config_file(config_path, std::ios::out | std::ios::trunc);
  
  if (!config_file.good()) {
    std::cerr << "[StateLogger WARNING] Failed to open metadata file: " << config_path << std::endl;
    return;
  }
  
  // Write JSON format
  config_file << "{\n";
  
  // Add logging metadata
  config_file << "  \"logging\": {\n";
  config_file << "    \"dt\": " << dt << ",\n";
  config_file << "    \"num_joints\": " << configured_num_joints_ << ",\n";
  config_file << "    \"num_actions\": " << configured_num_actions_ << ",\n";
  config_file << "    \"ring_capacity\": " << capacity_ << "\n";
  config_file << "  }";
  
  // Add robot configuration if present
  if (!robot_config_.empty()) {
    config_file << ",\n  \"robot_config\": {\n";
    
    bool first = true;
    for (const auto& [key, value] : robot_config_) {
      if (!first) {
        config_file << ",\n";
      }
      first = false;
      
      config_file << "    \"" << key << "\": ";
      
      // Handle different variant types
      std::visit([&config_file](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, std::string>) {
          config_file << "\"" << arg << "\"";
        } else if constexpr (std::is_same_v<T, bool>) {
          config_file << (arg ? "true" : "false");
        } else if constexpr (std::is_same_v<T, int>) {
          config_file << arg;
        } else if constexpr (std::is_same_v<T, double>) {
          config_file << std::setprecision(9) << arg;
        }
      }, value);
    }
    
    config_file << "\n  }";
  }
  
  config_file << "\n}\n";
  config_file.close();
  
  std::cout << "[StateLogger] Metadata written to: " << config_path << std::endl;
}

Entry StateLogger::makeZeroEntry_() const {
  Entry e;
  e.index = 0;
  e.timestamp = std::chrono::system_clock::time_point{};
  e.base_quat = {0.0, 0.0, 0.0, 0.0};
  e.base_ang_vel = {0.0, 0.0, 0.0};
  e.base_accel = {0.0, 0.0, 0.0};
  e.body_torso_quat = {0.0, 0.0, 0.0, 0.0};
  e.body_torso_ang_vel = {0.0, 0.0, 0.0};
  e.body_torso_accel = {0.0, 0.0, 0.0};
  if (configured_num_joints_ > 0) {
    e.body_q.assign(static_cast<size_t>(configured_num_joints_), 0.0);
    e.body_dq.assign(static_cast<size_t>(configured_num_joints_), 0.0);
  }
  if (configured_num_actions_ > 0) {
    e.last_action.assign(static_cast<size_t>(configured_num_actions_), 0.0);
  }
  // Hand data (7 motors each)
  e.left_hand_q.assign(7, 0.0);
  e.left_hand_dq.assign(7, 0.0);
  e.right_hand_q.assign(7, 0.0);
  e.right_hand_dq.assign(7, 0.0);
  e.last_left_hand_action.assign(7, 0.0);
  e.last_right_hand_action.assign(7, 0.0);
  // Post-state data (default to no post-state data)
  e.has_post_state_data = false;
  e.token_state.clear();
  return e;
}

