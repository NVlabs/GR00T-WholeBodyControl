/**
 * @file file_sink.hpp
 * @brief Simple CSV file writer used by StateLogger for split-file persistence.
 *
 * Each FileSink manages one output CSV file (e.g. `base_quat.csv`, `q.csv`).
 * It lazily opens the file and writes a header on the first call to
 * `writeLine()`, then appends one timestamped row per call.
 *
 * Header format varies by HeaderType:
 *  - XYZ:        x, y, z columns
 *  - QUATERNION: w, x, y, z columns
 *  - VECTOR:     v0, v1, …, vN columns (sized from the first data span)
 */

#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <span>
#include <string>

/**
 * @class FileSink
 * @brief Writes timestamped rows to a single CSV file for one signal type.
 */
class FileSink {
public:
  enum class HeaderType {
    XYZ,
    QUATERNION,
    VECTOR,
  };

  FileSink(bool enable_csv, std::string csv_dir, const std::string& file_prefix, const std::string& header_prefix, HeaderType type);
  void writeLine(std::uint64_t index, double t_ms, double t_realtime_ms, double t_monotonic_ms, double ros_timestamp, const std::span<const double>& arr);

private:
  std::string path_;
  std::string header_prefix_;
  std::ofstream file_;
  HeaderType type_;
  bool header_written = false;

  static bool fileIsEmpty(const std::string& path);
  void open();
  void writeHeaderCommon();
  void writeHeaderIfNecessary(std::size_t vector_size);
};

