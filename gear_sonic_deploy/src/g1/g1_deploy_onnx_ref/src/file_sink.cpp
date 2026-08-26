/**
 * @file file_sink.cpp
 * @brief Implementation of FileSink – single-signal CSV file writer.
 *
 * Each FileSink instance manages one CSV file.  The file is opened in append
 * mode (to support resumption after crash) and a header is written lazily on
 * the first `writeLine()` call, once the vector size is known.
 *
 * Common CSV columns: index, time_ms, time_realtime_ms, time_monotonic_ms, ros_timestamp.
 * Signal-specific columns are generated from the `header_prefix` + HeaderType.
 */

#include "file_sink.hpp"

/// Construct a FileSink; opens the file immediately if CSV is enabled.
FileSink::FileSink(bool enable_csv, std::string csv_dir, const std::string& file_prefix, const std::string& header_prefix, HeaderType type):
  path_(csv_dir + "/" + file_prefix + ".csv"),
  header_prefix_(header_prefix),
  type_(type)
{
  if (enable_csv) {
    open();
  }
}

void FileSink::writeLine(std::uint64_t index, double t_ms, double t_realtime_ms, double t_monotonic_ms, double ros_timestamp, const std::span<const double>& arr)
{
  if (!arr.empty() && file_.good()) {
    writeHeaderIfNecessary(arr.size());
    file_.setf(std::ios::fixed, std::ios::floatfield);
    file_ << std::setprecision(3);
    file_ << index << ',' << t_ms << ',';
    file_ << t_realtime_ms << ',' << t_monotonic_ms << ',';
    file_ << std::setprecision(9);
    file_ << ros_timestamp;
    for (double v : arr)
    {
      file_ << ',' << v;
    }
    file_ << std::endl;
  }
}

bool FileSink::fileIsEmpty(const std::string& path) {
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  if (!fin.good()) return true;
  fin.seekg(0, std::ios::end);
  return fin.tellg() == 0;
}

void FileSink::open() {
  file_.open(path_, std::ios::out | std::ios::app);
  if (!file_.good()) return;
  header_written = !fileIsEmpty(path_) ? true : false;
}

void FileSink::writeHeaderCommon() {
  file_ << "index,time_ms,time_realtime_ms,time_monotonic_ms,ros_timestamp";
}

void FileSink::writeHeaderIfNecessary(std::size_t vector_size) {
  if (header_written || !file_.good()) return;

  writeHeaderCommon();

  switch (type_) {
    case HeaderType::XYZ:
    {
      static std::array<std::string, 3> labels = {"x", "y", "z"};
      for (const auto& label : labels) {
        file_ << ',' << header_prefix_ << label;
      }
      break;
    }
    case HeaderType::QUATERNION:
      static std::array<std::string, 4> labels = {"w", "x", "y", "z"};
      for (const auto& label : labels) {
        file_ << ',' << header_prefix_ << label;
      }
      break;
    case HeaderType::VECTOR:
      for (size_t i = 0; i < vector_size; ++i) {
        file_ << ',' << header_prefix_ << "_" << i;
      }
      break;
  }
  file_ << std::endl;
  header_written = true;
}

