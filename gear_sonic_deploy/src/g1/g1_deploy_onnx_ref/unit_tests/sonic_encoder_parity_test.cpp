#include <gtest/gtest.h>

#include "../include/encoder.hpp"
#include "../include/math_utils.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kWindowLength = 10;
constexpr std::size_t kJointCount = 29;
constexpr std::size_t kOrientationCount = 6;
constexpr std::size_t kEncoderDimension = 1247;
constexpr std::size_t kTokenDimension = 64;
constexpr double kQuaternionNormTolerance = 1e-5;

// Cumulative offsets for every observation in observation_config.yaml.
constexpr std::array<std::size_t, 13> kSliceOffsets = {
    0,    // encoder_mode_4
    4,    // motion_joint_positions_10frame_step1
    294,  // motion_joint_velocities_10frame_step1
    584,  // motion_anchor_orientation_10frame_step1
    644,  // motion_anchor_orientation
    650,  // motion_joint_positions_lowerbody_10frame_step1
    770,  // motion_joint_velocities_lowerbody_10frame_step1
    890,  // vr_3point_local_target
    899,  // vr_3point_local_orn_target
    911,  // smpl_joints_4frame_step1
    1199, // smpl_anchor_orientation_4frame_step1
    1223, // motion_joint_positions_wrists_4frame_step1
    1247,
};
static_assert(kSliceOffsets.back() == kEncoderDimension);
static_assert(sizeof(float) == 4);

using Quaternion = std::array<double, 4>;
using JointRow = std::array<double, kJointCount>;
using JointWindow = std::array<JointRow, kWindowLength>;
using QuaternionWindow = std::array<Quaternion, kWindowLength>;

struct CanonicalCase {
  JointWindow positions;
  JointWindow velocities;
  Quaternion initial_observed_root_wxyz;
  Quaternion current_observed_root_wxyz;
  Quaternion initial_reference_root_wxyz;
  QuaternionWindow future_reference_root_wxyz;
};

bool IsFinite(double value) {
  constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ULL;
  return (std::bit_cast<std::uint64_t>(value) & exponent_mask) != exponent_mask;
}

bool IsFinite(float value) {
  constexpr std::uint32_t exponent_mask = 0x7f800000U;
  return (std::bit_cast<std::uint32_t>(value) & exponent_mask) != exponent_mask;
}

double ReadFiniteNumber(const nlohmann::json &value, const std::string &field_name) {
  if (!value.is_number()) {
    throw std::invalid_argument(field_name + " must contain only numbers");
  }
  const double result = value.get<double>();
  if (!IsFinite(result)) {
    throw std::invalid_argument(field_name + " must contain only finite values");
  }
  return result;
}

template <std::size_t Rows, std::size_t Columns>
std::array<std::array<double, Columns>, Rows>
ReadMatrix(const nlohmann::json &root, const std::string &field_name) {
  if (!root.contains(field_name) || !root.at(field_name).is_array() ||
      root.at(field_name).size() != Rows) {
    throw std::invalid_argument(field_name + " has the wrong row count");
  }
  std::array<std::array<double, Columns>, Rows> result{};
  const auto &rows = root.at(field_name);
  for (std::size_t row = 0; row < Rows; ++row) {
    if (!rows.at(row).is_array() || rows.at(row).size() != Columns) {
      throw std::invalid_argument(field_name + " has the wrong column count");
    }
    for (std::size_t column = 0; column < Columns; ++column) {
      result[row][column] = ReadFiniteNumber(
          rows.at(row).at(column), field_name + "[" + std::to_string(row) + "][" +
                                          std::to_string(column) + "]");
    }
  }
  return result;
}

Quaternion ReadQuaternion(const nlohmann::json &root, const std::string &field_name) {
  if (!root.contains(field_name) || !root.at(field_name).is_array() ||
      root.at(field_name).size() != 4) {
    throw std::invalid_argument(field_name + " must have shape [4]");
  }
  Quaternion result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = ReadFiniteNumber(
        root.at(field_name).at(index), field_name + "[" + std::to_string(index) + "]");
  }
  const double norm = std::sqrt(result[0] * result[0] + result[1] * result[1] +
                                result[2] * result[2] + result[3] * result[3]);
  if (!IsFinite(norm) || norm < 1e-12 || std::abs(norm - 1.0) > kQuaternionNormTolerance) {
    throw std::invalid_argument(field_name + " must be unit length within 1e-5");
  }
  return quat_unit_d(result);
}

CanonicalCase ParseCanonicalCase(const nlohmann::json &root) {
  if (!root.is_object()) {
    throw std::invalid_argument("canonical case must be a JSON object");
  }
  const std::set<std::string> expected_fields = {
      "current_observed_root_wxyz", "future_reference_root_wxyz",
      "initial_observed_root_wxyz", "initial_reference_root_wxyz", "positions",
      "velocities"};
  std::set<std::string> actual_fields;
  for (const auto &[key, unused] : root.items()) {
    static_cast<void>(unused);
    actual_fields.insert(key);
  }
  if (actual_fields != expected_fields) {
    throw std::invalid_argument("canonical case fields do not match the exact contract");
  }

  CanonicalCase result{};
  result.positions = ReadMatrix<kWindowLength, kJointCount>(root, "positions");
  result.velocities = ReadMatrix<kWindowLength, kJointCount>(root, "velocities");
  result.initial_observed_root_wxyz = ReadQuaternion(root, "initial_observed_root_wxyz");
  result.current_observed_root_wxyz = ReadQuaternion(root, "current_observed_root_wxyz");
  result.initial_reference_root_wxyz = ReadQuaternion(root, "initial_reference_root_wxyz");

  const auto future = ReadMatrix<kWindowLength, 4>(root, "future_reference_root_wxyz");
  for (std::size_t index = 0; index < kWindowLength; ++index) {
    nlohmann::json wrapped = {{"quaternion", future[index]}};
    result.future_reference_root_wxyz[index] = ReadQuaternion(wrapped, "quaternion");
  }
  return result;
}

CanonicalCase ReadCanonicalCase(const std::string &path) {
  std::ifstream stream(path);
  if (!stream) {
    throw std::runtime_error("failed to open canonical case JSON: " + path);
  }
  nlohmann::json document;
  try {
    stream >> document;
  } catch (const nlohmann::json::exception &error) {
    throw std::invalid_argument(std::string("failed to parse canonical case JSON: ") + error.what());
  }
  return ParseCanonicalCase(document);
}

std::array<double, kOrientationCount> Rotation6D(const Quaternion &quaternion) {
  const auto matrix = quat_to_rotation_matrix_d(quaternion);
  return {matrix[0][0], matrix[0][1], matrix[1][0],
          matrix[1][1], matrix[2][0], matrix[2][1]};
}

std::vector<float> PackCanonicalG1Case(const CanonicalCase &canonical) {
  std::vector<float> input(kEncoderDimension, 0.0f);
  std::size_t offset = kSliceOffsets[1];
  for (const auto &row : canonical.positions) {
    for (double value : row) {
      const float packed = static_cast<float>(value);
      if (!IsFinite(packed)) {
        throw std::invalid_argument("position float32 conversion must remain finite");
      }
      input.at(offset++) = packed;
    }
  }
  if (offset != kSliceOffsets[2]) {
    throw std::logic_error("position slice packing does not end at offset 294");
  }
  for (const auto &row : canonical.velocities) {
    for (double value : row) {
      const float packed = static_cast<float>(value);
      if (!IsFinite(packed)) {
        throw std::invalid_argument("velocity float32 conversion must remain finite");
      }
      input.at(offset++) = packed;
    }
  }
  if (offset != kSliceOffsets[3]) {
    throw std::logic_error("velocity slice packing does not end at offset 584");
  }

  const Quaternion q_apply = quat_unit_d(quat_mul_d(
      calc_heading_quat_d(canonical.initial_observed_root_wxyz),
      quat_conjugate_d(calc_heading_quat_d(canonical.initial_reference_root_wxyz))));
  const Quaternion current_inverse = quat_conjugate_d(canonical.current_observed_root_wxyz);
  for (const Quaternion &future_reference : canonical.future_reference_root_wxyz) {
    const Quaternion aligned = quat_unit_d(quat_mul_d(q_apply, future_reference));
    const Quaternion relative = quat_unit_d(quat_mul_d(current_inverse, aligned));
    for (double value : Rotation6D(relative)) {
      const float packed = static_cast<float>(value);
      if (!IsFinite(packed)) {
        throw std::invalid_argument("orientation float32 conversion must remain finite");
      }
      input.at(offset++) = packed;
    }
  }
  if (offset != kSliceOffsets[4]) {
    throw std::logic_error("orientation slice packing does not end at offset 644");
  }
  if (!std::all_of(input.begin() + static_cast<std::ptrdiff_t>(kSliceOffsets[4]), input.end(),
                   [](float value) { return value == 0.0f; })) {
    throw std::logic_error("inactive G1 encoder observations must remain zero");
  }
  return input;
}

template <typename Container>
void WriteRawFloat32(const std::string &path, const Container &values) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) {
    throw std::runtime_error("failed to open parity output: " + path);
  }
  stream.write(reinterpret_cast<const char *>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
  stream.close();
  if (!stream) {
    throw std::runtime_error("failed to write parity output: " + path);
  }
}

nlohmann::json ValidCanonicalJson() {
  nlohmann::json root = {
      {"current_observed_root_wxyz", {1.0, 0.0, 0.0, 0.0}},
      {"future_reference_root_wxyz", nlohmann::json::array()},
      {"initial_observed_root_wxyz", {1.0, 0.0, 0.0, 0.0}},
      {"initial_reference_root_wxyz", {1.0, 0.0, 0.0, 0.0}},
      {"positions", nlohmann::json::array()},
      {"velocities", nlohmann::json::array()},
  };
  for (std::size_t index = 0; index < kWindowLength; ++index) {
    root["positions"].push_back(std::vector<double>(kJointCount, 0.0));
    root["velocities"].push_back(std::vector<double>(kJointCount, 0.0));
    root["future_reference_root_wxyz"].push_back({1.0, 0.0, 0.0, 0.0});
  }
  return root;
}

} // namespace

TEST(SonicEncoderParity, RejectsWrongCanonicalShapesAndNonFiniteValues) {
  auto wrong_shape = ValidCanonicalJson();
  wrong_shape["positions"].erase(wrong_shape["positions"].begin());
  EXPECT_THROW(static_cast<void>(ParseCanonicalCase(wrong_shape)), std::invalid_argument);

  auto non_finite = ValidCanonicalJson();
  non_finite["velocities"][0][0] = std::numeric_limits<double>::infinity();
  EXPECT_THROW(static_cast<void>(ParseCanonicalCase(non_finite)), std::invalid_argument);
}

TEST(SonicEncoderParity, EncodesPinnedFloat32Input) {
  constexpr std::array<const char *, 4> environment_names = {
      "SONIC_PARITY_CASE_JSON", "SONIC_PARITY_MODEL", "SONIC_PARITY_INPUT_OUT",
      "SONIC_PARITY_TOKEN_OUT"};
  std::array<const char *, environment_names.size()> environment_values{};
  for (std::size_t index = 0; index < environment_names.size(); ++index) {
    environment_values[index] = std::getenv(environment_names[index]);
    if (environment_values[index] == nullptr || environment_values[index][0] == '\0') {
      GTEST_SKIP() << environment_names[index] << " is not set";
    }
  }

  CanonicalCase canonical{};
  ASSERT_NO_THROW(canonical = ReadCanonicalCase(environment_values[0]));
  std::vector<float> input;
  ASSERT_NO_THROW(input = PackCanonicalG1Case(canonical));
  ASSERT_EQ(input.size(), kEncoderDimension);
  for (float value : input) {
    ASSERT_TRUE(IsFinite(value));
  }

  EncoderEngine encoder;
  ASSERT_TRUE(encoder.Initialize(environment_values[1], false));
  ASSERT_EQ(encoder.GetInputTensorName(), "obs_dict");
  ASSERT_EQ(encoder.GetOutputTensorName(), "encoded_tokens");
  ASSERT_EQ(encoder.GetInputDimension(), kEncoderDimension);
  ASSERT_EQ(encoder.GetTokenDimension(), kTokenDimension);
  ASSERT_EQ(encoder.GetInputBuffer().size(), kEncoderDimension);
  std::copy(input.begin(), input.end(), encoder.GetInputBuffer().begin());
  ASSERT_TRUE(encoder.Encode());
  ASSERT_EQ(encoder.GetTokenBuffer().size(), kTokenDimension);
  for (float value : encoder.GetTokenBuffer()) {
    ASSERT_TRUE(IsFinite(value));
  }

  ASSERT_NO_THROW(WriteRawFloat32(environment_values[2], input));
  ASSERT_NO_THROW(WriteRawFloat32(environment_values[3], encoder.GetTokenBuffer()));
}
