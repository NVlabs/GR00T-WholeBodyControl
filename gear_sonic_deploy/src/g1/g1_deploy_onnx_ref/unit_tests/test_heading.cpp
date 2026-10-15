#include <gtest/gtest.h>

#include <tuple>
#include <type_traits>

#include "../include/math_utils.hpp"

namespace {

template <typename T>
std::array<T, 4> Heading(const std::array<T, 4>& q) {
  if constexpr (std::is_same_v<T, double>) {
    return calc_heading_quat_d(q);
  } else {
    return calc_heading_quat(q);
  }
}

template <typename T>
std::array<T, 4> InverseHeading(const std::array<T, 4>& q) {
  if constexpr (std::is_same_v<T, double>) {
    return calc_heading_quat_inv_d(q);
  } else {
    return calc_heading_quat_inv(q);
  }
}

template <typename T>
class HeadingTest : public ::testing::Test {
 protected:
  static constexpr double kTolerance = std::is_same_v<T, double> ? 1e-12 : 2e-6;

  void ExpectQuaternion(const std::array<T, 4>& actual, const std::array<double, 4>& expected) {
    for (size_t i = 0; i < 4; ++i) {
      EXPECT_NEAR(actual[i], expected[i], kTolerance) << "component " << i;
    }
  }
};

using ScalarTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(HeadingTest, ScalarTypes);

TYPED_TEST(HeadingTest, TiltedPoseMatchesTrainingTwist) {
  // Rz(0) * Ry(30 deg) * Rx(30 deg). Projected forward-axis yaw is zero,
  // but torch_transform.get_heading_q gives a -8.2132107 degree Z twist.
  const std::array<TypeParam, 4> q = {
      TypeParam(0.9330127018922194), TypeParam(0.25), TypeParam(0.25),
      TypeParam(-0.06698729810778066)};
  this->ExpectQuaternion(Heading(q), {0.9974325337111245, 0.0, 0.0, -0.07161243393857301});
  this->ExpectQuaternion(InverseHeading(q), {0.9974325337111245, 0.0, 0.0, 0.07161243393857301});
}

TYPED_TEST(HeadingTest, RelativeOrientationMatchesTraining6DObservation) {
  const std::array<TypeParam, 4> q = {
      TypeParam(0.9330127018922194), TypeParam(0.25), TypeParam(0.25),
      TypeParam(-0.06698729810778066)};
  // Identity reference: quat_inv(get_heading_q(q)) * identity.
  // Deployment flattens the first two rotation-matrix columns row-wise.
  const auto inverse = InverseHeading(q);
  const auto matrix = quat_to_rotation_matrix_d(std::array<double, 4>{inverse[0], inverse[1], inverse[2], inverse[3]});
  const std::array<double, 6> expected = {
      0.989743318610787, -0.14285714285714282,
      0.14285714285714282, 0.989743318610787, 0.0, 0.0};
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 2; ++col) {
      EXPECT_NEAR(matrix[row][col], expected[2 * row + col], TestFixture::kTolerance);
    }
  }
}

TYPED_TEST(HeadingTest, PureHorizontalAxisTiltHasNoTwist) {
  // A 60-degree tilt around (1, 1, 0) has zero Z twist; projected yaw is nonzero.
  const std::array<TypeParam, 4> q = {
      TypeParam(0.8660254037844386), TypeParam(0.3535533905932738),
      TypeParam(0.3535533905932738), TypeParam(0)};
  this->ExpectQuaternion(Heading(q), {1.0, 0.0, 0.0, 0.0});
  this->ExpectQuaternion(InverseHeading(q), {1.0, 0.0, 0.0, 0.0});
}

TYPED_TEST(HeadingTest, PitchThroughVerticalKeepsTwist) {
  const double yaw = 30.0 * std::acos(-1.0) / 180.0;
  for (double pitch_degrees : {80.0, 89.999, 90.0, 90.001, 100.0}) {
    const double pitch = pitch_degrees * std::acos(-1.0) / 180.0;
    // q_z(yaw) * q_y(pitch); forward-axis projection becomes singular at 90 deg.
    const std::array<TypeParam, 4> q = {
        TypeParam(std::cos(yaw / 2) * std::cos(pitch / 2)),
        TypeParam(-std::sin(yaw / 2) * std::sin(pitch / 2)),
        TypeParam(std::cos(yaw / 2) * std::sin(pitch / 2)),
        TypeParam(std::sin(yaw / 2) * std::cos(pitch / 2))};
    this->ExpectQuaternion(Heading(q), {std::cos(yaw / 2), 0.0, 0.0, std::sin(yaw / 2)});
  }
}

TYPED_TEST(HeadingTest, YawWrapAndQuaternionSignPreserveRotation) {
  for (double degrees : {-181.0, -180.0, -179.0, 0.0, 179.0, 180.0, 181.0}) {
    const double yaw = degrees * std::acos(-1.0) / 180.0;
    for (double sign : {-1.0, 1.0}) {
      const std::array<TypeParam, 4> q = {
          TypeParam(sign * std::cos(yaw / 2)), TypeParam(0), TypeParam(0),
          TypeParam(sign * std::sin(yaw / 2))};
      const auto heading = Heading(q);
      this->ExpectQuaternion(heading, {sign * std::cos(yaw / 2), 0.0, 0.0, sign * std::sin(yaw / 2)});
      const auto direction = quat_rotate(heading, std::array<TypeParam, 3>{1, 0, 0});
      EXPECT_NEAR(direction[0], std::cos(yaw), TestFixture::kTolerance);
      EXPECT_NEAR(direction[1], std::sin(yaw), TestFixture::kTolerance);
      this->ExpectQuaternion(quat_mul(heading, InverseHeading(q)), {1.0, 0.0, 0.0, 0.0});
    }
  }
}

TYPED_TEST(HeadingTest, DegenerateProjectionMatchesTrainingNormalization) {
  this->ExpectQuaternion(Heading(std::array<TypeParam, 4>{0, 1, 0, 0}), {0.0, 0.0, 0.0, 0.0});
  // Training clamps the projected norm to 1e-9 for both float and double.
  const std::array<TypeParam, 4> q = {TypeParam(6e-11), TypeParam(1), TypeParam(0), TypeParam(8e-11)};
  this->ExpectQuaternion(Heading(q), {0.06, 0.0, 0.0, 0.08});
}

}  // namespace
