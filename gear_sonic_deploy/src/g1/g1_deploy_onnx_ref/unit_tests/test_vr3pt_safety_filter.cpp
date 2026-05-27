#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include "input_interface/vr3pt_safety_filter.hpp"

static std::array<double, 9> ZeroPosition() {
  return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

static std::array<double, 12> IdentityOrientation() {
  return {
      1.0, 0.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 0.0,
      1.0, 0.0, 0.0, 0.0};
}

TEST(Vr3PtSafetyFilter, FirstFrameAlwaysAccepted) {
  Vr3PtSafetyFilter filter;
  const auto result = filter.Filter(ZeroPosition(), IdentityOrientation());
  EXPECT_FALSE(result.used_last_good);
  EXPECT_FALSE(result.estop_triggered);
}

TEST(Vr3PtSafetyFilter, SmallStepAccepted) {
  Vr3PtSafetyFilter filter;
  filter.Filter(ZeroPosition(), IdentityOrientation());
  auto position = ZeroPosition();
  position[0] = 0.01;
  const auto result = filter.Filter(position, IdentityOrientation());
  EXPECT_FALSE(result.used_last_good);
  EXPECT_DOUBLE_EQ(result.position[0], 0.01);
}

TEST(Vr3PtSafetyFilter, LargePositionStepRejectedAndFrozen) {
  Vr3PtSafetyFilter filter;
  filter.Filter(ZeroPosition(), IdentityOrientation());
  auto position = ZeroPosition();
  position[0] = 0.30;
  const auto result = filter.Filter(position, IdentityOrientation());
  EXPECT_TRUE(result.used_last_good);
  EXPECT_DOUBLE_EQ(result.position[0], 0.0);
}

TEST(Vr3PtSafetyFilter, LargeQuatStepRejected) {
  Vr3PtSafetyFilter filter;
  filter.Filter(ZeroPosition(), IdentityOrientation());
  auto orientation = IdentityOrientation();
  orientation[0] = std::cos(M_PI / 4.0);
  orientation[1] = std::sin(M_PI / 4.0);
  const auto result = filter.Filter(ZeroPosition(), orientation);
  EXPECT_TRUE(result.used_last_good);
}

TEST(Vr3PtSafetyFilter, NonUnitQuatIsNormalizedAndAccepted) {
  Vr3PtSafetyFilter filter;
  auto orientation = IdentityOrientation();
  orientation[0] = 2.0;
  orientation[4] = 2.0;
  orientation[8] = 2.0;
  const auto result = filter.Filter(ZeroPosition(), orientation);
  EXPECT_FALSE(result.used_last_good);
  EXPECT_DOUBLE_EQ(result.orientation[0], 1.0);
  EXPECT_DOUBLE_EQ(result.orientation[4], 1.0);
  EXPECT_DOUBLE_EQ(result.orientation[8], 1.0);
}

TEST(Vr3PtSafetyFilter, NaNRejected) {
  Vr3PtSafetyFilter filter;
  filter.Filter(ZeroPosition(), IdentityOrientation());
  auto position = ZeroPosition();
  position[3] = std::nan("");
  const auto result = filter.Filter(position, IdentityOrientation());
  EXPECT_TRUE(result.used_last_good);
}

TEST(Vr3PtSafetyFilter, EStopAfterStreak) {
  Vr3PtSafetyFilter::Config config;
  config.violation_streak_estop = 3;
  Vr3PtSafetyFilter filter(config);
  filter.Filter(ZeroPosition(), IdentityOrientation());
  auto bad_position = ZeroPosition();
  bad_position[0] = 0.50;
  for (int i = 0; i < 2; ++i) {
    const auto result = filter.Filter(bad_position, IdentityOrientation());
    EXPECT_FALSE(result.estop_triggered);
  }
  const auto result = filter.Filter(bad_position, IdentityOrientation());
  EXPECT_TRUE(result.estop_triggered);
}

TEST(Vr3PtSafetyFilter, GoodFrameResetsStreak) {
  Vr3PtSafetyFilter::Config config;
  config.violation_streak_estop = 3;
  Vr3PtSafetyFilter filter(config);
  filter.Filter(ZeroPosition(), IdentityOrientation());
  auto bad_position = ZeroPosition();
  bad_position[0] = 0.50;
  filter.Filter(bad_position, IdentityOrientation());
  filter.Filter(bad_position, IdentityOrientation());
  auto good_position = ZeroPosition();
  good_position[0] = 0.005;
  filter.Filter(good_position, IdentityOrientation());
  const auto result = filter.Filter(bad_position, IdentityOrientation());
  EXPECT_FALSE(result.estop_triggered);
}
