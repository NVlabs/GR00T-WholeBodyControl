#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <limits>
#include <optional>
#include <thread>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "input_interface/control_session_state.hpp"
#include "input_interface/zmq_manager.hpp"

namespace {

using Clock = std::chrono::steady_clock;

static_assert(!std::is_copy_constructible_v<ControlSessionState>);
static_assert(!std::is_copy_assignable_v<ControlSessionState>);
static_assert(!std::is_move_constructible_v<ControlSessionState>);
static_assert(!std::is_move_assignable_v<ControlSessionState>);

ControlSessionToken MakeToken(std::uint8_t discriminator) {
  ControlSessionToken token{};
  token[0] = discriminator;
  return token;
}

StreamedMotionMerger::IncomingData MakeJointMotion(
    int64_t start_frame, int64_t step, int frame_count) {
  StreamedMotionMerger::IncomingData data;
  data.protocol_version = 1;
  data.catch_up_enabled = false;
  data.num_frames = frame_count;
  data.num_joints = 1;
  data.num_quat_bodies = 1;
  data.joint_pos.assign(frame_count, std::vector<double>(1, 0.0));
  data.joint_vel.assign(frame_count, std::vector<double>(1, 0.0));
  data.body_quat.assign(
      frame_count,
      std::vector<std::array<double, 4>>(1, {1.0, 0.0, 0.0, 0.0}));
  for (int frame = 0; frame < frame_count; ++frame) {
    data.frame_indices.push_back(start_frame + step * frame);
  }
  return data;
}

TEST(ZMQInputLease, PlannerPublisherDisappearanceExpiresLease) {
  const auto now = Clock::time_point(std::chrono::seconds(10));
  const auto fresh = now - std::chrono::milliseconds(100);
  const auto stale =
      now - ZMQManager::INPUT_LEASE_TIMEOUT - std::chrono::milliseconds(1);

  EXPECT_TRUE(ZMQManager::IsInputLeaseFreshAt(
      ZMQManager::ManagedMode::PLANNER, fresh, std::nullopt, now));
  EXPECT_FALSE(ZMQManager::IsInputLeaseFreshAt(
      ZMQManager::ManagedMode::PLANNER, stale, std::nullopt, now));
  EXPECT_FALSE(ZMQManager::IsInputLeaseFreshAt(
      ZMQManager::ManagedMode::PLANNER, std::nullopt, fresh, now));
}

TEST(ZMQInputLease, PosePublisherDisappearanceExpiresLease) {
  const auto now = Clock::time_point(std::chrono::seconds(10));
  const auto fresh = now - std::chrono::milliseconds(100);
  const auto stale =
      now - ZMQManager::INPUT_LEASE_TIMEOUT - std::chrono::milliseconds(1);

  EXPECT_TRUE(ZMQManager::IsInputLeaseFreshAt(
      ZMQManager::ManagedMode::STREAMED_MOTION,
      std::nullopt,
      fresh,
      now));
  EXPECT_FALSE(ZMQManager::IsInputLeaseFreshAt(
      ZMQManager::ManagedMode::STREAMED_MOTION,
      std::nullopt,
      stale,
      now));
  EXPECT_FALSE(ZMQManager::IsInputLeaseFreshAt(
      ZMQManager::ManagedMode::STREAMED_MOTION,
      fresh,
      std::nullopt,
      now));
}

TEST(ZMQInputLease, RejectsTimestampTooFarInFuture) {
  const auto now = Clock::time_point(std::chrono::seconds(10));
  const auto future =
      now + ZMQManager::INPUT_LEASE_FUTURE_TOLERANCE +
      std::chrono::milliseconds(1);

  EXPECT_FALSE(ZMQManager::IsInputLeaseFreshAt(
      ZMQManager::ManagedMode::PLANNER, future, std::nullopt, now));
}

TEST(ZMQInputLease, PlannerSourceFrameMustAdvance) {
  EXPECT_TRUE(ZMQManager::IsPlannerFrameAdvancing(std::nullopt, 0));
  EXPECT_TRUE(ZMQManager::IsPlannerFrameAdvancing(0, 1));
  EXPECT_FALSE(ZMQManager::IsPlannerFrameAdvancing(1, 1));
  EXPECT_FALSE(ZMQManager::IsPlannerFrameAdvancing(2, 1));
  EXPECT_FALSE(ZMQManager::IsPlannerFrameAdvancing(std::nullopt, -1));
}

TEST(ZMQInputLease, CommandSourceIndexMustAdvance) {
  EXPECT_TRUE(ZMQManager::IsCommandIndexAdvancing(std::nullopt, 0));
  EXPECT_TRUE(ZMQManager::IsCommandIndexAdvancing(0, 1));
  EXPECT_FALSE(ZMQManager::IsCommandIndexAdvancing(1, 1));
  EXPECT_FALSE(ZMQManager::IsCommandIndexAdvancing(2, 1));
  EXPECT_FALSE(ZMQManager::IsCommandIndexAdvancing(std::nullopt, -1));
}

TEST(ZMQInputLease, ManagedPoseResetPreservesSourceSequence) {
  EXPECT_FALSE(ZMQEndpointInterface::ShouldResetSourceSequence(
      /*explicit_reset=*/false,
      /*has_control_session=*/true));
  EXPECT_TRUE(ZMQEndpointInterface::ShouldResetSourceSequence(
      /*explicit_reset=*/true,
      /*has_control_session=*/true));
  EXPECT_TRUE(ZMQEndpointInterface::ShouldResetSourceSequence(
      /*explicit_reset=*/false,
      /*has_control_session=*/false));
}

TEST(ControlSessionState, ReceiverEpochIsNonzeroAndStable) {
  ControlSessionState state;

  const auto first = state.ReceiverEpoch();
  const auto second = state.ReceiverEpoch();
  auto different = first;
  different[0] ^= 1;
  EXPECT_TRUE(ControlSessionState::IsValidToken(first));
  EXPECT_EQ(first, second);
  EXPECT_TRUE(state.IsReceiverEpoch(first));
  EXPECT_FALSE(state.IsReceiverEpoch(different));
}

TEST(ControlSessionState, WireTokenRequiresExactLengthAndNonzeroValue) {
  const auto valid = MakeToken(7);
  const std::vector<std::uint8_t> short_token(
      ControlSessionState::TOKEN_SIZE - 1, 1);
  const std::vector<std::uint8_t> long_token(
      ControlSessionState::TOKEN_SIZE + 1, 1);
  const ControlSessionToken zero{};

  EXPECT_EQ(
      ControlSessionState::ParseWireToken(valid),
      std::optional<ControlSessionToken>(valid));
  EXPECT_FALSE(ControlSessionState::ParseWireToken(short_token).has_value());
  EXPECT_FALSE(ControlSessionState::ParseWireToken(long_token).has_value());
  EXPECT_FALSE(ControlSessionState::ParseWireToken(zero).has_value());
  EXPECT_FALSE(ControlSessionState::IsValidToken(zero));
}

TEST(ControlSessionState, FirstPublisherClaimIsPermanentAndIdempotent) {
  ControlSessionState state;
  const auto receiver_epoch = state.ReceiverEpoch();
  const auto first_publisher = MakeToken(11);
  const auto other_publisher = MakeToken(12);

  EXPECT_FALSE(state.HasClaimedPublisher());
  EXPECT_EQ(
      state.ClaimPublisher(receiver_epoch, first_publisher),
      ControlSessionState::ClaimResult::CLAIMED);
  EXPECT_EQ(
      state.ClaimPublisher(receiver_epoch, first_publisher),
      ControlSessionState::ClaimResult::ALREADY_CLAIMED);
  EXPECT_EQ(
      state.ClaimPublisher(receiver_epoch, other_publisher),
      ControlSessionState::ClaimResult::REJECTED_DIFFERENT_PUBLISHER);

  EXPECT_TRUE(state.HasClaimedPublisher());
  EXPECT_TRUE(
      state.IsClaimedPublisher(receiver_epoch, first_publisher));
  EXPECT_FALSE(
      state.IsClaimedPublisher(receiver_epoch, other_publisher));
  EXPECT_EQ(state.ClaimedPublisher(), first_publisher);
}

TEST(ControlSessionState, InvalidPublisherCannotConsumeClaim) {
  ControlSessionState state;
  const auto receiver_epoch = state.ReceiverEpoch();
  const ControlSessionToken zero{};
  const auto valid_publisher = MakeToken(21);

  EXPECT_EQ(
      state.ClaimPublisher(receiver_epoch, zero),
      ControlSessionState::ClaimResult::REJECTED_INVALID_TOKEN);
  EXPECT_FALSE(state.HasClaimedPublisher());
  EXPECT_EQ(
      state.ClaimPublisher(receiver_epoch, valid_publisher),
      ControlSessionState::ClaimResult::CLAIMED);
}

TEST(ControlSessionState, WrongReceiverEpochCannotConsumeClaim) {
  ControlSessionState state;
  const auto receiver_epoch = state.ReceiverEpoch();
  auto wrong_epoch = receiver_epoch;
  wrong_epoch[0] ^= 1;
  const auto publisher = MakeToken(31);

  EXPECT_EQ(
      state.ClaimPublisher(wrong_epoch, publisher),
      ControlSessionState::ClaimResult::REJECTED_RECEIVER_EPOCH);
  EXPECT_FALSE(state.HasClaimedPublisher());
  EXPECT_FALSE(state.IsClaimedPublisher(wrong_epoch, publisher));
  EXPECT_EQ(
      state.ClaimPublisher(receiver_epoch, publisher),
      ControlSessionState::ClaimResult::CLAIMED);
}

TEST(ControlSessionState, ConcurrentPublishersCannotBothClaim) {
  ControlSessionState state;
  const auto receiver_epoch = state.ReceiverEpoch();
  constexpr std::size_t publisher_count = 8;
  std::array<ControlSessionToken, publisher_count> publishers{};
  std::array<ControlSessionState::ClaimResult, publisher_count> results{};
  std::array<std::thread, publisher_count> workers;
  std::atomic<bool> start{false};
  results.fill(
      ControlSessionState::ClaimResult::REJECTED_INVALID_TOKEN);

  for (std::size_t index = 0; index < publisher_count; ++index) {
    publishers[index] = MakeToken(
        static_cast<std::uint8_t>(index + 1));
    workers[index] = std::thread([&, index] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      results[index] =
          state.ClaimPublisher(receiver_epoch, publishers[index]);
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& worker : workers) {
    worker.join();
  }

  const auto claimed_count = std::count(
      results.begin(),
      results.end(),
      ControlSessionState::ClaimResult::CLAIMED);
  const auto rejected_count = std::count(
      results.begin(),
      results.end(),
      ControlSessionState::ClaimResult::REJECTED_DIFFERENT_PUBLISHER);
  EXPECT_EQ(claimed_count, 1);
  EXPECT_EQ(rejected_count, publisher_count - 1);

  const auto claimed_publisher = state.ClaimedPublisher();
  ASSERT_TRUE(claimed_publisher.has_value());
  EXPECT_EQ(
      std::count(
          publishers.begin(), publishers.end(), *claimed_publisher),
      1);
}

TEST(ZMQPackedMessage, RejectsUnknownAndOverflowingFieldMetadata) {
  ZMQPackedMessageSubscriber::FieldInfo unknown{
      "value", "object", {1}, false};
  std::size_t byte_size = 123;
  EXPECT_FALSE(unknown.TryComputeByteSize(byte_size));
  EXPECT_EQ(byte_size, 0U);

  ZMQPackedMessageSubscriber::FieldInfo overflow{
      "value",
      "f64",
      {std::numeric_limits<std::size_t>::max(), 2},
      false};
  EXPECT_FALSE(overflow.TryComputeByteSize(byte_size));
  EXPECT_EQ(byte_size, 0U);
}

TEST(StreamedMotionMergerSafety, SenderCannotDisableBoundedCatchUp) {
  StreamedMotionMerger merger;
  auto first = merger.MergeIncomingData(MakeJointMotion(0, 1, 512), 0);
  ASSERT_NE(first.motion, nullptr);
  EXPECT_EQ(first.motion->timesteps, 512);

  auto second = merger.MergeIncomingData(
      MakeJointMotion(512, 1, 512), 0);
  ASSERT_NE(second.motion, nullptr);
  EXPECT_TRUE(second.did_catchup_reset);
  EXPECT_LE(
      second.motion->timesteps,
      StreamedMotionMerger::MAX_MOTION_FRAMES);
}

TEST(StreamedMotionMergerSafety, RejectsFrameStepChange) {
  StreamedMotionMerger merger;
  ASSERT_NE(
      merger.MergeIncomingData(MakeJointMotion(0, 1, 2), 0).motion,
      nullptr);
  EXPECT_EQ(
      merger.MergeIncomingData(MakeJointMotion(2, 2, 2), 0).motion,
      nullptr);
}

TEST(StreamedMotionMergerSafety, RejectsSingletonChunk) {
  StreamedMotionMerger merger;
  EXPECT_EQ(
      merger.MergeIncomingData(MakeJointMotion(0, 1, 1), 0).motion,
      nullptr);
}

TEST(StreamedMotionMergerSafety, RejectsStridePhaseChange) {
  StreamedMotionMerger merger;
  ASSERT_NE(
      merger.MergeIncomingData(MakeJointMotion(0, 1000, 2), 0).motion,
      nullptr);
  EXPECT_EQ(
      merger.MergeIncomingData(MakeJointMotion(1999, 1000, 2), 0).motion,
      nullptr);
}

}  // namespace
