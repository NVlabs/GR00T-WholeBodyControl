#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "input_interface/control_session_state.hpp"
#include "input_interface/zmq_endpoint_interface.hpp"
#include "input_interface/zmq_manager.hpp"

using DecodedHeader = ZMQPackedMessageSubscriber::DecodedHeader;
using BufferView = ZMQPackedMessageSubscriber::BufferView;

struct ZMQWireTestAccess {
  static void DeliverCommand(
      ZMQManager& manager,
      const DecodedHeader& header,
      const std::vector<BufferView>& buffers) {
    manager.OnCommandReceived("command", header, buffers);
  }

  static void DeliverPlanner(
      ZMQManager& manager,
      const DecodedHeader& header,
      const std::vector<BufferView>& buffers) {
    manager.OnPlannerReceived("planner", header, buffers);
  }

  static CommandMessage LatestCommand(ZMQManager& manager) {
    std::lock_guard<std::mutex> lock(manager.command_mutex_);
    return manager.latest_command_;
  }

  static PlannerMessage LatestPlanner(ZMQManager& manager) {
    std::lock_guard<std::mutex> lock(manager.planner_mutex_);
    return manager.latest_planner_message_;
  }

  static std::optional<std::int64_t> LastCommandIndex(
      ZMQManager& manager) {
    std::lock_guard<std::mutex> lock(manager.command_mutex_);
    return manager.last_accepted_command_index_;
  }

  static std::optional<std::int64_t> LastPlannerFrame(
      ZMQManager& manager) {
    std::lock_guard<std::mutex> lock(manager.planner_mutex_);
    return manager.last_accepted_planner_frame_;
  }

  static void DeliverPose(
      ZMQEndpointInterface& endpoint,
      const DecodedHeader& header,
      const std::vector<BufferView>& buffers) {
    endpoint.OnPoseDataReceived("pose", header, buffers);
  }

  static std::uint64_t PoseReceiveCount(ZMQEndpointInterface& endpoint) {
    std::lock_guard<std::mutex> lock(endpoint.data_mutex_);
    return endpoint.receive_count_;
  }

  static bool HasBufferedPose(ZMQEndpointInterface& endpoint) {
    std::lock_guard<std::mutex> lock(endpoint.data_mutex_);
    return endpoint.has_new_data_;
  }

  static bool DecodeBufferedTokenPose(ZMQEndpointInterface& endpoint) {
    DataBuffer<HeadingState> heading_state;
    std::lock_guard<std::mutex> lock(endpoint.data_mutex_);
    endpoint.has_new_data_ = false;
    auto result = endpoint.DecodeIntoMotionSequence(
        0,
        endpoint.streamed_motion_,
        endpoint.stream_window_start_,
        heading_state);
    return result.protocol_version == 4 &&
           result.token_data.size() == 64;
  }

  static std::optional<std::int64_t> LastPoseFrame(
      ZMQEndpointInterface& endpoint) {
    std::lock_guard<std::mutex> lock(endpoint.data_mutex_);
    return endpoint.last_accepted_source_frame_;
  }
};

namespace {

class OwnedWire {
 public:
  explicit OwnedWire(int version) {
    header.version = version;
    header.endian = is_little_endian() ? "le" : "be";
  }

  template <typename T>
  void Add(
      std::string name,
      std::string dtype,
      std::vector<std::size_t> shape,
      std::span<const T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* begin =
        reinterpret_cast<const std::uint8_t*>(values.data());
    storage.emplace_back(begin, begin + values.size_bytes());
    header.fields.push_back(
        {std::move(name), std::move(dtype), std::move(shape), false});
  }

  template <typename T>
  void AddScalar(std::string name, std::string dtype, T value) {
    const std::array<T, 1> values{value};
    Add<T>(
        std::move(name),
        std::move(dtype),
        {1},
        std::span<const T>(values));
  }

  std::vector<BufferView> Views() const {
    std::vector<BufferView> views;
    views.reserve(storage.size());
    for (const auto& bytes : storage) {
      views.push_back({bytes.data(), bytes.size()});
    }
    return views;
  }

  DecodedHeader header;
  std::vector<std::vector<std::uint8_t>> storage;
};

ControlSessionToken MakeToken(std::uint8_t discriminator) {
  ControlSessionToken token{};
  token[0] = discriminator;
  return token;
}

ControlSessionToken DifferentToken(const ControlSessionToken& token) {
  auto different = token;
  different[0] ^= 0x80;
  if (!ControlSessionState::IsValidToken(different)) {
    different[1] = 1;
  }
  return different;
}

OwnedWire MakeCommand(
    const ControlSessionToken& receiver_epoch,
    const ControlSessionToken& publisher_session,
    std::int64_t command_index,
    bool claim,
    bool start,
    bool stop,
    bool planner) {
  OwnedWire wire(2);
  wire.Add<std::uint8_t>(
      "receiver_epoch", "u8", {receiver_epoch.size()}, receiver_epoch);
  wire.Add<std::uint8_t>(
      "publisher_session",
      "u8",
      {publisher_session.size()},
      publisher_session);
  wire.AddScalar("command_index", "i64", command_index);
  wire.AddScalar("claim", "bool", static_cast<std::uint8_t>(claim));
  wire.AddScalar("start", "bool", static_cast<std::uint8_t>(start));
  wire.AddScalar("stop", "bool", static_cast<std::uint8_t>(stop));
  wire.AddScalar("planner", "bool", static_cast<std::uint8_t>(planner));
  return wire;
}

OwnedWire MakePlanner(
    const ControlSessionToken& receiver_epoch,
    const ControlSessionToken& publisher_session,
    std::int64_t frame_index,
    float movement_x = 0.0F) {
  OwnedWire wire(2);
  const std::array<float, 3> movement{movement_x, 0.0F, 0.0F};
  const std::array<float, 3> facing{1.0F, 0.0F, 0.0F};
  wire.Add<std::uint8_t>(
      "receiver_epoch", "u8", {receiver_epoch.size()}, receiver_epoch);
  wire.Add<std::uint8_t>(
      "publisher_session",
      "u8",
      {publisher_session.size()},
      publisher_session);
  wire.AddScalar("frame_index", "i64", frame_index);
  wire.AddScalar(
      "mode",
      "i32",
      static_cast<std::int32_t>(LocomotionMode::IDLE));
  wire.Add<float>("movement", "f32", {movement.size()}, movement);
  wire.Add<float>("facing", "f32", {facing.size()}, facing);
  return wire;
}

OwnedWire MakeTokenPose(
    const ControlSessionToken& receiver_epoch,
    const ControlSessionToken& publisher_session,
    std::int64_t frame_index) {
  OwnedWire wire(4);
  std::array<float, 64> token_state{};
  token_state[0] = 0.25F;
  wire.Add<std::uint8_t>(
      "receiver_epoch", "u8", {receiver_epoch.size()}, receiver_epoch);
  wire.Add<std::uint8_t>(
      "publisher_session",
      "u8",
      {publisher_session.size()},
      publisher_session);
  wire.Add<float>(
      "token_state", "f32", {token_state.size()}, token_state);
  wire.AddScalar("frame_index", "i64", frame_index);
  return wire;
}

void DeliverCommand(ZMQManager& manager, OwnedWire& wire) {
  const auto views = wire.Views();
  ZMQWireTestAccess::DeliverCommand(manager, wire.header, views);
}

void DeliverPlanner(ZMQManager& manager, OwnedWire& wire) {
  const auto views = wire.Views();
  ZMQWireTestAccess::DeliverPlanner(manager, wire.header, views);
}

void DeliverPose(ZMQEndpointInterface& endpoint, OwnedWire& wire) {
  const auto views = wire.Views();
  ZMQWireTestAccess::DeliverPose(endpoint, wire.header, views);
}

TEST(ZMQNativeWire, CommandEnforcesSchemaOwnershipReplayAndFailSafeStop) {
  auto session = std::make_shared<ControlSessionState>();
  ZMQManager manager(
      "localhost",
      55991,
      "native_wire_pose",
      "native_wire_command",
      "native_wire_planner",
      false,
      false,
      session);
  const auto receiver_epoch = session->ReceiverEpoch();
  const auto wrong_epoch = DifferentToken(receiver_epoch);
  const auto owner = MakeToken(11);
  const auto other = MakeToken(12);

  auto malformed =
      MakeCommand(receiver_epoch, owner, 0, true, false, false, true);
  malformed.header.fields.back().name = "start";
  DeliverCommand(manager, malformed);
  EXPECT_FALSE(session->HasClaimedPublisher());
  EXPECT_FALSE(ZMQWireTestAccess::LastCommandIndex(manager).has_value());

  auto wrong_epoch_claim =
      MakeCommand(wrong_epoch, owner, 0, true, false, false, true);
  DeliverCommand(manager, wrong_epoch_claim);
  EXPECT_FALSE(session->HasClaimedPublisher());

  auto claim =
      MakeCommand(receiver_epoch, owner, 0, true, false, false, true);
  DeliverCommand(manager, claim);
  EXPECT_TRUE(session->IsClaimedPublisher(receiver_epoch, owner));
  EXPECT_EQ(ZMQWireTestAccess::LastCommandIndex(manager), 0);

  auto takeover =
      MakeCommand(receiver_epoch, other, 1, true, false, false, true);
  DeliverCommand(manager, takeover);
  EXPECT_TRUE(session->IsClaimedPublisher(receiver_epoch, owner));
  EXPECT_FALSE(session->IsClaimedPublisher(receiver_epoch, other));
  EXPECT_EQ(ZMQWireTestAccess::LastCommandIndex(manager), 0);

  auto start =
      MakeCommand(receiver_epoch, owner, 1, false, true, false, true);
  DeliverCommand(manager, start);
  auto accepted = ZMQWireTestAccess::LatestCommand(manager);
  ASSERT_TRUE(accepted.valid);
  EXPECT_TRUE(accepted.start);
  EXPECT_TRUE(accepted.planner);
  EXPECT_EQ(ZMQWireTestAccess::LastCommandIndex(manager), 1);

  auto replay =
      MakeCommand(receiver_epoch, owner, 1, false, true, false, false);
  DeliverCommand(manager, replay);
  accepted = ZMQWireTestAccess::LatestCommand(manager);
  EXPECT_TRUE(accepted.planner);
  EXPECT_EQ(ZMQWireTestAccess::LastCommandIndex(manager), 1);

  auto wrong_owner =
      MakeCommand(receiver_epoch, other, 2, false, true, false, false);
  DeliverCommand(manager, wrong_owner);
  accepted = ZMQWireTestAccess::LatestCommand(manager);
  EXPECT_TRUE(accepted.planner);
  EXPECT_EQ(ZMQWireTestAccess::LastCommandIndex(manager), 1);

  auto wrong_epoch_stop =
      MakeCommand(wrong_epoch, other, 0, false, false, true, false);
  DeliverCommand(manager, wrong_epoch_stop);
  EXPECT_FALSE(ZMQWireTestAccess::LatestCommand(manager).stop);

  auto current_epoch_stop =
      MakeCommand(receiver_epoch, other, 0, false, false, true, false);
  DeliverCommand(manager, current_epoch_stop);
  EXPECT_TRUE(ZMQWireTestAccess::LatestCommand(manager).stop);
  EXPECT_EQ(ZMQWireTestAccess::LastCommandIndex(manager), 1);
}

TEST(ZMQNativeWire, PlannerEnforcesSchemaOwnershipAndReplay) {
  auto session = std::make_shared<ControlSessionState>();
  ZMQManager manager(
      "localhost",
      55992,
      "native_wire_pose",
      "native_wire_command",
      "native_wire_planner",
      false,
      false,
      session);
  const auto receiver_epoch = session->ReceiverEpoch();
  const auto wrong_epoch = DifferentToken(receiver_epoch);
  const auto owner = MakeToken(21);
  const auto other = MakeToken(22);
  ASSERT_EQ(
      session->ClaimPublisher(receiver_epoch, owner),
      ControlSessionState::ClaimResult::CLAIMED);

  auto malformed = MakePlanner(receiver_epoch, owner, 10);
  malformed.header.fields.back().name = "movement";
  DeliverPlanner(manager, malformed);
  EXPECT_FALSE(ZMQWireTestAccess::LastPlannerFrame(manager).has_value());

  auto wrong_epoch_planner = MakePlanner(wrong_epoch, owner, 10);
  DeliverPlanner(manager, wrong_epoch_planner);
  EXPECT_FALSE(ZMQWireTestAccess::LastPlannerFrame(manager).has_value());

  auto wrong_owner_planner = MakePlanner(receiver_epoch, other, 10);
  DeliverPlanner(manager, wrong_owner_planner);
  EXPECT_FALSE(ZMQWireTestAccess::LastPlannerFrame(manager).has_value());

  auto valid = MakePlanner(receiver_epoch, owner, 10);
  DeliverPlanner(manager, valid);
  auto accepted = ZMQWireTestAccess::LatestPlanner(manager);
  ASSERT_TRUE(accepted.valid);
  EXPECT_EQ(accepted.frame_index, 10);
  EXPECT_DOUBLE_EQ(accepted.movement[0], 0.0);
  EXPECT_NE(
      accepted.timestamp, std::chrono::steady_clock::time_point{});

  auto replay = MakePlanner(receiver_epoch, owner, 10, 0.5F);
  DeliverPlanner(manager, replay);
  accepted = ZMQWireTestAccess::LatestPlanner(manager);
  EXPECT_EQ(accepted.frame_index, 10);
  EXPECT_DOUBLE_EQ(accepted.movement[0], 0.0);

  auto advancing = MakePlanner(receiver_epoch, owner, 11, 0.5F);
  DeliverPlanner(manager, advancing);
  accepted = ZMQWireTestAccess::LatestPlanner(manager);
  EXPECT_EQ(accepted.frame_index, 11);
  EXPECT_DOUBLE_EQ(accepted.movement[0], 0.5);
}

TEST(ZMQNativeWire, ManagedPoseEnforcesEnvelopeAndSourceReplay) {
  auto session = std::make_shared<ControlSessionState>();
  const auto receiver_epoch = session->ReceiverEpoch();
  const auto wrong_epoch = DifferentToken(receiver_epoch);
  const auto owner = MakeToken(31);
  const auto other = MakeToken(32);
  ASSERT_EQ(
      session->ClaimPublisher(receiver_epoch, owner),
      ControlSessionState::ClaimResult::CLAIMED);
  ZMQEndpointInterface endpoint(
      "localhost",
      55993,
      "native_wire_pose",
      false,
      false,
      session);

  auto malformed = MakeTokenPose(receiver_epoch, owner, 10);
  malformed.header.fields[1].name = "receiver_epoch";
  DeliverPose(endpoint, malformed);
  EXPECT_EQ(ZMQWireTestAccess::PoseReceiveCount(endpoint), 0);
  EXPECT_FALSE(ZMQWireTestAccess::HasBufferedPose(endpoint));

  auto wrong_epoch_pose = MakeTokenPose(wrong_epoch, owner, 10);
  DeliverPose(endpoint, wrong_epoch_pose);
  EXPECT_EQ(ZMQWireTestAccess::PoseReceiveCount(endpoint), 0);

  auto wrong_owner_pose = MakeTokenPose(receiver_epoch, other, 10);
  DeliverPose(endpoint, wrong_owner_pose);
  EXPECT_EQ(ZMQWireTestAccess::PoseReceiveCount(endpoint), 0);

  auto valid = MakeTokenPose(receiver_epoch, owner, 10);
  DeliverPose(endpoint, valid);
  EXPECT_EQ(ZMQWireTestAccess::PoseReceiveCount(endpoint), 1);
  EXPECT_TRUE(ZMQWireTestAccess::HasBufferedPose(endpoint));
  EXPECT_TRUE(ZMQWireTestAccess::DecodeBufferedTokenPose(endpoint));
  EXPECT_EQ(ZMQWireTestAccess::LastPoseFrame(endpoint), 10);

  auto replay = MakeTokenPose(receiver_epoch, owner, 10);
  DeliverPose(endpoint, replay);
  EXPECT_EQ(ZMQWireTestAccess::PoseReceiveCount(endpoint), 2);
  EXPECT_FALSE(ZMQWireTestAccess::DecodeBufferedTokenPose(endpoint));
  EXPECT_EQ(ZMQWireTestAccess::LastPoseFrame(endpoint), 10);

  auto advancing = MakeTokenPose(receiver_epoch, owner, 11);
  DeliverPose(endpoint, advancing);
  EXPECT_TRUE(ZMQWireTestAccess::DecodeBufferedTokenPose(endpoint));
  EXPECT_EQ(ZMQWireTestAccess::LastPoseFrame(endpoint), 11);

  auto takeover_pose = MakeTokenPose(receiver_epoch, other, 12);
  DeliverPose(endpoint, takeover_pose);
  EXPECT_EQ(ZMQWireTestAccess::PoseReceiveCount(endpoint), 3);
  EXPECT_EQ(ZMQWireTestAccess::LastPoseFrame(endpoint), 11);
}

}  // namespace
