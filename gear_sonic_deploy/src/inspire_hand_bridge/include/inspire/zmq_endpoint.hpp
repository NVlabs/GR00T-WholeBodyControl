#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include <msgpack.hpp>

#include "inspire/contract.hpp"

namespace inspire {

inline const msgpack::object* map_value(const msgpack::object_map& map, std::string_view key) {
  for (std::uint32_t i = 0; i < map.size; ++i) {
    const auto& candidate = map.ptr[i];
    if (candidate.key.type == msgpack::type::STR &&
        std::string_view(candidate.key.via.str.ptr, candidate.key.via.str.size) == key) {
      return &candidate.val;
    }
  }
  return nullptr;
}

inline CommandV1 unpack_command(const char* data, std::size_t size) {
  auto handle = msgpack::unpack(data, size);
  const auto& object = handle.get();
  if (object.type != msgpack::type::MAP) throw std::runtime_error("command must be a map");
  const auto& map = object.via.map;
  const auto* schema = map_value(map, "schema");
  const auto* order = map_value(map, "joint_order");
  const auto* sequence = map_value(map, "sequence");
  const auto* timestamp = map_value(map, "source_monotonic_ns");
  const auto* left = map_value(map, "left_q");
  const auto* right = map_value(map, "right_q");
  if (!schema || !order || !sequence || !timestamp || !left || !right) {
    throw std::runtime_error("command is missing a required field");
  }
  if (schema->as<std::string>() != kCommandSchema) throw std::runtime_error("unknown command schema");
  const auto decoded_order = order->as<std::array<std::string, kHandDof>>();
  for (std::size_t i = 0; i < kHandDof; ++i) {
    if (decoded_order[i] != kJointOrder[i]) throw std::runtime_error("wrong joint_order");
  }
  CommandV1 command;
  command.sequence = sequence->as<std::uint64_t>();
  command.source_monotonic_ns = timestamp->as<std::uint64_t>();
  command.left_q = left->as<HandVector>();
  command.right_q = right->as<HandVector>();
  if (const auto* frame = map_value(map, "frame_index")) {
    command.frame_index = frame->as<std::uint64_t>();
  }
  if (!finite(command.left_q) || !finite(command.right_q)) throw std::runtime_error("non-finite value");
  return command;
}

inline msgpack::sbuffer pack_state(const StateV1& state) {
  msgpack::sbuffer buffer;
  msgpack::packer<msgpack::sbuffer> packer(buffer);
  const bool velocities = state.left_dq.has_value() && state.right_dq.has_value();
  const bool command = state.accepted_command_sequence.has_value() &&
                       state.left_command_q.has_value() && state.right_command_q.has_value();
  const bool accepted_frame = command && state.accepted_frame_index.has_value();
  packer.pack_map(8 + (velocities ? 2 : 0) + (command ? 3 : 0) +
                  (accepted_frame ? 1 : 0));
  packer.pack("schema"); packer.pack(kStateSchema);
  packer.pack("sequence"); packer.pack(state.sequence);
  packer.pack("source_monotonic_ns"); packer.pack(state.source_monotonic_ns);
  packer.pack("gateway_monotonic_ns"); packer.pack(state.gateway_monotonic_ns);
  packer.pack("status"); packer.pack(std::string(to_string(state.status)));
  packer.pack("joint_order");
  std::array<std::string, kHandDof> order{};
  for (std::size_t i = 0; i < kHandDof; ++i) order[i] = kJointOrder[i];
  packer.pack(order);
  packer.pack("left_q"); packer.pack(state.left_q);
  packer.pack("right_q"); packer.pack(state.right_q);
  if (velocities) {
    packer.pack("left_dq"); packer.pack(*state.left_dq);
    packer.pack("right_dq"); packer.pack(*state.right_dq);
  }
  if (command) {
    packer.pack("accepted_command_sequence"); packer.pack(*state.accepted_command_sequence);
    packer.pack("left_command_q"); packer.pack(*state.left_command_q);
    packer.pack("right_command_q"); packer.pack(*state.right_command_q);
    if (accepted_frame) {
      packer.pack("accepted_frame_index"); packer.pack(*state.accepted_frame_index);
    }
  }
  return buffer;
}

}  // namespace inspire
