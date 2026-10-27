#pragma once

#include <cstdlib>
#include <functional>
#include <random>
#include <string>

#include "inspire/zmq_endpoint.hpp"

namespace inspire {

// Optional operator control, separate from the policy's ordinary action schema.
// All methods run on the Gateway's existing command/state thread.
class OperatorSession {
 public:
  explicit OperatorSession(bool enabled = enabled_from_environment(), std::string instance = "")
      : enabled_(enabled), instance_(instance.empty() ? new_id() : std::move(instance)) {}

  static bool enabled_from_environment() {
    const auto* value = std::getenv("G1_RL_SESSION_CONTROL");
    return value && std::string(value) == "1";
  }

  void observe(const std::string& status) {
    if (status == "FAULT" || status == "STALE") fault_latched_ = true;
  }

  // Only the explicit local RESET command may clear a fault or release ownership.
  void manual_reset() {
    fault_latched_ = false;
    managed_ = active_ = false;
    session_.clear();
    ++generation_;
  }

  // true means consumed (operator request or rejected old-session action).
  bool consume(const char* data, std::size_t size, const std::string& status,
               bool healthy, const std::function<bool(const std::string&)>& execute) {
    observe(status);  // Capture a fault before an ordinary STOP can replace it with DISARMED.
    auto decoded = msgpack::unpack(data, size);
    const auto& object = decoded.get();
    if (object.type != msgpack::type::MAP) throw std::runtime_error("command must be a map");
    const auto& map = object.via.map;
    const auto* schema = map_value(map, "schema");
    if (!schema || schema->as<std::string>() != "g1.operator.command.v1") {
      const auto* owner = map_value(map, "operator_session_id");
      // Legacy has no ownership field. A prior session cannot become legacy
      // merely because the operator RESET the Gateway or restarted its process.
      if (!managed_) return owner != nullptr;
      return !active_ || !owner || owner->as<std::string>() != session_;
    }
    const auto* request = map_value(map, "request_id");
    const auto* instance = map_value(map, "instance_id");
    const auto* generation = map_value(map, "generation");
    const auto* session = map_value(map, "session_id");
    const auto* operation = map_value(map, "operation");
    if (!request || !instance || !generation || !session || !operation)
      throw std::runtime_error("incomplete operator request");
    const auto id = request->as<std::string>();
    const auto owner = session->as<std::string>();
    const auto op = operation->as<std::string>();
    if (!valid_id(id) || !valid_id(owner)) throw std::runtime_error("invalid operator identity");
    // A duplicate cannot repeat a reset or a physical hold command.
    if (id == request_) return true;
    request_ = id;
    ok_ = false;
    error_.clear();
    if (!enabled_) error_ = "operator_control_disabled";
    else if (instance->as<std::string>() != instance_ ||
             (op != "DISARM" && generation->as<std::uint64_t>() != generation_))
      error_ = "stale_operator_request";
    else if (op != "PREPARE" && op != "DISARM") error_ = "unknown_operator_operation";
    else if (op == "PREPARE" && (fault_latched_ || !healthy || status != "DISARMED"))
      error_ = "prepare_requires_healthy_disarmed_without_fault";
    else if (op == "PREPARE" && owner == session_) error_ = "new_episode_identity_required";
    else if (op == "DISARM" && (!managed_ || owner != session_)) error_ = "wrong_operator_session";
    if (!error_.empty()) return true;

    ++generation_;  // Consume the permission before any potentially failing work.
    managed_ = true;
    active_ = false;
    if (op == "PREPARE") session_ = owner;
    try {
      ok_ = execute(op);
    } catch (...) {
      fault_latched_ = true;
      error_ = "operator_execution_failed";
      throw;
    }
    if (!ok_) {
      fault_latched_ = true;
      error_ = "operator_execution_failed";
    }
    active_ = ok_ && op == "PREPARE";
    return true;
  }

  msgpack::sbuffer state(msgpack::sbuffer original) const {
    auto decoded = msgpack::unpack(original.data(), original.size());
    const auto& map = decoded.get().via.map;
    msgpack::sbuffer result;
    msgpack::packer<msgpack::sbuffer> packer(result);
    packer.pack_map(map.size + 1);
    for (std::uint32_t i = 0; i < map.size; ++i) {
      packer.pack(map.ptr[i].key);
      packer.pack(map.ptr[i].val);
    }
    packer.pack("operator_session");
    packer.pack_map(9);
    packer.pack("enabled"); packer.pack(enabled_);
    packer.pack("instance_id"); packer.pack(instance_);
    packer.pack("generation"); packer.pack(generation_);
    packer.pack("request_id"); packer.pack(request_);
    packer.pack("session_id"); packer.pack(session_);
    packer.pack("active"); packer.pack(active_);
    packer.pack("ok"); packer.pack(ok_);
    packer.pack("error"); packer.pack(error_);
    packer.pack("fault_latched"); packer.pack(fault_latched_);
    return result;
  }

 private:
  static bool valid_id(const std::string& id) {
    return id.size() == 32 && id.find_first_not_of("0123456789abcdef") == std::string::npos;
  }
  static std::string new_id() {
    std::random_device random;
    std::string id;
    for (int i = 0; i < 32; ++i) id += "0123456789abcdef"[random() % 16];
    return id;
  }
  bool enabled_, managed_{false}, active_{false}, fault_latched_{false}, ok_{false};
  std::string instance_, request_, session_, error_;
  std::uint64_t generation_{0};
};

}  // namespace inspire
