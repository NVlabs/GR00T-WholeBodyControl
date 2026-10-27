#include <cassert>
#include <iostream>
#include <string>

#include "inspire/operator_session.hpp"

using inspire::OperatorSession;
const std::string instance(32, 'a'), first(32, 'b'), second(32, 'c');

msgpack::sbuffer request(const std::string& operation, int id, std::uint64_t generation,
                        const std::string& session = first, const std::string& boot = instance) {
  msgpack::sbuffer result;
  msgpack::packer<msgpack::sbuffer> p(result);
  p.pack_map(6);
  p.pack("schema"); p.pack("g1.operator.command.v1");
  p.pack("operation"); p.pack(operation);
  p.pack("request_id"); p.pack(std::string(31, '0') + "0123456789abcdef"[id]);
  p.pack("generation"); p.pack(generation);
  p.pack("instance_id"); p.pack(boot);
  p.pack("session_id"); p.pack(session);
  return result;
}

template <typename T> T receipt(const OperatorSession& op, const std::string& key) {
  const auto packed = op.state(inspire::pack_state(inspire::StateV1{}));
  auto handle = msgpack::unpack(packed.data(), packed.size());
  const auto* meta = inspire::map_value(handle.get().via.map, "operator_session");
  return inspire::map_value(meta->via.map, key)->as<T>();
}

bool action(OperatorSession& op, const std::string& session, const std::string& status = "ARMED") {
  msgpack::sbuffer result;
  msgpack::packer<msgpack::sbuffer> p(result);
  p.pack_map(session.empty() ? 1 : 2);
  p.pack("schema"); p.pack("fixture.action");
  if (!session.empty()) { p.pack("operator_session_id"); p.pack(session); }
  return !op.consume(result.data(), result.size(), status, true,
                     [](const std::string&) { assert(false); return false; });
}

int main() {
  int executed = 0;
  auto apply = [&](const std::string&) { ++executed; return true; };
  auto send = [&](OperatorSession& op, const msgpack::sbuffer& req,
                  const std::string& status = "DISARMED", bool healthy = true) {
    assert(op.consume(req.data(), req.size(), status, healthy, apply));
  };
  OperatorSession disabled(false, instance);
  send(disabled, request("PREPARE", 1, 0));
  assert(!receipt<bool>(disabled, "ok") && executed == 0);
  assert(action(disabled, ""));  // Legacy is unchanged when not managed.
  assert(!action(disabled, first));  // Old tagged packets are not legacy after reboot.

  OperatorSession op(true, instance);
  send(op, request("PREPARE", 1, 0), "RUNNING");
  send(op, request("PREPARE", 2, 0), "DISARMED", false);
  send(op, request("PREPARE", 3, 0, first, second));  // Wrong Gateway instance.
  assert(executed == 0);
  send(op, request("PREPARE", 4, 0));
  assert(receipt<bool>(op, "ok") && executed == 1);
  send(op, request("PREPARE", 4, 0));  // Duplicate: no reset/re-arm.
  assert(executed == 1 && receipt<std::uint64_t>(op, "generation") == 1);
  assert(action(op, first) && !action(op, second) && !action(op, ""));
  send(op, request("DISARM", 5, 0), "ARMED");  // Lost prepare ACK can still be cancelled.
  assert(executed == 2 && !receipt<bool>(op, "active"));
  assert(!action(op, first));  // Delayed ordinary actions stay revoked.
  send(op, request("PREPARE", 6, 0, second));  // Delayed old permission cannot re-arm.
  assert(executed == 2 && !receipt<bool>(op, "ok"));
  send(op, request("PREPARE", 7, 2, second));
  assert(executed == 3 && action(op, second) && !action(op, first));
  send(op, request("DISARM", 8, 3, first), "ARMED");
  assert(executed == 3 && receipt<bool>(op, "active"));  // Old owner cannot stop new one.

  assert(action(op, second, "FAULT"));  // Ordinary STOP path must latch the prior fault.
  send(op, request("DISARM", 9, 3, second), "DISARMED");
  send(op, request("PREPARE", 10, 4, first));
  assert(executed == 4 && !receipt<bool>(op, "ok"));  // STOP cannot erase fault history.
  op.manual_reset();
  assert(!receipt<bool>(op, "fault_latched") && action(op, ""));
  assert(!action(op, second));  // Old tagged actions stay rejected after a local RESET.

  OperatorSession failed(true, instance);
  auto req = request("PREPARE", 1, 0);
  assert(failed.consume(req.data(), req.size(), "DISARMED", true,
                        [](const std::string&) { return false; }));
  assert(receipt<bool>(failed, "fault_latched") && !action(failed, first));
  std::cout << "operator protocol: opt-in, state gates, replay, ownership, faults PASS\n";
}
