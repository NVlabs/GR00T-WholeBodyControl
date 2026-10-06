#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "inspire/direct_serial_protocol.hpp"

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

bool close(float left, float right) { return std::abs(left - right) < 1e-6F; }

}  // namespace

int main() {
  const inspire::HandVector command{0.F, 0.125F, 0.5F, 0.755F, 0.999F, 1.F};
  const auto request = inspire::serial::make_set_position_request(command);
  require(request.size() == 20, "set request size changed");
  require(request[0] == 0xEB && request[1] == 0x90 && request[2] == 1,
          "set request envelope changed");
  require(request.back() == inspire::serial::checksum(request),
          "set request checksum is wrong");
  require(request[9] == 125 && request[10] == 0, "125/1000 encoding is wrong");
  require(request[13] == 0xF3 && request[14] == 0x02, "755/1000 encoding is wrong");

  const auto get = inspire::serial::make_get_position_request();
  require(get.back() == inspire::serial::checksum(get), "get request checksum is wrong");

  std::array<std::uint8_t, inspire::serial::kAckSize> ack{
      0x90, 0xEB, 1, 0x04, 0x12, 0xCE, 0x05, 0x00, 0x00};
  ack.back() = inspire::serial::checksum(ack);
  inspire::serial::validate_ack(ack);

  const inspire::serial::SpeedVector speed{100, 250, 500, 750, 999, 1000};
  const auto set_speed = inspire::serial::make_set_speed_request(speed);
  require(set_speed[5] == 0xF2 && set_speed[6] == 0x05,
          "speed register address is wrong");
  require(set_speed[7] == 100 && set_speed[8] == 0, "speed encoding is wrong");
  require(set_speed[17] == 0xE8 && set_speed[18] == 0x03,
          "maximum speed encoding is wrong");
  require(set_speed.back() == inspire::serial::checksum(set_speed),
          "set speed checksum is wrong");
  const auto get_speed = inspire::serial::make_get_speed_request();
  require(get_speed[5] == 0xF2 && get_speed[6] == 0x05 && get_speed[7] == 0x0C,
          "get speed request is wrong");

  const auto clear_error = inspire::serial::make_clear_error_request();
  require(clear_error.size() == inspire::serial::kGetRequestSize,
          "clear-error request size changed");
  require(clear_error[3] == 0x04 && clear_error[4] == 0x12,
          "clear-error write envelope is wrong");
  require(clear_error[5] == 0xEC && clear_error[6] == 0x03 && clear_error[7] == 0x01,
          "clear-error register or value is wrong");
  require(clear_error.back() == inspire::serial::checksum(clear_error),
          "clear-error checksum is wrong");

  std::array<std::uint8_t, inspire::serial::kPositionResponseSize> response{};
  response[0] = 0x90;
  response[1] = 0xEB;
  response[2] = 1;
  response[3] = 0x0F;
  for (std::size_t index = 0; index < inspire::kHandDof; ++index) {
    const auto raw = static_cast<std::uint16_t>(std::lround(command[index] * 1000.F));
    response[7 + 2 * index] = static_cast<std::uint8_t>(raw & 0xFFU);
    response[8 + 2 * index] = static_cast<std::uint8_t>((raw >> 8U) & 0xFFU);
  }
  response.back() = inspire::serial::checksum(response);
  const auto decoded = inspire::serial::decode_position_response(response);
  for (std::size_t index = 0; index < inspire::kHandDof; ++index) {
    require(close(decoded[index], command[index]), "position response round trip changed a value");
  }

  auto speed_response = response;
  for (std::size_t index = 0; index < inspire::kHandDof; ++index) {
    speed_response[7 + 2 * index] = static_cast<std::uint8_t>(speed[index] & 0xFFU);
    speed_response[8 + 2 * index] =
        static_cast<std::uint8_t>((speed[index] >> 8U) & 0xFFU);
  }
  speed_response.back() = inspire::serial::checksum(speed_response);
  require(inspire::serial::decode_speed_response(speed_response) == speed,
          "speed response round trip changed a value");

  bool rejected = false;
  try {
    auto corrupt = response;
    corrupt.back() ^= 0x01;
    (void)inspire::serial::decode_position_response(corrupt);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "corrupt position checksum was accepted");

  rejected = false;
  try {
    auto request_header = response;
    request_header[0] = 0xEB;
    request_header[1] = 0x90;
    request_header.back() = inspire::serial::checksum(request_header);
    (void)inspire::serial::decode_position_response(request_header);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "request header was accepted as a response header");

  rejected = false;
  try {
    auto invalid = command;
    invalid[3] = 1.01F;
    (void)inspire::serial::make_set_position_request(invalid);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "out-of-range normalized command was accepted");

  rejected = false;
  try {
    auto invalid_speed = speed;
    invalid_speed[2] = 1001;
    (void)inspire::serial::make_set_speed_request(invalid_speed);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "out-of-range speed was accepted");
  return 0;
}
