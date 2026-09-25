/**
 * @file arm_compliance_subscriber.hpp
 * @brief ZMQ SUB thread that feeds JSON commands into arm_compliance::Controller.
 *
 * Connects to tcp://<host>:<port> (the publisher binds), subscribes to <topic>,
 * and expects single-frame messages of the form "<topic> <json>", e.g.
 *
 *     compliance {"profile": "HANDSHAKE"}
 *
 * which is what pyzmq's  socket.send_string(f"{topic} {json.dumps(cmd)}")  produces.
 * A two-frame message [topic, json] is accepted as well.
 */
#pragma once

#include <atomic>
#include <iostream>
#include <string>
#include <thread>

#include <zmq.hpp>

#include "arm_compliance.hpp"

namespace arm_compliance {

class Subscriber {
  public:
    explicit Subscriber(Controller& controller) : controller_(controller), context_(1) {}

    Subscriber(const Subscriber&) = delete;
    Subscriber& operator=(const Subscriber&) = delete;

    ~Subscriber() { Stop(); }

    void Start() {
      if (running_.exchange(true)) return;
      thread_ = std::thread(&Subscriber::Run, this);
    }

    void Stop() {
      if (!running_.exchange(false)) return;
      if (thread_.joinable()) thread_.join();
    }

  private:
    void Run() {
      const Config& cfg = controller_.config();
      const std::string endpoint = "tcp://" + cfg.host + ":" + std::to_string(cfg.port);
      try {
        zmq::socket_t socket(context_, zmq::socket_type::sub);
        socket.set(zmq::sockopt::rcvtimeo, 100);  // ms, so Stop() is responsive
        socket.set(zmq::sockopt::linger, 0);
        socket.set(zmq::sockopt::subscribe, cfg.topic);
        socket.connect(endpoint);
        std::cout << "[ArmCompliance] Listening on " << endpoint << " topic '" << cfg.topic << "'" << std::endl;

        while (running_) {
          zmq::message_t msg;
          auto res = socket.recv(msg, zmq::recv_flags::none);
          if (!res) continue;  // timeout

          std::string text = msg.to_string();
          std::string payload;
          if (msg.more()) {
            // Two-frame form: [topic][json]
            zmq::message_t body;
            if (!socket.recv(body, zmq::recv_flags::none)) continue;
            payload = body.to_string();
          } else {
            const auto space = text.find(' ');
            if (space == std::string::npos) continue;
            payload = text.substr(space + 1);
          }

          Command cmd;
          std::string err;
          if (!controller_.Parse(payload, cmd, err)) {
            std::cerr << "[ArmCompliance] Ignoring command: " << err << " | " << payload << std::endl;
            continue;
          }
          controller_.SetCommand(cmd);
        }
      } catch (const zmq::error_t& e) {
        std::cerr << "[ArmCompliance] ZMQ error: " << e.what()
                  << " — compliance commands disabled, holding current gains." << std::endl;
      }
    }

    Controller& controller_;
    zmq::context_t context_;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

}  // namespace arm_compliance
