#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include "../include/input_interface/zmq_packed_message_subscriber.hpp"

static void run_local_publisher(const std::string &bind_endpoint,
                                const std::string &topic,
                                int num_messages,
                                int interval_ms,
                                bool ramp_prefix)
{
  constexpr size_t HEADER_SIZE = 1024;
  
  try {
    zmq::context_t ctx(1);
    zmq::socket_t pub(ctx, zmq::socket_type::pub);
    pub.set(zmq::sockopt::sndhwm, 10);
    pub.bind(bind_endpoint);

    // Give subscribers time to connect
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    for (int i = 0; i < num_messages; ++i) {
      // Build JSON header with 3 fields: index, timestamp_ns, and fake_positions
      nlohmann::json header = {
        {"v", 1},
        {"endian", "le"},
        {"fields", nlohmann::json::array({
          nlohmann::json{{"name","index"},{"dtype","i32"},{"shape",{1}}},
          nlohmann::json{{"name","timestamp_ns"},{"dtype","i64"},{"shape",{1}}},
          nlohmann::json{{"name","fake_positions"},{"dtype","f32"},{"shape",{10,3}}}
        })}
      };

      std::string header_str = header.dump();
      
      // Prepare data
      int32_t idx = ramp_prefix ? static_cast<int32_t>(i) : 0;
      const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
      int64_t ts_ns = static_cast<int64_t>(now_ns);
      
      // Fake positions: 10x3 float array (120 bytes)
      std::vector<float> fake_positions(10 * 3);
      for (size_t j = 0; j < fake_positions.size(); ++j) {
        fake_positions[j] = static_cast<float>(i) + static_cast<float>(j) * 0.1f;
      }

      // Pack into single frame: [topic_prefix][1024-byte JSON header][fields...]
      const size_t packed_size = topic.size() + HEADER_SIZE + sizeof(idx) + sizeof(ts_ns) 
                                  + fake_positions.size() * sizeof(float);
      std::vector<unsigned char> packed_data(packed_size, 0);
      
      size_t offset = 0;
      
      // Copy topic prefix
      std::memcpy(packed_data.data() + offset, topic.data(), topic.size());
      offset += topic.size();
      
      // Copy JSON header (null-pad to HEADER_SIZE)
      std::memcpy(packed_data.data() + offset, header_str.data(), std::min(header_str.size(), HEADER_SIZE));
      offset += HEADER_SIZE;
      
      // Copy binary data (in order matching header.fields)
      std::memcpy(packed_data.data() + offset, &idx, sizeof(idx));
      offset += sizeof(idx);
      std::memcpy(packed_data.data() + offset, &ts_ns, sizeof(ts_ns));
      offset += sizeof(ts_ns);
      std::memcpy(packed_data.data() + offset, fake_positions.data(), fake_positions.size() * sizeof(float));

      // Send as single frame (no separate topic frame)
      zmq::message_t packed_msg(packed_data.size());
      std::memcpy(packed_msg.data(), packed_data.data(), packed_data.size());

      pub.send(packed_msg, zmq::send_flags::none);

      if (i % 20 == 0) {
        std::cout << "[TestPublisher] Sent idx=" << idx << " (packed_bytes=" << packed_size << ")" << std::endl;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
    }
  } catch (const zmq::error_t &e) {
    std::cerr << "[TestPublisher] ZMQ error: " << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "[TestPublisher] Error: " << e.what() << std::endl;
  }
}

int main(int argc, char** argv)
{
  // Parse command-line args
  bool use_conflate = false;
  if (argc > 1 && std::string(argv[1]) == "--conflate") {
    use_conflate = true;
  }

  // Test configuration
  const std::string host = "127.0.0.1";
  const int port = 5557;
  const std::string topic = "pose";
  const std::string endpoint = "tcp://" + host + ":" + std::to_string(port);

  std::cout << "[Test] Mode: " << (use_conflate ? "CONFLATE" : "NO-CONFLATE") << std::endl;
  std::cout << "[Test] Starting local ZMQ PUB at " << endpoint << std::endl;
  
  // Start the publisher first
  std::thread pub_thread(run_local_publisher, endpoint, topic, /*num_messages=*/400,
                         /*interval_ms=*/10, /*ramp_prefix=*/true);

  // Delay before starting subscriber to simulate late joiner
  std::this_thread::sleep_for(std::chrono::milliseconds(500));

  // Start the subscriber with configurable conflate mode
  std::cout << "[Test] Creating subscriber..." << std::endl;
  ZMQPackedMessageSubscriber sub(host, port, topic, /*timeout_ms=*/100, /*verbose=*/true, 
                                 use_conflate, /*rcv_hwm=*/ use_conflate ? 1 : -1);
  std::vector<int> received_indices;
  std::vector<double> latencies_ms;
  
  sub.SetOnDecodedMessage(
    [&](const std::string& t,
        const ZMQPackedMessageSubscriber::DecodedHeader& hdr,
        const std::vector<ZMQPackedMessageSubscriber::BufferView>& bufs){
      if (bufs.size() >= 3 && bufs[0].size >= sizeof(int32_t) && bufs[1].size >= sizeof(int64_t)) {
        int32_t idx;
        int64_t ts_ns;
        std::memcpy(&idx, bufs[0].data, sizeof(int32_t));
        std::memcpy(&ts_ns, bufs[1].data, sizeof(int64_t));
        
        received_indices.push_back(static_cast<int>(idx));
        const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count();
        double latency_ms = static_cast<double>(now_ns - ts_ns) / 1e6;
        latencies_ms.push_back(latency_ms);
        
        // Decode first few values from fake_positions to verify
        if (received_indices.size() % 20 == 0 && bufs[2].size >= 3 * sizeof(float)) {
          float pos[3];
          std::memcpy(pos, bufs[2].data, 3 * sizeof(float));
          std::cout << "[Received] idx=" << idx << " latency_ms=" << latency_ms 
                    << " pos[0]=" << pos[0] << "," << pos[1] << "," << pos[2] << std::endl;
        }
      }
    });
  std::cout << "[Test] Starting subscriber..." << std::endl;
  sub.Start();
  std::cout << "[Test] Subscriber started, waiting 2s..." << std::endl;

  // Let it run for a while
  std::this_thread::sleep_for(std::chrono::milliseconds(2000));

  // Stop subscriber and join publisher
  sub.Stop();
  if (pub_thread.joinable()) pub_thread.join();

  // Print results
  std::cout << "\n========== TEST RESULTS ==========" << std::endl;
  std::cout << "Mode: " << (use_conflate ? "CONFLATE" : "NO-CONFLATE") << std::endl;
  if (received_indices.empty()) {
    std::cout << "No messages received" << std::endl;
  } else {
    std::cout << "First index: " << received_indices.front() << std::endl;
    std::cout << "Last index: " << received_indices.back() << std::endl;
    std::cout << "Total received: " << received_indices.size() << std::endl;
    
    // Check for gaps (skipped indices) to verify conflate behavior
    int gaps = 0;
    for (size_t i = 1; i < received_indices.size(); ++i) {
      int gap = received_indices[i] - received_indices[i-1];
      if (gap > 1) gaps += (gap - 1);
    }
    std::cout << "Skipped indices: " << gaps << std::endl;
    
    if (!latencies_ms.empty()) {
      double minv = *std::min_element(latencies_ms.begin(), latencies_ms.end());
      double maxv = *std::max_element(latencies_ms.begin(), latencies_ms.end());
      double avg = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) / latencies_ms.size();
      std::cout << "Latency [min/avg/max] ms: " << minv << " / " << avg << " / " << maxv << std::endl;
    }
  }
  std::cout << "==================================" << std::endl;

  std::cout << "[Test] Done." << std::endl;
  return 0;
}
