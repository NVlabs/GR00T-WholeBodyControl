#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <zmq.hpp>
#include <nlohmann/json.hpp>

#include "../include/input_interface/zmq_packed_message_subscriber.hpp"

/**
 * Test program for ZMQPackedMessageSubscriber connecting to the Python pose estimation server.
 * 
 * This test connects to the Python server and dynamically detects all available fields.
 * Supports Protocol V1 (joint-based) and V2 (SMPL-based) with optional fields.
 * 
 * Usage:
 *   1. Start Python server: python pose_estimation_server_onboard_test.py
 *   2. Run this test: ./zmq_python_sender_test [--verbose] [--host HOST] [--port PORT]
 */

// Dynamic field storage
struct FieldData {
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  size_t expected_size = 0;
  size_t actual_size = 0;
  bool present = false;
  bool valid = false;
  
  // Sample values for printing
  std::vector<double> sample_values;
  
  void print_summary() const {
    std::cout << "    " << name << " (" << dtype << ") shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
      std::cout << shape[i];
      if (i < shape.size() - 1) std::cout << ",";
    }
    std::cout << "] - " << (valid ? "✓ VALID" : "✗ INVALID");
    if (valid && !sample_values.empty()) {
      std::cout << " | sample: [";
      for (size_t i = 0; i < std::min(size_t(3), sample_values.size()); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << sample_values[i];
      }
      if (sample_values.size() > 3) std::cout << ", ...";
      std::cout << "]";
    }
    std::cout << std::endl;
  }
};

struct MessageData {
  int version = 0;
  std::map<std::string, FieldData> fields;
  bool is_valid = false;
  
  void clear() {
    fields.clear();
    is_valid = false;
    version = 0;
  }
  
  void print_summary() const {
    std::cout << "  [MessageData] Protocol V" << version << " - " 
              << (is_valid ? "✓ VALID" : "✗ INVALID") << std::endl;
    std::cout << "  Fields detected: " << fields.size() << std::endl;
    for (const auto& [name, field] : fields) {
      field.print_summary();
    }
  }
};

int main(int argc, char** argv)
{
  // Parse command-line args
  bool verbose = false;
  std::string host = "localhost";
  int port = 5558;
  
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    } else if (arg == "--host") {
      if (i + 1 < argc) {
        host = argv[++i];
      }
    } else if (arg == "--port") {
      if (i + 1 < argc) {
        port = std::stoi(argv[++i]);
      }
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
      std::cout << "Options:" << std::endl;
      std::cout << "  --verbose, -v     Enable verbose output" << std::endl;
      std::cout << "  --host HOST       Server host (default: localhost)" << std::endl;
      std::cout << "  --port PORT       Server port (default: 5558)" << std::endl;
      std::cout << "  --help, -h        Show this help message" << std::endl;
      return 0;
    }
  }

  // Test configuration
  const std::string topic = "pose";

  std::cout << "ZMQ Python Sender Test" << std::endl;
  std::cout << "Connecting to: tcp://" << host << ":" << port << " (topic: " << topic << ")" << std::endl;
  if (verbose) {
    std::cout << "Verbose mode: enabled" << std::endl;
  }
  std::cout << "Waiting for messages... (Press Ctrl+C to stop)\n" << std::endl;

  // Statistics
  int message_count = 0;
  int valid_message_count = 0;
  int error_count = 0;
  std::vector<double> receive_intervals_ms;
  auto last_receive_time = std::chrono::steady_clock::now();
  
  // Track field changes
  std::map<std::string, FieldData> previous_fields;
  int last_protocol_version = -1;

  // Create subscriber with conflate to get latest data
  // Note: subscriber verbose is disabled by default, only enable for deep debugging
  ZMQPackedMessageSubscriber sub(host, port, topic, 
                                 /*timeout_ms=*/1000, 
                                 /*verbose=*/false, 
                                 /*conflate=*/true, 
                                 /*rcv_hwm=*/1);
  
  sub.SetOnDecodedMessage(
    [&](const std::string& t,
        const ZMQPackedMessageSubscriber::DecodedHeader& hdr,
        const std::vector<ZMQPackedMessageSubscriber::BufferView>& bufs) {
      
      message_count++;
      
      // Track receive interval
      auto now = std::chrono::steady_clock::now();
      double interval_ms = std::chrono::duration<double, std::milli>(now - last_receive_time).count();
      if (message_count > 1) {
        receive_intervals_ms.push_back(interval_ms);
      }
      last_receive_time = now;
      
      // Validate header structure
      if (hdr.fields.size() != bufs.size()) {
        std::cerr << "\nERROR: Field count (" << hdr.fields.size() 
                  << ") != buffer count (" << bufs.size() << ")" << std::endl;
        error_count++;
        return;
      }
      
      // Parse message data dynamically
      MessageData msg;
      msg.version = hdr.version;
      bool needs_swap = hdr.NeedsByteSwap();
      bool all_fields_valid = true;
      
      // First pass: detect all fields and calculate expected sizes
      for (size_t i = 0; i < hdr.fields.size(); ++i) {
        const auto& field = hdr.fields[i];
        const auto& buf = bufs[i];
        
        FieldData field_data;
        field_data.name = field.name;
        field_data.dtype = field.dtype;
        field_data.shape = field.shape;
        field_data.actual_size = buf.size;
        field_data.present = true;
        
        // Calculate expected size from shape and dtype
        size_t element_size = 0;
        if (field.dtype == "f32") element_size = sizeof(float);
        else if (field.dtype == "f64") element_size = sizeof(double);
        else if (field.dtype == "i32") element_size = sizeof(int32_t);
        else if (field.dtype == "i64") element_size = sizeof(int64_t);
        else if (field.dtype == "u8" || field.dtype == "bool") element_size = sizeof(uint8_t);
        
        if (element_size > 0) {
          size_t num_elements = 1;
          for (auto dim : field.shape) {
            num_elements *= dim;
          }
          field_data.expected_size = num_elements * element_size;
          field_data.valid = (field_data.actual_size == field_data.expected_size);
        } else {
          std::cerr << "\nWARNING: Unknown dtype '" << field.dtype << "' for field '" << field.name << "'" << std::endl;
          field_data.valid = false;
        }
        
        if (!field_data.valid) {
          all_fields_valid = false;
        }
        
        // Extract sample values for printing (first 3 elements)
        if (field_data.valid && buf.size > 0) {
          size_t num_samples = std::min(size_t(3), buf.size / element_size);
          field_data.sample_values.reserve(num_samples);
          
          if (field.dtype == "f32") {
            for (size_t s = 0; s < num_samples; ++s) {
              float val;
              std::memcpy(&val, static_cast<const uint8_t*>(buf.data) + s * sizeof(float), sizeof(float));
              if (needs_swap) val = byte_swap(val);
              field_data.sample_values.push_back(static_cast<double>(val));
            }
          } else if (field.dtype == "f64") {
            for (size_t s = 0; s < num_samples; ++s) {
              double val;
              std::memcpy(&val, static_cast<const uint8_t*>(buf.data) + s * sizeof(double), sizeof(double));
              if (needs_swap) val = byte_swap(val);
              field_data.sample_values.push_back(val);
            }
          } else if (field.dtype == "i32") {
            for (size_t s = 0; s < num_samples; ++s) {
              int32_t val;
              std::memcpy(&val, static_cast<const uint8_t*>(buf.data) + s * sizeof(int32_t), sizeof(int32_t));
              if (needs_swap) val = byte_swap(val);
              field_data.sample_values.push_back(static_cast<double>(val));
            }
          } else if (field.dtype == "i64") {
            for (size_t s = 0; s < num_samples; ++s) {
              int64_t val;
              std::memcpy(&val, static_cast<const uint8_t*>(buf.data) + s * sizeof(int64_t), sizeof(int64_t));
              if (needs_swap) val = byte_swap(val);
              field_data.sample_values.push_back(static_cast<double>(val));
            }
          } else if (field.dtype == "u8" || field.dtype == "bool") {
            for (size_t s = 0; s < num_samples; ++s) {
              uint8_t val;
              std::memcpy(&val, static_cast<const uint8_t*>(buf.data) + s * sizeof(uint8_t), sizeof(uint8_t));
              field_data.sample_values.push_back(static_cast<double>(val));
            }
          }
        }
        
        msg.fields[field.name] = field_data;
      }
      
      // Validate protocol-specific requirements
      msg.is_valid = all_fields_valid;
      
      // Protocol V1 requires: joint_pos, joint_vel, body_quat, frame_index
      if (msg.version == 1) {
        bool has_required_v1 = msg.fields.count("joint_pos") && 
                               msg.fields.count("joint_vel") && 
                               (msg.fields.count("body_quat") || msg.fields.count("body_quat_w")) && 
                               (msg.fields.count("frame_index") || msg.fields.count("last_smpl_global_frames"));
        if (!has_required_v1) {
          std::cerr << "\nERROR: Protocol V1 missing required fields" << std::endl;
          msg.is_valid = false;
          all_fields_valid = false;
        }
      }
      // Protocol V2 requires: smpl_joints, smpl_pose, body_quat, frame_index
      else if (msg.version == 2) {
        bool has_required_v2 = (msg.fields.count("smpl_joints") || msg.fields.count("body_pos")) && 
                               msg.fields.count("smpl_pose") &&
                               (msg.fields.count("body_quat") || msg.fields.count("body_quat_w")) && 
                               (msg.fields.count("frame_index") || msg.fields.count("last_smpl_global_frames"));
        if (!has_required_v2) {
          std::cerr << "\nERROR: Protocol V2 missing required fields (need: smpl_joints, smpl_pose, body_quat, frame_index)" << std::endl;
          msg.is_valid = false;
          all_fields_valid = false;
        }
      }
      
      // Detect field changes (protocol version or field set)
      bool fields_changed = false;
      if (message_count > 1) {
        // Check if protocol version changed
        if (last_protocol_version != msg.version) {
          std::cerr << "\n⚠⚠⚠ WARNING: Protocol version changed from " << last_protocol_version 
                    << " to " << msg.version << " at message #" << message_count << std::endl;
          fields_changed = true;
        }
        
        // Check if field set changed (names, shapes, or dtypes)
        if (previous_fields.size() != msg.fields.size()) {
          std::cerr << "\n⚠⚠⚠ WARNING: Field count changed from " << previous_fields.size() 
                    << " to " << msg.fields.size() << " at message #" << message_count << std::endl;
          fields_changed = true;
        } else {
          // Same number of fields - check if any changed
          for (const auto& [name, field] : msg.fields) {
            if (previous_fields.count(name) == 0) {
              std::cerr << "\n⚠⚠⚠ WARNING: New field '" << name << "' appeared at message #" << message_count << std::endl;
              fields_changed = true;
              break;
            }
            const auto& prev_field = previous_fields[name];
            if (prev_field.dtype != field.dtype || prev_field.shape != field.shape) {
              std::cerr << "\n⚠⚠⚠ WARNING: Field '" << name << "' changed at message #" << message_count << std::endl;
              std::cerr << "    Old: dtype=" << prev_field.dtype << " shape=[";
              for (size_t j = 0; j < prev_field.shape.size(); ++j) {
                std::cerr << prev_field.shape[j];
                if (j < prev_field.shape.size() - 1) std::cerr << ",";
              }
              std::cerr << "]" << std::endl;
              std::cerr << "    New: dtype=" << field.dtype << " shape=[";
              for (size_t j = 0; j < field.shape.size(); ++j) {
                std::cerr << field.shape[j];
                if (j < field.shape.size() - 1) std::cerr << ",";
              }
              std::cerr << "]" << std::endl;
              fields_changed = true;
              break;
            }
          }
          
          // Check for removed fields
          for (const auto& [name, prev_field] : previous_fields) {
            if (msg.fields.count(name) == 0) {
              std::cerr << "\n⚠⚠⚠ WARNING: Field '" << name << "' removed at message #" << message_count << std::endl;
              fields_changed = true;
              break;
            }
          }
        }
      }
      
      // Update tracking
      previous_fields = msg.fields;
      last_protocol_version = msg.version;
      
      // Print first message OR when fields change
      if (message_count == 1 || fields_changed) {
        if (fields_changed) {
          std::cout << "\n!!! FIELD CONFIGURATION CHANGED !!!" << std::endl;
        }
        
        std::cout << "\n=== " << (message_count == 1 ? "First Message Detected" : "Message Field Update") << " ===" << std::endl;
        std::cout << "Message #" << message_count << std::endl;
        std::cout << "Protocol Version: " << msg.version << std::endl;
        std::cout << "Fields detected: " << msg.fields.size() << std::endl;
        for (const auto& [name, field] : msg.fields) {
          std::cout << "  - " << name << " (" << field.dtype << ") shape=[";
          for (size_t j = 0; j < field.shape.size(); ++j) {
            std::cout << field.shape[j];
            if (j < field.shape.size() - 1) std::cout << ",";
          }
          std::cout << "]";
          if (field.valid) {
            std::cout << " ✓";
          } else {
            std::cout << " ✗ (size mismatch: expected " << field.expected_size 
                      << " bytes, got " << field.actual_size << ")";
          }
          std::cout << std::endl;
        }
        std::cout << "==============================\n" << std::endl;
      }
      
      // Print results
      if (msg.is_valid) {
        valid_message_count++;
        if (verbose) {
          std::cout << "\n[Message #" << message_count << " - interval: " << interval_ms << " ms]" << std::endl;
          msg.print_summary();
        } else if (message_count % 30 == 0) {
          // Print periodic update without verbose mode
          std::cout << "\r[Received: " << message_count << " msgs (V" << msg.version 
                    << ", " << msg.fields.size() << " fields), Latest interval: " 
                    << static_cast<int>(interval_ms) << " ms]" << std::flush;
        }
      } else {
        std::cerr << "\nERROR: Message validation failed" << std::endl;
        msg.print_summary();
        error_count++;
      }
    });

  sub.Start();

  // Run until interrupted
  try {
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      
      // Print periodic status (every 5 seconds)
      static int last_status_count = 0;
      if (message_count > 0 && message_count - last_status_count >= 30) {
        last_status_count = message_count;
        
        if (!receive_intervals_ms.empty()) {
          double avg_interval = std::accumulate(receive_intervals_ms.begin(), receive_intervals_ms.end(), 0.0) 
                                / receive_intervals_ms.size();
          double avg_fps = 1000.0 / avg_interval;
          std::cout << "\n[Status] Received: " << message_count << " msgs (" 
                    << valid_message_count << " valid, " << error_count << " errors) | "
                    << "Avg: " << avg_fps << " Hz" << std::endl;
        } else {
          std::cout << "\n[Status] Received: " << message_count << " msgs (" 
                    << valid_message_count << " valid, " << error_count << " errors)" << std::endl;
        }
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
  }

  // Cleanup
  std::cout << "\n\nStopping..." << std::endl;
  sub.Stop();

  // Print final statistics
  std::cout << "\n=== Test Results ===" << std::endl;
  std::cout << "Total messages: " << message_count << " (" << valid_message_count << " valid, " 
            << error_count << " errors)" << std::endl;
  
  if (!receive_intervals_ms.empty()) {
    double min_interval = *std::min_element(receive_intervals_ms.begin(), receive_intervals_ms.end());
    double max_interval = *std::max_element(receive_intervals_ms.begin(), receive_intervals_ms.end());
    double avg_interval = std::accumulate(receive_intervals_ms.begin(), receive_intervals_ms.end(), 0.0) 
                          / receive_intervals_ms.size();
    double avg_fps = 1000.0 / avg_interval;
    
    std::cout << "Receive rate: " << avg_fps << " Hz (avg: " << avg_interval << " ms, "
              << "min: " << min_interval << " ms, max: " << max_interval << " ms)" << std::endl;
  }
  std::cout << "====================" << std::endl;

  return (error_count > 0) ? 1 : 0;
}

