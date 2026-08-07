/**
 * @file observation_config.hpp
 * @brief YAML-driven observation and encoder configuration for the G1 policy.
 *
 * This header provides data structures and a simple parser for the
 * `observation_config.yaml` file that controls which sensor / state
 * observations are fed to the RL policy and (optionally) the encoder.
 *
 * ## Config File Format (simplified YAML)
 *
 * ```yaml
 * observations:
 *   - name: "motion_joint_positions"
 *     enabled: true
 *   - name: "token_state"
 *     enabled: true
 *
 * encoder:
 *   dimension: 64
 *   use_fp16: false
 *   encoder_observations:
 *     - name: "body_joint_positions"
 *       enabled: true
 *   encoder_modes:
 *     - name: "g1"
 *       mode_id: 0
 *       required_observations:
 *         - body_joint_positions
 *         - body_joint_velocities
 * ```
 *
 * ## Key Types
 *
 *   - ObservationConfig – name + enabled flag for one observation.
 *   - EncoderModeConfig – per-mode observation requirements.
 *   - EncoderConfig – dimension, FP16 flag, observation list, mode list.
 *   - FullObservationConfig – bundles observations + encoder config.
 *   - ObservationConfigParser – static parser with default fallback.
 */

#ifndef OBSERVATION_CONFIG_HPP
#define OBSERVATION_CONFIG_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

/**
 * @brief Configuration for a single observation type (name + enabled flag).
 */
struct ObservationConfig {
  std::string name;    ///< Name identifier for the observation type
  bool enabled;        ///< Whether this observation is active in the pipeline
  
  /**
   * @brief Constructs an observation configuration
   * @param n Name of the observation type
   * @param e Whether this observation should be enabled
   */
  ObservationConfig(const std::string& n, bool e) : name(n), enabled(e) {}
};

/**
 * @brief Configuration for a specific encoder mode
 * 
 * Each mode defines which observations it actually needs.
 * Observations not in this list will be zero-filled for this mode.
 */
struct EncoderModeConfig {
  std::string name;                          ///< Name of the mode (e.g., "g1", "teleop")
  int mode_id;                               ///< Numeric mode ID (matches encoder_mode value)
  std::vector<std::string> required_observations;  ///< Observations needed for this mode
  
  EncoderModeConfig() : mode_id(-1) {}
  EncoderModeConfig(const std::string& n, int id) : name(n), mode_id(id) {}
};

/**
 * @brief Configuration structure for the encoder (tokenizer)
 * 
 * This structure defines encoder settings from observation config.
 * The encoder model path is passed as a command-line argument.
 */
struct EncoderConfig {
  int dimension = 0;                                    ///< Token dimension (0 = no encoder)
  bool use_fp16 = false;                                ///< Use FP16 precision for TensorRT
  std::vector<ObservationConfig> encoder_observations;  ///< All possible observations (superset)
  std::vector<EncoderModeConfig> encoder_modes;         ///< Mode-specific observation requirements
};

/**
 * @brief Combined configuration containing observations and optional encoder
 */
struct FullObservationConfig {
  std::vector<ObservationConfig> observations;
  EncoderConfig encoder;
};

/**
 * @brief Simple YAML parser specifically designed for observation configuration files
 * 
 * This class provides static methods to parse YAML configuration files that define
 * which robot observations should be enabled or disabled. It supports a simple YAML
 * format with fallback to default configurations when files are missing or invalid.
 * 
 * Expected YAML format:
 * observations:
 *   - name: "observation_type"
 *     enabled: true/false
 */
class ObservationConfigParser {
public:
  /**
   * @brief Parses full configuration including observations and encoder from a YAML file
   * 
   * @param config_path Path to the YAML configuration file
   * @return FullObservationConfig containing observations and encoder configuration
   * 
   * This method parses both the observations section and optional encoder section.
   */
  static FullObservationConfig ParseFullConfig(const std::string& config_path) {
    FullObservationConfig full_config;
    std::ifstream file(config_path);
    
    if (!file.is_open()) {
      std::cout << "Warning: Could not open observation config file: " << config_path << std::endl;
      std::cout << "Using default observation configuration." << std::endl;
      full_config.observations = GetDefaultConfig();
      return full_config;
    }
    
    std::cout << "Parsing observation config file: " << config_path << std::endl;
    std::string line;
    bool in_observations_section = false;
    bool in_encoder_section = false;
    bool in_encoder_observations_section = false;
    bool in_encoder_modes_section = false;
    bool in_mode_observations_section = false;
    int line_number = 0;
    EncoderModeConfig current_mode;
    
    while (std::getline(file, line)) {
      line_number++;
      
      // Trim leading whitespace
      size_t first_non_space = line.find_first_not_of(" \t");
      if (first_non_space == std::string::npos) {
        line = "";
      } else {
        line = line.substr(first_non_space);
      }
      
      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') continue;
      
      // Check for main sections (must check before parsing items)
      if (line.find("encoder:") == 0 || line == "encoder:") {
        in_observations_section = false;
        in_encoder_section = true;
        in_encoder_observations_section = false;
        std::cout << "Found encoder section at line " << line_number << std::endl;
        continue;
      }
      
      if (line.find("observations:") == 0 || line == "observations:") {
        in_observations_section = true;
        in_encoder_section = false;
        in_encoder_observations_section = false;
        std::cout << "Found observations section at line " << line_number << std::endl;
        continue;
      }
      
      // Parse observations section (stop if we see any top-level section marker)
      if (in_observations_section && line.find("- name:") != std::string::npos) {
        std::string name = ExtractValue(line, "name:");
        if (std::getline(file, line)) {
          line_number++;
          size_t first_non_space = line.find_first_not_of(" \t");
          if (first_non_space != std::string::npos) {
            line = line.substr(first_non_space);
          }
          bool enabled = ExtractBoolValue(line, "enabled:");
          if (!name.empty()) {
            full_config.observations.emplace_back(name, enabled);
            std::cout << "  Parsed observation: " << name << " -> " << (enabled ? "enabled" : "disabled") << std::endl;
          }
        }
      }
      
      // Parse encoder section parameters (key-value format, no dashes)
      if (in_encoder_section && !in_encoder_observations_section && !in_encoder_modes_section) {
        if (line.find("dimension:") != std::string::npos) {
          std::string dim_str = ExtractValue(line, "dimension:");
          try {
            full_config.encoder.dimension = std::stoi(dim_str);
            std::cout << "  Encoder dimension: " << full_config.encoder.dimension << std::endl;
          } catch (...) {}
        }
        else if (line.find("use_fp16:") != std::string::npos) {
          full_config.encoder.use_fp16 = ExtractBoolValue(line, "use_fp16:");
          std::cout << "  Encoder use_fp16: " << (full_config.encoder.use_fp16 ? "true" : "false") << std::endl;
        }
        else if (line.find("encoder_observations:") != std::string::npos) {
          in_encoder_observations_section = true;
          in_encoder_modes_section = false;
          std::cout << "  Found encoder_observations subsection" << std::endl;
        }
        else if (line.find("encoder_modes:") != std::string::npos) {
          in_encoder_modes_section = true;
          in_encoder_observations_section = false;
          std::cout << "  Found encoder_modes subsection" << std::endl;
        }
      }
      
      // Also check for encoder_modes at the same indentation level (after encoder_observations)
      if (in_encoder_section && in_encoder_observations_section && line.find("encoder_modes:") != std::string::npos) {
        in_encoder_modes_section = true;
        in_encoder_observations_section = false;
        std::cout << "  Found encoder_modes subsection (after encoder_observations)" << std::endl;
      }
      
      // Parse encoder_observations subsection
      if (in_encoder_observations_section && line.find("- name:") != std::string::npos) {
        std::string name = ExtractValue(line, "name:");
        if (std::getline(file, line)) {
          line_number++;
          size_t first_non_space = line.find_first_not_of(" \t");
          if (first_non_space != std::string::npos) {
            line = line.substr(first_non_space);
          }
          bool enabled = ExtractBoolValue(line, "enabled:");
          if (!name.empty()) {
            full_config.encoder.encoder_observations.emplace_back(name, enabled);
            std::cout << "    Encoder obs: " << name << " -> " << (enabled ? "enabled" : "disabled") << std::endl;
          }
        }
      }
      
      // Parse encoder_modes subsection
      if (in_encoder_modes_section) {
        // Start of a new mode
        if (line.find("- name:") != std::string::npos) {
          // Save previous mode if it exists
          if (!current_mode.name.empty()) {
            full_config.encoder.encoder_modes.push_back(current_mode);
            std::cout << "    Saved mode '" << current_mode.name << "' with " << current_mode.required_observations.size() << " observations" << std::endl;
          }
          // Start new mode
          current_mode = EncoderModeConfig();
          current_mode.name = ExtractValue(line, "name:");
          in_mode_observations_section = false;
          std::cout << "    Found mode: " << current_mode.name << std::endl;
        }
        else if (line.find("mode_id:") != std::string::npos) {
          std::string id_str = ExtractValue(line, "mode_id:");
          try {
            current_mode.mode_id = std::stoi(id_str);
            std::cout << "      Mode ID: " << current_mode.mode_id << std::endl;
          } catch (...) {
            std::cerr << "      Warning: Invalid mode_id value: " << id_str << std::endl;
          }
        }
        else if (line.find("required_observations:") != std::string::npos) {
          in_mode_observations_section = true;
          std::cout << "      Found required_observations for mode " << current_mode.name << std::endl;
        }
        else if (in_mode_observations_section && line.find("- ") == 0) {
          // Parse observation list item
          std::string obs_name = line.substr(2);  // Skip "- "
          // Remove quotes and commas
          obs_name.erase(std::remove(obs_name.begin(), obs_name.end(), '"'), obs_name.end());
          obs_name.erase(std::remove(obs_name.begin(), obs_name.end(), ','), obs_name.end());
          // Trim whitespace
          size_t first = obs_name.find_first_not_of(" \t");
          size_t last = obs_name.find_last_not_of(" \t");
          if (first != std::string::npos && last != std::string::npos) {
            obs_name = obs_name.substr(first, last - first + 1);
          }
          if (!obs_name.empty()) {
            current_mode.required_observations.push_back(obs_name);
            std::cout << "        - " << obs_name << std::endl;
          }
        }
      }
    }
    
    // Save last mode if parsing encoder_modes
    if (in_encoder_modes_section && !current_mode.name.empty()) {
      full_config.encoder.encoder_modes.push_back(current_mode);
      std::cout << "    Saved mode '" << current_mode.name << "' with " << current_mode.required_observations.size() << " observations" << std::endl;
    }
    
    if (full_config.observations.empty()) {
      std::cout << "Warning: No valid observation configuration found. Using defaults." << std::endl;
      full_config.observations = GetDefaultConfig();
    }
    
    // ========================================================================
    // Validation: Check encoder section when token_state observation is used
    // ========================================================================
    // Logic:
    //   1. If token_state is enabled -> encoder section should exist (dimension > 0)
    //   2. If token_state is disabled -> encoder section is ignored
    // Note: model_path is now a command-line argument, not in config
    // ========================================================================
    bool has_token_state = false;
    for (const auto& obs : full_config.observations) {
      if (obs.name == "token_state" && obs.enabled) {
        has_token_state = true;
        break;
      }
    }
    
    if (has_token_state) {
      // token_state enabled: encoder section should exist with dimension
      if (full_config.encoder.dimension <= 0) {
        std::cerr << "✗ Error: 'token_state' observation is enabled but encoder section is missing or has invalid dimension!" << std::endl;
        std::cerr << "  Please add an 'encoder:' section in the config with 'dimension'" << std::endl;
        std::cerr << "  Or disable 'token_state' observation if you don't need it." << std::endl;
        // Return empty config to signal error
        full_config.observations.clear();
        return full_config;
      }
      std::cout << "✓ Token state observation validated with encoder configuration" << std::endl;
    } else {
      // token_state disabled: encoder section is IGNORED
      if (full_config.encoder.dimension > 0) {
        std::cout << "Note: Encoder section found but 'token_state' observation is not enabled - encoder will be ignored" << std::endl;
        full_config.encoder = EncoderConfig();  // Reset to empty
      }
    }
    
    std::cout << "Successfully parsed " << full_config.observations.size() << " observations" << std::endl;
    if (full_config.encoder.dimension > 0) {
      std::cout << "Encoder config found with " << full_config.encoder.encoder_observations.size() << " input observations" << std::endl;
    }
    
    return full_config;
  }

  /**
   * @brief Parses observation configuration from a YAML file (backward compatibility)
   * 
   * @param config_path Path to the YAML configuration file
   * @return Vector of ObservationConfig objects representing the parsed configuration
   * 
   * This method attempts to parse the specified YAML file and extract observation
   * configurations. If the file cannot be opened or contains no valid configurations,
   * it falls back to a set of default configurations.
   */
  static std::vector<ObservationConfig> ParseConfig(const std::string& config_path) {
    std::vector<ObservationConfig> configs;
    std::ifstream file(config_path);
    
    // Handle file opening errors gracefully by falling back to defaults
    if (!file.is_open()) {
      std::cout << "Warning: Could not open observation config file: " << config_path << std::endl;
      std::cout << "Using default observation configuration." << std::endl;
      return GetDefaultConfig();
    }
    
    std::cout << "Parsing observation config file: " << config_path << std::endl;
    std::string line;
    bool in_observations_section = false;  // Track whether we're inside the observations: section
    int line_number = 0;
    
    // Parse the file line by line using a simple state machine
    while (std::getline(file, line)) {
      line_number++;
      
      // Trim leading whitespace from the line
      size_t first_non_space = line.find_first_not_of(" \t");
      if (first_non_space == std::string::npos) {
        line = "";  // Line contains only whitespace
      } else {
        line = line.substr(first_non_space);
      }
      
      // Skip empty lines and comments
      if (line.empty() || line[0] == '#') continue;
      
      // Look for the main "observations:" section header
      if (line.find("observations:") != std::string::npos) {
        in_observations_section = true;
        std::cout << "Found observations section at line " << line_number << std::endl;
        continue;
      }
      
      // Parse individual observation entries within the observations section
      if (in_observations_section && line.find("- name:") != std::string::npos) {
        // Extract the observation name from current line
        std::string name = ExtractValue(line, "name:");
        
        // Read the next line which should contain the enabled status
        if (std::getline(file, line)) {
          line_number++;
          // Trim whitespace from the enabled line
          size_t first_non_space = line.find_first_not_of(" \t");
          if (first_non_space == std::string::npos) {
            line = "";  // Line contains only whitespace
          } else {
            line = line.substr(first_non_space);
          }
          bool enabled = ExtractBoolValue(line, "enabled:");
          
          // Store the parsed configuration if we got a valid name
          if (!name.empty()) {
            configs.emplace_back(name, enabled);
            std::cout << "  Parsed: " << name << " -> " << (enabled ? "enabled" : "disabled") << std::endl;
          }
        }
      }
    }
    
    // Fall back to defaults if no configurations were successfully parsed
    if (configs.empty()) {
      std::cout << "Warning: No valid observation configuration found in file. Using defaults." << std::endl;
      return GetDefaultConfig();
    }
    
    std::cout << "Successfully parsed " << configs.size() << " observation configurations" << std::endl;
    return configs;
  }

  /**
   * @brief Returns the default observation configuration
   * 
   * @return Vector of default ObservationConfig objects
   * 
   * This overloaded method provides a convenient way to get default configurations
   * without needing to specify a file path. Useful for testing or when no custom
   * configuration is needed.
   */
  static std::vector<ObservationConfig> ParseConfig() {
    return GetDefaultConfig();
  }
   
private:
  /**
   * @brief Extracts a string value from a YAML line after a specified key
   * 
   * @param line The YAML line to parse
   * @param key The key to search for (e.g., "name:")
   * @return The extracted value with quotes and whitespace removed
   * 
   * This helper method finds the specified key in a line and extracts the value
   * that follows it. It automatically removes surrounding quotes and whitespace
   * to return a clean string value.
   */
  static std::string ExtractValue(const std::string& line, const std::string& key) {
    size_t pos = line.find(key);
    if (pos == std::string::npos) return "";  // Key not found
    
    // Move position to after the key
    pos += key.length();
    std::string value = line.substr(pos);
    
    // Remove quotes and trim whitespace from both ends
    size_t first_non_space = value.find_first_not_of(" \t\"");
    if (first_non_space == std::string::npos) {
      return "";  // Value contains only whitespace/quotes
    }
    value = value.substr(first_non_space);
    
    size_t last_non_space = value.find_last_not_of(" \t\"");
    if (last_non_space != std::string::npos) {
      value = value.substr(0, last_non_space + 1);
    }
    
    return value;
  }
  
  /**
   * @brief Extracts a boolean value from a YAML line after a specified key
   * 
   * @param line The YAML line to parse
   * @param key The key to search for (e.g., "enabled:")
   * @return True if the value is "true", false otherwise
   * 
   * This helper method extracts a boolean value by first getting the string value
   * and then comparing it to "true". Any other value (including "false", empty strings,
   * or missing values) will return false.
   */
  static bool ExtractBoolValue(const std::string& line, const std::string& key) {
    std::string value = ExtractValue(line, key);
    return value == "true";  // Only "true" string evaluates to true
  }
  
  /**
   * @brief Provides a hardcoded set of default observation configurations
   * 
   * @return Vector of default ObservationConfig objects for typical robot operations
   * 
   * This method returns a predefined set of observation configurations that are
   * commonly used for robot control and monitoring. It includes joint positions,
   * velocities, orientations, and action history - all enabled by default.
   * Used as a fallback when configuration files are missing or invalid.
   */
  static std::vector<ObservationConfig> GetDefaultConfig() {
    std::cout << "Loading default observation configuration (single frame):" << std::endl;
    std::vector<ObservationConfig> default_configs = {
      {"motion_joint_positions", true},      ///< Joint position data for motion planning
      {"motion_joint_velocities", true},     ///< Joint velocity data for motion control
      {"motion_anchor_orientation", true},   ///< Base orientation for spatial awareness
      {"base_angular_velocity", true},       ///< Angular velocity of the robot base
      {"body_joint_positions", true},        ///< Body joint positions for posture control
      {"body_joint_velocities", true},       ///< Body joint velocities for dynamic control
      {"last_actions", true}                 ///< History of recent actions for temporal context
    };
    
    // Log the default configuration being loaded
    for (const auto& config : default_configs) {
      std::cout << "  Default: " << config.name << " -> " << (config.enabled ? "enabled" : "disabled") << std::endl;
    }
    
    return default_configs;
  }
};

#endif // OBSERVATION_CONFIG_HPP
