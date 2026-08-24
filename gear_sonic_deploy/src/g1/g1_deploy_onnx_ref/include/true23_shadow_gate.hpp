#pragma once

// Fail-closed contract for the first native G1 rev-1.0 23-DoF deployment
// surface.  Only read-only shadow operation is authorized.  This header has no
// Unitree SDK dependency so its metadata, shape, mode, and permutation rules
// can be tested without DDS or a robot.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

namespace gear_sonic::true23 {

inline constexpr int kArtifactSchemaVersion = 3;
inline constexpr int kSimulationSchemaVersion = 2;
inline constexpr int kRequiredModeMachine = 4;
inline constexpr int kOnnxOpset = 13;
inline constexpr int kEncoderInputDim = 267;
inline constexpr int kEncoderOutputDim = 64;
inline constexpr int kDecoderInputDim = 994;
inline constexpr int kDecoderOutputDim = 23;
inline constexpr int kHistoryLength = 10;
inline constexpr int kStableModeSamples = 5;
inline constexpr std::string_view kRobotModel = "g1_23dof_rev_1_0";
inline constexpr std::string_view kArtifactKind =
    "g1_23dof_validated_teleop_encoder_decoder_onnx_pair";
inline constexpr int kMujocoCandidateSchemaVersion = 1;
inline constexpr std::string_view kMujocoCandidateKind =
    "g1_true23_mujoco_candidate_onnx_pair";
inline constexpr std::string_view kMujocoCandidateEmbeddedKind =
    "g1_true23_mujoco_candidate_onnx";
inline constexpr std::string_view kMujocoCandidateStage =
    "mujoco_candidate";
inline constexpr int kMujocoPromotionSchemaVersion = 1;
inline constexpr std::string_view kMujocoPromotionKind =
    "g1_true23_mujoco_promoted_onnx_pair";
inline constexpr std::string_view kMujocoPromotedStage =
    "mujoco_sim2sim_promoted";
inline constexpr std::string_view kMujocoProducerKind =
    "g1_true23_mujoco_sim2sim_runner";
inline constexpr std::string_view kMujocoVersion = "3.2.3";
inline constexpr std::string_view kMujocoCandidateSourceKind =
    "paired_onnx_trained_candidate";
inline constexpr std::string_view kPinnedUnitreeRepository =
    "https://github.com/unitreerobotics/unitree_ros.git";
inline constexpr std::string_view kPinnedUnitreeRevision =
    "f3772ce54c56ef2d34c6aee8100bc768896c7d19";
inline constexpr std::string_view kPinnedUnitreeRootRelpath =
    "robots/g1_description";
inline constexpr std::string_view kPinnedUnitreeTextNormalization =
    "crlf_to_lf_and_ensure_final_newline_v1";
inline constexpr int kPinnedUnitreeFileCount = 29;
inline constexpr std::int64_t kPinnedUnitreeTotalBytes = 25211170;
inline constexpr std::string_view kPinnedUnitreeManifestSha256 =
    "a98562b34a591fd26a2f4024d84454aa0d3f40ca9067e4d17d900d00f18a492b";
inline constexpr std::string_view kPinnedUnitreeUrdfSha256 =
    "e2f55a541a485d486b376b752734cd1912c3b1b6e74f57e89e2e68691b5aa523";
inline constexpr std::string_view kPinnedUnitreeMjcfSha256 =
    "ea1ce67705253e73a91f9587aa70a34aaa2d17943517b2fe5ac209283b2c9e0c";
inline constexpr std::string_view kPinnedRobotConfigSha256 =
    "edaea59dc404e0b830e495e590d2ff1534775de03212fbf2b1b116b29aa9b11f";
inline constexpr std::string_view kPinnedMujocoRunnerSha256 =
    "be956d4da52cdbf70e72abb6fa82d7f9b7dd1d86fb0a9c678f58fc6384445226";
inline constexpr std::string_view kPinnedMujocoRuntimeSha256 =
    "0897a9c7ab77fd611c3869036b3b28976ccf3cebd2ef6fa4d585688658373e16";
inline constexpr std::string_view kPinnedMujocoConfigSha256 =
    "1ffce4afaeb82e20323e24cf4645a98259d9378ba5936e9d9d52e426241a25cf";
inline constexpr std::string_view kPinnedMujocoPhysicsContractSha256 =
    "666556144bdbf060d7b0d33ed1b3b58db999d04a1b42dc773505d989a0df9ce1";
inline constexpr std::string_view kObservationLayout =
    "canonical_il29_fixed_slots_v1";
inline constexpr std::string_view kDecoderOutputLayout =
    "native_physx_il23_bfs_v1";
inline constexpr std::string_view kOnnxMetadataKey = "g1_23dof_artifact";
inline constexpr std::string_view kEncoderRole = "teleop_encoder";
inline constexpr std::string_view kDecoderRole = "true23_decoder";
inline constexpr std::string_view kEncoderInputName = "teleop_obs";
inline constexpr std::string_view kEncoderOutputName = "token";
inline constexpr std::string_view kDecoderInputName = "obs_dict";
inline constexpr std::string_view kDecoderOutputName = "action";
inline constexpr std::string_view kNormalReferenceProfile =
    "true23_step5_0p1s";
inline constexpr std::string_view kLowLatencyReferenceProfile =
    "released_low_latency_step1_0p02s";
inline constexpr std::string_view kCausalHistoryReferenceProfile =
    "true23_causal_step1_history_0p02s_v1";
inline constexpr std::string_view kCausalHistoryContractSha256 =
    "bd046467325fe7f7f585fd692f01223ed7a3b2742c51ed414072c98fe12806f7";
inline constexpr std::string_view kNormalReleaseSha256 =
    "e6bdab3f64a39336b3d41877d4f497d05f58af275f288ec0e6746c283ded8909";
inline constexpr std::string_view kLowLatencyReleaseSha256 =
    "0031ae7db24747445d6eb7c27697640973a837546f0b8763e775143c47d4507c";
inline constexpr std::string_view kLowLatencyReleaseRevision =
    "7c90a56cfe04788c4f041daeef5b1e12930675ad";
inline constexpr std::string_view kNormalInitialPolicySha256 =
    "c247e5cf8bf06bc954db314013cba5ed8b56b6fe4c9a952c19f053583714f0bc";
inline constexpr std::string_view kLowLatencyInitialPolicySha256 =
    "39049d5018608f198dee1d2ea5e0f465d212dc4572c036d21378a36f80c1fc2f";
inline constexpr std::string_view kMotionDatasetRepository =
    "bones-studio/seed";
inline constexpr std::string_view kMotionDatasetRevision =
    "2f59b2077b9da34dd4e43618e705c7cb962c9a66";
inline constexpr std::string_view kMotionDatasetArchiveRelpath =
    "g1.tar.gz";
inline constexpr std::int64_t kMotionDatasetArchiveSizeBytes =
    23499973647;
inline constexpr std::string_view kMotionDatasetArchiveSha256 =
    "52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a";
inline constexpr std::string_view kProcessedMotionDatasetRootRelpath =
    "data/motion_lib_bones_seed/robot_filtered";
inline constexpr int kMinimumTrainingUpdates = 50;
inline constexpr std::array<double, 10> kNormalReferenceOffsets = {
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
};
inline constexpr std::array<double, 10> kLowLatencyReferenceOffsets = {
    0.0, 0.02, 0.04, 0.06, 0.08,
    0.1, 0.12, 0.14, 0.16, 0.18,
};

// Compact hardware/MuJoCo order -> actual Unitree motor slot.
inline constexpr std::array<int, 23> kHardwareJointIds = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    15, 16, 17, 18, 19, 22, 23, 24, 25, 26,
};
inline constexpr std::array<int, 6> kExcludedHardwareJointIds = {
    13, 14, 20, 21, 27, 28,
};

// hardware_compact[j] = native_physx[kNativeToHardwareCompact[j]].
inline constexpr std::array<int, 23> kNativeToHardwareCompact = {
    0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20,
    2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22,
};

// native_physx[i] = hardware_compact[kHardwareCompactToNative[i]].
inline constexpr std::array<int, 23> kHardwareCompactToNative = {
    0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3,
    9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22,
};

// canonical_il29[kNativeToCanonicalIl29[i]] = native_physx[i].
inline constexpr std::array<int, 23> kNativeToCanonicalIl29 = {
    0, 1, 2, 3, 4, 11, 12, 6, 7, 15, 16, 9,
    10, 19, 20, 13, 14, 21, 22, 17, 18, 23, 24,
};

inline constexpr std::array<int, 23> kSourceMj29ToNative = {
    0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3,
    9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26,
};

inline constexpr std::array<int, 23> kSourceIl29KeepIndices = {
    0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
};
inline constexpr std::array<int, 6> kSourceIl29ExcludedIndices = {
    5, 8, 25, 26, 27, 28,
};

inline constexpr std::array<std::string_view, 23> kNativeJointNames = {
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
};

inline constexpr std::array<std::string_view, 23> kHardwareJointNames = {
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
};

inline constexpr std::array<double, 23> kHardwareActionScale = {
    0.55, 0.35, 0.55, 0.35, 0.44, 0.44, 0.55, 0.35,
    0.55, 0.35, 0.44, 0.44, 0.55, 0.44, 0.44, 0.44,
    0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44,
};

constexpr std::array<double, 23> NativeActionScale() {
  std::array<double, 23> result{};
  for (std::size_t native = 0; native < result.size(); ++native) {
    result[native] =
        kHardwareActionScale[static_cast<std::size_t>(
            kHardwareCompactToNative[native])];
  }
  return result;
}

inline constexpr auto kNativeActionScale = NativeActionScale();

constexpr bool IsPermutation(const std::array<int, 23>& values) {
  std::array<bool, 23> seen{};
  for (const int value : values) {
    if (value < 0 || value >= 23 || seen[static_cast<std::size_t>(value)]) {
      return false;
    }
    seen[static_cast<std::size_t>(value)] = true;
  }
  return std::all_of(seen.begin(), seen.end(), [](bool value) { return value; });
}

constexpr bool AreInversePermutations() {
  for (std::size_t native = 0; native < 23; ++native) {
    const auto hardware =
        static_cast<std::size_t>(kHardwareCompactToNative[native]);
    if (kNativeToHardwareCompact[hardware] != static_cast<int>(native)) {
      return false;
    }
  }
  return true;
}

constexpr std::array<int, 23> NativeToHardwareMotorIds() {
  std::array<int, 23> result{};
  for (std::size_t native = 0; native < result.size(); ++native) {
    result[native] =
        kHardwareJointIds[static_cast<std::size_t>(
            kHardwareCompactToNative[native])];
  }
  return result;
}

inline constexpr auto kNativeToHardwareMotorIds =
    NativeToHardwareMotorIds();

static_assert(IsPermutation(kNativeToHardwareCompact));
static_assert(IsPermutation(kHardwareCompactToNative));
static_assert(AreInversePermutations());
static_assert(kNativeToHardwareMotorIds ==
              std::array<int, 23>{
                  0, 6, 12, 1, 7, 15, 22, 2, 8, 16, 23, 3,
                  9, 17, 24, 4, 10, 18, 25, 5, 11, 19, 26,
              });

template <typename T>
constexpr std::array<T, 23> NativeToHardwareCompact(
    const std::array<T, 23>& native) {
  std::array<T, 23> hardware{};
  for (std::size_t index = 0; index < hardware.size(); ++index) {
    hardware[index] =
        native[static_cast<std::size_t>(kNativeToHardwareCompact[index])];
  }
  return hardware;
}

template <typename T>
constexpr std::array<T, 23> HardwareCompactToNative(
    const std::array<T, 23>& hardware) {
  std::array<T, 23> native{};
  for (std::size_t index = 0; index < native.size(); ++index) {
    native[index] =
        hardware[static_cast<std::size_t>(kHardwareCompactToNative[index])];
  }
  return native;
}

template <typename T>
constexpr std::array<T, 23> HardwareSlotsToNative(
    const std::array<T, 29>& motor_slots) {
  std::array<T, 23> compact{};
  for (std::size_t index = 0; index < compact.size(); ++index) {
    compact[index] =
        motor_slots[static_cast<std::size_t>(kHardwareJointIds[index])];
  }
  return HardwareCompactToNative(compact);
}

enum class RequestedMode {
  Shadow,
  Control,
};

struct ShadowAuthorization {
  bool lowcmd_publisher_allowed = false;
  bool motion_mode_release_allowed = false;
  bool command_writer_allowed = false;
};

struct ModelSignature {
  std::size_t input_count = 0;
  std::size_t output_count = 0;
  std::string input_name;
  std::string output_name;
  std::vector<std::int64_t> input_shape;
  std::vector<std::int64_t> output_shape;
  bool input_float32 = false;
  bool output_float32 = false;
  std::vector<std::int64_t> default_opsets;
};

struct PairBinding {
  std::string encoder_filename;
  std::string decoder_filename;
  std::string metadata_filename;
  std::string encoder_onnx_sha256;
  std::string decoder_onnx_sha256;
  std::string encoder_path;
  std::string decoder_path;
  std::string metadata_path;
  std::string metadata_sha256;
};

struct ValidationResult {
  std::vector<std::string> errors;
  ShadowAuthorization authorization;

  [[nodiscard]] bool ok() const { return errors.empty(); }

  [[nodiscard]] std::string Message() const {
    std::ostringstream stream;
    for (std::size_t index = 0; index < errors.size(); ++index) {
      if (index != 0) {
        stream << "; ";
      }
      stream << errors[index];
    }
    return stream.str();
  }

  void Require() const {
    if (!ok()) {
      throw std::runtime_error("true23 shadow gate rejected artifact: " +
                               Message());
    }
  }
};

inline std::string Sha256Bytes(std::string_view bytes);
inline std::string Sha256CanonicalJson(const nlohmann::json& value);

namespace detail {

template <typename T, std::size_t N>
nlohmann::json ToJsonArray(const std::array<T, N>& values) {
  nlohmann::json result = nlohmann::json::array();
  for (const auto& value : values) {
    result.push_back(value);
  }
  return result;
}

template <std::size_t N>
nlohmann::json ToJsonArray(
    const std::array<std::string_view, N>& values) {
  nlohmann::json result = nlohmann::json::array();
  for (const auto value : values) {
    result.push_back(std::string(value));
  }
  return result;
}

inline const nlohmann::json* Find(
    const nlohmann::json& root,
    std::initializer_list<std::string_view> path) {
  const nlohmann::json* current = &root;
  for (const auto element : path) {
    if (!current->is_object()) {
      return nullptr;
    }
    const auto iterator = current->find(std::string(element));
    if (iterator == current->end()) {
      return nullptr;
    }
    current = &*iterator;
  }
  return current;
}

inline std::string PathName(
    std::initializer_list<std::string_view> path) {
  std::ostringstream stream;
  bool first = true;
  for (const auto element : path) {
    if (!first) {
      stream << '.';
    }
    first = false;
    stream << element;
  }
  return stream.str();
}

template <typename T>
void RequireEqual(
    const nlohmann::json& root,
    std::initializer_list<std::string_view> path,
    const T& expected,
    std::vector<std::string>& errors) {
  const auto* value = Find(root, path);
  const auto name = PathName(path);
  if (value == nullptr) {
    errors.push_back(name + " is missing");
    return;
  }
  try {
    if (*value != expected) {
      errors.push_back(name + " mismatch");
    }
  } catch (const std::exception&) {
    errors.push_back(name + " has invalid type");
  }
}

inline bool IsLowerSha256(const nlohmann::json& value) {
  if (!value.is_string()) {
    return false;
  }
  const auto text = value.get<std::string>();
  return text.size() == 64 &&
         std::all_of(text.begin(), text.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

inline void RequireSha256(
    const nlohmann::json& root,
    std::initializer_list<std::string_view> path,
    std::vector<std::string>& errors) {
  const auto* value = Find(root, path);
  const auto name = PathName(path);
  if (value == nullptr || !IsLowerSha256(*value)) {
    errors.push_back(name + " must be lowercase SHA-256");
  }
}

inline bool PositiveInteger(const nlohmann::json* value) {
  return value != nullptr && value->is_number_integer() &&
         !value->is_boolean() && value->get<std::int64_t>() > 0;
}

inline void RequireExactObjectKeys(
    const nlohmann::json* value,
    std::initializer_list<std::string_view> expected,
    std::string_view context,
    std::vector<std::string>& errors) {
  if (value == nullptr || !value->is_object()) {
    errors.emplace_back(std::string(context) + " must be an object");
    return;
  }
  if (value->size() != expected.size() ||
      !std::all_of(
          expected.begin(), expected.end(),
          [value](std::string_view key) {
            return value->contains(std::string(key));
          })) {
    errors.emplace_back(
        std::string(context) + " keys do not match schema");
  }
}

}  // namespace detail

inline nlohmann::json ReferenceProfileContract(
    std::string_view profile) {
  if (profile == kCausalHistoryReferenceProfile) {
    return {
        {"schema", "g1_true23_causal_history_profile_v1"},
        {"profile", std::string(profile)},
        {"source_sample_rate_hz", 50},
        {"source_sample_period_s", 0.02},
        {"architecture_initialization_profile",
         std::string(kLowLatencyReferenceProfile)},
        {"encoder_input_dim", 267},
        {"lower_body_term_dim", 240},
        {"lower_body_term_name", "causal_history_lower_body"},
        {"position_frame_count", 10},
        {"position_order", "oldest_to_anchor"},
        {"position_offsets_from_anchor_s",
         nlohmann::json::array(
             {-0.18, -0.16, -0.14, -0.12, -0.10,
              -0.08, -0.06, -0.04, -0.02, 0.0})},
        {"velocity_order", "oldest_to_anchor"},
        {"velocity_definition",
         "forward_difference_q_i_to_q_i_plus_1_over_0p02s"},
        {"anchor_frame", "q9"},
        {"proof_frame", "q10"},
        {"proof_frame_offset_from_anchor_s", 0.02},
        {"anchor_age_at_emission_s", 0.02},
        {"reference_channels_anchor", "q9"},
        {"reference_channels",
         nlohmann::json::array(
             {"lower_body_positions_and_velocities",
              "vr_3point_local_target",
              "vr_3point_local_orientation_target",
              "reference_pelvis_orientation",
              "buffered_robot_anchor_orientation",
              "reward_and_tracking_target"})},
        {"control_and_proprioception_frame", "q10_current"},
        {"future_samples_relative_to_emission", false},
        {"repeated_or_synthetic_future_frames", false},
        {"released_profile_relabel_permitted", false},
        {"retraining_required", true},
        {"contract_sha256", std::string(kCausalHistoryContractSha256)},
    };
  }
  int step = 0;
  double horizon = 0.0;
  nlohmann::json offsets;
  if (profile == kNormalReferenceProfile) {
    step = 5;
    horizon = 0.9;
    offsets = detail::ToJsonArray(kNormalReferenceOffsets);
  } else if (profile == kLowLatencyReferenceProfile) {
    step = 1;
    horizon = 0.18;
    offsets = detail::ToJsonArray(kLowLatencyReferenceOffsets);
  } else {
    throw std::invalid_argument("unsupported true23 reference profile");
  }
  return {
      {"profile", std::string(profile)},
      {"source_sample_rate_hz", 50},
      {"source_sample_period_s", 0.02},
      {"future_frame_count", 10},
      {"future_frame_step", step},
      {"future_frame_offsets_s", std::move(offsets)},
      {"horizon_s", horizon},
      {"command_layout", "positions_120_then_velocities_120"},
  };
}

inline nlohmann::json DecoderLayerDimsForReferenceProfile(
    std::string_view profile) {
  if (profile == kNormalReferenceProfile) {
    return nlohmann::json::array(
        {994, 2048, 2048, 1024, 1024, 512, 512, 23});
  }
  if (profile == kLowLatencyReferenceProfile ||
      profile == kCausalHistoryReferenceProfile) {
    return nlohmann::json::array(
        {994, 4096, 4096, 2048, 2048, 1024, 1024, 512, 512, 23});
  }
  throw std::invalid_argument("unsupported true23 reference profile");
}

inline nlohmann::json DecoderLinearIndicesForReferenceProfile(
    std::string_view profile) {
  if (profile == kNormalReferenceProfile) {
    return nlohmann::json::array({0, 2, 4, 6, 8, 10, 12});
  }
  if (profile == kLowLatencyReferenceProfile ||
      profile == kCausalHistoryReferenceProfile) {
    return nlohmann::json::array({0, 2, 4, 6, 8, 10, 12, 14, 16});
  }
  throw std::invalid_argument("unsupported true23 reference profile");
}

inline void ValidateReferenceProfileBinding(
    const nlohmann::json* profile,
    const nlohmann::json* contract,
    std::string_view context,
    std::vector<std::string>& errors) {
  if (profile == nullptr || !profile->is_string()) {
    errors.emplace_back(
        std::string(context) + ".reference_profile is missing");
    return;
  }
  const auto value = profile->get<std::string>();
  if (value != kNormalReferenceProfile &&
      value != kLowLatencyReferenceProfile &&
      value != kCausalHistoryReferenceProfile) {
    errors.emplace_back(
        std::string(context) + ".reference_profile is unsupported");
    return;
  }
  if (contract == nullptr || *contract != ReferenceProfileContract(value)) {
    errors.emplace_back(
        std::string(context) +
        ".reference_contract does not match immutable profile");
  }
}

inline void ValidateMotionDatasetEvidence(
    const nlohmann::json* dataset,
    std::string_view context,
    std::vector<std::string>& errors) {
  const auto prefix = std::string(context);
  if (dataset == nullptr || !dataset->is_object()) {
    errors.emplace_back(prefix + " must be an object");
    return;
  }
  detail::RequireExactObjectKeys(
      dataset,
      {"schema_version", "source_archive", "processed"},
      prefix, errors);
  detail::RequireExactObjectKeys(
      detail::Find(*dataset, {"source_archive"}),
      {"repository", "revision", "relpath", "size_bytes", "sha256"},
      prefix + ".source_archive", errors);
  detail::RequireExactObjectKeys(
      detail::Find(*dataset, {"processed"}),
      {"root_relpath", "file_count", "total_bytes", "manifest_sha256"},
      prefix + ".processed", errors);
  detail::RequireEqual(*dataset, {"schema_version"}, 1, errors);
  detail::RequireEqual(
      *dataset, {"source_archive", "repository"},
      std::string(kMotionDatasetRepository), errors);
  detail::RequireEqual(
      *dataset, {"source_archive", "revision"},
      std::string(kMotionDatasetRevision), errors);
  detail::RequireEqual(
      *dataset, {"source_archive", "relpath"},
      std::string(kMotionDatasetArchiveRelpath), errors);
  detail::RequireEqual(
      *dataset, {"source_archive", "size_bytes"},
      kMotionDatasetArchiveSizeBytes, errors);
  detail::RequireEqual(
      *dataset, {"source_archive", "sha256"},
      std::string(kMotionDatasetArchiveSha256), errors);
  detail::RequireEqual(
      *dataset, {"processed", "root_relpath"},
      std::string(kProcessedMotionDatasetRootRelpath), errors);
  detail::RequireSha256(
      *dataset, {"processed", "manifest_sha256"}, errors);
  for (const auto field : {"file_count", "total_bytes"}) {
    const auto* value =
        detail::Find(*dataset, {"processed", field});
    if (!detail::PositiveInteger(value)) {
      errors.emplace_back(
          prefix + ".processed." + field +
          " must be a positive integer");
    }
  }
}

inline void ValidateTrainingMaterialEvidence(
    const nlohmann::json* material,
    std::string_view profile,
    std::string_view context,
    std::vector<std::string>& errors);

inline void ValidateTrainingEvidence(
    const nlohmann::json* evidence,
    std::string_view profile,
    std::string_view context,
    std::vector<std::string>& errors) {
  const auto prefix = std::string(context);
  if (evidence == nullptr || !evidence->is_object()) {
    errors.emplace_back(prefix + " must be an object");
    return;
  }
  detail::RequireExactObjectKeys(
      evidence,
      {
          "schema_version",
          "kind",
          "producer",
          "robot_model",
          "global_step",
          "history_length",
          "observation_layout",
          "decoder_input_dim",
          "decoder_output_dim",
          "reference_profile",
          "reference_contract",
          "source_family",
          "source_revision",
          "source_checkpoint_sha256",
          "initial_policy_state_sha256",
          "training_start_global_step",
          "training_updates",
          "minimum_training_updates",
          "policy_state_sha256",
          "weights_only_initialization",
          "motion_dataset",
          "training_material",
      },
      prefix, errors);
  ValidateMotionDatasetEvidence(
      detail::Find(*evidence, {"motion_dataset"}),
      prefix + ".motion_dataset", errors);
  ValidateTrainingMaterialEvidence(
      detail::Find(*evidence, {"training_material"}),
      profile, prefix + ".training_material", errors);
  detail::RequireEqual(*evidence, {"schema_version"}, 3, errors);
  detail::RequireEqual(
      *evidence, {"kind"}, "g1_23dof_training_checkpoint", errors);
  detail::RequireEqual(
      *evidence, {"producer"},
      "gear_sonic.trl.callbacks.ModelSaveCallback", errors);
  detail::RequireEqual(
      *evidence, {"robot_model"}, std::string(kRobotModel), errors);
  detail::RequireEqual(
      *evidence, {"history_length"}, kHistoryLength, errors);
  detail::RequireEqual(
      *evidence, {"observation_layout"},
      std::string(kObservationLayout), errors);
  detail::RequireEqual(
      *evidence, {"decoder_input_dim"}, kDecoderInputDim, errors);
  detail::RequireEqual(
      *evidence, {"decoder_output_dim"}, kDecoderOutputDim, errors);
  detail::RequireEqual(
      *evidence, {"reference_contract"},
      ReferenceProfileContract(profile), errors);
  detail::RequireEqual(
      *evidence, {"weights_only_initialization"}, false, errors);
  detail::RequireEqual(
      *evidence, {"reference_profile"}, std::string(profile), errors);
  detail::RequireEqual(
      *evidence, {"minimum_training_updates"},
      kMinimumTrainingUpdates, errors);
  if (profile == kNormalReferenceProfile) {
    detail::RequireEqual(
        *evidence, {"source_family"}, "sonic_release", errors);
    detail::RequireEqual(
        *evidence, {"source_revision"}, nullptr, errors);
    detail::RequireEqual(
        *evidence, {"source_checkpoint_sha256"},
        std::string(kNormalReleaseSha256), errors);
    detail::RequireEqual(
        *evidence, {"initial_policy_state_sha256"},
        std::string(kNormalInitialPolicySha256), errors);
  } else if (profile == kLowLatencyReferenceProfile ||
             profile == kCausalHistoryReferenceProfile) {
    detail::RequireEqual(
        *evidence, {"source_family"}, "sonic_low_latency", errors);
    detail::RequireEqual(
        *evidence, {"source_revision"},
        std::string(kLowLatencyReleaseRevision), errors);
    detail::RequireEqual(
        *evidence, {"source_checkpoint_sha256"},
        std::string(kLowLatencyReleaseSha256), errors);
    detail::RequireEqual(
        *evidence, {"initial_policy_state_sha256"},
        std::string(kLowLatencyInitialPolicySha256), errors);
  }
  detail::RequireSha256(
      *evidence, {"initial_policy_state_sha256"}, errors);
  detail::RequireSha256(
      *evidence, {"policy_state_sha256"}, errors);
  const auto* initial =
      detail::Find(*evidence, {"initial_policy_state_sha256"});
  const auto* final = detail::Find(*evidence, {"policy_state_sha256"});
  if (initial != nullptr && final != nullptr && *initial == *final) {
    errors.emplace_back(prefix + " policy weights were not updated");
  }
  const auto* global_step = detail::Find(*evidence, {"global_step"});
  const auto* start =
      detail::Find(*evidence, {"training_start_global_step"});
  const auto* updates = detail::Find(*evidence, {"training_updates"});
  if (!detail::PositiveInteger(global_step) ||
      start == nullptr || !start->is_number_integer() ||
      start->is_boolean() || start->get<std::int64_t>() < 0 ||
      updates == nullptr || !updates->is_number_integer() ||
      updates->is_boolean() ||
      updates->get<std::int64_t>() < kMinimumTrainingUpdates ||
      global_step->get<std::int64_t>() -
              start->get<std::int64_t>() !=
          updates->get<std::int64_t>()) {
    errors.emplace_back(prefix + " does not prove minimum policy updates");
  }
}

inline bool IsNormalizedRelativePosixPath(std::string_view value) {
  if (value.empty() || value.front() == '/' ||
      value.find('\\') != std::string_view::npos ||
      value.find("//") != std::string_view::npos) {
    return false;
  }
  std::size_t start = 0;
  while (start <= value.size()) {
    const auto end = value.find('/', start);
    const auto part = value.substr(
        start,
        end == std::string_view::npos ? value.size() - start
                                      : end - start);
    if (part.empty() || part == "." || part == "..") {
      return false;
    }
    if (end == std::string_view::npos) {
      break;
    }
    start = end + 1;
  }
  return true;
}

inline void ValidateRuntimeSourceManifest(
    const nlohmann::json* manifest,
    std::string_view context,
    std::vector<std::string>& errors) {
  const auto prefix = std::string(context);
  if (manifest == nullptr || !manifest->is_object()) {
    errors.emplace_back(prefix + " must be an object");
    return;
  }
  detail::RequireExactObjectKeys(
      manifest,
      {"schema_version", "file_count", "total_bytes",
       "manifest_sha256", "files"},
      prefix, errors);
  detail::RequireEqual(*manifest, {"schema_version"}, 1, errors);
  detail::RequireSha256(*manifest, {"manifest_sha256"}, errors);
  const auto* file_count = detail::Find(*manifest, {"file_count"});
  const auto* total_bytes = detail::Find(*manifest, {"total_bytes"});
  const auto* files = detail::Find(*manifest, {"files"});
  if (!detail::PositiveInteger(file_count)) {
    errors.emplace_back(prefix + ".file_count must be a positive integer");
  }
  if (!detail::PositiveInteger(total_bytes)) {
    errors.emplace_back(prefix + ".total_bytes must be a positive integer");
  }
  if (files == nullptr || !files->is_array() || files->empty()) {
    errors.emplace_back(prefix + ".files must be a non-empty array");
    return;
  }
  if (file_count != nullptr && file_count->is_number_integer() &&
      !file_count->is_boolean() &&
      file_count->get<std::int64_t>() !=
          static_cast<std::int64_t>(files->size())) {
    errors.emplace_back(prefix + ".file_count differs from files");
  }
  std::int64_t recomputed_total = 0;
  std::string previous_path;
  for (std::size_t index = 0; index < files->size(); ++index) {
    const auto& record = (*files)[index];
    const auto record_context =
        prefix + ".files[" + std::to_string(index) + "]";
    detail::RequireExactObjectKeys(
        &record, {"relpath", "size_bytes", "sha256"},
        record_context, errors);
    const auto* relpath = detail::Find(record, {"relpath"});
    if (relpath == nullptr || !relpath->is_string() ||
        !IsNormalizedRelativePosixPath(
            relpath->get_ref<const std::string&>())) {
      errors.emplace_back(
          record_context + ".relpath must be normalized relative POSIX");
    } else {
      const auto& current = relpath->get_ref<const std::string&>();
      if (!previous_path.empty() && current <= previous_path) {
        errors.emplace_back(
            prefix + ".files must use sorted unique relpaths");
      }
      previous_path = current;
    }
    const auto* size = detail::Find(record, {"size_bytes"});
    if (size == nullptr || !size->is_number_integer() ||
        size->is_boolean() || size->get<std::int64_t>() < 0) {
      errors.emplace_back(
          record_context + ".size_bytes must be an integer >= 0");
    } else {
      recomputed_total += size->get<std::int64_t>();
    }
    detail::RequireSha256(record, {"sha256"}, errors);
  }
  if (total_bytes != nullptr && total_bytes->is_number_integer() &&
      !total_bytes->is_boolean() &&
      total_bytes->get<std::int64_t>() != recomputed_total) {
    errors.emplace_back(prefix + ".total_bytes differs from files");
  }
  const auto* expected_sha = detail::Find(*manifest, {"manifest_sha256"});
  if (expected_sha != nullptr && expected_sha->is_string() &&
      expected_sha->get_ref<const std::string&>() !=
          Sha256CanonicalJson(*files)) {
    errors.emplace_back(prefix + ".manifest_sha256 differs from files");
  }
}

inline void ValidateTrainingMaterialEvidence(
    const nlohmann::json* material,
    std::string_view profile,
    std::string_view context,
    std::vector<std::string>& errors) {
  (void)profile;
  const auto prefix = std::string(context);
  if (material == nullptr || !material->is_object()) {
    errors.emplace_back(prefix + " must be an object");
    return;
  }
  detail::RequireExactObjectKeys(
      material,
      {
          "schema_version",
          "resolved_config",
          "resolved_config_sha256",
          "material_config",
          "material_config_sha256",
          "runtime_source",
          "robot_assets",
      },
      prefix, errors);
  detail::RequireEqual(*material, {"schema_version"}, 1, errors);
  const auto* resolved = detail::Find(*material, {"resolved_config"});
  const auto* material_config =
      detail::Find(*material, {"material_config"});
  if (resolved == nullptr || !resolved->is_object()) {
    errors.emplace_back(prefix + ".resolved_config must be an object");
  } else {
    const auto* expected_sha =
        detail::Find(*material, {"resolved_config_sha256"});
    if (expected_sha == nullptr || !expected_sha->is_string() ||
        expected_sha->get_ref<const std::string&>() !=
            Sha256CanonicalJson(*resolved)) {
      errors.emplace_back(
          prefix + ".resolved_config_sha256 differs from snapshot");
    }
  }
  if (material_config == nullptr || !material_config->is_object()) {
    errors.emplace_back(prefix + ".material_config must be an object");
  } else {
    const auto* expected_sha =
        detail::Find(*material, {"material_config_sha256"});
    if (expected_sha == nullptr || !expected_sha->is_string() ||
        expected_sha->get_ref<const std::string&>() !=
            Sha256CanonicalJson(*material_config)) {
      errors.emplace_back(
          prefix + ".material_config_sha256 differs from config");
    }
  }
  ValidateRuntimeSourceManifest(
      detail::Find(*material, {"runtime_source"}),
      prefix + ".runtime_source", errors);
  ValidateRuntimeSourceManifest(
      detail::Find(*material, {"robot_assets"}),
      prefix + ".robot_assets", errors);
}

inline void ValidateMaterialProvenance(
    const nlohmann::json* provenance,
    std::string_view context,
    std::vector<std::string>& errors) {
  const auto prefix = std::string(context);
  if (provenance == nullptr || !provenance->is_object()) {
    errors.emplace_back(prefix + " must be an object");
    return;
  }
  detail::RequireExactObjectKeys(
      provenance,
      {"schema_version", "runtime_source", "motion_dataset"},
      prefix, errors);
  detail::RequireEqual(*provenance, {"schema_version"}, 1, errors);
  ValidateRuntimeSourceManifest(
      detail::Find(*provenance, {"runtime_source"}),
      prefix + ".runtime_source", errors);
  ValidateMotionDatasetEvidence(
      detail::Find(*provenance, {"motion_dataset"}),
      prefix + ".motion_dataset", errors);
}

inline void ValidateModelSignature(
    const ModelSignature& signature,
    std::string_view role,
    std::string_view expected_input_name,
    std::string_view expected_output_name,
    int expected_input_dim,
    int expected_output_dim,
    std::vector<std::string>& errors) {
  const auto prefix = std::string(role) + " ONNX ";
  if (signature.input_count != 1) {
    errors.emplace_back(prefix + "must have exactly one input");
  }
  if (signature.output_count != 1) {
    errors.emplace_back(prefix + "must have exactly one output");
  }
  if (signature.input_name != expected_input_name) {
    errors.emplace_back(prefix + "input name mismatch");
  }
  if (signature.output_name != expected_output_name) {
    errors.emplace_back(prefix + "output name mismatch");
  }
  if (signature.input_shape !=
      std::vector<std::int64_t>{1, expected_input_dim}) {
    errors.emplace_back(prefix + "input shape must be static [1," +
                        std::to_string(expected_input_dim) + "]");
  }
  if (signature.output_shape !=
      std::vector<std::int64_t>{1, expected_output_dim}) {
    errors.emplace_back(prefix + "output shape must be static [1," +
                        std::to_string(expected_output_dim) + "]");
  }
  if (!signature.input_float32) {
    errors.emplace_back(prefix + "input must be float32");
  }
  if (!signature.output_float32) {
    errors.emplace_back(prefix + "output must be float32");
  }
  if (signature.default_opsets !=
      std::vector<std::int64_t>{kOnnxOpset}) {
    errors.emplace_back(prefix + "default opset must be exactly 13");
  }
}

inline void ValidatePairModelSignatures(
    const ModelSignature& encoder,
    const ModelSignature& decoder,
    std::vector<std::string>& errors) {
  ValidateModelSignature(
      encoder, kEncoderRole, kEncoderInputName, kEncoderOutputName,
      kEncoderInputDim, kEncoderOutputDim, errors);
  ValidateModelSignature(
      decoder, kDecoderRole, kDecoderInputName, kDecoderOutputName,
      kDecoderInputDim, kDecoderOutputDim, errors);
}

inline void ValidateSidecar(
    const nlohmann::json& sidecar,
    std::vector<std::string>& errors) {
  using detail::RequireEqual;
  using detail::RequireSha256;

  RequireEqual(sidecar, {"schema_version"}, kArtifactSchemaVersion, errors);
  RequireEqual(sidecar, {"artifact_kind"}, std::string(kArtifactKind), errors);
  RequireEqual(sidecar, {"robot_model"}, std::string(kRobotModel), errors);
  RequireEqual(sidecar, {"mode_machine"}, kRequiredModeMachine, errors);
  RequireEqual(sidecar, {"action_dof"}, kDecoderOutputDim, errors);
  RequireEqual(sidecar, {"hardware_joint_ids"},
               detail::ToJsonArray(kHardwareJointIds), errors);
  RequireEqual(sidecar, {"excluded_hardware_joint_ids"},
               detail::ToJsonArray(kExcludedHardwareJointIds), errors);
  RequireEqual(sidecar, {"decoder_output_layout"},
               std::string(kDecoderOutputLayout), errors);
  RequireEqual(sidecar, {"observation_layout"},
               std::string(kObservationLayout), errors);
  RequireEqual(sidecar, {"history_length"}, kHistoryLength, errors);
  RequireEqual(sidecar, {"decoder_input_dim"}, kDecoderInputDim, errors);
  RequireEqual(sidecar, {"decoder_output_dim"}, kDecoderOutputDim, errors);
  RequireEqual(sidecar, {"onnx_opset"}, kOnnxOpset, errors);
  RequireEqual(sidecar, {"checkpoint_stage"}, "trained", errors);
  RequireEqual(sidecar, {"deployment_ready"}, true, errors);
  RequireEqual(sidecar, {"sim_validation_passed"}, true, errors);
  RequireEqual(sidecar, {"naive_output_masking"}, false, errors);
  ValidateReferenceProfileBinding(
      detail::Find(sidecar, {"reference_profile"}),
      detail::Find(sidecar, {"reference_contract"}),
      "sidecar", errors);
  const auto* profile_value =
      detail::Find(sidecar, {"reference_profile"});
  if (profile_value != nullptr && profile_value->is_string()) {
    ValidateTrainingEvidence(
        detail::Find(sidecar, {"training_evidence"}),
        profile_value->get_ref<const std::string&>(),
        "sidecar.training_evidence", errors);
  }
  RequireEqual(sidecar, {"validation", "pair_dry_run"}, true, errors);
  for (const auto role : {kEncoderRole, kDecoderRole}) {
    const auto* validation =
        detail::Find(sidecar, {"validation", role});
    if (validation == nullptr || !validation->is_object()) {
      errors.emplace_back(
          "validation." + std::string(role) + " must be an object");
      continue;
    }
    RequireEqual(sidecar,
                 {"validation", role, "onnx_checker_full_check"},
                 true, errors);
    RequireEqual(sidecar, {"validation", role, "shape_inference"},
                 true, errors);
    RequireEqual(sidecar, {"validation", role, "ort_provider"},
                 "CPUExecutionProvider", errors);
    RequireEqual(sidecar, {"validation", role, "parity_case_count"},
                 3, errors);
  }

  const auto* global_step =
      detail::Find(sidecar, {"training_evidence", "global_step"});
  if (!detail::PositiveInteger(global_step)) {
    errors.emplace_back(
        "training_evidence.global_step must be a positive integer");
  }
  const auto* sidecar_profile =
      detail::Find(sidecar, {"reference_profile"});
  const auto* training_profile =
      detail::Find(sidecar,
                   {"training_evidence", "reference_profile"});
  if (sidecar_profile == nullptr || training_profile == nullptr ||
      *sidecar_profile != *training_profile) {
    errors.emplace_back(
        "training_evidence.reference_profile pair binding mismatch");
  }

  RequireEqual(sidecar, {"simulation_evidence", "schema_version"},
               kSimulationSchemaVersion, errors);
  RequireEqual(sidecar, {"simulation_evidence", "computed_pass"}, true,
               errors);
  ValidateMaterialProvenance(
      detail::Find(
          sidecar, {"simulation_evidence", "material_provenance"}),
      "sidecar.simulation_evidence.material_provenance", errors);
  const auto* trained_motion_dataset = detail::Find(
      sidecar, {"training_evidence", "motion_dataset"});
  const auto* simulated_motion_dataset = detail::Find(
      sidecar,
      {"simulation_evidence", "material_provenance",
       "motion_dataset"});
  if (trained_motion_dataset == nullptr ||
      simulated_motion_dataset == nullptr ||
      *trained_motion_dataset != *simulated_motion_dataset) {
    errors.emplace_back(
        "training/simulation motion dataset binding mismatch");
  }
  const auto* trained_runtime_source = detail::Find(
      sidecar,
      {"training_evidence", "training_material", "runtime_source"});
  const auto* simulated_runtime_source = detail::Find(
      sidecar,
      {"simulation_evidence", "material_provenance",
       "runtime_source"});
  if (trained_runtime_source == nullptr ||
      simulated_runtime_source == nullptr ||
      *trained_runtime_source != *simulated_runtime_source) {
    errors.emplace_back(
        "training/simulation runtime source binding mismatch");
  }
  const auto* simulation_evidence =
      detail::Find(sidecar, {"simulation_evidence"});
  detail::RequireExactObjectKeys(
      simulation_evidence,
      {
          "schema_version",
          "computed_pass",
          "report_sha256",
          "report_payload_sha256",
          "checkpoint_sha256",
          "producer",
          "runtime_config",
          "material_provenance",
          "trace_manifest_sha256",
          "trace_count",
          "required_scenarios",
          "run_count",
          "total_episodes",
          "total_steps",
          "scenario_coverage",
          "simulator",
          "max_metrics",
      },
      "sidecar.simulation_evidence", errors);
  detail::RequireExactObjectKeys(
      detail::Find(sidecar, {"simulation_evidence", "producer"}),
      {"kind", "version", "runner_sha256"},
      "sidecar.simulation_evidence.producer", errors);
  detail::RequireExactObjectKeys(
      detail::Find(sidecar, {"simulation_evidence", "runtime_config"}),
      {"resolved_config_sha256", "material_config_sha256"},
      "sidecar.simulation_evidence.runtime_config", errors);
  detail::RequireExactObjectKeys(
      detail::Find(sidecar, {"simulation_evidence", "simulator"}),
      {"name", "version", "asset_sha256", "robot_config_sha256",
       "config_sha256", "runtime_config_sha256"},
      "sidecar.simulation_evidence.simulator", errors);
  detail::RequireExactObjectKeys(
      detail::Find(sidecar, {"simulation_evidence", "max_metrics"}),
      {"phantom_observation_max_abs", "max_recovery_time_s",
       "action_saturation_fraction", "mpjpe_m"},
      "sidecar.simulation_evidence.max_metrics", errors);
  detail::RequireExactObjectKeys(
      detail::Find(
          sidecar, {"simulation_evidence", "scenario_coverage"}),
      {"nominal", "disturbance_50", "disturbance_100"},
      "sidecar.simulation_evidence.scenario_coverage", errors);
  RequireEqual(
      sidecar, {"simulation_evidence", "producer", "kind"},
      "gear_sonic_true23_isaaclab_disturbance_validation", errors);
  RequireEqual(
      sidecar, {"simulation_evidence", "producer", "version"},
      1, errors);
  detail::RequireSha256(
      sidecar,
      {"simulation_evidence", "producer", "runner_sha256"}, errors);
  detail::RequireSha256(
      sidecar,
      {"simulation_evidence", "runtime_config",
       "resolved_config_sha256"},
      errors);
  detail::RequireSha256(
      sidecar,
      {"simulation_evidence", "runtime_config",
       "material_config_sha256"},
      errors);
  detail::RequireSha256(
      sidecar,
      {"simulation_evidence", "trace_manifest_sha256"}, errors);
  detail::RequireSha256(
      sidecar,
      {"simulation_evidence", "simulator",
       "runtime_config_sha256"},
      errors);
  const auto* trace_count =
      detail::Find(sidecar, {"simulation_evidence", "trace_count"});
  const auto* evidence_run_count =
      detail::Find(sidecar, {"simulation_evidence", "run_count"});
  if (!detail::PositiveInteger(trace_count) ||
      evidence_run_count == nullptr ||
      *trace_count != *evidence_run_count) {
    errors.emplace_back(
        "simulation_evidence trace_count/run_count mismatch");
  }
  RequireEqual(sidecar, {"simulation_evidence", "required_scenarios"},
               nlohmann::json::array(
                   {"nominal", "disturbance_50", "disturbance_100"}),
               errors);
  const auto* run_count =
      detail::Find(sidecar, {"simulation_evidence", "run_count"});
  if (run_count == nullptr || !run_count->is_number_integer() ||
      run_count->is_boolean() || run_count->get<std::int64_t>() < 9) {
    errors.emplace_back(
        "simulation_evidence.run_count must cover three seeds per scenario");
  }
  const auto require_coverage =
      [&](std::string_view scenario,
          std::string_view field,
          std::int64_t minimum) {
        const auto* value = detail::Find(
            sidecar,
            {"simulation_evidence", "scenario_coverage", scenario, field});
        if (value == nullptr || !value->is_number_integer() ||
            value->is_boolean() ||
            value->get<std::int64_t>() < minimum) {
          errors.emplace_back(
              "simulation_evidence.scenario_coverage." +
              std::string(scenario) + "." + std::string(field) +
              " is below required coverage");
        }
      };
  for (const auto scenario :
       {"nominal", "disturbance_50", "disturbance_100"}) {
    detail::RequireExactObjectKeys(
        detail::Find(
            sidecar,
            {"simulation_evidence", "scenario_coverage", scenario}),
        {"seed_count", "episodes", "steps"},
        "sidecar.simulation_evidence.scenario_coverage", errors);
    require_coverage(scenario, "seed_count", 3);
    require_coverage(scenario, "episodes", 64);
    require_coverage(scenario, "steps", 16000);
  }
  const auto require_total =
      [&](std::string_view field, std::int64_t minimum) {
        const auto* value =
            detail::Find(sidecar, {"simulation_evidence", field});
        if (value == nullptr || !value->is_number_integer() ||
            value->is_boolean() ||
            value->get<std::int64_t>() < minimum) {
          errors.emplace_back(
              "simulation_evidence." + std::string(field) +
              " is below required coverage");
        }
      };
  require_total("total_episodes", 192);
  require_total("total_steps", 48000);
  RequireEqual(sidecar, {"simulation_evidence", "simulator", "name"},
               "IsaacLab", errors);
  RequireSha256(sidecar,
                {"simulation_evidence", "simulator", "asset_sha256"},
                errors);
  RequireSha256(
      sidecar,
      {"simulation_evidence", "simulator", "robot_config_sha256"},
      errors);
  RequireSha256(sidecar,
                 {"simulation_evidence", "simulator", "config_sha256"},
                 errors);
  const auto* simulator_version =
      detail::Find(sidecar,
                   {"simulation_evidence", "simulator", "version"});
  if (simulator_version == nullptr || !simulator_version->is_string() ||
      simulator_version->get_ref<const std::string&>().empty()) {
    errors.emplace_back(
        "simulation_evidence.simulator.version must be non-empty");
  }
  RequireSha256(sidecar,
                {"simulation_evidence", "checkpoint_sha256"}, errors);
  RequireSha256(sidecar,
                {"simulation_evidence", "report_sha256"}, errors);
  RequireSha256(sidecar,
                {"simulation_evidence", "report_payload_sha256"}, errors);
  const auto require_metric_at_most =
      [&](std::string_view metric, double maximum) {
        const auto* value = detail::Find(
            sidecar, {"simulation_evidence", "max_metrics", metric});
        if (value == nullptr || !value->is_number() ||
            value->is_boolean()) {
          errors.emplace_back(
              "simulation_evidence.max_metrics." +
              std::string(metric) + " must be numeric");
          return;
        }
        const auto numeric = value->get<double>();
        if (!std::isfinite(numeric) || numeric < 0.0 ||
            numeric > maximum) {
          errors.emplace_back(
              "simulation_evidence.max_metrics." +
              std::string(metric) + " exceeds promotion threshold");
        }
      };
  require_metric_at_most("phantom_observation_max_abs", 1.0e-8);
  require_metric_at_most("max_recovery_time_s", 2.0);
  require_metric_at_most("action_saturation_fraction", 0.10);
  require_metric_at_most("mpjpe_m", 0.15);

  RequireEqual(sidecar,
               {"observation_contract",
                "native_il23_to_canonical_il29"},
               detail::ToJsonArray(kNativeToCanonicalIl29), errors);
  RequireEqual(sidecar,
               {"observation_contract", "source_il29_keep_indices"},
               detail::ToJsonArray(kSourceIl29KeepIndices), errors);
  RequireEqual(sidecar,
               {"observation_contract", "source_il29_excluded_indices"},
               detail::ToJsonArray(kSourceIl29ExcludedIndices), errors);
  RequireEqual(sidecar, {"observation_contract", "term_order"},
               nlohmann::json::array(
                   {"base_ang_vel", "joint_pos_rel", "joint_vel",
                    "previous_action", "projected_gravity"}),
               errors);
  RequireEqual(sidecar, {"observation_contract", "history_order"},
               "oldest_to_newest", errors);
  RequireEqual(
      sidecar,
      {"observation_contract", "missing_fill", "joint_pos_rel"},
      "fixed_default_relative_zero", errors);
  RequireEqual(sidecar,
               {"observation_contract", "missing_fill", "joint_vel"},
               "zero", errors);
  RequireEqual(
      sidecar,
      {"observation_contract", "missing_fill", "previous_action"},
      "zero_every_history_frame", errors);

  RequireEqual(sidecar,
               {"action_contract", "native_il23_joint_names"},
               detail::ToJsonArray(kNativeJointNames), errors);
  RequireEqual(sidecar,
               {"action_contract", "hardware_joint_names"},
               detail::ToJsonArray(kHardwareJointNames), errors);
  RequireEqual(sidecar,
               {"action_contract", "isaaclab_to_mujoco_dof"},
               detail::ToJsonArray(kNativeToHardwareCompact), errors);
  RequireEqual(sidecar,
               {"action_contract", "mujoco_to_isaaclab_dof"},
               detail::ToJsonArray(kHardwareCompactToNative), errors);
  RequireEqual(sidecar,
               {"action_contract", "hardware_action_scale"},
               detail::ToJsonArray(kHardwareActionScale), errors);
  RequireEqual(sidecar,
               {"action_contract", "native_il23_action_scale"},
               detail::ToJsonArray(kNativeActionScale), errors);

  constexpr std::array<std::string_view, 17> kRequiredHashKeys = {
           "checkpoint_sha256",
           "policy_state_sha256",
           "encoder_state_sha256",
           "decoder_state_sha256",
           "encoder_onnx_sha256",
           "decoder_onnx_sha256",
           "sim_report_sha256",
           "sim_report_payload_sha256",
           "contract_sha256",
           "robot_asset_sha256",
           "robot_config_sha256",
           "sim_config_sha256",
           "encoder_config_sha256",
           "decoder_config_sha256",
           "policy_config_sha256",
           "encoder_embedded_metadata_sha256",
           "decoder_embedded_metadata_sha256",
  };
  const auto* hashes = detail::Find(sidecar, {"hashes"});
  if (hashes == nullptr || !hashes->is_object() ||
      hashes->size() != kRequiredHashKeys.size()) {
    errors.emplace_back(
        "hashes must contain exactly the 17 paired artifact bindings");
  }
  for (const auto key : kRequiredHashKeys) {
    RequireSha256(sidecar, {"hashes", key}, errors);
  }
  RequireSha256(sidecar, {"metadata_payload_sha256"}, errors);
}

inline void ValidateEmbeddedContract(
    const nlohmann::json& embedded,
    std::vector<std::string>& errors) {
  using detail::RequireEqual;
  using detail::RequireSha256;

  RequireEqual(embedded, {"contract", "schema_version"},
               kArtifactSchemaVersion, errors);
  RequireEqual(embedded, {"contract", "robot_model"},
               std::string(kRobotModel), errors);
  RequireEqual(embedded, {"contract", "required_mode_machine"},
               kRequiredModeMachine, errors);
  RequireEqual(embedded, {"contract", "observation_layout"},
               std::string(kObservationLayout), errors);
  RequireEqual(embedded, {"contract", "history_length"}, kHistoryLength,
               errors);
  RequireEqual(embedded, {"contract", "history_order"},
               "oldest_to_newest", errors);
  RequireEqual(embedded, {"contract", "term_order"},
               nlohmann::json::array(
                   {"base_ang_vel", "joint_pos_rel", "joint_vel",
                    "previous_action", "projected_gravity"}),
               errors);
  RequireEqual(embedded, {"contract", "token_dim"}, kEncoderOutputDim,
               errors);
  RequireEqual(embedded, {"contract", "proprioception_dim"}, 930,
               errors);
  RequireEqual(embedded, {"contract", "decoder_input_dim"},
               kDecoderInputDim, errors);
  RequireEqual(embedded, {"contract", "decoder_output_dim"},
               kDecoderOutputDim, errors);
  RequireEqual(embedded, {"contract", "decoder_output_layout"},
               std::string(kDecoderOutputLayout), errors);
  const auto* reference_profile =
      detail::Find(embedded, {"contract", "reference_profile"});
  ValidateReferenceProfileBinding(
      reference_profile,
      detail::Find(embedded, {"contract", "reference_contract"}),
      "embedded.contract", errors);
  RequireEqual(embedded, {"contract", "source_il29_keep_indices"},
               detail::ToJsonArray(kSourceIl29KeepIndices), errors);
  RequireEqual(embedded, {"contract", "source_il29_excluded_indices"},
               detail::ToJsonArray(kSourceIl29ExcludedIndices), errors);
  RequireEqual(embedded,
               {"contract", "native_il23_to_canonical_il29"},
               detail::ToJsonArray(kNativeToCanonicalIl29), errors);
  RequireEqual(embedded, {"contract", "source_mj29_to_native_il23"},
               detail::ToJsonArray(kSourceMj29ToNative), errors);
  RequireEqual(
      embedded,
      {"contract", "missing_fill", "joint_pos_rel"},
      "fixed_default_relative_zero", errors);
  RequireEqual(embedded,
               {"contract", "missing_fill", "joint_vel"},
               "zero", errors);
  RequireEqual(
      embedded,
      {"contract", "missing_fill", "previous_action"},
      "zero_every_history_frame", errors);
  RequireEqual(embedded, {"contract", "hardware_joint_ids"},
               detail::ToJsonArray(kHardwareJointIds), errors);
  RequireEqual(embedded, {"contract", "excluded_hardware_joint_ids"},
               detail::ToJsonArray(kExcludedHardwareJointIds), errors);
  RequireEqual(embedded, {"contract", "native_il23_joint_names"},
               detail::ToJsonArray(kNativeJointNames), errors);
  RequireEqual(embedded, {"contract", "hardware_joint_names"},
               detail::ToJsonArray(kHardwareJointNames), errors);
  RequireEqual(embedded, {"contract", "hardware_action_scale"},
               detail::ToJsonArray(kHardwareActionScale), errors);
  RequireEqual(embedded, {"contract", "native_il23_action_scale"},
               detail::ToJsonArray(kNativeActionScale), errors);
  RequireEqual(embedded, {"contract", "isaaclab_to_mujoco_dof"},
               detail::ToJsonArray(kNativeToHardwareCompact), errors);
  RequireEqual(embedded, {"contract", "mujoco_to_isaaclab_dof"},
               detail::ToJsonArray(kHardwareCompactToNative), errors);

  RequireEqual(embedded, {"contract", "teleop_encoder", "input_dim"},
               kEncoderInputDim, errors);
  const bool causal_reference =
      reference_profile != nullptr && reference_profile->is_string() &&
      reference_profile->get_ref<const std::string&>() ==
          kCausalHistoryReferenceProfile;
  RequireEqual(
      embedded, {"contract", "teleop_encoder", "input_term_order"},
      nlohmann::json::array(
          {causal_reference ? "causal_history_lower_body"
                            : "command_multi_future_lower_body",
           "vr_3point_local_target",
           "vr_3point_local_orn_target",
           "motion_anchor_ori_b"}),
      errors);
  RequireEqual(embedded,
               {"contract", "teleop_encoder", "input_term_dims"},
               nlohmann::json::array({240, 9, 12, 6}), errors);
  RequireEqual(
      embedded, {"contract", "teleop_encoder", "layer_dims"},
      nlohmann::json::array({267, 2048, 1024, 512, 512, 64}), errors);
  RequireEqual(embedded,
               {"contract", "teleop_encoder", "linear_indices"},
               nlohmann::json::array({0, 2, 4, 6, 8}), errors);
  RequireEqual(embedded, {"contract", "teleop_encoder", "activation"},
               "SiLU", errors);
  RequireEqual(embedded, {"contract", "teleop_encoder", "token_count"},
               2, errors);
  RequireEqual(embedded, {"contract", "teleop_encoder", "token_width"},
               32, errors);
  RequireEqual(embedded, {"contract", "teleop_encoder", "output_dim"},
               kEncoderOutputDim, errors);
  RequireEqual(embedded, {"contract", "teleop_encoder", "fsq_level"},
               32, errors);
  RequireEqual(embedded, {"contract", "teleop_encoder", "fsq_formula"},
               "fsq_tanh_round_ste_even_levels_v1", errors);
  RequireSha256(
      embedded, {"contract", "teleop_encoder", "config_sha256"}, errors);

  if (reference_profile != nullptr && reference_profile->is_string()) {
    const auto profile = reference_profile->get<std::string>();
    if (profile == kNormalReferenceProfile ||
        profile == kLowLatencyReferenceProfile ||
        profile == kCausalHistoryReferenceProfile) {
      RequireEqual(
          embedded, {"contract", "decoder", "layer_dims"},
          DecoderLayerDimsForReferenceProfile(profile), errors);
      RequireEqual(
          embedded, {"contract", "decoder", "linear_indices"},
          DecoderLinearIndicesForReferenceProfile(profile), errors);
    }
  }
  RequireEqual(embedded, {"contract", "decoder", "activation"},
               "SiLU", errors);
  RequireSha256(embedded, {"contract", "decoder", "config_sha256"},
                errors);
  RequireSha256(embedded, {"contract", "policy_config_sha256"},
                errors);
  RequireSha256(
      embedded, {"contract", "sim_validation", "config_sha256"}, errors);
  RequireEqual(embedded, {"contract", "sim_validation", "control_hz"},
               50, errors);
  RequireEqual(
      embedded,
      {"contract", "sim_validation", "minimum_coverage",
       "seeds_per_scenario"},
      3, errors);
  RequireEqual(
      embedded,
      {"contract", "sim_validation", "minimum_coverage",
       "episodes_per_scenario"},
      64, errors);
  RequireEqual(
      embedded,
      {"contract", "sim_validation", "minimum_coverage",
       "seconds_per_episode"},
      5.0, errors);
  RequireEqual(
      embedded,
      {"contract", "sim_validation", "minimum_coverage",
       "steps_per_episode"},
      250, errors);

  RequireEqual(embedded, {"contract", "encoder_onnx", "opset"},
               kOnnxOpset, errors);
  RequireEqual(embedded, {"contract", "encoder_onnx", "input_name"},
               std::string(kEncoderInputName), errors);
  RequireEqual(embedded, {"contract", "encoder_onnx", "output_name"},
               std::string(kEncoderOutputName), errors);
  RequireEqual(embedded, {"contract", "encoder_onnx", "input_shape"},
               nlohmann::json::array({1, kEncoderInputDim}), errors);
  RequireEqual(embedded, {"contract", "encoder_onnx", "output_shape"},
               nlohmann::json::array({1, kEncoderOutputDim}), errors);
  RequireEqual(embedded, {"contract", "encoder_onnx", "input_dtype"},
               "float32", errors);
  RequireEqual(embedded, {"contract", "encoder_onnx", "output_dtype"},
               "float32", errors);
  RequireEqual(embedded, {"contract", "encoder_onnx", "dynamic_axes"},
               false, errors);

  RequireEqual(embedded, {"contract", "decoder_onnx", "opset"},
               kOnnxOpset, errors);
  RequireEqual(embedded, {"contract", "decoder_onnx", "input_name"},
               std::string(kDecoderInputName), errors);
  RequireEqual(embedded, {"contract", "decoder_onnx", "output_name"},
               std::string(kDecoderOutputName), errors);
  RequireEqual(embedded, {"contract", "decoder_onnx", "input_shape"},
               nlohmann::json::array({1, kDecoderInputDim}), errors);
  RequireEqual(embedded, {"contract", "decoder_onnx", "output_shape"},
               nlohmann::json::array({1, kDecoderOutputDim}), errors);
  RequireEqual(embedded, {"contract", "decoder_onnx", "input_dtype"},
               "float32", errors);
  RequireEqual(embedded, {"contract", "decoder_onnx", "output_dtype"},
               "float32", errors);
  RequireEqual(embedded, {"contract", "decoder_onnx", "dynamic_axes"},
               false, errors);
}

inline void ValidateEmbeddedMetadata(
    const nlohmann::json& embedded,
    std::string_view expected_role,
    std::vector<std::string>& errors) {
  using detail::RequireEqual;
  using detail::RequireSha256;

  RequireEqual(embedded, {"artifact_kind"}, std::string(kArtifactKind),
               errors);
  RequireEqual(embedded, {"artifact_role"}, std::string(expected_role),
               errors);
  RequireEqual(embedded, {"checkpoint_stage"}, "trained", errors);
  RequireEqual(embedded, {"sim_validation_computed"}, true, errors);
  const auto* global_step =
      detail::Find(embedded, {"training_global_step"});
  if (!detail::PositiveInteger(global_step)) {
    errors.emplace_back(
        "embedded training_global_step must be a positive integer");
  }
  RequireSha256(embedded, {"checkpoint_sha256"}, errors);
  RequireSha256(embedded, {"policy_state_sha256"}, errors);
  RequireSha256(embedded, {"encoder_state_sha256"}, errors);
  RequireSha256(embedded, {"decoder_state_sha256"}, errors);
  RequireSha256(embedded, {"sim_report_sha256"}, errors);
  RequireSha256(embedded, {"sim_report_payload_sha256"}, errors);
  ValidateEmbeddedContract(embedded, errors);
  const auto* profile =
      detail::Find(embedded, {"contract", "reference_profile"});
  if (profile != nullptr && profile->is_string()) {
    ValidateTrainingEvidence(
        detail::Find(embedded, {"training_evidence"}),
        profile->get_ref<const std::string&>(),
        "embedded.training_evidence", errors);
  }
}

inline void ValidateCrossBinding(
    const nlohmann::json& sidecar,
    const nlohmann::json& encoder_embedded,
    const nlohmann::json& decoder_embedded,
    const PairBinding& binding,
    std::vector<std::string>& errors) {
  const auto require_pair_hash =
      [&](std::string_view sidecar_key,
          std::string_view embedded_key,
          std::string_view label) {
        const auto* sidecar_value =
            detail::Find(sidecar, {"hashes", sidecar_key});
        const auto* encoder_value =
            detail::Find(encoder_embedded, {embedded_key});
        const auto* decoder_value =
            detail::Find(decoder_embedded, {embedded_key});
        if (sidecar_value == nullptr || encoder_value == nullptr ||
            decoder_value == nullptr ||
            *sidecar_value != *encoder_value ||
            *sidecar_value != *decoder_value) {
          errors.emplace_back(std::string(label) +
                              " pair binding mismatch");
        }
      };

  for (const auto [sidecar_key, embedded_key, label] :
       std::array<std::array<std::string_view, 3>, 6>{
           std::array<std::string_view, 3>{
               "checkpoint_sha256", "checkpoint_sha256", "checkpoint"},
           {"policy_state_sha256", "policy_state_sha256", "policy state"},
           {"encoder_state_sha256", "encoder_state_sha256",
            "encoder state"},
           {"decoder_state_sha256", "decoder_state_sha256",
            "decoder state"},
           {"sim_report_sha256", "sim_report_sha256",
            "simulation report"},
           {"sim_report_payload_sha256", "sim_report_payload_sha256",
            "simulation report payload"},
       }) {
    require_pair_hash(sidecar_key, embedded_key, label);
  }

  const auto* sidecar_step =
      detail::Find(sidecar, {"training_evidence", "global_step"});
  const auto* encoder_step =
      detail::Find(encoder_embedded, {"training_global_step"});
  const auto* decoder_step =
      detail::Find(decoder_embedded, {"training_global_step"});
  if (sidecar_step == nullptr || encoder_step == nullptr ||
      decoder_step == nullptr || *sidecar_step != *encoder_step ||
      *sidecar_step != *decoder_step) {
    errors.emplace_back("training global step pair binding mismatch");
  }

  const auto* sidecar_profile =
      detail::Find(sidecar, {"reference_profile"});
  const auto* encoder_profile =
      detail::Find(encoder_embedded,
                   {"contract", "reference_profile"});
  const auto* decoder_profile =
      detail::Find(decoder_embedded,
                   {"contract", "reference_profile"});
  const auto* training_profile =
      detail::Find(sidecar,
                   {"training_evidence", "reference_profile"});
  if (sidecar_profile == nullptr || encoder_profile == nullptr ||
      decoder_profile == nullptr || training_profile == nullptr ||
      *sidecar_profile != *encoder_profile ||
      *sidecar_profile != *decoder_profile ||
      *sidecar_profile != *training_profile) {
    errors.emplace_back("reference profile pair binding mismatch");
  }
  const auto* sidecar_reference_contract =
      detail::Find(sidecar, {"reference_contract"});
  const auto* encoder_reference_contract =
      detail::Find(encoder_embedded,
                   {"contract", "reference_contract"});
  const auto* decoder_reference_contract =
      detail::Find(decoder_embedded,
                   {"contract", "reference_contract"});
  if (sidecar_reference_contract == nullptr ||
      encoder_reference_contract == nullptr ||
      decoder_reference_contract == nullptr ||
      *sidecar_reference_contract != *encoder_reference_contract ||
      *sidecar_reference_contract != *decoder_reference_contract) {
    errors.emplace_back("reference contract pair binding mismatch");
  }
  const auto* sidecar_training_evidence =
      detail::Find(sidecar, {"training_evidence"});
  const auto* encoder_training_evidence =
      detail::Find(encoder_embedded, {"training_evidence"});
  const auto* decoder_training_evidence =
      detail::Find(decoder_embedded, {"training_evidence"});
  if (sidecar_training_evidence == nullptr ||
      encoder_training_evidence == nullptr ||
      decoder_training_evidence == nullptr ||
      *sidecar_training_evidence != *encoder_training_evidence ||
      *sidecar_training_evidence != *decoder_training_evidence) {
    errors.emplace_back("training lineage pair binding mismatch");
  }

  const auto* encoder_contract =
      detail::Find(encoder_embedded, {"contract"});
  const auto* decoder_contract =
      detail::Find(decoder_embedded, {"contract"});
  if (encoder_contract == nullptr || decoder_contract == nullptr ||
      *encoder_contract != *decoder_contract) {
    errors.emplace_back("encoder/decoder contract mismatch");
  } else {
    const auto* expected_contract_hash =
        detail::Find(sidecar, {"hashes", "contract_sha256"});
    if (expected_contract_hash == nullptr ||
        !expected_contract_hash->is_string() ||
        expected_contract_hash->get<std::string>() !=
            Sha256CanonicalJson(*encoder_contract)) {
      errors.emplace_back("embedded contract SHA-256 mismatch");
    }
  }

  const auto require_string =
      [&](std::initializer_list<std::string_view> path,
          const std::string& expected,
          std::string_view label) {
        const auto* value = detail::Find(sidecar, path);
        if (value == nullptr || !value->is_string() ||
            value->get<std::string>() != expected) {
          errors.emplace_back(std::string(label) + " binding mismatch");
        }
      };
  require_string({"encoder_onnx_filename"}, binding.encoder_filename,
                 "encoder filename");
  require_string({"decoder_onnx_filename"}, binding.decoder_filename,
                 "decoder filename");
  require_string({"metadata_filename"}, binding.metadata_filename,
                 "metadata filename");
  require_string({"hashes", "encoder_onnx_sha256"},
                 binding.encoder_onnx_sha256, "encoder ONNX SHA-256");
  require_string({"hashes", "decoder_onnx_sha256"},
                 binding.decoder_onnx_sha256, "decoder ONNX SHA-256");

  const std::array<std::string, 3> bound_filenames = {
      binding.encoder_filename,
      binding.decoder_filename,
      binding.metadata_filename,
  };
  const std::array<std::string, 3> bound_paths = {
      binding.encoder_path,
      binding.decoder_path,
      binding.metadata_path,
  };
  const auto all_distinct_nonempty =
      [](const std::array<std::string, 3>& values) {
        return std::all_of(
                   values.begin(), values.end(),
                   [](const std::string& value) {
                     return !value.empty();
                   }) &&
               values[0] != values[1] && values[0] != values[2] &&
               values[1] != values[2];
      };
  if (!all_distinct_nonempty(bound_filenames)) {
    errors.emplace_back(
        "encoder, decoder, and metadata filenames must be distinct");
  }
  if (!all_distinct_nonempty(bound_paths)) {
    errors.emplace_back(
        "encoder, decoder, and metadata paths must resolve distinctly");
  }

  const auto* encoder_metadata_hash =
      detail::Find(sidecar,
                   {"hashes", "encoder_embedded_metadata_sha256"});
  if (encoder_metadata_hash == nullptr ||
      !encoder_metadata_hash->is_string() ||
      encoder_metadata_hash->get<std::string>() !=
          Sha256CanonicalJson(encoder_embedded)) {
    errors.emplace_back("encoder embedded metadata SHA-256 mismatch");
  }
  const auto* decoder_metadata_hash =
      detail::Find(sidecar,
                   {"hashes", "decoder_embedded_metadata_sha256"});
  if (decoder_metadata_hash == nullptr ||
      !decoder_metadata_hash->is_string() ||
      decoder_metadata_hash->get<std::string>() !=
          Sha256CanonicalJson(decoder_embedded)) {
    errors.emplace_back("decoder embedded metadata SHA-256 mismatch");
  }

  auto unhashed_sidecar = sidecar;
  const auto erased = unhashed_sidecar.erase("metadata_payload_sha256");
  const auto* sidecar_payload_hash =
      detail::Find(sidecar, {"metadata_payload_sha256"});
  if (erased != 1 || sidecar_payload_hash == nullptr ||
      !sidecar_payload_hash->is_string() ||
      sidecar_payload_hash->get<std::string>() !=
          Sha256CanonicalJson(unhashed_sidecar)) {
    errors.emplace_back("metadata sidecar payload SHA-256 mismatch");
  }

  for (const auto [evidence_key, hashes_key, label] :
       std::array<std::array<std::string_view, 3>, 3>{
           std::array<std::string_view, 3>{
               "checkpoint_sha256", "checkpoint_sha256", "checkpoint"},
           {"report_sha256", "sim_report_sha256", "simulation report"},
           {"report_payload_sha256", "sim_report_payload_sha256",
            "simulation report payload"},
       }) {
    const auto* evidence =
        detail::Find(sidecar, {"simulation_evidence", evidence_key});
    const auto* hashes = detail::Find(sidecar, {"hashes", hashes_key});
    if (evidence == nullptr || hashes == nullptr ||
        *evidence != *hashes) {
      errors.emplace_back(std::string(label) +
                          " evidence/hash mismatch");
    }
  }
  for (const auto [simulator_key, hashes_key, label] :
       std::array<std::array<std::string_view, 3>, 3>{
           std::array<std::string_view, 3>{
               "asset_sha256", "robot_asset_sha256", "robot asset"},
           {"robot_config_sha256", "robot_config_sha256",
            "robot config"},
           {"config_sha256", "sim_config_sha256",
            "simulation config"},
       }) {
    const auto* simulator_value = detail::Find(
        sidecar,
        {"simulation_evidence", "simulator", simulator_key});
    const auto* hashes_value =
        detail::Find(sidecar, {"hashes", hashes_key});
    if (simulator_value == nullptr || hashes_value == nullptr ||
        *simulator_value != *hashes_value) {
      errors.emplace_back(std::string(label) +
                          " simulator/hash mismatch");
    }
  }

  if (encoder_contract != nullptr) {
    for (const auto [contract_section, contract_key, hashes_key, label] :
         std::array<std::array<std::string_view, 4>, 4>{
             std::array<std::string_view, 4>{
                 "teleop_encoder", "config_sha256",
                 "encoder_config_sha256", "encoder config"},
             {"decoder", "config_sha256", "decoder_config_sha256",
              "decoder config"},
             {"sim_validation", "config_sha256", "sim_config_sha256",
              "simulation config"},
             {"", "policy_config_sha256", "policy_config_sha256",
              "policy config"},
         }) {
      const nlohmann::json* contract_value = nullptr;
      if (contract_section.empty()) {
        contract_value =
            detail::Find(*encoder_contract, {contract_key});
      } else {
        contract_value = detail::Find(
            *encoder_contract, {contract_section, contract_key});
      }
      const auto* hashes_value =
          detail::Find(sidecar, {"hashes", hashes_key});
      if (contract_value == nullptr || hashes_value == nullptr ||
          *contract_value != *hashes_value) {
        errors.emplace_back(std::string(label) + " binding mismatch");
      }
    }
  }
}

inline ValidationResult ValidateShadowArtifact(
    const nlohmann::json& sidecar,
    const nlohmann::json& encoder_embedded,
    const nlohmann::json& decoder_embedded,
    const ModelSignature& encoder_signature,
    const ModelSignature& decoder_signature,
    const PairBinding& binding,
    RequestedMode requested_mode) {
  ValidationResult result;
  if (requested_mode != RequestedMode::Shadow) {
    result.errors.emplace_back(
        "native true23 control is unsupported; only shadow mode is allowed");
  }
  ValidateSidecar(sidecar, result.errors);
  ValidateEmbeddedMetadata(
      encoder_embedded, kEncoderRole, result.errors);
  ValidateEmbeddedMetadata(
      decoder_embedded, kDecoderRole, result.errors);
  ValidatePairModelSignatures(
      encoder_signature, decoder_signature, result.errors);
  ValidateCrossBinding(
      sidecar, encoder_embedded, decoder_embedded, binding, result.errors);

  // Deliberately never authorize mutation.  Future active-control work must
  // introduce a different audited type instead of flipping these defaults.
  result.authorization = ShadowAuthorization{};
  return result;
}

inline bool IsBaseFilename(std::string_view value) {
  return !value.empty() && value != "." && value != ".." &&
         value.find('/') == std::string_view::npos &&
         value.find('\\') == std::string_view::npos;
}

inline nlohmann::json MujocoCandidateBaseContract(
    const nlohmann::json& candidate) {
  nlohmann::json result = nlohmann::json::object();
  for (const auto key :
       {
           "schema_version",
           "robot_model",
           "mode_machine",
           "action_dof",
           "hardware_joint_ids",
           "excluded_hardware_joint_ids",
           "decoder_output_layout",
           "observation_layout",
           "history_length",
           "decoder_input_dim",
           "decoder_output_dim",
           "reference_profile",
           "reference_contract",
           "checkpoint_stage",
           "deployment_ready",
           "sim_validation_passed",
           "naive_output_masking",
           "observation_contract",
           "action_contract",
       }) {
    const auto iterator = candidate.find(key);
    if (iterator != candidate.end()) {
      result[key] = *iterator;
    }
  }
  return result;
}

inline void ValidateMujocoCandidateValidation(
    const nlohmann::json& candidate,
    std::vector<std::string>& errors) {
  const auto* validation = detail::Find(candidate, {"validation"});
  detail::RequireExactObjectKeys(
      validation,
      {"teleop_encoder", "true23_decoder", "pair_dry_run"},
      "candidate.validation", errors);
  for (const auto role : {kEncoderRole, kDecoderRole}) {
    const auto* record =
        validation == nullptr
            ? nullptr
            : detail::Find(*validation, {role});
    const auto context =
        "candidate.validation." + std::string(role);
    detail::RequireExactObjectKeys(
        record,
        {
            "onnx_checker_full_check",
            "shape_inference",
            "ort_provider",
            "parity_case_count",
            "parity_atol",
            "parity_rtol",
            "parity_max_abs_error",
            "parity_max_rel_error",
            "parity_inputs_sha256",
            "parity_outputs_sha256",
        },
        context, errors);
    if (record == nullptr || !record->is_object()) {
      continue;
    }
    detail::RequireEqual(
        *record, {"onnx_checker_full_check"}, true, errors);
    detail::RequireEqual(*record, {"shape_inference"}, true, errors);
    detail::RequireEqual(
        *record, {"ort_provider"}, "CPUExecutionProvider", errors);
    detail::RequireEqual(*record, {"parity_case_count"}, 3, errors);
    detail::RequireSha256(*record, {"parity_inputs_sha256"}, errors);
    detail::RequireSha256(*record, {"parity_outputs_sha256"}, errors);
    for (const auto key :
         {"parity_atol", "parity_rtol", "parity_max_abs_error",
          "parity_max_rel_error"}) {
      const auto* value = detail::Find(*record, {key});
      if (value == nullptr || !value->is_number() ||
          !std::isfinite(value->get<double>()) ||
          value->get<double>() < 0.0) {
        errors.emplace_back(
            context + "." + key +
            " must be a finite number >= 0");
      }
    }
  }

  const auto* dry_run =
      validation == nullptr
          ? nullptr
          : detail::Find(*validation, {"pair_dry_run"});
  detail::RequireExactObjectKeys(
      dry_run,
      {"performed", "device", "dtype", "token_sha256", "action_sha256"},
      "candidate.validation.pair_dry_run", errors);
  if (dry_run != nullptr && dry_run->is_object()) {
    detail::RequireEqual(*dry_run, {"performed"}, true, errors);
    detail::RequireEqual(*dry_run, {"device"}, "cpu", errors);
    detail::RequireEqual(*dry_run, {"dtype"}, "float32", errors);
    detail::RequireSha256(*dry_run, {"token_sha256"}, errors);
    detail::RequireSha256(*dry_run, {"action_sha256"}, errors);
  }
}

inline void ValidateMujocoCandidateSidecar(
    const nlohmann::json& candidate,
    std::vector<std::string>& errors) {
  detail::RequireExactObjectKeys(
      &candidate,
      {
          "schema_version",
          "robot_model",
          "mode_machine",
          "action_dof",
          "hardware_joint_ids",
          "excluded_hardware_joint_ids",
          "decoder_output_layout",
          "observation_layout",
          "history_length",
          "decoder_input_dim",
          "decoder_output_dim",
          "reference_profile",
          "reference_contract",
          "checkpoint_stage",
          "deployment_ready",
          "sim_validation_passed",
          "naive_output_masking",
          "observation_contract",
          "action_contract",
          "artifact_kind",
          "promotion_stage",
          "deployment_authorized",
          "encoder_onnx_filename",
          "decoder_onnx_filename",
          "metadata_filename",
          "onnx_opset",
          "asset_provenance",
          "training_evidence",
          "hashes",
          "validation",
          "metadata_payload_sha256",
      },
      "candidate", errors);
  detail::RequireEqual(
      candidate, {"schema_version"}, kArtifactSchemaVersion, errors);
  detail::RequireEqual(
      candidate, {"artifact_kind"},
      std::string(kMujocoCandidateKind), errors);
  detail::RequireEqual(
      candidate, {"promotion_stage"},
      std::string(kMujocoCandidateStage), errors);
  detail::RequireEqual(
      candidate, {"deployment_authorized"}, false, errors);
  detail::RequireEqual(
      candidate, {"robot_model"}, std::string(kRobotModel), errors);
  detail::RequireEqual(
      candidate, {"mode_machine"}, kRequiredModeMachine, errors);
  detail::RequireEqual(
      candidate, {"action_dof"}, kDecoderOutputDim, errors);
  detail::RequireEqual(
      candidate, {"hardware_joint_ids"},
      detail::ToJsonArray(kHardwareJointIds), errors);
  detail::RequireEqual(
      candidate, {"excluded_hardware_joint_ids"},
      detail::ToJsonArray(kExcludedHardwareJointIds), errors);
  detail::RequireEqual(
      candidate, {"decoder_output_layout"},
      std::string(kDecoderOutputLayout), errors);
  detail::RequireEqual(
      candidate, {"observation_layout"},
      std::string(kObservationLayout), errors);
  detail::RequireEqual(
      candidate, {"history_length"}, kHistoryLength, errors);
  detail::RequireEqual(
      candidate, {"decoder_input_dim"}, kDecoderInputDim, errors);
  detail::RequireEqual(
      candidate, {"decoder_output_dim"}, kDecoderOutputDim, errors);
  detail::RequireEqual(
      candidate, {"checkpoint_stage"}, "trained", errors);
  detail::RequireEqual(
      candidate, {"deployment_ready"}, false, errors);
  detail::RequireEqual(
      candidate, {"sim_validation_passed"}, false, errors);
  detail::RequireEqual(
      candidate, {"naive_output_masking"}, false, errors);
  detail::RequireEqual(
      candidate, {"onnx_opset"}, kOnnxOpset, errors);
  ValidateReferenceProfileBinding(
      detail::Find(candidate, {"reference_profile"}),
      detail::Find(candidate, {"reference_contract"}),
      "candidate", errors);
  const auto* profile =
      detail::Find(candidate, {"reference_profile"});
  if (profile != nullptr && profile->is_string()) {
    ValidateTrainingEvidence(
        detail::Find(candidate, {"training_evidence"}),
        profile->get_ref<const std::string&>(),
        "candidate.training_evidence", errors);
  }

  const auto* observation =
      detail::Find(candidate, {"observation_contract"});
  if (observation == nullptr || !observation->is_object()) {
    errors.emplace_back("candidate.observation_contract must be an object");
  } else {
    detail::RequireEqual(
        *observation, {"token_dim"}, kEncoderOutputDim, errors);
    detail::RequireEqual(
        *observation, {"proprioception_dim"},
        kDecoderInputDim - kEncoderOutputDim, errors);
    detail::RequireEqual(
        *observation, {"term_order"},
        nlohmann::json::array(
            {"base_ang_vel", "joint_pos_rel", "joint_vel",
             "previous_action", "projected_gravity"}),
        errors);
    detail::RequireEqual(
        *observation, {"history_order"}, "oldest_to_newest", errors);
    detail::RequireEqual(
        *observation, {"source_il29_keep_indices"},
        detail::ToJsonArray(kSourceIl29KeepIndices), errors);
    detail::RequireEqual(
        *observation, {"source_il29_excluded_indices"},
        detail::ToJsonArray(kSourceIl29ExcludedIndices), errors);
    detail::RequireEqual(
        *observation, {"native_il23_to_canonical_il29"},
        detail::ToJsonArray(kNativeToCanonicalIl29), errors);
    detail::RequireEqual(
        *observation, {"missing_fill"},
        nlohmann::json{
            {"joint_pos_rel", "fixed_default_relative_zero"},
            {"joint_vel", "zero"},
            {"previous_action", "zero_every_history_frame"},
        },
        errors);
  }
  const auto* action =
      detail::Find(candidate, {"action_contract"});
  if (action == nullptr || !action->is_object()) {
    errors.emplace_back("candidate.action_contract must be an object");
  } else {
    detail::RequireEqual(
        *action, {"native_il23_joint_names"},
        detail::ToJsonArray(kNativeJointNames), errors);
    detail::RequireEqual(
        *action, {"hardware_joint_names"},
        detail::ToJsonArray(kHardwareJointNames), errors);
    detail::RequireEqual(
        *action, {"isaaclab_to_mujoco_dof"},
        detail::ToJsonArray(kNativeToHardwareCompact), errors);
    detail::RequireEqual(
        *action, {"mujoco_to_isaaclab_dof"},
        detail::ToJsonArray(kHardwareCompactToNative), errors);
    detail::RequireEqual(
        *action, {"hardware_action_scale"},
        detail::ToJsonArray(kHardwareActionScale), errors);
    detail::RequireEqual(
        *action, {"native_il23_action_scale"},
        detail::ToJsonArray(kNativeActionScale), errors);
  }

  const auto* hashes = detail::Find(candidate, {"hashes"});
  detail::RequireExactObjectKeys(
      hashes,
      {
          "checkpoint_sha256",
          "policy_state_sha256",
          "encoder_state_sha256",
          "decoder_state_sha256",
          "encoder_onnx_sha256",
          "decoder_onnx_sha256",
          "training_evidence_sha256",
          "contract_sha256",
          "urdf_sha256",
          "mjcf_sha256",
          "robot_config_sha256",
          "asset_manifest_sha256",
          "encoder_embedded_metadata_sha256",
          "decoder_embedded_metadata_sha256",
      },
      "candidate.hashes", errors);
  if (hashes != nullptr && hashes->is_object()) {
    for (const auto key :
         {
             "checkpoint_sha256",
             "policy_state_sha256",
             "encoder_state_sha256",
             "decoder_state_sha256",
             "encoder_onnx_sha256",
             "decoder_onnx_sha256",
             "training_evidence_sha256",
             "contract_sha256",
             "urdf_sha256",
             "mjcf_sha256",
             "robot_config_sha256",
             "asset_manifest_sha256",
             "encoder_embedded_metadata_sha256",
             "decoder_embedded_metadata_sha256",
         }) {
      detail::RequireSha256(*hashes, {key}, errors);
    }
    detail::RequireEqual(
        *hashes, {"urdf_sha256"},
        std::string(kPinnedUnitreeUrdfSha256), errors);
    detail::RequireEqual(
        *hashes, {"mjcf_sha256"},
        std::string(kPinnedUnitreeMjcfSha256), errors);
    detail::RequireEqual(
        *hashes, {"asset_manifest_sha256"},
        std::string(kPinnedUnitreeManifestSha256), errors);
    detail::RequireEqual(
        *hashes, {"robot_config_sha256"},
        std::string(kPinnedRobotConfigSha256), errors);
    const auto* training =
        detail::Find(candidate, {"training_evidence"});
    const auto* training_sha =
        detail::Find(*hashes, {"training_evidence_sha256"});
    if (training == nullptr || training_sha == nullptr ||
        !training_sha->is_string() ||
        training_sha->get_ref<const std::string&>() !=
            Sha256CanonicalJson(*training)) {
      errors.emplace_back(
          "candidate training evidence SHA-256 mismatch");
    }
    const auto* trained_policy = detail::Find(
        candidate, {"training_evidence", "policy_state_sha256"});
    const auto* candidate_policy =
        detail::Find(*hashes, {"policy_state_sha256"});
    if (trained_policy == nullptr || candidate_policy == nullptr ||
        *trained_policy != *candidate_policy) {
      errors.emplace_back(
          "candidate policy hash differs from training evidence");
    }
    const auto contract = MujocoCandidateBaseContract(candidate);
    const auto* contract_sha =
        detail::Find(*hashes, {"contract_sha256"});
    if (contract_sha == nullptr || !contract_sha->is_string() ||
        contract_sha->get_ref<const std::string&>() !=
            Sha256CanonicalJson(contract)) {
      errors.emplace_back("candidate contract SHA-256 mismatch");
    }
  }

  const auto* provenance =
      detail::Find(candidate, {"asset_provenance"});
  detail::RequireExactObjectKeys(
      provenance,
      {
          "repository",
          "revision",
          "root_relpath",
          "text_normalization",
          "file_count",
          "total_bytes",
          "manifest_sha256",
          "urdf_sha256",
          "mjcf_sha256",
          "verified",
      },
      "candidate.asset_provenance", errors);
  if (provenance != nullptr && provenance->is_object()) {
    detail::RequireEqual(
        *provenance, {"repository"},
        std::string(kPinnedUnitreeRepository), errors);
    detail::RequireEqual(
        *provenance, {"revision"},
        std::string(kPinnedUnitreeRevision), errors);
    detail::RequireEqual(
        *provenance, {"root_relpath"},
        std::string(kPinnedUnitreeRootRelpath), errors);
    detail::RequireEqual(
        *provenance, {"text_normalization"},
        std::string(kPinnedUnitreeTextNormalization), errors);
    detail::RequireEqual(
        *provenance, {"file_count"}, kPinnedUnitreeFileCount, errors);
    detail::RequireEqual(
        *provenance, {"total_bytes"}, kPinnedUnitreeTotalBytes, errors);
    detail::RequireEqual(
        *provenance, {"manifest_sha256"},
        std::string(kPinnedUnitreeManifestSha256), errors);
    detail::RequireEqual(
        *provenance, {"urdf_sha256"},
        std::string(kPinnedUnitreeUrdfSha256), errors);
    detail::RequireEqual(
        *provenance, {"mjcf_sha256"},
        std::string(kPinnedUnitreeMjcfSha256), errors);
    detail::RequireEqual(*provenance, {"verified"}, true, errors);
  }
  ValidateMujocoCandidateValidation(candidate, errors);

  auto unhashed = candidate;
  const auto erased = unhashed.erase("metadata_payload_sha256");
  const auto* payload_hash =
      detail::Find(candidate, {"metadata_payload_sha256"});
  if (erased != 1 || payload_hash == nullptr ||
      !payload_hash->is_string() ||
      payload_hash->get_ref<const std::string&>() !=
          Sha256CanonicalJson(unhashed)) {
    errors.emplace_back("candidate metadata payload SHA-256 mismatch");
  }
}

inline void ValidateMujocoCandidateEmbeddedMetadata(
    const nlohmann::json& embedded,
    std::string_view expected_role,
    std::vector<std::string>& errors) {
  detail::RequireExactObjectKeys(
      &embedded,
      {
          "schema_version",
          "kind",
          "artifact_role",
          "promotion_stage",
          "deployment_authorized",
          "robot_model",
          "checkpoint_stage",
          "checkpoint_sha256",
          "policy_state_sha256",
          "encoder_state_sha256",
          "decoder_state_sha256",
          "training_evidence_sha256",
          "contract_sha256",
          "global_step",
          "reference_profile",
          "reference_contract",
          "encoder_input_dim",
          "token_dim",
          "decoder_input_dim",
          "decoder_output_dim",
          "naive_output_masking",
      },
      "candidate embedded metadata", errors);
  detail::RequireEqual(
      embedded, {"schema_version"},
      kMujocoCandidateSchemaVersion, errors);
  detail::RequireEqual(
      embedded, {"kind"},
      std::string(kMujocoCandidateEmbeddedKind), errors);
  detail::RequireEqual(
      embedded, {"artifact_role"}, std::string(expected_role), errors);
  detail::RequireEqual(
      embedded, {"promotion_stage"},
      std::string(kMujocoCandidateStage), errors);
  detail::RequireEqual(
      embedded, {"deployment_authorized"}, false, errors);
  detail::RequireEqual(
      embedded, {"robot_model"}, std::string(kRobotModel), errors);
  detail::RequireEqual(
      embedded, {"checkpoint_stage"}, "trained", errors);
  detail::RequireEqual(
      embedded, {"encoder_input_dim"}, kEncoderInputDim, errors);
  detail::RequireEqual(
      embedded, {"token_dim"}, kEncoderOutputDim, errors);
  detail::RequireEqual(
      embedded, {"decoder_input_dim"}, kDecoderInputDim, errors);
  detail::RequireEqual(
      embedded, {"decoder_output_dim"}, kDecoderOutputDim, errors);
  detail::RequireEqual(
      embedded, {"naive_output_masking"}, false, errors);
  ValidateReferenceProfileBinding(
      detail::Find(embedded, {"reference_profile"}),
      detail::Find(embedded, {"reference_contract"}),
      "candidate embedded metadata", errors);
  if (!detail::PositiveInteger(
          detail::Find(embedded, {"global_step"}))) {
    errors.emplace_back(
        "candidate embedded metadata global_step must be positive");
  }
  for (const auto key :
       {
           "checkpoint_sha256",
           "policy_state_sha256",
           "encoder_state_sha256",
           "decoder_state_sha256",
           "training_evidence_sha256",
           "contract_sha256",
       }) {
    detail::RequireSha256(embedded, {key}, errors);
  }
}

inline void ValidateMujocoPromotionEvidence(
    const nlohmann::json& promotion,
    const nlohmann::json& candidate,
    std::vector<std::string>& errors) {
  const auto* evidence =
      detail::Find(promotion, {"mujoco_evidence"});
  detail::RequireExactObjectKeys(
      evidence,
      {
          "schema_version",
          "computed_pass",
          "report_sha256",
          "report_payload_sha256",
          "source_artifact",
          "producer",
          "simulator",
          "trace_manifest_sha256",
          "trace_count",
          "scenario_count",
          "run_count",
          "episodes_per_scenario",
          "total_episodes",
          "total_records",
          "deterministic_onnx_mujoco_replay_verified",
          "summary_metrics",
      },
      "promotion.mujoco_evidence", errors);
  if (evidence == nullptr || !evidence->is_object()) {
    return;
  }
  detail::RequireEqual(
      *evidence, {"schema_version"}, kMujocoPromotionSchemaVersion,
      errors);
  detail::RequireEqual(*evidence, {"computed_pass"}, true, errors);
  detail::RequireEqual(
      *evidence, {"deterministic_onnx_mujoco_replay_verified"},
      true, errors);
  for (const auto key :
       {"report_sha256", "report_payload_sha256",
        "trace_manifest_sha256"}) {
    detail::RequireSha256(*evidence, {key}, errors);
  }
  detail::RequireEqual(*evidence, {"trace_count"}, 9, errors);
  detail::RequireEqual(*evidence, {"scenario_count"}, 3, errors);
  detail::RequireEqual(*evidence, {"run_count"}, 9, errors);
  detail::RequireEqual(
      *evidence, {"episodes_per_scenario"}, 66, errors);
  detail::RequireEqual(*evidence, {"total_episodes"}, 198, errors);
  detail::RequireEqual(*evidence, {"total_records"}, 49500, errors);

  const auto* source =
      detail::Find(*evidence, {"source_artifact"});
  detail::RequireExactObjectKeys(
      source,
      {
          "artifact_kind",
          "checkpoint_sha256",
          "policy_state_sha256",
          "encoder_onnx_sha256",
          "decoder_onnx_sha256",
          "candidate_manifest_sha256",
          "candidate_manifest_payload_sha256",
          "candidate_claimed_payload_sha256",
          "inference_runtime",
          "inference_threads",
      },
      "promotion.mujoco_evidence.source_artifact", errors);
  const auto* promoted_source =
      detail::Find(promotion, {"source_candidate"});
  if (source != nullptr && source->is_object() &&
      promoted_source != nullptr && promoted_source->is_object()) {
    for (const auto key :
         {
             "artifact_kind",
             "checkpoint_sha256",
             "policy_state_sha256",
             "encoder_onnx_sha256",
             "decoder_onnx_sha256",
             "candidate_manifest_sha256",
             "candidate_manifest_payload_sha256",
             "candidate_claimed_payload_sha256",
             "inference_runtime",
             "inference_threads",
         }) {
      const auto* evidence_value = detail::Find(*source, {key});
      const auto* source_value =
          detail::Find(*promoted_source, {key});
      if (evidence_value == nullptr || source_value == nullptr ||
          *evidence_value != *source_value) {
        errors.emplace_back(
            "MuJoCo evidence/source candidate mismatch: " +
            std::string(key));
      }
    }
  }

  const auto* producer = detail::Find(*evidence, {"producer"});
  detail::RequireExactObjectKeys(
      producer,
      {"kind", "version", "runner_sha256", "runtime_sha256"},
      "promotion.mujoco_evidence.producer", errors);
  if (producer != nullptr && producer->is_object()) {
    detail::RequireEqual(
        *producer, {"kind"}, std::string(kMujocoProducerKind), errors);
    detail::RequireEqual(*producer, {"version"}, 1, errors);
    detail::RequireSha256(*producer, {"runner_sha256"}, errors);
    detail::RequireSha256(*producer, {"runtime_sha256"}, errors);
    detail::RequireEqual(
        *producer, {"runner_sha256"},
        std::string(kPinnedMujocoRunnerSha256), errors);
    detail::RequireEqual(
        *producer, {"runtime_sha256"},
        std::string(kPinnedMujocoRuntimeSha256), errors);
  }

  const auto* simulator = detail::Find(*evidence, {"simulator"});
  detail::RequireExactObjectKeys(
      simulator,
      {
          "name",
          "version",
          "mjcf_sha256",
          "config_sha256",
          "physics_contract_sha256",
          "asset_manifest_sha256",
      },
      "promotion.mujoco_evidence.simulator", errors);
  if (simulator != nullptr && simulator->is_object()) {
    detail::RequireEqual(*simulator, {"name"}, "MuJoCo", errors);
    detail::RequireEqual(
        *simulator, {"version"}, std::string(kMujocoVersion), errors);
    for (const auto key :
         {"mjcf_sha256", "config_sha256",
          "physics_contract_sha256", "asset_manifest_sha256"}) {
      detail::RequireSha256(*simulator, {key}, errors);
    }
    detail::RequireEqual(
        *simulator, {"config_sha256"},
        std::string(kPinnedMujocoConfigSha256), errors);
    detail::RequireEqual(
        *simulator, {"physics_contract_sha256"},
        std::string(kPinnedMujocoPhysicsContractSha256), errors);
    const auto* candidate_mjcf =
        detail::Find(candidate, {"hashes", "mjcf_sha256"});
    const auto* simulator_mjcf =
        detail::Find(*simulator, {"mjcf_sha256"});
    if (candidate_mjcf == nullptr || simulator_mjcf == nullptr ||
        *candidate_mjcf != *simulator_mjcf) {
      errors.emplace_back(
          "MuJoCo evidence MJCF differs from candidate");
    }
    const auto* candidate_assets =
        detail::Find(candidate, {"hashes", "asset_manifest_sha256"});
    const auto* simulator_assets =
        detail::Find(*simulator, {"asset_manifest_sha256"});
    if (candidate_assets == nullptr || simulator_assets == nullptr ||
        *candidate_assets != *simulator_assets) {
      errors.emplace_back(
          "MuJoCo evidence asset manifest differs from candidate");
    }
  }

  const auto* metrics =
      detail::Find(*evidence, {"summary_metrics"});
  detail::RequireExactObjectKeys(
      metrics,
      {
          "run_count",
          "episode_count",
          "record_count",
          "termination_count",
          "nonfinite_count",
          "joint_limit_violation_count",
          "min_base_height_m",
          "max_tilt_rad",
          "max_tracking_rmse_rad",
          "max_abs_joint_velocity_radps",
          "max_abs_applied_torque_nm",
          "max_abs_native_action",
          "max_abs_native_action_raw",
          "max_action_saturation_fraction",
          "minimum_recovery_fraction",
          "max_recovery_time_s",
      },
      "promotion.mujoco_evidence.summary_metrics", errors);
  if (metrics != nullptr && metrics->is_object()) {
    detail::RequireEqual(*metrics, {"run_count"}, 9, errors);
    detail::RequireEqual(*metrics, {"episode_count"}, 198, errors);
    detail::RequireEqual(*metrics, {"record_count"}, 49500, errors);
    detail::RequireEqual(*metrics, {"termination_count"}, 0, errors);
    detail::RequireEqual(*metrics, {"nonfinite_count"}, 0, errors);
    detail::RequireEqual(
        *metrics, {"joint_limit_violation_count"}, 0, errors);
    const auto require_range =
        [&](std::string_view key, double minimum, double maximum) {
          const auto* value = detail::Find(*metrics, {key});
          if (value == nullptr || !value->is_number() ||
              !std::isfinite(value->get<double>()) ||
              value->get<double>() < minimum ||
              value->get<double>() > maximum) {
            errors.emplace_back(
                "promotion.mujoco_evidence.summary_metrics." +
                std::string(key) + " outside promotion envelope");
          }
        };
    require_range("min_base_height_m", 0.45, 10.0);
    require_range("max_tilt_rad", 0.0, 1.0);
    require_range("max_tracking_rmse_rad", 0.0, 0.75);
    require_range("max_abs_joint_velocity_radps", 0.0, 37.0);
    require_range("max_abs_applied_torque_nm", 0.0, 139.0);
    require_range("max_abs_native_action", 0.0, 20.0);
    require_range("max_abs_native_action_raw", 0.0, 1.0e9);
    require_range(
        "max_action_saturation_fraction", 0.0, 0.1);
    require_range("minimum_recovery_fraction", 1.0, 1.0);
    require_range("max_recovery_time_s", 0.0, 2.0);
  }
}

inline void ValidateMujocoPromotionSidecar(
    const nlohmann::json& promotion,
    const nlohmann::json& candidate,
    const PairBinding& binding,
    std::vector<std::string>& errors) {
  detail::RequireExactObjectKeys(
      &promotion,
      {
          "schema_version",
          "kind",
          "robot_model",
          "promotion_stage",
          "deployment_authorized",
          "active_motor_control_authorized",
          "checkpoint_stage",
          "source_candidate",
          "mujoco_evidence",
          "deployment_conditions",
          "promotion_payload_sha256",
      },
      "promotion", errors);
  detail::RequireEqual(
      promotion, {"schema_version"},
      kMujocoPromotionSchemaVersion, errors);
  detail::RequireEqual(
      promotion, {"kind"}, std::string(kMujocoPromotionKind), errors);
  detail::RequireEqual(
      promotion, {"robot_model"}, std::string(kRobotModel), errors);
  detail::RequireEqual(
      promotion, {"promotion_stage"},
      std::string(kMujocoPromotedStage), errors);
  detail::RequireEqual(
      promotion, {"deployment_authorized"}, true, errors);
  detail::RequireEqual(
      promotion, {"active_motor_control_authorized"}, false, errors);
  detail::RequireEqual(
      promotion, {"checkpoint_stage"}, "trained", errors);

  const auto* conditions =
      detail::Find(promotion, {"deployment_conditions"});
  detail::RequireExactObjectKeys(
      conditions,
      {
          "mode_machine",
          "paired_onnx_bytes_must_remain_unchanged",
          "live_shadow_required",
          "gantry_or_rated_support_required_for_first_actuation",
          "free_standing_first_actuation_authorized",
      },
      "promotion.deployment_conditions", errors);
  if (conditions != nullptr && conditions->is_object()) {
    detail::RequireEqual(
        *conditions, {"mode_machine"}, kRequiredModeMachine, errors);
    detail::RequireEqual(
        *conditions, {"paired_onnx_bytes_must_remain_unchanged"},
        true, errors);
    detail::RequireEqual(
        *conditions, {"live_shadow_required"}, true, errors);
    detail::RequireEqual(
        *conditions,
        {"gantry_or_rated_support_required_for_first_actuation"},
        true, errors);
    detail::RequireEqual(
        *conditions, {"free_standing_first_actuation_authorized"},
        false, errors);
  }

  const auto* source =
      detail::Find(promotion, {"source_candidate"});
  detail::RequireExactObjectKeys(
      source,
      {
          "checkpoint_filename",
          "encoder_onnx_filename",
          "decoder_onnx_filename",
          "metadata_filename",
          "artifact_kind",
          "checkpoint_sha256",
          "policy_state_sha256",
          "encoder_onnx_sha256",
          "decoder_onnx_sha256",
          "candidate_manifest_sha256",
          "candidate_manifest_payload_sha256",
          "candidate_claimed_payload_sha256",
          "inference_runtime",
          "inference_threads",
      },
      "promotion.source_candidate", errors);
  if (source != nullptr && source->is_object()) {
    detail::RequireEqual(
        *source, {"artifact_kind"},
        std::string(kMujocoCandidateSourceKind), errors);
    detail::RequireEqual(
        *source, {"encoder_onnx_filename"},
        binding.encoder_filename, errors);
    detail::RequireEqual(
        *source, {"decoder_onnx_filename"},
        binding.decoder_filename, errors);
    detail::RequireEqual(
        *source, {"metadata_filename"},
        binding.metadata_filename, errors);
    detail::RequireEqual(
        *source, {"encoder_onnx_sha256"},
        binding.encoder_onnx_sha256, errors);
    detail::RequireEqual(
        *source, {"decoder_onnx_sha256"},
        binding.decoder_onnx_sha256, errors);
    detail::RequireEqual(
        *source, {"candidate_manifest_sha256"},
        binding.metadata_sha256, errors);
    detail::RequireEqual(
        *source, {"inference_runtime"}, "onnxruntime_cpu", errors);
    detail::RequireEqual(
        *source, {"inference_threads"}, 1, errors);
    for (const auto key :
         {
             "checkpoint_sha256",
             "policy_state_sha256",
             "encoder_onnx_sha256",
             "decoder_onnx_sha256",
             "candidate_manifest_sha256",
             "candidate_manifest_payload_sha256",
             "candidate_claimed_payload_sha256",
         }) {
      detail::RequireSha256(*source, {key}, errors);
    }
    const auto* checkpoint_filename =
        detail::Find(*source, {"checkpoint_filename"});
    if (checkpoint_filename == nullptr ||
        !checkpoint_filename->is_string() ||
        !IsBaseFilename(
            checkpoint_filename->get_ref<const std::string&>()) ||
        !checkpoint_filename->get_ref<const std::string&>()
             .ends_with(".promotion.pt")) {
      errors.emplace_back(
          "promotion.source_candidate.checkpoint_filename must be "
          "a basename ending .promotion.pt");
    }
    const auto* candidate_hashes =
        detail::Find(candidate, {"hashes"});
    if (candidate_hashes != nullptr &&
        candidate_hashes->is_object()) {
      for (const auto key :
           {"checkpoint_sha256", "policy_state_sha256",
            "encoder_onnx_sha256", "decoder_onnx_sha256"}) {
        const auto* source_value = detail::Find(*source, {key});
        const auto* candidate_value =
            detail::Find(*candidate_hashes, {key});
        if (source_value == nullptr || candidate_value == nullptr ||
            *source_value != *candidate_value) {
          errors.emplace_back(
              "promotion/candidate hash mismatch: " +
              std::string(key));
        }
      }
    }
    const auto* manifest_payload =
        detail::Find(*source, {"candidate_manifest_payload_sha256"});
    if (manifest_payload == nullptr ||
        !manifest_payload->is_string() ||
        manifest_payload->get_ref<const std::string&>() !=
            Sha256CanonicalJson(candidate)) {
      errors.emplace_back(
          "candidate manifest canonical payload SHA-256 mismatch");
    }
    const auto* claimed_payload =
        detail::Find(*source, {"candidate_claimed_payload_sha256"});
    const auto* candidate_payload =
        detail::Find(candidate, {"metadata_payload_sha256"});
    if (claimed_payload == nullptr || candidate_payload == nullptr ||
        *claimed_payload != *candidate_payload) {
      errors.emplace_back(
          "candidate claimed payload SHA-256 mismatch");
    }
  }

  auto unhashed = promotion;
  const auto erased = unhashed.erase("promotion_payload_sha256");
  const auto* payload_hash =
      detail::Find(promotion, {"promotion_payload_sha256"});
  if (erased != 1 || payload_hash == nullptr ||
      !payload_hash->is_string() ||
      payload_hash->get_ref<const std::string&>() !=
          Sha256CanonicalJson(unhashed)) {
    errors.emplace_back("promotion payload SHA-256 mismatch");
  }
  ValidateMujocoPromotionEvidence(promotion, candidate, errors);
}

inline void ValidateMujocoPromotionCrossBinding(
    const nlohmann::json& promotion,
    const nlohmann::json& candidate,
    const nlohmann::json& encoder_embedded,
    const nlohmann::json& decoder_embedded,
    const PairBinding& binding,
    std::vector<std::string>& errors) {
  const auto require_candidate_hash =
      [&](std::string_view candidate_key,
          std::string_view embedded_key,
          std::string_view label) {
        const auto* candidate_value =
            detail::Find(candidate, {"hashes", candidate_key});
        const auto* encoder_value =
            detail::Find(encoder_embedded, {embedded_key});
        const auto* decoder_value =
            detail::Find(decoder_embedded, {embedded_key});
        if (candidate_value == nullptr || encoder_value == nullptr ||
            decoder_value == nullptr ||
            *candidate_value != *encoder_value ||
            *candidate_value != *decoder_value) {
          errors.emplace_back(
              "candidate/embedded " + std::string(label) +
              " mismatch");
        }
      };
  for (const auto values :
       std::array<std::array<std::string_view, 3>, 6>{
           std::array<std::string_view, 3>{
               "checkpoint_sha256", "checkpoint_sha256",
               "checkpoint"},
           {"policy_state_sha256", "policy_state_sha256", "policy"},
           {"encoder_state_sha256", "encoder_state_sha256",
            "encoder state"},
           {"decoder_state_sha256", "decoder_state_sha256",
            "decoder state"},
           {"training_evidence_sha256",
            "training_evidence_sha256", "training evidence"},
           {"contract_sha256", "contract_sha256", "contract"},
       }) {
    require_candidate_hash(values[0], values[1], values[2]);
  }
  const auto* candidate_profile =
      detail::Find(candidate, {"reference_profile"});
  const auto* encoder_profile =
      detail::Find(encoder_embedded, {"reference_profile"});
  const auto* decoder_profile =
      detail::Find(decoder_embedded, {"reference_profile"});
  if (candidate_profile == nullptr || encoder_profile == nullptr ||
      decoder_profile == nullptr ||
      *candidate_profile != *encoder_profile ||
      *candidate_profile != *decoder_profile) {
    errors.emplace_back(
        "candidate/embedded reference profile mismatch");
  }
  const auto* candidate_contract =
      detail::Find(candidate, {"reference_contract"});
  const auto* encoder_contract =
      detail::Find(encoder_embedded, {"reference_contract"});
  const auto* decoder_contract =
      detail::Find(decoder_embedded, {"reference_contract"});
  if (candidate_contract == nullptr || encoder_contract == nullptr ||
      decoder_contract == nullptr ||
      *candidate_contract != *encoder_contract ||
      *candidate_contract != *decoder_contract) {
    errors.emplace_back(
        "candidate/embedded reference contract mismatch");
  }
  const auto* candidate_step =
      detail::Find(candidate, {"training_evidence", "global_step"});
  const auto* encoder_step =
      detail::Find(encoder_embedded, {"global_step"});
  const auto* decoder_step =
      detail::Find(decoder_embedded, {"global_step"});
  if (candidate_step == nullptr || encoder_step == nullptr ||
      decoder_step == nullptr ||
      *candidate_step != *encoder_step ||
      *candidate_step != *decoder_step) {
    errors.emplace_back("candidate/embedded global step mismatch");
  }
  const auto* encoder_hash = detail::Find(
      candidate, {"hashes", "encoder_embedded_metadata_sha256"});
  if (encoder_hash == nullptr || !encoder_hash->is_string() ||
      encoder_hash->get_ref<const std::string&>() !=
          Sha256CanonicalJson(encoder_embedded)) {
    errors.emplace_back(
        "candidate encoder embedded metadata SHA-256 mismatch");
  }
  const auto* decoder_hash = detail::Find(
      candidate, {"hashes", "decoder_embedded_metadata_sha256"});
  if (decoder_hash == nullptr || !decoder_hash->is_string() ||
      decoder_hash->get_ref<const std::string&>() !=
          Sha256CanonicalJson(decoder_embedded)) {
    errors.emplace_back(
        "candidate decoder embedded metadata SHA-256 mismatch");
  }
  const auto* candidate_encoder =
      detail::Find(candidate, {"hashes", "encoder_onnx_sha256"});
  const auto* candidate_decoder =
      detail::Find(candidate, {"hashes", "decoder_onnx_sha256"});
  if (candidate_encoder == nullptr ||
      !candidate_encoder->is_string() ||
      candidate_encoder->get_ref<const std::string&>() !=
          binding.encoder_onnx_sha256) {
    errors.emplace_back("candidate encoder ONNX byte hash mismatch");
  }
  if (candidate_decoder == nullptr ||
      !candidate_decoder->is_string() ||
      candidate_decoder->get_ref<const std::string&>() !=
          binding.decoder_onnx_sha256) {
    errors.emplace_back("candidate decoder ONNX byte hash mismatch");
  }
  detail::RequireEqual(
      candidate, {"encoder_onnx_filename"},
      binding.encoder_filename, errors);
  detail::RequireEqual(
      candidate, {"decoder_onnx_filename"},
      binding.decoder_filename, errors);
  detail::RequireEqual(
      candidate, {"metadata_filename"},
      binding.metadata_filename, errors);
  if (binding.encoder_path == binding.decoder_path ||
      binding.encoder_path == binding.metadata_path ||
      binding.decoder_path == binding.metadata_path) {
    errors.emplace_back(
        "candidate encoder, decoder, and metadata paths must differ");
  }
  ValidateMujocoPromotionSidecar(
      promotion, candidate, binding, errors);
}

inline ValidationResult ValidateMujocoShadowPromotion(
    const nlohmann::json& promotion,
    const nlohmann::json& candidate,
    const nlohmann::json& encoder_embedded,
    const nlohmann::json& decoder_embedded,
    const ModelSignature& encoder_signature,
    const ModelSignature& decoder_signature,
    const PairBinding& binding,
    RequestedMode requested_mode) {
  ValidationResult result;
  if (requested_mode != RequestedMode::Shadow) {
    result.errors.emplace_back(
        "MuJoCo promotion is valid only for shadow mode; active "
        "true23 control remains unsupported");
  }
  ValidateMujocoCandidateSidecar(candidate, result.errors);
  ValidateMujocoCandidateEmbeddedMetadata(
      encoder_embedded, kEncoderRole, result.errors);
  ValidateMujocoCandidateEmbeddedMetadata(
      decoder_embedded, kDecoderRole, result.errors);
  ValidatePairModelSignatures(
      encoder_signature, decoder_signature, result.errors);
  ValidateMujocoPromotionCrossBinding(
      promotion, candidate, encoder_embedded, decoder_embedded,
      binding, result.errors);

  // Promotion authorizes unchanged ONNX bytes for read-only shadow use only.
  // It never grants a publisher, writer, or motion-mode transition.
  result.authorization = ShadowAuthorization{};
  return result;
}

enum class ModeObservation {
  DuplicateTick,
  TickRegression,
  Waiting,
  Ready,
  WrongMode,
  LatchedFailure,
};

class ModeMachineShadowGate {
 public:
  ModeObservation Observe(std::uint32_t tick, std::uint8_t mode_machine) {
    if (latched_failure_) {
      return ModeObservation::LatchedFailure;
    }
    if (ready_ && mode_machine != kRequiredModeMachine) {
      ready_ = false;
      stable_samples_ = 0;
      latched_failure_ = true;
      return ModeObservation::LatchedFailure;
    }
    if (mode_machine != kRequiredModeMachine) {
      stable_samples_ = 0;
      return ModeObservation::WrongMode;
    }
    if (have_tick_ && tick == last_tick_) {
      return ModeObservation::DuplicateTick;
    }
    if (have_tick_ &&
        static_cast<std::int32_t>(tick - last_tick_) < 0) {
      ready_ = false;
      stable_samples_ = 0;
      latched_failure_ = true;
      return ModeObservation::TickRegression;
    }
    have_tick_ = true;
    last_tick_ = tick;

    if (stable_samples_ < kStableModeSamples) {
      ++stable_samples_;
    }
    if (stable_samples_ >= kStableModeSamples) {
      ready_ = true;
      return ModeObservation::Ready;
    }
    return ModeObservation::Waiting;
  }

  [[nodiscard]] bool ready() const { return ready_ && !latched_failure_; }
  [[nodiscard]] bool latched_failure() const { return latched_failure_; }
  [[nodiscard]] int stable_samples() const { return stable_samples_; }

 private:
  std::uint32_t last_tick_ = 0;
  int stable_samples_ = 0;
  bool have_tick_ = false;
  bool ready_ = false;
  bool latched_failure_ = false;
};

// Small self-contained SHA-256 implementation used only to bind the ONNX file
// read by C++ to the hash in its validated metadata sidecar.
class Sha256 {
 public:
  Sha256()
      : state_{0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
               0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U} {}

  void Update(const std::uint8_t* data, std::size_t length) {
    for (std::size_t index = 0; index < length; ++index) {
      buffer_[buffer_size_++] = data[index];
      if (buffer_size_ == buffer_.size()) {
        Transform();
        bit_length_ += 512;
        buffer_size_ = 0;
      }
    }
  }

  [[nodiscard]] std::string FinalHex() {
    const auto original_bits =
        bit_length_ + static_cast<std::uint64_t>(buffer_size_) * 8U;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56) {
      while (buffer_size_ < 64) {
        buffer_[buffer_size_++] = 0;
      }
      Transform();
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56) {
      buffer_[buffer_size_++] = 0;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
      buffer_[buffer_size_++] =
          static_cast<std::uint8_t>((original_bits >> shift) & 0xffU);
    }
    Transform();

    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (const auto value : state_) {
      stream << std::setw(8) << value;
    }
    return stream.str();
  }

 private:
  static constexpr std::array<std::uint32_t, 64> kRoundConstants = {
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
      0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
      0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
      0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
      0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
      0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
      0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
      0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
      0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
  };

  static constexpr std::uint32_t RotateRight(std::uint32_t value,
                                              std::uint32_t count) {
    return (value >> count) | (value << (32U - count));
  }

  void Transform() {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      const std::size_t offset = index * 4;
      words[index] =
          (static_cast<std::uint32_t>(buffer_[offset]) << 24U) |
          (static_cast<std::uint32_t>(buffer_[offset + 1]) << 16U) |
          (static_cast<std::uint32_t>(buffer_[offset + 2]) << 8U) |
          static_cast<std::uint32_t>(buffer_[offset + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const std::uint32_t s0 =
          RotateRight(words[index - 15], 7U) ^
          RotateRight(words[index - 15], 18U) ^
          (words[index - 15] >> 3U);
      const std::uint32_t s1 =
          RotateRight(words[index - 2], 17U) ^
          RotateRight(words[index - 2], 19U) ^
          (words[index - 2] >> 10U);
      words[index] =
          words[index - 16] + s0 + words[index - 7] + s1;
    }

    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const std::uint32_t sigma1 =
          RotateRight(e, 6U) ^ RotateRight(e, 11U) ^
          RotateRight(e, 25U);
      const std::uint32_t choose = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 =
          h + sigma1 + choose + kRoundConstants[index] + words[index];
      const std::uint32_t sigma0 =
          RotateRight(a, 2U) ^ RotateRight(a, 13U) ^
          RotateRight(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sigma0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_length_ = 0;
};

inline std::string Sha256Bytes(std::string_view bytes) {
  Sha256 digest;
  digest.Update(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
  return digest.FinalHex();
}

inline std::string Sha256CanonicalJson(const nlohmann::json& value) {
  std::string canonical = value.dump();
  canonical.push_back('\n');
  return Sha256Bytes(canonical);
}

namespace detail {

class ProtobufCursor {
 public:
  explicit ProtobufCursor(std::span<const std::uint8_t> bytes)
      : bytes_(bytes) {}

  [[nodiscard]] bool empty() const { return offset_ == bytes_.size(); }

  std::uint64_t ReadVarint() {
    std::uint64_t value = 0;
    for (int shift = 0; shift < 64; shift += 7) {
      if (offset_ >= bytes_.size()) {
        throw std::runtime_error("truncated ONNX protobuf varint");
      }
      const auto byte = bytes_[offset_++];
      value |= static_cast<std::uint64_t>(byte & 0x7fU) << shift;
      if ((byte & 0x80U) == 0) {
        return value;
      }
    }
    throw std::runtime_error("oversized ONNX protobuf varint");
  }

  std::pair<std::uint32_t, std::uint32_t> ReadKey() {
    const auto key = ReadVarint();
    const auto field = static_cast<std::uint32_t>(key >> 3U);
    const auto wire = static_cast<std::uint32_t>(key & 0x07U);
    if (field == 0) {
      throw std::runtime_error("invalid zero ONNX protobuf field");
    }
    return {field, wire};
  }

  std::span<const std::uint8_t> ReadLengthDelimited() {
    const auto length = ReadVarint();
    if (length > bytes_.size() - offset_) {
      throw std::runtime_error(
          "truncated ONNX protobuf length-delimited field");
    }
    const auto result =
        bytes_.subspan(offset_, static_cast<std::size_t>(length));
    offset_ += static_cast<std::size_t>(length);
    return result;
  }

  void Skip(std::uint32_t wire) {
    switch (wire) {
      case 0:
        static_cast<void>(ReadVarint());
        return;
      case 1:
        SkipFixed(8);
        return;
      case 2:
        static_cast<void>(ReadLengthDelimited());
        return;
      case 5:
        SkipFixed(4);
        return;
      default:
        throw std::runtime_error(
            "unsupported ONNX protobuf wire type");
    }
  }

 private:
  void SkipFixed(std::size_t length) {
    if (length > bytes_.size() - offset_) {
      throw std::runtime_error("truncated ONNX protobuf fixed field");
    }
    offset_ += length;
  }

  std::span<const std::uint8_t> bytes_;
  std::size_t offset_ = 0;
};

inline std::optional<std::int64_t> ParseDefaultOpset(
    std::span<const std::uint8_t> bytes) {
  ProtobufCursor cursor(bytes);
  std::string domain;
  std::optional<std::int64_t> version;
  while (!cursor.empty()) {
    const auto [field, wire] = cursor.ReadKey();
    if (field == 1 && wire == 2) {
      const auto value = cursor.ReadLengthDelimited();
      domain.assign(
          reinterpret_cast<const char*>(value.data()), value.size());
    } else if (field == 2 && wire == 0) {
      version = static_cast<std::int64_t>(cursor.ReadVarint());
    } else {
      cursor.Skip(wire);
    }
  }
  if (version.has_value() &&
      (domain.empty() || domain == "ai.onnx")) {
    return version;
  }
  return std::nullopt;
}

}  // namespace detail

inline std::vector<std::int64_t> InspectDefaultOnnxOpsets(
    std::span<const std::uint8_t> model_bytes) {
  detail::ProtobufCursor cursor(model_bytes);
  std::vector<std::int64_t> result;
  while (!cursor.empty()) {
    const auto [field, wire] = cursor.ReadKey();
    if (field == 8 && wire == 2) {
      const auto opset =
          detail::ParseDefaultOpset(cursor.ReadLengthDelimited());
      if (opset.has_value()) {
        result.push_back(*opset);
      }
    } else {
      cursor.Skip(wire);
    }
  }
  return result;
}

inline std::vector<std::int64_t> InspectDefaultOnnxOpsetsFile(
    const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) {
    throw std::runtime_error("cannot open ONNX for opset inspection: " +
                             path);
  }
  const auto end = stream.tellg();
  if (end <= 0) {
    throw std::runtime_error(
        "ONNX is empty or unreadable during opset inspection: " + path);
  }
  constexpr std::uint64_t kMaximumModelBytes = 1024ULL * 1024ULL * 1024ULL;
  const auto size = static_cast<std::uint64_t>(end);
  if (size > kMaximumModelBytes) {
    throw std::runtime_error(
        "ONNX exceeds 1 GiB inspection limit: " + path);
  }
  stream.seekg(0, std::ios::beg);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  if (!stream || stream.gcount() !=
                     static_cast<std::streamsize>(bytes.size())) {
    throw std::runtime_error(
        "failed to read complete ONNX for opset inspection: " + path);
  }
  return InspectDefaultOnnxOpsets(bytes);
}

inline std::string Sha256File(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot open file for SHA-256: " + path);
  }
  Sha256 digest;
  std::array<char, 1024 * 1024> buffer{};
  while (stream) {
    stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = stream.gcount();
    if (count > 0) {
      digest.Update(
          reinterpret_cast<const std::uint8_t*>(buffer.data()),
          static_cast<std::size_t>(count));
    }
  }
  if (!stream.eof()) {
    throw std::runtime_error("failed while hashing file: " + path);
  }
  return digest.FinalHex();
}

}  // namespace gear_sonic::true23
