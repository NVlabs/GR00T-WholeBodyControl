#include <pybind11/pybind11.h>
#include <pybind11/chrono.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <thread>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <nlohmann/json.hpp>
#include "PXREARobotSDK.h"


using json = nlohmann::json;

constexpr const char* kPositiveJointTimestampContract =
    "pico_local_pose_timestamp_positive_v1";
constexpr const char* kUnavailableJointTimestampContract =
    "pico_local_pose_timestamp_unavailable_zero_v1";
constexpr const char* kHeadOnlyJointTimestampContract =
    "pico_local_pose_head_only_timestamp_v1";
constexpr const char* kLocalPoseBodyTimestampContract =
    "latest_pico_local_pose_timestamp_v1";
constexpr const char* kHeadLocalPoseBodyTimestampContract =
    "pico_head_local_pose_timestamp_v1";
constexpr const char* kPacketHealthBodyTimestampContract =
    "same_packet_health_timestamp_v1";

std::array<double, 7> LeftControllerPose;
std::array<double, 7> RightControllerPose;
std::array<double, 7> HeadsetPose;
int64_t LeftControllerTimeStampNs = 0;
int64_t RightControllerTimeStampNs = 0;

struct ControllerSideTrackingHealthState {
    bool deviceValid = false;
    bool isTrackedAvailable = false;
    bool isTracked = false;
    bool trackingStateAvailable = false;
    int trackingState = 0;
    bool valid = false;
};

struct ControllerTrackingHealthState {
    bool available = false;
    bool valid = false;
    int schemaVersion = 0;
    int64_t sampleSequence = 0;
    int64_t timestampNs = 0;
    std::string clientBuild;
    ControllerSideTrackingHealthState left;
    ControllerSideTrackingHealthState right;
};

ControllerTrackingHealthState ControllerTrackingHealth;
bool GrootTeleopSafetyProtocolSeen = false;

std::array<std::array<double, 7>, 26> LeftHandTrackingState;
double LeftHandScale = 1.0;
int LeftHandIsActive = 0;
std::array<std::array<double, 7>, 26> RightHandTrackingState;
double RightHandScale = 1.0;
int RightHandIsActive = 0;

// Whole body motion data - 24 joints for body tracking
std::array<std::array<double, 7>, 24> BodyJointsPose;  // Position and rotation for each joint
std::array<std::array<double, 6>, 24> BodyJointsVelocity;  // Velocity and angular velocity for each joint
std::array<std::array<double, 6>, 24> BodyJointsAcceleration;  // Acceleration and angular acceleration for each joint
std::array<int64_t, 24> BodyJointsTimestamp;  // IMU timestamp for each joint
int64_t BodyTimeStampNs = 0;  // Body data timestamp
std::string BodyTimestampContract;
std::string BodyJointTimestampContract;
bool BodyDataAvailable = false;  // Flag to indicate if body data is available
int64_t BodySampleSequence = 0;
bool BodySampleSequenceAvailable = false;

struct BodyTrackingHealthState {
    bool available = false;
    bool valid = false;
    int schemaVersion = 0;
    int64_t sampleSequence = 0;
    int64_t timestampNs = 0;
    std::string clientBuild;
    int calibrationResult = 0;
    bool calibrated = false;
    int trackingMode = 0;
    int connectStateResult = 0;
    int trackerCount = 0;
    int uniqueTrackerCount = 0;
    int bodyStateResult = 0;
    bool isTracking = false;
    int trackingStateCode = 0;
    int bodyStateCode = 0;
    int bodyErrorCode = 0;
    int connectedBandCount = 0;
    int bodyDataResult = 0;
    int bodyRoleCount = 0;
};

BodyTrackingHealthState BodyTrackingHealth;

std::array<std::array<double, 7>, 3> MotionTrackerPose;  // Position and rotation for each joint
std::array<std::array<double, 6>, 3> MotionTrackerVelocity;  // Velocity and angular velocity for each joint
std::array<std::array<double, 6>, 3> MotionTrackerAcceleration;  // Acceleration and angular acceleration for each joint
std::array<std::string, 3> MotionTrackerSerialNumbers;  // Serial numbers of the motion trackers
int64_t MotionTimeStampNs = 0;  // Motion data timestamp
int NumMotionDataAvailable = 0;  // number of motion trackers


bool LeftMenuButton;
double LeftTrigger;
double LeftGrip;
std::array<double, 2> LeftAxis{0.0, 0.0};
bool LeftAxisClick;
bool LeftPrimaryButton;
bool LeftSecondaryButton;

bool RightMenuButton;
double RightTrigger;
double RightGrip;
std::array<double, 2> RightAxis{0.0, 0.0};
bool RightAxisClick;
bool RightPrimaryButton;
bool RightSecondaryButton;

int64_t TimeStampNs;

std::mutex controllerMutex;
std::mutex headsetPoseMutex;
std::mutex timestampMutex;
std::mutex leftHandMutex;
std::mutex rightHandMutex;
std::mutex bodyMutex;  // Mutex for body tracking data
std::mutex motionMutex;



int64_t jsonExactInt64(const json& value, const char* fieldName) {
    if (value.is_number_unsigned()) {
        const auto unsignedValue = value.get<uint64_t>();
        if (unsignedValue > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument(std::string(fieldName) + " exceeds int64 range");
        }
        return static_cast<int64_t>(unsignedValue);
    }
    if (!value.is_number_integer()) {
        throw std::invalid_argument(std::string(fieldName) + " must be an exact integer");
    }
    return value.get<int64_t>();
}

int jsonBinaryFlag(const json& value, const char* fieldName) {
    const int64_t decoded = jsonExactInt64(value, fieldName);
    if (decoded != 0 && decoded != 1) {
        throw std::invalid_argument(std::string(fieldName) + " must be 0 or 1");
    }
    return static_cast<int>(decoded);
}

int jsonBoundedInt(
    const json& value,
    const char* fieldName,
    int minimum,
    int maximum) {
    const int64_t decoded = jsonExactInt64(value, fieldName);
    if (decoded < minimum || decoded > maximum) {
        throw std::invalid_argument(std::string(fieldName) + " is out of range");
    }
    return static_cast<int>(decoded);
}

double jsonFiniteDouble(
    const json& value,
    const char* fieldName,
    double minimum,
    double maximum) {
    if (!value.is_number()) {
        throw std::invalid_argument(std::string(fieldName) + " must be numeric");
    }
    const double decoded = value.get<double>();
    if (!std::isfinite(decoded) || decoded < minimum || decoded > maximum) {
        throw std::invalid_argument(std::string(fieldName) + " is out of range");
    }
    return decoded;
}

template <std::size_t N>
std::array<double, N> stringToFiniteArray(const std::string& input) {
    if (std::count(input.begin(), input.end(), ',') != static_cast<int>(N - 1)) {
        throw std::invalid_argument("numeric array has the wrong number of fields");
    }

    std::array<double, N> result{};
    std::stringstream ss(input);
    std::string value;
    std::size_t i = 0;
    while (std::getline(ss, value, ',')) {
        if (i >= N || value.empty()) {
            throw std::invalid_argument("numeric array contains an empty or extra field");
        }
        std::size_t parsed = 0;
        const double number = std::stod(value, &parsed);
        while (parsed < value.size() &&
               std::isspace(static_cast<unsigned char>(value[parsed]))) {
            ++parsed;
        }
        if (parsed != value.size() || !std::isfinite(number)) {
            throw std::invalid_argument("numeric array contains a malformed or non-finite field");
        }
        result[i++] = number;
    }
    if (i != N) {
        throw std::invalid_argument("numeric array has the wrong number of fields");
    }
    return result;
}

std::array<double, 7> stringToPoseArray(const std::string& poseStr) {
    auto result = stringToFiniteArray<7>(poseStr);
    const double quaternionNorm = std::sqrt(
        result[3] * result[3] + result[4] * result[4] +
        result[5] * result[5] + result[6] * result[6]);
    if (quaternionNorm < 0.5 || quaternionNorm > 1.5) {
        throw std::invalid_argument("pose contains an invalid quaternion");
    }
    return result;
}

std::array<double, 6> stringToVelocityArray(const std::string& velocityStr) {
    return stringToFiniteArray<6>(velocityStr);
}

std::string trimCopy(const std::string& input) {
    const auto first = std::find_if_not(
        input.begin(), input.end(),
        [](unsigned char character) { return std::isspace(character); });
    const auto last = std::find_if_not(
        input.rbegin(), input.rend(),
        [](unsigned char character) { return std::isspace(character); }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

struct ControllerFrame {
    std::array<double, 7> pose{};
    double trigger = 0.0;
    double grip = 0.0;
    bool menuButton = false;
    std::array<double, 2> axis{};
    bool axisClick = false;
    bool primaryButton = false;
    bool secondaryButton = false;
};

struct HandFrame {
    std::array<std::array<double, 7>, 26> joints{};
    double scale = 1.0;
    int isActive = 0;
};

struct BodyFrame {
    std::array<std::array<double, 7>, 24> poses{};
    std::array<std::array<double, 6>, 24> velocities{};
    std::array<std::array<double, 6>, 24> accelerations{};
    std::array<int64_t, 24> timestamps{};
    int64_t timestampNs = 0;
    std::string timestampContract;
    std::string jointTimestampContract;
    int64_t sampleSequence = 0;
    bool hasSampleSequence = false;
    bool hasDeclaredLength = false;
};

struct MotionFrame {
    std::array<std::array<double, 7>, 3> poses{};
    std::array<std::array<double, 6>, 3> velocities{};
    std::array<std::array<double, 6>, 3> accelerations{};
    std::array<std::string, 3> serialNumbers{};
    int64_t timestampNs = 0;
    int count = 0;
};

struct GrootTeleopSafetyFrame {
    std::string clientBuild;
};

struct ParsedTrackingFrame {
    std::optional<std::pair<ControllerFrame, ControllerFrame>> controllers;
    std::optional<int64_t> controllerSampleSequence;
    std::optional<ControllerTrackingHealthState> controllerHealth;
    std::optional<std::array<double, 7>> headsetPose;
    std::optional<HandFrame> leftHand;
    std::optional<HandFrame> rightHand;
    std::optional<BodyFrame> body;
    std::optional<BodyTrackingHealthState> bodyHealth;
    std::optional<GrootTeleopSafetyFrame> safetyProtocol;
    std::optional<MotionFrame> motion;
    std::optional<int64_t> timestampNs;
};

ControllerFrame parseControllerFrame(const json& controller) {
    ControllerFrame result;
    result.pose = stringToPoseArray(controller.at("pose").get<std::string>());
    result.trigger =
        jsonFiniteDouble(controller.at("trigger"), "controller trigger", 0.0, 1.0);
    result.grip =
        jsonFiniteDouble(controller.at("grip"), "controller grip", 0.0, 1.0);
    result.menuButton = controller.at("menuButton").get<bool>();
    result.axis[0] =
        jsonFiniteDouble(controller.at("axisX"), "controller axisX", -1.0, 1.0);
    result.axis[1] =
        jsonFiniteDouble(controller.at("axisY"), "controller axisY", -1.0, 1.0);
    result.axisClick = controller.at("axisClick").get<bool>();
    result.primaryButton = controller.at("primaryButton").get<bool>();
    result.secondaryButton = controller.at("secondaryButton").get<bool>();
    return result;
}

void writeControllerFramesUnlocked(
    const ControllerFrame& left,
    const ControllerFrame& right,
    int64_t timestampNs) {
    LeftControllerPose = left.pose;
    LeftTrigger = left.trigger;
    LeftGrip = left.grip;
    LeftMenuButton = left.menuButton;
    LeftAxis = left.axis;
    LeftAxisClick = left.axisClick;
    LeftPrimaryButton = left.primaryButton;
    LeftSecondaryButton = left.secondaryButton;
    RightControllerPose = right.pose;
    RightTrigger = right.trigger;
    RightGrip = right.grip;
    RightMenuButton = right.menuButton;
    RightAxis = right.axis;
    RightAxisClick = right.axisClick;
    RightPrimaryButton = right.primaryButton;
    RightSecondaryButton = right.secondaryButton;
    LeftControllerTimeStampNs = timestampNs;
    RightControllerTimeStampNs = timestampNs;
}

void clearControllerFramesUnlocked() {
    LeftControllerPose = {};
    RightControllerPose = {};
    LeftControllerTimeStampNs = 0;
    RightControllerTimeStampNs = 0;
    LeftTrigger = 0.0;
    LeftGrip = 0.0;
    LeftMenuButton = false;
    LeftAxis = {};
    LeftAxisClick = false;
    LeftPrimaryButton = false;
    LeftSecondaryButton = false;
    RightTrigger = 0.0;
    RightGrip = 0.0;
    RightMenuButton = false;
    RightAxis = {};
    RightAxisClick = false;
    RightPrimaryButton = false;
    RightSecondaryButton = false;
}

void clearControllerHealthUnlocked() {
    ControllerTrackingHealth = {};
}

BodyFrame parseBodyFrame(const json& body) {
    BodyFrame result;
    if (!body.is_object()) {
        throw std::invalid_argument("Body section must be an object");
    }
    if (body.contains("len")) {
        const int declaredLength =
            jsonBoundedInt(body.at("len"), "body len", 0, 24);
        if (declaredLength != 24) {
            throw std::invalid_argument("body len must be exactly 24");
        }
        result.hasDeclaredLength = true;
    }
    if (body.contains("sampleSequence")) {
        result.sampleSequence =
            jsonExactInt64(body.at("sampleSequence"), "body sampleSequence");
        if (result.sampleSequence < 0) {
            throw std::invalid_argument(
                "body sampleSequence must be non-negative");
        }
        result.hasSampleSequence = true;
    }
    const bool hasBodyTimestamp = body.contains("timeStampNs");
    if (hasBodyTimestamp) {
        result.timestampNs =
            jsonExactInt64(body.at("timeStampNs"), "body timeStampNs");
        if (result.timestampNs <= 0) {
            throw std::invalid_argument("body timestamp must be positive");
        }
    }
    const auto& joints = body.at("joints");
    if (!joints.is_array() || joints.size() != 24) {
        throw std::invalid_argument("body frame must contain exactly 24 joints");
    }

    int64_t latestJointTimestampNs = 0;
    bool sawZeroJointTimestamp = false;
    bool sawPositiveJointTimestamp = false;
    for (std::size_t i = 0; i < joints.size(); ++i) {
        const auto& joint = joints.at(i);
        result.poses[i] =
            stringToPoseArray(joint.at("p").get<std::string>());
        const bool hasLinearDerivatives = joint.contains("va");
        const bool hasAngularDerivatives = joint.contains("wva");
        if (!hasLinearDerivatives || !hasAngularDerivatives) {
            throw std::invalid_argument(
                "body joint must contain both va and wva derivative fields");
        }
        // XRoboToolkit Body packets differ from Motion packets:
        //   va   = linear velocity xyz, linear acceleration xyz
        //   wva  = angular velocity xyz, angular acceleration xyz
        // Normalize both into explicit [linear xyz, angular xyz] arrays.
        const auto linearDerivatives =
            stringToVelocityArray(joint.at("va").get<std::string>());
        const auto angularDerivatives =
            stringToVelocityArray(joint.at("wva").get<std::string>());
        for (std::size_t axis = 0; axis < 3; ++axis) {
            result.velocities[i][axis] = linearDerivatives[axis];
            result.velocities[i][axis + 3] = angularDerivatives[axis];
            result.accelerations[i][axis] =
                linearDerivatives[axis + 3];
            result.accelerations[i][axis + 3] =
                angularDerivatives[axis + 3];
        }
        if (!joint.contains("t")) {
            throw std::invalid_argument(
                "body joint must contain a source timestamp");
        }
        result.timestamps[i] =
            jsonExactInt64(joint.at("t"), "body joint timestamp");
        if (result.timestamps[i] < 0) {
            throw std::invalid_argument(
                "body joint timestamp must be non-negative");
        }
        sawZeroJointTimestamp =
            sawZeroJointTimestamp || result.timestamps[i] == 0;
        sawPositiveJointTimestamp =
            sawPositiveJointTimestamp || result.timestamps[i] > 0;
        latestJointTimestampNs =
            std::max(latestJointTimestampNs, result.timestamps[i]);
    }

    const bool hasMixedJointTimestamps =
        sawZeroJointTimestamp && sawPositiveJointTimestamp;
    bool hasExactHeadOnlyTimestamp = result.timestamps[15] > 0;
    for (std::size_t i = 0; i < result.timestamps.size(); ++i) {
        if (i != 15 && result.timestamps[i] != 0) {
            hasExactHeadOnlyTimestamp = false;
        }
    }
    if (hasMixedJointTimestamps && !hasExactHeadOnlyTimestamp) {
        throw std::invalid_argument(
            "mixed body joint timestamps require exact positive HEAD index 15 only");
    }
    if (hasExactHeadOnlyTimestamp) {
        if (hasBodyTimestamp) {
            throw std::invalid_argument(
                "HEAD-only PICO joint timestamp cannot carry an independent body timestamp");
        }
        // Current PICO firmware timestamps only the canonical HEAD role. The
        // other 23 zero values remain explicit and are never synthesized.
        result.timestampNs = result.timestamps[15];
        result.timestampContract = kHeadLocalPoseBodyTimestampContract;
        result.jointTimestampContract = kHeadOnlyJointTimestampContract;
    } else if (sawZeroJointTimestamp) {
        if (hasBodyTimestamp) {
            throw std::invalid_argument(
                "zero PICO joint timestamps cannot carry an independent body timestamp");
        }
        // PICO Integration SDK 3.1.2 on current PICO OS reports an honest
        // all-zero localPose.TimeStamp for otherwise valid XR24 samples. Keep
        // those zero values intact. The same-packet hardened health timestamp
        // is bound below only as the advancing body sample timestamp.
        result.timestampNs = 0;
        result.timestampContract = kPacketHealthBodyTimestampContract;
        result.jointTimestampContract = kUnavailableJointTimestampContract;
    } else if (!hasBodyTimestamp) {
        // The official XRoboToolkit Unity Client v1.1.1 does not timestamp
        // the nested Body object. Its HEAD joint carries the advancing PICO
        // source clock while the packet-level timestamp can keep advancing
        // over a cached body pose after tracker/calibration loss.
        result.timestampNs = latestJointTimestampNs;
        if (result.timestampNs <= 0) {
            throw std::invalid_argument(
                "body frame has no positive body-specific source timestamp");
        }
        result.timestampContract = kLocalPoseBodyTimestampContract;
        result.jointTimestampContract = kPositiveJointTimestampContract;
    } else {
        result.timestampContract = kLocalPoseBodyTimestampContract;
        result.jointTimestampContract = kPositiveJointTimestampContract;
    }
    return result;
}

GrootTeleopSafetyFrame parseGrootTeleopSafety(
    const json& safetyProtocol) {
    constexpr const char* expectedName = "groot-wbc";
    constexpr const char* expectedClientBuild =
        "xrobotoolkit-pico-health-v1";
    constexpr const char* bodyHealthCapability =
        "body_tracking_health_v1";
    constexpr const char* controllerHealthCapability =
        "controller_tracking_health_v1";

    if (!safetyProtocol.is_object()) {
        throw std::invalid_argument(
            "GrootTeleopSafety section must be an object");
    }
    if (safetyProtocol.at("name").get<std::string>() != expectedName) {
        throw std::invalid_argument("GrootTeleopSafety name is unsupported");
    }
    if (jsonExactInt64(
            safetyProtocol.at("schemaVersion"),
            "GrootTeleopSafety schemaVersion") != 1) {
        throw std::invalid_argument(
            "GrootTeleopSafety schemaVersion is unsupported");
    }

    GrootTeleopSafetyFrame result;
    result.clientBuild =
        safetyProtocol.at("clientBuild").get<std::string>();
    if (result.clientBuild != expectedClientBuild) {
        throw std::invalid_argument(
            "GrootTeleopSafety clientBuild is unsupported");
    }

    const auto& capabilities = safetyProtocol.at("capabilities");
    if (!capabilities.is_array()) {
        throw std::invalid_argument(
            "GrootTeleopSafety capabilities must be an array");
    }
    bool bodyHealthCapabilityFound = false;
    bool controllerHealthCapabilityFound = false;
    for (const auto& capability : capabilities) {
        if (!capability.is_string()) {
            throw std::invalid_argument(
                "GrootTeleopSafety capability must be a string");
        }
        const std::string decoded = capability.get<std::string>();
        if (decoded == bodyHealthCapability) {
            bodyHealthCapabilityFound = true;
        } else if (decoded == controllerHealthCapability) {
            controllerHealthCapabilityFound = true;
        }
    }
    if (!bodyHealthCapabilityFound ||
        !controllerHealthCapabilityFound) {
        throw std::invalid_argument(
            "GrootTeleopSafety lacks required tracking-health capabilities");
    }
    return result;
}

ControllerSideTrackingHealthState parseControllerSideTrackingHealth(
    const json& side,
    const char* sideName) {
    if (!side.is_object()) {
        throw std::invalid_argument(
            std::string("ControllerTrackingStatus ") + sideName +
            " section must be an object");
    }

    ControllerSideTrackingHealthState result;
    result.deviceValid = side.at("deviceValid").get<bool>();
    result.isTrackedAvailable =
        side.at("isTrackedAvailable").get<bool>();
    result.isTracked = side.at("isTracked").get<bool>();
    result.trackingStateAvailable =
        side.at("trackingStateAvailable").get<bool>();
    const std::string trackingStateField =
        std::string("ControllerTrackingStatus ") + sideName +
        " trackingState";
    result.trackingState =
        jsonBoundedInt(
            side.at("trackingState"),
            trackingStateField.c_str(),
            0,
            63);
    result.valid = side.at("valid").get<bool>();

    const bool independentlyValid =
        result.deviceValid &&
        result.isTrackedAvailable &&
        result.isTracked &&
        result.trackingStateAvailable &&
        (result.trackingState & 3) == 3;
    if (result.valid != independentlyValid) {
        throw std::invalid_argument(
            std::string("ControllerTrackingStatus ") + sideName +
            " valid flag contradicts tracking fields");
    }
    return result;
}

ControllerTrackingHealthState parseControllerTrackingHealth(
    const json& health,
    const GrootTeleopSafetyFrame& safetyProtocol) {
    constexpr const char* expectedClientBuild =
        "xrobotoolkit-pico-health-v1";

    if (!health.is_object()) {
        throw std::invalid_argument(
            "ControllerTrackingStatus section must be an object");
    }

    ControllerTrackingHealthState result;
    result.available = true;
    result.schemaVersion =
        jsonBoundedInt(
            health.at("schemaVersion"),
            "ControllerTrackingStatus schemaVersion",
            1,
            1);
    result.clientBuild = health.at("clientBuild").get<std::string>();
    if (result.clientBuild != expectedClientBuild ||
        result.clientBuild != safetyProtocol.clientBuild) {
        throw std::invalid_argument(
            "ControllerTrackingStatus clientBuild does not match safety protocol");
    }
    result.sampleSequence =
        jsonExactInt64(
            health.at("sampleSequence"),
            "ControllerTrackingStatus sampleSequence");
    if (result.sampleSequence < 0) {
        throw std::invalid_argument(
            "ControllerTrackingStatus sampleSequence must be non-negative");
    }
    result.timestampNs =
        jsonExactInt64(
            health.at("timeStampNs"),
            "ControllerTrackingStatus timeStampNs");
    if (result.timestampNs <= 0) {
        throw std::invalid_argument(
            "ControllerTrackingStatus timeStampNs must be positive");
    }
    result.left =
        parseControllerSideTrackingHealth(health.at("left"), "left");
    result.right =
        parseControllerSideTrackingHealth(health.at("right"), "right");
    result.valid = health.at("valid").get<bool>();

    const bool independentlyValid =
        result.left.valid && result.right.valid;
    if (result.valid != independentlyValid) {
        throw std::invalid_argument(
            "ControllerTrackingStatus valid flag contradicts side health");
    }
    return result;
}

BodyTrackingHealthState parseBodyTrackingHealth(
    const json& health,
    const GrootTeleopSafetyFrame& safetyProtocol) {
    constexpr const char* expectedClientBuild =
        "xrobotoolkit-pico-health-v1";
    constexpr int minimumApiResult = std::numeric_limits<int>::min();
    constexpr int maximumApiResult = std::numeric_limits<int>::max();

    if (!health.is_object()) {
        throw std::invalid_argument(
            "BodyTrackingStatus section must be an object");
    }

    BodyTrackingHealthState result;
    result.available = true;
    result.schemaVersion =
        jsonBoundedInt(
            health.at("schemaVersion"),
            "BodyTrackingStatus schemaVersion",
            1,
            1);
    result.clientBuild = health.at("clientBuild").get<std::string>();
    if (result.clientBuild != expectedClientBuild ||
        result.clientBuild != safetyProtocol.clientBuild) {
        throw std::invalid_argument(
            "BodyTrackingStatus clientBuild does not match safety protocol");
    }
    result.calibrationResult =
        jsonBoundedInt(
            health.at("calibrationResult"),
            "BodyTrackingStatus calibrationResult",
            minimumApiResult,
            maximumApiResult);
    result.calibrated = health.at("calibrated").get<bool>();
    result.trackingMode =
        jsonBoundedInt(
            health.at("trackingMode"),
            "BodyTrackingStatus trackingMode",
            minimumApiResult,
            maximumApiResult);
    result.connectStateResult =
        jsonBoundedInt(
            health.at("connectStateResult"),
            "BodyTrackingStatus connectStateResult",
            minimumApiResult,
            maximumApiResult);
    result.trackerCount =
        jsonBoundedInt(
            health.at("trackerCount"),
            "BodyTrackingStatus trackerCount",
            0,
            6);
    result.uniqueTrackerCount =
        jsonBoundedInt(
            health.at("uniqueTrackerCount"),
            "BodyTrackingStatus uniqueTrackerCount",
            0,
            6);
    if (result.uniqueTrackerCount > result.trackerCount) {
        throw std::invalid_argument(
            "BodyTrackingStatus uniqueTrackerCount exceeds trackerCount");
    }
    result.bodyStateResult =
        jsonBoundedInt(
            health.at("bodyStateResult"),
            "BodyTrackingStatus bodyStateResult",
            minimumApiResult,
            maximumApiResult);
    result.isTracking = health.at("isTracking").get<bool>();
    result.trackingStateCode =
        jsonBoundedInt(
            health.at("trackingStateCode"),
            "BodyTrackingStatus trackingStateCode",
            minimumApiResult,
            maximumApiResult);
    result.bodyStateCode =
        jsonBoundedInt(
            health.at("bodyStateCode"),
            "BodyTrackingStatus bodyStateCode",
            minimumApiResult,
            maximumApiResult);
    result.bodyErrorCode =
        jsonBoundedInt(
            health.at("bodyErrorCode"),
            "BodyTrackingStatus bodyErrorCode",
            minimumApiResult,
            maximumApiResult);
    result.connectedBandCount =
        jsonBoundedInt(
            health.at("connectedBandCount"),
            "BodyTrackingStatus connectedBandCount",
            0,
            12);
    result.bodyDataResult =
        jsonBoundedInt(
            health.at("bodyDataResult"),
            "BodyTrackingStatus bodyDataResult",
            minimumApiResult,
            maximumApiResult);
    result.bodyRoleCount =
        jsonBoundedInt(
            health.at("bodyRoleCount"),
            "BodyTrackingStatus bodyRoleCount",
            0,
            24);
    result.valid = health.at("valid").get<bool>();
    result.sampleSequence =
        jsonExactInt64(
            health.at("sampleSequence"),
            "BodyTrackingStatus sampleSequence");
    if (result.sampleSequence < 0) {
        throw std::invalid_argument(
            "BodyTrackingStatus sampleSequence must be non-negative");
    }
    result.timestampNs =
        jsonExactInt64(
            health.at("timeStampNs"),
            "BodyTrackingStatus timeStampNs");
    if (result.timestampNs <= 0) {
        throw std::invalid_argument(
            "BodyTrackingStatus timeStampNs must be positive");
    }

    const bool independentlyValid =
        result.calibrationResult == 0 &&
        result.calibrated &&
        result.trackingMode == 0 &&
        result.connectStateResult == 0 &&
        result.trackerCount >= 2 &&
        result.uniqueTrackerCount >= 2 &&
        result.bodyStateResult == 0 &&
        result.isTracking &&
        result.trackingStateCode == 0 &&
        result.bodyStateCode == 1 &&
        result.connectedBandCount >= 2 &&
        result.bodyDataResult == 0 &&
        result.bodyRoleCount == 24;
    if (result.valid && !independentlyValid) {
        throw std::invalid_argument(
            "BodyTrackingStatus valid flag contradicts health fields");
    }
    return result;
}

MotionFrame parseMotionFrame(const json& motion, int64_t frameTimestampNs) {
    MotionFrame result;
    // The official v1.1.1 client likewise places the source timestamp on the
    // complete packet rather than inside Motion.
    result.timestampNs = motion.contains("timeStampNs")
        ? jsonExactInt64(motion.at("timeStampNs"), "motion timeStampNs")
        : frameTimestampNs;
    if (result.timestampNs <= 0) {
        throw std::invalid_argument("motion timestamp must be positive");
    }
    const auto& joints = motion.at("joints");
    if (!joints.is_array()) {
        throw std::invalid_argument("motion joints must be an array");
    }
    const int64_t activeTrackerCount = motion.contains("len")
        ? jsonExactInt64(motion.at("len"), "motion len")
        : static_cast<int64_t>(joints.size());
    if (activeTrackerCount < 2 || activeTrackerCount > 3 ||
        joints.size() < static_cast<std::size_t>(activeTrackerCount)) {
        throw std::invalid_argument("motion frame must contain two or three trackers");
    }

    std::set<std::string> uniqueSerialNumbers;
    for (std::size_t i = 0;
         i < static_cast<std::size_t>(activeTrackerCount);
         ++i) {
        const auto& joint = joints.at(i);
        result.poses[i] =
            stringToPoseArray(joint.at("p").get<std::string>());
        result.serialNumbers[i] =
            trimCopy(joint.at("sn").get<std::string>());
        if (result.serialNumbers[i].empty() ||
            !uniqueSerialNumbers.insert(result.serialNumbers[i]).second) {
            throw std::invalid_argument("motion tracker serial numbers must be unique and non-empty");
        }
        if (joint.contains("va")) {
            result.velocities[i] =
                stringToVelocityArray(joint.at("va").get<std::string>());
        }
        if (joint.contains("wva")) {
            result.accelerations[i] =
                stringToVelocityArray(joint.at("wva").get<std::string>());
        }
    }
    result.count = static_cast<int>(activeTrackerCount);
    return result;
}

HandFrame parseHandFrame(const json& hand, const char* side) {
    HandFrame result;
    result.scale =
        jsonFiniteDouble(hand.at("scale"), "hand scale", 0.0, 10.0);
    result.isActive =
        jsonBinaryFlag(hand.at("isActive"), "hand isActive");
    const auto& joints = hand.at("HandJointLocations");
    if (!joints.is_array() || joints.size() != result.joints.size()) {
        throw std::invalid_argument(
            std::string(side) + " hand must contain exactly 26 joints");
    }
    for (std::size_t i = 0; i < result.joints.size(); ++i) {
        result.joints[i] =
            stringToPoseArray(joints.at(i).at("p").get<std::string>());
    }
    return result;
}

void clearBodyUnlocked() {
    BodyJointsPose = {};
    BodyJointsVelocity = {};
    BodyJointsAcceleration = {};
    BodyJointsTimestamp = {};
    BodyTimeStampNs = 0;
    BodyTimestampContract.clear();
    BodyJointTimestampContract.clear();
    BodyDataAvailable = false;
    BodySampleSequence = 0;
    BodySampleSequenceAvailable = false;
}

void clearBodyHealthUnlocked() {
    BodyTrackingHealth = {};
}

void writeBodyFrameUnlocked(const BodyFrame& body) {
    BodyJointsPose = body.poses;
    BodyJointsVelocity = body.velocities;
    BodyJointsAcceleration = body.accelerations;
    BodyJointsTimestamp = body.timestamps;
    BodyTimeStampNs = body.timestampNs;
    BodyTimestampContract = body.timestampContract;
    BodyJointTimestampContract = body.jointTimestampContract;
    BodyDataAvailable = true;
    BodySampleSequence = body.sampleSequence;
    BodySampleSequenceAvailable = body.hasSampleSequence;
}

void clearMotionUnlocked() {
    MotionTrackerPose = {};
    MotionTrackerVelocity = {};
    MotionTrackerAcceleration = {};
    MotionTrackerSerialNumbers = {};
    MotionTimeStampNs = 0;
    NumMotionDataAvailable = 0;
}

void commitTrackingFrame(const ParsedTrackingFrame& frame) {
    std::scoped_lock lock(
        controllerMutex,
        headsetPoseMutex,
        timestampMutex,
        leftHandMutex,
        rightHandMutex,
        bodyMutex,
        motionMutex);
    if (GrootTeleopSafetyProtocolSeen &&
        !frame.safetyProtocol.has_value()) {
        throw std::invalid_argument(
            "GrootTeleopSafety protocol disappeared during device session");
    }
    if (frame.safetyProtocol.has_value()) {
        GrootTeleopSafetyProtocolSeen = true;
    }
    if (frame.safetyProtocol.has_value()) {
        if (frame.controllerHealth.has_value()) {
            ControllerTrackingHealth = *frame.controllerHealth;
            if (frame.controllerHealth->valid) {
                writeControllerFramesUnlocked(
                    frame.controllers->first,
                    frame.controllers->second,
                    *frame.timestampNs);
            } else {
                clearControllerFramesUnlocked();
            }
        } else {
            clearControllerHealthUnlocked();
            clearControllerFramesUnlocked();
        }
    } else if (frame.controllers.has_value()) {
        // Stock XRoboToolkit controller frames remain visible for diagnostics
        // but never advertise hardened controller-health support.
        writeControllerFramesUnlocked(
            frame.controllers->first,
            frame.controllers->second,
            *frame.timestampNs);
    }
    if (frame.headsetPose.has_value()) {
        HeadsetPose = *frame.headsetPose;
    }
    if (frame.timestampNs.has_value()) {
        TimeStampNs = *frame.timestampNs;
    }
    if (frame.leftHand.has_value()) {
        LeftHandTrackingState = frame.leftHand->joints;
        LeftHandScale = frame.leftHand->scale;
        LeftHandIsActive = frame.leftHand->isActive;
    }
    if (frame.rightHand.has_value()) {
        RightHandTrackingState = frame.rightHand->joints;
        RightHandScale = frame.rightHand->scale;
        RightHandIsActive = frame.rightHand->isActive;
    }
    if (frame.safetyProtocol.has_value()) {
        if (frame.bodyHealth.has_value()) {
            // PICO BodyTracking and raw MotionTracking are mutually exclusive.
            // Never retain raw tracker state from an earlier Motion-mode frame
            // beside a current hardened Body-mode sample.
            clearMotionUnlocked();
            BodyTrackingHealth = *frame.bodyHealth;
            if (frame.bodyHealth->valid) {
                writeBodyFrameUnlocked(*frame.body);
            } else {
                clearBodyUnlocked();
            }
        } else {
            clearBodyHealthUnlocked();
            clearBodyUnlocked();
        }
    } else if (frame.body.has_value()) {
        // Stock XRoboToolkit frames remain decodable for diagnostics. They
        // never advertise hardened health support, so downstream safety code
        // can reject them without losing visibility into the raw body stream.
        writeBodyFrameUnlocked(*frame.body);
    }
    if (frame.motion.has_value()) {
        MotionTrackerPose = frame.motion->poses;
        MotionTrackerVelocity = frame.motion->velocities;
        MotionTrackerAcceleration = frame.motion->accelerations;
        MotionTrackerSerialNumbers = frame.motion->serialNumbers;
        MotionTimeStampNs = frame.motion->timestampNs;
        NumMotionDataAvailable = frame.motion->count;
    }
}

void invalidateAllTracking(bool resetSafetyProtocol) {
    std::scoped_lock lock(
        controllerMutex,
        headsetPoseMutex,
        timestampMutex,
        leftHandMutex,
        rightHandMutex,
        bodyMutex,
        motionMutex);
    clearControllerFramesUnlocked();
    clearControllerHealthUnlocked();
    HeadsetPose = {};
    TimeStampNs = 0;
    LeftHandTrackingState = {};
    LeftHandScale = 1.0;
    LeftHandIsActive = 0;
    RightHandTrackingState = {};
    RightHandScale = 1.0;
    RightHandIsActive = 0;
    clearBodyUnlocked();
    clearBodyHealthUnlocked();
    if (resetSafetyProtocol) {
        GrootTeleopSafetyProtocolSeen = false;
    }
    clearMotionUnlocked();
}

void OnPXREAClientCallback(void* context, PXREAClientCallbackType type, int status, void* userData)
{
    switch (type)
    {
    case PXREAServerConnect:
        std::cout << "server connect\n" << std::endl;
        break;
    case PXREAServerDisconnect:
        std::cout << "server disconnect\n" << std::endl;
        invalidateAllTracking(true);
        break;
    case PXREADeviceFind:
        std::cout << "device found\n" << (const char*)userData << std::endl;
        break;
    case PXREADeviceMissing:
        std::cout << "device missing\n" << (const char*)userData << std::endl;
        invalidateAllTracking(true);
        break;
    case PXREADeviceConnect:
        std::cout << "device connect\n" << (const char*)userData << status << std::endl;
        break;
    case PXREADeviceStateJson: {
        try {
            if (userData == nullptr) {
                throw std::invalid_argument(
                    "device-state callback has no payload");
            }
            auto& dsj = *static_cast<PXREADevStateJson*>(userData);
            json data = json::parse(dsj.stateJson);
            if (!data.is_object() || !data.contains("value")) {
                throw std::invalid_argument(
                    "device-state payload requires value");
            }
            auto value = json::parse(data.at("value").get<std::string>());
            if (!value.is_object()) {
                throw std::invalid_argument("tracking value must be an object");
            }

            ParsedTrackingFrame frame;
            if (value.contains("GrootTeleopSafety")) {
                frame.safetyProtocol =
                    parseGrootTeleopSafety(
                        value.at("GrootTeleopSafety"));
            }
            if (value.contains("BodyTrackingStatus")) {
                if (!frame.safetyProtocol.has_value()) {
                    throw std::invalid_argument(
                        "BodyTrackingStatus requires GrootTeleopSafety");
                }
                frame.bodyHealth =
                    parseBodyTrackingHealth(
                        value.at("BodyTrackingStatus"),
                        *frame.safetyProtocol);
            }
            if (value.contains("ControllerTrackingStatus")) {
                if (!frame.safetyProtocol.has_value()) {
                    throw std::invalid_argument(
                        "ControllerTrackingStatus requires GrootTeleopSafety");
                }
                frame.controllerHealth =
                    parseControllerTrackingHealth(
                        value.at("ControllerTrackingStatus"),
                        *frame.safetyProtocol);
            }
            if (value.contains("timeStampNs")) {
                const int64_t timestampNs =
                    jsonExactInt64(
                        value.at("timeStampNs"), "frame timeStampNs");
                if (timestampNs <= 0) {
                    throw std::invalid_argument(
                        "frame timestamp must be positive");
                }
                frame.timestampNs = timestampNs;
            }
            if (value.contains("Controller")) {
                const auto& controllers = value.at("Controller");
                if (!frame.timestampNs.has_value() ||
                    !controllers.is_object() ||
                    !controllers.contains("left") || !controllers.contains("right")) {
                    throw std::invalid_argument(
                        "controller frame requires both sides and a positive timestamp");
                }
                if (controllers.contains("sampleSequence")) {
                    const int64_t sampleSequence =
                        jsonExactInt64(
                            controllers.at("sampleSequence"),
                            "Controller sampleSequence");
                    if (sampleSequence <= 0) {
                        throw std::invalid_argument(
                            "Controller sampleSequence must be positive");
                    }
                    frame.controllerSampleSequence = sampleSequence;
                }
                const ControllerFrame left =
                    parseControllerFrame(controllers.at("left"));
                const ControllerFrame right =
                    parseControllerFrame(controllers.at("right"));
                frame.controllers = std::make_pair(left, right);
            }
            if (frame.safetyProtocol.has_value() &&
                frame.controllerHealth.has_value()) {
                if (frame.controllerHealth->valid &&
                    frame.controllerHealth->sampleSequence <= 0) {
                    throw std::invalid_argument(
                        "valid ControllerTrackingStatus requires a positive sampleSequence");
                }
                if (frame.controllers.has_value()) {
                    if (!frame.controllerSampleSequence.has_value()) {
                        throw std::invalid_argument(
                            "hardened Controller data requires sampleSequence");
                    }
                    if (frame.controllerHealth->sampleSequence <= 0 ||
                        frame.controllerHealth->sampleSequence !=
                            *frame.controllerSampleSequence) {
                        throw std::invalid_argument(
                            "Controller sampleSequence does not match ControllerTrackingStatus");
                    }
                    if (!frame.timestampNs.has_value() ||
                        frame.controllerHealth->timestampNs !=
                            *frame.timestampNs) {
                        throw std::invalid_argument(
                            "ControllerTrackingStatus timeStampNs does not match controller frame");
                    }
                } else if (frame.controllerHealth->valid) {
                    throw std::invalid_argument(
                        "valid ControllerTrackingStatus requires Controller data");
                }
            }
            if (frame.safetyProtocol.has_value() &&
                !frame.controllerHealth.has_value() &&
                frame.controllers.has_value()) {
                throw std::invalid_argument(
                    "hardened Controller data requires ControllerTrackingStatus");
            }
            if (value.contains("Head")) {
                const auto& headset = value.at("Head");
                frame.headsetPose = stringToPoseArray(
                    headset.at("pose").get<std::string>());
            }
            if (value.contains("Hand")) {
                const auto& hands = value.at("Hand");
                if (!hands.is_object()) {
                    throw std::invalid_argument(
                        "Hand section must be an object");
                }
                if (hands.contains("leftHand")) {
                    frame.leftHand =
                        parseHandFrame(hands.at("leftHand"), "left");
                }
                if (hands.contains("rightHand")) {
                    frame.rightHand =
                        parseHandFrame(hands.at("rightHand"), "right");
                }
            }
            if (value.contains("Body")) {
                frame.body = parseBodyFrame(value.at("Body"));
            }
            if (frame.body.has_value() &&
                frame.body->jointTimestampContract ==
                    kUnavailableJointTimestampContract) {
                if (!frame.safetyProtocol.has_value() ||
                    !frame.bodyHealth.has_value() ||
                    !frame.bodyHealth->valid ||
                    !frame.timestampNs.has_value() ||
                    frame.bodyHealth->timestampNs != *frame.timestampNs) {
                    throw std::invalid_argument(
                        "zero PICO joint timestamps require valid same-packet hardened health");
                }
                frame.body->timestampNs = *frame.timestampNs;
            }
            if (frame.safetyProtocol.has_value() &&
                frame.bodyHealth.has_value() &&
                frame.bodyHealth->valid) {
                if (frame.bodyHealth->sampleSequence <= 0) {
                    throw std::invalid_argument(
                        "valid BodyTrackingStatus requires a positive sampleSequence");
                }
                if (!frame.timestampNs.has_value() ||
                    frame.bodyHealth->timestampNs != *frame.timestampNs) {
                    throw std::invalid_argument(
                        "BodyTrackingStatus timeStampNs does not match body frame");
                }
                if (!frame.body.has_value()) {
                    throw std::invalid_argument(
                        "valid BodyTrackingStatus requires Body data");
                }
                if (!frame.body->hasDeclaredLength) {
                    throw std::invalid_argument(
                        "hardened Body data requires len");
                }
                if (!frame.body->hasSampleSequence ||
                    frame.body->sampleSequence !=
                        frame.bodyHealth->sampleSequence) {
                    throw std::invalid_argument(
                        "Body sampleSequence does not match BodyTrackingStatus");
                }
            }
            if (frame.safetyProtocol.has_value() &&
                !frame.bodyHealth.has_value() &&
                frame.body.has_value()) {
                throw std::invalid_argument(
                    "hardened Body data requires BodyTrackingStatus");
            }
            if (value.contains("Motion")) {
                if (!frame.timestampNs.has_value()) {
                    throw std::invalid_argument(
                        "motion frame requires a positive frame timestamp");
                }
                frame.motion =
                    parseMotionFrame(value.at("Motion"), *frame.timestampNs);
            }
            commitTrackingFrame(frame);
        } catch (const std::exception& e) {
            invalidateAllTracking(false);
            std::cerr << "Tracking frame rejected: " << e.what() << std::endl;
        } catch (...) {
            invalidateAllTracking(false);
            std::cerr << "Tracking frame rejected: unknown parsing error" << std::endl;
        }
        break;
    }
    }
}

void init() {
    invalidateAllTracking(true);
    if (PXREAInit(NULL, OnPXREAClientCallback, PXREAFullMask) != 0) {
        throw std::runtime_error("PXREAInit failed");
    }
}

void deinit() {
    PXREADeinit();
    invalidateAllTracking(true);
}

std::array<double, 7> getLeftControllerPose() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return LeftControllerPose;
}

std::array<double, 7> getRightControllerPose() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return RightControllerPose;
}

int64_t getLeftControllerTimeStampNs() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return LeftControllerTimeStampNs;
}

int64_t getRightControllerTimeStampNs() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return RightControllerTimeStampNs;
}

pybind11::dict getControllerSnapshot() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    pybind11::dict snapshot;
    snapshot["left_pose"] = LeftControllerPose;
    snapshot["right_pose"] = RightControllerPose;
    snapshot["left_trigger_value"] = LeftTrigger;
    snapshot["right_trigger_value"] = RightTrigger;
    snapshot["left_squeeze_value"] = LeftGrip;
    snapshot["right_squeeze_value"] = RightGrip;
    snapshot["left_thumbstick"] = LeftAxis;
    snapshot["right_thumbstick"] = RightAxis;
    snapshot["left_thumbstick_click"] = LeftAxisClick;
    snapshot["right_thumbstick_click"] = RightAxisClick;
    snapshot["left_primary_click"] = LeftPrimaryButton;
    snapshot["left_secondary_click"] = LeftSecondaryButton;
    snapshot["right_primary_click"] = RightPrimaryButton;
    snapshot["right_secondary_click"] = RightSecondaryButton;
    snapshot["left_menu_button"] = LeftMenuButton;
    snapshot["right_menu_button"] = RightMenuButton;
    snapshot["left_timestamp_ns"] = LeftControllerTimeStampNs;
    snapshot["right_timestamp_ns"] = RightControllerTimeStampNs;
    snapshot["health_supported"] = GrootTeleopSafetyProtocolSeen;
    snapshot["health_available"] = ControllerTrackingHealth.available;
    snapshot["health_valid"] = ControllerTrackingHealth.valid;
    snapshot["health_schema_version"] =
        ControllerTrackingHealth.schemaVersion;
    snapshot["health_sample_sequence"] =
        ControllerTrackingHealth.sampleSequence;
    snapshot["health_timestamp_ns"] =
        ControllerTrackingHealth.timestampNs;
    snapshot["health_client_build"] =
        ControllerTrackingHealth.clientBuild;
    snapshot["health_left_device_valid"] =
        ControllerTrackingHealth.left.deviceValid;
    snapshot["health_left_is_tracked_available"] =
        ControllerTrackingHealth.left.isTrackedAvailable;
    snapshot["health_left_is_tracked"] =
        ControllerTrackingHealth.left.isTracked;
    snapshot["health_left_tracking_state_available"] =
        ControllerTrackingHealth.left.trackingStateAvailable;
    snapshot["health_left_tracking_state"] =
        ControllerTrackingHealth.left.trackingState;
    snapshot["health_left_valid"] =
        ControllerTrackingHealth.left.valid;
    snapshot["health_right_device_valid"] =
        ControllerTrackingHealth.right.deviceValid;
    snapshot["health_right_is_tracked_available"] =
        ControllerTrackingHealth.right.isTrackedAvailable;
    snapshot["health_right_is_tracked"] =
        ControllerTrackingHealth.right.isTracked;
    snapshot["health_right_tracking_state_available"] =
        ControllerTrackingHealth.right.trackingStateAvailable;
    snapshot["health_right_tracking_state"] =
        ControllerTrackingHealth.right.trackingState;
    snapshot["health_right_valid"] =
        ControllerTrackingHealth.right.valid;
    return snapshot;
}

std::array<double, 7> getHeadsetPose() {
    std::lock_guard<std::mutex> lock(headsetPoseMutex);
    return HeadsetPose;
}

double getLeftTrigger() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return LeftTrigger;
}

double getLeftGrip() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return LeftGrip;
}

double getRightTrigger() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return RightTrigger;
}

double getRightGrip() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return RightGrip;
}

bool getLeftMenuButton() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return LeftMenuButton;
}

bool getRightMenuButton() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return RightMenuButton;
}

bool getLeftAxisClick() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return LeftAxisClick;
}

bool getRightAxisClick() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return RightAxisClick;
}

std::array<double, 2> getLeftAxis() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return LeftAxis;
}


std::array<double, 2> getRightAxis() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return RightAxis;
}

bool getLeftPrimaryButton() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return LeftPrimaryButton;
}

bool getRightPrimaryButton() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return RightPrimaryButton;
}

bool getLeftSecondaryButton() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return LeftSecondaryButton;
}

bool getRightSecondaryButton() {
    std::lock_guard<std::mutex> lock(controllerMutex);
    return RightSecondaryButton;
}

int64_t getTimeStampNs() {
    std::lock_guard<std::mutex> lock(timestampMutex);
    return TimeStampNs;
}

std::array<std::array<double, 7>, 26> getLeftHandTrackingState() {
    std::lock_guard<std::mutex> lock(leftHandMutex);
    return LeftHandTrackingState;
}

int getLeftHandScale() {
    std::lock_guard<std::mutex> lock(leftHandMutex);
    return LeftHandScale;
}

int getLeftHandIsActive() {
    std::lock_guard<std::mutex> lock(leftHandMutex);
    return LeftHandIsActive;
}

std::array<std::array<double, 7>, 26> getRightHandTrackingState() {
    std::lock_guard<std::mutex> lock(rightHandMutex);
    return RightHandTrackingState;
}

int getRightHandScale() {
    std::lock_guard<std::mutex> lock(rightHandMutex);
    return RightHandScale;
}

int getRightHandIsActive() {
    std::lock_guard<std::mutex> lock(rightHandMutex);
    return RightHandIsActive;
}

// Body tracking functions
bool isBodyDataAvailable() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyDataAvailable;
}

std::array<std::array<double, 7>, 24> getBodyJointsPose() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsPose;
}

pybind11::dict getBodySnapshot() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    pybind11::dict snapshot;
    snapshot["contract"] =
        "xrt_xr24_body_tracker_fused_ankles_atomic_v1";
    snapshot["source_coherence_contract"] =
        "same_packet_xr24_body_tracking_v1";
    snapshot["available"] = BodyDataAvailable;
    snapshot["timestamp_ns"] = BodyTimeStampNs;
    snapshot["body_timestamp_contract"] = BodyTimestampContract;
    snapshot["joint_timestamp_contract"] = BodyJointTimestampContract;
    snapshot["poses"] = BodyJointsPose;
    snapshot["velocities"] = BodyJointsVelocity;
    snapshot["accelerations"] = BodyJointsAcceleration;
    snapshot["derivative_layout_contract"] =
        "linear_xyz_then_angular_xyz_v1";
    snapshot["joint_timestamps_ns"] = BodyJointsTimestamp;
    if (BodySampleSequenceAvailable) {
        snapshot["sample_sequence"] = BodySampleSequence;
    } else {
        snapshot["sample_sequence"] = pybind11::none();
    }
    snapshot["health_supported"] = GrootTeleopSafetyProtocolSeen;
    snapshot["health_available"] = BodyTrackingHealth.available;
    snapshot["health_valid"] = BodyTrackingHealth.valid;
    snapshot["health_schema_version"] = BodyTrackingHealth.schemaVersion;
    snapshot["health_sample_sequence"] =
        BodyTrackingHealth.sampleSequence;
    snapshot["health_timestamp_ns"] = BodyTrackingHealth.timestampNs;
    snapshot["health_client_build"] = BodyTrackingHealth.clientBuild;
    snapshot["health_calibration_result"] =
        BodyTrackingHealth.calibrationResult;
    snapshot["health_calibrated"] = BodyTrackingHealth.calibrated;
    snapshot["health_tracking_mode"] = BodyTrackingHealth.trackingMode;
    snapshot["health_connect_state_result"] =
        BodyTrackingHealth.connectStateResult;
    snapshot["health_tracker_count"] = BodyTrackingHealth.trackerCount;
    snapshot["health_unique_tracker_count"] =
        BodyTrackingHealth.uniqueTrackerCount;
    snapshot["health_body_state_result"] =
        BodyTrackingHealth.bodyStateResult;
    snapshot["health_is_tracking"] = BodyTrackingHealth.isTracking;
    snapshot["health_tracking_state_code"] =
        BodyTrackingHealth.trackingStateCode;
    snapshot["health_body_state_code"] =
        BodyTrackingHealth.bodyStateCode;
    snapshot["health_body_error_code"] =
        BodyTrackingHealth.bodyErrorCode;
    snapshot["health_connected_band_count"] =
        BodyTrackingHealth.connectedBandCount;
    snapshot["health_body_data_result"] =
        BodyTrackingHealth.bodyDataResult;
    snapshot["health_body_role_count"] =
        BodyTrackingHealth.bodyRoleCount;
    return snapshot;
}

std::array<std::array<double, 6>, 24> getBodyJointsVelocity() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsVelocity;
}

std::array<std::array<double, 6>, 24> getBodyJointsAcceleration() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsAcceleration;
}

std::array<int64_t, 24> getBodyJointsTimestamp() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyJointsTimestamp;
}

int64_t getBodyTimeStampNs() {
    std::lock_guard<std::mutex> lock(bodyMutex);
    return BodyTimeStampNs;
}

int numMotionDataAvailable() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return NumMotionDataAvailable;
}

std::vector<std::array<double, 7>> getMotionTrackerPose() {
    std::lock_guard<std::mutex> lock(motionMutex);
    std::vector<std::array<double, 7>> result;
    for (int i = 0; i < NumMotionDataAvailable; i++) {
        result.push_back(MotionTrackerPose[i]);
    }
    return result;
}

std::vector<std::array<double, 6>> getMotionTrackerVelocity() {
    std::lock_guard<std::mutex> lock(motionMutex);
    std::vector<std::array<double, 6>> result;
    for (int i = 0; i < NumMotionDataAvailable; i++) {
        result.push_back(MotionTrackerVelocity[i]);
    }
    return result;
}

std::vector<std::array<double, 6>> getMotionTrackerAcceleration() {
    std::lock_guard<std::mutex> lock(motionMutex);
    std::vector<std::array<double, 6>> result;
    for (int i = 0; i < NumMotionDataAvailable; i++) {
        result.push_back(MotionTrackerAcceleration[i]);
    }
    return result;
}

std::vector<std::string> getMotionTrackerSerialNumbers() {
    std::lock_guard<std::mutex> lock(motionMutex);
    std::vector<std::string> result;
    for (int i = 0; i < NumMotionDataAvailable; i++) {
        result.push_back(MotionTrackerSerialNumbers[i]);
    }
    return result;
}

int64_t getMotionTimeStampNs() {
    std::lock_guard<std::mutex> lock(motionMutex);
    return MotionTimeStampNs;
}

pybind11::dict getMotionTrackerSnapshot() {
    std::lock_guard<std::mutex> lock(motionMutex);
    std::vector<std::string> serialNumbers;
    std::vector<std::array<double, 7>> poses;
    std::vector<std::array<double, 6>> velocities;
    std::vector<std::array<double, 6>> accelerations;
    serialNumbers.reserve(NumMotionDataAvailable);
    poses.reserve(NumMotionDataAvailable);
    velocities.reserve(NumMotionDataAvailable);
    accelerations.reserve(NumMotionDataAvailable);
    for (int i = 0; i < NumMotionDataAvailable; ++i) {
        serialNumbers.push_back(MotionTrackerSerialNumbers[i]);
        poses.push_back(MotionTrackerPose[i]);
        velocities.push_back(MotionTrackerVelocity[i]);
        accelerations.push_back(MotionTrackerAcceleration[i]);
    }
    pybind11::dict snapshot;
    snapshot["available"] =
        NumMotionDataAvailable > 0 && MotionTimeStampNs > 0;
    snapshot["count"] = NumMotionDataAvailable;
    snapshot["timestamp_ns"] = MotionTimeStampNs;
    snapshot["serial_numbers"] = serialNumbers;
    snapshot["poses"] = poses;
    snapshot["velocities"] = velocities;
    snapshot["accelerations"] = accelerations;
    return snapshot;
}

pybind11::dict getXr24AnkleSnapshot() {
    // BodyTracking and MotionTracking are mutually exclusive PICO modes.
    // Locking both retained states prevents a torn host read, but cannot make
    // them one source packet. This diagnostic getter must remain unapproved.
    std::scoped_lock lock(bodyMutex, motionMutex);

    std::vector<std::string> serialNumbers;
    std::vector<std::array<double, 7>> trackerPoses;
    std::vector<std::array<double, 6>> trackerVelocities;
    std::vector<std::array<double, 6>> trackerAccelerations;
    serialNumbers.reserve(NumMotionDataAvailable);
    trackerPoses.reserve(NumMotionDataAvailable);
    trackerVelocities.reserve(NumMotionDataAvailable);
    trackerAccelerations.reserve(NumMotionDataAvailable);
    for (int i = 0; i < NumMotionDataAvailable; ++i) {
        serialNumbers.push_back(MotionTrackerSerialNumbers[i]);
        trackerPoses.push_back(MotionTrackerPose[i]);
        trackerVelocities.push_back(MotionTrackerVelocity[i]);
        trackerAccelerations.push_back(MotionTrackerAcceleration[i]);
    }

    pybind11::dict body;
    body["available"] = BodyDataAvailable;
    body["timestamp_ns"] = BodyTimeStampNs;
    body["body_timestamp_contract"] = BodyTimestampContract;
    body["joint_timestamp_contract"] = BodyJointTimestampContract;
    body["poses"] = BodyJointsPose;
    body["velocities"] = BodyJointsVelocity;
    body["accelerations"] = BodyJointsAcceleration;
    body["joint_timestamps_ns"] = BodyJointsTimestamp;
    if (BodySampleSequenceAvailable) {
        body["sample_sequence"] = BodySampleSequence;
    } else {
        body["sample_sequence"] = pybind11::none();
    }
    body["health_supported"] = GrootTeleopSafetyProtocolSeen;
    body["health_available"] = BodyTrackingHealth.available;
    body["health_valid"] = BodyTrackingHealth.valid;
    body["health_schema_version"] = BodyTrackingHealth.schemaVersion;
    body["health_sample_sequence"] = BodyTrackingHealth.sampleSequence;
    body["health_timestamp_ns"] = BodyTrackingHealth.timestampNs;
    body["health_client_build"] = BodyTrackingHealth.clientBuild;
    body["health_calibration_result"] =
        BodyTrackingHealth.calibrationResult;
    body["health_calibrated"] = BodyTrackingHealth.calibrated;
    body["health_tracking_mode"] = BodyTrackingHealth.trackingMode;
    body["health_connect_state_result"] =
        BodyTrackingHealth.connectStateResult;
    body["health_tracker_count"] = BodyTrackingHealth.trackerCount;
    body["health_unique_tracker_count"] =
        BodyTrackingHealth.uniqueTrackerCount;
    body["health_body_state_result"] =
        BodyTrackingHealth.bodyStateResult;
    body["health_is_tracking"] = BodyTrackingHealth.isTracking;
    body["health_tracking_state_code"] =
        BodyTrackingHealth.trackingStateCode;
    body["health_body_state_code"] = BodyTrackingHealth.bodyStateCode;
    body["health_body_error_code"] = BodyTrackingHealth.bodyErrorCode;
    body["health_connected_band_count"] =
        BodyTrackingHealth.connectedBandCount;
    body["health_body_data_result"] =
        BodyTrackingHealth.bodyDataResult;
    body["health_body_role_count"] = BodyTrackingHealth.bodyRoleCount;

    pybind11::dict trackers;
    trackers["available"] =
        NumMotionDataAvailable > 0 && MotionTimeStampNs > 0;
    trackers["count"] = NumMotionDataAvailable;
    trackers["timestamp_ns"] = MotionTimeStampNs;
    trackers["serial_numbers"] = serialNumbers;
    trackers["poses"] = trackerPoses;
    trackers["velocities"] = trackerVelocities;
    trackers["accelerations"] = trackerAccelerations;

    pybind11::dict snapshot;
    snapshot["contract"] =
        "xrt_xr24_plus_retained_motion_unapproved_v0";
    snapshot["derivative_layout_contract"] =
        "linear_xyz_then_angular_xyz_v1";
    snapshot["source_coherence_contract"] =
        "mutually_exclusive_body_and_motion_retained_state_v0";
    snapshot["body"] = body;
    snapshot["trackers"] = trackers;
    return snapshot;
}

int DeviceControlJsonWrapper(const std::string& dev_id, const std::string& json_str) {
    const int rc = PXREADeviceControlJson(dev_id.c_str(), json_str.c_str());
    if (rc != 0) {
        throw std::runtime_error("device_control_json failed");
    }
    return rc; // 0
}

int SendBytesToDeviceWrapper(const std::string& dev_id, pybind11::bytes blob) {
    std::string s = blob;  // copy Python bytes to std::string
    const int rc = PXREASendBytesToDevice(dev_id.c_str(), s.data(), static_cast<unsigned>(s.size()));
    if (rc != 0) {
        throw std::runtime_error("send_bytes_to_device failed");
    }
    return rc; // 0
}


PYBIND11_MODULE(xrobotoolkit_sdk, m) {
    m.def("init", &init, "Initialize the PXREARobot SDK.");
    m.def("close", &deinit, "Deinitialize the PXREARobot SDK.");
    m.def("get_left_controller_pose", &getLeftControllerPose, "Get the left controller pose.");
    m.def("get_right_controller_pose", &getRightControllerPose, "Get the right controller pose.");
    m.def("get_controller_snapshot", &getControllerSnapshot, "Atomically get both controllers, their inputs, source timestamps, and tracking health.");
    m.def("get_left_controller_timestamp_ns", &getLeftControllerTimeStampNs, "Get the timestamp of the latest left-controller sample.");
    m.def("get_right_controller_timestamp_ns", &getRightControllerTimeStampNs, "Get the timestamp of the latest right-controller sample.");
    m.def("get_headset_pose", &getHeadsetPose, "Get the headset pose.");
    m.def("get_left_trigger", &getLeftTrigger, "Get the left trigger value.");
    m.def("get_left_grip", &getLeftGrip, "Get the left grip value.");
    m.def("get_right_trigger", &getRightTrigger, "Get the right trigger value.");
    m.def("get_right_grip", &getRightGrip, "Get the right grip value.");
    m.def("get_left_menu_button", &getLeftMenuButton, "Get the left menu button state.");
    m.def("get_right_menu_button", &getRightMenuButton, "Get the right menu button state.");
    m.def("get_left_axis_click", &getLeftAxisClick, "Get the left axis click state.");
    m.def("get_right_axis_click", &getRightAxisClick, "Get the right axis click state.");
    m.def("get_left_axis", &getLeftAxis, "Get the left axis values (x, y).");
    m.def("get_right_axis", &getRightAxis, "Get the right axis values (x, y).");
    m.def("get_X_button", &getLeftPrimaryButton, "Get the left primary button state.");
    m.def("get_A_button", &getRightPrimaryButton, "Get the right primary button state.");
    m.def("get_Y_button", &getLeftSecondaryButton, "Get the left secondary button state.");
    m.def("get_B_button", &getRightSecondaryButton, "Get the right secondary button state.");
    m.def("get_time_stamp_ns", &getTimeStampNs, "Get the timestamp in nanoseconds.");
    m.def("get_left_hand_tracking_state", &getLeftHandTrackingState, "Get the left hand state.");
    m.def("get_right_hand_tracking_state", &getRightHandTrackingState, "Get the right hand state.");
    m.def("get_left_hand_is_active", &getLeftHandIsActive, "Get the left hand tracking quality (0 = low, 1 = high).");
    m.def("get_right_hand_is_active", &getRightHandIsActive, "Get the right hand tracking quality (0 = low, 1 = high).");
    
    // Body tracking functions
    m.def("is_body_data_available", &isBodyDataAvailable, "Check if body tracking data is available.");
    m.def("get_body_snapshot", &getBodySnapshot, "Atomically get body availability, pose, velocity, acceleration, joint timestamps, and source timestamp.");
    m.def("get_body_joints_pose", &getBodyJointsPose, "Get the body joints pose data (24 joints, 7 values each: x,y,z,qx,qy,qz,qw).");
    m.def("get_body_joints_velocity", &getBodyJointsVelocity, "Get the body joints velocity data (24 joints, 6 values each: vx,vy,vz,wx,wy,wz).");
    m.def("get_body_joints_acceleration", &getBodyJointsAcceleration, "Get the body joints acceleration data (24 joints, 6 values each: ax,ay,az,wax,way,waz).");
    m.def("get_body_joints_timestamp", &getBodyJointsTimestamp, "Get the body joints IMU timestamp data (24 joints).");
    m.def("get_body_timestamp_ns", &getBodyTimeStampNs, "Get the body data timestamp in nanoseconds.");

    // Motion tracker functions
    m.def("num_motion_data_available", &numMotionDataAvailable, "Check if motion tracker data is available.");
    m.def("get_motion_tracker_snapshot", &getMotionTrackerSnapshot, "Atomically get motion tracker count, serials, pose, velocity, acceleration, and source timestamp.");
    m.def("get_xr24_ankle_snapshot", &getXr24AnkleSnapshot, "Get one host-atomic but source-unapproved XR24 body plus retained motion-tracker diagnostic snapshot.");
    m.def("get_motion_tracker_pose", &getMotionTrackerPose, "Get the motion tracker pose data (3 trackers, 7 values each: x,y,z,qx,qy,qz,qw).");
    m.def("get_motion_tracker_velocity", &getMotionTrackerVelocity, "Get the motion tracker velocity data (3 trackers, 6 values each: vx,vy,vz,wx,wy,wz).");
    m.def("get_motion_tracker_acceleration", &getMotionTrackerAcceleration, "Get the motion tracker acceleration data (3 trackers, 6 values each: ax,ay,az,wax,way,waz).");
    m.def("get_motion_tracker_serial_numbers", &getMotionTrackerSerialNumbers, "Get the serial numbers of the motion trackers.");
    m.def("get_motion_timestamp_ns", &getMotionTimeStampNs, "Get the motion data timestamp in nanoseconds.");
    

    // send json bytes functions
    m.def("device_control_json", &DeviceControlJsonWrapper, "Send a JSON control command to a device");
    m.def("send_bytes_to_device", &SendBytesToDeviceWrapper, "Send raw bytes to a device");
    
    m.doc() = "Python bindings for PXREARobot SDK using pybind11.";
}
