#ifndef __MANUS_SDK_TYPES_H__
#define __MANUS_SDK_TYPES_H__

// Set up a Doxygen group.
/** @addtogroup ManusSdkTypes
 *  @{
 */

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>

/******************************************************************************
 * Preprocessor defines.
 *****************************************************************************/

/// @brief Used to descriptively refer to the number of fingers on a hand.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define NUM_FINGERS_ON_HAND 5

/// @brief Used to descriptively refer to the number of flex sensor segments.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define NUM_FLEX_SEGMENTS_PER_FINGER 2

/// @brief Used to descriptively refer to the maximum IMUs count on a glove.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUM_IMUS_ON_GLOVE (NUM_FINGERS_ON_HAND + 1)

/// @brief Used to descriptively refer to the maximum number of Polygon users.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_USERS 16 // we should never reach that

/// @brief Used to descriptively refer to the maximum user name length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_USERNAME 64

/// @brief Used to descriptively refer to the maximum body measurement name length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_MEASUREMENT 64

/// Host name length is based on a post made here:
/// https://community.cisco.com/t5/other-network-architecture/maximum-length-hostname/td-p/529327
/// Which in turn was based on: https://www.ietf.org/rfc/rfc1035.txt
/// @brief Used to descriptively refer to the maximum host name length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_HOST_NAME 256

/// @brief Used to descriptively refer to the maximum IP address length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
/// It is based on the length of an IPv6 address. Example:
/// "2001:0db8:0000:0000:0000:8a2e:0370:7334".
#define MAX_NUM_CHARS_IN_IP_ADDRESS 40

/// @brief Used to descriptively refer to the maximum tracker name length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_TRACKER_ID 32

/// @brief Used to descriptively refer to the maximum tracker manufacturer length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_TRACKER_MANUFACTURER 32

/// @brief Used to descriptively refer to the maximum tracker manufacturer length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_TRACKER_PRODUCTNAME 32

/// @brief Used to descriptively refer to the maximum target name length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_TARGET_ID 32

/// @brief Used to descriptively refer to the maximum version string length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_VERSION 16

/// @brief Used to descriptively refer to the value given to glove and dongle IDs when they are uninitialized.
#define UNINITIALIZED_ID 0

/// @brief Used to descriptively refer to the maximum number of hosts that we support when finding manus core.
#define MAX_NUMBER_OF_HOSTS 100

/// @brief Used to descriptively refer to the max number of supported dongles.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUMBER_OF_DONGLES 16 // we only have 35 radio channels available. if we got 16 dongles, expect wireless issues! and no 16 usb ports is also not likely.

/// @brief Used to descriptively refer to the maximum license type string length.
#define MAX_NUM_CHARS_IN_LICENSE_TYPE 64

/// @brief Used to descriptively refer to the maximum calibration step title length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_CALIBRATION_TITLE 64

/// @brief Used to descriptively refer to the maximum calibration step description length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_CALIBRATION_DESCRIPTION 256

#define MAX_NUM_CHARS_IN_CALIBRATION_FILE 65536

/// @brief Constants for the maximum number of devices considered plausible.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUMBER_OF_GLOVES  (MAX_NUMBER_OF_DONGLES* 2)
#define MAX_NUMBER_OF_HAPTICS_DONGLES  MAX_NUMBER_OF_DONGLES

/// @brief Constants for the maximum number of skeletons.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUMBER_OF_SKELETONS  32

/// @brief Constants for the maximum number of users.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUMBER_OF_USERS  MAX_USERS

/// @brief Constants for the maximum number of trackers for each polygon skeleton.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define NUMBER_OF_TRACKERS_PER_POLYGON_SKELETON 8

/// @brief Constants for the maximum number of trackers.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUMBER_OF_TRACKERS (MAX_NUMBER_OF_USERS * NUMBER_OF_TRACKERS_PER_POLYGON_SKELETON)

/// @brief Used to descriptively refer to the maximum node name length.
/// Used with arrays to make them more descriptive than simply using the
/// number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_NODE_NAME 256 // this is for a UTF8 string , NOT an ASCII CHAR array (same base type though)


/// @brief Used to descriptively refer to the max number of chain nodes.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_CHAIN_LENGTH 32

/// @brief Used to descriptively refer to the max number of fingers we support per hand in chains (not the same as a glove, as we can retarget alien hands with more or less fingers).
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUM_FINGER_IDS 10

/// @brief Used to descriptively refer to the max number of toes we support per foot in chains ( we can retarget alien feet with more or less toes).
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUM_TOE_IDS 10

/// @brief Used to descriptively refer to the max length of a system error message string.
/// Used with arrays to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_SYSTEM_ERROR_MESSAGE 256

/// @brief Used to descriptively refer to the max length of a debugging id string.
/// Used with arrays to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUM_CHAR_DEBUGGING_ID 64

/// @brief Used to descriptively refer to the max number of ergonomics data.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUMBER_OF_ERGONOMICS_DATA MAX_NUMBER_OF_GLOVES

/// @brief Used to descriptively refer to the max number of sessions connected to Core.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUMBER_OF_SESSIONS 8 // this is not the real limit for Core but just for the SDKClient to handle

/// @brief Used to descriptively refer to the max number of skeletons for each session.
/// Used with arrays and loops to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUMBER_OF_SKELETONS_PER_SESSION	32	

/// @brief Used to descriptively refer to the max length of a skeleton name string.
/// Used with arrays to make them more descriptive than simply using
/// the number, and to make changing the number easier and safer.
#define MAX_NUM_CHARS_IN_SKELETON_NAME 256 // we already encountered 34 char names in unreal, but its utf8 so enbiggen even more!

/// @brief Used to descriptively refer to the max length of a timecode interface string.
#define MAX_NUM_CHARS_IN_TIMECODE_INTERFACE_STRINGS 64

/// @brief Used to descriptively refer to the max amount of timecode interfaces.
#define MAX_NUMBER_OF_AUDIO_INTERFACES 32

/// @brief Used to descriptively refer to the maximum number of bone weights per vertex.
#define MAX_BONE_WEIGHTS_PER_VERTEX 4

/// @brief Used to descriptively refer to the maximum number of nodes per each estimation skeleton
#define MAX_NUMBER_OF_NODES_PER_ESTIMATION_SKELETON 40

/// @brief Used to define the max amount of gestures in one chunk
#define MAX_GESTURE_DATA_CHUNK_SIZE 64

#define MAX_NUM_CHARS_IN_OPENXR_FEATURE_STRING 64

/// @brief Used to define the max amount of net devices
#define MAX_NUMBER_OF_NETDEVICES  MAX_NUMBER_OF_DONGLES

/// @brief Matchmaker requirements.
#define BROADCAST_ADDRESS "255.255.255.255"
#define BROADCAST_PORT "62687"
#define SECONDS_TO_FIND_HOSTS 2
#define SECONDS_TO_FIND_LOCAL_HOSTS 2
#define DEFAULT_BUFFER_SIZE 512

/******************************************************************************
 * Enums.
 *****************************************************************************/

/// @brief The return values that can be given by SDK wrapper functions.
typedef enum SDKReturnCode
{
	/// No issues occurred.
	SDKReturnCode_Success,

	/// Something went wrong, but no specific reason can be given.
	SDKReturnCode_Error,

	/// One of the arguments given had an invalid value.
	SDKReturnCode_InvalidArgument,

	/// The size of an argument given (e.g. an array) does not match the size
	/// of the data that it is intended to hold.
	SDKReturnCode_ArgumentSizeMismatch,

	/// A string of an unsupported size was encountered.
	SDKReturnCode_UnsupportedStringSizeEncountered,

	/// The Core SDK is not available.
	SDKReturnCode_SdkNotAvailable,

	/// The network host finder is not available.
	SDKReturnCode_HostFinderNotAvailable,

	/// The data requested is not available.
	SDKReturnCode_DataNotAvailable,

	/// Failed to allocate memory for something.
	SDKReturnCode_MemoryError,

	/// Something went wrong in the SDK internally.
	SDKReturnCode_InternalError,

	/// The function was not intended to be called at this time.
	SDKReturnCode_FunctionCalledAtWrongTime,

	/// No connection to Core was made.
	SDKReturnCode_NotConnected,

	/// The connection with Core timed out.
	SDKReturnCode_ConnectionTimeout,

	/// using an uninitialized ID is bad.
	SDKReturnCode_InvalidID,

	/// memory unallocated or just a null pointer passed where it wasn't supposed to be!
	SDKReturnCode_NullPointer,

	/// null sequence type for polygon calibration
	SDKReturnCode_InvalidSequence,

	/// don't forget to set the coordinate system type or there will be trouble
	SDKReturnCode_NoCoordinateSystemSet,

	/// if everything is being terminated. don't restart
	SDKReturnCode_SdkIsTerminating,

	/// the stub has been reset but someone is tryign to use it anyway. usually after a shutdown of the SDK.
	SDKReturnCode_StubNullPointer,

	/// Skeleton could not be loaded. usually when using more then the max skeletons per session (32).
	SDKReturnCode_SkeletonNotLoaded,
	
	/// Function not available for this version of the SDK
	SDKReturnCode_FunctionNotAvailable,

	SDKReturnCode_MAX_SIZE
} SDKReturnCode;

/// @brief Used to tell what severity the log is
typedef enum LogSeverity
{
	LogSeverity_Debug,
	LogSeverity_Info,
	LogSeverity_Warn,
	LogSeverity_Error
} LogSeverity;


/// @brief Used to tell what client is using the wrapper.
/// This makes the session easier to identify in the landscape.
typedef enum SessionType
{
	SessionType_Unknown,
	SessionType_UnityPlugin,
	SessionType_UnrealPlugin,
	SessionType_CoreSDK,
	SessionType_Xsens,
	SessionType_Optitrack,
	SessionType_MotionBuilder,
	SessionType_VRED,
	SessionType_OpenXR,
	SessionType_Qualisys,
	SessionType_Vicon,
	SessionType_Nokov,
	SessionType_IcIdo,
	SessionType_Siemens
} SessionType;

/// @brief Describes the different types of trackers that can be used.
typedef enum TrackerType
{
	TrackerType_Unknown,
	TrackerType_Head,
	TrackerType_Waist,
	TrackerType_LeftHand,
	TrackerType_RightHand,
	TrackerType_LeftFoot,
	TrackerType_RightFoot,
	TrackerType_LeftUpperArm,
	TrackerType_RightUpperArm,
	TrackerType_LeftUpperLeg,
	TrackerType_RightUpperLeg,
	TrackerType_Controller,
	TrackerType_Camera,

	TrackerType_MAX_SIZE
} TrackerType;

/// @brief Describes the tracking quality.
typedef enum TrackingQuality
{
	TrackingQuality_Untrackable,
	TrackingQuality_BadTracking,
	TrackingQuality_Trackable,
} TrackerQuality;

/// @brief Describes the different types of tracker systems.
typedef enum TrackerSystemType
{
	TrackerSystemType_Unknown,
	TrackerSystemType_Antilatency,
	TrackerSystemType_ART,
	TrackerSystemType_OpenVR,
	TrackerSystemType_Optitrack,
	TrackerSystemType_Vicon,
	TrackerSystemType_OpenXR,
} TrackerSystemType;

/// @brief Describes the paired state of the device.
typedef enum DevicePairedState
{
	DevicePairedState_Unknown,
	DevicePairedState_Paired,
	DevicePairedState_Unpaired,
	DevicePairedState_Pairing,
} DevicePairedState;

/// @brief Describes the different types of device classes.
typedef enum DeviceClassType
{
	DeviceClassType_Unknown,
	DeviceClassType_Dongle,
	DeviceClassType_Glove,
	DeviceClassType_Glongle
} DeviceClassType;

/// @brief Describes the different types of Manus devices. 
typedef enum DeviceFamilyType
{
	DeviceFamilyType_Unknown,
	DeviceFamilyType_Prime1,
	DeviceFamilyType_Prime2,
	DeviceFamilyType_PrimeX, // TODO obsolete?
	DeviceFamilyType_Metaglove,
	DeviceFamilyType_Prime3,
	DeviceFamilyType_Virtual,
	DeviceFamilyType_MetaglovePro,
} DeviceFamilyType;

/// @brief Describes the different types of profile used during the calibration.
typedef enum ProfileType
{
	ProfileType_Hands,
	ProfileType_FullBody,

	ProfileType_MAX_SIZE
} ProfileType;

/// @brief The different types of body measurements used for the polygon calibration.
typedef enum MeasurementType
{
	MeasurementType_Unknown,
	MeasurementType_PlayerHeight,
	MeasurementType_SpineLength,
	MeasurementType_NeckLength,
	MeasurementType_UpperArmLength,
	MeasurementType_LowerArmLength,
	MeasurementType_ArmLength,
	MeasurementType_ArmSpan,
	MeasurementType_UpperLegLength,
	MeasurementType_LowerLegLength,
	MeasurementType_LegLength,
	MeasurementType_HandLength,
	MeasurementType_FootLength,
	MeasurementType_HipWidth,
	MeasurementType_ShoulderWidth,
	MeasurementType_ShoulderHeight,
	MeasurementType_HeadLength,
	MeasurementType_Thickness,
	MeasurementType_ArmRatio,
	MeasurementType_LegRatio,

	MeasurementType_MAX_SIZE // Warning, this value is used to define the UserProfile.Measurement[SIZE]
} MeasurementType;

/// @brief Describes the different types of tracker offsets.
typedef enum TrackerOffsetType
{
	TrackerOffsetType_Unknown,
	TrackerOffsetType_HeadTrackerToHead,
	TrackerOffsetType_HeadTrackerToTopOfHead,

	TrackerOffsetType_LeftHandTrackerToWrist,
	TrackerOffsetType_RightHandTrackerToWrist,
	TrackerOffsetType_LeftFootTrackerToAnkle,
	TrackerOffsetType_RightFootTrackerToAnkle,

	TrackerOffsetType_HipTrackerToHip,
	TrackerOffsetType_HipTrackerToLeftLeg,
	TrackerOffsetType_HipTrackerToRightLeg,

	TrackerOffsetType_LeftUpperArmTrackerToElbow,
	TrackerOffsetType_RightUpperArmTrackerToElbow,
	TrackerOffsetType_LeftUpperArmTrackerToShoulder,
	TrackerOffsetType_RightUpperArmTrackerToShoulder,

	TrackerOffsetType_MAX_SIZE // Warning, this value is used to define the UserProfile.TrackerOffset[SIZE]
} TrackerOffsetType;

/// @brief Describes the different types of extra tracker offsets.
typedef enum ExtraTrackerOffsetType
{
	ExtraTrackerOffsetType_Unknown,
	ExtraTrackerOffsetType_HeadForward,
	ExtraTrackerOffsetType_HipForward,
	ExtraTrackerOffsetType_HipHeight,
	ExtraTrackerOffsetType_MAX_SIZE // Warning, this value is used to define the UserProfile.TrackerOffset[SIZE]
} ExtraTrackerOffsetType;

/// @brief Describes the different types of body measurement units. 
typedef enum MeasurementUnit
{
	MeasurementUnit_Meters,
	MeasurementUnit_Percentage,
} MeasurementUnit;

/// @brief Describes the different types of body measurement categories used for the polygon calibration.
typedef enum MeasurementCategory
{
	MeasurementCategory_Misc,
	MeasurementCategory_Generic,
	MeasurementCategory_Arms,
	MeasurementCategory_Legs,
	MeasurementCategory_Body,
} MeasurementCategory;

/// @brief Describes the different possibilities for the update status. 
typedef enum UpdateStatusEnum
{
	UpdateStatusEnum_Unknown,
	UpdateStatusEnum_NoUpdateAvailable,
	UpdateStatusEnum_UpdateAvailable,
	UpdateStatusEnum_MandatoryUpdateAvailable,
	UpdateStatusEnum_Updating,
} UpdateStatusEnum;

/// @brief Describes the different skeleton types. 
typedef enum SkeletonType
{
	SkeletonType_Invalid,
	SkeletonType_Hand,
	SkeletonType_Body,
	SkeletonType_Both
} SkeletonType;

/// @brief Describes the possible data that can be used for the skeleton animation.
typedef enum SkeletonTargetType
{
	SkeletonTargetType_Invalid,
	SkeletonTargetType_UserData,
	SkeletonTargetType_UserIndexData,
	SkeletonTargetType_AnimationData,
	SkeletonTargetType_GloveData
} SkeletonTargetType;

/// @brief Describes the possible nodes types used when setting up the skeleton.
typedef enum NodeType
{
	NodeType_Invalid,
	NodeType_Joint,
	NodeType_Mesh,
	NodeType_Leaf,
	NodeType_Collider
} NodeType;

/// @brief Describes the settings that can be applied to a node, it is defined as a flag so that more than one setting can be set.
typedef enum NodeSettingsFlag
{
	NodeSettingsFlag_None = 0,
	NodeSettingsFlag_IK = 1 << 0,
	NodeSettingsFlag_Foot = 1 << 1,
	NodeSettingsFlag_RotationOffset = 1 << 2,
	NodeSettingsFlag_Leaf = 1 << 3,
} NodeSettingsFlag;

/// @brief Describes the possible chain types used when setting up the skeleton.
typedef enum ChainType
{
	ChainType_Invalid,
	ChainType_Arm,
	ChainType_Leg,
	ChainType_Neck,
	ChainType_Spine,
	ChainType_FingerThumb,
	ChainType_FingerIndex,
	ChainType_FingerMiddle,
	ChainType_FingerRing,
	ChainType_FingerPinky,
	ChainType_Pelvis,
	ChainType_Head,
	ChainType_Shoulder,
	ChainType_Hand,
	ChainType_Foot,
	ChainType_Toe
} ChainType;

/// @brief Describes the possible collider types.
typedef enum CollisionType
{
	CollisionType_None,
	CollisionType_Discrete,
	CollisionType_Continuous,
} CollisionType;

/// @brief Describes the possible collider types.
typedef enum ColliderType
{
	ColliderType_Invalid,
	ColliderType_Sphere,
	ColliderType_Capsule,
	ColliderType_Box,
} ColliderType;

/// @brief Describes the possible chain side.
typedef enum Side
{
	Side_Invalid,
	Side_Left,
	Side_Right,
	Side_Center
} Side;

/// @brief Describes which sensor data the hand motion is based on.
typedef enum HandMotion
{
	HandMotion_None,
	HandMotion_IMU,
	HandMotion_Tracker,
	HandMotion_Tracker_RotationOnly,
	HandMotion_Auto
} HandMotion;

/// @brief Describes the direction of the coordinate system axis in 3d space.
typedef enum AxisDirection
{
	AxisDirection_Invalid,
	AxisDirection_Backward,
	AxisDirection_Left,
	AxisDirection_Down,
	AxisDirection_Up,
	AxisDirection_Right,
	AxisDirection_Forward
} AxisDirection;

/// @brief Describes the view of the coordinate system axis.
/// Consider yourself sitting in front of your computer screen.
/// From Viewer means it goes into the screen, so away from you.
/// To Viewer means the axis goes from the screen towards you.
typedef enum AxisView
{
	AxisView_Invalid,

	AxisView_ZFromViewer,
	AxisView_YFromViewer,
	AxisView_XFromViewer,

	AxisView_XToViewer,
	AxisView_YToViewer,
	AxisView_ZToViewer
} AxisView;

/// @brief Describes the polarity of the coordinate system axis.
typedef enum AxisPolarity
{
	AxisPolarity_Invalid,

	AxisPolarity_NegativeZ,
	AxisPolarity_NegativeY,
	AxisPolarity_NegativeX,

	AxisPolarity_PositiveX,
	AxisPolarity_PositiveY,
	AxisPolarity_PositiveZ
} AxisPolarity;

/// @brief Describes the possible types for system messages received from core. 
typedef enum SystemMessageType
{
	SystemMessageType_Unknown,
	SystemMessageType_LibDebugReplugDongle,
	SystemMessageType_LibDebugRxStall,
	SystemMessageType_LibDebugTxStall,
	
	SystemMessageType_TrackerError,
	SystemMessageType_TrackerOk,
	SystemMessageType_TrackerSystemOutOfDate,
	
	SystemMessageType_GloveSanityErrorPSOCInit,
	SystemMessageType_GloveSanityErrorQCBatV,
	SystemMessageType_GloveSanityErrorQCLRACalib,
	SystemMessageType_GloveSanityErrorQCFlexInit,
	SystemMessageType_GloveSanityErrorQCIMUInit,
	SystemMessageType_GloveSanityErrorQCIMUCalib,
	SystemMessageType_GloveSanityErrorQCID,
	SystemMessageType_GloveSanityErrorQCInterCPU,
	
	SystemMessageType_SessionConnectionVersionMismatch,

	SystemMessageType_TemporarySkeletonModified,

	SystemMessageType_SessionRefusedDueToLicenseIssue,

	SystemMessageType_LaunchDevTools
} SystemMessageType;

/// @brief Describes the possible types for the ergonomics data.
typedef enum ErgonomicsDataType
{
//	ErgonomicsDataType_Invalid,

	ErgonomicsDataType_LeftFingerThumbMCPSpread,
	ErgonomicsDataType_LeftFingerThumbMCPStretch,
	ErgonomicsDataType_LeftFingerThumbPIPStretch,
	ErgonomicsDataType_LeftFingerThumbDIPStretch,

	ErgonomicsDataType_LeftFingerIndexMCPSpread,
	ErgonomicsDataType_LeftFingerIndexMCPStretch,
	ErgonomicsDataType_LeftFingerIndexPIPStretch,
	ErgonomicsDataType_LeftFingerIndexDIPStretch,

	ErgonomicsDataType_LeftFingerMiddleMCPSpread,
	ErgonomicsDataType_LeftFingerMiddleMCPStretch,
	ErgonomicsDataType_LeftFingerMiddlePIPStretch,
	ErgonomicsDataType_LeftFingerMiddleDIPStretch,

	ErgonomicsDataType_LeftFingerRingMCPSpread,
	ErgonomicsDataType_LeftFingerRingMCPStretch,
	ErgonomicsDataType_LeftFingerRingPIPStretch,
	ErgonomicsDataType_LeftFingerRingDIPStretch,

	ErgonomicsDataType_LeftFingerPinkyMCPSpread,
	ErgonomicsDataType_LeftFingerPinkyMCPStretch,
	ErgonomicsDataType_LeftFingerPinkyPIPStretch,
	ErgonomicsDataType_LeftFingerPinkyDIPStretch,


	ErgonomicsDataType_RightFingerThumbMCPSpread,
	ErgonomicsDataType_RightFingerThumbMCPStretch,
	ErgonomicsDataType_RightFingerThumbPIPStretch,
	ErgonomicsDataType_RightFingerThumbDIPStretch,

	ErgonomicsDataType_RightFingerIndexMCPSpread,
	ErgonomicsDataType_RightFingerIndexMCPStretch,
	ErgonomicsDataType_RightFingerIndexPIPStretch,
	ErgonomicsDataType_RightFingerIndexDIPStretch,

	ErgonomicsDataType_RightFingerMiddleMCPSpread,
	ErgonomicsDataType_RightFingerMiddleMCPStretch,
	ErgonomicsDataType_RightFingerMiddlePIPStretch,
	ErgonomicsDataType_RightFingerMiddleDIPStretch,

	ErgonomicsDataType_RightFingerRingMCPSpread,
	ErgonomicsDataType_RightFingerRingMCPStretch,
	ErgonomicsDataType_RightFingerRingPIPStretch,
	ErgonomicsDataType_RightFingerRingDIPStretch,

	ErgonomicsDataType_RightFingerPinkyMCPSpread,
	ErgonomicsDataType_RightFingerPinkyMCPStretch,
	ErgonomicsDataType_RightFingerPinkyPIPStretch,
	ErgonomicsDataType_RightFingerPinkyDIPStretch,

	ErgonomicsDataType_MAX_SIZE
} ErgonomicsDataType;

/// @brief Describes the possible Manus license types.
typedef enum LicenseType
{
	LicenseType_Undefined,
	LicenseType_Polygon,
	LicenseType_CoreXO,
	LicenseType_CorePro,
	LicenseType_CoreXOPro,

	LicenseType_CoreX,
	LicenseType_CoreO,
	LicenseType_CoreQ,
	LicenseType_CoreXPro,
	LicenseType_CoreOPro,
	LicenseType_CoreQPro,
	LicenseType_CoreXOQPro,
	LicenseType_CoreXR,
	LicenseType_CorePrimeThree,
	LicenseType_Feature,

} LicenseType;

/// @brief The possible FPS rates
typedef enum TimecodeFPS
{
	TimecodeFPS_Undefined,
	TimecodeFPS_23_976,
	TimecodeFPS_24,
	TimecodeFPS_25,
	TimecodeFPS_29_97,
	TimecodeFPS_30,
	TimecodeFPS_48,
	TimecodeFPS_50,
	TimecodeFPS_59_94,
	TimecodeFPS_60,
	TimecodeFPS_29_97DF,
	TimecodeFPS_59_94DF,
} TimecodeFPS;

typedef enum FingerJointType
{
	FingerJointType_Invalid,
	FingerJointType_Metacarpal,
	FingerJointType_Proximal,
	FingerJointType_Intermediate,
	FingerJointType_Distal, //thumb doesn't have it 
	FingerJointType_Tip
}FingerJointType;

typedef enum SetGloveCalibrationReturnCode
{
	SetGloveCalibrationReturnCode_Error,
	SetGloveCalibrationReturnCode_Success,
	SetGloveCalibrationReturnCode_VersionError,
	SetGloveCalibrationReturnCode_WrongSideError,
	SetGloveCalibrationReturnCode_GloveNotFoundError,
	SetGloveCalibrationReturnCode_UserServiceError,
	SetGloveCalibrationReturnCode_DeserializationError,

	SetGloveCalibrationReturnCode_MAX_SIZE
} SetGloveCalibrationReturnCode;

/******************************************************************************
 * Structs.
 *****************************************************************************/

/// @brief A 3D vector, used for translations.
typedef struct ManusVec3
{
	float x; //default = 0.0f;
	float y; //default = 0.0f;
	float z; //default = 0.0f;
} ManusVec3;

/// @brief A 2D vector, used for translations.
typedef struct ManusVec2
{
	float x; //default = 0.0f;
	float y; //default = 0.0f;
} ManusVec2;

/// @brief A quaternion, used for rotations.
typedef struct ManusQuaternion
{
	float w; //default = 1.0f;
	float x; //default = 0.0f;
	float y; //default = 0.0f;
	float z; //default = 0.0f;
} ManusQuaternion;

/// @brief Transform containing position, rotation and scaling.
typedef struct ManusTransform
{
	ManusVec3 position;
	ManusQuaternion rotation;
	ManusVec3 scale;
} ManusTransform;

/// @brief Color containing red, green, blue and alpha.
typedef struct Color
{
	float r; //default = 0.0f;
	float g; //default = 0.0f;
	float b; //default = 0.0f;
	float a; //default = 0.0f;
}Color;

/// @brief A Timestamp
typedef struct ManusTimestampInfo
{
	uint16_t fraction; //is either frame in timecode or miliseconds in non timecode
	uint8_t second;
	uint8_t minute;
	uint8_t hour;
	uint8_t day; //is 0 in timecode
	uint8_t month; //is 0 in timecode
	uint32_t year; //is 0 in timecode
	bool timecode;
} ManusTimestampInfo;

/// @brief A compressed timestamp
typedef struct ManusTimestamp
{
	uint64_t time;
} ManusTimestamp;

/// @brief Information regarding IMU sensors used for calibration
typedef struct IMUCalibrationInfo
{
	uint32_t mag;					// Magnometer calibration level 0-3
	uint32_t acc;					// Accelerometer caibraton level 0-3
	uint32_t gyr;					// Gyroscope calibration level 0-3
	uint32_t sys;					// System accuracy
} IMUCalibrationInfo;

/// @brief Used to describe hardware, firmware or ManusCore version.
typedef struct Version
{
	uint32_t major;
	uint32_t minor;
	uint32_t patch;
	char label[MAX_NUM_CHARS_IN_VERSION];
	char sha[MAX_NUM_CHARS_IN_VERSION];
	char tag[MAX_NUM_CHARS_IN_VERSION];
} Version;

typedef struct FirmwareVersion
{
	int32_t version;
	ManusTimestamp timestamp;
} FirmwareVersion;

/// @brief Stores a single version string.
typedef struct ManusVersion 
{
	char versionInfo[MAX_NUM_CHARS_IN_VERSION];
} ManusVersion;

/// @brief Contains information for connecting to a host running Manus Core.
/// Note that if one of these values is blank, the other will be used when
/// connecting.
typedef struct ManusHost
{
	char hostName[MAX_NUM_CHARS_IN_HOST_NAME]; // TODO utf8 compliant?
	char ipAddress[MAX_NUM_CHARS_IN_IP_ADDRESS];
	Version manusCoreVersion;
} ManusHost;

/******************************************************************************
 * Tracking
 *****************************************************************************/



/// @brief Stores the name of a tracker.
typedef struct TrackerId
{
	char id[MAX_NUM_CHARS_IN_TRACKER_ID]; // todo. make this UTF8 compliant ?
} TrackerId;

/// @brief Stores all the tracker data that can be sent or received.
typedef struct TrackerData
{
	ManusTimestamp lastUpdateTime; //default = 0;

	TrackerId trackerId;

	uint32_t userId; //default = 0; 
	
	bool isHmd; //default = false;
	TrackerType trackerType; //default = TrackerType::TrackerType_Unknown;

	ManusQuaternion rotation;
	ManusVec3 position;

	TrackingQuality quality; //default = TrackingQuality::TrackingQuality_Untrackable;
} TrackerData;


/// @brief Stores the information sent by the tracker stream. 
typedef struct TrackerStreamInfo
{
	ManusTimestamp publishTime; //default = 0;	//	DateTime.UtcNow.
	uint32_t trackerCount; //default = 0;
} TrackerStreamInfo;

// ------------------------------------------------------------------------------------------------------------------------------
// end of Tracking
// ------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------------
// Gesture Stream
// ------------------------------------------------------------------------------------------------------------------------------

typedef struct GestureProbability
{
	uint32_t id;
	float percent;
} GestureProbability;

typedef struct GestureProbabilities
{
	uint32_t id; //default = 0;
	bool isUserID; //default = false;
	uint32_t totalGestureCount; //default = 0;
	GestureProbability gestureData[MAX_GESTURE_DATA_CHUNK_SIZE]; //default = { 0 };
	uint32_t gestureCount; //default = 0;
} GestureProbabilities;

typedef struct GestureStreamInfo
{
	ManusTimestamp publishTime; //default = 0;	//	DateTime.UtcNow.
	uint32_t gestureProbabilitiesCount; //default = 0;
} GestureStreamInfo;


// ------------------------------------------------------------------------------------------------------------------------------
// end of Gesture Stream
// ------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------------
// Skeleton Stream
// ------------------------------------------------------------------------------------------------------------------------------

/// @brief Stores the information regarding each skeleton node. 
/// The transform is defined as a local or global transform depending on the coordinate system set when initializing the SDK.
/// See functions CoreSdk_InitializeCoordinateSystemWithVUH and CoreSdk_InitializeCoordinateSystemWithDirection.
typedef struct SkeletonNode
{
	uint32_t id;
	ManusTransform transform;
} SkeletonNode;

/// @brief Stores the information regarding the skeletons that have been added to Manus Core.
typedef struct SkeletonInfo
{
	uint32_t id; //default = 0;
	uint32_t nodesCount; //default = 0;
	ManusTimestamp publishTime; //default = 0;	//	DateTime.UtcNow.
} SkeletonInfo;

/// @brief Stores the information regarding the skeletons coming from the estimation system in core
typedef struct RawSkeletonInfo
{
	uint32_t gloveId; //default = 0;
	uint32_t nodesCount; //default = 0;
	ManusTimestamp publishTime; //default = 0;	//	DateTime.UtcNow.
} RawSkeletonInfo;

/// @brief Stores the information sent by the skeleton stream. 
typedef struct SkeletonStreamInfo
{
	ManusTimestamp publishTime; //default = 0;	//	DateTime.UtcNow.
	uint32_t skeletonsCount; //default = 0;
} SkeletonStreamInfo;


// ------------------------------------------------------------------------------------------------------------------------------
// end of Skeleton Stream
// ------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------------
// Ergonomics
// ------------------------------------------------------------------------------------------------------------------------------

/// @brief Stores the received ergonomics data.
typedef struct ErgonomicsData
{
	uint32_t id; //default = 0;
	bool isUserID; //default = false;
	float data[ErgonomicsDataType_MAX_SIZE]; //default = { 0 };
} ErgonomicsData;

/// @brief Stores the information sent by the ergonomics stream.
typedef struct ErgonomicsStream
{
	ManusTimestamp publishTime; //default = 0;
	ErgonomicsData data[MAX_NUMBER_OF_ERGONOMICS_DATA];
	uint32_t dataCount; //default = 0;
} ErgonomicsStream;

// ------------------------------------------------------------------------------------------------------------------------------
// end of Ergonomics
// ------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------------
// Landscape
// ------------------------------------------------------------------------------------------------------------------------------

/// @brief Stores all the received dongle data.
typedef struct DongleLandscapeData
{
	uint32_t id;
	DeviceClassType classType;
	DeviceFamilyType familyType;
	bool isHaptics;

	Version hardwareVersion;
	Version firmwareVersion;
	ManusTimestamp firmwareTimestamp;

	uint32_t chargingState;

	int32_t channel;

	UpdateStatusEnum updateStatus;

	char licenseType[MAX_NUM_CHARS_IN_LICENSE_TYPE];

	ManusTimestamp lastSeen; // this may be used in playback/recording TBD

	uint32_t leftGloveID;
	uint32_t rightGloveID;

	LicenseType licenseLevel; //default = LicenseType::LicenseType_Undefined;
	ManusTimestamp licenseExpiration;

	uint32_t netDeviceID;
} DongleLandscapeData;

/// @brief Stores all the received glove data.
typedef struct GloveLandscapeData
{
	uint32_t id;
	DeviceClassType classType;
	DeviceFamilyType familyType;
	Side side;
	bool isHaptics;

	DevicePairedState pairedState;
	uint32_t dongleID;

	Version hardwareVersion;
	Version firmwareVersion;
	ManusTimestamp firmwareTimestamp;

	UpdateStatusEnum updateStatus;

	uint32_t batteryPercentage;
	int32_t transmissionStrength;

	IMUCalibrationInfo iMUCalibrationInfo[MAX_NUM_IMUS_ON_GLOVE];

	ManusTimestamp lastSeen; // this may be used in playback/recording TBD
	bool excluded;

	uint32_t netDeviceID;
} GloveLandscapeData;

/// @brief Stores informations regarding the lengths of different parts of the body.
typedef struct Measurement
{
	MeasurementType	entryType; //default = MeasurementType::MeasurementType_Unknown;
	float value; //default = 0.0f;

	MeasurementUnit unit; //default = MeasurementUnit_Meters;
	MeasurementCategory category; //default = MeasurementCategory_Misc;
	char displayName[MAX_NUM_CHARS_IN_MEASUREMENT];
} Measurement;

/// @brief Stores the local offsets to the trackers.
typedef struct TrackerOffset
{
	TrackerOffsetType entryType;
	ManusVec3 translation;
	ManusQuaternion rotation;
} TrackerOffset;

/// @brief Stores possible extra offsets to the trackers.
typedef struct ExtraTrackerOffset
{
	ExtraTrackerOffsetType entryType; //default = ExtraTrackerOffsetType::ExtraTrackerOffsetType_Unknown;
	float value; //default = 0.0f;
} ExtraTrackerOffset;

/// @brief Stores all the received tracker data.
typedef struct TrackerLandscapeData
{
	char id[MAX_NUM_CHARS_IN_TRACKER_ID];
	TrackerType type; //default = TrackerType::TrackerType_Unknown;
	TrackerSystemType systemType; //default = TrackerSystemType::TrackerSystemType_Unknown;
	uint32_t user; //default = 0;
	bool isHMD; //default = false;
	char manufacturer[MAX_NUM_CHARS_IN_TRACKER_MANUFACTURER]; // default = "Unknown"
	char productName[MAX_NUM_CHARS_IN_TRACKER_PRODUCTNAME]; // default = "Unknown"
} TrackerLandscapeData;

/// @brief Stores all the received user profile data.
typedef struct UserProfileLandscapeData
{
	ProfileType profileType; //default = ProfileType::ProfileType_Hands;
	Measurement measurements[MeasurementType_MAX_SIZE];
	TrackerOffset trackerOffsets[TrackerOffsetType_MAX_SIZE];
	ExtraTrackerOffset extraTrackerOffsets[ExtraTrackerOffsetType_MAX_SIZE];
} UserProfileLandscapeData;

/// @brief Stores all the received user data.
typedef struct UserLandscapeData
{
	uint32_t id; //default = 0;
	char name[MAX_NUM_CHARS_IN_USERNAME];
	Color color;
	uint32_t dongleID; //default = 0;
	uint32_t leftGloveID; //default = 0;
	uint32_t rightGloveID; //default = 0;
	UserProfileLandscapeData profile;
	uint32_t userIndex; //default = 0;
} UserLandscapeData;

/// @brief Stores all the received skeleton data.
typedef struct SkeletonLandscapeData
{
	uint32_t id; //default = 0;
	char session[MAX_NUM_CHARS_IN_HOST_NAME];
	uint32_t userId; //default = 0;
	SkeletonType type; //default = SkeletonType::SkeletonType_Invalid;
	char rootBoneName[MAX_NUM_CHARS_IN_NODE_NAME];
	bool scaled; //default = false;
} SkeletonLandscapeData;

/// @brief Stores all the information related to the devices present in the landscape.
typedef struct DeviceLandscape
{
	DongleLandscapeData dongles[MAX_NUMBER_OF_DONGLES];
	uint32_t dongleCount; //default = 0;
	GloveLandscapeData gloves[MAX_NUMBER_OF_GLOVES];
	uint32_t gloveCount; //default = 0;
} DeviceLandscape;

/// @brief Stores all the information related to the users present in the landscape.
typedef struct UserLandscape
{
	UserLandscapeData users[MAX_USERS];
	uint32_t userCount; //default = 0;
} UserLandscape;

/// @brief Stores all the information related to the skeletons present in the landscape.
typedef struct SkeletonLandscape
{
	SkeletonLandscapeData skeletons[MAX_NUMBER_OF_SKELETONS];
	uint32_t skeletonCount; //default = 0;
} SkeletonLandscape;

/// @brief Stores all the information related to the trackers present in the landscape.
typedef struct TrackerLandscape
{
	TrackerLandscapeData trackers[MAX_NUMBER_OF_TRACKERS];
	uint32_t trackerCount; //default = 0;
} TrackerLandscape;

/// @brief Stores the license information.
typedef struct LicenseInfo
{
	bool recording; //default = false;
	bool exporting; //default = false;
	bool advancedExporting; //default = false;
	bool unitySession; //default = false;
	bool unrealSession; //default = false;
	bool openXRSession; //default = false;
	bool sdk; //default = false;
	bool raw; //default = false;
	bool mobuSession; //default = false;
	bool xsensSession; //default = false;
	bool optitrackSession; //default = false;
	bool qualisysSession; //default = false;
	bool viconSession; //default = false;
	bool nokovSession; //default = false;
	bool icidoSession; //default = false;
	bool siemensSession; //default = false;
	bool vredSession; //default = false;

} LicenseInfo;

/// @brief Stores the landscape settings.
typedef struct SettingsLandscape
{
	Version manusCoreVersion;
	LicenseInfo license;
	bool playbackMode; //default = false;
	bool ignoreSessionTimeOuts; //default = false;
	FirmwareVersion firmwareOne;
	FirmwareVersion firmwareTwo;
	bool recordingMode; //default = false;
	bool isNetDevice; // default = false;
	bool isConnectedAsNetDevice; // default = false;
}SettingsLandscape;

typedef struct TimecodeInterface
{
	char name[MAX_NUM_CHARS_IN_TIMECODE_INTERFACE_STRINGS];
	char api[MAX_NUM_CHARS_IN_TIMECODE_INTERFACE_STRINGS];
	int index; //default = -1;
} TimecodeInterface;

typedef struct TimeLandscape
{
	TimecodeInterface interfaces[MAX_NUMBER_OF_AUDIO_INTERFACES];
	uint32_t interfaceCount; //default = 0;
	TimecodeInterface currentInterface;
	TimecodeFPS fps; //default = Undefined;
	bool fakeTimecode; //default = false;
	bool useSyncPulse; //default = false;
	bool deviceKeepAlive; //default = false;
	bool syncStatus; //default = false;
	bool timecodeStatus; //default = false;
	int32_t ltcChannel; //default = -1;
} TimeLandscape;

/// @brief Contains information about a gesture
typedef struct GestureLandscapeData
{
	uint32_t id; //default = 0;
	char name[MAX_NUM_CHARS_IN_USERNAME];
} GestureLandscapeData;

/// @brief Contains information about a net devices
typedef struct NetDeviceLandscapeData
{
	uint32_t netDeviceID;
	char hostname[MAX_NUM_CHARS_IN_HOST_NAME];
	char ip[MAX_NUM_CHARS_IN_IP_ADDRESS];
} NetDeviceLandscapeData;

/// @brief Contains information about a single net device
typedef struct NetDevicesLandscape
{
	uint32_t numberOfNetDevices; // default = 0;
	NetDeviceLandscapeData netDevices[MAX_NUMBER_OF_NETDEVICES];
} NetDeviceLandscape;

/// @brief Stores the landscape data.
typedef struct Landscape
{
	DeviceLandscape gloveDevices;
	UserLandscape users;
	SkeletonLandscape skeletons;
	TrackerLandscape trackers;
	SettingsLandscape settings;
	NetDevicesLandscape netDevices;
	TimeLandscape time;
	uint32_t gestureCount; //default = 0;
} Landscape;

// ------------------------------------------------------------------------------------------------------------------------------
// end of Landscape
// ------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------------
// Skeleton
// ------------------------------------------------------------------------------------------------------------------------------
/// @brief Stores the inverse kinematics settings of a node, the ikAim is used to control the IK solve, 1 is default, -1 is inversed.
typedef struct NodeSettingsIK
{
	float ikAim;
} NodeSettingsIK;

/// @brief Stores the settings for a node of type foot, heightFromGround is used to set the height of the 3d model ankle from ground.
typedef struct NodeSettingsFoot
{
	float heightFromGround;
} NodeSettingsFoot;

/// @brief Stores the rotation offset of a node, this is used internally for building the skeleton.
typedef struct NodeSettingsRotationOffset
{
	ManusQuaternion value;
} NodeSettingsRotationOffset;

/// @brief Stores the settings of a node of type leaf, the direction vector is defined with respect to the previous node in the chain.
typedef struct NodeSettingsLeaf
{
	ManusVec3 direction;
	float length;
} NodeSettingsLeaf;

/// @brief Stores all the node settings.
typedef struct NodeSettings
{
	NodeSettingsFlag usedSettings;
	NodeSettingsIK ik;
	NodeSettingsFoot foot;
	NodeSettingsRotationOffset rotationOffset;
	NodeSettingsLeaf leaf;
} NodeSettings;

/// @brief Stores the node setup information. Each node represents a segment of the skeleton that can be animated, nodes combine together to form chains.
/// the parentID is used to identify the node with respect to which the current one will move. 
typedef struct NodeSetup
{
	uint32_t id; //default = 0;
	char name[MAX_NUM_CHARS_IN_NODE_NAME]; // this is a UTF8 string , NOT an ASCII CHAR array (same base type though)
	NodeType type; //default = NodeType::NodeType_Invalid;
	ManusTransform transform;
	uint32_t parentID; //default = 0;
	NodeSettings settings;
} NodeSetup;

/// @brief Stores all the settings of a chain of type pelvis.
typedef struct ChainSettingsPelvis
{
	float hipHeight; //default = 0.0f;
	float hipBendOffset; //default = 0.0f;
	float thicknessMultiplier; //default = 1.0f;
} ChainSettingsPelvis;

/// @brief  Stores all the settings of a chain of type leg.
typedef struct ChainSettingsLeg
{
	bool reverseKneeDirection; //default = false;
	float kneeRotationOffset; //default = 0.0f;
	float footForwardOffset; //default = 0.0f;
	float footSideOffset; //default = 0.0f;
} ChainSettingsLeg;

/// @brief  Stores all the settings of a chain of type spine.
typedef struct ChainSettingsSpine
{
	float spineBendOffset; //default = 0.0f;
} ChainSettingsSpine;

/// @brief  Stores all the settings of a chain of type neck.
typedef struct ChainSettingsNeck
{
	float neckBendOffset; //default = 0.0f;
} ChainSettingsNeck;

/// @brief  Stores all the settings of a chain of type head.
typedef struct ChainSettingsHead
{
	float headPitchOffset; //default = 0.0f;
	float headYawOffset; //default = 0.0f;
	float headTiltOffset; //default = 0.0f;
	bool useLeafAtEnd; //default = false;
} ChainSettingsHead;

/// @brief  Stores all the settings of a chain of type arm.
typedef struct ChainSettingsArm
{
	float armLengthMultiplier; //default = 0.0f;
	float elbowRotationOffset; //default = 0.0f;

	ManusVec3 armRotationOffset;

	ManusVec3 positionMultiplier;
	ManusVec3 positionOffset;
} ChainSettingsArm;

/// @brief  Stores all the settings of a chain of type shoulder.
typedef struct ChainSettingsShoulder
{
	float forwardOffset; //default = 0.0f;
	float shrugOffset; //default = 0.0f;

	float forwardMultiplier; //default = 0.0f;
	float shrugMultiplier; //default = 0.0f;
} ChainSettingsShoulder;

/// @brief  Stores all the settings of a chain of type finger.
typedef struct ChainSettingsFinger
{
	bool useLeafAtEnd; //default = false;
	int32_t metacarpalBoneId; //default = 0;
	int32_t handChainId; //default = 0;
	float fingerWidth; //default = 0;
} ChainSettingsFinger;

/// @brief  Stores all the settings of a chain of type hand.
typedef struct ChainSettingsHand
{
	int32_t fingerChainIds[MAX_NUM_FINGER_IDS];
	int32_t fingerChainIdsUsed; //default = 0;
	HandMotion handMotion; //default = HandMotion::HandMotion_None;
} ChainSettingsHand;

/// @brief  Stores all the settings of a chain of type foot.
typedef struct ChainSettingsFoot
{
	int32_t toeChainIds[MAX_NUM_TOE_IDS];
	int32_t toeChainIdsUsed; //default = 0;
} ChainSettingsFoot;

/// @brief  Stores all the settings of a chain of type toe.
typedef struct ChainSettingsToe
{
	int32_t footChainId; //default = 0;
	bool useLeafAtEnd; //default = false;
} ChainSettingsToe;

/// @brief  Stores all chain settings.
typedef struct ChainSettings
{
	ChainType usedSettings;
	ChainSettingsPelvis pelvis;
	ChainSettingsLeg leg;
	ChainSettingsSpine spine;
	ChainSettingsNeck neck;
	ChainSettingsHead head;
	ChainSettingsArm arm;
	ChainSettingsShoulder shoulder;
	ChainSettingsFinger finger;
	ChainSettingsHand hand;
	ChainSettingsFoot foot;
	ChainSettingsToe toe;
} ChainSettings;

/// @brief Stores the chain setup information.
typedef struct ChainSetup
{
	uint32_t id; //default = 0;
	ChainType type; //default = ChainType::ChainType_Invalid;
	ChainType dataType; //default = ChainType::ChainType_Invalid;
	uint32_t dataIndex; //default = 0;
	uint32_t nodeIdCount; //default = 0;
	uint32_t nodeIds[MAX_CHAIN_LENGTH];
	ChainSettings settings;
	Side side; //default = Side::Side_Invalid;
} ChainSetup;

/// @brief  Stores all the settings of a collider of type sphere.
typedef struct SphereColliderSetup
{
	float radius; //default = 0;
} SphereColliderSetup;

/// @brief  Stores all the settings of a collider of type capsule.
typedef struct CapsuleColliderSetup
{
	float radius; //default = 0;
	float length; //default = 0;
} CapsuleColliderSetup;

/// @brief  Stores all the settings of a collider of type box.
typedef struct BoxColliderSetup
{
	ManusVec3 size; //default = 0,0,0;
} BoxColliderSetup;

/// @brief Stores the collider setup information.
typedef struct ColliderSetup
{
	uint32_t nodeID; //default = 0;
	ManusVec3 localPosition; // default 0,0,0
	ManusVec3 localRotation; // default 0,0,0

	ColliderType type;//default = ColliderType::ColliderType_Invalid
	SphereColliderSetup sphere;
	CapsuleColliderSetup capsule;
	BoxColliderSetup box;
} ColliderSetup;

/// @brief Stores the vertex weight information.
typedef struct Weight
{
	uint32_t nodeID;
	float weightValue;
} Weight;

/// @brief Stores the vertex information.
typedef struct Vertex
{
	ManusVec3 position; // default 0,0,0
	uint32_t weightsCount; //default = 0;
	Weight weights[MAX_BONE_WEIGHTS_PER_VERTEX];
} Vertex;

/// @brief Stores the triangle information.
typedef struct Triangle
{
	int32_t vertexIndex1;
	int32_t vertexIndex2;
	int32_t vertexIndex3;
} Triangle;

/// @brief Stores the information regarding the user data used to animate the skeleton.
typedef struct SkeletonTargetUserData
{
	uint32_t userID; //default = 0;
} SkeletonTargetUserData;

/// @brief Stores the information regarding the user index data used to animate the skeleton.
typedef struct SkeletonTargetUserIndexData
{
	uint32_t userIndex; //default = 0;
} SkeletonTargetUserIndexData;

/// @brief Stores the information regarding the animation data used to animate the skeleton.
typedef struct SkeletonTargetAnimationData
{
	char id[MAX_NUM_CHARS_IN_TARGET_ID];
} SkeletonTargetAnimationData;

/// @brief Stores the information regarding the glove data used to animate the skeleton.
typedef struct SkeletonTargetGloveData
{
	uint32_t gloveID; //default = 0;
} SkeletonTargetGloveData;

/// @brief Stores all the possible skeleton settings.
typedef struct SkeletonSettings
{
	bool scaleToTarget; //default = false;
	bool useEndPointApproximations; //default = false;
	CollisionType collisionType; //default = CollisionType::CollisionType_None

	SkeletonTargetType targetType;
	SkeletonTargetUserData skeletonTargetUserData;
	SkeletonTargetUserIndexData skeletonTargetUserIndexData;
	SkeletonTargetAnimationData skeletonTargetAnimationData;
	SkeletonTargetGloveData skeletonGloveData;
} SkeletonSettings;

/// @brief Stores the skeleton setup information.
typedef struct SkeletonSetupInfo
{
	uint32_t id; //default = 0;
	SkeletonType type; //default = SkeletonType::SkeletonType_Invalid;
	SkeletonSettings settings;
	char name[MAX_NUM_CHARS_IN_SKELETON_NAME]; // this is a UTF8 string , NOT an ASCII CHAR array (same base type though)
} SkeletonSetupInfo;


/// @brief Stores the amount of nodes and chains in the skeleton setup.
typedef struct SkeletonSetupArraySizes
{
	uint32_t nodesCount; //default = 0;
	uint32_t chainsCount; //default = 0;
	uint32_t collidersCount; //default = 0;
	uint32_t meshCount; //default = 0;
} SkeletonSetupArraySizes;

/// @brief Stores the temporary skeleton information.
typedef struct TemporarySkeletonInfo
{
	char name[MAX_NUM_CHARS_IN_SKELETON_NAME]; // this is a UTF8 string , NOT an ASCII CHAR array (same base type though)
	uint32_t index; //default = UINT32_MAX;
} TemporarySkeletonInfo;

/// @brief Stores the temporary skeletons available for a specific session.
typedef struct TemporarySkeletonsInfoForSession
{
	uint32_t sessionId; //default = 0;
	char sessionName[MAX_NUM_CHARS_IN_HOST_NAME];
	uint32_t skeletonCount; //default = 0;
	TemporarySkeletonInfo skeletonInfo[MAX_NUMBER_OF_SKELETONS_PER_SESSION];
} TemporarySkeletonsInfoForSession;

/// @brief Stores the temporary skeletons available for a specific session.
typedef struct TemporarySkeletonCountForSession
{
	uint32_t sessionId; //default = 0;
	char sessionName[MAX_NUM_CHARS_IN_HOST_NAME];
	uint32_t skeletonCount; //default = 0;
} TemporarySkeletonCountForSession;

/// @brief Stores the temporary skeleton available for all sessions connected to Core.
typedef struct TemporarySkeletonCountForAllSessions
{
	uint32_t sessionsCount; //default = 0;
	TemporarySkeletonCountForSession temporarySkeletonCountForSessions[MAX_NUMBER_OF_SESSIONS];
} TemporarySkeletonCountForAllSessions;

/// @brief ONLY USED INTERNALLY
/// @brief Stores the temporary skeleton available for all sessions connected to Core.
typedef struct TemporarySkeletonSessionsData
{
	uint32_t sessionsCount; //default = 0;
	TemporarySkeletonsInfoForSession temporarySkeletonsSessions[MAX_NUMBER_OF_SESSIONS];
} TemporarySkeletonSessionsData;

/// @brief Stores the data associated to System messages received from Core.
typedef struct SystemMessage
{
	SystemMessageType type;
	char infoString[MAX_NUM_CHARS_IN_SYSTEM_ERROR_MESSAGE];
	uint32_t infoUInt;
} SystemMessage;

typedef struct NodeInfo
{
	uint32_t nodeId; //default = 0;
	uint32_t parentId; //default = 0;
	ChainType chainType;
	Side side;
	FingerJointType fingerJointType;
}NodeInfo;

// ------------------------------------------------------------------------------------------------------------------------------
// end of Skeleton
// ------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------------
// Glove calibration
// ------------------------------------------------------------------------------------------------------------------------------

typedef struct GloveCalibrationArgs {
	uint32_t gloveId; //default = 0;
} GloveCalibrationArgs;

typedef struct GloveCalibrationStepArgs {
	uint32_t gloveId; //default = 0;
	uint32_t stepIndex; //default = 0;
} GloveCalibrationStepArgs;

/// @brief Stores the data associated to a single calibration step for a specific glove.
/// if the time value is negative, it means that its a continuous step and the value is estimated
typedef struct GloveCalibrationStepData {
	uint32_t index; //default = 0;
	char title[MAX_NUM_CHARS_IN_CALIBRATION_TITLE];
	char description[MAX_NUM_CHARS_IN_CALIBRATION_DESCRIPTION];
	float time; //default = 0
} GloveCalibrationStepData;

// ------------------------------------------------------------------------------------------------------------------------------
// end of Glove calibration
// ------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------------
// Coordinate system settings
// ------------------------------------------------------------------------------------------------------------------------------

/// @brief Stores the information regarding the coordinate system used by the client, defined as VUH (view, up, handedness). 
typedef struct CoordinateSystemVUH
{
	AxisView view;
	AxisPolarity up;
	Side handedness;
	float unitScale; // in meters
} CoordinateSystemVUH;

/// @brief Stores the information regarding the coordinate system used by the client, defined by each axis direction.
typedef struct CoordinateSystemDirection
{
	AxisDirection x;
	AxisDirection y;
	AxisDirection z;
	float unitScale; // in meters
} CoordinateSystemDirection;

// ------------------------------------------------------------------------------------------------------------------------------
// end of Coordinate system settings
// ------------------------------------------------------------------------------------------------------------------------------

// ------------------------------------------------------------------------------------------------------------------------------
// Callbacks
// ------------------------------------------------------------------------------------------------------------------------------

/// @brief Describes the ConnectedToCoreCallback function type
typedef void(*ConnectedToCoreCallback_t)(const ManusHost* const p_Host);
/// @brief Describes the DisconnectedToCoreCallback function type
typedef void(*DisconnectedFromCoreCallback_t)(const ManusHost* const p_Host);
/// @brief Describes the LoggingCallback function type
typedef void(*LoggingCallback_t)(LogSeverity p_Severity, const char* const p_Log, uint32_t p_Length);
/// @brief Describes the SkeletonStreamCallback function type
typedef void(*SkeletonStreamCallback_t)(const SkeletonStreamInfo* const p_SkeletonInfo);
/// @brief Describes the LandscapeStreamCallback function type
typedef void(*LandscapeStreamCallback_t)(const Landscape* const p_Landscape);
/// @brief Describes the TrackerStreamCallback function type
typedef void(*TrackerStreamCallback_t)(const TrackerStreamInfo* const p_TrackerStreamInfo);
/// @brief Describes the ErgonomicsStreamCallback type
typedef void(*ErgonomicsStreamCallback_t)(const ErgonomicsStream* const p_Ergonomics);
/// @brief Describes the SystemStreamCallback function type
typedef void(*SystemStreamCallback_t)(const SystemMessage* const p_SystemMessage);
/// @brief Describes the RawSkeletonStreamCallback function type
typedef void(*RawSkeletonStreamCallback_t)(const SkeletonStreamInfo* const p_SkeletonInfo);
/// @brief Describes the GestureStreamCallback function type
typedef void(*GestureStreamCallback_t)(const GestureStreamInfo* const p_GestureStream);

// ------------------------------------------------------------------------------------------------------------------------------
// end of Callbacks
// ------------------------------------------------------------------------------------------------------------------------------
#ifdef __cplusplus
} // extern "C"
#endif

// Close the Doxygen group.
/** @} */

#endif // #ifndef H_CORESDKWRAPPERTYPES
