#ifndef __MANUS_SDK_H__
#define __MANUS_SDK_H__

// Set up a Doxygen group.
/** @addtogroup ManusSDK
 *  @{
 */

#ifdef __cplusplus
extern "C"
{
#endif

#include "ManusSDKTypes.h"
#include "ManusSDKTypeInitializers.h"

#include <stdint.h>

#ifdef MANUS_SDK_STATIC
#define CORESDK_API
#else
#ifdef _WIN32
	// MANUS_SDK_EXPORTS defines if the functions are exported to or
	// imported from a DLL.
#ifdef MANUS_SDK_EXPORTS
#define CORESDK_API __declspec(dllexport)
#else
#define CORESDK_API __declspec(dllimport)
#endif
#elif defined(__linux__)
#define CORESDK_API
#else
#error Unrecognized platform.
#endif
#endif

	/******************************************************************************
	 * Wrapper startup and shutdown.
	 *****************************************************************************/

	 /// @brief Initialize the wrapper. 
	 /// Call this before using the wrapper.
	 /// @param p_TypeOfSession 
	 /// @param p_Remote If set to true, this is essentially the same as calling the old CoreSdk_Initialize
	 /// @return SDKReturnCode_Success if successful, 
	 /// SDKReturnCode_FunctionCalledAtWrongTime if the function was not intended to be called at this time.
	CORESDK_API SDKReturnCode CoreSdk_Initialize(SessionType p_TypeOfSession, bool p_Remote);

	/// @brief Shut down the wrapper. 
	/// This needs to be called last.
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_ShutDown();

	/******************************************************************************
	 * Utility functions.
	 *****************************************************************************/

	 /// @brief Check if the wrapper DLL was built in the debug configuration.
	 /// @param p_WasBuiltInDebugConfiguration 
	 /// @return SDKReturnCode_Success if successful.
	CORESDK_API SDKReturnCode CoreSdk_WasDllBuiltInDebugConfiguration(bool* p_WasBuiltInDebugConfiguration);

	/// @brief Gets the timestamp info (more readable form of timestamp).
	/// @param p_Timestamp Timestamp to get info from
	/// @param p_Info Info of the timestamp
	/// @return SDKReturnCode_Success if successful.
	CORESDK_API SDKReturnCode CoreSdk_GetTimestampInfo(ManusTimestamp p_Timestamp, ManusTimestampInfo* p_Info);

	/// @brief Sets the timestamp according to the info (more readable form of timestamp).
	/// @param p_Timestamp the Timestamp to set info of
	/// @param p_Info Info to get info from
	/// @return SDKReturnCode_Success if successful.
	CORESDK_API SDKReturnCode CoreSdk_SetTimestampInfo(ManusTimestamp* p_Timestamp, ManusTimestampInfo p_Info);


	/******************************************************************************
	 * Connection handling.
	 *****************************************************************************/

	 /// @brief Start a background task that looks for hosts running Manus Core.
	 /// Call this first when looking for hosts to connect to.
	 /// Underlying function will sleep for p_WaitSeconds to allow servers to reply.
	 /// @param p_WaitSeconds defines the time the function will sleep for to allow servers to reply,
	 /// @param p_LoopbackOnly if true it looks for hosts only locally, if false it looks for hosts anywhere in the network,
	 /// @return SDKReturnCode_Success if successful, 
	 /// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	 /// SDKReturnCode_FunctionCalledAtWrongTime if the function was not intended to be called at this time.
	CORESDK_API SDKReturnCode CoreSdk_LookForHosts(uint32_t p_WaitSeconds = 1, bool p_LoopbackOnly = false);

	/// @brief Get the number of hosts running Manus Core that were found.
	/// This is the second function to call when looking for hosts to connect to.
	/// @param p_NumberOfAvailableHostsFound 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_FunctionCalledAtWrongTime if the function was not intended to be called at this time.
	CORESDK_API SDKReturnCode CoreSdk_GetNumberOfAvailableHostsFound(uint32_t* p_NumberOfAvailableHostsFound);

	/// @brief Fill the given array with information on the hosts that were found.
	/// This is the third and final function to call when looking for hosts to
	/// connect to.
	/// @param p_AvailableHostsFound 
	/// @param p_NumberOfHostsThatFitInArray 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_FunctionCalledAtWrongTime if the function was not intended to be called at this time
	/// SDKReturnCode_ArgumentSizeMismatch if p_NumberOfHostsThatFitInArray does not match the number of available hosts found in the network,
	/// SDKReturnCode_InvalidArgument if the provided p_AvailableHostsFound is a null pointer, 
	/// SDKReturnCode_InternalError if the number of available hosts found is higher than the defined MAX_NUMBER_OF_HOSTS,
	/// SDKReturnCode_UnsupportedStringSizeEncountered if the name of a host is higher than the defined MAX_NUM_CHARS_IN_HOST_NAME,
	/// SDKReturnCode_Error if copying the string representing the host name was not successful,
	/// SDKReturnCode_NullPointer if invalid host data was found.
	CORESDK_API SDKReturnCode CoreSdk_GetAvailableHostsFound(ManusHost* p_AvailableHostsFound, const uint32_t p_NumberOfHostsThatFitInArray);


	/// @brief Check if the wrapper is connected to ManusCore.
	/// @param p_ConnectedToCore 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_GetIsConnectedToCore(bool* p_ConnectedToCore);

	/// @brief Connect to the preset GRPC address.
	/// @return SDKReturnCode_Success
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_ConnectGRPC();

	/// @brief Disconnects from the current host
	/// @return SDKReturnCode_Success
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_Disconnect();

	/// @brief Connect to a host using the given host information.
	/// @param p_Host 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_SdkIsTerminating if the sdk is being terminated,
	/// SDKReturnCode_FunctionCalledAtWrongTime if the function was not intended to be called at this time,
	/// SDKReturnCode_NoCoordinateSystemSet if the user tried to connect without first setting up the coordinate system settings.
	CORESDK_API SDKReturnCode CoreSdk_ConnectToHost(ManusHost p_Host);

	/// @brief Initialize a coordinate system of type CoordinateSystemVUH. (View Up Handedness)
	/// This has to be called before a connection is made.
	/// @param p_CoordinateSystem 
	/// @param p_UseWorldCoordinates if true all data will be defined with respect to a world coordinate system, 
	/// if false all data will be defined with respect to a local one, 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_InitializeCoordinateSystemWithVUH(CoordinateSystemVUH p_CoordinateSystem, bool p_UseWorldCoordinates);

	/// @brief Initialize a coordinate system of type CoordinateSystemDirection.
	/// This has to be called before a connection is made.
	/// @param p_CoordinateSystem 
	/// @param p_UseWorldCoordinates if true all data will be defined with respect to a world coordinate system, 
	/// if false all data will be defined with respect to a local one, 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_InitializeCoordinateSystemWithDirection(CoordinateSystemDirection p_CoordinateSystem, bool p_UseWorldCoordinates);

	/// @brief Get the SDK and Core version used and check if they are compatible with each other. 
	/// @param p_SdkVersion 
	/// @param p_CoreVersion 
	/// @param p_AreVersionsCompatible 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_GetVersionsAndCheckCompatibility(ManusVersion* p_SdkVersion, ManusVersion* p_CoreVersion, bool* p_AreVersionsCompatible);

	/// @brief Get the current session Id.
	/// @param p_SessionId 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_InvalidID if the session id is zero, which means it was not correctly assigned.
	CORESDK_API SDKReturnCode CoreSdk_GetSessionId(uint32_t* p_SessionId);

	/// @brief Register the callback function that is called when manus core gets connected to the sdk.
	/// @param p_ConnectedCallback 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForOnConnect(ConnectedToCoreCallback_t p_ConnectedCallback);

	/// @brief Register the callback function that is called when manus core gets disconnected from the sdk.
	/// @param p_DisconnectedCallback 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForOnDisconnect(DisconnectedFromCoreCallback_t p_DisconnectedCallback);

	/// @brief Register the callback function that is called when something is logged in the SDK.
	/// @param p_LoggingCallback 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForOnLog(LoggingCallback_t p_LoggingCallback);

	/// @brief Register the callback function that is called when skeleton data comes in.
	/// @param p_SkeletonStreamCallback 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForSkeletonStream(SkeletonStreamCallback_t p_SkeletonStreamCallback);

	/// @brief Register the callback function that is called when tracker data comes in.
	/// @param p_TrackerStreamCallback 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForTrackerStream(TrackerStreamCallback_t p_TrackerStreamCallback);

	/// @brief Register the callback function that is called when the landscape data comes in.
	/// @param p_LandscapeStreamCallback 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForLandscapeStream(LandscapeStreamCallback_t p_LandscapeStreamCallback);

	/// @brief Register the callback function that is called when core system related data comes in.
	/// @param p_SystemStreamCallback 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForSystemStream(SystemStreamCallback_t p_SystemStreamCallback);

	/// @brief Register the callback function that is called when ergonomics data comes in.
	/// @param p_ErgonomicsStreamCallback 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForErgonomicsStream(ErgonomicsStreamCallback_t p_ErgonomicsStreamCallback);

	/// @brief Register the callback function that is called when skeleton data comes in.
	/// @param p_SkeletonStreamCallback 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForRawSkeletonStream(RawSkeletonStreamCallback_t p_RawSkeletonStreamCallback);

	/// @brief Register the callback function that is called when gesture data comes in.
	/// @param p_GestureStreamCallback 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_RegisterCallbackForGestureStream(GestureStreamCallback_t p_GestureStreamCallback);

	/******************************************************************************
	 * Basic glove interaction.
	 *****************************************************************************/

	/// @brief Try to pair a glove to a dongle
	/// @param p_GloveID 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_PairGlove(uint32_t p_GloveID, bool* p_Result);

	/// @brief Try to unpair a glove to a dongle
	/// @param p_GloveID 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available.
	CORESDK_API SDKReturnCode CoreSdk_UnpairGlove(uint32_t p_GloveID, bool* p_Result);

	/// @brief Vibrate the motor on the given fingers of a haptic glove.
	/// The order of the fingers is Thumb, Index, Middle, Ring, Pinky.
	/// @param p_DongleId 
	/// @param p_HandType 
	/// @param p_Powers strength of the vibration, should be an array of 5 values
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	/// @deprecated Use CoreSdk_VibrateFingersForGlove or CoreSdk_VibrateFingersForSkeleton instead
	[[deprecated("Use CoreSdk_VibrateFingersForGlove or CoreSdk_VibrateFingersForSkeleton instead")]]
	CORESDK_API SDKReturnCode CoreSdk_VibrateFingers(uint32_t p_DongleId, Side p_HandType, const float* p_Powers);

	/// @brief Vibrate the motor on the given fingers of a haptic glove.
	/// The order of the fingers is Thumb, Index, Middle, Ring, Pinky.
	/// @param p_GloveId 
	/// @param p_Powers strength of the vibration, should be an array of 5 values
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_VibrateFingersForGlove(uint32_t p_GloveId, const float* p_Powers);

	/// @brief Vibrate the motor on the fingers of a haptic glove associated to skeleton with given id.
	/// The order of the fingers is Thumb, Index, Middle, Ring, Pinky.
	/// @param p_SkeletonId 
	/// @param p_HandType 
	/// @param p_Powers strength of the vibration, should be an array of 5 values
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_VibrateFingersForSkeleton(uint32_t p_SkeletonId, Side p_HandType, const float* p_Powers);

	/// @brief Get a glove ID for the given hand of the given user identified by its Id.
	/// @param p_UserId 
	/// @param p_HandType 
	/// @param p_GloveId 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidArgument if the provided p_HandType was neither left nor right,
	/// SDKReturnCode_InvalidID if no user with the provided p_UserId was found in the landscape.
	CORESDK_API SDKReturnCode CoreSdk_GetGloveIdOfUser_UsingUserId(uint32_t p_UserId, Side p_HandType, uint32_t* p_GloveId);

	/// @brief Get the number of gloves that are available.
	/// @param p_NumberOfAvailableGloves 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InternalError if the number of available gloves found is higher than the defined MAX_NUMBER_OF_GLOVES.
	CORESDK_API SDKReturnCode CoreSdk_GetNumberOfAvailableGloves(uint32_t* p_NumberOfAvailableGloves);

	/// @brief Fill the given array with the IDs of all available gloves.
	/// The size of the given array must match the number of available gloves.
	/// Note that the number of available gloves can change at any time.
	/// @param p_IdsOfAvailableGloves 
	/// @param p_NumberOfIdsThatFitInArray 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_ArgumentSizeMismatch if p_NumberOfIdsThatFitInArray does not match with the number of available gloves found,
	/// SDKReturnCode_InvalidArgument if p_NumberOfIdsThatFitInArray is zero or is higher than the defined MAX_NUMBER_OF_GLOVES.
	CORESDK_API SDKReturnCode CoreSdk_GetIdsOfAvailableGloves(uint32_t* p_IdsOfAvailableGloves, uint32_t p_NumberOfIdsThatFitInArray);

	/// @brief Get glove id's for given dongle id.
	/// @param p_DongleId 
	/// @param p_LeftGloveId 
	/// @param p_RightGloveId 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	/// @deprecated Get this through the Landscape instead 
	[[deprecated("Get this through the Landscape instead")]]
	CORESDK_API SDKReturnCode CoreSdk_GetGlovesForDongle(uint32_t p_DongleId, uint32_t* p_LeftGloveId, uint32_t* p_RightGloveId);

	/// @brief Get data for the glove with the given glove ID.
	/// @param p_GloveId 
	/// @param p_GloveData 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_DataNotAvailable if no glove with the provided p_GloveId was found in the landscape.
	/// @deprecated Get this through the Landscape instead
	[[deprecated("Get this through the Landscape instead")]]
	CORESDK_API SDKReturnCode CoreSdk_GetDataForGlove_UsingGloveId(uint32_t p_GloveId, GloveLandscapeData* p_GloveData);

	/// @brief Gets the data for a dongle with the given id.
	/// @param p_DongleId 
	/// @param p_DongleData 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_DataNotAvailable if no dongle with the provided p_DongleId was found in the landscape.
	CORESDK_API SDKReturnCode CoreSdk_GetDataForDongle(uint32_t p_DongleId, DongleLandscapeData* p_DongleData);

	/// @brief Get the number of available dongles.
	/// @param p_NumberOfDongles 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InternalError if the number of dongles found is higher than the defined MAX_NUMBER_OF_DONGLES.
	/// @deprecated Get this through the Landscape instead
	[[deprecated("Get this through the Landscape instead")]]
	CORESDK_API SDKReturnCode CoreSdk_GetNumberOfDongles(uint32_t* p_NumberOfDongles);

	/// @brief Fill the given array with the IDs of all available dongles.
	/// The size of the given array must match the number of available dongles.
	/// Note that the number of available  dongles can change at any time.
	/// @param p_DongleIds 
	/// @param p_NumberOfIdsThatFitInArray 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_ArgumentSizeMismatch if p_NumberOfIdsThatFitInArray does not match with the number of dongles found,
	/// SDKReturnCode_InvalidArgument if p_NumberOfIdsThatFitInArray is zero or higher than the defined MAX_NUMBER_OF_HAPTICS_DONGLES.
	/// @deprecated Get this through the Landscape instead
	[[deprecated("Get this through the Landscape instead")]]
	CORESDK_API SDKReturnCode CoreSdk_GetDongleIds(uint32_t* p_DongleIds, uint32_t p_NumberOfIdsThatFitInArray);

	/******************************************************************************
	 * Haptics module.
	 *****************************************************************************/

	 /// @brief Get the number of available haptics dongles.
	 /// @param p_NumberOfHapticsDongles 
	 /// @return SDKReturnCode_Success if successful, 
	 /// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	 /// SDKReturnCode_NotConnected if there is no connection to core,
	 /// SDKReturnCode_InternalError if the number of haptic dongles found is higher than the defined MAX_NUMBER_OF_HAPTICS_DONGLES.
	 /// @deprecated Get this through the Landscape instead
	[[deprecated("Get this through the Landscape instead")]]
	CORESDK_API SDKReturnCode CoreSdk_GetNumberOfHapticsDongles(uint32_t* p_NumberOfHapticsDongles);

	/// @brief Fill the given array with the IDs of all available haptics dongles.
	/// The size of the given array must match the number of available haptics dongles.
	/// Note that the number of available haptics dongles can change at any time.
	/// @param p_HapticsDongleIds 
	/// @param p_NumberOfIdsThatFitInArray 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_ArgumentSizeMismatch if p_NumberOfIdsThatFitInArray does not match with the number of haptic dongles found, 
	/// SDKReturnCode_InvalidArgument if p_NumberOfIdsThatFitInArray is zero or higher than the defined MAX_NUMBER_OF_HAPTICS_DONGLES.
	/// @deprecated Get this through the Landscape instead
	[[deprecated("Get this through the Landscape instead")]]
	CORESDK_API SDKReturnCode CoreSdk_GetHapticsDongleIds(uint32_t* p_HapticsDongleIds, uint32_t p_NumberOfIdsThatFitInArray);

	/******************************************************************************
	 * Users.
	 *****************************************************************************/

	 /// @brief Get the number of available users.
	 /// Note that this is reliant on the landscape, and the first second after connecting the client to the core it will not yet have landscape
	 /// data, so the number of users wil be 0. Since there will always be a default users, just wait until the count is above 0 and then proceed.
	 /// @param p_NumberOfAvailableUsers
	 /// @return SDKReturnCode_Success if successful, 
	 /// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	 /// SDKReturnCode_NotConnected if there is no connection to core,
	 /// SDKReturnCode_InternalError if the number of available users found is higher than the defined MAX_NUMBER_OF_USERS.
	CORESDK_API SDKReturnCode CoreSdk_GetNumberOfAvailableUsers(uint32_t* p_NumberOfAvailableUsers);

	/// @brief Fill the given array with the IDs of all available users.
	/// The size of the given array must match the number of available users.
	/// Note that the number of available users can change at any time depending on the landscape.
	/// @param p_IdsOfAvailableUsers, The array of ids that will get filled
	/// @param p_NumberOfIdsThatFitInArray 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidArgument if p_NumberOfIdsThatFitInArray is null or higher than the defined MAX_NUMBER_OF_USERS, or 
	/// if p_IdsOfAvailableUsers is a null array.
	/// SDKReturnCode_ArgumentSizeMismatch if p_NumberOfIdsThatFitInArray does not match with number of available users found.
	CORESDK_API SDKReturnCode CoreSdk_GetIdsOfAvailableUsers(uint32_t* p_IdsOfAvailableUsers, uint32_t p_NumberOfIdsThatFitInArray);

	/******************************************************************************
	 * Tracking.
	 *****************************************************************************/

	 /// @brief Get data for a tracker at a given index in the TrackerStream callback.
	 /// @param p_TrackerIndex 
	 /// @param p_TrackerData 
	 /// @return SDKReturnCode_Success if successful, 
	 /// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	 /// SDKReturnCode_NotConnected if there is no connection to core,
	 /// SDKReturnCode_ArgumentSizeMismatch if p_TrackerIndex is higher than the total number of trackers received from the tracker stream. 
	CORESDK_API SDKReturnCode CoreSdk_GetTrackerData(uint32_t p_TrackerIndex, TrackerData* p_TrackerData);

	/// @brief Get the number of available trackers.
	/// @param p_NumberOfAvailableTrackers 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InternalError if the number of trackers found is higher than the defined MAX_NUMBER_OF_TRACKERS.
	CORESDK_API SDKReturnCode CoreSdk_GetNumberOfAvailableTrackers(uint32_t* p_NumberOfAvailableTrackers);

	/// @brief Fill the given array with the IDs of available trackers.
	/// The size of the given array must match the number of available trackers.
	/// Note that the number of available trackers can change at any time.
	/// @param p_IdsOfAvailableTrackers 
	/// @param p_NumberOfIdsThatFitInArray 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidArgument if p_NumberOfIdsThatFitInArray is zero or higher than the defined MAX_NUMBER_OF_TRACKERS, or
	/// if p_IdsOfAvailableTrackers is a null array.
	/// SDKReturnCode_ArgumentSizeMismatch if p_NumberOfIdsThatFitInArray does not match with the number of available trackers found.
	CORESDK_API SDKReturnCode CoreSdk_GetIdsOfAvailableTrackers(TrackerId* p_IdsOfAvailableTrackers, uint32_t p_NumberOfIdsThatFitInArray);

	/// @brief Get the number of available trackers for a user with given Id.
	/// @param p_NumberOfAvailableTrackers 
	/// @param p_UserId 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InternalError if the number of trackers found is higher than the defined MAX_NUMBER_OF_TRACKERS.
	CORESDK_API SDKReturnCode CoreSdk_GetNumberOfAvailableTrackersForUserId(uint32_t* p_NumberOfAvailableTrackers, uint32_t p_UserId);

	/// @brief Fill the given array with the IDs of available trackers for a user with given Id.
	/// The size of the given array must match the number of available trackers.
	/// Note that the number of available trackers can change at any time.
	/// @param p_IdsOfAvailableTrackers 
	/// @param p_UserId 
	/// @param p_NumberOfIdsThatFitInArray 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidArgument if p_NumberOfIdsThatFitInArray is zero or higher than the defined MAX_NUMBER_OF_TRACKERS, or
	/// if p_IdsOfAvailableTrackers is a null array.
	/// SDKReturnCode_ArgumentSizeMismatch if p_NumberOfIdsThatFitInArray does not match with the number of available trackers found.
	CORESDK_API SDKReturnCode CoreSdk_GetIdsOfAvailableTrackersForUserId(TrackerId* p_IdsOfAvailableTrackers, uint32_t p_UserId, uint32_t p_NumberOfIdsThatFitInArray);

	/// @brief Get the number of available trackers for a user with given Id.
	/// @param p_NumberOfAvailableTrackers 
	/// @param p_UserIndex 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InternalError if the number of trackers found is higher than the defined MAX_NUMBER_OF_TRACKERS.
	CORESDK_API SDKReturnCode CoreSdk_GetNumberOfAvailableTrackersForUserIndex(uint32_t* p_NumberOfAvailableTrackers, uint32_t p_UserIndex);

	/// @brief Fill the given array with the IDs of available trackers for a user with given Id.
	/// The size of the given array must match the number of available trackers.
	/// Note that the number of available trackers can change at any time.
	/// @param p_IdsOfAvailableTrackers 
	/// @param p_UserIndex 
	/// @param p_NumberOfIdsThatFitInArray 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidArgument if p_NumberOfIdsThatFitInArray is zero or higher than the defined MAX_NUMBER_OF_TRACKERS, or
	/// if p_IdsOfAvailableTrackers is a null array.
	/// SDKReturnCode_ArgumentSizeMismatch if p_NumberOfIdsThatFitInArray does not match with the number of available trackers found.
	CORESDK_API SDKReturnCode CoreSdk_GetIdsOfAvailableTrackersForUserIndex(TrackerId* p_IdsOfAvailableTrackers, uint32_t p_UserIndex, uint32_t p_NumberOfIdsThatFitInArray);



	/// @brief Get data for the tracker with the given tracker Id.
	/// @param p_TrackerId 
	/// @param p_TrackerData 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_DataNotAvailable if no tracker with the given p_TrackerId was found in the landscape.
	CORESDK_API SDKReturnCode CoreSdk_GetDataForTracker_UsingTrackerId(TrackerId p_TrackerId, TrackerData* p_TrackerData);

	/// @brief Get data for the tracker with the given user id and type.
	/// @param p_UserId 
	/// @param p_TrackerType 
	/// @param p_TrackerData  
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_DataNotAvailable if no tracker with the given p_TrackerId and the given p_TrackerType was found in the landscape.
	CORESDK_API SDKReturnCode CoreSdk_GetDataForTracker_UsingIdAndType(uint32_t p_UserId, uint32_t p_TrackerType, TrackerData* p_TrackerData);

	/// @brief Send data to Core for a tracker.
	/// @param p_TrackerData 
	/// @param p_NumberOfTrackers amount of trackers for which we send the information
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_NullPointer if the provided p_TrackerData is a null pointer,
	/// SDKReturnCode_InvalidArgument if p_NumberOfTrackers is null or higher than the defined MAX_NUMBER_OF_TRACKERS.
	CORESDK_API SDKReturnCode CoreSdk_SendDataForTrackers(const TrackerData* p_TrackerData, uint32_t p_NumberOfTrackers);

	/// @brief Sets the offset of a tracker for a specified user
	/// @param p_UserId
	/// @param p_TrackerOffset
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_SetTrackerOffset(uint32_t p_UserId, const TrackerOffset* p_TrackerOffset);

	/******************************************************************************
	 * Gestures.
	 *****************************************************************************/

	 /// @brief Get the landscape data for all the gestures in the latest landscape.
	 /// @param p_LandscapeDataArray 
	 /// @param p_ArraySize 
	 /// @return SDKReturnCode_Success if successful, 
	 /// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	 /// SDKReturnCode_NotConnected if there is no connection to core,
	 /// SDKReturnCode_ArgumentSizeMismatch if p_ArraySize is higher than the total number of gestures in the landscape stream. 
	CORESDK_API SDKReturnCode CoreSdk_GetGestureLandscapeData(GestureLandscapeData* p_LandscapeDataArray, uint32_t p_ArraySize);

	/// @brief The start data index allows you to get more data if there are more than MAX_GESTURE_DATA_CHUNK_SIZE gestures.
	 /// @param p_GestureStreamDataIndex 
	 /// @param p_StartDataIndex 
	 /// @param p_GestureProbabilitiesCollection 
	 /// @return SDKReturnCode_Success if successful, or if there is no more data to get
	 /// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	 /// SDKReturnCode_NotConnected if there is no connection to core,
	 /// SDKReturnCode_ArgumentSizeMismatch if p_ArraySize is higher than the total number of gestures in the landscape stream. 
	CORESDK_API SDKReturnCode CoreSdk_GetGestureStreamData(uint32_t p_GestureStreamDataIndex, uint32_t p_StartDataIndex, GestureProbabilities* p_GestureProbabilitiesCollection);

	/******************************************************************************
	 * Glove Calibration.
	 *****************************************************************************/

	/// @brief Start the calibration for a glove with the id that is specified in the arguments.
	/// @param p_CalibrationArgs 
	/// @param p_Result 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core
	CORESDK_API SDKReturnCode CoreSdk_GloveCalibrationStart(GloveCalibrationArgs p_CalibrationArgs, bool* p_Result);

	/// @brief Stop the calibration for a glove with the id that is specified in the arguments.
	/// @param p_CalibrationArgs 
	/// @param p_Result 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core
	CORESDK_API SDKReturnCode CoreSdk_GloveCalibrationStop(GloveCalibrationArgs p_CalibrationArgs, bool* p_Result);
	
	/// @brief Finish the calibration for a glove with the id that is specified in the arguments.
	/// @param p_CalibrationArgs 
	/// @param p_Result 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core
	CORESDK_API SDKReturnCode CoreSdk_GloveCalibrationFinish(GloveCalibrationArgs p_CalibrationArgs, bool* p_Result);


	/// @brief Get the number of steps in a glove calibration sequence.
	/// @param p_CalibrationArgs 
	/// @param p_NumberOfSteps 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core
	CORESDK_API SDKReturnCode CoreSdk_GloveCalibrationGetNumberOfSteps(GloveCalibrationArgs p_CalibrationArgs, uint32_t* p_NumberOfSteps);

	/// @brief Get information data about a specific calibration step for a glove.
	/// @param p_CalibrationStepArgs 
	/// @param p_Data 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core
	CORESDK_API SDKReturnCode CoreSdk_GloveCalibrationGetStepData(GloveCalibrationStepArgs p_CalibrationStepArgs, GloveCalibrationStepData* p_Data);

	/// @brief Start a specific calibration step for a glove.
	/// @param p_CalibrationStepArgs 
	/// @param p_Result 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core
	CORESDK_API SDKReturnCode CoreSdk_GloveCalibrationStartStep(GloveCalibrationStepArgs p_CalibrationStepArgs, bool* p_Result);


	CORESDK_API SDKReturnCode CoreSdk_GetGloveCalibrationSize(uint32_t p_GloveId, uint32_t* p_Size);

	CORESDK_API SDKReturnCode CoreSdk_GetGloveCalibration(unsigned char* p_CalibrationBytes, uint32_t p_BytesLength);

	CORESDK_API SDKReturnCode CoreSdk_SetGloveCalibration(uint32_t p_GloveId, unsigned char* p_CalibrationBytes, SetGloveCalibrationReturnCode* p_Result);

	/******************************************************************************
	 * Skeletal System.
	 *****************************************************************************/

	/// @brief Get information about the final animated skeleton with given index.
	/// @param p_SkeletonIndex 
	/// @param p_Info 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_ArgumentSizeMismatch if p_SkeletonIndex is higher than the total number of skeletons received from the skeleton stream.
	CORESDK_API SDKReturnCode CoreSdk_GetSkeletonInfo(uint32_t p_SkeletonIndex, SkeletonInfo* p_Info);

	/// @brief Get data for the final animated skeleton with given index.
	/// The size of the given array must match the nodes count.
	/// Note that the nodes count can change at any time. 
	/// @param p_SkeletonIndex 
	/// @param p_Nodes 
	/// @param p_NodeCount 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_ArgumentSizeMismatch if p_SkeletonIndex is higher than the total number of skeletons received from the skeleton stream or 
	/// if the provided p_NodeCount is different from the node count of that skeleton. 
	CORESDK_API SDKReturnCode CoreSdk_GetSkeletonData(uint32_t p_SkeletonIndex, SkeletonNode* p_Nodes, uint32_t p_NodeCount);

	/// @brief Determine how the raw data should be moving
	/// @param p_HandMotion 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	CORESDK_API SDKReturnCode CoreSdk_SetRawSkeletonHandMotion(HandMotion p_HandMotion);

	/// @brief Get the raw skeleton hand motion type
	/// @param p_HandMotion 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	CORESDK_API SDKReturnCode CoreSdk_GetRawSkeletonHandMotion(HandMotion* p_HandMotion);

	/// @brief Get information about the raw skeleton data with given index.
	 /// @param p_SkeletonIndex 
	 /// @param p_Info 
	 /// @return SDKReturnCode_Success if successful, 
	 /// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	 /// SDKReturnCode_NotConnected if there is no connection to core,
	 /// SDKReturnCode_ArgumentSizeMismatch if p_SkeletonIndex is higher than the total number of skeletons received from the skeleton stream.
	CORESDK_API SDKReturnCode CoreSdk_GetRawSkeletonInfo(uint32_t p_SkeletonIndex, RawSkeletonInfo* p_Info);

	/// @brief Get data for the raw skeleton with given index.
	/// The size of the given array must match the nodes count.
	/// Note that the nodes count can change at any time. 
	/// @param p_SkeletonIndex 
	/// @param p_Nodes 
	/// @param p_NodeCount 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_ArgumentSizeMismatch if p_SkeletonIndex is higher than the total number of skeletons received from the skeleton stream or 
	/// if the provided p_NodeCount is different from the node count of that skeleton. 
	CORESDK_API SDKReturnCode CoreSdk_GetRawSkeletonData(uint32_t p_SkeletonIndex, SkeletonNode* p_Nodes, uint32_t p_NodeCount);

	/// @brief Identifies if the glove associated to the skeleton with given id and side is a haptic one.
	/// @param p_SkeletonId 
	/// @param p_HandType 
	/// @param p_IsHaptics 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_DataNotAvailable if the requested glove is not found or if the skeleton does not have any user associated to it,
	/// SDKReturnCode_InvalidArgument if the provided p_HandType is neither Left nor Right.
	CORESDK_API SDKReturnCode CoreSdk_DoesSkeletonGloveSupportHaptics(uint32_t p_SkeletonId, Side p_HandType, bool* p_IsHaptics);
	/******************************************************************************
	* Skeletal System Setup.
	 *****************************************************************************/

	 /// @brief Create a new SkeletonSetup with the given information and returns the index on which it is saved.
	 /// @param p_Skeleton The SkeletonSetup information used by this skeleton.
	 /// @param p_SkeletonSetupIndex The index which is used by this SkeletonSetup.
	 /// @return SDKReturnCode_Success if successful, 
	 /// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	 /// SDKReturnCode_NotConnected if there is no connection to core,
	 /// SDKReturnCode_ArgumentSizeMismatch if the maximum number of temporary skeletons per session has been reached,
	 /// when this happens the user can either load or clear a temporary skeleton in order to create a new skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_CreateSkeletonSetup(SkeletonSetupInfo p_Skeleton, uint32_t* p_SkeletonSetupIndex);

	/// @brief Add a node to a SkeletonSetup at a given index.
	/// @param p_SkeletonSetupIndex 
	/// @param p_Node 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_AddNodeToSkeletonSetup(uint32_t p_SkeletonSetupIndex, NodeSetup p_Node);

	/// @brief Add a chain to a SkeletonSetup at a given index. 
	/// @param p_SkeletonSetupIndex 
	/// @param p_Chain 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_AddChainToSkeletonSetup(uint32_t p_SkeletonSetupIndex, ChainSetup p_Chain);

	/// @brief Add a collider to a SkeletonSetup at a given index. 
	/// @param p_SkeletonSetupIndex 
	/// @param p_Collider 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_AddColliderToSkeletonSetup(uint32_t p_SkeletonSetupIndex, ColliderSetup p_Collider);

	/// @brief Add a mesh setup to a SkeletonSetup at a given index. 
	/// @param p_SkeletonSetupIndex 
	/// @param p_NodeID 
	/// @param p_MeshSetupIndex 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_AddMeshSetupToSkeletonSetup(uint32_t p_SkeletonSetupIndex, uint32_t p_NodeID, uint32_t* p_MeshSetupIndex);

	/// @brief Add a Vertex to a mesh setup at a given index. 
	/// @param p_SkeletonSetupIndex 
	/// @param p_MeshSetupIndex 
	/// @param p_Vertex 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_AddVertexToMeshSetup(uint32_t p_SkeletonSetupIndex, uint32_t p_MeshSetupIndex, Vertex p_Vertex);


	/// @brief Add a Triangle to a mesh setup at a given index. 
	/// @param p_SkeletonSetupIndex 
	/// @param p_MeshSetupIndex 
	/// @param p_Triangle 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_AddTriangleToMeshSetup(uint32_t p_SkeletonSetupIndex, uint32_t p_MeshSetupIndex, Triangle p_Triangle);

	/// @brief Overwrite an existing SkeletonSetup at a given index.
	/// This will remove all chains and nodes from the current skeleton at this index.
	/// @param p_SkeletonSetupIndex 
	/// @param p_SkeletonSetup 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup,
	/// SDKReturnCode_NullPointer if the skeleton setup was not correctly saved, thus the new skeleton setup is a null pointer.
	CORESDK_API SDKReturnCode CoreSdk_OverwriteSkeletonSetup(uint32_t p_SkeletonSetupIndex, SkeletonSetupInfo p_SkeletonSetup);

	/// @brief Overwrite a chain in a SkeletonSetup, the chain to be overwritten is idenfified by p_Chain.id. 
	/// @param p_SkeletonSetupIndex The SkeletonSetup's index, not to be confused by the Chain's ID.
	/// @param p_Chain The chain to replace the matching (via id) chain with.
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_OverwriteChainToSkeletonSetup(uint32_t p_SkeletonSetupIndex, ChainSetup p_Chain);

	/// @brief Overwrite a node in a SkeletonSetup, the node to be overwritten is identified by p_Node.id. 
	/// @param p_SkeletonSetupIndex The SkeletonSetup's index, not to be confused by the Chain's ID. 
	/// @param p_Node The node to replace the matching (via id) node with.
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_OverwriteNodeToSkeletonSetup(uint32_t p_SkeletonSetupIndex, NodeSetup p_Node);

	/// @brief Get the skeleton info (number of nodes and chains) for a SkeletonSetup at a given index.
	/// @param p_SkeletonSetupIndex 
	/// @param p_SkeletonInfo 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_GetSkeletonSetupArraySizes(uint32_t p_SkeletonSetupIndex, SkeletonSetupArraySizes* p_SkeletonInfo);

	/// @brief Allocate chains for a SkeletonSetup at a given index.
	/// @param p_SkeletonSetupIndex 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_AllocateChainsForSkeletonSetup(uint32_t p_SkeletonSetupIndex);
	CORESDK_API SDKReturnCode CoreSdk_PrepareSkeletonSetup(uint32_t p_SkeletonSetupIndex);

	/// @brief Get setup info for a SkeletonSetup at a given index.
	/// @param p_SkeletonSetupIndex 
	/// @param p_SDK 
	/// @return SDKReturnCode_Success if successful,
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// /// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_GetSkeletonSetupInfo(uint32_t p_SkeletonSetupIndex, SkeletonSetupInfo* p_SDK);

	/// @brief Get setup chains for a SkeletonSetup at a given index.
	/// The size of the given array must match the chains count.
	/// Note that the chains count can change at any time. 
	/// @param p_SkeletonSetupIndex 
	/// @param p_SDK 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_NullPointer if the provided p_SDK (used to store the setup chains) is a null pointer, it should be initialized to
	/// an array with size equal to the chains count,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	/// @deprecated Use GetSkeletonSetupChainsArray() instead
	[[deprecated ("Use GetSkeletonSetupChainsArray() instead")]]
	CORESDK_API SDKReturnCode CoreSdk_GetSkeletonSetupChains(uint32_t p_SkeletonSetupIndex, ChainSetup* p_SDK);

	/// @brief Get setup chains for a SkeletonSetup at a given index.
	/// The size of the given array must match the chains count.
	/// Note that the chains count can change at any time. 
	/// @param p_SkeletonSetupIndex 
	/// @param p_ChainSetupArray
	/// @param p_ArraySize
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_NullPointer if the provided p_ChainSetupArray (used to store the setup chains) is a null pointer, it should be initialized to
	/// an array with size equal to the chains count,
	/// SDKReturnCode_ArgumentSizeMismatch if the provided p_ArraySize is not equal to the chains count,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_GetSkeletonSetupChainsArray(uint32_t p_SkeletonSetupIndex, ChainSetup* p_ChainSetupArray, uint32_t p_ArraySize);

	/// @brief Get setup nodes for a SkeletonSetup at a given index.
	/// The size of the given array must match the nodes count.
	/// Note that the nodes count can change at any time.
	/// @param p_SkeletonSetupIndex 
	/// @param p_SDK 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_NullPointer if the provided p_SDK (used to store the setup nodes) is a null pointer, it should be initialized to
	/// an array with size equal to the nodes count,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	/// @deprecated Use GetSkeletonSetupNodesArray() instead 
	[[deprecated ("Use GetSkeletonSetupNodesArray() instead")]]
	CORESDK_API SDKReturnCode CoreSdk_GetSkeletonSetupNodes(uint32_t p_SkeletonSetupIndex, NodeSetup* p_SDK);

	/// @brief Get setup nodes for a SkeletonSetup at a given index.
	/// The size of the given array must match the nodes count.
	/// Note that the nodes count can change at any time.
	/// @param p_SkeletonSetupIndex 
	/// @param p_NodeSetupArray
	/// @param p_ArraySize
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_NullPointer if the provided p_SDK (used to store the setup nodes) is a null pointer, it should be initialized to
	/// an array with size equal to the nodes count,
	/// SDKReturnCode_ArgumentSizeMismatch if the provided p_ArraySize is not equal to the nodes count, 
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_GetSkeletonSetupNodesArray(uint32_t p_SkeletonSetupIndex, NodeSetup* p_NodeSetupArray, uint32_t p_ArraySize);

	/// @brief Get setup colliders for a SkeletonSetup at a given index.
	/// The size of the given array must match the colliders count.
	/// Note that the colliders count can change at any time.
	/// @param p_SkeletonSetupIndex 
	/// @param p_SDK 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_NullPointer if the provided p_SDK (used to store the setup colliders) is a null pointer, it should be initialized to
	/// an array with size equal to the colliders count,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	/// @deprecated 
	[[deprecated]]
	CORESDK_API SDKReturnCode CoreSdk_GetSkeletonSetupColliders(uint32_t p_SkeletonSetupIndex, ColliderSetup* p_SDK);

	/// @brief Get the amount of nodes for the raw skeleton with given id
	/// @param p_GloveId 
	/// @param p_NodeCount
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided glove id does not match any skeleton in core
	CORESDK_API SDKReturnCode CoreSdk_GetRawSkeletonNodeCount(uint32_t p_GloveId, uint32_t& p_NodeCount);

	/// @brief Get the information for nodes of the raw skeleton with given id 
	/// The size of the given array p_NodeInfo must match the node count retrieved from CoreSdk_GetRawSkeletonNodeCount.
	/// @param p_GloveId 
	/// @param p_NodeInfo
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided glove id does not match any skeleton in core
	/// @deprecated Use CoreSdk_GetRawSkeletonNodeInfoArray instead
	[[deprecated("Use CoreSdk_GetRawSkeletonNodeInfoArray instead")]]
	CORESDK_API SDKReturnCode CoreSdk_GetRawSkeletonNodeInfo(uint32_t p_GloveId, NodeInfo* p_NodeInfo);


	/// @brief Get the information for nodes of the raw skeleton with given id 
	/// The size of the given array p_NodeInfo and p_NodeCount must match the node count retrieved from CoreSdk_GetRawSkeletonNodeCount.
	/// @param p_GloveId 
	/// @param p_NodeInfoArray
	/// @param p_ArraySize 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_InvalidID if the provided glove id does not match any skeleton in core,
	/// SDKReturnCode_ArgumentSizeMismatch if p_ArraySize does not match node count retrieved from CoreSdk_GetRawSkeletonNodeCount.
	CORESDK_API SDKReturnCode CoreSdk_GetRawSkeletonNodeInfoArray(uint32_t p_GloveId, NodeInfo* p_NodeInfoArray, uint32_t p_ArraySize);

	/// @brief Sends a skeleton setup to core to become a skeleton upon which data is applied.
	/// Returns the skeleton ID of the loaded skeleton, this ID is used to identify the skeleton in the SkeletonStreamCallback.
	/// Loading a skeleton into core for retargetting removes it from the temporary skeletons array, thus causing
	/// the p_SkeletonSetupIndex to no longer be valid after this call.
	/// @param p_SkeletonSetupIndex The SkeletonSetup's index, will become invalid after this function.
	/// @param p_SkeletonId This is the ID of the loaded skeleton.
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_LoadSkeleton(uint32_t p_SkeletonSetupIndex, uint32_t* p_SkeletonId);

	/// @brief Unload a skeleton with a certain ID from Manus Core, so data will no longer be generated for it.
	/// @param p_SkeletonId The Skeleton's ID which is assigned by Manus Core.
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_UnloadSkeleton(uint32_t p_SkeletonId);

	/// @brief Save a SkeletonSetup at a given index and sessionId into Manus Core for use in the Dev Tools.
	/// p_IsSkeletonModified is intended to be used mainly by the Dev Tools, it is set to true after saving any modification to the skeleton, 
	/// this triggers the OnSyStemCallback which is used in the SDK to be notified about a change to its temporary skeletons.
	/// @param p_SkeletonSetupIndex 
	/// @param p_SessionId 
	/// @param p_IsSkeletonModified 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_SaveTemporarySkeleton(uint32_t p_SkeletonSetupIndex, uint32_t p_SessionId, bool p_IsSkeletonModified);

	/// @brief Send a temporary skeleton to Core to compress it (convert it into an array of bytes) and then get the compressed data back in the SDK, 
	/// the skeleton compressed data are stored internally. It returns the length of the bytes array for correct memory allocation.
	/// This function must be called before CoreSdk_GetCompressedTemporarySkeletonData.
	/// @param p_SkeletonSetupIndex 
	/// @param p_SessionId 
	/// @param p_TemporarySkeletonLengthInBytes 
	/// @return 
	CORESDK_API SDKReturnCode CoreSdk_CompressTemporarySkeletonAndGetSize(uint32_t p_SkeletonSetupIndex, uint32_t p_SessionId, uint32_t* p_TemporarySkeletonLengthInBytes);

	/// @brief Get the compressed temporary skeleton data which are stored internally when calling function CoreSdk_CompressTemporarySkeletonAndGetSize.
	/// The size of p_TemporarySkeletonData must match with p_TemporarySkeletonLengthInBytes.
	/// @param p_TemporarySkeletonData 
	/// @param p_TemporarySkeletonLengthInBytes 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_GetCompressedTemporarySkeletonData(unsigned char* p_TemporarySkeletonData, uint32_t p_TemporarySkeletonLengthInBytes);

	/// @brief Get a SkeletonSetup at a given index and sessionId from Manus Core.
	/// This function does NOT return a loaded skeleton!
	/// @param p_SkeletonSetupIndex 
	/// @param p_SessionId 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	CORESDK_API SDKReturnCode CoreSdk_GetTemporarySkeleton(uint32_t p_SkeletonSetupIndex, uint32_t p_SessionId);

	/// @brief Send the compressed data (an array of bytes) retrieved from a file to Core to be decompressed and converted back into a temporary skeleton.
	/// The received temporary skeleton is saved internally with the given index and to the given session id.
	/// The size of p_TemporarySkeletonData must match with p_TemporarySkeletonLengthInBytes.
	/// @param p_SkeletonSetupIndex 
	/// @param p_SessionId 
	/// @param p_TemporarySkeletonData
	/// @param p_TemporarySkeletonLengthInBytes
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	CORESDK_API SDKReturnCode CoreSdk_GetTemporarySkeletonFromCompressedData(uint32_t p_SkeletonSetupIndex, uint32_t p_SessionId, unsigned char* p_TemporarySkeletonData, uint32_t p_TemporarySkeletonLengthInBytes);

	/// @brief Clear a SkeletonSetup at a given index and sessionId in Manus Core and the SDK.
	/// The given setup index will no longer be valid after this call.
	/// @param p_SkeletonSetupIndex 
	/// @param p_SessionId 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_InvalidID if the provided p_SkeletonSetupIndex does not correspond to any skeleton setup.
	CORESDK_API SDKReturnCode CoreSdk_ClearTemporarySkeleton(uint32_t p_SkeletonSetupIndex, uint32_t p_SessionId);

	/// @brief Clear all temporary skeletons associated to the current session both in the sdk and core.
	/// All skeleton setups will no longer be valid after this call.
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_NotConnected if there is no connection to core,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	CORESDK_API SDKReturnCode CoreSdk_ClearAllTemporarySkeletons();

	/// @brief Get temporary skeleton counts for all sessions connected to Core.
	/// @param p_temporarySkeletonSessionsData 
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_GetTemporarySkeletonCountForAllSessions(TemporarySkeletonCountForAllSessions* p_TemporarySkeletonCountForAllSessions);

	/// @brief Get information (name and index) for the SkeletonSetups of the specified session.
	/// @param p_SessionId
	/// @param p_temporarySkeletonSessionsData
	/// @return SDKReturnCode_Success if successful, 
	/// SDKReturnCode_SdkNotAvailable if the Core SDK is not available,
	/// SDKReturnCode_StubNullPointer if the stub has been reset but someone is trying to use it anyway. This usually happens after a shutdown of the SDK,
	/// SDKReturnCode_NotConnected if there is no connection to core.
	CORESDK_API SDKReturnCode CoreSdk_GetTemporarySkeletonsInfoForSession(uint32_t p_SessionId, TemporarySkeletonsInfoForSession* p_TemporarySkeletonsInfoForSession);

#ifdef __cplusplus
} // extern "C"
#endif

// Close the Doxygen group.
/** @} */

#endif // #ifndef H_MANUS_SDK
