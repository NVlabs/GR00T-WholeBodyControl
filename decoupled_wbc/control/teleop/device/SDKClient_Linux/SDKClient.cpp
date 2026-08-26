#include "SDKClient.hpp"
#include "ManusSDKTypes.h"
#include "ClientLogging.hpp"
#include <fstream>
#include <iostream>
#include <thread>
#include <map>
#include <vector>
#include <algorithm>

#include "ClientPlatformSpecific.hpp"

#define GO_TO_DISPLAY(p_Key,p_Function) if (GetKeyDown(p_Key)) { ClearConsole();\
		m_CurrentInteraction = std::bind(&SDKClient::p_Function, this); return ClientReturnCode::ClientReturnCode_Success;}

#define GO_TO_MENU_IF_REQUESTED() if (GetKeyDown('Q')) { ClearConsole();\
		m_CurrentInteraction = nullptr; return ClientReturnCode::ClientReturnCode_Success;}

using ManusSDK::ClientLog;

std::map<std::string, std::vector<double>> output_map;

SDKClient* SDKClient::s_Instance = nullptr;

std::vector<long int> idleft={3822396207, 3998055887};
std::vector<long int> idright={3762867141, 831307785};

SDKClient::SDKClient()
{
	s_Instance = this;

	// using initializers like these ensure that the data is set to its default values.
	ErgonomicsData_Init(&m_LeftGloveErgoData);
	ErgonomicsData_Init(&m_RightGloveErgoData);

	TestTimestamp();
}

SDKClient::~SDKClient()
{
	s_Instance = nullptr;
}

/// @brief Initialize the sample console and the SDK.
/// This function attempts to resize the console window and then proceeds to initialize the SDK's interface.
ClientReturnCode SDKClient::Initialize()
{
	// if (!PlatformSpecificInitialization())
	// {
	// 	return ClientReturnCode::ClientReturnCode_FailedPlatformSpecificInitialization;
	// }

	const ClientReturnCode t_IntializeResult = InitializeSDK();
	if (t_IntializeResult != ClientReturnCode::ClientReturnCode_Success)
	{
		ClientLog::error("Failed to initialize the SDK. Are you sure the correct ManusSDKLibary is used?");
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief The main SDKClient loop.
/// This is a simple state machine which switches between different substates.
ClientReturnCode SDKClient::Run()
{
	ConnectingToCore();

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief When you are done with the SDK, don't forget to nicely shut it down
/// this will close all connections to the host, close any threads and clean up after itself
/// after this is called it is expected to exit the client program. If not it needs to call initialize again.
ClientReturnCode SDKClient::ShutDown()
{
	const SDKReturnCode t_Result = CoreSdk_ShutDown();
	if (t_Result != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to shut down the SDK wrapper. The value returned was {}.", (int32_t)t_Result);
		return ClientReturnCode::ClientReturnCode_FailedToShutDownSDK;
	}

	if (!PlatformSpecificShutdown())
	{
		return ClientReturnCode::ClientReturnCode_FailedPlatformSpecificShutdown;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Gets called when the client is connects to manus core
/// Using this callback is optional.
/// In this sample we use this callback to change the client's state and switch to another screen
void SDKClient::OnConnectedCallback(const ManusHost* const p_Host)
{
	ClientLog::print("Connected to manus core.");

	//No need to initialize these as they get filled in the CoreSdk_GetVersionsAndCheckCompatibility
	ManusVersion t_SdkVersion;
	ManusVersion t_CoreVersion;
	bool t_IsCompatible;

	const SDKReturnCode t_Result = CoreSdk_GetVersionsAndCheckCompatibility(&t_SdkVersion, &t_CoreVersion, &t_IsCompatible);

	if (t_Result == SDKReturnCode::SDKReturnCode_Success)
	{
		const std::string t_Versions = "Sdk version : " + std::string(t_SdkVersion.versionInfo) + ", Core version : " + std::string(t_CoreVersion.versionInfo) + ".";

		if (t_IsCompatible)
		{
			ClientLog::print("Versions are compatible.{}", t_Versions);
		}
		else
		{
			ClientLog::warn("Versions are not compatible with each other.{}", t_Versions);
		}
	}
	else
	{
		ClientLog::error("Failed to get the versions from the SDK. The value returned was {}.", (int32_t)t_Result);
	}

	uint32_t t_SessionId;
	const SDKReturnCode t_SessionIdResult = CoreSdk_GetSessionId(&t_SessionId);
	if (t_SessionIdResult == SDKReturnCode::SDKReturnCode_Success && t_SessionId != 0)
	{
		ClientLog::print("Session Id: {}", t_SessionId);
		s_Instance->m_SessionId = t_SessionId;
	}
	else
	{
		ClientLog::print("Failed to get the Session ID from Core. The value returned was{}.", (int32_t)t_SessionIdResult);
	}

	ManusHost t_Host(*p_Host);
	s_Instance->m_Host = std::make_unique<ManusHost>(t_Host);

	// Set the hand motion mode of the RawSkeletonStream. This is optional and can be set to any of the HandMotion enum values. Default = None
	const SDKReturnCode t_HandMotionResult = CoreSdk_SetRawSkeletonHandMotion(HandMotion_Auto);
	if (t_HandMotionResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::print("Failed to set the hand motion mode. The value returned was {}.", (int32_t)t_HandMotionResult);
	}

	// Only setting state to displaying data on automatic reconnect
	if (s_Instance->m_State == ClientState::ClientState_Disconnected)
	{
		s_Instance->m_State = ClientState::ClientState_DisplayingData;
	}
}

/// @brief Gets called when the client disconnects from manus core.
/// This callback is optional and in the sample changes the client's state.
void SDKClient::OnDisconnectedCallback(const ManusHost* const p_Host)
{
	ClientLog::print("Disconnected from manus core.");
	s_Instance->m_TimeSinceLastDisconnect = std::chrono::high_resolution_clock::now();
	ManusHost t_Host(*p_Host);
	s_Instance->m_Host = std::make_unique<ManusHost>(t_Host);
	s_Instance->m_State = ClientState::ClientState_Disconnected;
}

void SDKClient::OnLogCallback(LogSeverity p_Severity, const char* const p_Log, uint32_t p_Length)
{
	if (!s_Instance)return;

	auto t_Log = new SDKLog();
	t_Log->severity = p_Severity;
	t_Log->string = std::string(p_Log);
	s_Instance->m_LogMutex.lock();
	s_Instance->m_Logs.push_back(t_Log);
	s_Instance->m_LogMutex.unlock();
}

/// @brief This gets called when the client is connected to manus core
/// @param p_SkeletonStreamInfo contains the meta data on how much data regarding the skeleton we need to get from the SDK.
void SDKClient::OnSkeletonStreamCallback(const SkeletonStreamInfo* const p_SkeletonStreamInfo)
{
	if (s_Instance)
	{
		ClientSkeletonCollection* t_NxtClientSkeleton = new ClientSkeletonCollection();
		t_NxtClientSkeleton->skeletons.resize(p_SkeletonStreamInfo->skeletonsCount);

		for (uint32_t i = 0; i < p_SkeletonStreamInfo->skeletonsCount; i++)
		{
			CoreSdk_GetSkeletonInfo(i, &t_NxtClientSkeleton->skeletons[i].info);
			t_NxtClientSkeleton->skeletons[i].nodes.resize(t_NxtClientSkeleton->skeletons[i].info.nodesCount);
			t_NxtClientSkeleton->skeletons[i].info.publishTime = p_SkeletonStreamInfo->publishTime;
			CoreSdk_GetSkeletonData(i, t_NxtClientSkeleton->skeletons[i].nodes.data(), t_NxtClientSkeleton->skeletons[i].info.nodesCount);
		}
		s_Instance->m_SkeletonMutex.lock();
		if (s_Instance->m_NextSkeleton != nullptr) delete s_Instance->m_NextSkeleton;
		s_Instance->m_NextSkeleton = t_NxtClientSkeleton;
		s_Instance->m_SkeletonMutex.unlock();
	}
}

/// @brief This gets called when the client is connected to manus core. It sends the skeleton data coming from the estimation system, before the retargeting to the client skeleton model. 
/// @param p_RawSkeletonStreamInfo contains the meta data on how much data regarding the raw skeleton we need to get from the SDK.
void SDKClient::OnRawSkeletonStreamCallback(const SkeletonStreamInfo* const p_RawSkeletonStreamInfo)
{
	// std::cout<<"OnRawSkeletonStreamCallback"<<std::endl;
	if (s_Instance)
	{
		ClientRawSkeletonCollection* t_NxtClientRawSkeleton = new ClientRawSkeletonCollection();
		t_NxtClientRawSkeleton->skeletons.resize(p_RawSkeletonStreamInfo->skeletonsCount);

		for (uint32_t i = 0; i < p_RawSkeletonStreamInfo->skeletonsCount; i++)
		{
			CoreSdk_GetRawSkeletonInfo(i, &t_NxtClientRawSkeleton->skeletons[i].info);
			t_NxtClientRawSkeleton->skeletons[i].nodes.resize(t_NxtClientRawSkeleton->skeletons[i].info.nodesCount);
			t_NxtClientRawSkeleton->skeletons[i].info.publishTime = p_RawSkeletonStreamInfo->publishTime;
			CoreSdk_GetRawSkeletonData(i, t_NxtClientRawSkeleton->skeletons[i].nodes.data(), t_NxtClientRawSkeleton->skeletons[i].info.nodesCount);
			
			std::string glove_id = std::to_string(t_NxtClientRawSkeleton->skeletons[i].info.gloveId);
			std::vector<double> positions;
			std::vector<double> orientations;
			for (uint32_t j = 0; j < t_NxtClientRawSkeleton->skeletons[i].info.nodesCount; j++){
				positions.push_back(t_NxtClientRawSkeleton->skeletons[i].nodes[j].transform.position.x);
				positions.push_back(t_NxtClientRawSkeleton->skeletons[i].nodes[j].transform.position.y);
				positions.push_back(t_NxtClientRawSkeleton->skeletons[i].nodes[j].transform.position.z);
				orientations.push_back(t_NxtClientRawSkeleton->skeletons[i].nodes[j].transform.rotation.w);
				orientations.push_back(t_NxtClientRawSkeleton->skeletons[i].nodes[j].transform.rotation.x);
				orientations.push_back(t_NxtClientRawSkeleton->skeletons[i].nodes[j].transform.rotation.y);
				orientations.push_back(t_NxtClientRawSkeleton->skeletons[i].nodes[j].transform.rotation.z);
			}
			output_map[glove_id +"_position"] = positions;
			output_map[glove_id +"_orientation"] = orientations;
		}
		s_Instance->m_RawSkeletonMutex.lock();
		if (s_Instance->m_NextRawSkeleton != nullptr) delete s_Instance->m_NextRawSkeleton;
		s_Instance->m_NextRawSkeleton = t_NxtClientRawSkeleton;
		s_Instance->m_RawSkeletonMutex.unlock();
	}
}

/// @brief This gets called when receiving tracker information from core
/// @param p_TrackerStreamInfo contains the meta data on how much data regarding the trackers we need to get from the SDK.
void SDKClient::OnTrackerStreamCallback(const TrackerStreamInfo* const p_TrackerStreamInfo)
{
	if (s_Instance)
	{
		TrackerDataCollection* t_TrackerData = new TrackerDataCollection();

		t_TrackerData->trackerData.resize(p_TrackerStreamInfo->trackerCount);

		for (uint32_t i = 0; i < p_TrackerStreamInfo->trackerCount; i++)
		{
			CoreSdk_GetTrackerData(i, &t_TrackerData->trackerData[i]);
		}
		s_Instance->m_TrackerMutex.lock();
		if (s_Instance->m_NextTrackerData != nullptr) delete s_Instance->m_NextTrackerData;
		s_Instance->m_NextTrackerData = t_TrackerData;
		s_Instance->m_TrackerMutex.unlock();
	}
}

/// @brief This gets called when receiving gesture data from Manus Core
/// In our sample we only save the first glove's gesture data.
/// Gesture data gets generated and sent when glove data changes, this means that the stream
/// does not always contain ALL of the devices, because some may not have had new data since
/// the last time the gesture data was sent.
/// @param p_GestureStream contains the basic info to retrieve gesture data.
void SDKClient::OnGestureStreamCallback(const GestureStreamInfo* const p_GestureStream)
{
	if (s_Instance)
	{
		for (uint32_t i = 0; i < p_GestureStream->gestureProbabilitiesCount; i++)
		{
			GestureProbabilities t_Probs;
			CoreSdk_GetGestureStreamData(i, 0, &t_Probs);
			if (t_Probs.isUserID)continue;
			if (t_Probs.id != s_Instance->m_FirstLeftGloveID && t_Probs.id != s_Instance->m_FirstRightGloveID)continue;
			ClientGestures* t_Gest = new ClientGestures();
			t_Gest->info = t_Probs;
			t_Gest->probabilities.reserve(t_Gest->info.totalGestureCount);
			uint32_t t_BatchCount = (t_Gest->info.totalGestureCount / MAX_GESTURE_DATA_CHUNK_SIZE) + 1;
			uint32_t t_ProbabilityIdx = 0;
			for (uint32_t b = 0; b < t_BatchCount; b++)
			{
				for (uint32_t j = 0; j < t_Probs.gestureCount; j++)
				{
					t_Gest->probabilities.push_back(t_Probs.gestureData[j]);
				}
				t_ProbabilityIdx += t_Probs.gestureCount;
				CoreSdk_GetGestureStreamData(i, t_ProbabilityIdx, &t_Probs); //this will get more data, if needed for the next iteration.
			}

			s_Instance->m_GestureMutex.lock();
			if (t_Probs.id == s_Instance->m_FirstLeftGloveID)
			{
				if (s_Instance->m_NewFirstLeftGloveGestures != nullptr) delete s_Instance->m_NewFirstLeftGloveGestures;
				s_Instance->m_NewFirstLeftGloveGestures = t_Gest;
			}
			else
			{
				if (s_Instance->m_NewFirstRightGloveGestures != nullptr) delete s_Instance->m_NewFirstRightGloveGestures;
				s_Instance->m_NewFirstRightGloveGestures = t_Gest;
			}
			s_Instance->m_GestureMutex.unlock();
		}
	}
}

/// @brief This gets called when receiving landscape information from core
/// @param p_Landscape contains the new landscape from core.
void SDKClient::OnLandscapeCallback(const Landscape* const p_Landscape)
{
	if (s_Instance == nullptr)return;

	Landscape* t_Landscape = new Landscape(*p_Landscape);
	s_Instance->m_LandscapeMutex.lock();
	if (s_Instance->m_NewLandscape != nullptr) delete s_Instance->m_NewLandscape;
	s_Instance->m_NewLandscape = t_Landscape;
	s_Instance->m_NewGestureLandscapeData.resize(t_Landscape->gestureCount);
	CoreSdk_GetGestureLandscapeData(s_Instance->m_NewGestureLandscapeData.data(), (uint32_t)s_Instance->m_NewGestureLandscapeData.size());
	s_Instance->m_LandscapeMutex.unlock();
}


/// @brief This gets called when receiving a system message from Core.
/// @param p_SystemMessage contains the system message received from core.
void SDKClient::OnSystemCallback(const SystemMessage* const p_SystemMessage)
{
	if (s_Instance)
	{
		s_Instance->m_SystemMessageMutex.lock();

		switch (p_SystemMessage->type)
		{
		case SystemMessageType::SystemMessageType_TemporarySkeletonModified:
			// if the message was triggered by a temporary skeleton being modified then save the skeleton index,
			// this information will be used to get and load the skeleton into core
			s_Instance->m_ModifiedSkeletonIndex = p_SystemMessage->infoUInt;
			break;
		default:
			s_Instance->m_SystemMessageCode = p_SystemMessage->type;
			s_Instance->m_SystemMessage = p_SystemMessage->infoString;
			break;
		}
		s_Instance->m_SystemMessageMutex.unlock();
	}
}

/// @brief This gets called when receiving ergonomics data from Manus Core
/// In our sample we only save the first left and first right glove's latests ergonomics data.
/// Ergonomics data gets generated and sent when glove data changes, this means that the stream
/// does not always contain ALL of the devices, because some may not have had new data since
/// the last time the ergonomics data was sent.
/// @param p_Ergo contains the ergonomics data for each glove connected to Core.
void SDKClient::OnErgonomicsCallback(const ErgonomicsStream* const p_Ergo)
{
	if (s_Instance)
	{
		long int left_glove_id = 0;
		long int right_glove_id = 0;
		for (uint32_t i = 0; i < p_Ergo->dataCount; i++)
		{
			if (p_Ergo->data[i].isUserID)continue;

			ErgonomicsData* t_Ergo = nullptr;
			if (std::find(idleft.begin(), idleft.end(), p_Ergo->data[i].id) != idleft.end())
			{
				t_Ergo = &s_Instance->m_LeftGloveErgoData;
				left_glove_id = p_Ergo->data[i].id;
			}
			if (std::find(idright.begin(), idright.end(), p_Ergo->data[i].id) != idright.end())
			{
				t_Ergo = &s_Instance->m_RightGloveErgoData;
				right_glove_id = p_Ergo->data[i].id;
			}
			if (t_Ergo == nullptr)continue;
			CoreSdk_GetTimestampInfo(p_Ergo->publishTime, &s_Instance->m_ErgoTimestampInfo);
			t_Ergo->id = p_Ergo->data[i].id;
			t_Ergo->isUserID = p_Ergo->data[i].isUserID;
			// Create a vector to store the angles of the hand
			std::vector<double> orientations_left;
			std::vector<double> orientations_right;
			for (int j = 0; j < ErgonomicsDataType::ErgonomicsDataType_MAX_SIZE; j++)
			{
				t_Ergo->data[j] = p_Ergo->data[i].data[j];
				double roundedValue = s_Instance->RoundFloatValue(t_Ergo->data[j], 2);
				if (t_Ergo == &s_Instance->m_LeftGloveErgoData)
                {
                    orientations_left.push_back(roundedValue);
                }
                else if (t_Ergo == &s_Instance->m_RightGloveErgoData)
                {
                    orientations_right.push_back(roundedValue);
                }
			}
			if(left_glove_id!=0){
			output_map[std::to_string(left_glove_id)+"_angle"] = orientations_left;
			}
			if(right_glove_id!=0){
			output_map[std::to_string(right_glove_id)+"_angle"] = orientations_right;
			}
		}		
	}
}

/// @brief Round the given float value so that it has no more than the given number of decimals.
float SDKClient::RoundFloatValue(float p_Value, int p_NumDecimalsToKeep)
{
	// Since C++11, powf is supposed to be declared in <cmath>.
	// Unfortunately, gcc decided to be non-compliant on this for no apparent
	// reason, so now we have to do this.
	// https://stackoverflow.com/questions/5483930/powf-is-not-a-member-of-std
	float t_Power = static_cast<float>(std::pow(
		10.0,
		static_cast<double>(p_NumDecimalsToKeep)));
	return std::round(p_Value * t_Power) / t_Power;
}

/// @brief Set the position that the next log message will appear at.
/// Using this allows us to have somewhat of a static, yet flexible layout of logging.
void SDKClient::AdvanceConsolePosition(short int p_Y)
{
	if (p_Y < 0)
	{
		m_ConsoleCurrentOffset = 0;
	}
	else
	{
		m_ConsoleCurrentOffset += p_Y;
	}

	ApplyConsolePosition(m_ConsoleCurrentOffset);
}

/// @brief Initialize the sdk, register the callbacks and set the coordinate system.
/// This needs to be done before any of the other SDK functions can be used.
ClientReturnCode SDKClient::InitializeSDK()
{
	// ClientLog::print("Select what mode you would like to start in (and press enter to submit)");
	// ClientLog::print("[1] Core Integrated - This will run standalone without the need for a MANUS Core connection");
	// ClientLog::print("[2] Core Local - This will connect to a MANUS Core running locally on your machine");
	// ClientLog::print("[3] Core Remote - This will search for a MANUS Core running locally on your network");
	// std::string t_ConnectionTypeInput;
	// std::cin >> t_ConnectionTypeInput;

	std::string t_ConnectionTypeInput = "1";

	switch (t_ConnectionTypeInput[0])
	{
	case '1':
		m_ConnectionType = ConnectionType::ConnectionType_Integrated;
		m_State = ClientState::ClientState_ConnectingToCore;
		break;
	case '2':
		m_ConnectionType = ConnectionType::ConnectionType_Local;
		m_State = ClientState::ClientState_LookingForHosts;
		break;
	case '3':
		m_ConnectionType = ConnectionType::ConnectionType_Remote;
		m_State = ClientState::ClientState_LookingForHosts;
		break;
	default:
		m_ConnectionType = ConnectionType::ConnectionType_Invalid;
		m_State = ClientState::ClientState_Starting;
		ClientLog::print("Invalid input, try again");
		return InitializeSDK();
	}

	// Invalid connection type detected
	if (m_ConnectionType == ConnectionType::ConnectionType_Invalid
		|| m_ConnectionType == ConnectionType::ClientState_MAX_CLIENT_STATE_SIZE)
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;

	// before we can use the SDK, some internal SDK bits need to be initialized.
	// however after initializing, the SDK is not yet connected to a host or doing anything network related just yet.
	bool t_Remote = m_ConnectionType != ConnectionType::ConnectionType_Integrated;
	const SDKReturnCode t_InitializeResult = CoreSdk_Initialize(m_ClientType, t_Remote);
	if (t_InitializeResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to initialize the Manus Core SDK. The value returned was {}.", (int32_t)t_InitializeResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	const ClientReturnCode t_CallBackResults = RegisterAllCallbacks();
	if (t_CallBackResults != ::ClientReturnCode::ClientReturnCode_Success)
	{
		ClientLog::error("Failed to initialize callbacks.");
		return t_CallBackResults;
	}

	// after everything is registered and initialized as seen above
	// we must also set the coordinate system being used for the data in this client.
	// (each client can have their own settings. unreal and unity for instance use different coordinate systems)
	// if this is not set, the SDK will not connect to any Manus core host.
	// The coordinate system used for this example is z-up, x-positive, right-handed and in meter scale.
	CoordinateSystemVUH t_VUH;
	CoordinateSystemVUH_Init(&t_VUH);
	t_VUH.handedness = Side::Side_Right;
	t_VUH.up = AxisPolarity::AxisPolarity_PositiveZ;
	t_VUH.view = AxisView::AxisView_XFromViewer;
	t_VUH.unitScale = 1.0f; //1.0 is meters, 0.01 is cm, 0.001 is mm.

	// The above specified coordinate system is used to initialize and the coordinate space is specified (world/local).
	const SDKReturnCode t_CoordinateResult = CoreSdk_InitializeCoordinateSystemWithVUH(t_VUH, true);

	/* this is an example if you want to use the other coordinate system instead of VUH (view, up, handedness)
	CoordinateSystemDirection t_Direction;
	t_Direction.x = AxisDirection::AD_Right;
	t_Direction.y = AxisDirection::AD_Up;
	t_Direction.z = AxisDirection::AD_Forward;
	const SDKReturnCode t_InitializeResult = CoreSdk_InitializeCoordinateSystemWithDirection(t_Direction, true);
	*/

	if (t_CoordinateResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to initialize the Manus Core SDK coordinate system. The value returned was {}.", (int32_t)t_InitializeResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Used to restart and initialize the SDK to make sure a new connection can be set up.
/// This function is used by the client to shutdown the SDK and any connections it has.
/// After that it reinitializes the SDK so that it is ready to connect to a new Manus Core.
ClientReturnCode SDKClient::RestartSDK()
{
	const SDKReturnCode t_ShutDownResult = CoreSdk_ShutDown();
	if (t_ShutDownResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to shutdown the SDK. The value returned was {}.", (int32_t)t_ShutDownResult);
		return ClientReturnCode::ClientReturnCode_FailedToShutDownSDK;
	}

	const ClientReturnCode t_IntializeResult = InitializeSDK();
	if (t_IntializeResult != ClientReturnCode::ClientReturnCode_Success)
	{
		ClientLog::error("Failed to initialize the SDK functionality. The value returned was {}.", (int32_t)t_IntializeResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Used to register the callbacks between sdk and core.
/// Callbacks that are registered functions that get called when a certain 'event' happens, such as data coming in from Manus Core.
/// All of these are optional, but depending on what data you require you may or may not need all of them.
ClientReturnCode SDKClient::RegisterAllCallbacks()
{
	// Register the callback for when manus core is connected to the SDK
	// it is optional, but helps trigger your client nicely if needed.
	// see the function OnConnectedCallback for more details
	const SDKReturnCode t_RegisterConnectCallbackResult = CoreSdk_RegisterCallbackForOnConnect(*OnConnectedCallback);
	if (t_RegisterConnectCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for after connecting to Manus Core. The value returned was {}.", (int32_t)t_RegisterConnectCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	// Register the callback for when manus core is disconnected to the SDK
	// it is optional, but helps trigger your client nicely if needed.
	// see OnDisconnectedCallback for more details.
	const SDKReturnCode t_RegisterDisconnectCallbackResult = CoreSdk_RegisterCallbackForOnDisconnect(*OnDisconnectedCallback);
	if (t_RegisterDisconnectCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for after disconnecting from Manus Core. The value returned was {}.", (int32_t)t_RegisterDisconnectCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	// Register the callback for logging from the SDK
	// it is optional, but it might help display the logs at the right time/place
	const SDKReturnCode t_RegisterLogCallbackResult = CoreSdk_RegisterCallbackForOnLog(*OnLogCallback);
	if (t_RegisterLogCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for logging from the SDK. The value returned was {}.", (int32_t)t_RegisterLogCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	// Register the callback for when manus core is sending Skeleton data
	// it is optional, but without it you can not see any resulting skeleton data.
	// see OnSkeletonStreamCallback for more details.
	const SDKReturnCode t_RegisterSkeletonCallbackResult = CoreSdk_RegisterCallbackForSkeletonStream(*OnSkeletonStreamCallback);
	if (t_RegisterSkeletonCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for processing skeletal data from Manus Core. The value returned was {}.", (int32_t)t_RegisterSkeletonCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	// Register the callback for when manus core is sending landscape data
	// it is optional, but this allows for a reactive adjustment of device information.
	const SDKReturnCode t_RegisterLandscapeCallbackResult = CoreSdk_RegisterCallbackForLandscapeStream(*OnLandscapeCallback);
	if (t_RegisterLandscapeCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback for landscape from Manus Core. The value returned was {}.", (int32_t)t_RegisterLandscapeCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	// Register the callback for when manus core is sending System messages
	// This is usually not used by client applications unless they want to show errors/events from core.
	// see OnSystemCallback for more details.
	const SDKReturnCode t_RegisterSystemCallbackResult = CoreSdk_RegisterCallbackForSystemStream(*OnSystemCallback);
	if (t_RegisterSystemCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for system feedback from Manus Core. The value returned was {}.", (int32_t)t_RegisterSystemCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	// Register the callback for when manus core is sending Ergonomics data
	// it is optional, but helps trigger your client nicely if needed.
	// see OnErgonomicsCallback for more details.
	const SDKReturnCode t_RegisterErgonomicsCallbackResult = CoreSdk_RegisterCallbackForErgonomicsStream(*OnErgonomicsCallback);
	if (t_RegisterErgonomicsCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for ergonomics data from Manus Core. The value returned was {}.", (int32_t)t_RegisterErgonomicsCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	// Register the callback for when manus core is sending Raw Skeleton data
	// it is optional, but without it you can not see any resulting skeleton data.
	// see OnSkeletonStreamCallback for more details.
	const SDKReturnCode t_RegisterRawSkeletonCallbackResult = CoreSdk_RegisterCallbackForRawSkeletonStream(*OnRawSkeletonStreamCallback);
	if (t_RegisterRawSkeletonCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for processing raw skeletal data from Manus Core. The value returned was {}.", (int32_t)t_RegisterRawSkeletonCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	// Register the callback for when manus core is sending Tracker data
	// it is optional, but without it you can not see any resulting tracker data.
	// see OnTrackerStreamCallback for more details.
	const SDKReturnCode t_RegisterTrackerCallbackResult = CoreSdk_RegisterCallbackForTrackerStream(*OnTrackerStreamCallback);
	if (t_RegisterTrackerCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for processing tracker data from Manus Core. The value returned was {}.", (int32_t)t_RegisterTrackerCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	// Register the callback for when manus core is sending Raw Skeleton data
	// it is optional, but without it you can not see any resulting skeleton data.
	// see OnGestureStreamCallback for more details.
	const SDKReturnCode t_RegisterGestureCallbackResult = CoreSdk_RegisterCallbackForGestureStream(*OnGestureStreamCallback);
	if (t_RegisterGestureCallbackResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to register callback function for processing gesture data from Manus Core. The value returned was {}.", (int32_t)t_RegisterGestureCallbackResult);
		return ClientReturnCode::ClientReturnCode_FailedToInitialize;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Simple example of the SDK looking for manus core hosts on the network
/// and display them on screen.
ClientReturnCode SDKClient::LookingForHosts()
{
	ClientLog::print("Looking for hosts...");

	// Underlying function will sleep for m_SecondsToFindHosts to allow servers to reply.
	bool t_ConnectLocally = m_ConnectionType == ConnectionType::ConnectionType_Local;
	const SDKReturnCode t_StartResult = CoreSdk_LookForHosts(m_SecondsToFindHosts, t_ConnectLocally);
	if (t_StartResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to look for hosts. The error given was {}.", (int32_t)t_StartResult);

		return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
	}

	m_NumberOfHostsFound = 0;
	const SDKReturnCode t_NumberResult = CoreSdk_GetNumberOfAvailableHostsFound(&m_NumberOfHostsFound);
	if (t_NumberResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get the number of available hosts. The error given was {}.", (int32_t)t_NumberResult);

		return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
	}

	if (m_NumberOfHostsFound == 0)
	{
		ClientLog::warn("No hosts found.");
		m_State = ClientState::ClientState_NoHostsFound;

		return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
	}

	m_AvailableHosts.reset(new ManusHost[m_NumberOfHostsFound]);
	const SDKReturnCode t_HostsResult = CoreSdk_GetAvailableHostsFound(m_AvailableHosts.get(), m_NumberOfHostsFound);
	if (t_HostsResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get the available hosts. The error given was {}.", (int32_t)t_HostsResult);

		return ClientReturnCode::ClientReturnCode_FailedToFindHosts;
	}

	if (t_ConnectLocally)
	{
		m_State = ClientState::ClientState_ConnectingToCore;
		return ClientReturnCode::ClientReturnCode_Success;
	}

	m_State = ClientState::ClientState_PickingHost;
	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief When no available hosts are found the user can either retry or exit.
ClientReturnCode SDKClient::NoHostsFound()
{
	if (m_ConsoleClearTickCount == 0)
	{
		AdvanceConsolePosition(-1);
		ClientLog::print("No hosts were found. Retry?");
		ClientLog::print("[R]   retry");
		ClientLog::print("[ESC] exit");
	}

	if (GetKeyDown('R'))
	{
		ClientLog::print("Retrying.");

		m_State = ClientState::ClientState_LookingForHosts;
	}

	// Note: escape is handled by default below.
	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief Print the found hosts and give the user the option to select one.
ClientReturnCode SDKClient::PickingHost()
{
	if (m_ConsoleClearTickCount == 0)
	{
		AdvanceConsolePosition(-1);

		ClientLog::print("[R]   retry   [ESC] exit");
		ClientLog::print("Pick a host to connect to.");
		ClientLog::print("Found the following hosts:");

		// Note: only 10 hosts are shown, to match the number of number keys, for easy selection.
		for (unsigned int t_HostNumber = 0; t_HostNumber < 10 && t_HostNumber < m_NumberOfHostsFound; t_HostNumber++)
		{
			ClientLog::print(
				"[{}] hostname \"{}\", IP address \"{}\" Version {}.{}.{}",
				t_HostNumber,
				m_AvailableHosts[t_HostNumber].hostName,
				m_AvailableHosts[t_HostNumber].ipAddress,
				m_AvailableHosts[t_HostNumber].manusCoreVersion.major,
				m_AvailableHosts[t_HostNumber].manusCoreVersion.minor,
				m_AvailableHosts[t_HostNumber].manusCoreVersion.patch);
		}
	}

	for (unsigned int t_HostNumber = 0; t_HostNumber < 10 && t_HostNumber < m_NumberOfHostsFound; t_HostNumber++)
	{
		if (GetKeyDown('0' + t_HostNumber))
		{
			ClientLog::print("Selected host {}.", t_HostNumber);

			m_HostToConnectTo = t_HostNumber;
			m_State = ClientState::ClientState_ConnectingToCore;

			break;
		}
	}

	if (GetKeyDown('R'))
	{
		ClientLog::print("Retrying.");
		m_State = ClientState::ClientState_LookingForHosts;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief After a connection option was selected, the client will now try to connect to manus core via the SDK.
ClientReturnCode SDKClient::ConnectingToCore()
{
	SDKReturnCode t_ConnectResult = SDKReturnCode::SDKReturnCode_Error;
	switch (m_ConnectionType)
	{
	case ConnectionType::ConnectionType_Integrated:
	{
		ManusHost t_Empty;
		ManusHost_Init(&t_Empty);
		t_ConnectResult = CoreSdk_ConnectToHost(t_Empty);
		break;
	}
	case ConnectionType::ConnectionType_Local:
	{
		t_ConnectResult = CoreSdk_ConnectToHost(m_AvailableHosts[0]);
		break;
	}
	case ConnectionType::ConnectionType_Remote:
	{
		t_ConnectResult = CoreSdk_ConnectToHost(m_AvailableHosts[m_HostToConnectTo]);
		break;
	}
	}

	if (t_ConnectResult == SDKReturnCode::SDKReturnCode_NotConnected)
	{
		m_State = ClientState::ClientState_NoHostsFound;

		return ClientReturnCode::ClientReturnCode_Success; // Differentiating between error and no connect 
	}
	if (t_ConnectResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to connect to Core. The error given was {}.", (int32_t)t_ConnectResult);

		return ClientReturnCode::ClientReturnCode_FailedToConnect;
	}

	m_State = ClientState::ClientState_DisplayingData;

	// Note: a log message from somewhere in the SDK during the connection process can cause text
	// to permanently turn green after this step. Adding a sleep here of 2+ seconds "fixes" the
	// issue. It seems to be caused by a threading issue somewhere, resulting in a log call being
	// interrupted while it is printing the green [info] text. The log output then gets stuck in
	// green mode.

	return ClientReturnCode::ClientReturnCode_Success;
}


/// @brief Some things happen before every display update, no matter what state.
/// They happen here, such as the updating of the landscape and the generated tracker
/// @return 
ClientReturnCode SDKClient::UpdateBeforeDisplayingData()
{
	AdvanceConsolePosition(-1);

	m_SkeletonMutex.lock();
	if (m_NextSkeleton != nullptr)
	{
		if (m_Skeleton != nullptr)delete m_Skeleton;
		m_Skeleton = m_NextSkeleton;
		m_NextSkeleton = nullptr;
	}
	m_SkeletonMutex.unlock();

	m_RawSkeletonMutex.lock();
	if (m_NextRawSkeleton != nullptr)
	{
		if (m_RawSkeleton != nullptr)delete m_RawSkeleton;
		m_RawSkeleton = m_NextRawSkeleton;
		m_NextRawSkeleton = nullptr;
	}
	m_RawSkeletonMutex.unlock();

	m_TrackerMutex.lock();
	if (m_NextTrackerData != nullptr)
	{
		if (m_TrackerData != nullptr)delete m_TrackerData;
		m_TrackerData = m_NextTrackerData;
		m_NextTrackerData = nullptr;
	}
	m_TrackerMutex.unlock();

	m_GestureMutex.lock();
	if (m_NewFirstLeftGloveGestures != nullptr)
	{
		if (m_FirstLeftGloveGestures != nullptr) delete m_FirstLeftGloveGestures;
		m_FirstLeftGloveGestures = m_NewFirstLeftGloveGestures;
		m_NewFirstLeftGloveGestures = nullptr;
	}
	if (m_NewFirstRightGloveGestures != nullptr)
	{
		if (m_FirstRightGloveGestures != nullptr) delete m_FirstRightGloveGestures;
		m_FirstRightGloveGestures = m_NewFirstRightGloveGestures;
		m_NewFirstRightGloveGestures = nullptr;
	}
	m_GestureMutex.unlock();

	m_LandscapeMutex.lock();
	if (m_NewLandscape != nullptr)
	{
		if (m_Landscape != nullptr)
		{
			delete m_Landscape;
		}
		m_Landscape = m_NewLandscape;
		m_NewLandscape = nullptr;
		m_GestureLandscapeData.swap(m_NewGestureLandscapeData);
	}
	m_LandscapeMutex.unlock();

	m_FirstLeftGloveID = 0;
	m_FirstRightGloveID = 0;
	if (m_Landscape == nullptr)return ClientReturnCode::ClientReturnCode_Success;
	for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
	{
		if (m_FirstLeftGloveID == 0 && m_Landscape->gloveDevices.gloves[i].side == Side::Side_Left)
		{
			m_FirstLeftGloveID = m_Landscape->gloveDevices.gloves[i].id;
			continue;
		}
		if (m_FirstRightGloveID == 0 && m_Landscape->gloveDevices.gloves[i].side == Side::Side_Right)
		{
			m_FirstRightGloveID = m_Landscape->gloveDevices.gloves[i].id;
			continue;
		}
	}

	return ClientReturnCode::ClientReturnCode_Success;
}


/// @brief Once the connections are made we loop this function
/// it calls all the input handlers for different aspects of the SDK
/// and then prints any relevant data of it.
ClientReturnCode SDKClient::DisplayingData()
{
	ClientLog::print("<<Main Menu>> [ESC] quit");
	ClientLog::print("[G] Go To Gloves & Dongle Menu");
	ClientLog::print("[S] Go To Skeleton Menu");
	ClientLog::print("[X] Go To Temporary Skeleton Menu");
	ClientLog::print("[T] Go To Tracker Menu");
	ClientLog::print("[D] Go To Landscape Time Info");
	ClientLog::print("[J] Go To Gestures Menu");
	ClientLog::print("[C] Go To Glove Calibration Menu");
	ClientLog::print("[P] Go To Pairing Menu");

	AdvanceConsolePosition(10);

	GO_TO_DISPLAY('G', DisplayingDataGlove)
	GO_TO_DISPLAY('S', DisplayingDataSkeleton)
	GO_TO_DISPLAY('X', DisplayingDataTemporarySkeleton)
	GO_TO_DISPLAY('T', DisplayingDataTracker)
	GO_TO_DISPLAY('D', DisplayingLandscapeTimeData)
	GO_TO_DISPLAY('J', DisplayingDataGestures)
	GO_TO_DISPLAY('C', DisplayingGloveCalibration)
	GO_TO_DISPLAY('P', DisplayingPairing)

	PrintSystemMessage();

	// AdvanceConsolePosition(10);
	// GO_TO_DISPLAY('G', DisplayingDataGlove)
	// PrintSystemMessage();

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief display the ergonomics data of the gloves, and handles haptic commands.
/// @return 
ClientReturnCode SDKClient::DisplayingDataGlove()
{
	ClientLog::print("[Q] Back  <<Gloves & Dongles>> [ESC] quit");
	ClientLog::print("Haptic keys: left:([1]-[5] = pinky-thumb.) right:([6]-[0] = thumb-pinky.)");

	AdvanceConsolePosition(3);

	GO_TO_MENU_IF_REQUESTED()

	HandleHapticCommands();

	PrintErgonomicsData();
	PrintDongleData();
	PrintSystemMessage();

	return ClientReturnCode::ClientReturnCode_Success;
}

ClientReturnCode SDKClient::DisplayingDataSkeleton()
{
	ClientLog::print("[Q] Back  <<Skeleton>> [ESC] quit");
	ClientLog::print("<Skeleton> [B] Load Left Hand Skeleton [N] Load Right Hand Skeleton [M] Unload Skeleton");
	ClientLog::print("<Skeleton> [D] Toggle Send to DevTools ({})", m_SendToDevTools);
	ClientLog::print("<Skeleton Haptics> left:([1]-[5] = pinky-thumb) right:([6]-[0] = thumb-pinky)");

	AdvanceConsolePosition(4);

	GO_TO_MENU_IF_REQUESTED()

	HandleSkeletonCommands();
	HandleSkeletonHapticCommands();

	PrintSkeletonData();
	
	AdvanceConsolePosition(1);
	PrintSystemMessage();

	return ClientReturnCode::ClientReturnCode_Success;
}

ClientReturnCode SDKClient::DisplayingDataTracker()
{
	ClientLog::print("[Q] Back  <<Tracker>> [ESC] quit");
	ClientLog::print("[O] Toggle Test Tracker [G] Toggle per user tracker display [T] Set Tracker Offset to Left Wrist");

	AdvanceConsolePosition(3);

	GO_TO_MENU_IF_REQUESTED()
	HandleTrackerCommands();
	
	PrintRawSkeletonData();

	PrintTrackerData();
	PrintSystemMessage();

	return ClientReturnCode::ClientReturnCode_Success;
}

ClientReturnCode SDKClient::DisplayingDataTemporarySkeleton()
{
	ClientLog::print("[Q] Back  <<Temporary Skeleton>> [ESC] quit");
	ClientLog::print("<Skeleton>[A] Auto allocate chains and load skeleton");
	ClientLog::print("<Skeleton>[B] Build Temporary Skeleton [C] Clear Temporary Skeleton [D] Clear All Temporary Skeletons For The Current Session");
	ClientLog::print("<Skeleton>[E] Save Temporary Skeleton To File, [F] Get Temporary Skeleton From File");

	AdvanceConsolePosition(5);

	GO_TO_MENU_IF_REQUESTED()

	HandleTemporarySkeletonCommands();

	PrintTemporarySkeletonInfo();
	GetTemporarySkeletonIfModified();

	AdvanceConsolePosition(3);
	PrintSystemMessage();

	return ClientReturnCode::ClientReturnCode_Success;
}

ClientReturnCode SDKClient::DisplayingLandscapeTimeData()
{
	ClientLog::print("[Q] Back  <<Landscape Time Data>> [ESC] quit");

	AdvanceConsolePosition(2);

	GO_TO_MENU_IF_REQUESTED()

	PrintLandscapeTimeData();

	PrintSystemMessage();

	return ClientReturnCode::ClientReturnCode_Success;
}

ClientReturnCode SDKClient::DisplayingDataGestures()
{
	ClientLog::print("[Q] Back  <<Gesture Data>> [ESC] quit");
	ClientLog::print("<Gestures>[H] Show other Hand");

	AdvanceConsolePosition(3);

	GO_TO_MENU_IF_REQUESTED()

	HandleGesturesCommands();

	PrintGestureData();

	AdvanceConsolePosition(1);

	PrintSystemMessage();

	return ClientReturnCode::ClientReturnCode_Success;
}

ClientReturnCode SDKClient::DisplayingGloveCalibration()
{
	ClientLog::print("[Q] Back  <<Glove Calibration>> [ESC] quit");
	ClientLog::print("<Glove Calibration>[H] Toggle other Hand [P] Add 1 to the calibration step [M] Remove 1 from the calibration step");
	ClientLog::print("<Glove Calibration>[S] Start glove calibration [C] Cancel glove calibration [F] Finish glove calibration");
	ClientLog::print("<Glove Calibration>[E] Execute calibration step");

	AdvanceConsolePosition(5);

	GO_TO_MENU_IF_REQUESTED()

	HandleGloveCalibrationCommands();

	PrintGloveCalibrationData();

	AdvanceConsolePosition(3);
	PrintSystemMessage();

	return ClientReturnCode::ClientReturnCode_Success;
}

ClientReturnCode SDKClient::DisplayingPairing()
{
	ClientLog::print("[Q] Back  <<Glove Pairing>> [ESC] quit");
	ClientLog::print("<<Glove Pairing>> [P] Pair first available Glove");
	ClientLog::print("<<Glove Pairing>> [U] Unpair first available Glove");


	GO_TO_MENU_IF_REQUESTED()

	HandlePairingCommands();

	AdvanceConsolePosition(4);
	PrintSystemMessage();

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief When the SDK loses the connection with Core the user can either 
/// close the sdk or try to reconnect to a local or to a remote host.
ClientReturnCode SDKClient::DisconnectedFromCore()
{
	if (m_Host == nullptr) { return ClientReturnCode::ClientReturnCode_FailedToConnect; }

	AdvanceConsolePosition(-1);

	auto t_Duration = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - m_TimeSinceLastDisconnect).count();
	ClientLog::print("The SDK lost connection with Manus Core {} seconds ago.", t_Duration);
	ClientLog::print("[P] Pick a new host.   [ESC] exit");

	AdvanceConsolePosition(3);

	bool t_ConnectLocally = m_ConnectionType == ConnectionType::ConnectionType_Local;
	if (t_ConnectLocally)
	{
		ClientLog::print("Automatically trying to reconnect to local host.");

		ClientReturnCode t_ReconnectResult = ReconnectingToCore();
		if (t_ReconnectResult != ClientReturnCode::ClientReturnCode_FailedToConnect)
		{
			return t_ReconnectResult;
		}
	}
	else
	{
		ClientLog::print("[R] Try to reconnect to the last host {} at {}.", m_Host->hostName, m_Host->ipAddress);
		if (GetKeyDown('R'))
		{
			ClientLog::print("Reconnecting");

			ClientReturnCode t_ReconnectResult = ReconnectingToCore(m_SecondsToAttemptReconnecting, m_MaxReconnectionAttempts);
			if (t_ReconnectResult != ClientReturnCode::ClientReturnCode_FailedToConnect)
			{
				return t_ReconnectResult;
			}
		}
	}

	AdvanceConsolePosition(10);


	if (GetKeyDown('P'))
	{
		ClientLog::print("Picking new host.");

		// Restarting and initializing CoreConnection to make sure a new connection can be set up
		const ClientReturnCode t_RestartResult = RestartSDK();
		if (t_RestartResult != ClientReturnCode::ClientReturnCode_Success)
		{
			ClientLog::error("Failed to Restart CoreConnection.");
			return ClientReturnCode::ClientReturnCode_FailedToRestart;
		}

		m_State = ClientState::ClientState_LookingForHosts;
	}

	return ClientReturnCode::ClientReturnCode_Success;
}

/// @brief It is called when the sdk is disconnected from Core and the user select one of the options to reconnect. 
ClientReturnCode SDKClient::ReconnectingToCore(int32_t p_ReconnectionTime, int32_t p_ReconnectionAttempts)
{
	if (p_ReconnectionTime <= 0) { p_ReconnectionTime = std::numeric_limits<int32_t>::max(); }
	if (p_ReconnectionAttempts <= 0) { p_ReconnectionAttempts = std::numeric_limits<int32_t>::max(); }

	// Restarting and initializing CoreConnection to make sure a new connection can be set up
	const ClientReturnCode t_RestartResult = RestartSDK();
	if (t_RestartResult != ClientReturnCode::ClientReturnCode_Success)
	{
		ClientLog::error("Failed to Restart CoreConnection.");
		return ClientReturnCode::ClientReturnCode_FailedToRestart;
	}

	std::chrono::high_resolution_clock::time_point t_Start = std::chrono::high_resolution_clock::now();
	int t_Attempt = 0;
	while ((p_ReconnectionAttempts > 0) && (p_ReconnectionTime > 0))
	{
		ClientLog::print("Trying to reconnect to {} at {}. Attempt {}.", m_Host->hostName, m_Host->ipAddress, t_Attempt);
		ClientLog::print("Attempts remaining: {}. Seconds before time out: {}.", p_ReconnectionAttempts, p_ReconnectionTime);
		
		bool t_ConnectLocally = m_ConnectionType == ConnectionType::ConnectionType_Local;
		if (t_ConnectLocally)
		{
			ClientReturnCode t_ConnectionResult = LookingForHosts();
			if (t_ConnectionResult == ClientReturnCode::ClientReturnCode_Success)
			{
				ClientLog::print("Reconnected to ManusCore.");
				return ClientReturnCode::ClientReturnCode_Success;
			}
		}
		else
		{
			SDKReturnCode t_ConnectionResult = CoreSdk_ConnectToHost(*m_Host.get());
			if (t_ConnectionResult == SDKReturnCode::SDKReturnCode_Success)
			{
				ClientLog::print("Reconnected to ManusCore.");
				return ClientReturnCode::ClientReturnCode_Success;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(m_SleepBetweenReconnectingAttemptsInMs));
		p_ReconnectionTime -= static_cast<int32_t>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - t_Start).count());
		--p_ReconnectionAttempts;
		++t_Attempt;
	}

	ClientLog::print("Failed to reconnect to ManusCore.");
	m_State = ClientState::ClientState_Disconnected;
	return ClientReturnCode::ClientReturnCode_FailedToConnect;
}


/// @brief Prints the ergonomics data of a hand.
/// @param p_ErgoData 
void SDKClient::PrintHandErgoData(ErgonomicsData& p_ErgoData, bool p_Left)
{
	const std::string t_FingerNames[NUM_FINGERS_ON_HAND] = { "[thumb] ", "[index] ", "[middle]", "[ring]  ", "[pinky] " };
	const std::string t_FingerJointNames[NUM_FINGERS_ON_HAND] = { "mcp", "pip", "dip" };
	const std::string t_ThumbJointNames[NUM_FINGERS_ON_HAND] = { "cmc", "mcp", "ip " };

	int t_DataOffset = 0;
	if (!p_Left)t_DataOffset = 20;

	const std::string* t_JointNames = t_ThumbJointNames;
	for (unsigned int t_FingerNumber = 0; t_FingerNumber < NUM_FINGERS_ON_HAND; t_FingerNumber++)
	{
		ClientLog::printWithPadding("{} {} spread: {}, {} stretch: {}, {} stretch: {}, {} stretch: {} ", 6,
		t_FingerNames[t_FingerNumber], // Name of the finger.
		t_JointNames[0],
		RoundFloatValue(p_ErgoData.data[t_DataOffset], 2),
		t_JointNames[0],
		RoundFloatValue(p_ErgoData.data[t_DataOffset + 1], 2),
		t_JointNames[1],
		RoundFloatValue(p_ErgoData.data[t_DataOffset + 2], 2),
		t_JointNames[2],
		RoundFloatValue(p_ErgoData.data[t_DataOffset + 3], 2));
		t_JointNames = t_FingerJointNames;
		t_DataOffset += 4;
	}
}

/// @brief Print the ergonomics data received from Core.
void SDKClient::PrintErgonomicsData()
{
	// for testing purposes we only look at the first 2 gloves available
	ClientLog::print(" -- Ergo Timestamp {}:{}:{}.{} ~ {}/{}/{}(D/M/Y)",
		m_ErgoTimestampInfo.hour, m_ErgoTimestampInfo.minute, m_ErgoTimestampInfo.second, m_ErgoTimestampInfo.fraction,
		m_ErgoTimestampInfo.day, m_ErgoTimestampInfo.month, std::to_string(m_ErgoTimestampInfo.year));
	ClientLog::print(" -- Left Glove -- 0x{} - Angles in degrees", m_FirstLeftGloveID);
	if (m_LeftGloveErgoData.id == m_FirstLeftGloveID)
	{
		PrintHandErgoData(m_LeftGloveErgoData, true);
	}
	else
	{
		ClientLog::print(" ...No Data...");
	}
	ClientLog::print(" -- Right Glove -- 0x{} - Angles in degrees", m_FirstRightGloveID);
	if (m_RightGloveErgoData.id == m_FirstRightGloveID)
	{
		PrintHandErgoData(m_RightGloveErgoData, false);
	}
	else
	{
		ClientLog::print(" ...No Data...");
	}

	AdvanceConsolePosition(14);
}

std::string ConvertDeviceClassTypeToString(DeviceClassType p_Type)
{
	switch (p_Type)
	{
	case DeviceClassType_Dongle:
		return "Dongle";
	case DeviceClassType_Glove:
		return "Glove";
	case DeviceClassType_Glongle:
		return "Glongle (Glove Dongle)";
	default:
		return "Unknown";
	}
}

std::string ConvertDeviceFamilyTypeToString(DeviceFamilyType p_Type)
{
	switch (p_Type)
	{
	case DeviceFamilyType_Prime1:
		return "Prime 1";
	case DeviceFamilyType_Prime2:
		return "Prime 2";
	case DeviceFamilyType_PrimeX:
		return "Prime X";
	case DeviceFamilyType_Metaglove:
		return "Metaglove";
	case DeviceFamilyType_Prime3:
		return "Prime 3";
	case DeviceFamilyType_Virtual:
		return "Virtual";
	case DeviceFamilyType_MetaglovePro:
		return "Metaglove Pro";
	default:
		return "Unknown";
	}
}

/// @brief Print the ergonomics data received from Core.
void SDKClient::PrintDongleData()
{
	// get a dongle id
	uint32_t t_DongleCount = m_Landscape->gloveDevices.dongleCount;
	if (t_DongleCount == 0) return; // we got no gloves to work on anyway!

	DongleLandscapeData t_DongleData;

	for (uint32_t i = 0; i < t_DongleCount; i++)
	{
		t_DongleData = m_Landscape->gloveDevices.dongles[i];
		ClientLog::print(" -- Dongle -- 0x{}", t_DongleData.id);

		ClientLog::print(" Type: {} - {}",
			ConvertDeviceClassTypeToString(t_DongleData.classType),
			ConvertDeviceFamilyTypeToString(t_DongleData.familyType));
		ClientLog::print(" License: {}", t_DongleData.licenseType);

		AdvanceConsolePosition(4);
	}
}

/// @brief Prints the last received system messages received from Core.
void SDKClient::PrintSystemMessage()
{
	m_SystemMessageMutex.lock();
	ClientLog::print(""); //blank line
	ClientLog::print("Received System data:{} / code:{}", m_SystemMessage, (int32_t)m_SystemMessageCode);
	m_SystemMessageMutex.unlock();
}

/// @brief Prints the finalized skeleton data received from Core.
/// Since our console cannot render this data visually in 3d (its not in the scope of this client)
/// we only display a little bit of data of the skeletons we receive.
void SDKClient::PrintSkeletonData()
{
	if (m_Skeleton == nullptr || m_Skeleton->skeletons.size() == 0)
	{
		return;
	}

	ClientLog::print("Received Skeleton data. skeletons:{} first skeleton id:{}", m_Skeleton->skeletons.size(), (int32_t)m_Skeleton->skeletons[0].info.id);

	AdvanceConsolePosition(2);
}

void SDKClient::PrintRawSkeletonData()
{
	if (m_RawSkeleton == nullptr || m_RawSkeleton->skeletons.size() == 0)
	{
		return;
	}
	ClientLog::print("Received Skeleton glove data from Core. skeletons:{} first skeleton glove id:{}", m_RawSkeleton->skeletons.size(), m_RawSkeleton->skeletons[0].info.gloveId);

	if (m_FirstLeftGloveID == 0 && m_FirstRightGloveID == 0) return; // no gloves connected to core

	uint32_t t_NodeCount = 0;
	uint32_t t_GloveId;
	if (m_FirstLeftGloveID != 0)
	{
		t_GloveId = m_FirstLeftGloveID;
	}
	else
	{
		t_GloveId = m_FirstRightGloveID;
	}
	// we need to know the number of nodes in the hierarchy, so we can initialize an array with the correct size.
	SDKReturnCode t_Result = CoreSdk_GetRawSkeletonNodeCount(t_GloveId, t_NodeCount);
	if (t_Result != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get Raw Skeleton Node Count. The error given was {}.", (int32_t)t_Result);
		return;
	}

	// now get the hierarchy data, this can be used to reconstruct the positions of each node in case the user set up the system with a local coordinate system, and to know what each node is exactly.
	// having a node position defined as local means that this will be related to its parent 
	NodeInfo* t_NodeInfo = new NodeInfo[t_NodeCount];
	t_Result = CoreSdk_GetRawSkeletonNodeInfoArray(t_GloveId, t_NodeInfo, t_NodeCount);
	if (t_Result != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get Raw Skeleton Hierarchy. The error given was {}.", (int32_t)t_Result);
		return;
	}

	ClientLog::print("Fetched array with Raw Skeleton Hierarchy with {} nodes.", (int32_t)t_NodeCount);
	AdvanceConsolePosition(2);
}

/// @brief Prints the tracker data
/// Since our console cannot render this data visually in 3d (its not in the scope of this client)
/// we only display a little bit of data of the trackers we receive.
void SDKClient::PrintTrackerData()
{
	ClientLog::print("Tracker test active: {}.", m_TrackerTest); //To show that test tracker is being sent to core
	ClientLog::print("Per user tracker display: {}.", m_TrackerDataDisplayPerUser);

	AdvanceConsolePosition(2);

	if (m_TrackerDataDisplayPerUser)
	{
		PrintTrackerDataPerUser();
		AdvanceConsolePosition(10);
	}
	else
	{
		PrintTrackerDataGlobal();
		AdvanceConsolePosition(3);
	}

	// now, as a test, print the tracker data received from the stream
	if (m_TrackerData == nullptr || m_TrackerData->trackerData.size() == 0)
	{
		return;
	}

	ClientLog::print("Received Tracker data. number of received trackers:{} first tracker type:{}", m_TrackerData->trackerData.size(), (int32_t)m_TrackerData->trackerData[0].trackerType);
}

/// @brief Prints the tracker data without taking users into account.
/// This shows how one can get the tracker data that is being streamed without caring about users.
void SDKClient::PrintTrackerDataGlobal()
{
	uint32_t t_NumberOfAvailabletrackers = 0;
	SDKReturnCode t_TrackerResult = CoreSdk_GetNumberOfAvailableTrackers(&t_NumberOfAvailabletrackers);
	if (t_TrackerResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get tracker data. The error given was {}.", (int32_t)t_TrackerResult);
		return;
	}

	ClientLog::print("received available trackers :{} ", t_NumberOfAvailabletrackers);

	if (t_NumberOfAvailabletrackers == 0) return; // nothing else to do.
	TrackerId* t_TrackerId = new TrackerId[t_NumberOfAvailabletrackers];
	t_TrackerResult = CoreSdk_GetIdsOfAvailableTrackers(t_TrackerId, t_NumberOfAvailabletrackers);
	if (t_TrackerResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get tracker data. The error given was {}.", (int32_t)t_TrackerResult);
		return;
	}
}

/// @brief Prints the tracker data per user, this shows how to access this data for each user.
void SDKClient::PrintTrackerDataPerUser()
{
	uint32_t t_NumberOfAvailableUsers = 0;
	SDKReturnCode t_UserResult = CoreSdk_GetNumberOfAvailableUsers(&t_NumberOfAvailableUsers);
	if (t_UserResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get user count. The error given was {}.", (int32_t)t_UserResult);
		return;
	}
	if (t_NumberOfAvailableUsers == 0) return; // nothing to get yet


	for (uint32_t i = 0; i < t_NumberOfAvailableUsers; i++)
	{
		uint32_t t_NumberOfAvailabletrackers = 0;
		SDKReturnCode t_TrackerResult = CoreSdk_GetNumberOfAvailableTrackersForUserIndex(&t_NumberOfAvailabletrackers, i);
		if (t_TrackerResult != SDKReturnCode::SDKReturnCode_Success)
		{
			ClientLog::error("Failed to get tracker data. The error given was {}.", (int32_t)t_TrackerResult);
			return;
		}

		if (t_NumberOfAvailabletrackers == 0) continue;

		ClientLog::print("received available trackers for user index[{}] :{} ", i, t_NumberOfAvailabletrackers);

		if (t_NumberOfAvailabletrackers == 0) return; // nothing else to do.
		TrackerId* t_TrackerId = new TrackerId[t_NumberOfAvailabletrackers];
		t_TrackerResult = CoreSdk_GetIdsOfAvailableTrackersForUserIndex(t_TrackerId, i, t_NumberOfAvailabletrackers);
		if (t_TrackerResult != SDKReturnCode::SDKReturnCode_Success)
		{
			ClientLog::error("Failed to get tracker data. The error given was {}.", (int32_t)t_TrackerResult);
			return;
		}
	}
}

std::string GetFPSEnumName(TimecodeFPS p_FPS)
{
	switch (p_FPS)
	{
	case TimecodeFPS::TimecodeFPS_23_976:
		return "23.976 FPS (24 dropframe)";
	case TimecodeFPS::TimecodeFPS_24:
		return "24 FPS";
	case TimecodeFPS::TimecodeFPS_25:
		return "25 FPS";
	case TimecodeFPS::TimecodeFPS_29_97:
		return "29.97 FPS (30 dropframe)";
	case TimecodeFPS::TimecodeFPS_30:
		return "30 FPS";
	case TimecodeFPS::TimecodeFPS_50:
		return "50 FPS";
	case TimecodeFPS::TimecodeFPS_59_94:
		return "59.94 FPS (60 dropframe)";
	case TimecodeFPS::TimecodeFPS_60:
		return "60 FPS";
	default:
		return "Undefined FPS";
	}
}

void SDKClient::PrintLandscapeTimeData()
{
	ClientLog::print("Total count of Interfaces: {}", m_Landscape->time.interfaceCount);
	ClientLog::print("Current Interface: {} {} at index {}", m_Landscape->time.currentInterface.name, m_Landscape->time.currentInterface.api, m_Landscape->time.currentInterface.index);

	ClientLog::print("FPS: {}", GetFPSEnumName(m_Landscape->time.fps));
	ClientLog::print("Fake signal: {} | Sync Pulse: {} | Sync Status: {}", m_Landscape->time.fakeTimecode, m_Landscape->time.useSyncPulse, m_Landscape->time.syncStatus);
	ClientLog::print("Device keep alive: {} | Timecode Status: {}", m_Landscape->time.deviceKeepAlive, m_Landscape->time.timecodeStatus);

	AdvanceConsolePosition(6);
}

void SDKClient::PrintGestureData()
{
	ClientGestures* t_Gest = m_FirstLeftGloveGestures;
	std::string t_Side = "Left";
	if (!m_ShowLeftGestures)
	{
		t_Side = "Right";
		t_Gest = m_FirstRightGloveGestures;
	}

	if (t_Gest == nullptr)
	{
		ClientLog::print("No Gesture information for first {} glove.", t_Side);
		AdvanceConsolePosition(1);
		return;
	}
	ClientLog::print("Total count of gestures for the {} glove: {}", t_Side, t_Gest->info.totalGestureCount);
	uint32_t t_Max = t_Gest->info.totalGestureCount;
	if (t_Max > 20) t_Max = 20;
	ClientLog::print("Showing result of first {} gestures.", t_Max);
	for (uint32_t i = 0; i < t_Max; i++)
	{
		char* t_Name = "";
		for (uint32_t g = 0; g < m_GestureLandscapeData.size(); g++)
		{
			if (m_GestureLandscapeData[g].id == t_Gest->probabilities[i].id)
			{
				t_Name = m_GestureLandscapeData[g].name;
			}
		}
		ClientLog::print("Gesture {} ({}) has a probability of {}%.", t_Name, t_Gest->probabilities[i].id, t_Gest->probabilities[i].percent * 100.0f);
	}

	AdvanceConsolePosition(t_Max + 2);
}

void SDKClient::PrintGloveCalibrationData()
{
	std::string t_Side = "Left";
	if (!m_CalibrateLeftHand)
	{
		t_Side = "Right";
	}
	ClientLog::print("Calibrating: {} - Step {} ", t_Side, m_CalibrationStep);
	ClientLog::print("Glove: {}", m_CalibrationGloveId);
	ClientLog::print("Number of steps: {}", m_NumberOfCalibrationSteps);
	ClientLog::print("Title: {}", m_StepData.title);
	ClientLog::print("Description: {}", m_StepData.description);
	ClientLog::print("Time: {}", m_StepData.time);
	ClientLog::print("In progress: {}", m_IsCalibrationInProgress);
	ClientLog::print("Message: {}", m_CalibrationMessage);

	AdvanceConsolePosition(6);
}

/// @brief Prints the type of the first chain generated by the AllocateChain function, this is used for testing.
void SDKClient::PrintSkeletonInfo()
{
	switch (m_ChainType)
	{
		case ChainType::ChainType_FingerIndex:
		{
			ClientLog::print("received Skeleton chain type: ChainType_FingerIndex");
			break;
		}
		case ChainType::ChainType_FingerMiddle:
		{
			ClientLog::print("received Skeleton chain type: ChainType_FingerMiddle");
			break;
		}
		case ChainType::ChainType_FingerPinky:
		{
			ClientLog::print("received Skeleton chain type: ChainType_FingerPinky");
			break;
		}
		case ChainType::ChainType_FingerRing:
		{
			ClientLog::print("received Skeleton chain type: ChainType_FingerRing");
			break;
		}
		case ChainType::ChainType_FingerThumb:
		{
			ClientLog::print("received Skeleton chain type: ChainType_FingerThumb");
			break;
		}
		case ChainType::ChainType_Hand:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Hand");
			break;
		}
		case ChainType::ChainType_Head:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Head");
			break;
		}
		case ChainType::ChainType_Leg:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Leg");
			break;
		}
		case ChainType::ChainType_Neck:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Neck");
			break;
		}
		case ChainType::ChainType_Pelvis:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Pelvis");
			break;
		}
		case ChainType::ChainType_Shoulder:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Shoulder");
			break;
		}
		case ChainType::ChainType_Spine:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Spine");
			break;
		}
		case ChainType::ChainType_Arm:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Arm");
			break;
		}
		case ChainType_Foot:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Foot");
			break;
		}
		case ChainType_Toe:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Toe");
			break;
		}
		case ChainType::ChainType_Invalid:
		default:
		{
			ClientLog::print("received Skeleton chain type: ChainType_Invalid");
			break;
		}
	}
	AdvanceConsolePosition(2);
}

/// @brief This support function checks if a temporary skeleton related to the current session has been modified and gets it.
/// Whenever the Dev Tools saves the changes to the skeleton, it sets the boolean t_IsSkeletonModified to true for that skeleton.
/// With callback OnSystemCallback this application can be notified when one of its temporary skeletons has been modified, so it can get it and, potentially, load it.
void SDKClient::GetTemporarySkeletonIfModified()
{
	// if a temporary skeleton associated to the current session has been modified we can get it and, potentially, load it 
	if (m_ModifiedSkeletonIndex != UINT_MAX)
	{
		// get the temporary skeleton
		uint32_t t_SessionId = m_SessionId;
		SDKReturnCode t_Res = CoreSdk_GetTemporarySkeleton(m_ModifiedSkeletonIndex, t_SessionId);
		if (t_Res != SDKReturnCode::SDKReturnCode_Success)
		{
			ClientLog::error("Failed to get temporary skeleton. The error given was {}.", (int32_t)t_Res);
			return;
		}

		// At this point if we are satisfied with the modifications to the skeleton we can load it into Core.
		// Remember to always call function CoreSdk_ClearTemporarySkeleton after loading a temporary skeleton,
		// this will keep the temporary skeleton list in sync between Core and the SDK.

		//uint32_t t_ID = 0;
		//SDKReturnCode t_Res = CoreSdk_LoadSkeleton(m_ModifiedSkeletonIndex, &t_ID);
		//if (t_Res != SDKReturnCode::SDKReturnCode_Success)
		//{
		//	ClientLog::error("Failed to load skeleton. The error given was {}.", t_Res);
		//	return;
		//}
		//if (t_ID == 0)
		//{
		//	ClientLog::error("Failed to give skeleton an ID.");
		//}
		//m_LoadedSkeletons.push_back(t_ID);
		//t_Res = CoreSdk_ClearTemporarySkeleton(m_ModifiedSkeletonIndex, m_SessionId);
		//if (t_Res != SDKReturnCode::SDKReturnCode_Success)
		//{
		//	ClientLog::error("Failed to clear temporary skeleton. The error given was {}.", t_Res);
		//	return;
		//}
		m_ModifiedSkeletonIndex = UINT_MAX;
	}
}
/// @brief This support function gets the temporary skeletons for all sessions connected to Core and it prints the total number of temporary skeletons associated to the current session.
/// Temporary skeletons are used for auto allocation and in the Dev Tools.
void SDKClient::PrintTemporarySkeletonInfo()
{
	ClientLog::print("Number of temporary skeletons in the SDK: {} ", m_TemporarySkeletons.size());

	static uint32_t t_TotalNumberOfTemporarySkeletonsInCore = 0;
	auto t_TimeSinceLastTemporarySkeletonUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - m_LastTemporarySkeletonUpdate).count();
	if (t_TimeSinceLastTemporarySkeletonUpdate < static_cast<unsigned int>(MILLISECONDS_BETWEEN_TEMPORARY_SKELETONS_UPDATE))
	{
		ClientLog::print("Total number of temporary skeletons in core: {} ", t_TotalNumberOfTemporarySkeletonsInCore);
		return;
	}
	TemporarySkeletonCountForAllSessions t_TemporarySkeletonCountForAllSessions;
	SDKReturnCode t_Res = CoreSdk_GetTemporarySkeletonCountForAllSessions(&t_TemporarySkeletonCountForAllSessions);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get all temporary skeletons. The error given was {}.", (int32_t)t_Res);
		return;
	}

	t_TotalNumberOfTemporarySkeletonsInCore = 0;
	for (uint32_t i = 0; i < t_TemporarySkeletonCountForAllSessions.sessionsCount; i++)
	{
		t_TotalNumberOfTemporarySkeletonsInCore += t_TemporarySkeletonCountForAllSessions.temporarySkeletonCountForSessions[i].skeletonCount;
	}

	// print total number of temporary skeletons:
	ClientLog::print("Total number of temporary skeletons in core: {} ", t_TotalNumberOfTemporarySkeletonsInCore);
	m_LastTemporarySkeletonUpdate = std::chrono::high_resolution_clock::now();
}

void SDKClient::HandlePairingCommands()
{
	if (GetKeyDown('P')) // press P to pair
	{
		PairGlove();
	}

	if (GetKeyDown('U')) // press U to Unpair
	{
		UnpairGlove();
	}
}

/// @brief This showcases haptics support on gloves.
/// Send haptics commands to connected gloves if specific keys were pressed.
/// We have two ways of sending the vibration commands, based on the dongle id or based on the skeleton id.
void SDKClient::HandleHapticCommands()
{
	const int t_LeftHand = 0;
	const int t_RightHand = 1;

	uint32_t t_GloveIDs[2] = { 0, 0 };

	for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
	{
		GloveLandscapeData t_GloveLandscapeData = m_Landscape->gloveDevices.gloves[i];
		if (t_GloveLandscapeData.side == Side_Left && t_GloveLandscapeData.isHaptics)
		{
			t_GloveIDs[t_LeftHand] = t_GloveLandscapeData.id;
			break;
		}
	}

	for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
	{
		GloveLandscapeData t_GloveLandscapeData = m_Landscape->gloveDevices.gloves[i];
		if (t_GloveLandscapeData.side == Side_Right && t_GloveLandscapeData.isHaptics)
		{
			t_GloveIDs[t_RightHand] = t_GloveLandscapeData.id;
			break;
		}
	}

	ClientHapticSettings t_HapticState[NUMBER_OF_HANDS_SUPPORTED]{};

	// The strange key number sequence here results from having gloves lie in front of you, and have the keys and haptics in the same order.
	t_HapticState[t_LeftHand].shouldHapticFinger[0] = GetKey('5'); // left thumb
	t_HapticState[t_LeftHand].shouldHapticFinger[1] = GetKey('4'); // left index
	t_HapticState[t_LeftHand].shouldHapticFinger[2] = GetKey('3'); // left middle
	t_HapticState[t_LeftHand].shouldHapticFinger[3] = GetKey('2'); // left ring
	t_HapticState[t_LeftHand].shouldHapticFinger[4] = GetKey('1'); // left pinky
	t_HapticState[t_RightHand].shouldHapticFinger[0] = GetKey('6'); // right thumb
	t_HapticState[t_RightHand].shouldHapticFinger[1] = GetKey('7'); // right index
	t_HapticState[t_RightHand].shouldHapticFinger[2] = GetKey('8'); // right middle
	t_HapticState[t_RightHand].shouldHapticFinger[3] = GetKey('9'); // right ring
	t_HapticState[t_RightHand].shouldHapticFinger[4] = GetKey('0'); // right pinky

	const float s_FullPower = 1.0f;

	for (unsigned int t_HandNumber = 0; t_HandNumber < NUMBER_OF_HANDS_SUPPORTED; t_HandNumber++)
	{
		//If the Glove ID is 0, no glove was found for that side
		if (t_GloveIDs[t_HandNumber] == 0)
		{
			continue;
		}

		float t_HapticsPowers[NUM_FINGERS_ON_HAND]{};
		for (unsigned int t_FingerNumber = 0; t_FingerNumber < NUM_FINGERS_ON_HAND; t_FingerNumber++)
		{
			t_HapticsPowers[t_FingerNumber] = t_HapticState[t_HandNumber].shouldHapticFinger[t_FingerNumber] ? s_FullPower : 0.0f;
		}

		CoreSdk_VibrateFingersForGlove(t_GloveIDs[t_HandNumber], t_HapticsPowers);
	}
}

/// @brief Handles the console commands for the skeletons.
void SDKClient::HandleSkeletonCommands()
{
	if (GetKeyDown('B'))
	{
		LoadTestSkeleton(Side_Left);
	}
	if (GetKeyDown('N'))
	{
		LoadTestSkeleton(Side_Right);
	}
	if (GetKeyDown('M'))
	{
		UnloadTestSkeleton();
	}
	if (GetKeyDown('D'))
	{
		//Toggle send to DevTools
		m_SendToDevTools = !m_SendToDevTools;
	}
}

/// @brief This showcases haptics support on the skeletons.
/// Send haptics commands to the gloves connected to a skeleton if specific keys were pressed.
/// We have two ways of sending the vibration commands, based on the dongle id or based on the skeleton id.
void SDKClient::HandleSkeletonHapticCommands()
{
	if (m_Skeleton == nullptr || m_Skeleton->skeletons.size() == 0)
	{
		return;
	}

	ClientHapticSettings t_HapticState[NUMBER_OF_HANDS_SUPPORTED]{};
	const int t_LeftHand = 0;
	const int t_RightHand = 1;

	// The strange key number sequence here results from having gloves lie in front of you, and have the keys and haptics in the same order.
	t_HapticState[t_LeftHand].shouldHapticFinger[0] = GetKey('5'); // left thumb
	t_HapticState[t_LeftHand].shouldHapticFinger[1] = GetKey('4'); // left index
	t_HapticState[t_LeftHand].shouldHapticFinger[2] = GetKey('3'); // left middle
	t_HapticState[t_LeftHand].shouldHapticFinger[3] = GetKey('2'); // left ring
	t_HapticState[t_LeftHand].shouldHapticFinger[4] = GetKey('1'); // left pinky
	t_HapticState[t_RightHand].shouldHapticFinger[0] = GetKey('6'); // right thumb
	t_HapticState[t_RightHand].shouldHapticFinger[1] = GetKey('7'); // right index
	t_HapticState[t_RightHand].shouldHapticFinger[2] = GetKey('8'); // right middle
	t_HapticState[t_RightHand].shouldHapticFinger[3] = GetKey('9'); // right ring
	t_HapticState[t_RightHand].shouldHapticFinger[4] = GetKey('0'); // right pinky

	// Note: this timer is apparently not very accurate.
	// It is good enough for this test client, but should probably be replaced for other uses.
	static std::chrono::high_resolution_clock::time_point s_TimeOfLastHapticsCommandSent;
	const std::chrono::high_resolution_clock::time_point s_Now = std::chrono::high_resolution_clock::now();
	const long long s_MillisecondsSinceLastHapticCommand = std::chrono::duration_cast<std::chrono::milliseconds>(s_Now - s_TimeOfLastHapticsCommandSent).count();

	if (s_MillisecondsSinceLastHapticCommand < static_cast<unsigned int>(MINIMUM_MILLISECONDS_BETWEEN_HAPTICS_COMMANDS))
	{
		return;
	}

	const Side s_Hands[NUMBER_OF_HANDS_SUPPORTED] = { Side::Side_Left, Side::Side_Right };
	const float s_FullPower = 1.0f;

	// The preferred way of sending the haptics commands is based on skeleton id

	for (unsigned int t_HandNumber = 0; t_HandNumber < NUMBER_OF_HANDS_SUPPORTED; t_HandNumber++)
	{
		float t_HapticsPowers[NUM_FINGERS_ON_HAND]{};
		for (unsigned int t_FingerNumber = 0; t_FingerNumber < NUM_FINGERS_ON_HAND; t_FingerNumber++)
		{
			t_HapticsPowers[t_FingerNumber] = t_HapticState[t_HandNumber].shouldHapticFinger[t_FingerNumber] ? s_FullPower : 0.0f;
		}
		bool t_IsHaptics = false;
		if (CoreSdk_DoesSkeletonGloveSupportHaptics(m_Skeleton->skeletons[0].info.id, s_Hands[t_HandNumber], &t_IsHaptics) != SDKReturnCode::SDKReturnCode_Success)
		{
			continue;
		}
		if (!t_IsHaptics)
		{
			continue;
		}
		CoreSdk_VibrateFingersForSkeleton(m_Skeleton->skeletons[0].info.id, s_Hands[t_HandNumber], t_HapticsPowers);
	}
}

/// @brief Handles the console commands for the temporary skeletons.
void SDKClient::HandleTemporarySkeletonCommands()
{
	if (GetKeyDown('A'))
	{
		AllocateChains();
	}
	if (GetKeyDown('B'))
	{
		BuildTemporarySkeleton();
	}
	if (GetKeyDown('C'))
	{
		ClearTemporarySkeleton();
	}
	if (GetKeyDown('D'))
	{
		ClearAllTemporarySkeletons();
	}
	if (GetKeyDown('E'))
	{
		SaveTemporarySkeletonToFile();
	}
	if (GetKeyDown('F'))
	{
		GetTemporarySkeletonFromFile();
	}
}

/// @brief This support function is used to set a test tracker and add it to the landscape. 
void SDKClient::HandleTrackerCommands()
{
	if (GetKeyDown('O'))
	{
		m_TrackerTest = !m_TrackerTest;
	}

	if (GetKeyDown('G'))
	{
		m_TrackerDataDisplayPerUser = !m_TrackerDataDisplayPerUser;
	}

	if (GetKeyDown('T'))
	{
		SetTrackerOffset();
	}

	if (m_TrackerTest)
	{
		m_TrackerOffset += 0.0005f;
		if (m_TrackerOffset >= 10.0f)
		{
			m_TrackerOffset = 0.0f;
		}

		TrackerId t_TrackerId;
		CopyString(t_TrackerId.id, sizeof(t_TrackerId.id), std::string("Test Tracker"));
		TrackerData t_TrackerData = {};
		t_TrackerData.isHmd = false;
		t_TrackerData.trackerId = t_TrackerId;
		t_TrackerData.trackerType = TrackerType::TrackerType_Unknown;
		t_TrackerData.position = { 0.0f, m_TrackerOffset, 0.0f };
		t_TrackerData.rotation = { 1.0f, 0.0f, 0.0f, 0.0f };
		t_TrackerData.quality = TrackerQuality::TrackingQuality_Trackable;
		TrackerData t_TrackerDatas[MAX_NUMBER_OF_TRACKERS];
		t_TrackerDatas[0] = t_TrackerData;

		const SDKReturnCode t_TrackerSend = CoreSdk_SendDataForTrackers(t_TrackerDatas, 1);
		if (t_TrackerSend != SDKReturnCode::SDKReturnCode_Success)
		{
			ClientLog::error("Failed to send tracker data. The error given was {}.", (int32_t)t_TrackerSend);
			return;
		}
	}
}

/// @brief This support function is used to set the tracker offset. It sets the tracker offset for the left hand tracker to the wrist.
void SDKClient::SetTrackerOffset() {

	TrackerOffset t_TrackerOffset;

	//Positions the tracker 3mm right, 58mm up and 43mm forward relative to the wrist. This value is illustrative and should be adjusted to the actual tracker position.
	t_TrackerOffset.translation = { 0.003f, -0.058f, -0.043f };
	t_TrackerOffset.rotation = { 1.0f, 0.0f, 0.0f, 0.0f };
	t_TrackerOffset.entryType = TrackerOffsetType::TrackerOffsetType_LeftHandTrackerToWrist;

	auto t_UserId = m_Landscape->users.users[0].id;

	SDKReturnCode t_SetTrackerReturn = CoreSdk_SetTrackerOffset(t_UserId, &t_TrackerOffset);
	if (t_SetTrackerReturn != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to set tracker offset. The error given was {}.", (int32_t)t_SetTrackerReturn);
		return;
	}
}

/// @brief Handles the console commands for the gestures.
void SDKClient::HandleGesturesCommands()
{
	if (GetKeyDown('H'))
	{
		m_ShowLeftGestures = !m_ShowLeftGestures;
	}
}

void SDKClient::ExecuteGloveCalibrationStep(GloveCalibrationStepArgs p_Args)
{
	bool t_Result;
	auto t_Res = CoreSdk_GloveCalibrationStartStep(p_Args, &t_Result);

	if (t_Res == SDKReturnCode_Success && t_Result)
		m_CalibrationMessage = "Step finished";
	else
		m_CalibrationMessage = "Step failed";

	m_IsCalibrationInProgress = false;
}


/// @brief Handles the console commands for the glove calibration.
void SDKClient::HandleGloveCalibrationCommands()
{
	// Reset data
	m_NumberOfCalibrationSteps = 0;
	m_CalibrationGloveId = 0;
	m_ValidStepData = false;
	GloveCalibrationStepData_Init(&m_StepData);

	if (GetKeyDown('H'))
	{
		m_CalibrateLeftHand = !m_CalibrateLeftHand;
	}

	// Get glove id
	if (m_Landscape == nullptr || m_Landscape->users.userCount == 0)
		return;

	uint32_t t_GloveID = m_Landscape->users.users[0].leftGloveID;
	if (!m_CalibrateLeftHand)
		t_GloveID = m_Landscape->users.users[0].rightGloveID;

	m_CalibrationGloveId = t_GloveID;

	// Check if glove ID is valid, if not return
	if (m_CalibrationGloveId == 0)
		return;

	GloveCalibrationArgs t_Args;
	t_Args.gloveId = m_CalibrationGloveId;

	// Get the number of calibration steps
	auto t_Res = CoreSdk_GloveCalibrationGetNumberOfSteps(t_Args, &m_NumberOfCalibrationSteps);
	if (t_Res != SDKReturnCode_Success || m_NumberOfCalibrationSteps == 0)
		return;

	if (GetKeyDown('P') && m_CalibrationStep < 9)
	{
		m_CalibrationStep++;
	}

	if (GetKeyDown('M') && m_CalibrationStep != 0)
	{
		m_CalibrationStep--;
	}

	// Clamp current step to max number of steps
	if (m_CalibrationStep >= m_NumberOfCalibrationSteps)
		m_CalibrationStep = m_NumberOfCalibrationSteps - 1;

	// Get step data
	GloveCalibrationStepArgs t_StepArgs;
	t_StepArgs.gloveId = t_Args.gloveId;
	t_StepArgs.stepIndex = m_CalibrationStep;
	t_Res = CoreSdk_GloveCalibrationGetStepData(t_StepArgs, &m_StepData);
	if (t_Res != SDKReturnCode_Success)
		return;

	// Dont allow starting/stopping/etc while in progress
	if (m_IsCalibrationInProgress)
		return;

	if (GetKeyDown('S')) // Start
	{
		bool t_Result = false;
		t_Res = CoreSdk_GloveCalibrationStart(t_Args, &t_Result);

		if (t_Res == SDKReturnCode::SDKReturnCode_Success && t_Result)
			m_CalibrationMessage = "Glove calibration started";
		else
			m_CalibrationMessage = "Failed to start glove calibration";
	}

	if (GetKeyDown('C')) // Cancel
	{
		bool t_Result = false;
		t_Res = CoreSdk_GloveCalibrationStop(t_Args, &t_Result);

		if (t_Res == SDKReturnCode::SDKReturnCode_Success && t_Result)
			m_CalibrationMessage = "Glove calibration stopped";
		else
			m_CalibrationMessage = "Failed to stop glove calibration";
	}

	if (GetKeyDown('F')) // Finish
	{
		bool t_Result = false;
		t_Res = CoreSdk_GloveCalibrationFinish(t_Args, &t_Result);

		if (t_Res == SDKReturnCode::SDKReturnCode_Success && t_Result)
			m_CalibrationMessage = "Glove calibration finished";
		else
			m_CalibrationMessage = "Failed to finish glove calibration";
	}

	if (GetKeyDown('E')) // Execute step
	{
		m_CalibrationMessage = "Step " + std::to_string(m_CalibrationStep) + " in progress";
		m_IsCalibrationInProgress = true;
		std::thread t_Thread(&SDKClient::ExecuteGloveCalibrationStep, this, t_StepArgs);
		t_Thread.detach();
	}
}

/// @brief Skeletons are pretty extensive in their data setup
/// so we have several support functions so we can correctly receive and parse the data, 
/// this function helps setup the data.
/// @param p_Id the id of the created node setup
/// @param p_ParentId the id of the node parent
/// @param p_PosX X position of the node, this is defined with respect to the global coordinate system or the local one depending on 
/// the parameter p_UseWorldCoordinates set when initializing the sdk,
/// @param p_PosY Y position of the node this is defined with respect to the global coordinate system or the local one depending on 
/// the parameter p_UseWorldCoordinates set when initializing the sdk,
/// @param p_PosZ Z position of the node this is defined with respect to the global coordinate system or the local one depending on 
/// the parameter p_UseWorldCoordinates set when initializing the sdk,
/// @param p_Name the name of the node setup
/// @return the generated node setup
NodeSetup SDKClient::CreateNodeSetup(uint32_t p_Id, uint32_t p_ParentId, float p_PosX, float p_PosY, float p_PosZ, std::string p_Name)
{
	NodeSetup t_Node;
	NodeSetup_Init(&t_Node);
	t_Node.id = p_Id; //Every ID needs to be unique per node in a skeleton.
	CopyString(t_Node.name, sizeof(t_Node.name), p_Name);
	t_Node.type = NodeType::NodeType_Joint;
	//Every node should have a parent unless it is the Root node.
	t_Node.parentID = p_ParentId; //Setting the node ID to its own ID ensures it has no parent.
	t_Node.settings.usedSettings = NodeSettingsFlag::NodeSettingsFlag_None;

	t_Node.transform.position.x = p_PosX;
	t_Node.transform.position.y = p_PosY;
	t_Node.transform.position.z = p_PosZ;
	return t_Node;
}

ManusVec3 SDKClient::CreateManusVec3(float p_X, float p_Y, float p_Z)
{
	ManusVec3 t_Vec;
	t_Vec.x = p_X;
	t_Vec.y = p_Y;
	t_Vec.z = p_Z;
	return t_Vec;
}

/// @brief This support function sets up the nodes for the skeleton hand
/// In order to have any 3d positional/rotational information from the gloves or body,
/// one needs to setup the skeleton on which this data should be applied.
/// In the case of this sample we create a Hand skeleton for which we want to get the calculated result.
/// The ID's for the nodes set here are the same IDs which are used in the OnSkeletonStreamCallback,
/// this allows us to create the link between Manus Core's data and the data we enter here.
bool SDKClient::SetupHandNodes(uint32_t p_SklIndex, Side p_Side)
{
	// Define number of fingers per hand and number of joints per finger
	const uint32_t t_NumFingers = 5;
	const uint32_t t_NumJoints = 4;

	// Because the skeleton is build up in world space coordinates all joints have to be defined in coordate space as well
	// If it is required for the skeleton to work in local space, substract all previous positions from the vectors
	static ManusVec3 s_LeftHandPositions[t_NumFingers * t_NumJoints]
	{
		CreateManusVec3(0.025320f, -0.024950f, 0.000000f), //Thumb CMC joint
		CreateManusVec3(0.025320f + 0.032742f, -0.024950f, 0.000000f), //Thumb MCP joint
		CreateManusVec3(0.025320f + 0.032742f + 0.028739f, -0.024950f, 0.000000f), //Thumb IP joint
		CreateManusVec3(0.025320f + 0.032742f + 0.028739f + 0.028739f, -0.024950f, 0.000000f), //Thumb Tip joint

		CreateManusVec3(0.052904f, -0.011181f, 0.000000f), //Index MCP joint
		CreateManusVec3(0.052904f + 0.038257f, -0.011181f, 0.000000f), //Index PIP joint
		CreateManusVec3(0.052904f + 0.038257f + 0.020884f, -0.011181f, 0.000000f), //Index DIP joint
		CreateManusVec3(0.052904f + 0.038257f + 0.020884f + 0.018759f, -0.011181f, 0.000000f), //Index Tip joint

		CreateManusVec3(0.051287f, 0.000000f, 0.000000f),  //Middle MCP joint
		CreateManusVec3(0.051287f + 0.041861f, 0.000000f, 0.000000f),  //Middle PIP joint
		CreateManusVec3(0.051287f + 0.041861f + 0.024766f, 0.000000f, 0.000000f),  //Middle DIP joint
		CreateManusVec3(0.051287f + 0.041861f + 0.024766f + 0.019683f, 0.000000f, 0.000000f),  //Middle Tip joint
														   
		CreateManusVec3(0.049802f, 0.011274f, 0.000000f ), //Ring MCP joint
		CreateManusVec3(0.049802f + 0.039736f, 0.011274f, 0.000000f),  //Ring PIP joint
		CreateManusVec3(0.049802f + 0.039736f + 0.023564f, 0.011274f, 0.000000f),  //Ring DIP joint
		CreateManusVec3(0.049802f + 0.039736f + 0.023564f + 0.019868f, 0.011274f, 0.000000f),  //Ring Tip joint
														   
		CreateManusVec3(0.047309f, 0.020145f,0.000000f),   //Pinky MCP joint
		CreateManusVec3(0.047309f + 0.033175f, 0.020145f, 0.000000f),  //Pinky PIP joint
		CreateManusVec3(0.047309f + 0.033175f + 0.018020f, 0.020145f, 0.000000f),  //Pinky DIP joint
		CreateManusVec3(0.047309f + 0.033175f + 0.018020f + 0.019129f, 0.020145f, 0.000000f),  //Pinky Tip joint
	};

	static ManusVec3 s_RightHandPositions[t_NumFingers * t_NumJoints]
	{
		CreateManusVec3(0.025320f, 0.024950f, 0.000000f), //Thumb CMC joint
		CreateManusVec3(0.025320f + 0.032742f, 0.024950f, 0.000000f), //Thumb MCP joint
		CreateManusVec3(0.025320f + 0.032742f + 0.028739f, 0.024950f, 0.000000f), //Thumb IP joint
		CreateManusVec3(0.025320f + 0.032742f + 0.028739f + 0.028739f, 0.024950f, 0.000000f), //Thumb Tip joint

		CreateManusVec3(0.052904f, 0.011181f, 0.000000f), //Index MCP joint
		CreateManusVec3(0.052904f + 0.038257f, 0.011181f, 0.000000f), //Index PIP joint
		CreateManusVec3(0.052904f + 0.038257f + 0.020884f, 0.011181f, 0.000000f), //Index DIP joint
		CreateManusVec3(0.052904f + 0.038257f + 0.020884f + 0.018759f, 0.011181f, 0.000000f), //Index Tip joint

		CreateManusVec3(0.051287f, 0.000000f, 0.000000f), //Middle MCP joint
		CreateManusVec3(0.051287f + 0.041861f, 0.000000f, 0.000000f), //Middle PIP joint
		CreateManusVec3(0.051287f + 0.041861f + 0.024766f, 0.000000f, 0.000000f), //Middle DIP joint
		CreateManusVec3(0.051287f + 0.041861f + 0.024766f + 0.019683f, 0.000000f, 0.000000f), //Middle Tip joint

		CreateManusVec3(0.049802f, -0.011274f, 0.000000f),//Ring MCP joint
		CreateManusVec3(0.049802f + 0.039736f, -0.011274f, 0.000000f), //Ring PIP joint
		CreateManusVec3(0.049802f + 0.039736f + 0.023564f, -0.011274f, 0.000000f), //Ring DIP joint
		CreateManusVec3(0.049802f + 0.039736f + 0.023564f + 0.019868f, -0.011274f, 0.000000f), //Ring Tip joint

		CreateManusVec3(0.047309f, -0.020145f, 0.000000f),//Pinky MCP joint
		CreateManusVec3(0.047309f + 0.033175f, -0.020145f, 0.000000f), //Pinky PIP joint
		CreateManusVec3(0.047309f + 0.033175f + 0.018020f, -0.020145f, 0.000000f), //Pinky DIP joint
		CreateManusVec3(0.047309f + 0.033175f + 0.018020f + 0.019129f, -0.020145f, 0.000000f), //Pinky Tip joint
	};

	// Create an array with the initial position of each hand node. 
	// Note, these values are just an example of node positions and refer to the hand laying on a flat surface.
	ManusVec3* t_Fingers;
	if (p_Side == Side::Side_Left)
	{
		t_Fingers = s_LeftHandPositions;
	}
	else
	{
		t_Fingers = s_RightHandPositions;
	}

	// skeleton entry is already done. just the nodes now.
	// setup a very simple node hierarchy for fingers
	// first setup the root node
	// 
	// root, This node has ID 0 and parent ID 0, to indicate it has no parent.
	SDKReturnCode t_Res = CoreSdk_AddNodeToSkeletonSetup(p_SklIndex, CreateNodeSetup(0, 0, 0, 0, 0, "Hand"));
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Add Node To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return false;
	}

	// then loop for 5 fingers
	int t_FingerId = 0;
	for (uint32_t i = 0; i < t_NumFingers; i++)
	{
		uint32_t t_ParentID = 0;
		// then the digits of the finger that are linked to the root of the finger.
		for (uint32_t j = 0; j < t_NumJoints; j++)
		{
			t_Res = CoreSdk_AddNodeToSkeletonSetup(p_SklIndex, CreateNodeSetup(1 + t_FingerId + j, t_ParentID, t_Fingers[i * 4 + j].x, t_Fingers[i * 4 + j].y, t_Fingers[i * 4 + j].z, "fingerdigit"));
			if (t_Res != SDKReturnCode::SDKReturnCode_Success)
			{
				printf("Failed to Add Node To Skeleton Setup. The error given %d.", t_Res);
				return false;
			}
			t_ParentID = 1 + t_FingerId + j;
		}
		t_FingerId += t_NumJoints;
	}
	return true;
}

/// @brief This function sets up some basic hand chains.
/// Chains are required for a Skeleton to be able to be animated, it basically tells Manus Core
/// which nodes belong to which body part and what data needs to be applied to which node.
/// @param p_SklIndex The index of the temporary skeleton on which the chains will be added.
/// @return Returns true if everything went fine, otherwise returns false.
bool SDKClient::SetupHandChains(uint32_t p_SklIndex, Side p_Side)
{
	// Add the Hand chain, this identifies the wrist of the hand
	{
		ChainSettings t_ChainSettings;
		ChainSettings_Init(&t_ChainSettings);
		t_ChainSettings.usedSettings = ChainType::ChainType_Hand;
		t_ChainSettings.hand.handMotion = HandMotion::HandMotion_IMU;
		t_ChainSettings.hand.fingerChainIdsUsed = 5; //we will have 5 fingers
		t_ChainSettings.hand.fingerChainIds[0] = 1; //links to the other chains we will define further down
		t_ChainSettings.hand.fingerChainIds[1] = 2;
		t_ChainSettings.hand.fingerChainIds[2] = 3;
		t_ChainSettings.hand.fingerChainIds[3] = 4;
		t_ChainSettings.hand.fingerChainIds[4] = 5;

		ChainSetup t_Chain;
		ChainSetup_Init(&t_Chain);
		t_Chain.id = 0; //Every ID needs to be unique per chain in a skeleton.
		t_Chain.type = ChainType::ChainType_Hand;
		t_Chain.dataType = ChainType::ChainType_Hand;
		t_Chain.side = p_Side;
		t_Chain.dataIndex = 0;
		t_Chain.nodeIdCount = 1;
		t_Chain.nodeIds[0] = 0; //this links to the hand node created in the SetupHandNodes
		t_Chain.settings = t_ChainSettings;

		SDKReturnCode t_Res = CoreSdk_AddChainToSkeletonSetup(p_SklIndex, t_Chain);
		if (t_Res != SDKReturnCode::SDKReturnCode_Success)
		{
			ClientLog::error("Failed to Add Chain To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
			return false;
		}
	}

	// Add the 5 finger chains
	const ChainType t_FingerTypes[5] = { ChainType::ChainType_FingerThumb,
		ChainType::ChainType_FingerIndex,
		ChainType::ChainType_FingerMiddle,
		ChainType::ChainType_FingerRing,
		ChainType::ChainType_FingerPinky };
	for (int i = 0; i < 5; i++)
	{
		ChainSettings t_ChainSettings;
		ChainSettings_Init(&t_ChainSettings);
		t_ChainSettings.usedSettings = t_FingerTypes[i];
		t_ChainSettings.finger.handChainId = 0; //This links to the wrist chain above.
		//This identifies the metacarpal bone, if none exists, or the chain is a thumb it should be set to -1.
		//The metacarpal bone should not be part of the finger chain, unless you are defining a thumb which does need it.
		t_ChainSettings.finger.metacarpalBoneId = -1;
		t_ChainSettings.finger.useLeafAtEnd = false; //this is set to true if there is a leaf bone to the tip of the finger.
		ChainSetup t_Chain;
		ChainSetup_Init(&t_Chain);
		t_Chain.id = i + 1; //Every ID needs to be unique per chain in a skeleton.
		t_Chain.type = t_FingerTypes[i];
		t_Chain.dataType = t_FingerTypes[i];
		t_Chain.side = p_Side;
		t_Chain.dataIndex = 0;
		if (i == 0) // Thumb
		{
			t_Chain.nodeIdCount = 4; //The amount of node id's used in the array
			t_Chain.nodeIds[0] = 1; //this links to the hand node created in the SetupHandNodes
			t_Chain.nodeIds[1] = 2; //this links to the hand node created in the SetupHandNodes
			t_Chain.nodeIds[2] = 3; //this links to the hand node created in the SetupHandNodes
			t_Chain.nodeIds[3] = 4; //this links to the hand node created in the SetupHandNodes
		}
		else // All other fingers
		{
			t_Chain.nodeIdCount = 4; //The amount of node id's used in the array
			t_Chain.nodeIds[0] = (i * 4) + 1; //this links to the hand node created in the SetupHandNodes
			t_Chain.nodeIds[1] = (i * 4) + 2; //this links to the hand node created in the SetupHandNodes
			t_Chain.nodeIds[2] = (i * 4) + 3; //this links to the hand node created in the SetupHandNodes
			t_Chain.nodeIds[3] = (i * 4) + 4; //this links to the hand node created in the SetupHandNodes
		}
		t_Chain.settings = t_ChainSettings;

		SDKReturnCode t_Res = CoreSdk_AddChainToSkeletonSetup(p_SklIndex, t_Chain);
		if (t_Res != SDKReturnCode::SDKReturnCode_Success)
		{
			return false;
		}
	}
	return true;
}

/// @brief This function sets up a very minimalistic hand skeleton.
/// In order to have any 3d positional/rotational information from the gloves or body,
/// one needs to setup a skeleton on which this data can be applied.
/// In the case of this sample we create a Hand skeleton in order to get skeleton information
/// in the OnSkeletonStreamCallback function. This sample does not contain any 3D rendering, so
/// we will not be applying the returned data on anything.
void SDKClient::LoadTestSkeleton(Side p_Side)
{
	uint32_t t_SklIndex = 0;

	SkeletonSetupInfo t_SKL;
	SkeletonSetupInfo_Init(&t_SKL);
	t_SKL.type = SkeletonType::SkeletonType_Hand;
	t_SKL.settings.scaleToTarget = true;
	t_SKL.settings.useEndPointApproximations = true;
	t_SKL.settings.targetType = SkeletonTargetType::SkeletonTargetType_UserIndexData;
	//If the user does not exist then the added skeleton will not be animated.
	//Same goes for any other skeleton made for invalid users/gloves.
	t_SKL.settings.skeletonTargetUserIndexData.userIndex = 0;

	CopyString(t_SKL.name, sizeof(t_SKL.name), p_Side == Side::Side_Left ? std::string("LeftHand") : std::string("RightHand"));

	SDKReturnCode t_Res = CoreSdk_CreateSkeletonSetup(t_SKL, &t_SklIndex);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Create Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}
	m_TemporarySkeletons.push_back(t_SklIndex);

	// setup nodes and chains for the skeleton hand
	if (!SetupHandNodes(t_SklIndex, p_Side)) return;
	if (!SetupHandChains(t_SklIndex, p_Side)) return;

	if (m_SendToDevTools)
	{
		SendLoadedSkeleton(t_SklIndex);
	}

	// load skeleton 
	uint32_t t_ID = 0;
	t_Res = CoreSdk_LoadSkeleton(t_SklIndex, &t_ID);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to load skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}
	RemoveIndexFromTemporarySkeletonList(t_SklIndex);

	if (t_ID == 0)
	{
		ClientLog::error("Failed to give skeleton an ID.");
	}
	m_LoadedSkeletons.push_back(t_ID);
}

/// @brief This support function is used to unload a skeleton from Core. 
void SDKClient::UnloadTestSkeleton()
{
	if (m_LoadedSkeletons.size() == 0)
	{
		ClientLog::error("There was no skeleton for us to unload.");
		return;
	}
	SDKReturnCode t_Res = CoreSdk_UnloadSkeleton(m_LoadedSkeletons[0]);
	m_LoadedSkeletons.erase(m_LoadedSkeletons.begin());
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to unload skeleton. The error given was {}.", (int32_t)t_Res);

		return;
	}
}

/// @brief
/// Sends the loaded skeleton to the Core SDK.
/// This function saves the temporary skeleton with the given session ID and returns an error if the operation fails.
void SDKClient::SendLoadedSkeleton(uint32_t p_SklIndex)
{
	// save the temporary skeleton 
	SDKReturnCode t_Res = CoreSdk_SaveTemporarySkeleton(p_SklIndex, m_SessionId, true);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to save temporary skeleton. The error given was {}.", (int32_t)t_Res);
	}
}


/// @brief This support function sets up an incomplete hand skeleton and then uses manus core to allocate chains for it.
void SDKClient::AllocateChains()
{
	m_ChainType = ChainType::ChainType_Invalid;

	uint32_t t_SklIndex = 0;

	SkeletonSettings t_Settings;
	SkeletonSettings_Init(&t_Settings);
	t_Settings.scaleToTarget = true;
	t_Settings.targetType = SkeletonTargetType::SkeletonTargetType_UserData;
	t_Settings.skeletonTargetUserData.userID = 0;

	SkeletonSetupInfo t_SKL;
	SkeletonSetupInfo_Init(&t_SKL);
	t_SKL.id = 0;
	t_SKL.type = SkeletonType::SkeletonType_Hand;
	t_SKL.settings = t_Settings;
	CopyString(t_SKL.name, sizeof(t_SKL.name), std::string("hand"));

	SDKReturnCode t_Res = CoreSdk_CreateSkeletonSetup(t_SKL, &t_SklIndex);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Create Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}
	m_TemporarySkeletons.push_back(t_SklIndex);

	// setup nodes for the skeleton hand
	SetupHandNodes(t_SklIndex, Side::Side_Left);

	// allocate chains for skeleton 
	t_Res = CoreSdk_AllocateChainsForSkeletonSetup(t_SklIndex);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to allocate chains for skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// get the skeleton info
	SkeletonSetupArraySizes t_SkeletonInfo;
	t_Res = CoreSdk_GetSkeletonSetupArraySizes(t_SklIndex, &t_SkeletonInfo);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get info about skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}

	ChainSetup* t_Chains = new ChainSetup[t_SkeletonInfo.chainsCount];
	// now get the chain data
	t_Res = CoreSdk_GetSkeletonSetupChainsArray(t_SklIndex, t_Chains, t_SkeletonInfo.chainsCount);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get skeleton setup chains. The error given was {}.", (int32_t)t_Res);
		delete[] t_Chains;
		return;
	}
	// as proof store the first chain type
	m_ChainType = t_Chains[0].dataType;

	// but since we want to cleanly load the skeleton without holding everything up
	// we need to set its side first
	for (size_t i = 0; i < t_SkeletonInfo.chainsCount; i++)
	{
		if (t_Chains[i].dataType == ChainType::ChainType_Hand)
		{
			t_Chains[i].side = Side::Side_Left; // we're just picking a side here. 

			t_Res = CoreSdk_OverwriteChainToSkeletonSetup(t_SklIndex, t_Chains[i]);
			if (t_Res != SDKReturnCode::SDKReturnCode_Success)
			{
				ClientLog::error("Failed to overwrite Chain To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
				delete[] t_Chains;
				return;
			}
			break; // no need to continue checking the others.
		}
	}
	// cleanup
	delete[] t_Chains;

	// load skeleton so it is done. 
	uint32_t t_ID = 0;
	t_Res = CoreSdk_LoadSkeleton(t_SklIndex, &t_ID);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to load skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}
	RemoveIndexFromTemporarySkeletonList(t_SklIndex);

	if (t_ID == 0)
	{
		ClientLog::error("Failed to give skeleton an ID.");
	}
	m_LoadedSkeletons.push_back(t_ID);
}

/// @brief This support function is used for the manual allocation of the skeleton chains with means of a temporary skeleton.
/// Temporary Skeletons can be helpful when the user wants to modify the skeletons chains or nodes more than once before retargeting,
/// as they can be saved in core and retrieved later by the user to apply further modifications. This can be done easily with means of the Dev Tools.
/// The current function contains an example that shows how to create a temporary skeleton, modify its chains, nodes and 
/// skeleton info, and save it into Core. This process should be used during development.
/// After that, the final skeleton should be loaded into core for retargeting and for getting the actual data,as displayed in the LoadTestSkeleton 
/// function.
void SDKClient::BuildTemporarySkeleton()
{
	// define the session Id for which we want to save  
	// in this example we want to save a skeleton for the current session so we use our own Session Id
	uint32_t t_SessionId = m_SessionId;

	bool t_IsSkeletonModified = false; // this bool is set to true by the Dev Tools after saving any modification to the skeleton, 
	// this triggers the OnSyStemCallback which is used in the SDK to be notified about a change to its temporary skeletons.
	// for the purpose of this example setting this bool to true is not really necessary.

	// first create a skeleton setup of type Body
	uint32_t t_SklIndex = 0;

	SkeletonSettings t_Settings;
	SkeletonSettings_Init(&t_Settings);
	t_Settings.scaleToTarget = true;
	t_Settings.targetType = SkeletonTargetType::SkeletonTargetType_UserData;
	t_Settings.skeletonTargetUserData.userID = 0; // this needs to be a real user Id when retargeting, when editing the temporary skeleton this may (hopefully) not cause issues

	SkeletonSetupInfo t_SKL;
	SkeletonSetupInfo_Init(&t_SKL);
	t_SKL.id = 0;
	t_SKL.type = SkeletonType::SkeletonType_Body;
	t_SKL.settings = t_Settings;
	CopyString(t_SKL.name, sizeof(t_SKL.name), std::string("body"));

	SDKReturnCode t_Res = CoreSdk_CreateSkeletonSetup(t_SKL, &t_SklIndex);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Create Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}
	m_TemporarySkeletons.push_back(t_SklIndex);
	//Add 3 nodes to the skeleton setup
	t_Res = CoreSdk_AddNodeToSkeletonSetup(t_SklIndex, CreateNodeSetup(0, 0, 0, 0, 0, "root"));
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Add Node To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}

	t_Res = CoreSdk_AddNodeToSkeletonSetup(t_SklIndex, CreateNodeSetup(1, 0, 0, 1, 0, "branch"));
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Add Node To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}

	t_Res = CoreSdk_AddNodeToSkeletonSetup(t_SklIndex, CreateNodeSetup(2, 1, 0, 2, 0, "leaf"));
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Add Node To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}

	//Add one chain of type Leg to the skeleton setup
	ChainSettings t_ChainSettings;
	ChainSettings_Init(&t_ChainSettings);
	t_ChainSettings.usedSettings = ChainType::ChainType_Leg;
	t_ChainSettings.leg.footForwardOffset = 0;
	t_ChainSettings.leg.footSideOffset = 0;
	t_ChainSettings.leg.reverseKneeDirection = false;
	t_ChainSettings.leg.kneeRotationOffset = 0;

	ChainSetup t_Chain;
	ChainSetup_Init(&t_Chain);
	t_Chain.id = 0;
	t_Chain.type = ChainType::ChainType_Leg;
	t_Chain.dataType = ChainType::ChainType_Leg;
	t_Chain.dataIndex = 0;
	t_Chain.nodeIdCount = 3;
	t_Chain.nodeIds[0] = 0;
	t_Chain.nodeIds[1] = 1;
	t_Chain.nodeIds[2] = 2;
	t_Chain.settings = t_ChainSettings;
	t_Chain.side = Side::Side_Left;

	t_Res = CoreSdk_AddChainToSkeletonSetup(t_SklIndex, t_Chain);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Add Chain To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// save the temporary skeleton 
	t_Res = CoreSdk_SaveTemporarySkeleton(t_SklIndex, t_SessionId, t_IsSkeletonModified);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to save temporary skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// if we want to go on with the modifications to the same temporary skeleton 
	// get the skeleton
	t_Res = CoreSdk_GetTemporarySkeleton(t_SklIndex, t_SessionId);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get temporary skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// now add second chain to the same temporary skeleton
	t_ChainSettings.usedSettings = ChainType::ChainType_Head;

	t_Chain.id = 1;
	t_Chain.type = ChainType::ChainType_Head;
	t_Chain.dataType = ChainType::ChainType_Head;
	t_Chain.dataIndex = 0;
	t_Chain.nodeIdCount = 1;
	t_Chain.nodeIds[0] = 0;
	t_Chain.settings = t_ChainSettings;
	t_Chain.side = Side::Side_Center;

	t_Res = CoreSdk_AddChainToSkeletonSetup(t_SklIndex, t_Chain);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Add Chain To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// save the temporary skeleton 
	t_Res = CoreSdk_SaveTemporarySkeleton(t_SklIndex, t_SessionId, t_IsSkeletonModified);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to save temporary skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// get the skeleton info (number of nodes and chains for that skeleton)
	SkeletonSetupArraySizes t_SkeletonInfo;
	t_Res = CoreSdk_GetSkeletonSetupArraySizes(t_SklIndex, &t_SkeletonInfo);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get info about skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// now get the chain data
	ChainSetup* t_Chains = new ChainSetup[t_SkeletonInfo.chainsCount];
	t_Res = CoreSdk_GetSkeletonSetupChainsArray(t_SklIndex, t_Chains, t_SkeletonInfo.chainsCount);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get skeleton setup chains. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// get the node data 
	NodeSetup* t_Nodes = new NodeSetup[t_SkeletonInfo.nodesCount];
	t_Res = CoreSdk_GetSkeletonSetupNodesArray(t_SklIndex, t_Nodes, t_SkeletonInfo.nodesCount);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get skeleton setup nodes. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// just as an example try to get the skeleton setup info
	SkeletonSetupInfo t_SKeletonSetupInfo;
	SkeletonSetupInfo_Init(&t_SKeletonSetupInfo);
	t_Res = CoreSdk_GetSkeletonSetupInfo(t_SklIndex, &t_SKeletonSetupInfo);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to overwrite Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// if we want to modify the skeleton setup or if we want to apply some changes to the chains or nodes:
	// first overwrite the existing skeleton setup and then re-add all the chains and nodes to it
	SkeletonSettings_Init(&t_Settings);
	t_Settings.targetType = SkeletonTargetType::SkeletonTargetType_GloveData;

	t_SKL.settings = t_Settings;
	CopyString(t_SKL.name, sizeof(t_SKL.name), std::string("body2"));

	// this way we overwrite the temporary skeleton with index t_SklIndex with the modified skeleton setup
	t_Res = CoreSdk_OverwriteSkeletonSetup(t_SklIndex, t_SKL);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to overwrite Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// modify chains and nodes
	t_Chains[0].side = Side::Side_Right;
	t_Nodes[0].type = NodeType::NodeType_Mesh;

	// add all the existing nodes to the new skeleton setup
	for (size_t i = 0; i < t_SkeletonInfo.nodesCount; i++)
	{
		t_Res = CoreSdk_AddNodeToSkeletonSetup(t_SklIndex, t_Nodes[i]);
		if (t_Res != SDKReturnCode::SDKReturnCode_Success)
		{
			ClientLog::error("Failed to Add Node To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
			return;
		}
	}

	// then add all the existing chains to the new skeleton setup
	for (size_t i = 0; i < t_SkeletonInfo.chainsCount; i++)
	{
		t_Res = CoreSdk_AddChainToSkeletonSetup(t_SklIndex, t_Chains[i]);
		if (t_Res != SDKReturnCode::SDKReturnCode_Success)
		{
			ClientLog::error("Failed to Add Chains To Skeleton Setup. The error given was {}.", (int32_t)t_Res);
			return;
		}
	}

	// cleanup
	delete[] t_Chains;
	delete[] t_Nodes;

	// save temporary skeleton 
	// in the Dev Tools this bool is set to true when saving the temporary skeleton, this triggers OnSystemCallback which 
	// notifies the SDK sessions about a modifications to one of their temporary skeletons.
	// setting the bool to true in this example is not really necessary, it's just for testing purposes.
	t_IsSkeletonModified = true;
	t_Res = CoreSdk_SaveTemporarySkeleton(t_SklIndex, t_SessionId, t_IsSkeletonModified);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to save temporary skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}
}

/// @brief This support function is used to clear a temporary skeleton from the temporary skeleton list, for example it can be used when the 
/// max number of temporary skeletons has been reached for a specific session.
void SDKClient::ClearTemporarySkeleton()
{
	// clear the first element of the temporary skeleton list
	if (m_TemporarySkeletons.size() == 0)
	{
		ClientLog::error("There are no Temporary Skeletons to clear!");
		return;
	}
	uint32_t t_SklIndex = m_TemporarySkeletons[0];
	SDKReturnCode t_Res = CoreSdk_ClearTemporarySkeleton(t_SklIndex, m_SessionId);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Clear Temporary Skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}
	m_TemporarySkeletons.erase(m_TemporarySkeletons.begin());
}

/// @brief This support function is used to clear all temporary skeletons associated to the current SDK session, it can be used when the 
/// max number of temporary skeletons has been reached for the current session and we want to make room for more.
void SDKClient::ClearAllTemporarySkeletons()
{
	if (m_TemporarySkeletons.size() == 0)
	{
		ClientLog::error("There are no Temporary Skeletons to clear!");
		return;
	}
	SDKReturnCode t_Res = CoreSdk_ClearAllTemporarySkeletons();
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Clear All Temporary Skeletons. The error given was {}.", (int32_t)t_Res);
		return;
	}
	m_TemporarySkeletons.clear();
}

void SDKClient::SaveTemporarySkeletonToFile()
{
	// this example shows how to save a temporary skeleton to a file
	// first create a temporary skeleton:

	// define the session Id for which we want to save  
	uint32_t t_SessionId = m_SessionId;

	bool t_IsSkeletonModified = false; // setting this bool to true is not necessary here, it is mostly used by the Dev Tools
	// to notify the SDK sessions about their skeleton being modified.

	// first create a skeleton setup
	uint32_t t_SklIndex = 0;

	SkeletonSetupInfo t_SKL;
	SkeletonSetupInfo_Init(&t_SKL);
	t_SKL.type = SkeletonType::SkeletonType_Hand;
	t_SKL.settings.scaleToTarget = true;
	t_SKL.settings.targetType = SkeletonTargetType::SkeletonTargetType_GloveData;
	t_SKL.settings.skeletonTargetUserIndexData.userIndex = 0;

	CopyString(t_SKL.name, sizeof(t_SKL.name), std::string("LeftHand"));

	SDKReturnCode t_Res = CoreSdk_CreateSkeletonSetup(t_SKL, &t_SklIndex);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Create Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}
	m_TemporarySkeletons.push_back(t_SklIndex);

	// setup nodes and chains for the skeleton hand
	if (!SetupHandNodes(t_SklIndex, Side_Left)) return;
	if (!SetupHandChains(t_SklIndex, Side_Left)) return;

	// save the temporary skeleton 
	t_Res = CoreSdk_SaveTemporarySkeleton(t_SklIndex, t_SessionId, t_IsSkeletonModified);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to save temporary skeleton. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// now compress the temporary skeleton data and get the size of the compressed data:
	uint32_t t_TemporarySkeletonLengthInBytes;

	t_Res = CoreSdk_CompressTemporarySkeletonAndGetSize(t_SklIndex, t_SessionId, &t_TemporarySkeletonLengthInBytes);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to compress temporary skeleton and get size. The error given was {}.", (int32_t)t_Res);
		return;
	}
	unsigned char* t_TemporarySkeletonData = new unsigned char[t_TemporarySkeletonLengthInBytes];

	// get the array of bytes with the compressed temporary skeleton data, remember to always call function CoreSdk_CompressTemporarySkeletonAndGetSize
	// before trying to get the compressed temporary skeleton data
	t_Res = CoreSdk_GetCompressedTemporarySkeletonData(t_TemporarySkeletonData, t_TemporarySkeletonLengthInBytes);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to get compressed temporary skeleton data. The error given was {}.", (int32_t)t_Res);
		return;
	}

	// now save the data into a .mskl file
	// as an example we save the temporary skeleton in a folder called ManusTemporarySkeleton inside the documents directory
	// get the path for the documents directory
	std::string t_DirectoryPathString = GetDocumentsDirectoryPath_UTF8();

	// create directory name and file name for storing the temporary skeleton
	std::string t_DirectoryPath =
		t_DirectoryPathString
		+ s_SlashForFilesystemPath
		+ "ManusTemporarySkeleton";

	CreateFolderIfItDoesNotExist(t_DirectoryPath);

	std::string t_DirectoryPathAndFileName =
		t_DirectoryPath
		+ s_SlashForFilesystemPath
		+ "TemporarySkeleton.mskl";

	// write the temporary skeleton data to .mskl file
	std::ofstream t_File = GetOutputFileStream(t_DirectoryPathAndFileName);
	t_File.write((char*)t_TemporarySkeletonData, t_TemporarySkeletonLengthInBytes);
	t_File.close();

	t_Res = CoreSdk_ClearTemporarySkeleton(t_SklIndex, t_SessionId);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Clear Temporary Skeleton after saving. The error given was {}.", (int32_t)t_Res);
		return;
	}
	RemoveIndexFromTemporarySkeletonList(t_SklIndex);
}


void SDKClient::GetTemporarySkeletonFromFile()
{
	// this example shows how to load a temporary skeleton data from a file

	// as an example we try to get the temporary skeleton data previously saved as .mskl file in directory Documents/ManusTemporarySkeleton
	// get the path for the documents directory
	std::string t_DirectoryPathString = GetDocumentsDirectoryPath_UTF8();

	// check if directory exists
	std::string t_DirectoryPath =
		t_DirectoryPathString
		+ s_SlashForFilesystemPath
		+ "ManusTemporarySkeleton";

	if (!DoesFolderOrFileExist(t_DirectoryPath))
	{
		ClientLog::warn("Failed to read from client file, the mentioned directory does not exist");
		return;
	}

	// create string with file name
	std::string t_DirectoryPathAndFileName =
		t_DirectoryPath
		+ s_SlashForFilesystemPath
		+ "TemporarySkeleton.mskl";

	// read from file
	std::ifstream t_File = GetInputFileStream(t_DirectoryPathAndFileName);

	if (!t_File)
	{
		ClientLog::warn("Failed to read from client file, the file does not exist in the mentioned directory");
		return;
	}

	// get file dimension
	t_File.seekg(0, t_File.end);
	int t_FileLength = (int)t_File.tellg();
	t_File.seekg(0, t_File.beg);

	// get temporary skeleton data from file
	unsigned char* t_TemporarySkeletonData = new unsigned char[t_FileLength];
	t_File.read((char*)t_TemporarySkeletonData, t_FileLength);
	t_File.close();


	// save the zipped temporary skeleton information, they will be used internally for sending the data to Core
	uint32_t t_TemporarySkeletonLengthInBytes = t_FileLength;

	if (t_TemporarySkeletonData == nullptr)
	{
		ClientLog::warn("Failed to read the compressed temporary skeleton data from file");
		delete[] t_TemporarySkeletonData;
		return;
	}

	// create a skeleton setup where we will store the temporary skeleton retrieved from file
	SkeletonSetupInfo t_SKL;
	SkeletonSetupInfo_Init(&t_SKL);
	uint32_t t_SklIndex = 0;
	SDKReturnCode t_Res = CoreSdk_CreateSkeletonSetup(t_SKL, &t_SklIndex);
	if (t_Res != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to Create Skeleton Setup. The error given was {}.", (int32_t)t_Res);
		return;
	}
	m_TemporarySkeletons.push_back(t_SklIndex);

	// associate the retrieved temporary skeleton to the current session id
	uint32_t t_SessionId = m_SessionId;

	// load the temporary skeleton data retrieved from the zipped file and save it with index t_SklIndex and session id of the current session
	SDKReturnCode t_Result = CoreSdk_GetTemporarySkeletonFromCompressedData(t_SklIndex, t_SessionId, t_TemporarySkeletonData, t_TemporarySkeletonLengthInBytes);
	if (t_Result != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::warn("Failed to load temporary skeleton data from client file in Core, the error code was: {}.", (int32_t)t_Result);
		return;
	}

	delete[] t_TemporarySkeletonData;
}

void SDKClient::TestTimestamp()
{
	ManusTimestamp t_TS;
	ManusTimestamp_Init(&t_TS);
	ManusTimestampInfo t_TSInfo;
	ManusTimestampInfo_Init(&t_TSInfo);
	t_TSInfo.fraction = 30;
	t_TSInfo.second = 15;
	t_TSInfo.minute = 30;
	t_TSInfo.hour = 12;
	t_TSInfo.day = 1;
	t_TSInfo.month = 8;
	t_TSInfo.year = 2012;
	t_TSInfo.timecode = true;

	CoreSdk_SetTimestampInfo(&t_TS, t_TSInfo);

	ManusTimestampInfo t_TSInfo2;
	CoreSdk_GetTimestampInfo(t_TS, &t_TSInfo2);
}

void SDKClient::RemoveIndexFromTemporarySkeletonList(uint32_t p_Idx)
{
	for (int i = 0; i < m_TemporarySkeletons.size(); i++)
	{
		if (m_TemporarySkeletons[i] == p_Idx)
		{
			m_TemporarySkeletons.erase(m_TemporarySkeletons.begin() + i);
		}
	}
}

void SDKClient::PairGlove()
{
	bool t_GloveFound = false;
	uint32_t t_GloveToPairId = -1;

	for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
	{
		if (m_Landscape->gloveDevices.gloves[i].pairedState == DevicePairedState_Unpaired)
		{
			t_GloveToPairId = m_Landscape->gloveDevices.gloves[i].id;
			t_GloveFound = true;
			break;
		}
	}

	if (!t_GloveFound)
	{
		ClientLog::warn("No glove available for pairing found.");
		return;
	}

	bool t_Paired;
	SDKReturnCode t_PairGloveResult = CoreSdk_PairGlove(t_GloveToPairId, &t_Paired);
	if (t_PairGloveResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to pair glove with ID: {}. The error given was {}", t_GloveToPairId, (int32_t)t_PairGloveResult);
		return;
	}

	if (!t_Paired)
	{
		ClientLog::error("Failed to pair glove with ID: {}", t_GloveToPairId);
		return;
	}

	ClientLog::print("Succesfully paired glove with ID: {}", t_GloveToPairId);

	return;
}

void SDKClient::UnpairGlove()
{
	bool t_GloveFound = false;
	uint32_t t_GloveToUnPairId = -1;

	for (size_t i = 0; i < m_Landscape->gloveDevices.gloveCount; i++)
	{
		if (m_Landscape->gloveDevices.gloves[i].id == m_Landscape->gloveDevices.gloves[i].dongleID)
			continue;

		if (m_Landscape->gloveDevices.gloves[i].pairedState == DevicePairedState_Paired)
		{
			t_GloveToUnPairId = m_Landscape->gloveDevices.gloves[i].id;
			t_GloveFound = true;
			break;
		}
	}

	if (!t_GloveFound)
	{
		ClientLog::error("No glove to unpair found");
		return;
	}

	bool t_Unpaired;
	SDKReturnCode t_UnpairGloveResult = CoreSdk_UnpairGlove(t_GloveToUnPairId, &t_Unpaired);
	if (t_UnpairGloveResult != SDKReturnCode::SDKReturnCode_Success)
	{
		ClientLog::error("Failed to unpair glove with ID: {}. The error given was ", t_GloveToUnPairId, (int32_t)t_UnpairGloveResult);
		return;
	}

	ClientLog::print("Succesfully unpaired glove with ID: {}", t_GloveToUnPairId);
	return;
}
/// @brief Prints the type of the first chain generated by the AllocateChain function, this is used for testing.
void SDKClient::PrintLogs()
{
	std::vector<SDKLog*> t_SDKLogs;
	s_Instance->m_LogMutex.lock();
	t_SDKLogs.swap(s_Instance->m_Logs);
	s_Instance->m_LogMutex.unlock();

	for (size_t i = 0; i < t_SDKLogs.size(); i++)
	{
		ClientLog::print(t_SDKLogs[i]->string.c_str());
		delete t_SDKLogs[i];
	}
}
