/**
 * @file PXREARobotSDK.h
 * @brief Robot SDK header file for client-side communication
 */
#ifndef PXREACLIENTSDK_H
#define PXREACLIENTSDK_H
#ifdef _WIN32
#if defined(PXREACLIENTSDK_LIBRARY)
#  define PXREACLIENTSDK_EXPORT __declspec(dllexport)
#else
#  define PXREACLIENTSDK_EXPORT __declspec(dllimport)
#endif
#endif

#ifdef __linux__
#if defined(PXREACLIENTSDK_LIBRARY)
#  define PXREACLIENTSDK_EXPORT __attribute__((visibility("default")))
#else
#  define PXREACLIENTSDK_EXPORT __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum PXREAClientCallbackType
{
    /// @brief Server connected
    PXREAServerConnect          = 1<<2,
    /// @brief Server disconnected
    PXREAServerDisconnect       = 1<<3,
    /// @brief Device online
    PXREADeviceFind             = 1<<4,
    /// @brief Device offline
    PXREADeviceMissing          = 1<<5,
    /// @brief Device connected
    PXREADeviceConnect          = 1<<9,
    /// @brief Device state in JSON format
    PXREADeviceStateJson        = 1<<25,
    /// @brief Custom message
    PXREADeviceCustomMessage    = 1<<26,
    /// @brief Mask for enabling all callbacks
    PXREAFullMask               = 0xffffffff
};








/// @brief Device state in JSON format
typedef struct {
    /// @brief Device serial number
    char devID[32];
    /// @brief JSON string containing device state information
    char stateJson[16352];
}PXREADevStateJson;


typedef struct {
    /// @brief Device serial number
    char devID[32];
    /// @brief Data size
    uint64_t dataSize;
    /// @brief Data pointer, valid within callback
    const char* dataPtr;
}PXREADevCustomMessage;




/**
 * @brief Client callback for receiving server messages
 * @param context Callback context, passed from #Init parameter 1 context
 * @param type Callback type
 * @param status Callback status code
 * @param userData Callback data pointer, determined by parameter 2 type
 */
typedef void(*pfPXREAClientCallback)(void* context,PXREAClientCallbackType type,int status,void* userData);

/**
 * @brief SDK initialization interface
 * @details Connect to service and register callback
 * @param context Callback context for passing user-defined data to callback function
 * @param cliCallback Callback function pointer for listening to server messages
 * @param mask Callback mask for filtering certain server messages
 */
PXREACLIENTSDK_EXPORT int PXREAInit(void* context,pfPXREAClientCallback cliCallback,unsigned mask);
/**
 * @brief Termination interface
 * @details Disconnect from service
 */
PXREACLIENTSDK_EXPORT int PXREADeinit();

/**
 * @brief Send JSON format command to device
 * @param devID Device serial number
 * @param parameterJson Function and parameters in JSON format, refer to robot SDK documentation for specific usage
 * @return 0 Success
 * @return -1 Failure
 */
PXREACLIENTSDK_EXPORT int PXREADeviceControlJson(const char *devID,const char *parameterJson);
/**
 * @brief Send byte stream to specified device
 * @note This command is suitable for SDK caller's custom messages
 * @param devID Device serial number
 * @param data Starting address of byte stream
 * @param len Length of byte stream
 * @return 0 Success
 * @return -1 Failure
 */
PXREACLIENTSDK_EXPORT int PXREASendBytesToDevice(const char* devID,const char* data,unsigned len);
#ifdef __cplusplus
}
#endif

#endif // PXREACLIENTSDK_H
