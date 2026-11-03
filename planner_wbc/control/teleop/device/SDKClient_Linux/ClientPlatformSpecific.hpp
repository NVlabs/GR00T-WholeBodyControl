#ifndef _CLIENT_PLATFORM_SPECIFIC_HPP_
#define _CLIENT_PLATFORM_SPECIFIC_HPP_

#include "ClientPlatformSpecificTypes.hpp"

// size_t
#include <cstddef>
// std::string
#include <string>

// Set up a Doxygen group.
/** @addtogroup SDKClient
 *  @{
 */

class SDKClientPlatformSpecific
{
protected:
	/// @brief Initialise things only needed for this platform.
	bool PlatformSpecificInitialization(void);

	/// @brief Shut down things only needed for this platform.
	bool PlatformSpecificShutdown(void);

	/// @brief Update the current keyboard state.
	void UpdateInput(void);

	/// @brief Copy the given string into the given target.
	static bool CopyString(
		char* const p_Target,
		const size_t p_MaxLengthThatWillFitInTarget,
		const std::string& p_Source);

	/// @brief Resize the window, so the log messages will fit.
	/// Make sure not to enter illegal sizes in here or SetConsoleWindowInfo
	/// and SetConsoleScreenBufferSize can generate an error.
	/// This is not code directly needed for the SDK to function, but its handy
	/// for a quick and simple output for this client.
	bool ResizeWindow(
		const short int p_ConsoleWidth,
		const short int p_ConsoleHeight,
		const short int p_ConsoleScrollback);

	/// @brief Set the current console position.
	/// This handles the platform-specific part of advancing the console
	/// position.
	void ApplyConsolePosition(
		const int p_ConsoleCurrentOffset);

	/// @brief Clear the console window.
	static void ClearConsole(void);

	/// @brief Gets key's current state.
	/// True is pressed false is not pressed.
	bool GetKey(const int p_Key);

	/// @brief Returns true first time it is called when key is pressed.
	bool GetKeyDown(const int p_Key);

	/// @brief Returns true first time it is called when key is released.
	bool GetKeyUp(const int p_Key);

	/// @brief Get the path to the user's Documents folder.
	/// The string should be in UTF-8 format.
	std::string GetDocumentsDirectoryPath_UTF8(void);

	/// @brief Get an input stream for the given file.
	/// The file's path should be in UTF-8 format.
	std::ifstream GetInputFileStream(std::string p_Path_UTF8);

	/// @brief Get an output stream for the given file.
	/// The file's path should be in UTF-8 format.
	std::ofstream GetOutputFileStream(std::string p_Path_UTF8);

	/// @brief Check if the given folder or file exists.
	/// The folder path given should be in UTF-8 format.
	bool DoesFolderOrFileExist(std::string p_Path_UTF8);

	/// @brief Create the given folder if it does not exist.
	/// The folder path given should be in UTF-8 format.
	void CreateFolderIfItDoesNotExist(std::string p_Path_UTF8);

	/// @brief The slash character that is used in the filesystem.
	static const std::string s_SlashForFilesystemPath;
};

// Close the Doxygen group.
/** @} */

#endif
