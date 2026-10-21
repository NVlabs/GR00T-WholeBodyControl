#include "ClientPlatformSpecific.hpp"

// signal types
#include <csignal>
// std::filesystem
#include <filesystem>
// std::*fstream
#include <fstream>
// CoreSdk_ShutDown
#include "ManusSDK.h"
// std::map
#include <map>
// Ncurses terminal functions.
#include <ncursesw/ncurses.h>
// Termios terminal functions.
#include <termios.h>
#include <cstring>
#include <unistd.h>
// spdlog replacement
#include "ClientLogging.hpp"

using namespace ManusSDK;

/// @brief Reset a signal handler to its default, and then call it.
/// For signal types and explanation, see:
/// https://www.gnu.org/software/libc/manual/html_node/Standard-Signals.html
#define CALL_DEFAULT_SIGNAL_HANDLER(p_SignalType) \
	/* Reset the handler for this signal to the default. */ \
	signal(p_SignalType, SIG_DFL); \
	/* Re-raise this signal, causing the normal handler to run. */ \
	raise(p_SignalType);

const std::string SDKClientPlatformSpecific::s_SlashForFilesystemPath = "/";

/// @brief Handle a signal telling the SDK client to quit.
/// A generic signal used to "politely ask a program to terminate".
/// On Linux, this can be sent by using the Gnome System Monitor and telling
/// the SDK client process to end.
static void HandleTerminationSignal(int p_Parameter)
{
	ClientLog::error(
		"Termination signal sent with parameter {}.",
		p_Parameter);

	CoreSdk_ShutDown();

	CALL_DEFAULT_SIGNAL_HANDLER(SIGTERM);
}

/// @brief Handle an interrupt signal.
/// Called when the INTR character is typed - usually ctrl + c.
static void HandleInterruptSignal(int p_Parameter)
{
	ClientLog::error(
		"Interrupt signal sent with parameter {}.",
		p_Parameter);

	CoreSdk_ShutDown();

	CALL_DEFAULT_SIGNAL_HANDLER(SIGINT);
}

/// @brief Handle a quit signal.
/// Called when the QUIT character is typed - usually ctrl + \.
static void HandleQuitSignal(int p_Parameter)
{
	ClientLog::error(
		"Quit signal sent with parameter {}.",
		p_Parameter);

	CoreSdk_ShutDown();

	CALL_DEFAULT_SIGNAL_HANDLER(SIGQUIT);
}

/// @brief Handle a hangup signal.
/// Called to report that the user's terminal has disconnected.
/// This can happen when connecting over the network, for example.
/// It also happens if the terminal window is closed while debugging.
static void HandleHangupSignal(int p_Parameter)
{
	ClientLog::error(
		"Hang-up signal sent with parameter {}.",
		p_Parameter);

	CoreSdk_ShutDown();

	CALL_DEFAULT_SIGNAL_HANDLER(SIGHUP);
}

/// @brief Initialise Ncurses so that we can check for input.
static bool InitializeNcurses(void)
{
	const WINDOW* const t_Window = initscr();
	if (!t_Window)
	{
		ClientLog::error("Failed to initialise the screen.");

		return false;
	}

	// Don't buffer input until a newline or carriage return is typed.
	if (cbreak() != OK)
	{
		ClientLog::error("Failed to make input unbuffered.");

		return false;
	}

	// Don't echo input.
	if (noecho() != OK)
	{
		ClientLog::error("Failed to disable input echoing.");

		return false;
	}

	// Don't make newlines when the return key is pressed.
	if (nonl() != OK)
	{
		ClientLog::error("Failed to disable newlines.");

		return false;
	}

	// Do not flush the screen when interrupt/break/quit is pressed.
	if (intrflush(stdscr, FALSE) != OK)
	{
		ClientLog::error("Failed to disable screen flushing.");

		return false;
	}

	// Make getch non-blocking.
	if (nodelay(stdscr, TRUE) != OK)
	{
		ClientLog::error("Failed to make nodelay non-blocking.");

		return false;
	}

	// Enable handling the keypad ("function keys" like the arrow keys).
	if (keypad(stdscr, TRUE) != OK)
	{
		ClientLog::error("Failed to enable keypad input.");

		return false;
	}

	return true;
}

/// @brief Initialise Termios so that terminal output is correct.
static bool InitializeTermios(void)
{
	// For some reason, printf and spdlog strings require a carriage return
	// in addition to a newline after running initscr().
	// This code sets the "output modes" flag to treat newline characters
	// as newline+carriage-return characters.
	// https://arstechnica.com/civis/viewtopic.php?t=70699
	termios t_Settings;
	if (tcgetattr(STDIN_FILENO, &t_Settings) != 0)
	{
		ClientLog::error("Failed to get Termios settings.");

		return false;
	}

	t_Settings.c_oflag |= ONLCR;
	if (tcsetattr(0, TCSANOW, &t_Settings) != 0)
	{
		ClientLog::error("Failed to set Termios settings.");

		return false;
	}

	return true;
}

/// @brief Register our signal handling functions.
static bool SetUpSignalHandlers(void)
{
	{
		const __sighandler_t t_OldTerminationHandler = signal(
			SIGTERM,
			HandleTerminationSignal);
		if (t_OldTerminationHandler == SIG_ERR)
		{
			ClientLog::error("Failed to set termination signal handler.");
			return false;
		}
	}

	{
		const __sighandler_t t_OldInterruptHandler = signal(
			SIGINT,
			HandleInterruptSignal);
		if (t_OldInterruptHandler == SIG_ERR)
		{
			ClientLog::error("Failed to set interrupt signal handler.");
			return false;
		}
	}

	{
		const __sighandler_t t_OldQuitHandler = signal(
			SIGQUIT,
			HandleQuitSignal);
		if (t_OldQuitHandler == SIG_ERR)
		{
			ClientLog::error("Failed to set quit signal handler.");
			return false;
		}
	}

	{
		const __sighandler_t t_OldHangupHandler = signal(
			SIGHUP,
			HandleHangupSignal);
		if (t_OldHangupHandler == SIG_ERR)
		{
			ClientLog::error("Failed to set hang-up signal handler.");
			return false;
		}
	}

	return true;
}

/// @brief Handles keyboard input.
/// Requires things to be set up with Ncurses and Termios.
class ClientInput
{
public:
	/// @brief Update the state of the keyboard.
	void Update(void)
	{
		// Reset the state of the last update.
		for (
			InputMap_t::iterator t_Key = m_PressedLastUpdate.begin();
			t_Key != m_PressedLastUpdate.end();
			t_Key++)
		{
			t_Key->second = false;
		}

		// Copy the current state to the last update's state, and clear the
		// current state.
		for (
			InputMap_t::iterator t_Key = m_CurrentlyPressed.begin();
			t_Key != m_CurrentlyPressed.end();
			t_Key++)
		{
			m_PressedLastUpdate[t_Key->first] =
				t_Key->second;
			t_Key->second = false;
		}

		// Get the new state.
		int t_Ch = getch();
		while (t_Ch != ERR)
		{
			if (t_Ch >= 'a' && t_Ch <= 'z')
			{
				// Unlike with Windows' GetAsyncKeyState(), upper case and
				// lower case characters have different key numbers with
				// getch().
				// Since all WasKeyPressed calls (as of writing this) use
				// upper case, lower case keys need to be converted to work
				// on Linux.
				// Note that this does break the ability to check for lower
				// case key presses.
				t_Ch = toupper(t_Ch);
			}

			m_CurrentlyPressed[t_Ch] = true;

			t_Ch = getch();
		}
	}

	/// @brief Get the key's current state.
	/// Note that unlike IsPressed(), this also stores the result for use in
	/// the next key state check.
	bool GetKey(const int p_Key)
	{
		bool t_IsPressed = IsPressed(p_Key);
		m_PreviousKeyState[p_Key] = t_IsPressed;

		return t_IsPressed;
	}

	/// @brief Was this key pressed since the last check?
	/// Note that unlike WasJustPressed(), this checks if the key was pressed
	/// since the last time a GetKey* function was called.
	bool GetKeyDown(const int p_Key)
	{
		const bool t_IsPressed = IsPressed(p_Key);

		const auto t_PreviousState = m_PreviousKeyState.find(p_Key);
		const bool t_PreviousValue =
			t_PreviousState == m_PreviousKeyState.end()
				? false
				: t_PreviousState->second;

		const bool t_Down = t_IsPressed && !t_PreviousValue;
		m_PreviousKeyState[p_Key] = t_Down;

		return t_Down;
	}

	/// @brief Was this key released since the last check?
	/// Note that unlike WasJustReleased(), this checks if the key was released
	/// since the last time a GetKey* function was called.
	bool GetKeyUp(const int p_Key)
	{
		const bool t_IsPressed = IsPressed(p_Key);

		const auto t_PreviousState = m_PreviousKeyState.find(p_Key);
		const bool t_PreviousValue =
			t_PreviousState == m_PreviousKeyState.end()
				? false
				: t_PreviousState->second;

		const bool t_Up = !t_IsPressed && t_PreviousValue;
		m_PreviousKeyState[p_Key] = t_Up;

		return t_Up;
	}

private:
	/// @brief Get the key's current state.
	/// Note that unlike GetKey(), this does not store the result for use in
	/// the next key state check.
	bool IsPressed(const int p_Key) const
	{
		auto t_CurrentlyPressed = m_CurrentlyPressed.find(p_Key);
		if (t_CurrentlyPressed == m_CurrentlyPressed.end())
		{
			return false;
		}

		return t_CurrentlyPressed->second;
	}

	/// @brief Was this key pressed since the last input update?
	/// Note that unlike GetKeyDown(), this function will return the same value
	/// until the next keyboard state update.
	bool WasJustPressed(const int p_Key) const
	{
		return !WasPressedLastUpdate(p_Key) && IsPressed(p_Key);
	}

	/// @brief Was this key released since the last input update?
	/// Note that unlike GetKeyUp(), this function will return the same value
	/// until the next keyboard state update.
	bool WasJustReleased(const int p_Key) const
	{
		return WasPressedLastUpdate(p_Key) && !IsPressed(p_Key);
	}

	/// @brief Was this key pressed the previous input update?
	bool WasPressedLastUpdate(const int p_Key) const
	{
		auto t_StateLastUpdate = m_PressedLastUpdate.find(p_Key);
		if (t_StateLastUpdate == m_PressedLastUpdate.end())
		{
			return false;
		}

		return t_StateLastUpdate->second;
	}

	typedef std::map<int, bool> InputMap_t;
	InputMap_t m_CurrentlyPressed;
	InputMap_t m_PressedLastUpdate;
	// The Windows GetKey* functions only update the key state when a GetKey*
	// function gets called. To make the Linux input work the same way, this
	// map is used.
	InputMap_t m_PreviousKeyState;
};

static ClientInput g_Input;

bool SDKClientPlatformSpecific::PlatformSpecificInitialization(void)
{
	const bool t_NcursesResult = InitializeNcurses();
	const bool t_TermiosResult = InitializeTermios();
	const bool t_SignalResult = SetUpSignalHandlers();

	return t_NcursesResult && t_TermiosResult && t_SignalResult;
}

bool SDKClientPlatformSpecific::PlatformSpecificShutdown(void)
{
	// Ncurses.
	endwin();

	return true;
}

void SDKClientPlatformSpecific::UpdateInput(void)
{
	g_Input.Update();
}

/*static*/ bool SDKClientPlatformSpecific::CopyString(
	char* const p_Target,
	const size_t p_MaxLengthThatWillFitInTarget,
	const std::string& p_Source)
{
	if (!p_Target)
	{
		ClientLog::error(
			"Tried to copy a string, but the target was null. The string was \"{}\".",
			p_Source.c_str());

		return false;
	}

	if (p_MaxLengthThatWillFitInTarget == 0)
	{
		ClientLog::error(
			"Tried to copy a string, but the target's size is zero. The string was \"{}\".",
			p_Source.c_str());

		return false;
	}

	if (p_MaxLengthThatWillFitInTarget <= p_Source.length())
	{
		ClientLog::error(
			"Tried to copy a string that was longer than {} characters, which makes it too big for its target buffer. The string was \"{}\".",
			p_MaxLengthThatWillFitInTarget,
			p_Source.c_str());

		return false;
	}

	strcpy(p_Target, p_Source.c_str());

	return true;
}

bool SDKClientPlatformSpecific::ResizeWindow(
	const short int p_ConsoleWidth,
	const short int p_ConsoleHeight,
	const short int p_ConsoleScrollback)
{
	// https://apple.stackexchange.com/questions/33736/can-a-terminal-window-be-resized-with-a-terminal-command/47841#47841
	// Use a control sequence to resize the window.
	// Seems to be supported by the default terminal used in Gnome, as well
	// as Mac OS.
	// \\e[ -> ASCII ESC character (number 27, or 0x1B)
	//      -> control sequence introducer
	// 8;   -> resize the window
	// y;xt -> The first number is the height, the second the width.
	// Scrollback can't and doesn't need to be set here for Linux.
	printf("\e[8;%d;%dt", p_ConsoleHeight, p_ConsoleWidth);

	ClearConsole();

	// None of the ncurses functions for resizing the terminal actually
	// seem to do anything.
	/*if (resizeterm(180, 180) != OK)
	{
		return false;
	}*/

	return true;
}

void SDKClientPlatformSpecific::ApplyConsolePosition(
	const int p_ConsoleCurrentOffset)
{
	printf("\e[%d;1H", p_ConsoleCurrentOffset);
}

/*static*/ void SDKClientPlatformSpecific::ClearConsole(void)
{
	// https://stackoverflow.com/questions/4062045/clearing-terminal-in-linux-with-c-code
	// Use a control sequence to clear the terminal and move the cursor.
	// Seems to be supported by the default terminal used in Gnome,
	// as well as Mac OS.
	// \\e[ -> ASCII ESC character (number 27, or 0x1B) -> control sequence introducer
	// 2    -> the entire screen
	// J    -> clear the screen
	//printf("\e[2J\n");

	// Move the cursor to row 1 column 1.
	//printf("\e[1;1H\n");

	if (clear() != OK)
	{
		ClientLog::error("Failed to clear the screen.");
	}

	refresh();
}

bool SDKClientPlatformSpecific::GetKey(const int p_Key)
{
	return g_Input.GetKey(p_Key);
}

bool SDKClientPlatformSpecific::GetKeyDown(const int p_Key)
{
	return g_Input.GetKeyDown(p_Key);
}

bool SDKClientPlatformSpecific::GetKeyUp(const int p_Key)
{
	return g_Input.GetKeyUp(p_Key);
}

std::string SDKClientPlatformSpecific::GetDocumentsDirectoryPath_UTF8(void)
{
	const char* const t_Xdg = getenv("XDG_DOCUMENTS_DIR");

	// Backup - the documents folder is usually going to be in $HOME/Documents.
	const char* const t_Home = getenv("HOME");
	if (!t_Xdg && !t_Home)
	{
		return std::string("");
	}

	const std::string t_DocumentsDir =
		(!t_Xdg || strlen(t_Xdg) == 0)
			? std::string(t_Home) + std::string("/Documents")
			: std::string(t_Xdg);

	return t_DocumentsDir;
}

std::ifstream SDKClientPlatformSpecific::GetInputFileStream(
	std::string p_Path_UTF8)
{
	return std::ifstream(p_Path_UTF8, std::ifstream::binary);
}

std::ofstream SDKClientPlatformSpecific::GetOutputFileStream(
	std::string p_Path_UTF8)
{
	return std::ofstream(p_Path_UTF8, std::ofstream::binary);
}

bool SDKClientPlatformSpecific::DoesFolderOrFileExist(std::string p_Path_UTF8)
{
	return std::filesystem::exists(p_Path_UTF8);
}

void SDKClientPlatformSpecific::CreateFolderIfItDoesNotExist(
	std::string p_Path_UTF8)
{
	if (!DoesFolderOrFileExist(p_Path_UTF8))
	{
		std::filesystem::create_directory(p_Path_UTF8);
	}
}
