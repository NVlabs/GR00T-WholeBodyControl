#ifndef _CLIENT_PLATFORM_SPECIFIC_TYPES_HPP_
#define _CLIENT_PLATFORM_SPECIFIC_TYPES_HPP_

#include <climits>

// Set up a Doxygen group.
/** @addtogroup SDKClient
 *  @{
 */

// Virtual key code characters for WasKeyPressed().
// Based on key codes used with GetAsyncKeyState().
// Manually checked what values getch() returns for these keys.
// https://docs.microsoft.com/en-us/windows/win32/inputdev/virtual-key-codes
enum MicrosoftVirtualKeyCodes
{
	VK_TAB    = 9, // ASCII tab character value
	VK_RETURN = 13, // ASCII carriage return character value
	VK_ESCAPE = 27, // ASCII ESC character value
	VK_DOWN   = 258,
	VK_UP     = 259,
	VK_LEFT   = 260,
	VK_RIGHT  = 261,
	VK_HOME   = 262,
	VK_BACK   = 263,
	VK_F1     = 265,
	VK_F2     = 266,
	VK_F3     = 267,
	VK_F4     = 268,
	VK_F5     = 269,
	VK_F6     = 270,
	VK_F7     = 271,
	VK_F8     = 272,
	VK_F9     = 273,
	VK_F10    = 274,
	VK_F11    = 275,
	VK_F12    = 276,
	VK_DELETE = 330,
	VK_INSERT = 331,
	VK_NEXT   = 338,
	VK_PRIOR  = 339,
	VK_END    = 360
};

// Close the Doxygen group.
/** @} */

#endif
