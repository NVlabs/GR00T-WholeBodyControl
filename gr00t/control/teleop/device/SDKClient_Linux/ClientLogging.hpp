#ifndef __CLIENT_LOGGING_HPP__
#define __CLIENT_LOGGING_HPP__
#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>

namespace ManusSDK
{
	class ClientLog
	{
	public:
		//Simple
		static void debug(const char* p_String)
		{
			std::cout << "DEBUG: " << p_String << std::endl;
		}

		static void info(const char* p_String)
		{
			std::cout << "INFO: " << p_String << std::endl;
		}

		static void warn(const char* p_String)
		{
			std::cout << "WARN: " << p_String << std::endl;
		}

		static void error(const char* p_String)
		{
			std::cout << "ERROR: " << p_String << std::endl;
		}

		static void print(const char* p_String)
		{
			std::cout << p_String << std::endl;
		}

		//Templated
		template<typename... Args>
		static void debug(const char* p_Format, Args &&...p_Args)
		{
			std::string formatted_string = p_Format;
			replacePlaceholders(formatted_string, std::forward<Args>(p_Args)...);
			std::cout << "DEBUG: " << formatted_string << std::endl;
		}

		template<typename... Args>
		static void info(const char* p_Format, Args &&...p_Args)
		{
			std::string formatted_string = p_Format;
			replacePlaceholders(formatted_string, std::forward<Args>(p_Args)...);
			std::cout << "INFO: " << formatted_string << std::endl;
		}

		template<typename... Args>
		static void warn(const char* p_Format, Args &&...p_Args)
		{
			std::string formatted_string = p_Format;
			replacePlaceholders(formatted_string, std::forward<Args>(p_Args)...);
			std::cout << "WARN: " << formatted_string << std::endl;
		}

		template<typename... Args>
		static void error(const char* p_Format, Args &&...p_Args)
		{
			std::string formatted_string = p_Format;
			replacePlaceholders(formatted_string, std::forward<Args>(p_Args)...);
			std::cout << "ERROR: " << formatted_string << std::endl;
		}
		
		template<typename... Args>
		static void print(const char* p_Format, Args &&...p_Args)
		{
			std::string formatted_string = p_Format;
			replacePlaceholders(formatted_string, std::forward<Args>(p_Args)...);
			std::cout << formatted_string << std::endl;
		}

		template<typename... Args>
		static void printWithPadding(const char* p_Format, int p_Padding, Args &&...p_Args)
		{
			std::string formatted_string = p_Format;
			replacePlaceholdersWithPadding(formatted_string, p_Padding, std::forward<Args>(p_Args)...);
			std::cout << formatted_string << std::endl;
		}

	private:
		// Base case for recursion: no arguments left
		static void replacePlaceholders(std::string&) {
			// No more arguments, so nothing to replace
		}

		// Replace placeholders {} with actual values (only supports empty curly brackets, no formatting)
		template<typename T, typename... Args>
		static void replacePlaceholders(std::string& p_Format, T&& p_Value, Args&&... p_Args) {
			size_t pos = p_Format.find("{}");
			if (pos != std::string::npos) {
				p_Format.replace(pos, 2, to_string(std::forward<T>(p_Value)));
				replacePlaceholders(p_Format, std::forward<Args>(p_Args)...);
			}
		}

		// Base case for recursion: no arguments left
		static void replacePlaceholdersWithPadding(std::string&, int p_Padding) {
			// No more arguments, so nothing to replace
		}
		// Replace placeholders {} with actual values (only supports empty curly brackets, no formatting)
		template<typename T, typename... Args>
		static void replacePlaceholdersWithPadding(std::string& p_Format, int p_Padding, T&& p_Value, Args&&... p_Args) {
			size_t pos = p_Format.find("{}");
			if (pos != std::string::npos) {
				std::ostringstream oss;
				oss << std::setw(p_Padding) << to_string(std::forward<T>(p_Value));
				p_Format.replace(pos, 2, oss.str());
				replacePlaceholdersWithPadding(p_Format, p_Padding, std::forward<Args>(p_Args)...);
			}
		}

		// Utility to convert various types to string
		template<typename T>
		static std::string to_string(const T& p_Value) {
			std::ostringstream oss;
			oss << p_Value;
			return oss.str();
		}

		// Specialization for uint8_t
		static std::string to_string(const uint8_t& p_Value) {
			std::ostringstream oss;
			oss << static_cast<int>(p_Value);
			return oss.str();
		}

		// Specialization for uint16_t
		static std::string to_string(const uint16_t& p_Value) {
			std::ostringstream oss;
			oss << static_cast<int>(p_Value);
			return oss.str();
		}

		// Specialization for uint32_t
		static std::string to_string(const uint32_t& p_Value) {
			std::ostringstream oss;
			oss << std::hex << p_Value;
			return oss.str();
		}

		// Specialization for bool
		static std::string to_string(const bool& p_Value) {
			std::ostringstream oss;
			oss << std::setw(5) << (p_Value ? "true" : "false");
			return oss.str();
		}
	};
}

#endif
