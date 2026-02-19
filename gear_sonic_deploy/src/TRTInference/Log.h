#pragma once

#include <string.h>
#include <iostream>

#define __FILENAME__ (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define LOG_MSG(stream, category, msg) (stream << category << " [" << __FILENAME__ << "(" << __LINE__ << ")]: " << msg << std::endl)

#define LOG_ERROR(msg)   LOG_MSG(std::cerr, "ERROR", msg)
#define LOG_WARNING(msg) LOG_MSG(std::cout, "WARNING", msg)
#define LOG_DEBUG(msg)   LOG_MSG(std::cout, "DEBUG", msg)
#define LOG_INFO(msg)    LOG_MSG(std::cout, "INFO", msg)

