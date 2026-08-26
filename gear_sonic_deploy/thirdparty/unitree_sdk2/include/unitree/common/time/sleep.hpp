#ifndef __UT_SLEEP_HPP__
#define __UT_SLEEP_HPP__

#include <unitree/common/time/time_tool.hpp>

namespace unitree
{
namespace common
{
void MicroSleep(uint64_t microsecond);
void MilliSleep(uint64_t millisecond);
void Sleep(uint64_t second);

}
}

#endif//__UT_SLEEP_HPP__
