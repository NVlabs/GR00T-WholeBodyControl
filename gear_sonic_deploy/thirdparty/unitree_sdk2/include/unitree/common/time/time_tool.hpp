#ifndef __UT_TIME_TOOL_HPP__
#define __UT_TIME_TOOL_HPP__

#include <unitree/common/decl.hpp>

namespace unitree
{
namespace common
{
#define UT_NUMER_NANO   1000000000
#define UT_NUMER_MICRO  1000000
#define UT_NUMER_MILLI  1000

//default time format: "YEAR-MON-DAY HOUR:MIN:SEC"
#define UT_TIME_FORMAT_STR  "%d-%02d-%02d %02d:%02d:%02d"

//default precise time format: "YEAR-MON-DAY HOUR:MIN:SEC.[MILLI/MICRO]"
#define UT_TIME_MICROSEC_FORMAT_STR "%d-%02d-%02d %02d:%02d:%02d.%06d"
#define UT_TIME_MILLISEC_FORMAT_STR "%d-%02d-%02d %02d:%02d:%02d.%03d"

void GetCurrentTimeval(struct timeval& tv);
void GetCurrentTimespec(struct timespec& ts);

uint64_t GetCurrentTime();
uint64_t GetCurrentTimeNanosecond();
uint64_t GetCurrentTimeMicrosecond();
uint64_t GetCurrentTimeMillisecond();

uint64_t GetCurrentCpuTimeNanosecond();
uint64_t GetCurrentThreadCpuTimeNanosecond();
uint64_t GetCurrentMonotonicTimeNanosecond();

uint64_t GetCurrentCpuTimeMicrosecond();
uint64_t GetCurrentThreadCpuTimeMicrosecond();
uint64_t GetCurrentMonotonicTimeMicrosecond();

uint64_t TimevalToMicrosecond(const struct timeval& tv);
uint64_t TimevalToMillisecond(const struct timeval& tv);

uint64_t TimespecToMicrosecond(const struct timespec& ts);
uint64_t TimespecToMillisecond(const struct timespec& ts);

void MicrosecondToTimeval(uint64_t microsec, struct timeval& tv);
void MillisecondToTimeval(uint64_t millisec, struct timeval& tv);

void MicrosecondToTimespec(uint64_t microsec, struct timespec& ts);
void MillisecondToTimespec(uint64_t millisec, struct timespec& ts);

std::string TimeFormatString(struct tm* tmptr, const char* format = UT_TIME_FORMAT_STR);
std::string TimeFormatString(struct tm* tmptr, uint64_t precise, const char* format);
std::string TimeFormatString(uint64_t sec, const char* format = UT_TIME_FORMAT_STR);

std::string TimeMicrosecondFormatString(const uint64_t& microsec,
    const char* format = UT_TIME_MICROSEC_FORMAT_STR);

std::string TimeMillisecondFormatString(const uint64_t& millisec,
    const char* format = UT_TIME_MILLISEC_FORMAT_STR);

std::string GetTimeString();
std::string GetTimeMicrosecondString();
std::string GetTimeMillisecondString();

class Timer
{
public:
    Timer();
    ~Timer();

    void Start();
    void Restart();

    uint64_t Stop();

private:
    uint64_t mMicrosecond;
};

}
}

#endif//__UT_TIME_TOOL_HPP__
