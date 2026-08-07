#ifndef __UT_LOG_DECL_HPP__
#define __UT_LOG_DECL_HPP__

#include <unitree/common/exception.hpp>
#include <unitree/common/os.hpp>
#include <unitree/common/lock/lock.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/thread/recurrent_thread.hpp>
#include <unitree/common/filesystem/file.hpp>

/*
 * log buffer size
 */
#define UT_LOG_BUFFER_SIZE          65535           //64K-1
#define UT_LOG_CRT_BUFFER_SIZE      64512           //63K
#define UT_LOG_MAX_BUFFER_SIZE      4194303         //4M-1
/*
 * log file size
 */
#define UT_LOG_FILE_SIZE            104857600       //100M
#define UT_LOG_MAX_FILE_SIZE        10737418240     //10G
#define UT_LOG_MIN_FILE_SIZE        UT_LOG_MAX_BUFFER_SIZE

/*
 * log write interval(micro second)
 */
#define UT_LOG_WRITE_INTER          100000          //100ms
#define UT_LOG_MAX_WRITE_INTER      1000000         //1s
#define UT_LOG_MIN_WRITE_INTER      2000            //2ms

/*
 * log file number
 */
#define UT_LOG_FILE_NUMBER          2
#define UT_LOG_MAX_FILE_NUMBER      100
#define UT_LOG_MIN_FILE_NUMBER      UT_LOG_FILE_NUMBER

#define UT_LOG_FILE_EXT             ".LOG"

//write log macro wrapper
#define __UT_LOG(logger, level, ...)\
    do {                            \
        if (logger != NULL)         \
        {                           \
            logger->Log(level, __VA_ARGS__);    \
        }                           \
    } while (0)

#define __UT_CRIT_LOG(logger, key, code, ...)   \
    do {                            \
        if (logger != NULL)         \
        {                           \
            logger->CritLog(UT_LOG_CRIT, key, code, __VA_ARGS__);\
        }                           \
    } while (0)


//debug
#define LOG_DEBUG(logger, ...)      \
    __UT_LOG(logger, UT_LOG_DEBUG, __VA_ARGS__)

//info
#define LOG_INFO(logger, ...)       \
    __UT_LOG(logger, UT_LOG_INFO, __VA_ARGS__)

//warning
#define LOG_WARNING(logger, ...)    \
    __UT_LOG(logger, UT_LOG_WARNING, __VA_ARGS__)

//error
#define LOG_ERROR(logger, ...)      \
    __UT_LOG(logger, UT_LOG_ERROR, __VA_ARGS__)

//fatal
#define LOG_FATAL(logger, ...)      \
    __UT_LOG(logger, UT_LOG_FATAL, __VA_ARGS__)

//critical log. the 1st args is CRITICAL KEY 
#define CRIT_LOG(logger, ...)       \
    __UT_CRIT_LOG(logger, __VA_ARGS__)

#define CRIT_LOG(logger, ...)       \
    __UT_CRIT_LOG(logger, __VA_ARGS__)

//write log format macro wrapper
/*
 * FMT_DEBUG(logger, ("key1", val1)("key2", val2)("keyn", ""));
 */
#define __UT_LOG_FMT(logger, level, keyvalues)  \
    do {                                        \
        if (logger != NULL)                     \
        {                                       \
            logger->LogFormat(level, unitree::common::LogBuilder() keyvalues);    \
        }                                       \
    } while (0)

#define __UT_CRIT_LOG_FMT(logger, key, code, keyvalues)    \
    do {                                        \
        if (logger != NULL)                     \
        {                                       \
            logger->CritLogFormat(UT_LOG_CRIT, key, code, unitree::common::LogBuilder() keyvalues);   \
        }                                       \
    } while (0)

//debug
#define FMT_DEBUG(logger, keyvalues)            \
    __UT_LOG_FMT(logger, UT_LOG_DEBUG, keyvalues)

//info
#define FMT_INFO(logger, keyvalues)             \
    __UT_LOG_FMT(logger, UT_LOG_INFO, keyvalues)

//warning
#define FMT_WARNING(logger, keyvalues)          \
    __UT_LOG_FMT(logger, UT_LOG_WARNING, keyvalues)

//error
#define FMT_ERROR(logger, keyvalues)            \
    __UT_LOG_FMT(logger, UT_LOG_ERROR, keyvalues)

//fatal
#define FMT_FATAL(logger, keyvalues)            \
    __UT_LOG_FMT(logger, UT_LOG_FATAL, keyvalues)

#define CRIT_FMT(logger, critkey, keyvalues)    \
    __UT_CRIT_LOG_FMT(logger, critkey, keyvalues)


/*
//declare and define log level
#define UT_LOG_DECL_LEVEL(name, level, desc)    \
    const int32_t name = level;                 \
    const std::string name##_DESC = desc;
#define UT_LOG_DESC_LEVEL(name) name##_DESC
*/

//declare and define log store type 
#define UT_LOG_DECL_STORE_TYPE(name, type, desc)\
    const int32_t name = type;                  \
    const std::string name##_DESC = desc;
#define UT_LOG_DESC_STORE_TYPE(name) name##_DESC

//define log level
#define UT_LOG_NONE                     0
#define UT_LOG_CRIT                     1
#define UT_LOG_FATAL                    2
#define UT_LOG_ERROR                    3
#define UT_LOG_WARNING                  4
#define UT_LOG_INFO                     5
#define UT_LOG_DEBUG                    6
#define UT_LOG_ALL                      7

#define UT_LOG_DESC_NONE                "NONE"
#define UT_LOG_DESC_CRIT                "CRIT"
#define UT_LOG_DESC_FATAL               "FATAL"
#define UT_LOG_DESC_ERROR               "ERROR"
#define UT_LOG_DESC_WARNING             "WARNING"
#define UT_LOG_DESC_INFO                "INFO"
#define UT_LOG_DESC_DEBUG               "DEBUG"
#define UT_LOG_DESC_ALL                 "ALL"

//define log store type
#define UT_LOG_STORE_FILE_ASYNC         0
#define UT_LOG_STORE_FILE               1
#define UT_LOG_STORE_STDOUT             2
#define UT_LOG_STORE_STDERR             3

#define UT_LOG_STORE_DESC_FILE_ASYNC    "FILEASYNC"
#define UT_LOG_STORE_DESC_ASYNC_FILE    "ASYNCFILE"
#define UT_LOG_STORE_DESC_ASYNC         "ASYNC"
#define UT_LOG_STORE_DESC_FILE          "FILE"
#define UT_LOG_STORE_DESC_STDOUT        "STDOUT"
#define UT_LOG_STORE_DESC_STDERR        "STDERR"

namespace unitree
{
namespace common
{
static inline int32_t GetLogLevel(const std::string& desc)
{
    if (desc == UT_LOG_DESC_NONE)           {
        return UT_LOG_NONE;     }
    else if (desc == UT_LOG_DESC_CRIT)      {
        return UT_LOG_CRIT;     }
    else if (desc == UT_LOG_DESC_FATAL)     {
        return UT_LOG_FATAL;    }
    else if (desc == UT_LOG_DESC_ERROR)     {
        return UT_LOG_ERROR;    }
    else if (desc == UT_LOG_DESC_WARNING)   {
        return UT_LOG_WARNING;  }
    else if (desc == UT_LOG_DESC_INFO)      {
        return UT_LOG_INFO;     }
    else if (desc == UT_LOG_DESC_DEBUG)     {
        return UT_LOG_DEBUG;    }
    else if (desc == UT_LOG_DESC_ALL)       {
        return UT_LOG_ALL;      }

    UT_THROW(CommonException, std::string("unknown log level desc:") + desc);
}

static inline const char* GetLogLevelDesc(int32_t level)
{
    switch (level)
    {
    case UT_LOG_NONE:
        return UT_LOG_DESC_NONE;
    case UT_LOG_CRIT:
        return UT_LOG_DESC_CRIT;
    case UT_LOG_FATAL:
        return UT_LOG_DESC_FATAL;
    case UT_LOG_ERROR:
        return UT_LOG_DESC_ERROR;
    case UT_LOG_WARNING:
        return UT_LOG_DESC_WARNING;
    case UT_LOG_INFO:
        return UT_LOG_DESC_INFO;
    case UT_LOG_DEBUG:
        return UT_LOG_DESC_DEBUG;
    case UT_LOG_ALL:
        return UT_LOG_DESC_ALL;
    }

    UT_THROW(CommonException, "unknown log level");
}

static inline int32_t GetLogStoreType(const std::string& desc)
{
    if (desc == UT_LOG_STORE_DESC_FILE_ASYNC ||
        desc == UT_LOG_STORE_DESC_ASYNC_FILE ||
        desc == UT_LOG_STORE_DESC_ASYNC)      {
        return UT_LOG_STORE_FILE_ASYNC;  }
    else if (desc == UT_LOG_STORE_DESC_FILE)  {
        return UT_LOG_STORE_FILE;        }
    else if (desc == UT_LOG_STORE_DESC_STDOUT){
        return UT_LOG_STORE_STDOUT;      }
    else if (desc == UT_LOG_STORE_DESC_STDERR){
        return UT_LOG_STORE_STDERR;      }

    UT_THROW(CommonException, std::string("unknown log store type desc:") + desc);
}

static inline const char* GetLogStoreTypeDesc(int32_t type)
{
    switch (type)
    {
    case UT_LOG_STORE_FILE_ASYNC:
        return UT_LOG_STORE_DESC_FILE_ASYNC;
    case UT_LOG_STORE_FILE:
        return UT_LOG_STORE_DESC_FILE;
    case UT_LOG_STORE_STDOUT:
        return UT_LOG_STORE_DESC_STDOUT;
    case UT_LOG_STORE_STDERR:
        return UT_LOG_STORE_DESC_STDERR;
    }

    UT_THROW(CommonException, "unknown log store type");
}
}
}

#endif//__UT_LOG_DECL_HPP__
