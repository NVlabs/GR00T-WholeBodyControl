#ifndef __UT_DECL_HPP__
#define __UT_DECL_HPP__

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <exception>
#include <execinfo.h>
#include <sched.h>
#include <signal.h>
#include <string.h>
#include <strings.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/sysinfo.h>
#include <sys/syscall.h>
#include <sys/resource.h>
#include <sys/timerfd.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netdb.h>
#include <time.h>
#include <poll.h>
#include <pthread.h>
#include <pwd.h>
#include <limits.h>
#include <fcntl.h>
#include <dirent.h>
#include <utime.h>
#include <atomic>
#include <string>
#include <iostream>
#include <sstream>
#include <typeinfo>
#include <list>
#include <vector>
#include <set>
#include <map>
#include <unordered_map>
#include <functional>
#include <iomanip>
#include <memory>
#include <regex>

#ifdef __GLIBC__
#define UT_UNLIKELY(x)  __builtin_expect(!!(x), 0)
#define UT_LIKELY(x)    __builtin_expect(!!(x), 1)
#else
#define UT_UNLIKELY(x)  (x)
#define UT_LIKELY(x)    (x)
#endif//__GLIBC__

#define __UT_CAT(x, y)  x##y
#define UT_CAT(x, y)    __UT_CAT(x, y)

#define __UT_STR(x)     #x
#define UT_STR(x)       __UT_STR(x)

#define UT_QUEUE_MAX_LEN        INT_MAX
#define UT_PATH_MAX_LEN         1024
#define UT_THREAD_NAME_MAX_LEN  16

#define UT_DECL_ERR(name, code, desc)   \
    const int32_t name = code; const std::string name##_DESC = desc;

#define UT_DESC_ERR(name) name##_DESC

#ifndef SYS_gettid
#define SYS_gettid __NR_gettid
#endif//SYS_gettid

#define UT_SAFE_DEL(x)  \
    if ((x) != NULL) { delete (x); (x) = NULL; }

namespace unitree
{
namespace common
{
static const std::string UT_EMPTY_STR = "";
}
}

#endif//__UT_DECL_HPP__
