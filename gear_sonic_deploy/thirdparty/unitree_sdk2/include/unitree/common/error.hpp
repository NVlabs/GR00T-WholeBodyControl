#ifndef __UT_ERROR_HPP__
#define __UT_ERROR_HPP__

#include <unitree/common/decl.hpp>

namespace unitree
{
//Declare error
UT_DECL_ERR(UT_OK,              0,      "Success")
UT_DECL_ERR(UT_ERR_COMMON,      1001,   "common error")
UT_DECL_ERR(UT_ERR_BADCAST,     1002,   "Bad cast error")
UT_DECL_ERR(UT_ERR_FUTURE,      1003,   "Future error")
UT_DECL_ERR(UT_ERR_FUTURE_FAULT,1004,   "Future fault error")
UT_DECL_ERR(UT_ERR_JSON,        1005,   "Json data error")
UT_DECL_ERR(UT_ERR_SYSTEM,      1006,   "System error")
UT_DECL_ERR(UT_ERR_FILE,        1007,   "File operation error")
UT_DECL_ERR(UT_ERR_SOCKET,      1008,   "Socket operaton error")
UT_DECL_ERR(UT_ERR_IO,          1009,   "IO operaton error")
UT_DECL_ERR(UT_ERR_LOCK,        1010,   "Lock operation error")
UT_DECL_ERR(UT_ERR_NETWORK,     1011,   "Network error")
UT_DECL_ERR(UT_ERR_TIMEOUT,     1012,   "Timeout error")
UT_DECL_ERR(UT_ERR_UNKNOWN,     -1,     "Unknown error")
}

#endif//__UT_ERROR_HPP__
