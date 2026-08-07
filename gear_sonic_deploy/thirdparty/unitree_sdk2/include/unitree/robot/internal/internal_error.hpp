#ifndef __UT_ROBOT_INTERNAL_ERROR_HPP__
#define __UT_ROBOT_INTERNAL_ERROR_HPP__

#include <unitree/common/decl.hpp>

namespace unitree
{
namespace robot
{
UT_DECL_ERR(UT_ROBOT_OK,                            0,      "Success.")
UT_DECL_ERR(UT_ROBOT_ERR_UNKNOWN,                   3001,   "Unknown error.")
UT_DECL_ERR(UT_ROBOT_ERR_CLIENT_SEND,               3102,   "Send request error.")
UT_DECL_ERR(UT_ROBOT_ERR_CLIENT_API_NOT_REG,        3103,   "Api is not registed.")
UT_DECL_ERR(UT_ROBOT_ERR_CLIENT_API_TIMEOUT,        3104,   "Call api timeout error.")
UT_DECL_ERR(UT_ROBOT_ERR_CLIENT_API_NOT_MATCH,      3105,   "Response api not match error.")
UT_DECL_ERR(UT_ROBOT_ERR_CLIENT_API_DATA,           3106,   "Response data error.")
UT_DECL_ERR(UT_ROBOT_ERR_CLIENT_LEASE_INVALID,      3107,   "Lease is invalid.")

UT_DECL_ERR(UT_ROBOT_ERR_SERVER_SEND,               3201,   "Send response error.")
UT_DECL_ERR(UT_ROBOT_ERR_SERVER_INTERNAL,           3202,   "Server internal error.")
UT_DECL_ERR(UT_ROBOT_ERR_SERVER_API_NOT_IMPL,       3203,   "Api not implement error.")
UT_DECL_ERR(UT_ROBOT_ERR_SERVER_API_PARAMETER,      3204,   "Api parameter error.")
UT_DECL_ERR(UT_ROBOT_ERR_SERVER_LEASE_DENIED,       3205,   "Request denied by lease.")
UT_DECL_ERR(UT_ROBOT_ERR_SERVER_LEASE_NOT_EXIST,    3206,   "Lease not exist in server cache.")
UT_DECL_ERR(UT_ROBOT_ERR_SERVER_LEASE_EXIST,        3207,   "Lease is already exist in server cache.")
}
}

#endif//__UT_ROBOT_INTERNAL_ERROR_HPP__
