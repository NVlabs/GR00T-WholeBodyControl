#ifndef __UT_ROBOT_INTERNAL_REQUEST_RESPONSE_HPP__
#define __UT_ROBOT_INTERNAL_REQUEST_RESPONSE_HPP__

#include <unitree/robot/internal/internal_idl_decl/Request_.hpp>
#include <unitree/robot/internal/internal_idl_decl/Response_.hpp>

namespace unitree
{
namespace robot
{
using RequestIdentity = unitree_api::msg::dds_::RequestIdentity_;
using RequestLease = unitree_api::msg::dds_::RequestLease_;
using RequestPolicy = unitree_api::msg::dds_::RequestPolicy_;
using RequestHeader = unitree_api::msg::dds_::RequestHeader_;
using Request = unitree_api::msg::dds_::Request_;

using ResponseStatus = unitree_api::msg::dds_::ResponseStatus_;
using ResponseHeader = unitree_api::msg::dds_::ResponseHeader_;
using Response = unitree_api::msg::dds_::Response_;
}
}

#endif//__UT_ROBOT_INTERNAL_REQUEST_RESPONSE_HPP__
