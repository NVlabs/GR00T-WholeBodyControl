#ifndef __UT_ROBOT_SDK_CHANNEL_NAMER_HPP__
#define __UT_ROBOT_SDK_CHANNEL_NAMER_HPP__

#include <unitree/common/decl.hpp>

namespace unitree
{
namespace robot
{
const std::string ROBOT_SDK_CHANNEL_PREFIX = "rt/api/";
const std::string ROBOT_SDK_CHANNEL_SUFFIX_CLIENT = "/request";
const std::string ROBOT_SDK_CHANNEL_SUFFIX_SERVER = "/response";

/*
 * @brief
 * @class: ChannelNamer
 */
class ChannelNamer
{
public:
    virtual std::string GetSendChannelName(const std::string& name) = 0;
    virtual std::string GetRecvChannelName(const std::string& name) = 0;
};

using ChannelNamerPtr = std::shared_ptr<ChannelNamer>;

/*
 * @brief
 * @class: ClientChannelNamer
 */
class ClientChannelNamer : public ChannelNamer
{
public:
    ClientChannelNamer();
    ~ClientChannelNamer();

protected:
    std::string GetSendChannelName(const std::string& name);
    std::string GetRecvChannelName(const std::string& name);
};

using ClientChannelNamerPtr = std::shared_ptr<ClientChannelNamer>;

/*
 * @brief
 * @class: ServerChannelNamer
 */
class ServerChannelNamer : public ChannelNamer
{
public:
    ServerChannelNamer();
    virtual ~ServerChannelNamer();

protected:
    std::string GetSendChannelName(const std::string& name);
    std::string GetRecvChannelName(const std::string& name);
};

using ServerChannelNamerPtr = std::shared_ptr<ServerChannelNamer>;

}
}

#endif//__UT_ROBOT_SDK_CHANNEL_NAMER_HPP__
