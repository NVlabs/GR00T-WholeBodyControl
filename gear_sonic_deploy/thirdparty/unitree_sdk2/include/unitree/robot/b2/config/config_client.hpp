#ifndef __UT_ROBOT_B2_CONFIG_CLIENT_HPP__
#define __UT_ROBOT_B2_CONFIG_CLIENT_HPP__

#include <unitree/robot/client/client.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/ConfigChangeStatus_.hpp>

namespace unitree
{
namespace robot
{
namespace b2
{
/*
 * @brief ConfigMeta
 */
class ConfigMeta
{
public:
    ConfigMeta() : size(-1), epoch(-1)
    {}
    ~ConfigMeta()
    {}

public:
    std::string name;
    std::string lastModified;
    int32_t size;
    int32_t epoch;
};

/*
 * @brief ConfigClient
 */

using ConfigChangeStatusCallback = std::function<void(const std::string&,const std::string&)>;

class ConfigClient : public Client
{
public:
    explicit ConfigClient();
    ~ConfigClient();

    void Init();

    int32_t Set(const std::string& name, const std::string& content);
    int32_t Get(const std::string& name, std::string& content);
    int32_t Del(const std::string& name);
    int32_t Meta(const std::string& name, ConfigMeta& meta);
    int32_t Meta(const std::string& name, std::string& meta);

    void SubscribeChangeStatus(const std::string& name, const ConfigChangeStatusCallback& callback);

private:
    void ChangeStatusMessageHandler(const void* message);

private:
    unitree::common::Mutex mLock;
    std::map<std::string,ConfigChangeStatusCallback> mNameCallbackMap;
    ChannelSubscriberPtr<unitree_go::msg::dds_::ConfigChangeStatus_> mSubscriberPtr;
};

}
}
}

#endif//__UT_ROBOT_B2_CONFIG_CLIENT_HPP__

