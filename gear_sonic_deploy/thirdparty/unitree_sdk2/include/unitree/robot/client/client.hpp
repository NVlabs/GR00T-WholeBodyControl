#ifndef __UT_ROBOT_SDK_CLIENT_HPP__
#define __UT_ROBOT_SDK_CLIENT_HPP__

#include <unitree/robot/client/client_base.hpp>
#include <unitree/robot/client/lease_client.hpp>

#define UT_ROBOT_CLIENT_REG_API_NO_PROI(apiId) \
    UT_ROBOT_CLIENT_REG_API(apiId, 0)

#define UT_ROBOT_CLIENT_REG_API(apiId, priority) \
    RegistApi(apiId, priority)

namespace unitree
{
namespace robot
{
/*
 * @brief
 * @class: Client
 */
class Client: public ClientBase
{
public:
    explicit Client(const std::string& name, bool enableLease = false);
    virtual ~Client();

    void WaitLeaseApplied();
    int64_t GetLeaseId();

    const std::string& GetApiVersion() const;
    std::string GetServerApiVersion();

protected:
    void SetApiVersion(const std::string& apiVersion);

    int32_t Noop();

    int32_t Call(int32_t apiId, const std::string& parameter, std::string& data);
    int32_t Call(int32_t apiId, const std::string& parameter);

    int32_t Call(int32_t apiId, const std::vector<uint8_t>& parameter, std::vector<uint8_t>& data);
    int32_t Call(int32_t apiId, const std::vector<uint8_t>& parameter);

    int32_t Call(int32_t apiId, const std::string& parameter, const std::vector<uint8_t>& binary);

    void RegistApi(int32_t apiId, int32_t priority = 0);
    int32_t CheckApi(int32_t apiId, int32_t& priority, int64_t& leaseId);

private:
    bool mEnableLease;
    std::string mApiVersion;
    std::unordered_map<int32_t,int32_t> mApiMap;
    LeaseClientPtr mLeaseClientPtr;
};

using ClientPtr = std::shared_ptr<Client>;

}
}

#endif//__UT_ROBOT_SDK_CLIENT_HPP__
