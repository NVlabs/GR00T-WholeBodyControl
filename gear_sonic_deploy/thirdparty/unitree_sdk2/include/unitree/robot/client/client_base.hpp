#ifndef __UT_ROBOT_SDK_CLIENT_BASE_HPP__
#define __UT_ROBOT_SDK_CLIENT_BASE_HPP__

#include <unitree/robot/client/client_stub.hpp>

namespace unitree
{
namespace robot
{
/*
 * @brief
 * @default client timeout. 1s
 */
const int64_t ROBOT_CLIENT_TIMEOUT = 1000000;

/*
 * @brief
 * @class: ClientBase
 */
class ClientBase
{
public:
    explicit ClientBase(const std::string& name);
    virtual ~ClientBase();

    virtual void Init() = 0;

    void SetTimeout(int64_t timeout);
    void SetTimeout(float timeout);

protected:
    int32_t Call(int32_t apiId, const std::string& parameter, std::string& data, int32_t priority, int64_t leaseId);
    int32_t Call(int32_t apiId, const std::string& parameter, int32_t priority, int64_t leaseId);

    int32_t Call(int32_t apiId, const std::vector<uint8_t>& parameter, std::vector<uint8_t>& bin_data, int32_t priority, int64_t leaseId);
    int32_t Call(int32_t apiId, const std::vector<uint8_t>& parameter, int32_t priority, int64_t leaseId);

    int32_t Call(int32_t apiId, const std::string& parameter, const std::vector<uint8_t>& binary, int32_t priority, int64_t leaseId);

    int32_t Call(int32_t apiId, const std::string& parameter, std::string& data, int32_t priority, int64_t leaseId, int64_t timeout);

    void SetHeader(RequestHeader& header, int32_t apiId, int64_t leaseId, int32_t priority, bool noReply);

private:
    int64_t mTimeout;
    ClientStubPtr mClientStubPtr;
};

using ClientBasePtr = std::shared_ptr<ClientBase>;

}
}

#endif//__UT_ROBOT_SDK_CLIENT_BASE_HPP__
