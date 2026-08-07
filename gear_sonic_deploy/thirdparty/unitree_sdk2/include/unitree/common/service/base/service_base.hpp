#ifndef __UT_SERVICE_BASE_HPP__
#define __UT_SERVICE_BASE_HPP__

#include <unitree/common/service/base/service_config.hpp>

namespace unitree
{
namespace common
{
class ServiceBase
{
public:
    ServiceBase();
    virtual ~ServiceBase();

    virtual void Parse(const std::string& configFileName);

    virtual void Init();

    virtual void Register();

    virtual void Start();

    virtual void Wait();

    virtual void Stop();

protected:
    /*
     * Parse config content
     */
    virtual void ParseConfigContent(const std::string& content);

    /*
     * Get field: ServiceName
     */
    const std::string& GetServiceName() const;

    /*
     * Get field: CpuIds
     */
    const std::string& GetCpuIds() const;

    /*
     * Get top-level field/value
     */
    const Any& GetGlobalParameter(const std::string& name) const;

    /*
     * Get top-level field: Parameter
     */
    bool HasParameter(const std::string& name) const;

    /*
     * Get top-level field: Parameter
     */
    const JsonMap& GetParameter() const;

    /*
     * Get field from top-level field: Parameter
     */
    const Any& GetParameter(const std::string& name) const;

    /*
     * Get field from top-level field: Parameter
     */
    template<typename T>
    const T& GetParameter(const std::string& name) const
    {
        return mConfig.GetParameter<T>(name);
    }

    /*
     * Get field from top-level field: Parameter
     */
    template<typename T>
    T GetNumberParameter(const std::string& name) const
    {
        return mConfig.GetNumberParameter<T>(name);
    }

    /*
     * Get field from top-level field: Parameter
     */
    template<typename T>
    T GetParameter(const std::string& name, const T& defValue) const
    {
        return mConfig.GetParameter<T>(name, defValue);
    }

    template<typename T>
    T GetNumberParameter(const std::string& name, const T& defValue) const
    {
        return mConfig.GetNumberParameter<T>(name, defValue);
    }

private:
    ServiceConfig mConfig;
};
typedef std::shared_ptr<ServiceBase> ServicePtr;

}
}
#endif//__UT_SERVICE_BASE_HPP__
