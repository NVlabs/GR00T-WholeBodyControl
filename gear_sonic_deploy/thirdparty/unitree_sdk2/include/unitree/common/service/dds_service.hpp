#ifndef __UT_DDS_SERVICE_HPP__
#define __UT_DDS_SERVICE_HPP__

#include <unitree/common/service/base/service_base.hpp>
#include <unitree/common/service/base/service_config.hpp>
#include <unitree/common/dds/dds_easy_model.hpp>

namespace unitree
{
namespace common
{
class DdsService : public ServiceBase
{
public:
    DdsService()
    {
        mLogger = GetLogger("/unitree/service/dds_service");
        mQuit = false;
    }

    virtual ~DdsService()
    {}

    virtual void Register()
    {}

    virtual void Init()
    {
        mModel.Init(AnyCast<JsonMap>(GetGlobalParameter("DdsParameter")));
        LOG_INFO(mLogger, "parse init dds model success.");
    }

    virtual void Start()
    {}

    virtual void Wait()
    {
        while (!mQuit) { sleep(1); }
    }

    virtual void Stop()
    {
        mQuit = true;
    }

public:
    virtual void Parse(const std::string& configFileName)
    {
        ServiceBase::Parse(configFileName);
        LOG_INFO(mLogger, "parse config success. filename:", configFileName);
    }

protected:
    template<typename MSG>
    void RegistTopicMessageHandler(const std::string& topic, const DdsMessageHandler& handler)
    {
        mModel.SetTopic<MSG>(topic, handler);
        LOG_INFO(mLogger, "regist topic reader callback. topic:", topic);
    }

    template<typename MSG>
    void RegistTopic(const std::string& topic)
    {
        mModel.SetTopic<MSG>(topic);
        LOG_INFO(mLogger, "regist topic. topic:", topic);
    }

    /*
     * Write message to topic
     */
    template<typename MSG>
    void WriteMessage(const std::string& topic, const MSG& message)
    {
        mModel.WriteMessage<MSG>(topic, message);
    }

private:
    bool mQuit;
    DdsEasyModel mModel;
    Logger* mLogger;
};

}
}
#endif//__UT_DDS_SERVICE_HPP__
