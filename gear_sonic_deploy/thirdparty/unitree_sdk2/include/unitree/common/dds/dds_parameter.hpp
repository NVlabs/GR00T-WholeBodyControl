#ifndef __UT_DDS_PARAMETER_HPP__
#define __UT_DDS_PARAMETER_HPP__

#include <unitree/common/decl.hpp>
#include <unitree/common/dds/dds_qos_parameter.hpp>

#define UT_DDS_PARAM_KEY_PARTICIPANT        "Participant"
#define UT_DDS_PARAM_KEY_DOMAINID           "DomainId"
#define UT_DDS_PARAM_KEY_CONFIG             "Config"
#define UT_DDS_PARAM_KEY_NAME               "Name"
#define UT_DDS_PARAM_KEY_TOPIC              "Topic"
#define UT_DDS_PARAM_KEY_TOPICNAME          "TopicName"
#define UT_DDS_PARAM_KEY_PUBLISHER          "Publisher"
#define UT_DDS_PARAM_KEY_SUBSCRIBER         "Subscriber"
#define UT_DDS_PARAM_KEY_WRITER             "Writer"
#define UT_DDS_PARAM_KEY_READER             "Reader"
#define UT_DDS_PARAM_KEY_QOS                "Qos"

namespace unitree
{
namespace common
{
class DdsQosParameterHolder
{
public:
    DdsQosParameterHolder();

    virtual ~DdsQosParameterHolder();

    void SetQos(const DdsQosParameter& qos);
    const DdsQosParameter& GetQos() const;

protected:
    DdsQosParameter mQos;
};

class DdsParticipantParameter : public DdsQosParameterHolder
{
public:
    DdsParticipantParameter();
    DdsParticipantParameter(uint32_t domainId, const std::string& config = "");

    ~DdsParticipantParameter();

    void SetDomainId(int32_t domainId);
    uint32_t GetDomainId() const;

    void SetConfig(const std::string& config);
    const std::string& GetConfig() const;

private:
    uint32_t mDomainId;
    std::string mConfig;
};

class DdsTopicParameter : public DdsQosParameterHolder
{
public:
    DdsTopicParameter();
    DdsTopicParameter(const std::string& name);

    ~DdsTopicParameter();

    void SetName(const std::string& name);
    const std::string& GetName() const;

private:
    std::string mName;
};

class DdsTopicParameterHolder
{
public:
    DdsTopicParameterHolder();
    DdsTopicParameterHolder(const std::string& topicName);

    virtual ~DdsTopicParameterHolder();

    void SetTopicName(const std::string& topicName);
    const std::string& GetTopicName() const;

private:
    std::string mTopicName;
};

class DdsWriterParameter : public DdsTopicParameterHolder, public DdsQosParameterHolder
{
public:
    DdsWriterParameter();
    DdsWriterParameter(const std::string& topicName);

    ~DdsWriterParameter();
};

class DdsWriterParameterHolder
{
public:
    DdsWriterParameterHolder();

    virtual ~DdsWriterParameterHolder();

    void SetWriter(const DdsWriterParameter& writer);
    const DdsWriterParameter& GetWriter() const;

private:
    DdsWriterParameter mWriter;
};

class DdsReaderParameter : public DdsTopicParameterHolder, public DdsQosParameterHolder
{
public:
    DdsReaderParameter();
    DdsReaderParameter(const std::string& topicName);

    virtual ~DdsReaderParameter();
};

class DdsReaderParameterHolder
{
public:
    DdsReaderParameterHolder();

    virtual ~DdsReaderParameterHolder();

    void SetReader(const DdsReaderParameter& reader);
    const DdsReaderParameter& GetReader() const;

private:
    DdsReaderParameter mReader;
};

class DdsPublisherParameter : public DdsQosParameterHolder
{
public:
    DdsPublisherParameter();

    ~DdsPublisherParameter();

    void AppendWriter(const DdsWriterParameter& writer);
    void SetWriter(const std::vector<DdsWriterParameter>& writer);
    const std::vector<DdsWriterParameter>& GetWriter() const;

private:
    std::vector<DdsWriterParameter> mWriter;
};

class DdsSubscriberParameter : public DdsQosParameterHolder
{
public:
    DdsSubscriberParameter();
    ~DdsSubscriberParameter();

    void AppendReader(const DdsReaderParameter& reader);
    void SetReader(const std::vector<DdsReaderParameter>& reader);
    const std::vector<DdsReaderParameter>& GetReader() const;

private:
    std::vector<DdsReaderParameter> mReader;
};

class DdsParameter
{
public:
    DdsParameter();
    DdsParameter(const JsonMap& param);

    ~DdsParameter();

    void Init(const JsonMap& param);

    uint32_t GetDomainId();
    const std::string& GetConfig() const;

    const DdsParticipantParameter& GetParticipant();

    void AppendTopic(const DdsTopicParameter& topic);
    const std::map<std::string,DdsTopicParameter>& GetTopic() const;

    void AppendPublisher(const DdsPublisherParameter& publisher);
    void SetPublisher(const std::vector<DdsPublisherParameter>& publisher);
    const std::vector<DdsPublisherParameter>& GetPublisher() const;

    void AppendSubscriber(const DdsSubscriberParameter& subscriber);
    void SetSubscriber(const std::vector<DdsSubscriberParameter>& subscriber);
    const std::vector<DdsSubscriberParameter>& GetSubscriber() const;

    const DdsQosParameter& GetParticipantQos() const;
    const DdsQosParameter& GetTopicQos() const;
    const DdsQosParameter& GetPublisherQos() const;
    const DdsQosParameter& GetSubscriberQos() const;
    const DdsQosParameter& GetWriterQos() const;
    const DdsQosParameter& GetReaderQos() const;

private:
    uint32_t mDomainId;
    std::string mConfig;

    DdsQosParameter mParticipantQos;
    DdsQosParameter mTopicQos;
    DdsQosParameter mPublisherQos;
    DdsQosParameter mSubscriberQos;
    DdsQosParameter mWriterQos;
    DdsQosParameter mReaderQos;

    DdsParticipantParameter mParticipant;
    std::map<std::string,DdsTopicParameter> mTopic;
    std::vector<DdsPublisherParameter> mPublisher;
    std::vector<DdsSubscriberParameter> mSubscriber;

};

}
}
#endif//__UT_DDS_PARAMETER_HPP__
