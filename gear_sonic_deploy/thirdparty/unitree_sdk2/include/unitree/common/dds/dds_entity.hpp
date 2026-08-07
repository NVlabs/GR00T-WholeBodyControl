#ifndef __UT_DDS_ENTITY_HPP__
#define __UT_DDS_ENTITY_HPP__

#include <dds/dds.hpp>
#include <unitree/common/log/log.hpp>
#include <unitree/common/block_queue.hpp>
#include <unitree/common/thread/thread.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/time/sleep.hpp>
#include <unitree/common/dds/dds_exception.hpp>
#include <unitree/common/dds/dds_callback.hpp>
#include <unitree/common/dds/dds_qos.hpp>
#include <unitree/common/dds/dds_traits.hpp>

#define __UT_DDS_NULL__ ::dds::core::null

/*
 * dds wait sub/pub matched default time slice.
 * default 10000 us
 */
#define __UT_DDS_WAIT_MATCHED_TIME_SLICE 10000
#define __UT_DDS_WAIT_MATCHED_TIME_MAX   1000000

using namespace org::eclipse::cyclonedds;

namespace unitree
{
namespace common
{
class DdsLogger
{
public:
    DdsLogger();
    virtual ~DdsLogger();

protected:
    Logger* mLogger;
};

/*
 * @brief: DdsParticipant
 */
class DdsParticipant : public DdsLogger
{
public:
    using NATIVE_TYPE = ::dds::domain::DomainParticipant;

    explicit DdsParticipant(uint32_t domainId, const DdsParticipantQos& qos, const std::string& config = "");
    ~DdsParticipant();

    const NATIVE_TYPE& GetNative() const;

private:
    NATIVE_TYPE mNative;
};

using DdsParticipantPtr = std::shared_ptr<DdsParticipant>;


/*
 * @brief: DdsPublisher
 */
class DdsPublisher : public DdsLogger
{
public:
    using NATIVE_TYPE = ::dds::pub::Publisher;

    explicit DdsPublisher(const DdsParticipantPtr& participant, const DdsPublisherQos& qos);
    ~DdsPublisher();

    const NATIVE_TYPE& GetNative() const;

private:
    NATIVE_TYPE mNative;
};

using DdsPublisherPtr = std::shared_ptr<DdsPublisher>;


/*
 * @brief: DdsSubscriber
 */
class DdsSubscriber : public DdsLogger
{
public:
    using NATIVE_TYPE = ::dds::sub::Subscriber;

    explicit DdsSubscriber(const DdsParticipantPtr& participant, const DdsSubscriberQos& qos);
    ~DdsSubscriber();

    const NATIVE_TYPE& GetNative() const;

private:
    NATIVE_TYPE mNative;
};

using DdsSubscriberPtr = std::shared_ptr<DdsSubscriber>;


/*
 * @brief: DdsTopic
 */
template<typename MSG>
class DdsTopic : public DdsLogger
{
public:
    using NATIVE_TYPE = ::dds::topic::Topic<MSG>;

    explicit DdsTopic(const DdsParticipantPtr& participant, const std::string& name, const DdsTopicQos& qos) :
        mNative(__UT_DDS_NULL__)
    {
        UT_DDS_EXCEPTION_TRY

        auto topicQos = participant->GetNative().default_topic_qos();
        qos.CopyToNativeQos(topicQos);

        mNative = NATIVE_TYPE(participant->GetNative(), name, topicQos);

        UT_DDS_EXCEPTION_CATCH(mLogger, true)
    }

    ~DdsTopic()
    {
        mNative = __UT_DDS_NULL__;
    }

    const NATIVE_TYPE& GetNative() const
    {
        return mNative;
    }

private:
    NATIVE_TYPE mNative;
};

template<typename MSG>
using DdsTopicPtr = std::shared_ptr<DdsTopic<MSG>>;


/*
 * @brief: DdsWriter
 */
template<typename MSG>
class DdsWriter : public DdsLogger
{
public:
    using NATIVE_TYPE = ::dds::pub::DataWriter<MSG>;

    explicit DdsWriter(const DdsPublisherPtr publisher, const DdsTopicPtr<MSG>& topic, const DdsWriterQos& qos) :
        mNative(__UT_DDS_NULL__)
    {
        UT_DDS_EXCEPTION_TRY

        auto writerQos = publisher->GetNative().default_datawriter_qos();
        qos.CopyToNativeQos(writerQos);

        mNative = NATIVE_TYPE(publisher->GetNative(), topic->GetNative(), writerQos);

        UT_DDS_EXCEPTION_CATCH(mLogger, true)
    }

    ~DdsWriter()
    {
        mNative = __UT_DDS_NULL__;
    }

    const NATIVE_TYPE& GetNative() const
    {
        return mNative;
    }

    bool Write(const MSG& message, int64_t waitMicrosec)
    {
        if (waitMicrosec > 0)
        {
            WaitReader(waitMicrosec);
        }

        UT_DDS_EXCEPTION_TRY
        {
            mNative.write(message);
            return true;
        }
        UT_DDS_EXCEPTION_CATCH(mLogger, false)

        return false;
    }

private:
    void WaitReader(int64_t waitMicrosec)
    {
        if (waitMicrosec < __UT_DDS_WAIT_MATCHED_TIME_SLICE)
        {
            return;
        }

        int64_t waitTime = (waitMicrosec / 2);
        if (waitTime > __UT_DDS_WAIT_MATCHED_TIME_MAX)
        {
            waitTime = __UT_DDS_WAIT_MATCHED_TIME_MAX;
        }

        while (waitTime > 0 && mNative.publication_matched_status().current_count() == 0)
        {
            MicroSleep(__UT_DDS_WAIT_MATCHED_TIME_SLICE);
            waitTime -=__UT_DDS_WAIT_MATCHED_TIME_SLICE;
        }
    }

private:
    NATIVE_TYPE mNative;
};

template<typename MSG>
using DdsWriterPtr = std::shared_ptr<DdsWriter<MSG>>;


/*
 * @brief: DdsReaderListener
 */
template<typename MSG>
class DdsReaderListener : public ::dds::sub::NoOpDataReaderListener<MSG>, DdsLogger
{
public:
    using NATIVE_TYPE = ::dds::sub::DataReaderListener<MSG>;
    using MSG_PTR = std::shared_ptr<MSG>;

    explicit DdsReaderListener() :
        mHasQueue(false), mQuit(false), mMask(::dds::core::status::StatusMask::none()), mLastDataAvailableTime(0)
    {}

    ~DdsReaderListener()
    {
        if (mHasQueue)
        {
            mQuit = true;
            mDataQueuePtr->Interrupt(false);
            mDataQueueThreadPtr->Wait();
        }
    }

    void SetCallback(const DdsReaderCallback& cb)
    {
        if (cb.HasMessageHandler())
        {
            mMask |= ::dds::core::status::StatusMask::data_available();
        }

        mCallbackPtr.reset(new DdsReaderCallback(cb));
    }

    void SetQueue(int32_t len)
    {
        if (len <= 0)
        {
            return;
        }

        mHasQueue = true;
        mDataQueuePtr.reset(new BlockQueue<MSG_PTR>(len));

        auto queueThreadFunc = [this]() {
            while (true)
            {
                if (mCallbackPtr && mCallbackPtr->HasMessageHandler())
                {
                    break;
                }
                else
                {
                    MicroSleep(__UT_DDS_WAIT_MATCHED_TIME_SLICE);
                }
            }
            while (!mQuit)
            {
                MSG_PTR dataPtr;
                if (mDataQueuePtr->Get(dataPtr))
                {
                    if (dataPtr)
                    {
                        mCallbackPtr->OnDataAvailable(dataPtr.get());
                    }
                }
            }
            return 0;
        };

        mDataQueueThreadPtr = CreateThreadEx("rlsnr", UT_CPU_ID_NONE, queueThreadFunc);
    }

    int64_t GetLastDataAvailableTime() const
    {
        return mLastDataAvailableTime;
    }

    NATIVE_TYPE* GetNative() const
    {
        return (NATIVE_TYPE*)this;
    }

    const ::dds::core::status::StatusMask& GetStatusMask() const
    {
        return mMask;
    }

private:
    void on_data_available(::dds::sub::DataReader<MSG>& reader)
    {
        ::dds::sub::LoanedSamples<MSG> samples;
        samples = reader.take();

        if (samples.length() <= 0)
        {
            return;
        }

        typename ::dds::sub::LoanedSamples<MSG>::const_iterator iter;
        for (iter=samples.begin(); iter<samples.end(); ++iter)
        {
            const MSG& m = iter->data();
            if (iter->info().valid())
            {
                mLastDataAvailableTime = GetCurrentMonotonicTimeNanosecond();

                if (mHasQueue)
                {
                    if (!mDataQueuePtr->Put(MSG_PTR(new MSG(m)), true))
                    {
                        LOG_WARNING(mLogger, "earliest mesage was evicted. type:", DdsGetTypeName(MSG));
                    }
                }
                else
                {
                    mCallbackPtr->OnDataAvailable((const void*)&m);
                }
            }
        }
    }

private:
    bool mHasQueue;
    volatile bool mQuit;

    ::dds::core::status::StatusMask mMask;
    int64_t mLastDataAvailableTime;

    DdsReaderCallbackPtr mCallbackPtr;
    BlockQueuePtr<MSG_PTR> mDataQueuePtr;
    ThreadPtr mDataQueueThreadPtr;
};

template<typename MSG>
using DdsReaderListenerPtr = std::shared_ptr<DdsReaderListener<MSG>>;


/*
 * @brief: DdsReader
 */
template<typename MSG>
class DdsReader : public DdsLogger
{
public:
    using NATIVE_TYPE = ::dds::sub::DataReader<MSG>;

    explicit DdsReader(const DdsSubscriberPtr& subscriber, const DdsTopicPtr<MSG>& topic, const DdsReaderQos& qos) :
        mNative(__UT_DDS_NULL__)
    {
        UT_DDS_EXCEPTION_TRY

        auto readerQos = subscriber->GetNative().default_datareader_qos();
        qos.CopyToNativeQos(readerQos);

        mNative = NATIVE_TYPE(subscriber->GetNative(), topic->GetNative(), readerQos);

        UT_DDS_EXCEPTION_CATCH(mLogger, true)
    }

    ~DdsReader()
    {
        mNative = __UT_DDS_NULL__;
    }

    const NATIVE_TYPE& GetNative() const
    {
        return mNative;
    }

    void SetListener(const DdsReaderCallback& cb, int32_t qlen)
    {
        mListener.SetCallback(cb);
        mListener.SetQueue(qlen);
        mNative.listener(mListener.GetNative(), mListener.GetStatusMask());
    }

    int64_t GetLastDataAvailableTime() const
    {
        return mListener.GetLastDataAvailableTime();
    }

private:
    NATIVE_TYPE mNative;
    DdsReaderListener<MSG> mListener;
};

template<typename MSG>
using DdsReaderPtr = std::shared_ptr<DdsReader<MSG>>;

}
}

#endif//__UT_DDS_ENTITY_HPP__
