/*
 * Copyright(c) 2006 to 2021 ZettaScale Technology and others
 *
 * This program and the accompanying materials are made available under the
 * terms of the Eclipse Public License v. 2.0 which is available at
 * http://www.eclipse.org/legal/epl-2.0, or the Eclipse Distribution License
 * v. 1.0 which is available at
 * http://www.eclipse.org/org/documents/edl-v10.php.
 *
 * SPDX-License-Identifier: EPL-2.0 OR BSD-3-Clause
 */


/**
 * @file
 */

#ifndef CYCLONEDDS_SUB_QOS_DATA_READER_QOS_DELEGATE_HPP_
#define CYCLONEDDS_SUB_QOS_DATA_READER_QOS_DELEGATE_HPP_

#include <dds/core/detail/conformance.hpp>
#include <org/eclipse/cyclonedds/topic/qos/TopicQosDelegate.hpp>

struct _DDS_NamedDataReaderQos;

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace sub
{
namespace qos
{

class OMG_DDS_API DataReaderQosDelegate
{
public:
    DataReaderQosDelegate();
    DataReaderQosDelegate(const org::eclipse::cyclonedds::topic::qos::TopicQosDelegate& tqos);

    void policy(const dds::core::policy::UserData&            user_data);
    void policy(const dds::core::policy::Durability&          durability);
    void policy(const dds::core::policy::Deadline&            deadline);
    void policy(const dds::core::policy::LatencyBudget&       budget);
    void policy(const dds::core::policy::Liveliness&          liveliness);
    void policy(const dds::core::policy::Reliability&         reliability);
    void policy(const dds::core::policy::DestinationOrder&    order);
    void policy(const dds::core::policy::History&             history);
    void policy(const dds::core::policy::ResourceLimits&      resources);
    void policy(const dds::core::policy::Ownership&           ownership);
    void policy(const dds::core::policy::TimeBasedFilter&     tfilter);
    void policy(const dds::core::policy::ReaderDataLifecycle& lifecycle);
#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT
    void policy(const dds::core::policy::DataRepresentation&  datarepresentation);
    void policy(const dds::core::policy::TypeConsistencyEnforcement& typeconsistencyenforcement);
#endif //  OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

    template <typename POLICY> const POLICY& policy() const;
    template <typename POLICY> POLICY& policy();

    /* The returned ddsc QoS has to be freed. */
    dds_qos_t* ddsc_qos() const;
    void ddsc_qos(const dds_qos_t* qos);

    void named_qos(const struct _DDS_NamedDataReaderQos &qos);

    void check() const;

    bool operator==(const DataReaderQosDelegate& other) const;
    DataReaderQosDelegate& operator =(const org::eclipse::cyclonedds::topic::qos::TopicQosDelegate& tqos);

private:
    dds::core::policy::UserData                user_data_;
    dds::core::policy::Durability              durability_;
    dds::core::policy::Deadline                deadline_;
    dds::core::policy::LatencyBudget           budget_;
    dds::core::policy::Liveliness              liveliness_;
    dds::core::policy::Reliability             reliability_;
    dds::core::policy::DestinationOrder        order_;
    dds::core::policy::History                 history_;
    dds::core::policy::ResourceLimits          resources_;
    dds::core::policy::Ownership               ownership_;
    dds::core::policy::TimeBasedFilter         tfilter_;
    dds::core::policy::ReaderDataLifecycle     lifecycle_;
#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT
    dds::core::policy::DataRepresentation      datarepresentation_;
    dds::core::policy::TypeConsistencyEnforcement typeconsistencyenforcement_;
#endif //  OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT
};



//==============================================================================


template<>
inline const dds::core::policy::Durability&
DataReaderQosDelegate::policy<dds::core::policy::Durability>() const
{
    return durability_;
}
template<>
inline dds::core::policy::Durability&
DataReaderQosDelegate::policy<dds::core::policy::Durability>()
{
    return durability_;
}


template<>
inline const dds::core::policy::UserData&
DataReaderQosDelegate::policy<dds::core::policy::UserData>() const
{
    return user_data_;
}
template<>
inline dds::core::policy::UserData&
DataReaderQosDelegate::policy<dds::core::policy::UserData>()
{
    return user_data_;
}


template<> inline const dds::core::policy::Deadline&
DataReaderQosDelegate::policy<dds::core::policy::Deadline>() const
{
    return deadline_;
}
template<>
inline dds::core::policy::Deadline&
DataReaderQosDelegate::policy<dds::core::policy::Deadline>()
{
    return deadline_;
}


template<> inline const dds::core::policy::LatencyBudget&
DataReaderQosDelegate::policy<dds::core::policy::LatencyBudget>() const
{
    return budget_;
}
template<>
inline dds::core::policy::LatencyBudget&
DataReaderQosDelegate::policy<dds::core::policy::LatencyBudget>()
{
    return budget_;
}


template<> inline const dds::core::policy::Liveliness&
DataReaderQosDelegate::policy<dds::core::policy::Liveliness>() const
{
    return liveliness_;
}
template<>
inline dds::core::policy::Liveliness&
DataReaderQosDelegate::policy<dds::core::policy::Liveliness>()
{
    return liveliness_;
}


template<> inline const dds::core::policy::Reliability&
DataReaderQosDelegate::policy<dds::core::policy::Reliability>() const
{
    return reliability_;
}
template<>
inline dds::core::policy::Reliability&
DataReaderQosDelegate::policy<dds::core::policy::Reliability>()
{
    return reliability_;
}


template<> inline const dds::core::policy::DestinationOrder&
DataReaderQosDelegate::policy<dds::core::policy::DestinationOrder>() const
{
    return order_;
}
template<>
inline dds::core::policy::DestinationOrder&
DataReaderQosDelegate::policy<dds::core::policy::DestinationOrder>()
{
    return order_;
}


template<> inline const dds::core::policy::History&
DataReaderQosDelegate::policy<dds::core::policy::History>() const
{
    return history_;
}
template<>
inline dds::core::policy::History&
DataReaderQosDelegate::policy<dds::core::policy::History>()
{
    return history_;
}


template<> inline const dds::core::policy::ResourceLimits&
DataReaderQosDelegate::policy<dds::core::policy::ResourceLimits>() const
{
    return resources_;
}
template<>
inline dds::core::policy::ResourceLimits&
DataReaderQosDelegate::policy<dds::core::policy::ResourceLimits>()
{
    return resources_;
}


template<> inline const dds::core::policy::Ownership&
DataReaderQosDelegate::policy<dds::core::policy::Ownership>() const
{
    return ownership_;
}
template<>
inline dds::core::policy::Ownership&
DataReaderQosDelegate::policy<dds::core::policy::Ownership>()
{
    return ownership_;
}


template<> inline const dds::core::policy::TimeBasedFilter&
DataReaderQosDelegate::policy<dds::core::policy::TimeBasedFilter>() const
{
    return tfilter_;
}
template<>
inline dds::core::policy::TimeBasedFilter&
DataReaderQosDelegate::policy<dds::core::policy::TimeBasedFilter>()
{
    return tfilter_;
}


template<> inline const dds::core::policy::ReaderDataLifecycle&
DataReaderQosDelegate::policy<dds::core::policy::ReaderDataLifecycle>() const
{
    return lifecycle_;
}
template<>
inline dds::core::policy::ReaderDataLifecycle&
DataReaderQosDelegate::policy<dds::core::policy::ReaderDataLifecycle>()
{
    return lifecycle_;
}

#ifdef OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT
template<> inline const  dds::core::policy::DataRepresentation&
DataReaderQosDelegate::policy<dds::core::policy::DataRepresentation>() const
{
    return datarepresentation_;
}
template<> inline dds::core::policy::DataRepresentation&
DataReaderQosDelegate::policy<dds::core::policy::DataRepresentation>()
{
    return datarepresentation_;
}

template<> inline const  dds::core::policy::TypeConsistencyEnforcement&
DataReaderQosDelegate::policy<dds::core::policy::TypeConsistencyEnforcement>() const
{
    return typeconsistencyenforcement_;
}
template<> inline dds::core::policy::TypeConsistencyEnforcement&
DataReaderQosDelegate::policy<dds::core::policy::TypeConsistencyEnforcement>()
{
    return typeconsistencyenforcement_;
}
#endif //  OMG_DDS_EXTENSIBLE_AND_DYNAMIC_TOPIC_TYPE_SUPPORT

}
}
}
}
}

#endif /* CYCLONEDDS_SUB_QOS_DATA_READER_QOS_DELEGATE_HPP_ */
