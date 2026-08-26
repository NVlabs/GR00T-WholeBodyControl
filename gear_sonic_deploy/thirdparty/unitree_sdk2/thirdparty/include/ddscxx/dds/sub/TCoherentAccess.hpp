#ifndef OMG_DDS_SUB_TCOHERENT_ACCESS_HPP_
#define OMG_DDS_SUB_TCOHERENT_ACCESS_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Corp.
 * Copyright 2010, Real-Time Innovations, Inc.
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <dds/core/Value.hpp>


namespace dds
{
namespace sub
{
template <typename DELEGATE>
class TCoherentAccess;
}
}

/**
 * @brief
 * Class for RAII way of beginning/ending coherent access.
 *
 * Coherent access indicates that the application is about to access
 * the data samples in any of the DataReader objects attached to the
 * Subscriber.
 *
 * The application is required to use this operation
 * only if Presentation QosPolicy of the Subscriber to which the
 * DataReader belongs has the access_scope set to "GROUP". In the
 * aforementioned case, the operation must be called
 * prior to calling any of the sample-accessing operations, i.e.
 * read and take on DataReader. Otherwise the sample-accessing
 * operations will throw a PreconditionNotMetError exception.
 *
 * Once the application has finished accessing the data samples
 * it must end the coherent access. It is not required for the
 * application to begin or end access if the Presentation QosPolicy
 * has the access_scope set to something other than GROUP. Beginning
 * or ending access in this case is not considered an error and has
 * no effect. Beginning and ending access may be nested. In that
 * case, the application end access as many times as it began
 * access.
 *
 * @code{.cpp}
 * dds::domain::DomainParticipant participant(org::eclipse::cyclonedds::domain::default_id());
 * dds::topic::Topic<Foo::Bar> topic(participant, "TopicName");
 *
 * dds::sub::qos::SubscriberQos sQos sQos = participant.default_subscriber_qos()
 *                                          << dds::core::policy::Presentation::TopicAccessScope(false, true);
 * dds::sub::Subscriber subscriber(participant, sQos);
 *
 * {
 *     std::vector< dds::sub::DataReader<Foo::Bar> > readers;
 *     // Start coherent access.
 *     dds::sub::TCoherentAccess coherentAccess(subscriber);
 *     // Find (previously created with the subscriber) datareaders that now got data.
 *     dds::sub::find< dds::sub::DataReader<Foo::Bar> >(subscriber,
 *                                                      dds::sub::status::DataState::any(),
 *                                                      back_inserter(readers));
 *     // Get data from the readers
 *     for (size_type i = 0; i < rv.size(); i++) {
 *         dds::sub::LoanedSamples<Foo::Bar> samples = readers[i].read()
 *         dds::sub::LoanedSamples<Type1>::const_iterator it;
 *         for (it = samples.begin(); it != samples.end(); iterator++) {
 *             const dds::sub::Sample<Foo::Bar>& sample = *it;
 *             const Foo::Bar& data = sample.data();
 *             const dds::sub::SampleInfo& info = sample.info();
 *             // Use sample data and meta information.
 *         }
 *     }
 * }
 * // CoherentAccess went out of scope: it is ended implicitly
 * @endcode
 *
 * @see for more information: @ref DCPS_Modules_Subscription "Subscription"
 * @see dds::sub::Subscriber
 */
template <typename DELEGATE>
class dds::sub::TCoherentAccess : public dds::core::Value<DELEGATE>
{
public:
    /**
     * Creating a CoherentAccess object, which will begin ‘coherent access’ of
     * received samples using DataReader objects attached to this Subscriber.
     *
     * @param sub The Subscriber to begin the coherent access on.
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The Subscriber has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The Subscriber has not yet been enabled.
     * @throws dds::core::OutOfResourcesError
     *                  The Data Distribution Service ran out of resources to
     *                  complete this operation.
     * @throws dds::core::NullReferenceError
     *                  The Subscriber was not properly created and references to dds::core::null.
     */
    explicit TCoherentAccess(const dds::sub::Subscriber& sub);

public:
    /**
     * This operation will explicitly end the coherent access.
     *
     * If the Subscriber already ended its coherent access (by a call to this very
     * operation), then a call to this operation will have no effect.
     *
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The Subscriber has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The Subscriber has not yet been enabled.
     * @throws dds::core::NullReferenceError
     *                  The Subscriber was not properly created and references to dds::core::null.
     */
    void end();

public:
    /**
     * The destruction of the CoherentAccess will implicitly end the coherent
     * access if not already ended by a call to end().
     *
     * When there is a problem with which end() would normally throw an exception,
     * then that exception is swallowed. Errors can be found in the logs.
     */
    ~TCoherentAccess();

private:
    TCoherentAccess(const TCoherentAccess&);
    TCoherentAccess& operator=(const TCoherentAccess&);
};


#endif /* OMG_TDDS_SUB_TCOHERENT_ACCESS_HPP_ */
