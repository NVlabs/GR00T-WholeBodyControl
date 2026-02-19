#ifndef OMG_TDDS_PUB_COHERENT_SET_HPP_
#define OMG_TDDS_PUB_COHERENT_SET_HPP_

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

#include <dds/pub/Publisher.hpp>

namespace dds
{
namespace pub
{
template <typename DELEGATE>
class TCoherentSet;
}
}


/**
 * @brief
 * Class for RAII way of beginning/ending coherent publication sets.
 *
 * A coherent set is a set of modifications that must be
 * propagated in such a way that they are interpreted at the
 * receivers' side as a consistent set of modifications; that
 * is, the receiver will only be able to access the data after
 * all the modifications in the set are available at the
 * receiver end.
 *
 * A connectivity change may occur in the middle
 * of a set of coherent changes; for example, the set of
 * partitions used by the Publisher or one of its Subscribers
 * may change, a late-joining DataReader may appear on the
 * network, or a communication failure may occur. In the event
 * that such a change prevents an entity from receiving the
 * entire set of coherent changes, that entity must behave as if
 * it had received none of the set.
 *
 * The support for
 * coherent changes enables a publishing application to change
 * the value of several data-instances that could belong to the
 * same or different topics and have those changes be seen
 * atomically by the readers. This is useful in cases where
 * the values are inter-related. For example, if there are two
 * data instances representing the altitude and velocity
 * vector of the same aircraft and both are changed, it may be
 * useful to communicate those values in such a way the reader
 * can see both together; otherwise, it may, for example,
 * erroneously interpret that the aircraft is on a collision course.
 *
 * @see for more information: @ref DCPS_Modules_Publication "Publication"
 * @see dds::pub::Publisher
 */
template <typename DELEGATE>
class dds::pub::TCoherentSet : public dds::core::Value<DELEGATE>
{
public:
   /**
    * Creating a CoherentSet object, which will begin a ‘coherent set’ of
    * modifications using DataWriter objects attached to this Publisher.
    *
    * A precondition for making coherent changes is that the PresentationQos of the
    * Publisher has its coherent_access attribute set to TRUE. If this is not the case,
    * the Publisher will not accept any coherent start requests and throw
    * dds::core::PreconditionNotMetError.
    *
    * @param pub The publisher to supsend publications on.
    * @throws dds::core::Error
    *                  An internal error has occurred.
    * @throws dds::core::AlreadyClosedError
    *                  The Publisher has already been closed.
    * @throws dds::core::NotEnabledError
    *                  The Publisher has not yet been enabled.
    * @throws dds::core::OutOfResourcesError
    *                  The Data Distribution Service ran out of resources to
    *                  complete this operation.
    * @throws dds::core::NullReferenceError
    *                  The Publisher was not properly created and references to dds::core::null.
    * @throws dds::core::PreconditionNotMetError
    *                  The coherent access attribute of the Publisher's PresentationQos is not set to true.
    */
    explicit TCoherentSet(const dds::pub::Publisher& pub);

public:
    /**
     * This operation will explicitly end the publication of an coherent set.
     *
     * If the Publisher already ended its coherent set (by a call to this very
     * operation), then a call to this operation will have no effect.
     *
     * @throws dds::core::Error
     *                  An internal error has occurred.
     * @throws dds::core::AlreadyClosedError
     *                  The Publisher has already been closed.
     * @throws dds::core::NotEnabledError
     *                  The Publisher has not yet been enabled.
     * @throws dds::core::NullReferenceError
     *                  The Publisher was not properly created and references to dds::core::null.
     */
    void end();

public:
    /**
     * The destruction of the CoherentSet will implicitly end the publication
     * of a coheren set if not already ended by a call to end().
     *
     * When there is a problem with which end() would normally throw an exception,
     * then that exception is swallowed. Errors can be found in the logs.
     */
    ~TCoherentSet();
};


#endif /* OMG_TDDS_PUB_COHERENT_SET_HPP_ */
