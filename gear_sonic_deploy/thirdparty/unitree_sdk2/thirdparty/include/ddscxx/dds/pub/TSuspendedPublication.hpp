#ifndef OMG_TDDS_PUB_SUSPENDED_PUBLICATION_HPP_
#define OMG_TDDS_PUB_SUSPENDED_PUBLICATION_HPP_

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
class TSuspendedPublication;
}
}

/**
 * @brief
 * Class for RAII way of suspending/resuming publication.
 *
 * Suspended publication indicates to the Service that the application is about
 * to make multiple modifications using DataWriter objects belonging to
 * the Publisher.
 *
 * It is a hint to the Service so it can optimize its performance by,
 * e.g., holding the dissemination of the modifications and then batching
 * them. It is not required that the Service use this hint in any way.
 *
 * When a Suspended publication is started, it must be matched by
 * a corresponding call to SuspendedPublication::resume() or the destruction
 * of the SuspendedPublication object (which is an implicit resume),
 * indicating that the set of modifications has
 * completed. If the Publisher is deleted before the resume,
 * any suspended updates yet to be published will be discarded.
 *
 * This object suspends the publication of all DataWriter objects contained by
 * the given Publisher. The data written, disposed or unregistered by a DataWriter is
 * stored in the history buffer of the DataWriter and therefore, depending on its QoS
 * settings, the following operations may block (see the operation descriptions for
 * more information):
 * - dds::pub::DataWriter.write (and its overloaded counterparts).
 * - dds::pub::DataWriter.operator<< (and its overloaded counterparts).
 * - dds::pub::DataWriter.unregister_instance (and its overloaded counterparts).
 * - dds::pub::DataWriter.dispose_instance (and its overloaded counterparts).
 *
 * @see for more information: @ref DCPS_Modules_Publication "Publication"
 * @see dds::pub::Publisher
 */
template <typename DELEGATE>
class dds::pub::TSuspendedPublication : public dds::core::Value<DELEGATE>
{
public:
    /**
     * Creating a SuspendedPublication object, which will suspend the
     * dissemination of the publications by all contained DataWriter objects
     * of the given Publisher.
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
     */
    explicit TSuspendedPublication(const dds::pub::Publisher& pub);

public:
    /**
     * This operation will explicitly resume the publication of all
     * DataWriter objects contained by the given Publisher at construction.
     *
     * All data held in the history
     * buffer of the DataWriter's is actively published to the consumers. When the
     * operation returns all DataWriter's have resumed the publication of suspended
     * updates.
     *
     * If the Publisher already resumed its publication (by a call to this very
     * operation), then a call to this operation will have no effect.
     *
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
     */
    void resume();

public:
    /**
     * The destruction of the SuspendedPublication will implicitly resume
     * the publication if not already resumed by a call to resume().
     *
     * All data held in the history
     * buffer of the DataWriter's is actively published to the consumers. When the
     * operation returns all DataWriter's have resumed the publication of suspended
     * updates.
     *
     * When there is a problem with which resume() would normally throw an exception,
     * then that exception is swallowed. Errors can be found in the logs.
     */
    ~TSuspendedPublication();
};


#endif /* OMG_TDDS_PUB_SUSPENDED_PUBLICATION_HPP_ */
