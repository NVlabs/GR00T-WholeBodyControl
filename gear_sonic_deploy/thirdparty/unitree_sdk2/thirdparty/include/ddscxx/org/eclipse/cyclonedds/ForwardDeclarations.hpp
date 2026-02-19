/*
 * Copyright(c) 2006 to 2020 ZettaScale Technology and others
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

#ifndef LITE_FORWARD_DECLARATIONS_HPP_
#define LITE_FORWARD_DECLARATIONS_HPP_

namespace dds
{
    namespace domain
    {
        class DomainParticipantListener;

        template <typename DELEGATE>
        class TDomainParticipant;
    }

    namespace sub
    {
        template <typename T, template <typename Q> class DELEGATE>
        class DataReader;

        template <typename T>
        class DataReaderListener;

        class SubscriberListener;

        template <typename DELEGATE>
        class TSubscriber;

        namespace detail
        {
            template <typename T>
                class DataReader;

            template <typename T>
                class LoanedSamplesHolder;

            template <typename T, typename SamplesFWIterator>
                class SamplesFWInteratorHolder;

            template <typename T, typename SamplesBIIterator>
                class SamplesBIIteratorHolder;
        }
    }

    namespace pub
    {
        class PublisherListener;

        template <typename DELEGATE>
        class TPublisher;
    }

    namespace topic
    {
        template <typename DELEGATE>
        class TTopicDescription;

        template <typename T>
        class TopicListener;

        template <typename T, template <typename Q> class DELEGATE>
        class Topic;

        template <typename T, template <typename Q> class DELEGATE>
        class ContentFilteredTopic;

        namespace detail
        {
            template <typename T>
            class ContentFilteredTopic;
        }
    }
}


namespace org
{
namespace eclipse
{
namespace cyclonedds
{

    namespace domain {
        class DomainParticipantDelegate;
    }

    namespace sub {
        class SubscriberDelegate;
    }

    namespace pub {
        class PublisherDelegate;
    }

    namespace topic {
        template <class TOPIC>
        class TopicTraits;

        class TopicDescriptionDelegate;
    }
}
}
}

#endif /* LITE_FORWARD_DECLARATIONS_HPP_ */
