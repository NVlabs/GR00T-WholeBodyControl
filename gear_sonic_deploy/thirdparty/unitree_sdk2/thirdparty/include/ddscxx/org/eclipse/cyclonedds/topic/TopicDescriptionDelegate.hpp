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

#ifndef CYCLONEDDS_TOPIC_TOPICDESCRIPTIONDELEGATE_HPP_
#define CYCLONEDDS_TOPIC_TOPICDESCRIPTIONDELEGATE_HPP_

#include <dds/domain/DomainParticipant.hpp>
#include <org/eclipse/cyclonedds/core/ObjectDelegate.hpp>

namespace org
{
namespace eclipse
{
namespace cyclonedds
{
namespace topic
{

class OMG_DDS_API TopicDescriptionDelegate : public virtual org::eclipse::cyclonedds::core::DDScObjectDelegate
{
public:
    typedef ::dds::core::smart_ptr_traits< TopicDescriptionDelegate >::ref_type ref_type;
    typedef ::dds::core::smart_ptr_traits< TopicDescriptionDelegate >::weak_ref_type weak_ref_type;

public:
    TopicDescriptionDelegate(const dds::domain::DomainParticipant& dp,
                             const std::string& name,
                             const std::string& type_name);
    virtual ~TopicDescriptionDelegate();

public:

    /**
     *  @internal Get the name used to create the TopicDescription.
     */
    const std::string& name() const;

    /**
     *  @internal The type_name used to create the TopicDescription.
     */
    const std::string& type_name() const;

    /**
     *  @internal This operation returns the DomainParticipant to which the
     * TopicDescription belongs.
     */
    const dds::domain::DomainParticipant& domain_participant() const;

    void incrNrDependents();

    void decrNrDependents();

    bool hasDependents() const;

    virtual std::string reader_expression() const = 0;

    //@todo virtual c_value *reader_parameters() const = 0;

    ddsi_sertype *get_ser_type() const;

protected:
    dds::domain::DomainParticipant myParticipant;
    std::string myTopicName;
    std::string myTypeName;
    uint32_t nrDependents;
    ddsi_sertype *ser_type_;
};

}
}
}
}

#endif /* CYCLONEDDS_TOPIC_TOPICDESCRIPTIONDELEGATE_HPP_ */
