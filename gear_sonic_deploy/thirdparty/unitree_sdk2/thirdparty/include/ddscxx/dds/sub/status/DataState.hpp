#ifndef OMG_DDS_SUB_DATA_STATE_HPP_
#define OMG_DDS_SUB_DATA_STATE_HPP_

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

#include <bitset>

#include <dds/core/types.hpp>


namespace dds
{
namespace sub
{
namespace status
{
class SampleState;
class ViewState;
class InstanceState;
class DataState;
}
}
}

/**
 * @brief
 * Class to hold SampleState information.
 *
 * For each sample, the Data Distribution Service internally maintains a
 * sample_state specific to each DataReader. The sample_state can either be
 * READ_SAMPLE_STATE or NOT_READ_SAMPLE_STATE.
 * - <b><i>read</i></b>
 *      - The DataReader has already accessed that
 *        sample by means of read. Had the sample been accessed by take it would
 *        no longer be available to the DataReader.
 * - <b><i>not_read</i></b>
 *      - The DataReader has not accessed that sample before.
 *
 * @see for more information: @ref DCPS_Modules_Subscription_SampleInfo "SampleInfo"
 */
class OMG_DDS_API dds::sub::status::SampleState : public std::bitset<OMG_DDS_STATE_BIT_COUNT>
{
public:
    /**
     * Convenience typedef for std::bitset<OMG_DDS_STATE_BIT_COUNT>.
     */
    typedef std::bitset<OMG_DDS_STATE_BIT_COUNT> MaskType;

public:
    /**
     * Construct a SampleState with default MaskType.
     */
    SampleState();

    /**
     * Construct a SampleState with MaskType of i.
     *
     * @param i MaskType
     */
    explicit SampleState(uint32_t i);

    /**
     * Copy constructor.
     * Construct a SampleState with existing SampleState.
     *
     * @param src the SampleState to copy from
     */
    SampleState(const SampleState& src);

    /**
     * Copy assignment operator.
     *
     * Copy the contents from one SampleState to another.
     *
     * @param src the SampleState to copy from
     *
     * @return reference to the SampleState instance that was copied to
     */
    SampleState& operator=(const SampleState& src) = default;

    /**
     * Construct a SampleState with existing MaskType.
     *
     * @param src the MaskType to copy from
     */
    SampleState(const MaskType& src);

public:
    /**
     * Get the READ_SAMPLE_STATE.
     *
     * <i>read</i>
     *      - The DataReader has already accessed that
     *        sample by means of read. Had the sample been accessed by take it would
     *        no longer be available to the DataReader.
     *
     * @return the read SampleState
     */
    inline static const SampleState read();

    /**
     * Get the NOT_READ_SAMPLE_STATE.
     *
     *<i>not_read</i>
     *      - The DataReader has not accessed that sample before.
     *
     * @return the not_read SampleState
     */
    inline static const SampleState not_read();

    /**
     * Get any SampleState.
     *
     * Either the sample has already been read or not read.
     *
     * @return any SampleState
     */
    inline static const SampleState any();
};



/**
 * @brief
 * Class to hold sample ViewState information.
 *
 * For each instance (identified by the key), the Data Distribution Service internally
 * maintains a view_state relative to each DataReader. The ViewSate can
 * either be NEW_VIEW_STATE or NOT_NEW_VIEW_STATE.
 * - <b><i>new_view</i></b>
 *      - Either this is the first time that the DataReader
 *        has ever accessed samples of that instance, or else that the DataReader has
 *        accessed previous samples of the instance, but the instance has since been reborn
 *        (i.e. become not-alive and then alive again).
 * - <b><i>not_new_view</i></b>
 *      - The DataReader has already accessed
 *        samples of the same instance and that the instance has not been reborn since.
 *
 * @see for more information: @ref DCPS_Modules_Subscription_SampleInfo "SampleInfo"
 */
class OMG_DDS_API dds::sub::status::ViewState : public std::bitset<OMG_DDS_STATE_BIT_COUNT>
{
public:
    /**
     * Convenience typedef for std::bitset<OMG_DDS_STATE_BIT_COUNT>.
     */
    typedef std::bitset<OMG_DDS_STATE_BIT_COUNT> MaskType;

public:
    /**
     * Construct a ViewState with default MaskType.
     */
    ViewState();

    /**
     * Construct a ViewState with MaskType of i.
     *
     * @param m the MaskType
     */
    explicit ViewState(uint32_t m);

    /**
     * Copy constructor.
     *
     * Construct a ViewState with existing ViewState.
     *
     * @param src the ViewState to copy from
     */
    ViewState(const ViewState& src);

    /**
     * Copy assignment operator.
     *
     * Copy the contents from one ViewState to another.
     *
     * @param src the ViewState to copy from
     *
     * @return reference to the ViewState instance that was copied to
     */
    ViewState& operator=(const ViewState& src) = default;

    /**
     * Construct a ViewState with existing MaskType.
     *
     * @param src the MaskType to copy from
     */
    ViewState(const MaskType& src);

public:
    /**
     * Get the NEW_VIEW_STATE.
     *
     * <i>new_view</i>
     *      - Either this is the first time that the DataReader
     *        has ever accessed samples of that instance, or else that the DataReader has
     *        accessed previous samples of the instance, but the instance has since been reborn
     *        (i.e. become not-alive and then alive again).
     *
     * @return the new_view ViewState
     */
    inline static const ViewState new_view();

    /**
     * Get the NOT_NEW_VIEW_STATE.
     *
     * <i>not_new_view</i>
     *      - The DataReader has already accessed
     *        samples of the same instance and that the instance has not been reborn since.
     *
     * @return the not_new_view ViewState
     */
    inline static const ViewState not_new_view();

    /**
     * Get any ViewState.
     *
     * Either the sample has already been seen or not seen.
     *
     * @return the any ViewState
     */
    inline static const ViewState any();

};



/**
 * @brief
 * Class to hold sample InstanceState information.
 *
 * For each instance the Data Distribution Service internally maintains an
 * InstanceState. The InstanceState can be:
 * - <b><i>alive</i></b>, which indicates that
 *      - samples have been received for the instance
 *      - there are live DataWriter objects writing the instance
 *      - the instance has not been explicitly disposed of (or else samples have been
 *        received after it was disposed of)
 * - <b><i>not_alive_disposed</i></b>, which indicates that
 *     - the instance was disposed
 *       of by a DataWriter, either explicitly by means of the dispose operation or
 *       implicitly in case the autodispose_unregistered_instances field of the
 *       WriterDataLyfecycle QosPolicy equals TRUE when the instance gets
 *       unregistered, WriterDataLifecycle QosPolicy and no new
 *       samples for that instance have been written afterwards.
 * - <b><i>not_alive_no_writers</i></b>, which indicates that
 *      - the instance has been
 *        declared as not-alive by the DataReader because it detected that there are no live
 *        DataWriter objects writing that instance.
 *
 * @see for more information: @ref DCPS_Modules_Subscription_SampleInfo "SampleInfo"
 */
class OMG_DDS_API dds::sub::status::InstanceState : public std::bitset<OMG_DDS_STATE_BIT_COUNT>
{
public:
    /**
     * Convenience typedef for std::bitset<OMG_DDS_STATE_BIT_COUNT>.
     */
    typedef std::bitset<OMG_DDS_STATE_BIT_COUNT> MaskType;

public:
    /**
     * Construct an InstanceState with no state flags set.
     */
    InstanceState();

    /**
     * Construct an InstanceState with an uint32_t m, representing a bit array.
     *
     * @param m the bit array to initialize the bitset with
     */
    explicit InstanceState(uint32_t m);

    /**
     * Copy constructor.
     *
     * Construct an InstanceState with existing InstanceState.
     *
     * @param src the InstanceState to copy from
     */
    InstanceState(const InstanceState& src);

    /**
     * Copy assignment operator.
     *
     * Copy the contents from one InstanceState to another.
     *
     * @param src the InstanceState to copy from
     *
     * @return reference to the InstanceState instance that was copied to
     */
    InstanceState& operator=(const InstanceState& src) = default;

    /**
     * Construct an InstanceState with existing MaskType.
     *
     * @param src the bitset to copy from
     */
    InstanceState(const MaskType& src);

public:
    /**
     * Get ALIVE_INSTANCE_STATE.
     *
     * <i>alive</i>, which indicates that
     *      - samples have been received for the instance
     *      - there are live DataWriter objects writing the instance
     *      - the instance has not been explicitly disposed of (or else samples have been
     *        received after it was disposed of)
     *
     * @return the alive InstanceState
     */
    inline static const InstanceState alive();

    /**
     * Get NOT_ALIVE_DISPOSED_INSTANCE_STATE.
     *
     * <i>not_alive_disposed</i>, which indicates that
     *     - the instance was disposed
     *       of by a DataWriter, either explicitly by means of the dispose operation or
     *       implicitly in case the autodispose_unregistered_instances field of the
     *       WriterDataLyfecycle QosPolicy equals TRUE when the instance gets
     *       unregistered, WriterDataLifecycle QosPolicy and no new
     *       samples for that instance have been written afterwards.
     *
     * @return the not_alive_disposed InstanceState
     */
    inline static const InstanceState not_alive_disposed();

    /**
     * Get NOT_ALIVE_NO_WRITERS_INSTANCE_STATE.
     *
     * <i>not_alive_no_writers</i>, which indicates that
     *      - the instance has been
     *        declared as not-alive by the DataReader because it detected that there are no live
     *        DataWriter objects writing that instance.
     *
     * @return the not_alive_no_writers InstanceState
     */
    inline static const InstanceState not_alive_no_writers();

    /**
     * Get not_alive mask
     *
     * <i>not_alive</i> = not_alive_disposed | not_alive_no_writers:
     * - <i>not_alive_disposed</i>, which indicates that
     *     - the instance was disposed
     *       of by a DataWriter, either explicitly by means of the dispose operation or
     *       implicitly in case the autodispose_unregistered_instances field of the
     *       WriterDataLyfecycle QosPolicy equals TRUE when the instance gets
     *       unregistered, WriterDataLifecycle QosPolicy and no new
     *       samples for that instance have been written afterwards.
     * - <i>not_alive_no_writers</i>, which indicates that
     *      - the instance has been
     *        declared as not-alive by the DataReader because it detected that there are no live
     *        DataWriter objects writing that instance.
     *
     * @return the not_alive_mask InstanceState
     */
    inline static const InstanceState not_alive_mask();

    /**
     * Get any InstanceState.
     *
     * This Instance is either in existence or not in existence.
     *
     * @return the any InstanceState
     */
    inline static const InstanceState any();

};



/**
 * @brief
 * Class to hold sample DataState information.
 *
 * The DataState is part of dds::sub::SampleInfo but can also be used as data filter for
 * @ref dds::sub::DataReader "DataReaders".
 *
 * The DataState contains the following:
 *  - The sample_state of the Data value (i.e., if the sample has already been READ or NOT_READ by that same DataReader).
 *  - The view_state of the related instance (i.e., if the instance is NEW, or NOT_NEW for that DataReader).
 *  - The instance_state of the related instance (i.e., if the instance is ALIVE, NOT_ALIVE_DISPOSED, or NOT_ALIVE_NO_WRITERS).
 *
 * @see for more information: @ref DCPS_Modules_Subscription_SampleInfo "SampleInfo"
 */
class OMG_DDS_API dds::sub::status::DataState
{
public:
    /**
     * Create a DataState instance.
     *
     * Construct a DataState with:
     * - SampleState::any
     * - ViewState::any
     * - InstanceState::any
     *
     */
    DataState()
        : ss_(dds::sub::status::SampleState::any()),
          vs_(dds::sub::status::ViewState::any()),
          is_(dds::sub::status::InstanceState::any())
    { }

    /**
     * Create a DataState instance.
     *
     * Construct a DataState with:
     * - Given SampleState ss
     * - ViewState::any
     * - InstanceState::any
     *
     * @param ss the SampleState to construct DataState from
     */
    /* implicit */ DataState(const dds::sub::status::SampleState& ss)
        : ss_(ss),
          vs_(dds::sub::status::ViewState::any()),
          is_(dds::sub::status::InstanceState::any())
    { }

    /**
     * Create a DataState instance.
     *
     * Construct a DataState with:
     * - SampleState::any
     * - Given ViewState vs
     * - InstanceState::any
     *
     * @param vs the ViewState to construct DataState from
     */
    /* implicit */ DataState(const dds::sub::status::ViewState& vs)
        : ss_(dds::sub::status::SampleState::any()),
          vs_(vs),
          is_(dds::sub::status::InstanceState::any())
    { }

    /**
     * Create a DataState instance.
     *
     * Construct a DataState with:
     * - SampleState::any
     * - ViewState::any
     * - Given InstanceState is
     *
     * @param is InstanceState to construct DataState from
     */
    /* implicit */ DataState(const dds::sub::status::InstanceState& is)
        : ss_(dds::sub::status::SampleState::any()),
          vs_(dds::sub::status::ViewState::any()),
          is_(is)
    { }

    /**
     * Create a DataState instance.
     *
     * Construct a DataState with:
     * - Given SampleState ss
     * - Given ViewState vs
     * - Given InstanceState is
     *
     * @param ss SampleState
     * @param vs ViewState
     * @param is InstanceState
     */
    DataState(const dds::sub::status::SampleState& ss,
              const dds::sub::status::ViewState& vs,
              const dds::sub::status::InstanceState& is)
        : ss_(ss), vs_(vs), is_(is)
    { }

    /**
     * Set SampleState.
     *
     * @param ss SampleState
     */
    DataState& operator << (const dds::sub::status::SampleState& ss)
    {
        ss_ = ss;
        return *this;
    }

    /**
     * Set InstanceState.
     *
     * @param is InstanceState
     */
    DataState& operator << (const dds::sub::status::InstanceState& is)
    {
        is_ = is;
        return *this;
    }

    /**
     * Set ViewState.
     *
     * @param vs ViewState
     */
    DataState& operator << (const dds::sub::status::ViewState& vs)
    {
        vs_ = vs;
        return *this;
    }

    /**
     * Get SampleState.
     *
     * @param ss SampleState
     * @return the DataState
     */
    const DataState& operator >> (dds::sub::status::SampleState& ss) const
    {
        ss = ss_;
        return *this;
    }

    /**
     * Get InstanceState.
     *
     * @param is InstanceState
     * @return the DataState
     */
    const DataState& operator >> (dds::sub::status::InstanceState& is) const
    {
        is = is_;
        return *this;
    }

    /**
     * Get ViewState.
     *
     * @param vs ViewState
     * @return the DataState
     */
    const DataState& operator >> (dds::sub::status::ViewState& vs) const
    {
        vs = vs_;
        return *this;
    }

    /**
     * Check if this DataState is equal with another
     *
     * @return true if equal
     */
     bool operator ==(const DataState& o) const
     {
         return ((ss_ == o.ss_) && (vs_ == o.vs_) && (is_ == o.is_));
     }

     /**
      * Check if this DataState is not equal with another
      *
      * @return true if not equal
      */
     bool operator !=(const DataState& o) const
     {
         return !operator==(o);
     }

    /**
     * Get SampleState.
     *
     * @return the SampleState
     */
    const dds::sub::status::SampleState& sample_state() const
    {
        return ss_;
    }

    /**
     * Set SampleState.
     *
     * @param ss SampleState
     */
    void sample_state(const dds::sub::status::SampleState& ss)
    {
        *this << ss;
    }

    /**
     * Get InstanceState.
     *
     * @return the InstanceState
     */
    const dds::sub::status::InstanceState& instance_state() const
    {
        return is_;
    }

    /**
     * Set InstanceState.
     *
     * @param is InstanceState
     */
    void instance_state(const dds::sub::status::InstanceState& is)
    {
        *this << is;
    }

    /**
     * Get ViewState.
     *
     * @return the ViewState
     */
    const dds::sub::status::ViewState& view_state() const
    {
        return vs_;
    }

    /**
     * Set ViewState.
     *
     * @param vs ViewState
     */
    void view_state(const dds::sub::status::ViewState& vs)
    {
        *this << vs;
    }

    /**
     * Create a DataState instance.
     *
     * Return a DataState with:
     * - SampleState::any
     * - ViewState::any
     * - InstanceState::any
     *
     * @return the any DataState
     */
    static DataState any()
    {
        return DataState(dds::sub::status::SampleState::any(),
                         dds::sub::status::ViewState::any(),
                         dds::sub::status::InstanceState::any());
    }

    /**
     * Create a DataState instance.
     *
     * Return a DataState with:
     * - SampleState::not_read
     * - ViewState::any
     * - InstanceState::alive
     *
     * @return the new_data DataState
     */
    static DataState new_data()
    {
        return DataState(dds::sub::status::SampleState::not_read(),
                         dds::sub::status::ViewState::any(),
                         dds::sub::status::InstanceState::alive());
    }

    /**
     * Create a DataState instance.
     *
     * Return a DataState with:
     * - SampleState::any
     * - ViewState::any
     * - InstanceState::alive
     *
     * @return the any_data DataState
     */
    static DataState any_data()
    {
        return DataState(dds::sub::status::SampleState::any(),
                         dds::sub::status::ViewState::any(),
                         dds::sub::status::InstanceState::alive());
    }

    /**
     * Create a DataState instance.
     *
     * Return a DataState with:
     * - SampleState::any
     * - ViewState::new_view
     * - InstanceState::alive
     *
     * @return the new_instance DataState
     */
    static DataState new_instance()
    {
        return DataState(dds::sub::status::SampleState::any(),
                         dds::sub::status::ViewState::new_view(),
                         dds::sub::status::InstanceState::alive());
    }
private:
    dds::sub::status::SampleState ss_;
    dds::sub::status::ViewState vs_;
    dds::sub::status::InstanceState is_;

};

#endif /* OMG_DDS_SUB_DATA_STATE_HPP_ */
