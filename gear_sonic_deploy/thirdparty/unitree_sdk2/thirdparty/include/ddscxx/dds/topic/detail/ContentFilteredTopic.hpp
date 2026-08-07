#ifndef OMG_DDS_TOPIC_DETAIL_CONTENTFILTEREDTOPIC_HPP_
#define OMG_DDS_TOPIC_DETAIL_CONTENTFILTEREDTOPIC_HPP_

/* Copyright 2010, Object Management Group, Inc.
 * Copyright 2010, PrismTech, Inc.
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

#include <string>
#include <vector>

#include <dds/core/detail/conformance.hpp>
#include <dds/core/types.hpp>
#include <dds/topic/Topic.hpp>
#include <dds/topic/Filter.hpp>
#include <org/eclipse/cyclonedds/topic/TopicDescriptionDelegate.hpp>
#include <org/eclipse/cyclonedds/core/ScopedLock.hpp>
#include <org/eclipse/cyclonedds/sub/AnyDataReaderDelegate.hpp>

#ifdef OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT

// Required for C++11
#if __cplusplus == 201103L
#include <org/eclipse/cyclonedds/core/Missing.hpp>
#endif

// meta-programming helpers
namespace detail
{
template<typename T>
struct remove_first_from_tuple;

template<typename T, typename ... Ts>
struct remove_first_from_tuple<std::tuple<T, Ts...>>
{
  using type = std::tuple<Ts...>;
};

template<typename T>
using remove_first_from_tuple_t = typename remove_first_from_tuple<T>::type;

template<class F>
struct callable_traits;

template<class R, class ... Args>
struct callable_traits<R(Args...)>
{
  using return_type = R;

  static constexpr std::size_t Arity = sizeof...(Args);

  template < std::size_t N, std::enable_if_t<N<Arity> * = nullptr>
  struct argument
  {
    using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
  };

  using argument_tuple = std::tuple<Args...>;
};

// function pointer
template<class R, class ... Args>
struct callable_traits<R (*)(Args...)>
  : public callable_traits<R(Args...)> {};

// member function pointer
template<class C, class R, class ... Args>
struct callable_traits<R (C::*)(Args...)>
  : public callable_traits<R(C &, Args...)> {};

// const member function pointer
template<class C, class R, class ... Args>
struct callable_traits<R (C::*)(Args...) const>
  : public callable_traits<R(C &, Args...)> {};

// member object pointer
template<class C, class R>
struct callable_traits<R(C::*)>
  : public callable_traits<R(C &)> {};

// functor
template<class F>
struct callable_traits
{
private:
  using call_type = callable_traits<decltype(&F::operator())>;

public:
  using return_type = typename call_type::return_type;

  static constexpr std::size_t Arity = call_type::Arity - 1;

  template < std::size_t N, std::enable_if_t<N<Arity> * = nullptr>
  struct argument
  {
    using type = typename call_type::template argument<N + 1>::type;
  };

  using argument_tuple = remove_first_from_tuple_t<typename call_type::argument_tuple>;
};

template<class F>
struct callable_traits<F &>
  : public callable_traits<F> {};

template<class F>
struct callable_traits<F &&>
  : public callable_traits<F> {};

// Concepts of possible callbacks
template<class F, class T>
using sample_only_t = std::enable_if_t<callable_traits<std::decay_t<F>>::Arity == 1 &&
    std::is_same<typename callable_traits<std::decay_t<F>>::template argument<0>::type,
    const T &>::value &&
    std::is_same<typename callable_traits<std::decay_t<F>>::return_type, bool>::value
>;

template<class F>
using sample_info_only_t = std::enable_if_t<callable_traits<std::decay_t<F>>::Arity == 1 &&
    std::is_same<typename callable_traits<std::decay_t<F>>::template argument<0>::type,
    const dds::sub::SampleInfo &>::value &&
    std::is_same<typename callable_traits<std::decay_t<F>>::return_type, bool>::value
>;

template<class F, class T>
using sample_and_sample_info_t = std::enable_if_t<callable_traits<std::decay_t<F>>::Arity == 2 &&
    std::is_same<typename callable_traits<std::decay_t<F>>::template argument<0>::type,
    const T &>::value &&
    std::is_same<typename callable_traits<std::decay_t<F>>::template argument<1>::type,
    const dds::sub::SampleInfo &>::value &&
    std::is_same<typename callable_traits<std::decay_t<F>>::return_type, bool>::value
>;

} // namespace detail


namespace dds {
namespace topic {
namespace detail {

class FunctorHolderBase
{
public:
    FunctorHolderBase() = default;
    virtual ~FunctorHolderBase() = default;
    FunctorHolderBase(const FunctorHolderBase &) = default;
    FunctorHolderBase & operator=(const FunctorHolderBase &) = default;
    FunctorHolderBase(FunctorHolderBase &&) = default;
    FunctorHolderBase & operator=(FunctorHolderBase &&) = default;

    virtual bool check_sample(const void * sample, const dds_sample_info_t * sample_info) = 0;

    static bool c99_check_sample(const void *sample, void *arg)
    {
        auto funcHolder = static_cast<FunctorHolderBase *>(arg);
        return funcHolder->check_sample(sample, nullptr);
    }

    static bool c99_check_sample_info(const dds_sample_info_t * sampleinfo, void * arg)
    {
        auto funcHolder = static_cast<FunctorHolderBase *>(arg);
        return funcHolder->check_sample(nullptr, sampleinfo);
    }

    static bool c99_check_sample_and_sample_info(
      const void * sample,
      const dds_sample_info_t * sampleinfo, void * arg)
    {
        auto funcHolder = static_cast<FunctorHolderBase *>(arg);
        return funcHolder->check_sample(sample, sampleinfo);
    }
};

template<typename FUN, typename T, class = void>
class FunctorHolder;

template <typename FUN, typename T>
class FunctorHolder<FUN, T, ::detail::sample_only_t<FUN, T>>: public FunctorHolderBase
{
public:
    /* Remove const to be able to call non-const functors. */
    FunctorHolder(FUN functor)
    : myFunctor(std::move(functor)) {}

    bool check_sample(const void * sample, const dds_sample_info_t *) override
    {
        return myFunctor(*(reinterpret_cast<const T*>(sample)));
    }

private:
    FUN myFunctor;
};

template<typename FUN, typename T>
class FunctorHolder<FUN, T, ::detail::sample_info_only_t<FUN>>: public FunctorHolderBase
{
public:
    /* Remove const to be able to call non-const functors. */
    FunctorHolder(FUN functor)
    : myFunctor(std::move(functor)) {}

    bool check_sample(const void *, const dds_sample_info_t * sampleinfo) override
    {
      dds::sub::SampleInfo cxxSampleInfo;
      org::eclipse::cyclonedds::sub::AnyDataReaderDelegate::copy_sample_infos(
        *sampleinfo,
        cxxSampleInfo);
      return myFunctor(cxxSampleInfo);
    }

private:
    FUN myFunctor;
};

template<typename FUN, typename T>
class FunctorHolder<FUN, T, ::detail::sample_and_sample_info_t<FUN, T>>: public FunctorHolderBase
{
public:
    /* Remove const to be able to call non-const functors. */
    FunctorHolder(FUN functor)
    : myFunctor(std::move(functor)) {}

    bool check_sample(const void * sample, const dds_sample_info_t * sampleinfo) override
    {
      dds::sub::SampleInfo cxxSampleInfo;
      org::eclipse::cyclonedds::sub::AnyDataReaderDelegate::copy_sample_infos(*sampleinfo,
        cxxSampleInfo);
      return myFunctor(*(reinterpret_cast<const T*>(sample)), cxxSampleInfo);
    }

private:
    FUN myFunctor;
};

template <typename T>
class ContentFilteredTopic  :
    public virtual org::eclipse::cyclonedds::topic::TopicDescriptionDelegate,
    public virtual org::eclipse::cyclonedds::core::DDScObjectDelegate
{
public:
    ContentFilteredTopic(
        const dds::topic::Topic<T>& topic,
        const std::string& name,
        const dds::topic::Filter& filter)
        : org::eclipse::cyclonedds::core::DDScObjectDelegate(),
          org::eclipse::cyclonedds::topic::TopicDescriptionDelegate(topic.domain_participant(), name, topic.type_name()),
          myTopic(topic),
          myFilter(filter),
          myFunctor(nullptr)
    {
        topic.delegate()->incrNrDependents();
        this->myParticipant.delegate()->add_cfTopic(*this);
        this->ser_type_ = topic->get_ser_type();
    }

    virtual ~ContentFilteredTopic()
    {
        if (!this->closed) {
            try {
                this->close();
            } catch (...) {
                /* Empty: the exception throw should have already traced an error. */
            }
        }
        // delete the functor
        delete myFunctor;
        myFunctor = nullptr;
    }

    virtual void close()
    {
        org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);

        myTopic.delegate()->decrNrDependents();

        // Remove the ContentFilteredTopic from the list of topics in its participant.
        this->myParticipant.delegate()->remove_cfTopic(*this);

        org::eclipse::cyclonedds::core::ObjectDelegate::close();
    }

    void
    init(org::eclipse::cyclonedds::core::ObjectDelegate::weak_ref_type weak_ref)
    {
        /* Set weak_ref before passing ourselves to other isocpp objects. */
        this->set_weak_ref(weak_ref);
        /* Register topic at participant. */
        this->myParticipant.delegate()->add_cfTopic(*this);
    }

private:
#if 0
    void validate_filter()
    {
        q_expr expr = NULL;
        uint32_t length;
        c_value *params;

        length = myFilter.parameters_length();
        if (length < 100) {
            expr = q_parse(myFilter.expression().c_str());
            if (!expr ) {
                ISOCPP_THROW_EXCEPTION(ISOCPP_INVALID_ARGUMENT_ERROR,
                        "filter_expression '%s' is invalid", myFilter.expression().c_str());
            }
        } else {
            ISOCPP_THROW_EXCEPTION(ISOCPP_INVALID_ARGUMENT_ERROR,
                    "Invalid number of filter_parameters '%d', maximum is 99", length);
        }

        u_topic uTopic = (u_topic)(myTopic.delegate()->get_user_handle());

        params = reader_parameters();
        if (!u_topicContentFilterValidate2(uTopic, expr, params)) {
            ISOCPP_THROW_EXCEPTION(ISOCPP_INVALID_ARGUMENT_ERROR,
                    "filter_expression '%s' is invalid.", myFilter.expression().c_str());
        }
        q_dispose(expr);
        os_free(params);
    }
#endif
public:
    std::string reader_expression() const
    {
        std::string rExpr;

        rExpr += "select * from ";
        rExpr += myTopic.name();
        rExpr += " where ";
        rExpr += myFilter.expression();
        return rExpr;
    }
#if 0
    c_value *reader_parameters() const
    {
        c_value *params = NULL;
        uint32_t n, length;
        org::eclipse::cyclonedds::topic::FilterDelegate::const_iterator paramIterator;

        length = myFilter.parameters_length();
        params = (c_value *)os_malloc(length * sizeof(struct c_value));
        for (n = 0, paramIterator = myFilter.begin(); n < length; n++, paramIterator++) {
            params[n] = c_stringValue(const_cast<char *>(paramIterator->c_str()));
        }
        return params;
    }
#endif
    /**
    *  @internal Accessor to return the topic filter.
    * @return The dds::topic::Filter in effect on this topic.
    */
    const dds::topic::Filter& filter() const
    {
        return myFilter;
    }

    /**
     *  @internal Sets the filter parameters for this content filtered topic.
     * @param begin The iterator holding the first string param
     * @param end The last item in the string iteration
     */
    template <typename FWIterator>
    void filter_parameters(const FWIterator& begin, const FWIterator& end)
    {
        ISOCPP_THROW_EXCEPTION(ISOCPP_UNSUPPORTED_ERROR, "Changing of Filter parameters is currently not supported.");
        myFilter.parameters(begin, end);
        //@todo validate_filter();
    }

    const dds::topic::Topic<T>& topic() const
    {
        return myTopic;
    }

    const std::string& filter_expression() const
    {
        return myFilter.expression();
    }

    const dds::core::StringSeq filter_parameters() const
    {
        return dds::core::StringSeq(myFilter.begin(), myFilter.end());
    }
#if 0
    dds::topic::TTopicDescription<TopicDescriptionDelegate> clone()
    {
        org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);

        typename dds::topic::ContentFilteredTopic<T, ContentFilteredTopic>::DELEGATE_REF_T ref(
                new ContentFilteredTopic<T>(this->myTopic, this->myTopicName, this->myFilter));
        ref->init(ref);

        return dds::topic::ContentFilteredTopic<T, ContentFilteredTopic>(ref);
    }
#endif

    template<class Functor, ::detail::sample_only_t<Functor, T> * = nullptr>
    void filter_function(Functor && func)
    {
        dds_topic_filter flt;
        flt.mode = DDS_TOPIC_FILTER_SAMPLE_ARG;
        flt.f.sample_arg = &FunctorHolderBase::c99_check_sample;
        filter_function_internal(std::forward<Functor>(func), &flt);
    }

    template<class Functor, ::detail::sample_info_only_t<Functor> * = nullptr>
    void filter_function(Functor && func)
    {
        dds_topic_filter flt;
        flt.mode = DDS_TOPIC_FILTER_SAMPLEINFO_ARG;
        flt.f.sampleinfo_arg = &FunctorHolderBase::c99_check_sample_info;
        filter_function_internal(std::forward<Functor>(func), &flt);
     }

    template<class Functor, ::detail::sample_and_sample_info_t<Functor, T> * = nullptr>
    void filter_function(Functor && func)
    {
        dds_topic_filter flt;
        flt.mode = DDS_TOPIC_FILTER_SAMPLE_SAMPLEINFO_ARG;
        flt.f.sample_sampleinfo_arg = &FunctorHolderBase::c99_check_sample_and_sample_info;
        filter_function_internal(std::forward<Functor>(func), &flt);
    }

private:
    template <typename Functor>
    void filter_function_internal(Functor && func, dds_topic_filter * flt)
    {
        /* Make a private copy of the topic so my filter doesn't bother the original topic. */
        dds_qos_t* ddsc_qos = myTopic.qos()->ddsc_qos();
        ddsi_sertype *st = org::eclipse::cyclonedds::topic::TopicTraits<T>::getSerType();
        dds_entity_t cfTopic = dds_create_topic_sertype(
            myTopic.domain_participant().delegate()->get_ddsc_entity(), myTopic.name().c_str(), &st, ddsc_qos, NULL, NULL);
        dds_delete_qos(ddsc_qos);
        this->set_ddsc_entity(cfTopic);

        org::eclipse::cyclonedds::core::ScopedObjectLock scopedLock(*this);
        if (this->myFunctor)
        {
            delete this->myFunctor;
        }
        myFunctor = new FunctorHolder<Functor, T>(std::forward<Functor>(func));
        flt->arg = myFunctor;
        dds_set_topic_filter_extended(cfTopic, flt);
    }

    dds::topic::Topic<T> myTopic;
    dds::topic::Filter myFilter;
    FunctorHolderBase *myFunctor;
};

}
}
}

#endif /* OMG_DDS_CONTENT_SUBSCRIPTION_SUPPORT */

#endif /* OMG_DDS_TOPIC_DETAIL_CONTENTFILTEREDTOPIC_HPP_ */
