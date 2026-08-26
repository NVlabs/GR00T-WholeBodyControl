#ifndef OMG_DDS_CORE_REFMACROS_HPP_
#define OMG_DDS_CORE_REFMACROS_HPP_

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

#include <dds/core/macros.hpp>
#include <dds/core/ref_traits.hpp>

////////////////////////////////////////////////////////////////////////////////
// Macros
////////////////////////////////////////////////////////////////////////////////

#define DECLARE_TYPE_TRAITS(TYPE) \
    typedef TYPE  TYPE##_T;\
    typedef typename ::dds::core::smart_ptr_traits<TYPE>::ref_type       TYPE##_REF_T;\
    typedef typename ::dds::core::smart_ptr_traits<TYPE>::weak_ref_type  TYPE##_WEAK_REF_T;


/*
 * This macro defines all the functions that Reference Types have to implement.
 */
/*
 *
 *
 */
////////////////////////////////////////////////////////////////////////////////
// Defines all the types, functions and attributes required for a Reference type
// without default ctor.
//

#define OMG_DDS_REF_TYPE_BASE(TYPE, BASE, DELEGATE)     \
    public: \
    typedef typename ::dds::core::smart_ptr_traits< DELEGATE >::ref_type      DELEGATE_REF_T; \
    typedef typename ::dds::core::smart_ptr_traits< DELEGATE >::weak_ref_type DELEGATE_WEAK_REF_T; \
    \
    private:\
    const typename ::dds::core::Reference< DELEGATE >::DELEGATE_REF_T& impl() const \
    { return ::dds::core::Reference< DELEGATE >::impl_; }\
    typename ::dds::core::Reference< DELEGATE >::DELEGATE_REF_T& impl() \
    { return ::dds::core::Reference< DELEGATE >::impl_; }\
    \
    public: \
    typedef BASE< DELEGATE >                                                  BASE_T;\
    \
    public: \
    TYPE(const dds::core::null_type&) : dds::core::Reference< DELEGATE >(static_cast<DELEGATE*>(NULL)) { } \
    \
    TYPE& operator=(const dds::core::null_type& rhs) {\
        (void)rhs; \
        *this = TYPE(dds::core::null); \
        return *this; \
    } \
    \
    public:\
    TYPE(const DELEGATE_REF_T& ref) \
        : dds::core::Reference< DELEGATE >(ref)\
    { }

#define OMG_DDS_IMPLICIT_REF_BASE(TYPE)         \
    public:\
    template <typename H__> \
    TYPE(const H__& h)    \
    { \
        if (h.is_nil()) { \
            /* We got a null object and are not really able to do a typecheck here. */ \
            /* So, just set a null object. */ \
            *this = dds::core::null; \
        } else { \
            this->::dds::core::Reference< DELEGATE_T >::impl_ = ::std::dynamic_pointer_cast< DELEGATE_T >(h.delegate());  \
            if (h.delegate() != this->::dds::core::Reference< DELEGATE_T >::impl_) { \
                throw dds::core::IllegalOperationError(std::string("Attempted invalid cast: ") + typeid(h).name() + " to " + typeid(*this).name()); \
            } \
        } \
    }\
    \
    template <typename T__> \
    TYPE& \
    operator=(const T__& rhs) { \
        const TYPE &t = rhs; \
        if (this != &t) { \
            if (rhs.is_nil()) { \
                /* We got a null object and are not really able to do a typecheck here. */ \
                /* So, just set a null object. */ \
                *this = dds::core::null; \
            } else { \
                TYPE other(rhs); \
                /* Dont have to copy when the delegate impl is the same. */ \
                if (other.delegate() != this->::dds::core::Reference< DELEGATE_T >::impl_) { \
                    *this = other; \
                } \
            } \
        } \
        return *this; \
    }

#define OMG_DDS_EXPLICIT_REF_BASE_DECL(TYPE, FROM)         \
    public:\
    TYPE(const FROM& h);    \
    \
    TYPE& \
    operator=(const FROM& rhs);

#define OMG_DDS_EXPLICIT_REF_BASE(TYPE, FROM)         \
    public:\
    /* We need to support both const and non-const conversion explicitly, because it could */ \
    /* be that the class has a constructor that takes a single non-const template argument. */ \
    /* That non-const templated constructor is a closer match than TYPE(const FROM& h); when */ \
    /* creating the TYPE class from a non-const FROM object. */ \
    TYPE(FROM& h) { this->explicit_conversion(const_cast<const FROM&>(h)); }    \
    TYPE(const FROM& h) { this->explicit_conversion(h); }    \
    private:\
    void explicit_conversion(const FROM& h) \
    { \
        if (h.is_nil()) { \
            /* We got a null object and are not really able to do a typecheck here. */ \
            /* So, just set a null object. */ \
            *this = dds::core::null; \
        } else { \
            this->::dds::core::Reference<DELEGATE_T>::impl_ = ::std::dynamic_pointer_cast<DELEGATE_T>(h.delegate());  \
            if (h.delegate() != this->::dds::core::Reference< DELEGATE_T >::impl_) { \
                throw dds::core::IllegalOperationError(std::string("Attempted invalid cast: ") + typeid(h).name() + " to " + typeid(*this).name()); \
            } \
        } \
    }\
    \
    public:\
    TYPE& \
    operator=(const FROM& rhs) { \
        const TYPE &t = rhs; \
        if (this != &t) { \
            if (rhs.is_nil()) { \
                /* We got a null object and are not really able to do a typecheck here. */ \
                /* So, just set a null object. */ \
                *this = dds::core::null; \
            } else { \
                TYPE other(rhs); \
                /* Dont have to copy when the delegate is the same. */ \
                if (other.delegate() != this->::dds::core::Reference< DELEGATE_T >::impl_) { \
                    *this = other; \
                } \
            } \
        } \
        return *this; \
    }


////////////////////////////////////////////////////////////////////////////////
// Declares a reference type equipped with a default ctor.
//
#define OMG_DDS_REF_TYPE_PROTECTED_DC(TYPE, BASE, DELEGATE)                 \
public:                                                                     \
    typedef DELEGATE            DELEGATE_T;                                 \
    OMG_DDS_REF_TYPE_BASE(TYPE, BASE, DELEGATE_T)                           \
protected:                                                                  \
    TYPE() { }

#define OMG_DDS_REF_TYPE_PROTECTED_DC_T(TYPE, BASE, T_PARAM, DELEGATE)      \
public:                                                                     \
    typedef DELEGATE<T_PARAM>   DELEGATE_T;                                 \
    OMG_DDS_REF_TYPE_BASE(TYPE, BASE, DELEGATE_T)                           \
    protected:                                                              \
    TYPE() { }

#define OMG_DDS_REF_TYPE_NO_DC(TYPE, BASE, DELEGATE)                        \
public:                                                                     \
    typedef DELEGATE            DELEGATE_T;                                 \
    OMG_DDS_REF_TYPE_BASE(TYPE, BASE, DELEGATE_T)

#define OMG_DDS_REF_TYPE_DELEGATE_C(TYPE, BASE, DELEGATE)                   \
    OMG_DDS_REF_TYPE_PROTECTED_DC(TYPE, BASE, DELEGATE)                     \
    public:                                                                 \
    TYPE(DELEGATE_T *impl) : dds::core::Reference< DELEGATE_T >(impl) { }

#endif /* OMG_DDS_CORE_REFMACROS_HPP_ */
