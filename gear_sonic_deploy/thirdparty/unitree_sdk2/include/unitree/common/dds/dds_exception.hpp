#ifndef __UT_DDS_EXCEPTION_HPP__
#define __UT_DDS_EXCEPTION_HPP__

#include <unitree/common/exception.hpp>
#include <unitree/common/dds/dds_error.hpp>

#define __UT_DDS_EXCEPTION_MESSAGE(e, d)        \
    std::string("Catch dds::core exception. Class:") + __UT_STR(d) + ", Message:" + e.what();

#define __UT_DDS_EXCEPTION_CATCH(except, l, t)  \
catch (const except & e)                        \
{                                               \
    if (l || t)                                 \
    {                                           \
        std::string __t9b78e5r = __UT_DDS_EXCEPTION_MESSAGE(e, except);  \
        if (l)                                  \
        {                                       \
            LOG_ERROR(l, __t9b78e5r);           \
        }                                       \
        if (t)                                  \
        {                                       \
            UT_THROW(DdsException, __t9b78e5r); \
        }                                       \
    }                                           \
}

#define UT_DDS_EXCEPTION_TRY                    \
try                                             \
{

#define UT_DDS_EXCEPTION_CATCH(l, t)            \
}                                               \
__UT_DDS_EXCEPTION_CATCH(::dds::core::Error, l, t)                    \
__UT_DDS_EXCEPTION_CATCH(::dds::core::InvalidArgumentError, l, t)     \
__UT_DDS_EXCEPTION_CATCH(::dds::core::TimeoutError, l, t)             \
__UT_DDS_EXCEPTION_CATCH(::dds::core::UnsupportedError, l, t)         \
__UT_DDS_EXCEPTION_CATCH(::dds::core::AlreadyClosedError, l, t)       \
__UT_DDS_EXCEPTION_CATCH(::dds::core::IllegalOperationError, l, t)    \
__UT_DDS_EXCEPTION_CATCH(::dds::core::NotEnabledError, l, t)          \
__UT_DDS_EXCEPTION_CATCH(::dds::core::PreconditionNotMetError, l, t)  \
__UT_DDS_EXCEPTION_CATCH(::dds::core::ImmutablePolicyError, l, t)     \
__UT_DDS_EXCEPTION_CATCH(::dds::core::InconsistentPolicyError, l, t)  \
__UT_DDS_EXCEPTION_CATCH(::dds::core::OutOfResourcesError, l, t)      \
__UT_DDS_EXCEPTION_CATCH(::dds::core::InvalidDowncastError, l, t)     \
__UT_DDS_EXCEPTION_CATCH(::dds::core::NullReferenceError, l, t)       \
__UT_DDS_EXCEPTION_CATCH(::dds::core::InvalidDataError, l, t)         \
__UT_DDS_EXCEPTION_CATCH(::dds::core::Exception, l, t)                \
__UT_DDS_EXCEPTION_CATCH(std::exception, l, t)

namespace unitree
{
namespace common
{
UT_DECL_EXCEPTION(DdsException, UT_ERR_DDS, UT_DESC_ERR(UT_ERR_DDS))
}
}

#endif//__UT_DDS_EXCEPTION_HPP__
