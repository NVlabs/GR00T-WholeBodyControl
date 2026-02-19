#ifndef OMG_DDS_CORE_EXCEPTION_HPP_
#define OMG_DDS_CORE_EXCEPTION_HPP_

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

#include <stdexcept>
#include <string>
#include <dds/core/macros.hpp>

#if defined _MSC_VER
#   pragma warning (push)
#   pragma warning (disable:4275) // non dll-interface class 'std::foo_error' used as base for dll-interface class 'dds::core::BarError'
#endif

namespace dds
{
namespace core
{

/**
 * @brief
 * %Exception: base class for specified DDS Exceptions.
 *
 * DDS PIM Return Code            | DDS-PSM-CXX Exception Class   | std C++ Parent Exception
 * -------------------            | ---------------------------   | ------------------------
 * RETCODE_OK                     | Normal return; no exception   | N/A
 * RETCODE_NO_DATA                | Normal return with informational state attached    | N/A
 * RETCODE_ERROR                  | Error                         | std::logic_error
 * RETCODE_BAD_PARAMETER          | InvalidArgumentError          | std::invalid_argument
 * RETCODE_TIMEOUT                | TimeoutError                  | std::runtime_error
 * RETCODE_UNSUPPORTED            | UnsupportedError              | std::logic_error
 * RETCODE_ALREADY_DELETED        | AlreadyClosedError            | std::logic_error
 * RETCODE_ILLEGAL_OPERATION      | IllegalOperationError         | std::logic_error
 * RETCODE_NOT_ENABLED            | NotEnabledError               | std::logic_error
 * RETCODE_PRECONDITION_NOT_MET   | PreconditionNotMetError       | std::logic_error
 * RETCODE_IMMUTABLE_POLICY       | ImmutablePolicyError          | std::logic_error
 * RETCODE_INCONSISTENT_POLICY    | InconsistentPolicyError       | std::logic_error
 * RETCODE_OUT_OF_RESOURCES       | OutOfResourcesError           | std::runtime_error
 *
 * The DDS-PSM-Cxx maps error codes to C++ exceptions defined in the dds::core namespace and
 * inheriting from a base Exception class and the appropriate standard C++ exception.
 * Table 7.3 lists the mapping between error codes as defined in the DDS PIM and C++ exceptions
 * as used in this specification. Exceptions have value semantics; this means that they must
 * always have deep copy semantics.
 * The full list of exceptions is included in the file dds/core/Exceptions.hpp.
 *
 */
class OMG_DDS_API Exception
{
protected:
    Exception();
public:
    /** @cond */
    virtual ~Exception() throw();
    /** @endcond */

public:
    /**
     * Retrieve information about the exception that was thrown.
     *
     * Example
     * @code{.cpp}
     * try {
     *     // Do something that will trigger a dds exception, like:
     *     dds::domain::DomainParticipant participant = dds::core::null;
     *     participant.domain_id();
     * } catch (const dds::core::Exception& e) {
     *     std::cout << e.what() << std::endl;
     * }
     * @endcode
     * %Exception information (of the NullReferenceError in this case)
     * @code
     * Null reference: Reference[157] == dds::core::null
     * ========================================================================================
     * Context     : dds::domain::DomainParticipant::domain_id
     * Date        : Wed Oct 21 19:28:00 CET 2015
     * Node        : DeLorean
     * Process     : flux_capacitor <15423>
     * Thread      : mr_fusion b6f25700
     * Internals   : ReferenceImpl.hpp/157/V6.6.0
     * ----------------------------------------------------------------------------------------
     * Report      : Null reference: Reference[157] == dds::core::null
     * Internals   : dds::core::Reference<DELEGATE>::delegate/ReferenceImpl.hpp/157
     * @endcode
     *
     * @return Exception information
     */
    virtual const char* what() const throw() = 0;
};

/**
 * @brief
 * %Exception: Generic, unspecified error.
 */
class OMG_DDS_API Error : public Exception, public std::logic_error
{
/** @cond */
public:
    explicit Error(const std::string& msg);
    Error(const Error& src);
    virtual ~Error() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: The object target of this operation has already been closed.
 */
class OMG_DDS_API AlreadyClosedError : public Exception, public std::logic_error
{
/** @cond */
public:
    explicit AlreadyClosedError(const std::string& msg);
    AlreadyClosedError(const AlreadyClosedError& src);
    virtual ~AlreadyClosedError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: An operation was invoked on an inappropriate object or at an inappropriate time.
 *
 * This is determined by policies set by the specification or the Service implementation.
 *
 * There is no precondition that could be changed to make the operation succeed.
 */
class OMG_DDS_API IllegalOperationError : public Exception, public std::logic_error
{
/** @cond */
public:
    explicit IllegalOperationError(const std::string& msg);
    IllegalOperationError(const IllegalOperationError& src);
    virtual ~IllegalOperationError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: Application attempted to modify an immutable QosPolicy.
 */
class OMG_DDS_API ImmutablePolicyError : public Exception, public std::logic_error
{
/** @cond */
public:
    explicit ImmutablePolicyError(const std::string& msg);
    ImmutablePolicyError(const ImmutablePolicyError& src);
    virtual ~ImmutablePolicyError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: Application specified a set of policies that are not
 * consistent with each other.
 */
class OMG_DDS_API InconsistentPolicyError : public Exception, public std::logic_error
{
/** @cond */
public:
    explicit InconsistentPolicyError(const std::string& msg);
    InconsistentPolicyError(const InconsistentPolicyError& src);
    virtual ~InconsistentPolicyError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: Application is passing an invalid argument.
 */
class OMG_DDS_API InvalidArgumentError : public Exception, public std::invalid_argument
{
/** @cond */
public:
    explicit InvalidArgumentError(const std::string& msg);
    InvalidArgumentError(const InvalidArgumentError& src);
    virtual ~InvalidArgumentError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: Operation invoked on an Entity that is not yet enabled.
 */
class OMG_DDS_API NotEnabledError : public Exception, public std::logic_error
{
/** @cond */
public:
    explicit NotEnabledError(const std::string& msg);
    NotEnabledError(const NotEnabledError& src);
    virtual ~NotEnabledError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: Service ran out of the resources needed to complete the
 * operation.
 */
class OMG_DDS_API OutOfResourcesError : public Exception, public std::runtime_error
{
/** @cond */
public:
    explicit OutOfResourcesError(const std::string& msg);
    OutOfResourcesError(const OutOfResourcesError& src);
    virtual ~OutOfResourcesError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};


/**
 * @brief
 * %Exception: A pre-condition for the operation was not met.
 */
class OMG_DDS_API PreconditionNotMetError : public Exception, public std::logic_error
{
/** @cond */
public:
    explicit PreconditionNotMetError(const std::string& msg);
    PreconditionNotMetError(const PreconditionNotMetError& src);
    virtual ~PreconditionNotMetError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: The operation timed out.
 */
class OMG_DDS_API TimeoutError : public Exception, public std::runtime_error
{
/** @cond */
public:
    explicit TimeoutError(const std::string& msg);
    TimeoutError(const TimeoutError& src);
    virtual ~TimeoutError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: Unsupported operation.
 *
 * This can only be thrown by operations that are optional.
 */
class OMG_DDS_API UnsupportedError : public Exception, public std::logic_error
{
/** @cond */
public:
    explicit UnsupportedError(const std::string& msg);
    UnsupportedError(const UnsupportedError& src);
    virtual ~UnsupportedError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: Application has attempted to cast incompatible types.
 */
class OMG_DDS_API InvalidDowncastError : public Exception, public std::runtime_error
{
/** @cond */
public:
    explicit InvalidDowncastError(const std::string& msg);
    InvalidDowncastError(const InvalidDowncastError& src);
    virtual ~InvalidDowncastError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: Application used a null reference.
 *
 * Very likely, the used DDS object is a dds::core::null object.
 * @code{.cpp}
 * dds::domain::DomainParticipant participant = dds::core::null;
 * try {
 *     participant.domain_id();
 * } catch (const dds::core::NullReferenceError& e) {
 *     std::cout << e.what() << std::endl;
 * }
 * @endcode
 */
class OMG_DDS_API NullReferenceError : public Exception, public std::runtime_error
{
/** @cond */
public:
    explicit NullReferenceError(const std::string& msg);
    NullReferenceError(const NullReferenceError& src);
    virtual ~NullReferenceError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

/**
 * @brief
 * %Exception: Application provided invalid data
 */
class OMG_DDS_API InvalidDataError : public Exception, public std::logic_error
{
/** @cond */
public:
    explicit InvalidDataError(const std::string& msg);
    InvalidDataError(const InvalidDataError& src);
    virtual ~InvalidDataError() throw();

public:
    virtual const char* what() const throw();
/** @endcond */
};

}
}

#if defined _MSC_VER
#   pragma warning (pop)
#endif


#endif /* OMG_DDS_CORE_EXCEPTION_HPP_ */
