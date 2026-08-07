#ifndef __UT_EXCEPTION_HPP__
#define __UT_EXCEPTION_HPP__

#include <unitree/common/error.hpp>

#define UT_MAX_TRACE_ADDR_NUMBER 64

namespace unitree
{
namespace common
{
class Exception : public std::exception
{
public:
    Exception() throw()
        : Exception(UT_ERR_UNKNOWN, UT_DESC_ERR(UT_ERR_UNKNOWN))
    {}

    Exception(int32_t code, const std::string& message) throw()
        : mCode(code), mMessage(message), mLine(0)
    {}

    virtual ~Exception()
    {}

    int32_t GetCode() const
    {
        return mCode;
    }

    const std::string& GetMessage() const
    {
        return mMessage;
    }

    virtual const char* what() const noexcept
    {
        return mMessage.c_str();
    }

    virtual std::string GetClassName() const
    {
        return "Exception";
    }

    void Init(const char* file, const char* func, int32_t line)
    {
        mFile = file;
        mFunc = func;
        mLine = line;
    }

    std::string ToString() const
    {
        std::ostringstream os;
        AddDetail(os);

        return os.str();
    }

    std::string StackTrace() const
    {
        std::ostringstream os;
        AddDetail(os);
        AddBackTrace(os);

        return os.str();
    }

protected:
    void AddDetail(std::ostringstream& os) const
    {
        os << "Catch " << GetClassName() << " code:" << mCode
            << ", message:" << mMessage << std::endl;

        if (!mFile.empty() && !mFunc.empty() && mLine > 0)
        {
            os << " at __FILE__:" << mFile << ", __LINE__:"
                << mLine << ", __FUNCTION__:" << mFunc << std::endl;
        }
    }

    void AddBackTrace(std::ostringstream& os) const
    {
        os << "Stack:" << std::endl;

        void* addr[UT_MAX_TRACE_ADDR_NUMBER];
        int32_t number = backtrace(addr, UT_MAX_TRACE_ADDR_NUMBER);

        char **info = backtrace_symbols(addr, number);
        if(info == NULL)
        {
            return;
        }

        for(int32_t i=0; i<number; i++)
        {
            os << info[i] << std::endl;
        }

        free(info);
    }

protected:
    int32_t mCode;
    std::string mMessage;

    std::string mFile;
    std::string mFunc;
    int32_t mLine;

    std::string mWhat;
};

#define UT_THROW_0(ExceptionType)           \
    do                                      \
    {                                       \
        ExceptionType __temp_except_r38e2d; \
        __temp_except_r38e2d.Init(__FILE__,__PRETTY_FUNCTION__,__LINE__); \
        throw(__temp_except_r38e2d);        \
    } while(0);

#define UT_THROW(ExceptionType, args...)    \
    do                                      \
    {                                       \
        ExceptionType __temp_except_r38e2d(args);       \
        __temp_except_r38e2d.Init(__FILE__,__PRETTY_FUNCTION__,__LINE__); \
        throw(__temp_except_r38e2d);        \
    } while(0);

#define UT_THROW_IF(condition, ExceptionType, args...)  \
    if (condition)                          \
    {                                       \
        UT_THROW(ExceptionType, args);      \
    }

#define UT_EXCEPTION_TRY                    \
    try                                     \
    {

#define __UT_EXCEPTION_CATCH(except, l, t)  \
    catch (const except& e)                 \
    {                                       \
        if (l)                              \
        {                                   \
            LOG_ERROR(l, e.what());         \
        }                                   \
        if (t)                              \
        {                                   \
            throw e;                        \
        }                                   \
    }

#define UT_EXCEPTION_CATCH(l, t)            \
    }                                       \
    __UT_EXCEPTION_CATCH(unitree::common::Exception, l, t)      \
    __UT_EXCEPTION_CATCH(std::exception, l, t)

#define UT_DECL_EXCEPTION(ExceptionType, code, desc)    \
class ExceptionType : public unitree::common::Exception \
{                                           \
public:                                     \
    ExceptionType() throw()                 \
        : Exception(code, desc)             \
    {}                                      \
                                            \
    ExceptionType(const std::string& message) throw()   \
        : Exception(code, message)          \
    {}                                      \
                                            \
    std::string GetClassName() const        \
    {                                       \
        return #ExceptionType;              \
    }                                       \
};

UT_DECL_EXCEPTION(CommonException, UT_ERR_COMMON, UT_DESC_ERR(UT_ERR_COMMON))

UT_DECL_EXCEPTION(SystemException, UT_ERR_SYSTEM, UT_DESC_ERR(UT_ERR_SYSTEM))

UT_DECL_EXCEPTION(NetworkException, UT_ERR_NETWORK, UT_DESC_ERR(UT_ERR_NETWORK))

UT_DECL_EXCEPTION(FileException, UT_ERR_FILE, UT_DESC_ERR(UT_ERR_FILE))

UT_DECL_EXCEPTION(SocketException, UT_ERR_SOCKET, UT_DESC_ERR(UT_ERR_SOCKET))

UT_DECL_EXCEPTION(IOException, UT_ERR_IO, UT_DESC_ERR(UT_ERR_IO))

UT_DECL_EXCEPTION(LockException, UT_ERR_LOCK, UT_DESC_ERR(UT_ERR_LOCK))

UT_DECL_EXCEPTION(TimeoutException, UT_ERR_TIMEOUT, UT_DESC_ERR(UT_ERR_TIMEOUT))

UT_DECL_EXCEPTION(BadCastException, UT_ERR_BADCAST, UT_DESC_ERR(UT_ERR_BADCAST))

UT_DECL_EXCEPTION(JsonException, UT_ERR_JSON, UT_DESC_ERR(UT_ERR_JSON))

UT_DECL_EXCEPTION(FutureException, UT_ERR_FUTURE, UT_DESC_ERR(UT_ERR_FUTURE))

UT_DECL_EXCEPTION(FutureFaultException, UT_ERR_FUTURE_FAULT, UT_DESC_ERR(UT_ERR_FUTURE_FAULT))

}
}
#endif//__UT_EXCEPTION_HPP__
