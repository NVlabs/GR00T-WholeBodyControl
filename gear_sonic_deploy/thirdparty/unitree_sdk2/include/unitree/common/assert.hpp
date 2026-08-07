#ifndef __UT_ASSERT_HPP__
#define __UT_ASSERT_HPP__

#include <unitree/common/decl.hpp>

/*
 * Declare assert output
 */
#define UT_ASSERT_OUT(debug, file, func, line, r)           \
    if (debug)                                              \
    {                                                       \
        std::cout << "[" << ::time(NULL)                    \
            << "] [" << ::syscall(SYS_gettid)               \
            << "] UT_ASSERT DEBUG at __FILE__:" << file     \
            << ", __FUNCTION__:" << func                    \
            << ", __LINE__:" << line                        \
            << ", r:" << r                                  \
            << ", errno:" << errno                          \
            << std::endl;                                   \
    }                                                       \
    else                                                    \
    {                                                       \
        std::cout << "[" << ::time(NULL)                    \
            << "] [" << ::syscall(SYS_gettid)               \
            << "] UT_ASSERT ABORT at __FILE__:" << file     \
            << ", __FUNCTION__:" << func                    \
            << ", __LINE__:" << line                        \
            << ", r:" << r                                  \
            << ", errno:" << errno                          \
            << std::endl;                                   \
    }

#define UT_ASSERT_ABORT(debug, file, func, line, r)         \
    if (debug)                                              \
    {                                                       \
        UT_ASSERT_OUT(1, file, func, line, r);              \
    }                                                       \
    else                                                    \
    {                                                       \
        UT_ASSERT_OUT(0, file, func, line, r);              \
        abort();                                            \
    }

/*
 * Declare assert return value
 */
#define UT_ASSERT_EQ(x, r)                                  \
    unitree::common::AssertEqual(x, r, 0, __FILE__,                 \
        __PRETTY_FUNCTION__, __LINE__)

#define UT_ASSERT_EQ_DEBUG(x, r)                            \
    unitree::common::AssertEqual(x, r, 1, __FILE__,                 \
        __PRETTY_FUNCTION__, __LINE__)

#define UT_ASSERT_NOT_EQ(x, r)                              \
    unitree::common::AssertNotEqual(x, r, 0, __FILE__,              \
        __PRETTY_FUNCTION__, __LINE__)

#define UT_ASSERT_NOT_EQ_DEBUG(x, r)                        \
    unitree::common::AssertNotEqual(x, r, 1, __FILE__,              \
        __PRETTY_FUNCTION__, __LINE__)

/*
 * Declare assert return value and errno
 */
#define UT_ASSERT_ENO_EQ(x, r, eno)                         \
    unitree::common::AssertEqual(x, r, eno, 0, __FILE__,            \
        __PRETTY_FUNCTION__, __LINE__)

#define UT_ASSERT_ENO_EQ_DEBUG(x, r, eno)                   \
    unitree::common::AssertEqual(x, r, eno, 1, __FILE__,            \
        __PRETTY_FUNCTION__, __LINE__)

#define UT_ASSERT_ENO_EQ_EX(x, r, eno)                      \
    unitree::common::AssertEqualEx(x, r, eno, 0, __FILE__,          \
        __PRETTY_FUNCTION__, __LINE__)

#define UT_ASSERT_ENO_EQ_EX_DEBUG(x, r, eno)                \
    unitree::common::AssertEqualEx(x, r, eno, 1, __FILE__,          \
        __PRETTY_FUNCTION__, __LINE__)

/*
 * Declare assert wrapper
 */
#define UT_ASSERT_0(x)                  \
    UT_ASSERT_EQ(x, 0)

#define UT_ASSERT_DEBUG_0(x)            \
    UT_ASSERT_EQ_DEBUG(x, 0)

#define UT_ASSERT_ENO_0(x, eno)         \
    UT_ASSERT_ENO_EQ(x, 0, eno)

#define UT_ASSERT_ENO_DEBUG_0(x, eno)   \
    UT_ASSERT_ENO_EQ_DEBUG(x, 0, eno)


//Declare assert function
namespace unitree
{
namespace common
{
inline int AssertEqual(int r, int expectRet, bool debug,
    const char* file, const char* func, int line)
{
    if (UT_UNLIKELY(r != expectRet))
    {
        UT_ASSERT_ABORT(debug, file, func, line, r);
    }

    return r;
}

inline int AssertNotEqual(int r, int expectRet, bool debug,
    const char* file, const char* func, int line)
{
    if (UT_UNLIKELY(r == expectRet))
    {
        UT_ASSERT_ABORT(debug, file, func, line, r);
    }

    return r;
}

inline int AssertEqual(int r, int expectRet, int expectErrno, bool debug,
    const char* file, const char* func, int line)
{
    if (UT_UNLIKELY(r != expectRet) && UT_UNLIKELY(errno != expectErrno))
    {
        UT_ASSERT_ABORT(debug, file, func, line, r);
    }

    return r;
}

inline int AssertEqualEx(int r, int expectRet, int expectErrno, bool debug,
    const char* file, const char* func, int line)
{
    if (UT_UNLIKELY(r != 0) && UT_UNLIKELY(r != expectRet) && 
        UT_UNLIKELY(errno != expectErrno))
    {
        UT_ASSERT_ABORT(debug, file, func, line, r);
    }

    return r; 
}
}
}
#endif//__UT_ASSERT_HPP__
