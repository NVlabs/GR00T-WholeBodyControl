#ifndef __UT_THREAD_DECL_HPP__
#define __UT_THREAD_DECL_HPP__

#include <unitree/common/any.hpp>
#include <unitree/common/exception.hpp>
#include <unitree/common/lock/lock.hpp>
#include <unitree/common/time/time_tool.hpp>

#define __UT_THREAD_DECL_TMPL_FUNC_ARG__    \
    template<class Func, class... Args>

#define __UT_THREAD_TMPL_FUNC_ARG__         \
    Func&& func, Args&&... args

#define __UT_THREAD_BIND_FUNC_ARG__         \
    std::forward<Func>(func), std::forward<Args>(args)...

#define UT_CPU_ID_NONE  -1

#endif//__THREAD_DECL_HPP__
