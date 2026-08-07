#ifndef __UT_DDS_NATIVE_HPP__
#define __UT_DDS_NATIVE_HPP__

namespace unitree
{
namespace common
{
template<typename NATIVE>
class DdsNative
{
public:
    using NativeType = NATIVE;

    explicit DdsNative()
    {}

    virtual ~DdsNative()
    {}

    void SetNative(const NATIVE& native)
    {
        mNative = native;
    }

    const NATIVE& GetNative() const
    {
        return mNative;
    }

protected:
    NATIVE mNative;
};

}
}

#endif//
