#ifndef __UT_ANY_HPP__
#define __UT_ANY_HPP__

#include <unitree/common/exception.hpp>

namespace unitree
{
namespace common
{
class Any
{
public:
    Any()
        : mContent(0)
    {}

    template<typename ValueType>
    Any(const ValueType& value)
        : mContent(new Holder<ValueType>(value))
    {}

    Any(const char* s)
        : Any(std::string(s))
    {}

    Any(const char* s, size_t len)
        : Any(std::string(s, len))
    {}

    Any(const Any& other)
        : mContent(other.mContent ? other.mContent->Clone() : 0)
    {}

    ~Any()
    {
        delete mContent;
        mContent = 0;
    }

    Any& Swap(Any& other)
    {
        std::swap(mContent, other.mContent);
        return *this;
    }

    bool Empty() const
    {
        return mContent == 0;
    }

    const std::type_info& GetTypeInfo() const
    {
        return mContent ? mContent->GetTypeInfo() : typeid(void);
    }

    template<typename ValueType>
    Any& operator=(const ValueType& other)
    {
        Any(other).Swap(*this);
        return *this;
    }

    Any& operator=(Any other)
    {
        other.Swap(*this);
        return *this;
    }

public:
    class PlaceHolder
    {
    public:
        virtual ~PlaceHolder()
        {}

    public:
        virtual const std::type_info& GetTypeInfo() const = 0;
        virtual PlaceHolder* Clone() const = 0;
    };

    template<typename ValueType>
    class Holder : public PlaceHolder
    {
    public:
        explicit Holder(const ValueType& value)
            : mValue(value)
        {}

        virtual const std::type_info& GetTypeInfo() const
        {
            return typeid(ValueType);
        }

        virtual PlaceHolder* Clone() const
        {
            return new Holder(mValue);
        }

    public:
        ValueType mValue;
    };

public:
    PlaceHolder* mContent;
};

/*
 * static const Any
 */
static const Any UT_EMPTY_ANY = Any();

static inline bool IsBool(const Any& any)
{
    return any.GetTypeInfo() == typeid(bool);
}

static inline bool IsString(const Any& any)
{
    return any.GetTypeInfo() == typeid(std::string);
}

static inline bool IsInt8(const Any& any)
{
    return any.GetTypeInfo() == typeid(int8_t);
}

static inline bool IsUint8(const Any& any)
{
    return any.GetTypeInfo() == typeid(uint8_t);
}

static inline bool IsInt16(const Any& any)
{
    return any.GetTypeInfo() == typeid(int16_t);
}

static inline bool IsUint16(const Any& any)
{
    return any.GetTypeInfo() == typeid(uint16_t);
}

static inline bool IsInt(const Any& any)
{
    return any.GetTypeInfo() == typeid(int32_t);
}

static inline bool IsUint(const Any& any)
{
    return any.GetTypeInfo() == typeid(uint32_t);
}

static inline bool IsInt64(const Any& any)
{
    return any.GetTypeInfo() == typeid(int64_t);
}

static inline bool IsUint64(const Any& any)
{
    return any.GetTypeInfo() == typeid(uint64_t);
}

static inline bool IsFloat(const Any& any)
{
    return any.GetTypeInfo() == typeid(float);
}

static inline bool IsDouble(const Any& any)
{
    return any.GetTypeInfo() == typeid(double);
}

static inline bool IsLongDouble(const Any& any)
{
    return any.GetTypeInfo() == typeid(long double);
}

static inline bool IsInteger(const Any& any)
{
    return IsInt(any) || IsUint(any) || IsInt64(any) || IsUint64(any)
        || IsInt16(any) || IsUint16(any) || IsInt8(any) || IsUint8(any);
}

static inline bool IsNumber(const Any& any)
{
    return IsBool(any) || IsInteger(any) || IsFloat(any) || IsDouble(any)
        || IsLongDouble(any);
}

static inline bool IsBoolType(const std::type_info& t)
{
    return t == typeid(bool);
}

static inline bool IsInt8Type(const std::type_info& t)
{
    return t == typeid(int8_t);
}

static inline bool IsUint8Type(const std::type_info& t)
{
    return t == typeid(uint8_t);
}

static inline bool IsInt16Type(const std::type_info& t)
{
    return t == typeid(int16_t);
}

static inline bool IsUint16Type(const std::type_info& t)
{
    return t == typeid(uint16_t);
}

static inline bool IsIntType(const std::type_info& t)
{
    return t == typeid(int32_t);
}

static inline bool IsUintType(const std::type_info& t)
{
    return t == typeid(uint32_t);
}

static inline bool IsInt64Type(const std::type_info& t)
{
    return t == typeid(int64_t);
}

static inline bool IsUint64Type(const std::type_info& t)
{
    return t == typeid(uint64_t);
}

static inline bool IsIntegerType(const std::type_info& t)
{
    return IsIntType(t) || IsUintType(t) || IsInt64Type(t) || IsUint64Type(t) ||
           IsInt8Type(t) || IsUint8Type(t) || IsInt16Type(t) || IsUint16Type(t);
}

static inline bool IsFloatType(const std::type_info& t)
{
    return t == typeid(float);
}

static inline bool IsDoubleType(const std::type_info& t)
{
    return t == typeid(double);
}

static inline bool IsLongDoubleType(const std::type_info& t)
{
    return t == typeid(long double);
}

static inline bool IsNumberType(const std::type_info& t)
{
    return IsBoolType(t) || IsIntegerType(t) || IsFloatType(t)
        || IsDoubleType(t) || IsLongDoubleType(t);
}

static inline bool IsTypeEqual(const std::type_info& t1, const std::type_info& t2)
{
    return t1 == t2;
}

template<typename ValueType>
const ValueType& AnyCast(const Any* operand)
{
    const std::type_info& t1 = typeid(ValueType);
    const std::type_info& t2 = operand->GetTypeInfo();

    if (IsTypeEqual(t1, t2))
    {
        return ((Any::Holder<ValueType>*)(operand->mContent))->mValue;
    }

    UT_THROW(BadCastException, std::string("AnyCast error. target type is ")
        + t1.name() + ", but source type is " + t2.name());
}

template<typename ValueType>
const ValueType& AnyCast(const Any& operand)
{
    return AnyCast<ValueType>(&operand);
}

template<typename ValueType>
ValueType AnyNumberCast(const Any* operand)
{
    const std::type_info& t1 = typeid(ValueType);
    const std::type_info& t2 = operand->GetTypeInfo();

    if (IsNumberType(t1) && IsNumberType(t2))
    {
        if (IsTypeEqual(t1, t2))
        {
            return ((Any::Holder<ValueType>*)(operand->mContent))->mValue;
        }
        else if (IsFloatType(t2))
        {
            return (ValueType)((Any::Holder<float>*)(operand->mContent))->mValue;
        }
        else if (IsDoubleType(t2))
        {
            return (ValueType)((Any::Holder<double>*)(operand->mContent))->mValue;
        }
        else if (IsLongDoubleType(t2))
        {
            return (ValueType)((Any::Holder<long double>*)(operand->mContent))->mValue;
        }
        else if (IsInt8Type(t2))
        {
            return (ValueType)((Any::Holder<int8_t>*)(operand->mContent))->mValue;
        }
        else if (IsUint8Type(t2))
        {
            return (ValueType)((Any::Holder<uint8_t>*)(operand->mContent))->mValue;
        }
        else if (IsInt16Type(t2))
        {
            return (ValueType)((Any::Holder<int16_t>*)(operand->mContent))->mValue;
        }
        else if (IsUint16Type(t2))
        {
            return (ValueType)((Any::Holder<uint16_t>*)(operand->mContent))->mValue;
        }
        else if (IsIntType(t2))
        {
            return (ValueType)((Any::Holder<int32_t>*)(operand->mContent))->mValue;
        }
        else if (IsUintType(t2))
        {
            return (ValueType)((Any::Holder<uint32_t>*)(operand->mContent))->mValue;
        }
        else if (IsInt64Type(t2))
        {
            return (ValueType)((Any::Holder<int64_t>*)(operand->mContent))->mValue;
        }
        else if (IsUint64Type(t2))
        {
            return (ValueType)((Any::Holder<uint64_t>*)(operand->mContent))->mValue;
        }
        else
        {
            UT_THROW(BadCastException, std::string("AnyNumberCast error. unknown number type:", t2.name()));
        }
    }

    UT_THROW(BadCastException, std::string("AnyNumberCast error. not number type"));
}

template<typename ValueType>
ValueType AnyNumberCast(const Any& operand)
{
    return AnyNumberCast<ValueType>(&operand);
}

static inline const std::string& ToString(const Any& operand)
{
    if (operand.Empty())
    {
        return UT_EMPTY_STR;
    }

    return AnyCast<std::string>(operand);
}

static inline void StringTo(const std::string& s, Any& value)
{
    value = s;
}

static inline void StringTo(const char* s, Any& value)
{
    value = std::string(s);
}

static inline void StringTo(const char* s, size_t len, Any& value)
{
    value = std::string(s, len);
}

static inline void StringTo(const char* s, size_t pos, size_t len, Any& value)
{
    value = std::string(s, pos, len);
}

}
}
#endif//__UT_ANY_HPP__
