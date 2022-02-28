#ifndef CK_ELEMENT_WISE_OPERATION_HPP
#define CK_ELEMENT_WISE_OPERATION_HPP

#include "data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct PassThrough
{
    __host__ __device__ void operator()(float& y, const float& x) const { y = x; }

    __host__ __device__ void operator()(half_t& y, const half_t& x) const { y = x; }

    __host__ __device__ void operator()(ushort& y, const ushort& x) const { y = x; }

    __host__ __device__ void operator()(int32_t& y, const int32_t& x) const { y = x; }

    __host__ __device__ void operator()(int8_t& y, const int8_t& x) const { y = x; }
};

struct Add
{
    __host__ __device__ constexpr void operator()(float& y, const float& x0, const float& x1) const
    {
        y = x0 + x1;
    }

    __host__ __device__ constexpr void
    operator()(half_t& y, const half_t& x0, const half_t& x1) const
    {
        // FIXME - Use float (acc type) bias in the future.
        y = x0 + x1;
    }
};

struct AlphaBetaAdd
{
    AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta) {}

    __host__ __device__ constexpr void operator()(float& y, const float& x0, const float& x1) const
    {
        y = alpha_ * x0 + beta_ * x1;
    }

    __host__ __device__ constexpr void
    operator()(half_t& y, const half_t& x0, const half_t& x1) const
    {
        // FIXME - Let x0 be acc type
        y = static_cast<half_t>(alpha_ * static_cast<float>(x0) + beta_ * static_cast<float>(x1));
    }

    float alpha_;
    float beta_;
};

struct AddRelu
{
    __host__ __device__ constexpr void operator()(float& y, const float& x0, const float& x1) const
    {
        const float a = x0 + x1;
        y             = a > 0 ? a : 0;
    }

    __host__ __device__ constexpr void
    operator()(half_t& y, const half_t& x0, const half_t& x1) const
    {
        const half_t a = x0 + x1;
        y              = a > 0 ? a : 0;
    }
};

struct AddHardswish
{
    __host__ __device__ constexpr void operator()(float& y, const float& x0, const float& x1) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        y       = c;
    }

    __host__ __device__ constexpr void
    operator()(half_t& y, const half_t& x0, const half_t& x1) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        y       = c;
    }
};

struct AddReluAdd
{
    __host__ __device__ constexpr void
    operator()(half_t& y, const half_t& x0, const half_t& x1, const half_t& x2) const
    {
        half_t a = x0 + x1;
        half_t b = a > 0 ? a : 0;
        y        = b + x2;
    }

    __host__ __device__ constexpr void
    operator()(float& y, const float& x0, const float& x1, const float& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }

    __host__ __device__ constexpr void
    operator()(half_t& y, const float& x0, const half_t& x1, const half_t& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }
};

struct AddHardswishAdd
{
    __host__ __device__ constexpr void
    operator()(float& y, const float& x0, const float& x1, const float& x2) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
    }

    __host__ __device__ constexpr void
    operator()(half_t& y, const half_t& x0, const half_t& x1, const half_t& x2) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > float{6} ? float{6} : b) * a * float{0.166667};
        float d = c + x2;
        y       = d;
    }
};

// Unary operators are usually called element-wisely before the reduction is executed on the
// elements.
// They are needed for easy implementation of reduction types of AVG, NRM1, NRM2

template <typename Y, typename X, bool HasDividing = false>
struct UnaryIdentic
{
    __host__ __device__ UnaryIdentic(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ constexpr void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
    };
};

template <typename Y, typename X>
struct UnaryIdentic<Y, X, true>
{
    __host__ __device__ UnaryIdentic(const int32_t divider = 1) { divider_ = divider; };

    __host__ __device__ constexpr void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x) / type_convert<Y>(divider_);
    };

    int32_t divider_ = 1;
};

template <typename Y, typename X, bool HasDividing = false>
struct UnarySquare
{
    __host__ __device__ UnarySquare(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ constexpr void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
        y = y * y;
    };
};

template <typename Y, typename X>
struct UnarySquare<Y, X, true>
{
    __host__ __device__ UnarySquare(const int32_t divider = 1) { divider_ = divider; };

    __host__ __device__ constexpr void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
        y = y * y;
        y = y / type_convert<Y>(divider_);
    };

    int32_t divider_ = 1;
};

static inline __device__ half_t abs(half_t x) { return __habs(x); };
static inline __device__ half_t sqrtf(half_t x) { return hsqrt(x); };

template <typename Y, typename X, bool HasDividing = false>
struct UnaryAbs
{
    __host__ __device__ UnaryAbs(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ constexpr void operator()(Y& y, const X& x) const
    {
        y = abs(type_convert<Y>(x));
    };
};

template <typename Y, typename X>
struct UnaryAbs<Y, X, true>
{
    __host__ __device__ UnaryAbs(const int32_t divider = 1) { divider_ = divider; };

    __host__ __device__ constexpr void operator()(Y& y, const X& x) const
    {
        y = abs(type_convert<Y>(x) / type_convert<Y>(divider_));
    };

    int32_t divider_ = 1;
};

template <typename Y, typename X>
struct UnarySqrt
{
    __host__ __device__ UnarySqrt(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(Y& y, const X& x) const { y = sqrtf(type_convert<Y>(x)); };
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
#endif
