#pragma once
#include "data_type.hpp"
#include "math_v2.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct PassThrough
{
    __host__ __device__ void operator()(float& y, const float& x) const { y = x; }

    __host__ __device__ void operator()(half_t& y, const half_t& x) const { y = x; }

    __host__ __device__ void operator()(bhalf_t& y, const bhalf_t& x) const { y = x; }

    __host__ __device__ void operator()(int32_t& y, const int32_t& x) const { y = x; }

    __host__ __device__ void operator()(int8_t& y, const int8_t& x) const { y = x; }

    __host__ __device__ void operator()(double& y, const double& x) const { y = x; }
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

struct Normalize
{
    Normalize(float epsilon = 1e-4) : epsilon_(epsilon) {}

    __host__ __device__ constexpr void operator()(float& y,
                                                  const float& x,
                                                  const float& mean,
                                                  const float& mean_square,
                                                  const float& gamma,
                                                  const float& beta) const
    {
        float variance = mean_square - (mean * mean);
        y              = ((x - mean) / sqrtf(variance + epsilon_)) * gamma + beta;
    }

    float epsilon_;
};

// Unary operators are usually called element-wisely before/after the reduction is executed on the
// elements. They are needed for easy implementation of reduction types of AVG, NRM1, NRM2

template <typename Y, typename X, bool HasDividing = false>
struct UnaryIdentic;

template <>
struct UnaryIdentic<float, float, false>
{
    __host__ __device__ UnaryIdentic(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(float& y, const float& x) const { y = x; };
};

template <>
struct UnaryIdentic<float, float, true>
{
    __host__ __device__ UnaryIdentic(const int32_t divider = 1) { divider_ = divider; };

    __host__ __device__ void operator()(float& y, const float& x) const
    {
        y = x / type_convert<float>(divider_);
    };

    int32_t divider_ = 1;
};

template <>
struct UnaryIdentic<half_t, half_t, false>
{
    __host__ __device__ UnaryIdentic(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(half_t& y, const half_t& x) const { y = x; };
};

template <>
struct UnaryIdentic<double, double, false>
{
    __host__ __device__ UnaryIdentic(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(double& y, const double& x) const { y = x; };
};

template <>
struct UnaryIdentic<double, double, true>
{
    __host__ __device__ UnaryIdentic(const int32_t divider = 1) { divider_ = divider; };

    __host__ __device__ void operator()(double& y, const double& x) const
    {
        y = x / type_convert<double>(divider_);
    };

    int32_t divider_ = 1;
};

template <>
struct UnaryIdentic<int32_t, int32_t, false>
{
    __host__ __device__ UnaryIdentic(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(int32_t& y, const int32_t& x) const { y = x; };
};

template <>
struct UnaryIdentic<int32_t, int32_t, true>
{
    __host__ __device__ UnaryIdentic(const int32_t divider = 1) { divider_ = divider; };

    __host__ __device__ void operator()(int32_t& y, const int32_t& x) const { y = x / divider_; };

    int32_t divider_ = 1;
};

template <>
struct UnaryIdentic<int8_t, int8_t, false>
{
    __host__ __device__ UnaryIdentic(const int8_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(int8_t& y, const int8_t& x) const { y = x; };
};

template <typename Y, typename X, bool HasDividing = false>
struct UnarySquare;

template <>
struct UnarySquare<float, float, false>
{
    __host__ __device__ UnarySquare(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(float& y, const float& x) const { y = x * x; };
};

template <>
struct UnarySquare<float, float, true>
{
    __host__ __device__ UnarySquare(const int32_t divider = 1) { divider_ = divider; };

    __host__ __device__ void operator()(float& y, const float& x) const
    {
        y = x * x / type_convert<float>(divider_);
    };

    int32_t divider_ = 1;
};

template <>
struct UnarySquare<double, double, false>
{
    __host__ __device__ UnarySquare(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(double& y, const double& x) const { y = x * x; };
};

template <>
struct UnarySquare<double, double, true>
{
    __host__ __device__ UnarySquare(const int32_t divider = 1) { divider_ = divider; };

    __host__ __device__ void operator()(double& y, const double& x) const
    {
        y = x * x / type_convert<double>(divider_);
    };

    int32_t divider_ = 1;
};

template <typename Y, typename X>
struct UnaryAbs;

template <>
struct UnaryAbs<float, float>
{
    __host__ __device__ UnaryAbs(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(float& y, const float& x) const { y = ck::math::abs(x); };
};

template <>
struct UnaryAbs<half_t, half_t>
{
    __host__ __device__ UnaryAbs(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(half_t& y, const half_t& x) const { y = ck::math::abs(x); };
};

template <>
struct UnaryAbs<double, double>
{
    __host__ __device__ UnaryAbs(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(double& y, const double& x) const { y = ck::math::abs(x); };
};

template <>
struct UnaryAbs<int8_t, int8_t>
{
    __host__ __device__ UnaryAbs(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(int8_t& y, const int8_t& x) const { y = ck::math::abs(x); };
};

template <typename Y, typename X>
struct UnarySqrt;

template <>
struct UnarySqrt<float, float>
{
    __host__ __device__ UnarySqrt(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(float& y, const float& x) const { y = ck::math::sqrt(x); };
};

template <>
struct UnarySqrt<double, double>
{
    __host__ __device__ UnarySqrt(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(double& y, const double& x) const
    {
        y = ck::math::sqrt(x);
    };
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
