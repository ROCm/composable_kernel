#pragma once
#include "data_type.hpp"

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

struct RequantReluRequant
{
    // FIXME: We just need one scale for Relu / Leaky Relu / PRelu
    RequantReluRequant(float scaleGemm, float scaleRelu)
        : scaleGemm_(scaleGemm), scaleRelu_(scaleRelu)
    {
    }

    __host__ __device__ constexpr void operator()(int8_t& y, const int& x) const
    {
        float gemm_requant = scaleGemm_ * static_cast<float>(x);
        float relu         = gemm_requant > 0 ? gemm_requant : 0;
        float relu_requant = scaleRelu_ * relu;
        y                  = static_cast<int8_t>(relu_requant > 127 ? 127
                                                   : relu_requant < -128 ? -128 : relu_requant);
    }

    // for reference_gemm
    __host__ __device__ constexpr void operator()(float& y, const float& x) const
    {
        float gemm_requant = scaleGemm_ * x;
        float relu         = gemm_requant > 0 ? gemm_requant : 0;
        float relu_requant = scaleRelu_ * relu;
        y                  = static_cast<float>(relu_requant > 127 ? 127
                                                  : relu_requant < -128 ? -128 : relu_requant);
    }

    float scaleGemm_;
    float scaleRelu_;
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

    __host__ __device__ void operator()(float& y, const float& x) const { y = abs(x); };
};

template <>
struct UnaryAbs<half_t, half_t>
{
    __host__ __device__ UnaryAbs(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(half_t& y, const half_t& x) const { y = __habs(x); };
};

template <>
struct UnaryAbs<double, double>
{
    __host__ __device__ UnaryAbs(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(double& y, const double& x) const { y = abs(x); };
};

template <>
struct UnaryAbs<int8_t, int8_t>
{
    __host__ __device__ UnaryAbs(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(int8_t& y, const int8_t& x) const
    {
        int8_t sgn = x >> (8 - 1);

        y = (x ^ sgn) - sgn;
    };
};

template <typename Y, typename X>
struct UnarySqrt;

template <>
struct UnarySqrt<float, float>
{
    __host__ __device__ UnarySqrt(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(float& y, const float& x) const { y = sqrtf(x); };
};

template <>
struct UnarySqrt<double, double>
{
    __host__ __device__ UnarySqrt(const int32_t divider = 1) { (void)divider; };

    __host__ __device__ void operator()(double& y, const double& x) const { y = sqrt(x); };
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
