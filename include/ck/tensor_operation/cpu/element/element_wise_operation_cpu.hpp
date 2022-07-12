#pragma once
#include "ck/utility/data_type_cpu.hpp"

namespace ck {
namespace tensor_operation {
namespace cpu {
namespace element_wise {

using float8_t = ck::cpu::float8_t;
using float4_t = ck::cpu::float4_t;

struct PassThrough
{
    void operator()(float& y, const float& x) const { y = Apply(x); }
    void operator()(float4_t& y, const float4_t& x) const { y = Apply(x); }
    void operator()(float8_t& y, const float8_t& x) const { y = Apply(x); }

    float Apply(const float& x) const { return x; }
    float4_t Apply(const float4_t& x) const { return x; }
    float8_t Apply(const float8_t& x) const { return x; }

    static constexpr char* Name() { return "PassThrough"; }
};

struct Add
{
    void operator()(float& y, const float& x0, const float& x1) const { y = Apply(x0, x1); }

    void operator()(float4_t& y, const float4_t& x0, const float4_t& x1) const
    {
        y = Apply(x0, x1);
    }

    void operator()(float8_t& y, const float8_t& x0, const float8_t& x1) const
    {
        y = Apply(x0, x1);
    }

    float Apply(const float& x0, const float& x1) const { return x0 + x1; }
    float4_t Apply(const float4_t& x0, const float4_t& x1) const { return _mm_add_ps(x0, x1); }
    float8_t Apply(const float8_t& x0, const float8_t& x1) const { return _mm256_add_ps(x0, x1); }

    static constexpr char* Name() { return "Add"; }
};

struct Relu
{
    void operator()(float& y, const float& x) const { y = Apply(x); }
    void operator()(float4_t& y, const float4_t& x) const { y = Apply(x); }
    void operator()(float8_t& y, const float8_t& x) const { y = Apply(x); }

    float Apply(const float& x) const { return x > 0 ? x : 0; }
    float4_t Apply(const float4_t& x) const { return _mm_max_ps(x, _mm_setzero_ps()); }
    float8_t Apply(const float8_t& x) const { return _mm256_max_ps(x, _mm256_setzero_ps()); }

    static constexpr char* Name() { return "Relu"; }
};

struct AlphaBetaAdd
{
    AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta) {}

    void operator()(float& y, const float& x0, const float& x1) const { y = Apply(x0, x1); }

    void operator()(float4_t& y, const float4_t& x0, const float4_t& x1) const
    {
        y = Apply(x0, x1);
    }

    void operator()(float8_t& y, const float8_t& x0, const float8_t& x1) const
    {
        y = Apply(x0, x1);
    }

    float Apply(const float& x0, const float& x1) const { return alpha_ * x0 + beta_ * x1; }

    float4_t Apply(const float4_t& x0, const float4_t& x1) const
    {
        return _mm_add_ps(_mm_mul_ps(x0, _mm_set1_ps(alpha_)), _mm_mul_ps(x1, _mm_set1_ps(beta_)));
    }

    float8_t Apply(const float8_t& x0, const float8_t& x1) const
    {
        return _mm256_add_ps(_mm256_mul_ps(x0, _mm256_set1_ps(alpha_)),
                             _mm256_mul_ps(x1, _mm256_set1_ps(beta_)));
    }

    static constexpr char* Name() { return "AlphaBetaAdd"; }

    float alpha_;
    float beta_;
};

struct AddRelu
{
    void operator()(float& y, const float& x0, const float& x1) const { y = Apply(x0, x1); }

    void operator()(float4_t& y, const float4_t& x0, const float4_t& x1) const
    {
        y = Apply(x0, x1);
    }

    void operator()(float8_t& y, const float8_t& x0, const float8_t& x1) const
    {
        y = Apply(x0, x1);
    }

    float Apply(const float& x0, const float& x1) const
    {
        const float a = x0 + x1;
        return a > 0 ? a : 0;
    }

    float4_t Apply(const float4_t& x0, const float4_t& x1) const
    {
        return _mm_max_ps(_mm_add_ps(x0, x1), _mm_setzero_ps());
    }

    float8_t Apply(const float8_t& x0, const float8_t& x1) const
    {
        return _mm256_max_ps(_mm256_add_ps(x0, x1), _mm256_setzero_ps());
    }

    static constexpr char* Name() { return "AddRelu"; }
};

struct AddReluAdd
{
    void operator()(float& y, const float& x0, const float& x1, const float& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        float c = b + x2;
        y       = c;
    }

    void operator()(float4_t& y, const float4_t& x0, const float4_t& x1, const float4_t& x2) const
    {
        float4_t a = _mm_add_ps(x0, x1);
        float4_t b = _mm_max_ps(a, _mm_setzero_ps());
        y          = _mm_add_ps(b, x2);
    }

    void operator()(float8_t& y, const float8_t& x0, const float8_t& x1, const float8_t& x2) const
    {
        float8_t a = _mm256_add_ps(x0, x1);
        float8_t b = _mm256_max_ps(a, _mm256_setzero_ps());
        y          = _mm256_add_ps(b, x2);
    }

    float Apply(const float& x0, const float& x1, const float& x2) const
    {
        float a = x0 + x1;
        float b = a > 0 ? a : 0;
        return b + x2;
    }

    float4_t Apply(const float4_t& x0, const float4_t& x1, const float4_t& x2) const
    {
        float4_t a = _mm_add_ps(x0, x1);
        float4_t b = _mm_max_ps(a, _mm_setzero_ps());
        return _mm_add_ps(b, x2);
    }

    float8_t Apply(const float8_t& x0, const float8_t& x1, const float8_t& x2) const
    {
        float8_t a = _mm256_add_ps(x0, x1);
        float8_t b = _mm256_max_ps(a, _mm256_setzero_ps());
        return _mm256_add_ps(b, x2);
    }

    static constexpr char* Name() { return "AddReluAdd"; }
};

// Unary operators are usually called element-wisely before/after the reduction is executed on the
// elements. They are needed for easy implementation of reduction types of AVG, NRM1, NRM2

template <typename Y, typename X, bool HasDividing = false>
struct UnaryIdentic;

template <>
struct UnaryIdentic<float, float, false>
{
    UnaryIdentic(const int32_t divider = 1) { (void)divider; };

    void operator()(float& y, const float& x) const { y = x; };
};

template <>
struct UnaryIdentic<float, float, true>
{
    UnaryIdentic(const int32_t divider = 1) { divider_ = divider; };

    void operator()(float& y, const float& x) const { y = x / type_convert<float>(divider_); };

    int32_t divider_ = 1;
};

template <>
struct UnaryIdentic<float4_t, float4_t, false>
{
    UnaryIdentic(const int32_t divider = 1) { (void)divider; };

    void operator()(float4_t& y, const float4_t& x) const { y = x; };
};

template <>
struct UnaryIdentic<float4_t, float4_t, true>
{
    UnaryIdentic(const int32_t divider = 1) { divider_ = divider; };

    void operator()(float4_t& y, const float4_t& x) const
    {
        y = _mm_div_ps(x, _mm_set1_ps(static_cast<float>(divider_)));
    };

    int32_t divider_ = 1;
};

template <>
struct UnaryIdentic<float8_t, float8_t, false>
{
    UnaryIdentic(const int32_t divider = 1) { (void)divider; };

    void operator()(float8_t& y, const float8_t& x) const { y = x; };
};

template <>
struct UnaryIdentic<float8_t, float8_t, true>
{
    UnaryIdentic(const int32_t divider = 1) { divider_ = divider; };

    void operator()(float8_t& y, const float8_t& x) const
    {
        y = _mm256_div_ps(x, _mm256_set1_ps(static_cast<float>(divider_)));
    };

    int32_t divider_ = 1;
};

template <typename Y, typename X, bool HasDividing = false>
struct UnarySquare;

template <>
struct UnarySquare<float, float, false>
{
    UnarySquare(const int32_t divider = 1) { (void)divider; };

    void operator()(float& y, const float& x) const { y = x * x; };
};

template <>
struct UnarySquare<float, float, true>
{
    UnarySquare(const int32_t divider = 1) { divider_ = divider; };

    void operator()(float& y, const float& x) const { y = x * x / type_convert<float>(divider_); };

    int32_t divider_ = 1;
};

template <>
struct UnarySquare<float4_t, float4_t, false>
{
    UnarySquare(const int32_t divider = 1) { (void)divider; };

    void operator()(float4_t& y, const float4_t& x) const { y = _mm_mul_ps(x, x); };
};

template <>
struct UnarySquare<float4_t, float4_t, true>
{
    UnarySquare(const int32_t divider = 1) { divider_ = divider; };

    void operator()(float4_t& y, const float4_t& x) const
    {
        y = _mm_div_ps(_mm_mul_ps(x, x), _mm_set1_ps(static_cast<float>(divider_)));
    };

    int32_t divider_ = 1;
};

template <>
struct UnarySquare<float8_t, float8_t, false>
{
    UnarySquare(const int32_t divider = 1) { (void)divider; };

    void operator()(float8_t& y, const float8_t& x) const { y = _mm256_mul_ps(x, x); };
};

template <>
struct UnarySquare<float8_t, float8_t, true>
{
    UnarySquare(const int32_t divider = 1) { divider_ = divider; };

    void operator()(float8_t& y, const float8_t& x) const
    {
        y = _mm256_div_ps(_mm256_mul_ps(x, x), _mm256_set1_ps(static_cast<float>(divider_)));
    };

    int32_t divider_ = 1;
};

template <typename Y, typename X>
struct UnaryAbs;

template <>
struct UnaryAbs<float, float>
{
    UnaryAbs(const int32_t divider = 1) { (void)divider; };

    void operator()(float& y, const float& x) const { y = abs(x); };
};

template <>
struct UnaryAbs<float4_t, float4_t>
{
    UnaryAbs(const int32_t divider = 1) { (void)divider; };

    void operator()(float4_t& y, const float4_t& x) const
    {
        __m128 Mask = _mm_castsi128_ps(_mm_set1_epi32(~0x80000000));
        y           = _mm_and_ps(Mask, x);
    };
};

template <>
struct UnaryAbs<float8_t, float8_t>
{
    UnaryAbs(const int32_t divider = 1) { (void)divider; };

    void operator()(float8_t& y, const float8_t& x) const
    {
        __m256 Mask = _mm256_castsi256_ps(_mm256_set1_epi32(~0x80000000));
        y           = _mm256_and_ps(Mask, x);
    };
};

template <typename Y, typename X>
struct UnarySqrt;

template <>
struct UnarySqrt<float, float>
{
    void operator()(float& y, const float& x) const { y = sqrtf(x); };
};

template <>
struct UnarySqrt<float4_t, float4_t>
{
    void operator()(float4_t& y, const float4_t& x) const { y = _mm_sqrt_ps(x); };
};

template <>
struct UnarySqrt<float8_t, float8_t>
{
    void operator()(float8_t& y, const float8_t& x) const { y = _mm256_sqrt_ps(x); };
};

} // namespace element_wise
} // namespace cpu
} // namespace tensor_operation
} // namespace ck
