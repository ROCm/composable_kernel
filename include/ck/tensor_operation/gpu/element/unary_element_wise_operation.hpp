// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/utility/type_convert.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

#if CK_WORKAROUND_SWDEV_383542
extern "C" __device__ float __ocml_native_recip_f32(float);
#endif

struct PassThroughPack2
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    __host__ __device__ constexpr void operator()(ck::f8x2_t& y, const ck::half2_t& x) const
    {
        // fake conversion
        uint16_t t = ck::bit_cast<uint32_t>(x);
        y          = ck::bit_cast<ck::f8x2_t>(t);
    }

    __host__ __device__ constexpr void operator()(ck::half2_t& y, const ck::f8x2_t& x) const
    {
        auto t = type_convert<float2_t>(x);
        y      = type_convert<half2_t>(t);
    }

    __host__ __device__ constexpr void operator()(ck::half2_t& y, const ck::half2_t& x) const
    {
        y = x;
    }

    __host__ __device__ constexpr void operator()(ck::f8x2_t& y, const ck::f8x2_t& x) const
    {
        y = x;
    }

    __host__ __device__ constexpr void operator()(ck::float2_t& y, const ck::float2_t& x) const
    {
        y = x;
    }

    __host__ __device__ constexpr void operator()(ck::int8x2_t& y, const ck::int8x2_t& x) const
    {
        y = x;
    }

    __host__ __device__ constexpr void operator()(ck::bhalf2_t& y, const ck::bhalf2_t& x) const
    {
        y = x;
    }

    __host__ __device__ constexpr void operator()(ck::double2_t& y, const ck::double2_t& x) const
    {
        y = x;
    }

    constexpr const static bool is_pack2_invocable = true;
};

struct PassThrough
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<double, double>(double& y, const double& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<float, double>(float& y, const double& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<double, float>(double& y, const float& x) const
    {
        y = type_convert<double>(x);
    }

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<half_t, float>(half_t& y, const float& x) const
    {
        y = type_convert<half_t>(x);
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<int32_t, int32_t>(int32_t& y, const int32_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, float>(bhalf_t& y, const float& x) const
    {
        y = type_convert<bhalf_t>(x);
    }

    template <>
    __host__ __device__ void operator()<float, bhalf_t>(float& y, const bhalf_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, half_t>(bhalf_t& y, const half_t& x) const
    {
        y = type_convert<bhalf_t>(x);
    }

    template <>
    __host__ __device__ void operator()<float, half_t>(float& y, const half_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<int8_t, int8_t>(int8_t& y, const int8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<half_t, int8_t>(half_t& y, const int8_t& x) const
    {
        y = type_convert<half_t>(x);
    }

    template <>
    __host__ __device__ void operator()<int8_t, int32_t>(int8_t& y, const int32_t& x) const
    {
        y = type_convert<int8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<int8_t, float>(int8_t& y, const float& x) const
    {
        y = type_convert<int8_t>(x);
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    __host__ __device__ void operator()<int4_t, int4_t>(int4_t& y, const int4_t& x) const
    {
        y = x;
    }
#endif

    template <>
    __host__ __device__ void operator()<f8_t, f8_t>(f8_t& y, const f8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<float, f8_t>(float& y, const f8_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<f8_t, float>(f8_t& y, const float& x) const
    {
        y = type_convert<f8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<half_t, f8_t>(half_t& y, const f8_t& x) const
    {
        y = type_convert<half_t>(x);
    }

    template <>
    __host__ __device__ void operator()<f8_t, half_t>(f8_t& y, const half_t& x) const
    {
        y = type_convert<f8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<bf8_t, bf8_t>(bf8_t& y, const bf8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<float, bf8_t>(float& y, const bf8_t& x) const
    {
        y = type_convert<float>(x);
    }

    template <>
    __host__ __device__ void operator()<bf8_t, float>(bf8_t& y, const float& x) const
    {
        y = type_convert<bf8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<half_t, bf8_t>(half_t& y, const bf8_t& x) const
    {
        y = type_convert<half_t>(x);
    }

    template <>
    __host__ __device__ void operator()<bf8_t, half_t>(bf8_t& y, const half_t& x) const
    {
        y = ck::type_convert<bf8_t>(x);
    }
};

struct UnaryConvert
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        y = type_convert<Y>(x);
    }
};

struct ConvertBF16RTN
{
    // convert to bf16 using round to nearest (rtn)
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(is_same<Y, bhalf_t>::value, "Data type is not supported by this operation!");

        // check X datatype
        static_assert(is_same<X, float>::value || is_same<X, half_t>::value,
                      "Data type is not supported by this operation!");

        y = bf16_convert_rtn<Y>(x);
    }
};

struct ConvertF8SR
{
    // convert to fp8 using stochastic rounding (SR)
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(is_same<Y, f8_t>::value || is_same<Y, bf8_t>::value,
                      "Data type is not supported by this operation!");

        // check X datatype
        static_assert(is_same<X, float>::value || is_same<X, half_t>::value,
                      "Data type is not supported by this operation!");

        y = f8_convert_sr<Y>(x);
    }
};

struct ConvertF8RNE
{
    // convert to fp8 using rounding to nearest even
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        // check Y datatype
        static_assert(is_same<Y, f8_t>::value || is_same<Y, bf8_t>::value,
                      "Data type is not supported by this operation!");

        // check X datatype
        static_assert(is_same<X, float>::value || is_same<X, half_t>::value,
                      "Data type is not supported by this operation!");

        y = f8_convert_rne<Y>(x);
    }
};

struct Scale
{
    __host__ __device__ Scale(float scale) : scale_(scale) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        y = ck::type_convert<half_t>(scale_) * x;
    };

    template <>
    __host__ __device__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        const float x_tmp = ck::type_convert<float>(x);
        const float y_tmp = scale_ * x_tmp;
        y                 = ck::type_convert<bhalf_t>(y_tmp);
    };

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = scale_ * x;
    };

    template <>
    __host__ __device__ void operator()<double, double>(double& y, const double& x) const
    {
        y = scale_ * x;
    };

    float scale_;
};

struct ScaleAndResetNaNToMinusInfinity
{
    __host__ __device__ ScaleAndResetNaNToMinusInfinity(float scale) : scale_(scale) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = ck::math::isnan(x) ? -ck::NumericLimits<float>::Infinity() : scale_ * x;
    };

    float scale_;
};

struct UnaryDivide
{
    __host__ __device__ UnaryDivide(const int32_t divider = 1) : divider_(divider) {}

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = x / type_convert<T>(divider_);
    };

    int32_t divider_ = 1;
};

struct UnarySquare
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same_v<T, float> || is_same_v<T, half_t> || is_same_v<T, double> ||
                          is_same_v<T, int32_t> || is_same_v<T, int8_t>
#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
                          || is_same_v<T, int4_t>
#endif
                      ,
                      "Data type is not supported by this operation!");
        y = x * x;
    };
};

struct UnaryAbs
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::abs(x);
    };
};

struct UnarySqrt
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::sqrt(x);
    };
};

struct Relu
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        y = x > 0 ? x : 0;
    }

    template <>
    __host__ __device__ void operator()(bhalf_t& y, const bhalf_t& x) const
    {
        float x_f32 = ck::type_convert<float>(x);
        float y_f32 = x_f32 > 0 ? x_f32 : 0;
        y           = ck::type_convert<bhalf_t>(y_f32);
    }
};

// Fast GeLU
// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// host code use higher accuracy "exp" and "div"
// gpu code use lower accuracy "__expf" and "rcp" function
struct FastGelu
{
    template <typename Y, typename X>
    __host__ void operator()(Y& y, const X& x) const;

    template <typename Y, typename X>
    __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ void operator()<float, float>(float& y, const float& x) const
    {
        const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float emu = exp(-u);
        const float cdf = 0.5f + 0.5f * (2.f / (1.f + emu) - 1.f);

        y = x * cdf;
    }

    // device code, use lower precision "__expf" and "rcp"
    template <>
    __device__ void operator()<float, float>(float& y, const float& x) const
    {
        const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float emu = __expf(-u);

#if !CK_WORKAROUND_SWDEV_383542
        const float cdf = 0.5f + 0.5f * (2.f * __frcp_rn(1.f + emu) - 1.f);
#else
        const float cdf = 0.5f + 0.5f * (2.f * __ocml_native_recip_f32(1.f + emu) - 1.f);
#endif

        y = x * cdf;
    }

    template <>
    __host__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<half_t>(y_f);
    }

    template <>
    __device__ void operator()<half_t, half_t>(half_t& y, const half_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<half_t>(y_f);
    }

    template <>
    __host__ void operator()<half_t, float>(half_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<half_t>(y_f);
    }

    template <>
    __device__ void operator()<half_t, float>(half_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<half_t>(y_f);
    }
};

// https://paperswithcode.com/method/gelu
// y = 0.5*x*(1+erf(x/sqrt(2)))
struct Gelu
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ __device__ void operator()<float, float>(float& y, const float& x) const
    {
        y = 0.5f * x * (1.f + erf(float(0.70710678118f * x)));
    }

    template <>
    __host__ __device__ void operator()<ck::half_t, ck::half_t>(ck::half_t& y,
                                                                const ck::half_t& x) const
    {
        y = ck::half_t(0.5) * x * (ck::half_t(1) + ck::half_t(erf(float(0.70710678118f * x))));
    }
};

struct Sigmoid
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");
        constexpr T one = type_convert<T>(1);
        y               = one / (one + ck::math::exp(-x));
    };
};

struct TanH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::tanh(x);
    };
};

struct Swish
{
    Swish(float beta = 1.0f) : beta_(beta) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        static_assert(is_same<X, float>::value || is_same<X, double>::value ||
                          is_same<X, ck::half_t>::value,
                      "Data type is not supported by this operation!");

        static_assert(is_same<Y, float>::value || is_same<Y, double>::value ||
                          is_same<Y, ck::half_t>::value,
                      "Data type is not supported by this operation!");

        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<Y>(x / (1.f + ck::math::exp(bx)));
    };

    const float beta_;
};

struct SoftRelu
{
    SoftRelu(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha  = type_convert<T>(alpha_);
        constexpr T one = type_convert<T>(1);
        y               = ck::math::log(one + ck::math::exp(x * casted_alpha)) / casted_alpha;
    }
    const float alpha_;
};

struct Power
{
    Power(float alpha = 0.f, float beta = 1.f, float gamma = 2.f)
        : alpha_(alpha), beta_(beta), gamma_(gamma){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha     = type_convert<T>(alpha_);
        T casted_beta      = type_convert<T>(beta_);
        T casted_gamma     = type_convert<T>(gamma_);
        T shifted_scaled_x = casted_alpha + casted_beta * x;
        y                  = ck::math::pow(shifted_scaled_x, casted_gamma);
    }
    const float alpha_;
    const float beta_;
    const float gamma_;
};

struct ClippedRelu
{
    ClippedRelu(float alpha = 0.f, float beta = 1.f) : alpha_(alpha), beta_(beta){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        T casted_beta  = type_convert<T>(beta_);
        y              = ck::math::min(casted_beta, ck::math::max(casted_alpha, x));
    }
    const float alpha_;
    const float beta_;
};

struct LeakyRelu
{
    LeakyRelu(float alpha = 0.01f) : alpha_(alpha){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        y              = x >= 0 ? x : x * casted_alpha;
    }
    const float alpha_;
};

struct Elu
{
    Elu(float alpha = 1.f) : alpha_(alpha){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, half_t>::value || is_same<T, int32_t>::value ||
                          is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");
        T casted_alpha = type_convert<T>(alpha_);
        y              = x > 0 ? x : casted_alpha * ck::math::expm1(x);
    }
    const float alpha_;
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
