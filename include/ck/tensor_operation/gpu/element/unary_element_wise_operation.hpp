// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/math_v2.hpp"
#include "ck/utility/type_convert.hpp"
#include <cassert>

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct UnaryOpBase
{
    public:
    __host__ __device__ virtual ~UnaryOpBase() = default;

    __host__ __device__ UnaryOpBase()                   = default;
    __host__ __device__ UnaryOpBase(const UnaryOpBase&) = default;
    __host__ __device__ UnaryOpBase& operator=(const UnaryOpBase&) = default;
    __host__ __device__ UnaryOpBase(UnaryOpBase&&)                 = default;
    __host__ __device__ UnaryOpBase& operator=(UnaryOpBase&&) = default;

    __host__ __device__ virtual inline void operator()(float& y, const float& x) const = 0;

    __host__ __device__ virtual inline void operator()(double& y, const double& x) const = 0;

    __host__ __device__ virtual inline void operator()(int32_t& y, const int32_t& x) const = 0;

    __host__ __device__ virtual inline void operator()(int8_t& y, const int8_t& x) const = 0;

    __host__ __device__ virtual inline void operator()(half_t& y, const half_t& x) const = 0;

    __host__ __device__ virtual inline void operator()(bhalf_t& y, const bhalf_t& x) const = 0;
};

struct PassThroughPack2
{
    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

    __host__ __device__ constexpr void operator()(ck::half2_t& y, const ck::f8x2_t& x) const
    {
        auto t = type_convert<float2_t>(x);
        y      = type_convert<half2_t>(t);
    }
    constexpr const static bool is_pack2_invocable = true;
};

struct PassThrough : public UnaryOpBase
{

    __host__ __device__ inline void operator()(float& y, const float& x) const final { y = x; }

    __host__ __device__ inline void operator()(double& y, const double& x) const final { y = x; }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final { y = x; }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final { y = x; }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final { y = x; }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final { y = x; }

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const;

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
    __host__ __device__ void operator()<half_t, float>(half_t& y, const float& x) const
    {
        y = type_convert<half_t>(x);
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
    __host__ __device__ void operator()<half_t, int8_t>(half_t& y, const int8_t& x) const
    {
        y = type_convert<half_t>(x);
    }

    template <>
    __host__ __device__ void operator()<bhalf_t, int8_t>(bhalf_t& y, const int8_t& x) const
    {
        y = type_convert<bhalf_t>(x);
    }

    template <>
    __host__ __device__ void operator()<uint8_t, uint8_t>(uint8_t& y, const uint8_t& x) const
    {
        y = x;
    }

    template <>
    __host__ __device__ void operator()<int8_t, int32_t>(int8_t& y, const int32_t& x) const
    {
        y = type_convert<int8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<int32_t, int8_t>(int32_t& y, const int8_t& x) const
    {
        y = type_convert<int32_t>(x);
    }

    template <>
    __host__ __device__ void operator()<int8_t, float>(int8_t& y, const float& x) const
    {
        y = type_convert<int8_t>(x);
    }

    template <>
    __host__ __device__ void operator()<float, int8_t>(float& y, const int8_t& x) const
    {
        y = type_convert<float>(x);
    }

#ifdef CK_EXPERIMENTAL_BIT_INT_EXTENSION_INT4
    template <>
    __host__ __device__ void operator()<int4_t, int4_t>(int4_t& y, const int4_t& x) const
    {
        y = x;
    }
    template <>
    __host__ __device__ void operator()<int4_t, int>(int4_t& y, const int& x) const
    {
        y = type_convert<int4_t>(x);
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
    __host__ __device__ Scale(float scale = 1.f) : scale_(scale) {}

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        y = ck::type_convert<Y>(ck::type_convert<float>(x) * scale_);
    }

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

    template <>
    __host__ __device__ void operator()<int8_t, int8_t>(int8_t& y, const int8_t& x) const
    {
        y = ck::type_convert<int8_t>(scale_ * ck::type_convert<float>(x));
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
                          is_same<T, int32_t>::value || is_same<T, int8_t>::value,
                      "Data type is not supported by this operation!");

        y = x / type_convert<T>(divider_);
    };

    template <>
    __host__ __device__ void operator()<half_t>(half_t& y, const half_t& x) const
    {
        float x_         = type_convert<float>(x);
        float divider_f_ = type_convert<float>(divider_);

        y = type_convert<half_t>(x_ / divider_f_);
    };

    template <>
    __host__ __device__ void operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        float x_         = type_convert<float>(x);
        float divider_f_ = type_convert<float>(divider_);

        y = type_convert<bhalf_t>(x_ / divider_f_);
    };

    template <>
    __host__ __device__ void operator()<f8_t>(f8_t& y, const f8_t& x) const
    {
        float x_         = type_convert<float>(x);
        float divider_f_ = type_convert<float>(divider_);

        y = type_convert<f8_t>(x_ / divider_f_);
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

struct UnaryAbs : public UnaryOpBase
{
    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        y = ck::math::abs(x);
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        y = ck::math::abs(x);
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        y = ck::math::abs(x);
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        y = ck::math::abs(x);
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        y = ck::math::abs(x);
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
    {
        y = ck::math::abs(x);
    }

    __host__ __device__ void operator()(f8_t& y, const f8_t& x) const
    {
        y = ck::type_convert<f8_t>(ck::math::abs(ck::type_convert<float>(x)));
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

struct Relu : public UnaryOpBase
{
    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        y = x > 0 ? x : 0;
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        y = x > 0 ? x : 0;
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        y = x > 0 ? x : 0;
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        y = x > 0 ? x : 0;
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        y = x > 0 ? x : 0;
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
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
// gpu code use lower accuracy "_ocml_exp_f32" and "rcp" function
struct FastGelu
{
    template <typename Y, typename X>
    __host__ void operator()(Y& y, const X& x) const;

    template <typename Y, typename X>
    __device__ void operator()(Y& y, const X& x) const;

    template <>
    __host__ void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = -2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = exp(u);
        y               = x / (1.f + emu);
    }

    // device code, use lower precision "__ocml_exp_f32" and "rcp"
    template <>
    __device__ void operator()<float, float>(float& y, const float& x) const
    {
        // const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
        const float c1  = -2.0 * 0.035677f;
        const float c2  = -2.0 * 0.797885f;
        const float u   = x * (c1 * x * x + c2);
        const float emu = __ocml_exp_f32(u);

        y = x * ck::math::rcp(1.f + emu);
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

    template <>
    __host__ void operator()<bhalf_t, float>(bhalf_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<bhalf_t>(y_f);
    }

    template <>
    __device__ void operator()<bhalf_t, float>(bhalf_t& y, const float& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, x);

        y = type_convert<bhalf_t>(y_f);
    }

    template <>
    __device__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<bhalf_t>(y_f);
    }

    template <>
    __host__ void operator()<bhalf_t, bhalf_t>(bhalf_t& y, const bhalf_t& x) const
    {
        float y_f;

        this->operator()<float, float>(y_f, type_convert<float>(x));

        y = type_convert<bhalf_t>(y_f);
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

struct Sigmoid : public UnaryOpBase
{

    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        constexpr float one = type_convert<float>(1);
        y                   = one / (one + ck::math::exp(-x));
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        constexpr double one = type_convert<double>(1);
        y                    = one / (one + ck::math::exp(-x));
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        constexpr int32_t one = type_convert<int32_t>(1);
        y                     = one / (one + ck::math::exp(-x));
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        constexpr int8_t one = type_convert<int8_t>(1);
        y                    = one / (one + ck::math::exp(-x));
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        constexpr half_t one = type_convert<half_t>(1);
        y                    = one / (one + ck::math::exp(-x));
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
    {
        constexpr float one = type_convert<float>(1);
        float x_f32         = ck::type_convert<float>(x);
        float y_f32         = one / (one + ck::math::exp(x_f32));
        y                   = ck::type_convert<bhalf_t>(y_f32);
    }
};

struct Silu
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same_v<T, float> || is_same_v<T, double> || is_same_v<T, ck::half_t> ||
                          is_same_v<T, int8_t> || is_same_v<T, int32_t>,
                      "Data type is not supported by this operation!");
        constexpr T one = type_convert<T>(1);
        y               = x * (one / (one + ck::math::exp(-x)));
    };
};

struct TanH : public UnaryOpBase
{
    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        y = ck::math::tanh(x);
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        y = ck::math::tanh(x);
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        y = ck::math::tanh(x);
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        y = ck::math::tanh(x);
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        y = ck::math::tanh(x);
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
    {
        y = ck::math::tanh(x);
    }
};

struct ACos
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::acos(x);
    };
};

struct Neg
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::neg(x);
    };
};

struct ATan
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::atan(x);
    };
};

struct Sin
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::sin(x);
    };
};

struct ASinH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::asinh(x);
    };
};

struct Cos
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::cos(x);
    };
};

struct ACosH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::acosh(x);
    };
};

struct Tan
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::tan(x);
    };
};

struct ATanH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::atanh(x);
    };
};

struct SinH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::sinh(x);
    };
};

struct Ceil
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::ceil(x);
    };
};

struct Exp
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::exp(x);
    };
};

struct CosH
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::cosh(x);
    };
};

struct Floor
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::floor(x);
    };
};

struct Log
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::log(x);
    };
};

struct ASin
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::asin(x);
    };
};

struct Rcp
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(is_same<T, float>::value || is_same<T, double>::value ||
                          is_same<T, ck::half_t>::value || is_same<T, int8_t>::value ||
                          is_same<T, int32_t>::value,
                      "Data type is not supported by this operation!");

        y = ck::math::rcp(x);
    };
};

struct Swish : public UnaryOpBase
{
    __host__ __device__ Swish(float beta = 1.0f) : beta_(beta) {}

    __host__ __device__ float get_beta() const { return beta_; }

    const float beta_;

    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<float>(x / (1.f + ck::math::exp(bx)));
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<double>(x / (1.f + ck::math::exp(bx)));
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<int32_t>(x / (1.f + ck::math::exp(bx)));
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<int8_t>(x / (1.f + ck::math::exp(bx)));
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<half_t>(x / (1.f + ck::math::exp(bx)));
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
    {
        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<bhalf_t>(x / (1.f + ck::math::exp(bx)));
    }

    template <typename Y, typename X>
    __host__ __device__ void operator()(Y& y, const X& x) const
    {
        static_assert(is_same<X, float>::value || is_same<X, double>::value ||
                          is_same<X, half_t>::value,
                      "Data type is not supported by this operation!");

        static_assert(is_same<Y, float>::value || is_same<Y, double>::value ||
                          is_same<Y, half_t>::value,
                      "Data type is not supported by this operation!");

        float bx = -beta_ * type_convert<float>(x);
        y        = type_convert<Y>(x / (1.f + ck::math::exp(bx)));
    }
};

struct SoftRelu : public UnaryOpBase
{
    __host__ __device__ SoftRelu(float alpha = 1.0f) : alpha_(alpha) {}

    __host__ __device__ float get_alpha() const { return alpha_; }

    const float alpha_;

    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        float casted_alpha  = type_convert<float>(alpha_);
        constexpr float one = type_convert<float>(1);
        y                   = ck::math::log(one + ck::math::exp(x * casted_alpha)) / casted_alpha;
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        double casted_alpha  = type_convert<double>(alpha_);
        constexpr double one = type_convert<double>(1);
        y                    = ck::math::log(one + ck::math::exp(x * casted_alpha)) / casted_alpha;
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        int32_t casted_alpha  = type_convert<int32_t>(alpha_);
        constexpr int32_t one = type_convert<int32_t>(1);
        y                     = ck::math::log(one + ck::math::exp(x * casted_alpha)) / casted_alpha;
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        int8_t casted_alpha  = type_convert<int8_t>(alpha_);
        constexpr int8_t one = type_convert<int8_t>(1);
        y                    = ck::math::log(one + ck::math::exp(x * casted_alpha)) / casted_alpha;
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        half_t casted_alpha  = type_convert<half_t>(alpha_);
        constexpr half_t one = type_convert<half_t>(1);
        y                    = ck::math::log(one + ck::math::exp(x * casted_alpha)) / casted_alpha;
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
    {
        bhalf_t casted_alpha  = type_convert<bhalf_t>(alpha_);
        constexpr bhalf_t one = type_convert<bhalf_t>(1);
        y                     = ck::math::log(one + ck::math::exp(x * casted_alpha)) / casted_alpha;
    }
};

struct Power : public UnaryOpBase
{
    __host__ __device__ Power(float alpha = 0.f, float beta = 1.f, float gamma = 2.f)
        : alpha_(alpha), beta_(beta), gamma_(gamma)
    {
    }

    __host__ __device__ float get_alpha() const { return alpha_; }

    __host__ __device__ float get_beta() const { return beta_; }

    __host__ __device__ float get_gamma() const { return gamma_; }

    const float alpha_;
    const float beta_;
    const float gamma_;

    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        float casted_alpha = type_convert<float>(alpha_);
        float casted_beta  = type_convert<float>(beta_);
        float casted_gamma = type_convert<float>(gamma_);

        float shifted_scaled_x = casted_alpha + casted_beta * x;
        y                      = ck::math::pow(shifted_scaled_x, casted_gamma);
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        double casted_alpha = type_convert<double>(alpha_);
        double casted_beta  = type_convert<double>(beta_);
        double casted_gamma = type_convert<double>(gamma_);

        double shifted_scaled_x = casted_alpha + casted_beta * x;
        y                       = ck::math::pow(shifted_scaled_x, casted_gamma);
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        int32_t casted_alpha = type_convert<int32_t>(alpha_);
        int32_t casted_beta  = type_convert<int32_t>(beta_);
        int32_t casted_gamma = type_convert<int32_t>(gamma_);

        int32_t shifted_scaled_x = casted_alpha + casted_beta * x;
        y                        = ck::math::pow(shifted_scaled_x, casted_gamma);
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        int8_t casted_alpha = type_convert<int8_t>(alpha_);
        int8_t casted_beta  = type_convert<int8_t>(beta_);
        int8_t casted_gamma = type_convert<int8_t>(gamma_);

        int8_t shifted_scaled_x = casted_alpha + casted_beta * x;
        y                       = ck::math::pow(shifted_scaled_x, casted_gamma);
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        half_t casted_alpha = type_convert<half_t>(alpha_);
        half_t casted_beta  = type_convert<half_t>(beta_);
        half_t casted_gamma = type_convert<half_t>(gamma_);

        half_t shifted_scaled_x = casted_alpha + casted_beta * x;
        y                       = ck::math::pow(shifted_scaled_x, casted_gamma);
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
    {
        bhalf_t casted_alpha = type_convert<bhalf_t>(alpha_);
        bhalf_t casted_beta  = type_convert<bhalf_t>(beta_);
        bhalf_t casted_gamma = type_convert<bhalf_t>(gamma_);

        bhalf_t shifted_scaled_x = casted_alpha + casted_beta * x;
        y                        = ck::math::pow(shifted_scaled_x, casted_gamma);
    }
};

struct ClippedRelu : public UnaryOpBase
{
    __host__ __device__ ClippedRelu(float alpha = 0.f, float beta = 1.f)
        : alpha_(alpha), beta_(beta)
    {
    }

    __host__ __device__ float get_alpha() const { return alpha_; }

    __host__ __device__ float get_beta() const { return beta_; }

    const float alpha_;
    const float beta_;

    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        float casted_alpha = type_convert<float>(alpha_);
        float casted_beta  = type_convert<float>(beta_);
        y                  = ck::math::min(casted_beta, ck::math::max(casted_alpha, x));
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        double casted_alpha = type_convert<double>(alpha_);
        double casted_beta  = type_convert<double>(beta_);
        y                   = ck::math::min(casted_beta, ck::math::max(casted_alpha, x));
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        int32_t casted_alpha = type_convert<int32_t>(alpha_);
        int32_t casted_beta  = type_convert<int32_t>(beta_);
        y                    = ck::math::min(casted_beta, ck::math::max(casted_alpha, x));
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        int8_t casted_alpha = type_convert<int8_t>(alpha_);
        int8_t casted_beta  = type_convert<int8_t>(beta_);
        y                   = ck::math::min(casted_beta, ck::math::max(casted_alpha, x));
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        half_t casted_alpha = type_convert<half_t>(alpha_);
        half_t casted_beta  = type_convert<half_t>(beta_);
        y                   = ck::math::min(casted_beta, ck::math::max(casted_alpha, x));
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
    {
        bhalf_t casted_alpha = type_convert<bhalf_t>(alpha_);
        bhalf_t casted_beta  = type_convert<bhalf_t>(beta_);
        y                    = ck::math::min(casted_beta, ck::math::max(casted_alpha, x));
    }
};

struct LeakyRelu : public UnaryOpBase
{

    __host__ __device__ LeakyRelu(float alpha = 0.f) : alpha_(alpha) {}

    __host__ __device__ float get_alpha() const { return alpha_; }

    const float alpha_;

    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        float casted_alpha = type_convert<float>(alpha_);
        y                  = x >= 0 ? x : x * casted_alpha;
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        double casted_alpha = type_convert<double>(alpha_);
        y                   = x >= 0 ? x : x * casted_alpha;
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        int32_t casted_alpha = type_convert<int32_t>(alpha_);
        y                    = x >= 0 ? x : x * casted_alpha;
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        int8_t casted_alpha = type_convert<int8_t>(alpha_);
        y                   = x >= 0 ? x : x * casted_alpha;
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        half_t casted_alpha = type_convert<half_t>(alpha_);
        y                   = x >= 0 ? x : x * casted_alpha;
    }

    __host__ __device__ inline void operator()([[maybe_unused]] bhalf_t& y,
                                               [[maybe_unused]] const bhalf_t& x) const final
    {
    }
};

struct Elu : public UnaryOpBase
{

    __host__ __device__ Elu(float alpha = 1.f) : alpha_(alpha) {}

    __host__ __device__ float get_alpha() const { return alpha_; }

    const float alpha_;

    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        float casted_alpha = type_convert<float>(alpha_);
        y                  = x > 0 ? x : casted_alpha * ck::math::expm1(x);
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        double casted_alpha = type_convert<double>(alpha_);
        y                   = x > 0 ? x : casted_alpha * ck::math::expm1(x);
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        int32_t casted_alpha = type_convert<int32_t>(alpha_);
        y                    = x > 0 ? x : casted_alpha * ck::math::expm1(x);
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        int8_t casted_alpha = type_convert<int8_t>(alpha_);
        y                   = x > 0 ? x : casted_alpha * ck::math::expm1(x);
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        half_t casted_alpha = type_convert<half_t>(alpha_);
        y                   = x > 0 ? x : casted_alpha * ck::math::expm1(x);
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
    {
        bhalf_t casted_alpha = type_convert<bhalf_t>(alpha_);
        y                    = x > 0 ? x : casted_alpha * ck::math::expm1(x);
    }
};

struct Logistic : public UnaryOpBase
{

    __host__ __device__ Logistic(float alpha = 1.0f) : alpha_(alpha) {}

    __host__ __device__ float get_alpha() const { return alpha_; }

    const float alpha_;

    __host__ __device__ inline void operator()(float& y, const float& x) const final
    {
        float casted_alpha  = type_convert<float>(alpha_);
        constexpr float one = type_convert<float>(1);
        y                   = casted_alpha / (one + ck::math::exp(-x) * casted_alpha);
    }

    __host__ __device__ inline void operator()(double& y, const double& x) const final
    {
        double casted_alpha  = type_convert<double>(alpha_);
        constexpr double one = type_convert<double>(1);
        y                    = casted_alpha / (one + ck::math::exp(-x) * casted_alpha);
    }

    __host__ __device__ inline void operator()(int32_t& y, const int32_t& x) const final
    {
        int32_t casted_alpha  = type_convert<int32_t>(alpha_);
        constexpr int32_t one = type_convert<int32_t>(1);
        y                     = casted_alpha / (one + ck::math::exp(-x) * casted_alpha);
    }

    __host__ __device__ inline void operator()(int8_t& y, const int8_t& x) const final
    {
        int8_t casted_alpha  = type_convert<int8_t>(alpha_);
        constexpr int8_t one = type_convert<int8_t>(1);
        y                    = casted_alpha / (one + ck::math::exp(-x) * casted_alpha);
    }

    __host__ __device__ inline void operator()(half_t& y, const half_t& x) const final
    {
        half_t casted_alpha  = type_convert<half_t>(alpha_);
        constexpr half_t one = type_convert<half_t>(1);
        y                    = casted_alpha / (one + ck::math::exp(-x) * casted_alpha);
    }

    __host__ __device__ inline void operator()(bhalf_t& y, const bhalf_t& x) const final
    {
        bhalf_t casted_alpha  = type_convert<bhalf_t>(alpha_);
        constexpr bhalf_t one = type_convert<bhalf_t>(1);
        y                     = casted_alpha / (one + ck::math::exp(-x) * casted_alpha);
    }
};

struct ConvInvscale
{
    __host__ __device__ ConvInvscale(float scale_in  = 1.f,
                                     float scale_wei = 1.f,
                                     float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    __host__ __device__ void operator()(E& e, const C& c) const;

    template <>
    __host__ __device__ void operator()<f8_t, float>(f8_t& e, const float& c) const
    {
        e = type_convert<f8_t>(c / scale_in_ / scale_wei_ / scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

struct ConvScale
{
    __host__ __device__ ConvScale(float scale_in  = 1.f,
                                  float scale_wei = 1.f,
                                  float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    __host__ __device__ void operator()(E& e, const C& c) const;

    template <>
    __host__ __device__ void operator()<f8_t, float>(f8_t& e, const float& c) const
    {
        e = type_convert<f8_t>(c * scale_in_ * scale_wei_ * scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

struct ConvScaleRelu
{
    __host__ __device__ ConvScaleRelu(float scale_in  = 1.f,
                                      float scale_wei = 1.f,
                                      float scale_out = 1.f)
        : scale_in_(scale_in), scale_wei_(scale_wei), scale_out_(scale_out)
    {
    }

    template <typename E, typename C>
    __host__ __device__ void operator()(E& e, const C& c) const;

    template <>
    __host__ __device__ void operator()<f8_t, float>(f8_t& e, const float& c) const
    {
        float x;
        Relu{}(x, c * scale_in_ * scale_wei_);
        e = type_convert<f8_t>(x * scale_out_);
    };

    float scale_in_;
    float scale_wei_;
    float scale_out_;
};

// support fastconvert of int8 to fp16

template <typename InputDataType, typename OutputDataType, index_t RegPackNumber>
struct FastNumericArrayConverter
{
};

template <>
struct FastNumericArrayConverter<uint8_t, ck::half_t, 4>
{
    using InputArray  = vector_type<uint8_t, 4>;
    using OutputArray = vector_type<ck::half_t, 4>;

    __device__ static OutputArray convert(InputArray const& Input)
    {
        OutputArray Output;

        uint32_t* half_2       = reinterpret_cast<uint32_t*>(&Output);
        uint32_t const uint8_4 = reinterpret_cast<uint32_t const&>(Input);

        static constexpr uint32_t byte_selector_01 = 0x05010500;
        static constexpr uint32_t byte_selector_23 = 0x05030502;
        static constexpr uint32_t fp16_adder       = 0x64646464;
        half_2[0] = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_01);
        half_2[1] = __builtin_amdgcn_perm(fp16_adder, uint8_4, byte_selector_23);

        static constexpr uint32_t I8s_TO_F16s_MAGIC_NUM = 0x64806480;
        asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                     : "=v"(half_2[0])
                     : "v"(half_2[0]), "s"(I8s_TO_F16s_MAGIC_NUM));
        asm volatile("v_pk_add_f16 %0, %1, %2 neg_lo:[0,1] neg_hi:[0,1]"
                     : "=v"(half_2[1])
                     : "v"(half_2[1]), "s"(I8s_TO_F16s_MAGIC_NUM));

        return Output;
    }

    __device__ OutputArray operator()(InputArray const& Input) { return convert(Input); }
};

template <index_t N>
struct FastNumericArrayConverter<uint8_t, ck::half_t, N>
{
    static constexpr int VEC_WIDTH = 4;
    static_assert(!(N % VEC_WIDTH), "N must be multiple of 4.");

    using InputArray  = vector_type<uint8_t, N>;
    using OutputArray = vector_type<ck::half_t, N>;

    __device__ static OutputArray convert(InputArray const& Input)
    {
        FastNumericArrayConverter<uint8_t, ck::half_t, 4> converter;

        OutputArray Output;

        using Vec_InputArray  = vector_type<uint8_t, 4>;
        using Vec_OutputArray = vector_type<ck::half_t, 4>;

        Vec_OutputArray* half_4_ptr       = reinterpret_cast<Vec_OutputArray*>(&Output);
        Vec_InputArray const* uint8_4_ptr = reinterpret_cast<Vec_InputArray const*>(&Input);

        static_for<0, N / VEC_WIDTH, 1>{}(
            [&](auto i) { half_4_ptr[i] = converter(uint8_4_ptr[i]); });

        return Output;
    }

    __device__ OutputArray operator()(InputArray const& Input) { return convert(Input); }
};

struct DynamicUnaryOp
{

    DynamicUnaryOp& operator=(const DynamicUnaryOp& other)
    {
        if(this != &other)
        {
            unary_op_ptr_  = other.unary_op_ptr_;
            unary_op_type_ = other.unary_op_type_;
        }
        return *this;
    }

    __host__ __device__ DynamicUnaryOp() = delete;

    __host__ __device__ DynamicUnaryOp(const Swish& swish)
    {
        unary_op_type_ = UnaryOpType::Swish;
        beta           = swish.get_beta();
    }

    __host__ __device__ DynamicUnaryOp(const Swish&& swish)
    {
        unary_op_type_ = UnaryOpType::Swish;
        beta           = swish.get_beta();
    }

    __host__ __device__ DynamicUnaryOp(const Sigmoid&) { unary_op_type_ = UnaryOpType::Sigmoid; }

    __host__ __device__ DynamicUnaryOp(const Sigmoid&&) { unary_op_type_ = UnaryOpType::Sigmoid; }

    __host__ __device__ DynamicUnaryOp(const PassThrough&)
    {
        unary_op_type_ = UnaryOpType::PassThrough;
    }

    __host__ __device__ DynamicUnaryOp(const PassThrough&&)
    {
        unary_op_type_ = UnaryOpType::PassThrough;
    }

    __host__ __device__ DynamicUnaryOp(const Logistic& logistic)
    {
        unary_op_type_ = UnaryOpType::Logistic;
        alpha          = logistic.get_alpha();
    }

    __host__ __device__ DynamicUnaryOp(const Logistic&& logistic)
    {
        unary_op_type_ = UnaryOpType::Logistic;
        alpha          = logistic.get_alpha();
    }

    __host__ __device__ DynamicUnaryOp(const TanH&) { unary_op_type_ = UnaryOpType::TanH; }

    __host__ __device__ DynamicUnaryOp(const TanH&&) { unary_op_type_ = UnaryOpType::TanH; }

    __host__ __device__ DynamicUnaryOp(const Relu&) { unary_op_type_ = UnaryOpType::Relu; }

    __host__ __device__ DynamicUnaryOp(const Relu&&) { unary_op_type_ = UnaryOpType::Relu; }

    __host__ __device__ DynamicUnaryOp(const SoftRelu& softrelu)
    {
        unary_op_type_ = UnaryOpType::SoftRelu;
        alpha          = softrelu.get_alpha();
    }

    __host__ __device__ DynamicUnaryOp(const SoftRelu&& softrelu)
    {
        unary_op_type_ = UnaryOpType::SoftRelu;
        alpha          = softrelu.get_alpha();
    }

    __host__ __device__ DynamicUnaryOp(const UnaryAbs&) { unary_op_type_ = UnaryOpType::UnaryAbs; }

    __host__ __device__ DynamicUnaryOp(const UnaryAbs&&) { unary_op_type_ = UnaryOpType::UnaryAbs; }

    __host__ __device__ DynamicUnaryOp(const Power& pow)
    {
        unary_op_type_ = UnaryOpType::Power;
        alpha          = pow.get_alpha();
        beta           = pow.get_beta();
        gamma          = pow.get_gamma();
    }

    __host__ __device__ DynamicUnaryOp(const Power&& pow)
    {
        unary_op_type_ = UnaryOpType::Power;
        alpha          = pow.get_alpha();
        beta           = pow.get_beta();
        gamma          = pow.get_gamma();
    }

    __host__ __device__ DynamicUnaryOp(const ClippedRelu& clippedrelu)
    {
        unary_op_type_ = UnaryOpType::ClippedRelu;
        alpha          = clippedrelu.get_alpha();
        beta           = clippedrelu.get_beta();
    }

    __host__ __device__ DynamicUnaryOp(const ClippedRelu&& clippedrelu)
    {
        unary_op_type_ = UnaryOpType::ClippedRelu;
        alpha          = clippedrelu.get_alpha();
        beta           = clippedrelu.get_beta();
    }

    __host__ __device__ DynamicUnaryOp(const LeakyRelu& leakyrelu)
    {
        unary_op_type_ = UnaryOpType::LeakyRelu;
        alpha          = leakyrelu.get_alpha();
    }

    __host__ __device__ DynamicUnaryOp(const LeakyRelu&& leakyrelu)
    {
        unary_op_type_ = UnaryOpType::LeakyRelu;
        alpha          = leakyrelu.get_alpha();
    }

    __host__ __device__ DynamicUnaryOp(const Elu& elu)
    {
        unary_op_type_ = UnaryOpType::Elu;
        alpha          = elu.get_alpha();
    }

    __host__ __device__ DynamicUnaryOp(const Elu&& elu)
    {
        unary_op_type_ = UnaryOpType::Elu;
        alpha          = elu.get_alpha();
    }

    __host__ __device__ DynamicUnaryOp(const DynamicUnaryOp& dynamic_op)
        : unary_op_type_(dynamic_op.unary_op_type_),
          unary_op_ptr_(dynamic_op.unary_op_ptr_),
          alpha(dynamic_op.alpha),
          beta(dynamic_op.beta),
          gamma(dynamic_op.gamma)
    {
    }

    __host__ __device__ ~DynamicUnaryOp()
    {
        if(unary_op_ptr_)
            delete unary_op_ptr_;
    }

    __device__ void InitUnaryOpPtrOnDevice()
    {
        switch(unary_op_type_)
        {
        // printf("Inside InitUnaryOpPtrOnDevice\n");
        // printf("alpha %f beta %f gamma %f\n", alpha, beta, gamma);
        case(UnaryOpType::Swish): unary_op_ptr_ = new Swish(beta); break;
        case(UnaryOpType::Sigmoid): unary_op_ptr_ = new Sigmoid; break;
        case(UnaryOpType::PassThrough): unary_op_ptr_ = new PassThrough; break;
        case(UnaryOpType::Logistic): unary_op_ptr_ = new Logistic(alpha); break;
        case(UnaryOpType::TanH): unary_op_ptr_ = new TanH; break;
        case(UnaryOpType::Relu): unary_op_ptr_ = new Relu; break;
        case(UnaryOpType::SoftRelu): unary_op_ptr_ = new SoftRelu(alpha); break;
        case(UnaryOpType::UnaryAbs): unary_op_ptr_ = new UnaryAbs; break;
        case(UnaryOpType::Power): unary_op_ptr_ = new Power(alpha, beta, gamma); break;
        case(UnaryOpType::ClippedRelu): unary_op_ptr_ = new ClippedRelu(alpha, beta); break;
        case(UnaryOpType::LeakyRelu): unary_op_ptr_ = new LeakyRelu(alpha); break;
        case(UnaryOpType::Elu): unary_op_ptr_ = new Elu(alpha); break;

        default: unary_op_ptr_ = nullptr; break;
        }
    }

    template <typename Y, typename X>
    __device__ void operator()(Y& y, const X& x) const
    {
        isSupported<X, Y>();
        unary_op_ptr_->operator()(y, x);
    }

    template <typename Y, typename X>
    __host__ void operator()(Y& y, const X& x) const
    {
        isSupported<X, Y>();
        switch(unary_op_type_)
        {
        case(UnaryOpType::Swish): Swish{}.operator()(y, x); break;
        case(UnaryOpType::Sigmoid): Sigmoid{}.operator()(y, x); break;
        case(UnaryOpType::PassThrough): PassThrough{}.operator()(y, x); break;
        case(UnaryOpType::Logistic): Logistic{}.operator()(y, x); break;
        case(UnaryOpType::TanH): TanH{}.operator()(y, x); break;
        case(UnaryOpType::Relu): Relu{}.operator()(y, x); break;
        case(UnaryOpType::SoftRelu): SoftRelu{}.operator()(y, x); break;
        case(UnaryOpType::UnaryAbs): UnaryAbs{}.operator()(y, x); break;
        case(UnaryOpType::Power): Power{}.operator()(y, x); break;
        case(UnaryOpType::ClippedRelu): ClippedRelu{}.operator()(y, x); break;
        case(UnaryOpType::LeakyRelu): LeakyRelu{}.operator()(y, x); break;
        case(UnaryOpType::Elu): Elu{}.operator()(y, x); break;
        default: break;
        }
    }

    template <typename X, typename Y>
    __device__ __host__ constexpr void isSupported() const
    {

        static_assert(std::is_same<X, Y>::value, "X and Y must be of the same type");

        static_assert(is_same<X, float>::value || is_same<X, double>::value ||
                          is_same<X, bhalf_t>::value || is_same<X, half_t>::value ||
                          is_same<X, int32_t>::value || is_same<X, int8_t>::value,
                      "Data type is not supported by this operation!");
    }

    private:
    enum class UnaryOpType
    {
        Swish,
        Sigmoid,
        PassThrough,
        Logistic,
        TanH,
        Relu,
        SoftRelu,
        UnaryAbs,
        Power,
        ClippedRelu,
        LeakyRelu,
        Elu
    };

    public:
    UnaryOpType unary_op_type_;
    UnaryOpBase* unary_op_ptr_ = nullptr;
    float alpha;
    float beta;
    float gamma;
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
