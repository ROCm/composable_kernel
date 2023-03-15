#pragma once

#include "ck/utility/data_type.hpp"
// #include "ck/utility/get_id.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

// For Activation function which is piecewise linear function, such as relu, leaky relu ...etc
template <typename Activation>
struct Activation_Mul_Clamp
{
    Activation_Mul_Clamp(float requantScale, Activation activationOp)
        : requantScale_(requantScale), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void operator()(int8_t& y, const int32_t& x) const
    {
        float y_fp32 = ck::type_convert<float>(x);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __device__ constexpr void operator()(int32_t& y, const int32_t& x) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    __host__ constexpr void operator()(float& y, const float& x) const
    {
        // CAUSION - We might float in & float out in reference code
        activationOp_(y, x);
        y = math::clamp(requantScale_ * y, -128.f, 127.f);
    }

    float requantScale_;
    Activation activationOp_;
};

// Conv Perchannel quantization + Activation function which is piecewise linear function, such as
// relu, leaky relu ...etc
template <typename Activation>
struct Activation_Mul2_Clamp
{
    Activation_Mul2_Clamp(Activation activationOp) : activationOp_(activationOp) {}

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x, const float& requantScale) const
    {
        float y_fp32 = ck::type_convert<float>(x);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __device__ constexpr void
    operator()(int32_t& y, const int32_t& x, const float& requantScale) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    Activation activationOp_;
};

// For Activation function which is piecewise linear function, such as relu, leaky relu ...etc
template <typename Activation>
struct Add_Activation_Mul_Clamp
{
    Add_Activation_Mul_Clamp(float requantScale, Activation activationOp)
        : requantScale_(requantScale), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x, const int32_t& bias) const
    {
        float y_fp32 = ck::type_convert<float>(x + bias);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __host__ __device__ constexpr void
    operator()(int32_t& y, const int32_t& x, const int32_t& bias) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x + bias);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    float requantScale_;
    Activation activationOp_;
};

// Conv Perchannel quantization + Activation function which is piecewise linear function, such as
// relu, leaky relu ...etc
template <typename Activation>
struct Add_Activation_Mul2_Clamp
{
    Add_Activation_Mul2_Clamp(Activation activationOp) : activationOp_(activationOp) {}

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x, const int32_t& bias, const float& requantScale) const
    {
        float y_fp32 = ck::type_convert<float>(x + bias);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    __host__ __device__ constexpr void
    operator()(int32_t& y, const int32_t& x, const int32_t& bias, const float& requantScale) const
    {
        // CAUSION - We might type_convert to int8 in threadwise copy
        // eg. GridwiseGemmDlMultipleD_km_kn_mn
        float y_fp32 = ck::type_convert<float>(x + bias);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int32_t>(y_fp32);
    }

    Activation activationOp_;
};

// For Activation function which is non piecewise linear function, such as TanH, Sigmoid ...etc
template <typename Activation>
struct Add_Mul_Activation_Mul_Clamp
{
    Add_Mul_Activation_Mul_Clamp(float requantScale1, float requantScale2, Activation activationOp)
        : requantScale1_(requantScale1), requantScale2_(requantScale2), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x, const int32_t& bias) const
    {
        float y_fp32 = ck::type_convert<float>(x + bias);
        y_fp32       = requantScale1_ * y_fp32;
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(requantScale2_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    float requantScale1_;
    float requantScale2_;
    Activation activationOp_;
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
