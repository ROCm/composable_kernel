#pragma once

#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

// For Activation function which is piecewise linear function, such as relu, leaky relu ...etc
template <typename Activation>
struct Activation_Mul_Clamp
{
    Activation_Mul_Clamp(float multiplier, Activation activationOp)
        : multiplier_(multiplier), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void operator()(int8_t& y, const int32_t& x) const
    {
        float x_fp32 = ck::type_convert<float>(x);
        activationOp_(x_fp32, x_fp32);
        float y_fp32 = math::clamp(multiplier_ * x_fp32, -128.f, 127.f);
        y            = ck::type_convert<int8_t>(y_fp32);
    }

    __host__ __device__ constexpr void operator()(float& y, const int32_t& x) const
    {
        // We might type_convert to int8 after lambda in someplace
        float x_fp32 = ck::type_convert<float>(x);
        activationOp_(x_fp32, x_fp32);
        y = math::clamp(multiplier_ * x_fp32, -128.f, 127.f);
    }

    float multiplier_;
    Activation activationOp_;
};

// For Activation function which is piecewise linear function, such as relu, leaky relu ...etc
template <typename Activation>
struct Add_Activation_Mul_Clamp
{
    Add_Activation_Mul_Clamp(float multiplier, Activation activationOp)
        : multiplier_(multiplier), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x1, const int32_t& x2) const
    {
        float y_fp32 = ck::type_convert<float>(x1 + x2);
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(multiplier_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    float multiplier_;
    Activation activationOp_;
};

// For Activation function which is non piecewise linear function, such as TanH, Sigmoid ...etc
template <typename Activation>
struct Add_Mul_Activation_Mul_Clamp
{
    Add_Mul_Activation_Mul_Clamp(float multiplier1, float multiplier2, Activation activationOp)
        : multiplier1_(multiplier1), multiplier2_(multiplier2), activationOp_(activationOp)
    {
    }

    __host__ __device__ constexpr void
    operator()(int8_t& y, const int32_t& x1, const int32_t& x2) const
    {
        float y_fp32 = ck::type_convert<float>(x1 + x2);
        y_fp32       = multiplier1_ * y_fp32;
        activationOp_(y_fp32, y_fp32);
        y_fp32 = math::clamp(multiplier2_ * y_fp32, -128.f, 127.f);
        y      = ck::type_convert<int8_t>(y_fp32);
    }

    float multiplier1_;
    float multiplier2_;
    Activation activationOp_;
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
