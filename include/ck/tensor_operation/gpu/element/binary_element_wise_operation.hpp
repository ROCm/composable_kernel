#pragma once

#include "ck/utility/data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct Add
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 + x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 + x1;
    };

    // Question: should half_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 + x1;
    };

    // Question: should bhalf_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x0);
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x1_tmp + x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }
};

struct Subtract
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = x0 - x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = x0 - x1;
    };

    // Question: should half_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = x0 - x1;
    };

    // Question: should bhalf_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<bhalf_t>(bhalf_t& y, const bhalf_t& x0, const bhalf_t& x1) const
    {
        const float x1_tmp = ck::type_convert<float>(x0);
        const float x2_tmp = ck::type_convert<float>(x1);
        const float y_tmp  = x1_tmp - x2_tmp;
        y                  = ck::type_convert<bhalf_t>(y_tmp);
    }
};

struct AlphaBetaAdd
{
    AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};

    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        y = alpha_ * x0 + beta_ * x1;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        y = static_cast<double>(alpha_) * x0 + static_cast<double>(beta_) * x1;
    };

    // Question: should half_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        y = static_cast<half_t>(alpha_ * static_cast<float>(x0) + beta_ * static_cast<float>(x1));
    };

    float alpha_;
    float beta_;
};

struct AddRelu
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        const float a = x0 + x1;
        y             = a > 0.0f ? a : 0.0f;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        const double a = x0 + x1;
        y              = a > 0.0 ? a : 0.0;
    };

    // Question: should half_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        const half_t a = x0 + x1;
        y              = a > static_cast<half_t>(0.0f) ? a : static_cast<half_t>(0.0f);
    };
};

struct AddHardswish
{
    template <typename T>
    __host__ __device__ constexpr void operator()(T& y, const T& x0, const T& x1) const;

    template <>
    __host__ __device__ constexpr void
    operator()<float>(float& y, const float& x0, const float& x1) const
    {
        float a = x0 + x1;
        float b = a + float{3};
        float c = (b > 0) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        y       = c;
    };

    template <>
    __host__ __device__ constexpr void
    operator()<double>(double& y, const double& x0, const double& x1) const
    {
        double a = x0 + x1;
        double b = a + 3.0;
        double c = (b > 0) * (b > 6.0 ? 6.0 : b) * a * 0.166667;
        y        = c;
    };

    // Question: should half_t be supported ?
    template <>
    __host__ __device__ constexpr void
    operator()<half_t>(half_t& y, const half_t& x0, const half_t& x1) const
    {
        float a = x0 + x1;
        float b = a + 3.0f;
        float c = (b > 0) * (b > 6.0f ? 6.0f : b) * a * 0.166667f;
        y       = c;
    };
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
