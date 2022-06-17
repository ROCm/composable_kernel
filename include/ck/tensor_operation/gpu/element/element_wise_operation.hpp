#pragma once
#include "data_type.hpp"
#include "math_v2.hpp"
#include "unary_element_wise_operation.hpp"
#include "binary_element_wise_operation.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

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

struct Normalize
{
    Normalize(double epsilon = 1e-4) : epsilon_(epsilon) {}

    template <typename T>
    __host__ __device__ constexpr void operator()(
        T& y, const T& x, const T& mean, const T& mean_square, const T& gamma, const T& beta) const;

    template <>
    __host__ __device__ constexpr void operator()<float>(float& y,
                                                         const float& x,
                                                         const float& mean,
                                                         const float& mean_square,
                                                         const float& gamma,
                                                         const float& beta) const
    {
        using ck::math::sqrt;

        float variance = mean_square - (mean * mean);
        y = ((x - mean) / sqrt(variance + static_cast<float>(epsilon_))) * gamma + beta;
    };

    template <>
    __host__ __device__ constexpr void operator()<double>(double& y,
                                                          const double& x,
                                                          const double& mean,
                                                          const double& mean_square,
                                                          const double& gamma,
                                                          const double& beta) const
    {
        using ck::math::sqrt;

        double variance = mean_square - (mean * mean);
        y               = ((x - mean) / sqrt(variance + epsilon_)) * gamma + beta;
    };

    double epsilon_;
};

template <typename Y, typename X>
struct UnaryTypeConvert;

template <>
struct UnaryTypeConvert<float, ck::bhalf_t>
{
    __host__ __device__ void operator()(float& y, ck::bhalf_t& x) const
    {
        y = ck::type_convert<float, ck::bhalf_t>(x);
    };
};

template <>
struct UnaryTypeConvert<ck::bhalf_t, float>
{
    __host__ __device__ void operator()(ck::bhalf_t& y, float& x) const
    {
        y = ck::type_convert<ck::bhalf_t, float>(x);
    };
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
