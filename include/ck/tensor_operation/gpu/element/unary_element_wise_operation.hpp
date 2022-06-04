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

struct UnaryDivide
{
    __host__ __device__ UnaryDivide(const int32_t divider = 1) : divider_(divider){};

    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        impl_divide(y, x);
    };

    private:
    __host__ __device__ void impl_divide(float& y, const float& x) const
    {
        y = x / type_convert<float>(divider_);
    };

    __host__ __device__ void impl_divide(double& y, const double& x) const
    {
        y = x / type_convert<double>(divider_);
    };

    __host__ __device__ void impl_divide(int32_t& y, const int32_t& x) const
    {
        y = x / type_convert<int32_t>(divider_);
    };

    int32_t divider_ = 1;
};

struct UnarySquare
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        impl_square(y, x);
    };

    private:
    __host__ __device__ void impl_square(float& y, const float& x) const { y = x * x; };

    __host__ __device__ void impl_square(double& y, const double& x) const { y = x * x; };
};

struct UnaryAbs
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        y = ck::math::abs(x);
    };
};

struct UnarySqrt
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        y = ck::math::sqrt(x);
    };
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
