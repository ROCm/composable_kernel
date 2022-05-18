#pragma once
#include "data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace binary_element_wise {

struct Add
{
    __host__ __device__ constexpr void
    operator()(double& dst, const double& src1, const double& src2) const
    {
        dst = src1 + src2;
    }

    __host__ __device__ constexpr void
    operator()(float& dst, const float& src1, const float& src2) const
    {
        dst = src1 + src2;
    }
};

} // namespace binary_element_wise
} // namespace tensor_operation
} // namespace ck
