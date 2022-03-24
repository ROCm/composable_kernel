#pragma once
#include "data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace element_wise {

struct ReduceSum
{
    __host__ __device__ static constexpr float GetReduceZeroValue() { return float(0); }

    __host__ __device__ void Reduce(float& acc, float v) const { acc += v; }
};

struct ReduceSquareSum
{
    __host__ __device__ static constexpr float GetReduceZeroValue() { return float(0); }

    __host__ __device__ void Reduce(float& acc, float v) const { acc += v * v; }
};

} // namespace element_wise
} // namespace tensor_operation
} // namespace ck
