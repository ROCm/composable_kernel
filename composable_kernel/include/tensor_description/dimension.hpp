#ifndef CK_DIMENSION_HPP
#define CK_DIMENSION_HPP

#include "common_header.hpp"

namespace ck {

template <index_t Length>
struct Dimension
{
    __host__ __device__ static constexpr auto GetLength() { return Number<Length>{}; }
};

template <index_t Length, index_t Stride>
struct NativeDimension
{
    __host__ __device__ static constexpr auto GetLength() { return Number<Length>{}; }

    __host__ __device__ static constexpr auto GetStride() { return Number<Stride>{}; }

    __host__ __device__ static constexpr index_t CalculateOffset(index_t i) { return i * Stride; }

    __host__ __device__ static constexpr index_t CalculateOffsetDiff(index_t i_diff)
    {
        return i_diff * Stride;
    }
};

} // namespace ck
#endif
