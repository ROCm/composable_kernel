// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <array>

/// This file implements a generic GPU tensor "foreach" function. This
/// functionality turned out useful in separate parts of the testing
/// system, hence its implemented in a separate file. This version is
/// not particularly efficient (but it should at least be readable),
/// but it should be easy to replace the implementation in the future,
/// should that be needed.

namespace ck_tile::builder::test {

namespace detail {

constexpr int DEVICE_FOREACH_BLOCK_SIZE = 256;

template <int BLOCK_SIZE, size_t RANK, typename F>
__global__ __launch_bounds__(BLOCK_SIZE) //
    void foreach_kernel(const size_t numel, std::array<size_t, RANK> shape_scan, F f)
{
    const auto gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    for(size_t flat_idx = gid; flat_idx < numel; flat_idx += gridDim.x * BLOCK_SIZE)
    {
        // Compute the current index.
        std::array<size_t, RANK> index = {};

        size_t idx = flat_idx;
        for(size_t i = 0; i < RANK; ++i)
        {
            const auto scanned_dim = shape_scan[i];
            index[i]               = idx / scanned_dim;
            idx %= scanned_dim;
        }

        // Then invoke the callback with the index.
        f(index);
    }
}

} // namespace detail

template <size_t RANK>
__host__ __device__ size_t calculate_offset(const std::array<size_t, RANK>& index,
                                            const std::array<size_t, RANK>& strides)
{
    size_t offset = 0;
#pragma unroll
    for(size_t i = 0; i < RANK; ++i)
    {
        offset += index[i] * strides[i];
    }
    return offset;
}

template <size_t RANK, typename F>
void tensor_foreach(std::span<const size_t, RANK> shape, F f)
{
    constexpr int block_size = detail::DEVICE_FOREACH_BLOCK_SIZE;
    const auto kernel        = detail::foreach_kernel<block_size, RANK, F>;

    int occupancy;
    check_hip(hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, block_size, 0));

    int device;
    check_hip(hipGetDevice(&device));

    int multiprocessors;
    check_hip(
        hipDeviceGetAttribute(&multiprocessors, hipDeviceAttributeMultiprocessorCount, device));

    std::array<size_t, RANK> shape_scan;
    size_t numel = 1;
    for(int i = RANK; i > 0; --i)
    {
        shape_scan[i - 1] = numel;
        numel *= shape[i - 1];
    }

    // Reset any errors from previous launches.
    (void)hipGetLastError();

    kernel<<<occupancy * multiprocessors, block_size>>>(numel, shape_scan, f);
    check_hip(hipGetLastError());
}

} // namespace ck_tile::builder::test
