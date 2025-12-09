// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include "ck/ck.hpp"

namespace ck {
namespace ref {

// Generic kernel to pack strided tensor into contiguous layout
template <typename DataType>
__global__ void pack_strided_tensor(const DataType* __restrict__ src,
                                    DataType* __restrict__ dst,
                                    const index_t* src_lengths,
                                    const index_t* src_strides,
                                    int num_dims,
                                    long_index_t total_elements)
{
    const long_index_t tid         = blockIdx.x * blockDim.x + threadIdx.x;
    const long_index_t num_threads = blockDim.x * gridDim.x;

    for(long_index_t dst_idx = tid; dst_idx < total_elements; dst_idx += num_threads)
    {
        long_index_t remaining = dst_idx;
        long_index_t src_idx   = 0;

        for(int dim = num_dims - 1; dim >= 0; --dim)
        {
            index_t coord = remaining % src_lengths[dim];
            remaining /= src_lengths[dim];
            src_idx += coord * src_strides[dim];
        }

        dst[dst_idx] = src[src_idx];
    }
}

// Generic kernel to unpack contiguous tensor into strided layout
template <typename DataType>
__global__ void unpack_to_strided_tensor(const DataType* __restrict__ src,
                                         DataType* __restrict__ dst,
                                         const index_t* dst_lengths,
                                         const index_t* dst_strides,
                                         int num_dims,
                                         long_index_t total_elements)
{
    const long_index_t tid         = blockIdx.x * blockDim.x + threadIdx.x;
    const long_index_t num_threads = blockDim.x * gridDim.x;

    for(long_index_t src_idx = tid; src_idx < total_elements; src_idx += num_threads)
    {
        long_index_t remaining = src_idx;
        long_index_t dst_idx   = 0;

        for(int dim = num_dims - 1; dim >= 0; --dim)
        {
            index_t coord = remaining % dst_lengths[dim];
            remaining /= dst_lengths[dim];
            dst_idx += coord * dst_strides[dim];
        }

        dst[dst_idx] = src[src_idx];
    }
}

} // namespace ref
} // namespace ck
