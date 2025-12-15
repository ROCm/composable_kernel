// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"
#include "ck/host_utility/hip_check_error.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include <hip/hip_runtime.h>
#include <vector>

namespace ck {
namespace ref {

// RAII wrapper for device memory to prevent leaks
struct SimpleDeviceMem
{
    SimpleDeviceMem() = delete;

    SimpleDeviceMem(std::size_t mem_size) : p_mem_{}
    {
        HIP_CHECK_ERROR(hipMalloc(static_cast<void**>(&p_mem_), mem_size));
    }

    void* GetDeviceBuffer() { return p_mem_; }

    ~SimpleDeviceMem() { (void)hipFree(p_mem_); }

    void* p_mem_;
};

// Helper function to compute layout-aware strides for convolution tensors
// For channel-last layouts (GNHWC, GKYXC, GNHWK): C/K is the innermost dimension
// For channel-first layouts (GNCDHW, GKCZYX, GNKDHW): spatial dimensions are innermost
inline std::vector<index_t> compute_conv_tensor_strides(const std::vector<index_t>& lengths,
                                                        index_t ndim_spatial,
                                                        bool channel_last)
{
    std::vector<index_t> strides(lengths.size());

    if(channel_last)
    {
        // Channel-last layout: spatial dimensions come before C/K in memory
        // lengths[0] = G, lengths[1] = N/K, lengths[2] = C/K, lengths[3...] = spatial
        // Memory order: G, N/K, spatial..., C/K
        strides[2]     = 1; // C/K is innermost
        index_t stride = static_cast<index_t>(lengths[2]);

        // Spatial dimensions in reverse order
        for(int i = ndim_spatial + 2; i >= 3; --i)
        {
            strides[i] = stride;
            stride *= lengths[i];
        }

        // N/K
        strides[1] = stride;
        stride *= lengths[1];

        // G
        strides[0] = stride;
    }
    else
    {
        // Row-major layout (channel-first or fallback)
        // Memory order follows index order: G, N/K, C/K, spatial...
        index_t stride = 1;
        for(int i = lengths.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= lengths[i];
        }
    }

    return strides;
}

// Template helper to detect if a layout is channel-last (C or K as innermost dimension)
template <typename Layout>
constexpr bool is_channel_last_layout()
{
    using namespace ck::tensor_layout::convolution;

    // Input layouts with C last
    if constexpr(std::is_same_v<Layout, NWC> || std::is_same_v<Layout, NHWC> ||
                 std::is_same_v<Layout, NDHWC> || std::is_same_v<Layout, GNWC> ||
                 std::is_same_v<Layout, GNHWC> || std::is_same_v<Layout, GNDHWC> ||
                 std::is_same_v<Layout, NWGC> || std::is_same_v<Layout, NHWGC> ||
                 std::is_same_v<Layout, NDHWGC>)
    {
        return true;
    }
    // Weight layouts with C last
    else if constexpr(std::is_same_v<Layout, KXC> || std::is_same_v<Layout, KYXC> ||
                      std::is_same_v<Layout, KZYXC> || std::is_same_v<Layout, GKXC> ||
                      std::is_same_v<Layout, GKYXC> || std::is_same_v<Layout, GKZYXC> ||
                      std::is_same_v<Layout, KXGC> || std::is_same_v<Layout, KYXGC> ||
                      std::is_same_v<Layout, KZYXGC>)
    {
        return true;
    }
    // Output layouts with K last
    else if constexpr(std::is_same_v<Layout, NWK> || std::is_same_v<Layout, NHWK> ||
                      std::is_same_v<Layout, NDHWK> || std::is_same_v<Layout, GNWK> ||
                      std::is_same_v<Layout, GNHWK> || std::is_same_v<Layout, GNDHWK> ||
                      std::is_same_v<Layout, NWGK> || std::is_same_v<Layout, NHWGK> ||
                      std::is_same_v<Layout, NDHWGK>)
    {
        return true;
    }
    else
    {
        return false;
    }
}

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
