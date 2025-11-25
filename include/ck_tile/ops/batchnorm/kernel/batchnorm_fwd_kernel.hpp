// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batchnorm/block/block_welford.hpp"
#include "ck_tile/ops/batchnorm/pipeline/batchnorm_problem.hpp"
#include "ck_tile/ops/batchnorm/pipeline/batchnorm_shape.hpp"

namespace ck_tile {

// BatchnormFwd: Forward pass batch normalization kernel
// Performs:
// 1. Welford reduction to compute mean and variance across spatial dimensions (H*W)
// 2. Normalization: y = (x - mean) / sqrt(variance + epsilon)
// 
// For now, simplified version without scale/bias
template <typename Problem_>
struct BatchnormFwd
{
    using Problem         = remove_cvref_t<Problem_>;
    using XDataType       = typename Problem::XDataType;
    using ComputeDataType = typename Problem::ComputeDataType;
    using YDataType       = typename Problem::YDataType;
    using BlockShape      = typename Problem::BlockShape;

    static constexpr index_t kBlockSize = BlockShape::kBlockSize;

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    CK_TILE_DEVICE void operator()(const XDataType* p_x,
                                   YDataType* p_y,
                                   index_t N,
                                   index_t C,
                                   index_t H,
                                   index_t W,
                                   ComputeDataType epsilon) const
    {
        // For batchnorm: input shape is [N, C, H, W]
        // We reduce over H*W (spatial dimensions) for each N*C combination
        // Each block handles one or more (N,C) combinations

        const index_t spatial_size = H * W;
        const index_t nc_size = N * C;
        
        const index_t thread_id = get_thread_id();
        const index_t block_id = get_block_id();
        
        // For POC: simple mapping - each block handles one (N,C) pair
        // More sophisticated tiling will come later
        const index_t nc_idx = block_id;
        
        if(nc_idx >= nc_size)
            return;
        
        // Calculate n and c from linear index
        const index_t n = nc_idx / C;
        const index_t c = nc_idx % C;
        
        // Calculate base offset for this (N,C) pair
        const index_t base_offset = n * C * H * W + c * H * W;
        
        // Thread-local Welford statistics
        ComputeDataType thread_mean = type_convert<ComputeDataType>(0);
        ComputeDataType thread_m2 = type_convert<ComputeDataType>(0);
        index_t thread_count = 0;
        
        // Each thread processes a portion of the spatial dimension
        // Simple strided access pattern
        for(index_t idx = thread_id; idx < spatial_size; idx += kBlockSize)
        {
            const index_t offset = base_offset + idx;
            ComputeDataType val = type_convert<ComputeDataType>(p_x[offset]);
            
            // Online Welford update
            thread_count++;
            ComputeDataType delta = val - thread_mean;
            thread_mean += delta / type_convert<ComputeDataType>(thread_count);
            ComputeDataType delta2 = val - thread_mean;
            thread_m2 += delta * delta2;
        }
        
        // Allocate shared memory for block-level reduction
        __shared__ char smem[BlockWelford<ComputeDataType>::template GetSmemSize<index_t, kBlockSize>()];
        
        // Block-level Welford reduction
        ComputeDataType block_mean = thread_mean;
        ComputeDataType block_var = thread_m2;
        index_t block_count = thread_count;
        
        BlockWelford<ComputeDataType>::template Run<index_t, kBlockSize>(
            block_mean, block_var, block_count, smem);
        
        // Now all threads have the same mean and variance
        // Normalize and write output
        ComputeDataType inv_std = type_convert<ComputeDataType>(1) / 
            ck_tile::sqrt(block_var + epsilon);
        
        for(index_t idx = thread_id; idx < spatial_size; idx += kBlockSize)
        {
            const index_t offset = base_offset + idx;
            ComputeDataType val = type_convert<ComputeDataType>(p_x[offset]);
            ComputeDataType normalized = (val - block_mean) * inv_std;
            p_y[offset] = type_convert<YDataType>(normalized);
        }
    }

    // Validate arguments
    CK_TILE_HOST static bool IsSupportedArgument(index_t N, 
                                                 index_t C, 
                                                 index_t H, 
                                                 index_t W)
    {
        // For POC, accept all sizes
        // Later we can add alignment requirements
        if(N <= 0 || C <= 0 || H <= 0 || W <= 0)
        {
            return false;
        }
        return true;
    }
};

} // namespace ck_tile
