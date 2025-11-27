// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batchnorm/block/block_welford.hpp"
#include "ck_tile/ops/batchnorm/pipeline/batchnorm_problem.hpp"
#include "ck_tile/ops/batchnorm/pipeline/batchnorm_shape.hpp"

namespace ck_tile {

// Host-side arguments for batchnorm forward pass
struct BatchnormFwdHostArgs
{
    const void* p_x;     // [N, C, H, W] input tensor (required)
    const void* p_gamma; // [C] scale parameter (required, use all 1.0 if not needed)
    const void* p_beta;  // [C] bias parameter (required, use all 0.0 if not needed)
    
    void* p_y;           // [N, C, H, W] output tensor (required)
    
    void* p_running_mean;     // [C] running mean (nullptr if not used)
    void* p_running_var;      // [C] running variance (nullptr if not used)
    void* p_save_mean;        // [C] save mean for backward (nullptr if not used)
    void* p_save_inv_std;     // [C] save inv_std for backward (nullptr if not used)
    
    float epsilon;
    float momentum;
    
    index_t N, C, H, W;
    
    // Note: save/update flags are now in Traits (compile-time), not here (runtime)
};

// BatchnormFwd: Forward pass batch normalization kernel
template <typename Problem_>
struct BatchnormFwd
{
    using Problem         = remove_cvref_t<Problem_>;
    using XDataType       = typename Problem::XDataType;
    using ComputeDataType = typename Problem::ComputeDataType;
    using YDataType       = typename Problem::YDataType;
    using BlockShape      = typename Problem::BlockShape;

    static constexpr index_t kBlockSize = BlockShape::kBlockSize;

    // Kernel arguments
    struct BatchnormFwdKargs
    {
        const void* p_x;
        const void* p_gamma;
        const void* p_beta;
        void* p_y;
        void* p_running_mean;
        void* p_running_var;
        void* p_save_mean;
        void* p_save_inv_std;
        
        float epsilon;
        float momentum;
        
        index_t N, C, H, W;
        
        // Note: save/update flags now come from Problem::Traits (compile-time)
    };

    using Kargs = BatchnormFwdKargs;  // Alias for convenience
    using Hargs = BatchnormFwdHostArgs;

    // Convert host args to kernel args
    CK_TILE_HOST static constexpr Kargs MakeKernelArgs(const Hargs& hargs)
    {
        return Kargs{hargs.p_x,
                     hargs.p_gamma,
                     hargs.p_beta,
                     hargs.p_y,
                     hargs.p_running_mean,
                     hargs.p_running_var,
                     hargs.p_save_mean,
                     hargs.p_save_inv_std,
                     hargs.epsilon,
                     hargs.momentum,
                     hargs.N,
                     hargs.C,
                     hargs.H,
                     hargs.W};
    }

    // Grid size calculation
    CK_TILE_HOST static constexpr auto GridSize(const Hargs& hargs)
    {
        return dim3(hargs.C);  // One block per channel
    }

    // Block size
    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return kBlockSize;
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        // Cast pointers to typed pointers
        const XDataType* p_x = static_cast<const XDataType*>(kargs.p_x);
        YDataType* p_y = static_cast<YDataType*>(kargs.p_y);
        
        const index_t N = kargs.N;
        const index_t C = kargs.C;
        const index_t H = kargs.H;
        const index_t W = kargs.W;
        const ComputeDataType epsilon = static_cast<ComputeDataType>(kargs.epsilon);

        const index_t spatial_size = H * W;
        const index_t per_channel_size = N * spatial_size;  // Reduce over N×H×W
        
        const index_t thread_id = get_thread_id();
        const index_t block_id = get_block_id();
        
        // Each block handles one channel
        const index_t c = block_id;
        
        if(c >= C)
            return;
        
        // Thread-local Welford statistics
        ComputeDataType thread_mean = type_convert<ComputeDataType>(0);
        ComputeDataType thread_m2 = type_convert<ComputeDataType>(0);
        index_t thread_count = 0;
        
        // Each thread processes elements across ALL samples (N) and spatial positions (H×W)
        // for this channel
        for(index_t idx = thread_id; idx < per_channel_size; idx += kBlockSize)
        {
            // Calculate which sample (n) and spatial position (hw) this is
            const index_t n = idx / spatial_size;
            const index_t hw = idx % spatial_size;
            
            // Memory layout: [N, C, H, W] with C-major for H×W
            const index_t offset = n * C * H * W + c * H * W + hw;
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
        
        // Load scale (gamma) and bias (beta) for this channel
        // Following old CK pattern: gamma/beta are ALWAYS provided (no nullptr checks)
        // All threads load (efficient, no branching)
        const ComputeDataType* p_gamma = static_cast<const ComputeDataType*>(kargs.p_gamma);
        const ComputeDataType* p_beta = static_cast<const ComputeDataType*>(kargs.p_beta);
        
        const ComputeDataType gamma = p_gamma[c];
        const ComputeDataType beta = p_beta[c];
        
        // Compute inverse standard deviation
        ComputeDataType inv_std = type_convert<ComputeDataType>(1) / 
            ck_tile::sqrt(block_var + epsilon);
        
        // Normalize and write output with scale and bias
        // Formula: y = gamma * (x - mean) / std + beta
        //        = gamma * (x - mean) * inv_std + beta
        for(index_t idx = thread_id; idx < per_channel_size; idx += kBlockSize)
        {
            const index_t n = idx / spatial_size;
            const index_t hw = idx % spatial_size;
            
            const index_t offset = (n * C * H * W) + (c * H * W) + hw;
            ComputeDataType val = type_convert<ComputeDataType>(p_x[offset]);
            
            // Apply batch normalization with scale and bias
            ComputeDataType normalized = gamma * ((val - block_mean) * inv_std) + beta;
            
            p_y[offset] = type_convert<YDataType>(normalized);
        }
        
        // Save mean and inverse std for backward pass (compile-time check)
        if constexpr(Problem::Traits::kSaveMeanInvStd)
        {
            if(thread_id == 0)
            {
                using MeanVarDataType = typename Problem::MeanVarDataType;
                MeanVarDataType* p_save_mean = static_cast<MeanVarDataType*>(kargs.p_save_mean);
                MeanVarDataType* p_save_inv_std = static_cast<MeanVarDataType*>(kargs.p_save_inv_std);
                
                p_save_mean[c] = type_convert<MeanVarDataType>(block_mean);
                p_save_inv_std[c] = type_convert<MeanVarDataType>(inv_std);
            }
        }
        
        // Update running mean and variance (compile-time check)
        if constexpr(Problem::Traits::kUpdateMovingAverage)
        {
            if(thread_id == 0)
            {
                using MeanVarDataType = typename Problem::MeanVarDataType;
                MeanVarDataType* p_running_mean = static_cast<MeanVarDataType*>(kargs.p_running_mean);
                MeanVarDataType* p_running_var = static_cast<MeanVarDataType*>(kargs.p_running_var);
                
                const ComputeDataType momentum = static_cast<ComputeDataType>(kargs.momentum);
                const ComputeDataType one_minus_momentum = type_convert<ComputeDataType>(1) - momentum;
                
                // Exponential moving average: new = (1-momentum)*old + momentum*current
                ComputeDataType old_mean = type_convert<ComputeDataType>(p_running_mean[c]);
                ComputeDataType old_var = type_convert<ComputeDataType>(p_running_var[c]);
                
                p_running_mean[c] = type_convert<MeanVarDataType>(one_minus_momentum * old_mean + momentum * block_mean);
                p_running_var[c] = type_convert<MeanVarDataType>(one_minus_momentum * old_var + momentum * block_var);
            }
        }
    }

    // Validate arguments
    CK_TILE_HOST static bool IsSupportedArgument(const Hargs& hargs)
    {
        // Basic validation
        if(hargs.N <= 0 || hargs.C <= 0 || hargs.H <= 0 || hargs.W <= 0)
        {
            return false;
        }
        
        // Validate required pointers
        if(hargs.p_x == nullptr || hargs.p_y == nullptr ||
           hargs.p_gamma == nullptr || hargs.p_beta == nullptr)
        {
            return false;
        }
        
        // Validate optional pointers based on Traits (compile-time)
        if constexpr(Problem::Traits::kUpdateMovingAverage)
        {
            if(hargs.p_running_mean == nullptr || hargs.p_running_var == nullptr)
            {
                return false;
            }
        }
        
        if constexpr(Problem::Traits::kSaveMeanInvStd)
        {
            if(hargs.p_save_mean == nullptr || hargs.p_save_inv_std == nullptr)
            {
                return false;
            }
        }
        
        return true;
    }
};

} // namespace ck_tile
