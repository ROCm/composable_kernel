// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batchnorm/block/block_welford.hpp"
#include "ck_tile/ops/batchnorm/pipeline/batchnorm_fwd_pipeline.hpp"
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
    using GammaDataType   = typename Problem::GammaDataType;
    using BetaDataType    = typename Problem::BetaDataType;
    using ComputeDataType = typename Problem::ComputeDataType;
    using YDataType       = typename Problem::YDataType;
    using MeanVarDataType = typename Problem::MeanVarDataType;
    using BlockShape      = typename Problem::BlockShape;

    static constexpr index_t kBlockSize = BlockShape::BlockSize;

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

    // Shared memory size (must be constexpr for __shared__ allocation)
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return BatchnormFwdPipeline<Problem>::GetSmemSize();
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        using Pipeline = BatchnormFwdPipeline<Problem>;
        
        const index_t N = kargs.N;
        const index_t C = kargs.C;
        const index_t H = kargs.H;
        const index_t W = kargs.W;
        
        const index_t block_id = get_block_id();
        const index_t c = block_id;  // Channel index
        
        if(c >= C)
            return;
        
        const index_t spatial_size = H * W;
        const index_t per_channel_size = N * spatial_size;
        
        // Use block dimensions from BlockShape (like layernorm2d)
        static constexpr index_t Block_M = BlockShape::Block_M;
        static constexpr index_t Block_N = BlockShape::Block_N;
        
        // Create tensor views WITHOUT distributions (will be applied in pipeline)
        const auto x_window = [&]() {
            const XDataType* p_x = static_cast<const XDataType*>(kargs.p_x);
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                p_x + c * spatial_size,
                make_tuple(N, spatial_size),
                make_tuple(C * spatial_size, 1),
                number<1>{},
                number<1>{});
            
            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<false, false>{});
            
            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {0, 0});
        }();
        
        const auto gamma_window = [&]() {
            const GammaDataType* p_gamma = static_cast<const GammaDataType*>(kargs.p_gamma);
            const auto tmp_ = make_naive_tensor_view_packed<address_space_enum::global>(
                p_gamma + c,
                make_tuple(1),
                number<1>{});
            
            const auto tmp2_ = pad_tensor_view(tmp_, make_tuple(number<Block_M>{}), sequence<false>{});
            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}), {0});
        }();
        
        const auto beta_window = [&]() {
            const BetaDataType* p_beta = static_cast<const BetaDataType*>(kargs.p_beta);
            const auto tmp_ = make_naive_tensor_view_packed<address_space_enum::global>(
                p_beta + c,
                make_tuple(1),
                number<1>{});
            
            const auto tmp2_ = pad_tensor_view(tmp_, make_tuple(number<Block_M>{}), sequence<false>{});
            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}), {0});
        }();
        
        auto y_window = [&]() {
            YDataType* p_y = static_cast<YDataType*>(kargs.p_y);
            const auto tmp_ = make_naive_tensor_view<address_space_enum::global>(
                p_y + c * spatial_size,
                make_tuple(N, spatial_size),
                make_tuple(C * spatial_size, 1),
                number<1>{},
                number<1>{});
            
            const auto tmp2_ = pad_tensor_view(
                tmp_, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<false, false>{});
            
            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {0, 0});
        }();
        
        // Allocate shared memory (use kernel's constexpr GetSmemSize)
        __shared__ char smem[GetSmemSize()];
        
        // Cast pointers for optional features
        MeanVarDataType* p_running_mean = static_cast<MeanVarDataType*>(kargs.p_running_mean);
        MeanVarDataType* p_running_var = static_cast<MeanVarDataType*>(kargs.p_running_var);
        MeanVarDataType* p_save_mean = static_cast<MeanVarDataType*>(kargs.p_save_mean);
        MeanVarDataType* p_save_inv_std = static_cast<MeanVarDataType*>(kargs.p_save_inv_std);
        
        // Call pipeline with properly distributed tile windows
        Pipeline{}(x_window,
                   gamma_window,
                   beta_window,
                   y_window,
                   p_running_mean,
                   p_running_var,
                   p_save_mean,
                   p_save_inv_std,
                   static_cast<ComputeDataType>(kargs.epsilon),
                   static_cast<ComputeDataType>(kargs.momentum),
                   per_channel_size,
                   c,
                   smem);
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
