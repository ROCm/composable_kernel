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
    const void* p_x;     // [N, H, W, C] input tensor (required, NHWC layout)
    const void* p_gamma; // [C] scale parameter (required, use all 1.0 if not needed)
    const void* p_beta;  // [C] bias parameter (required, use all 0.0 if not needed)
    
    void* p_y;           // [N, H, W, C] output tensor (required, NHWC layout)
    
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
    using Pipeline        = BatchnormFwdPipeline<Problem>;  // Class-level, like layernorm2d
    using XDataType       = typename Problem::XDataType;
    using GammaDataType   = typename Problem::GammaDataType;
    using BetaDataType    = typename Problem::BetaDataType;
    using ComputeDataType = typename Problem::ComputeDataType;
    using YDataType       = typename Problem::YDataType;
    using MeanVarDataType = typename Problem::MeanVarDataType;
    using BlockShape      = typename Problem::BlockShape;

    static constexpr index_t kBlockSize = BlockShape::BlockSize;
    static constexpr index_t Block_M    = BlockShape::Block_M;
    static constexpr index_t Block_N    = BlockShape::Block_N;
    static constexpr index_t Vector_M   = BlockShape::Vector_M;
    static constexpr index_t Vector_N   = BlockShape::Vector_N;

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

    // Block size (wave32/64 aware like layernorm2d)
    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? BlockShape::template GetBlockSize<true>()
                           : BlockShape::template GetBlockSize<false>();
    }

    // Shared memory size (must be constexpr for __shared__ allocation)
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Pipeline::GetSmemSize();
    }

    CK_TILE_DEVICE void operator()(Kargs kargs) const
    {
        const index_t c = get_block_id();
        
        const index_t spatial_size = kargs.H * kargs.W;
        const index_t per_channel_size = kargs.N * spatial_size;
        
        // NHWC layout: channels are contiguous!
        const auto x_window = [&]() {
            const XDataType* p_x = static_cast<const XDataType*>(kargs.p_x);
            const XDataType* p_x_channel = p_x + c;  // Offset by c (channel stride = 1!)
            
            const auto x_view = make_naive_tensor_view<address_space_enum::global>(
                p_x_channel,
                make_tuple(kargs.N, spatial_size),  // [N, H×W]
                make_tuple(spatial_size*kargs.C, kargs.C),  // NHWC strides: [H×W×C, C]
                number<Vector_M>{},
                number<Vector_N>{});
            
            const auto tmp2_ = pad_tensor_view(
                x_view, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<false, false>{});
            
            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {0, 0});
        }();
        
        auto y_window = [&]() {
            YDataType* p_y = static_cast<YDataType*>(kargs.p_y);
            YDataType* p_y_channel = p_y + c;  // Offset by c (NHWC)
            
            const auto y_view = make_naive_tensor_view<address_space_enum::global>(
                p_y_channel,
                make_tuple(kargs.N, spatial_size),  // [N, H×W]
                make_tuple(spatial_size*kargs.C, kargs.C),  // NHWC strides
                number<Vector_M>{},
                number<Vector_N>{});
            
            const auto tmp2_ = pad_tensor_view(
                y_view, make_tuple(number<Block_M>{}, number<Block_N>{}), sequence<false, false>{});
            
            return make_tile_window(tmp2_, make_tuple(number<Block_M>{}, number<Block_N>{}), {0, 0});
        }();
        
        // Allocate shared memory (use kernel's constexpr GetSmemSize)
        __shared__ char smem[GetSmemSize()];
        
        // Cast pointers for optional features
        MeanVarDataType* p_running_mean = static_cast<MeanVarDataType*>(kargs.p_running_mean);
        MeanVarDataType* p_running_var = static_cast<MeanVarDataType*>(kargs.p_running_var);
        MeanVarDataType* p_save_mean = static_cast<MeanVarDataType*>(kargs.p_save_mean);
        MeanVarDataType* p_save_inv_std = static_cast<MeanVarDataType*>(kargs.p_save_inv_std);
        
        // Call pipeline with x/y windows and gamma/beta pointers
        const GammaDataType* p_gamma = static_cast<const GammaDataType*>(kargs.p_gamma);
        const BetaDataType* p_beta = static_cast<const BetaDataType*>(kargs.p_beta);
        
        Pipeline{}(x_window,
                   p_gamma,
                   p_beta,
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
