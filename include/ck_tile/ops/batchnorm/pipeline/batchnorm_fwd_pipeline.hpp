// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batchnorm/block/block_welford.hpp"
#include "ck_tile/ops/batchnorm/pipeline/batchnorm_fwd_policy.hpp"

namespace ck_tile {

// BatchnormFwdPipeline: Computation logic for batch normalization
// Takes tile windows and performs Welford reduction + normalization
template <typename Problem_, typename Policy_ = BatchnormFwdPipelineDefaultPolicy>
struct BatchnormFwdPipeline
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using XDataType       = typename Problem::XDataType;
    using GammaDataType   = typename Problem::GammaDataType;
    using BetaDataType    = typename Problem::BetaDataType;
    using ComputeDataType = typename Problem::ComputeDataType;
    using YDataType       = typename Problem::YDataType;
    using MeanVarDataType = typename Problem::MeanVarDataType;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename XWindow,
              typename GammaWindow,
              typename BetaWindow,
              typename YWindow>
    CK_TILE_DEVICE void operator()(const XWindow& x_window_,
                                   const GammaWindow& gamma_window_,
                                   const BetaWindow& beta_window_,
                                   YWindow& y_window_,
                                   MeanVarDataType* p_running_mean,
                                   MeanVarDataType* p_running_var,
                                   MeanVarDataType* p_save_mean,
                                   MeanVarDataType* p_save_inv_std,
                                   ComputeDataType epsilon,
                                   ComputeDataType momentum,
                                   [[maybe_unused]] index_t per_channel_size,
                                   index_t channel_idx,
                                   void* smem) const
    {
        const index_t thread_id = get_thread_id();
        
        // Apply tile distributions (like layernorm2d does)
        // Note: x_window and y_window are NOT const (need to move them)
        auto x_window =
            make_tile_window(x_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        const auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        const auto beta_window = make_tile_window(
            beta_window_, Policy::template MakeGammaBetaBlockTileDistribution<Problem>());
        auto y_window = 
            make_tile_window(y_window_, Policy::template MakeXBlockTileDistribution<Problem>());
        
        // Load gamma/beta once (constant per channel)
        [[maybe_unused]]const auto gamma = load_tile(gamma_window);
        [[maybe_unused]]const auto beta = load_tile(beta_window);
        
        // Calculate how many tiles needed (like layernorm2d two-pass)
        constexpr index_t Block_N = Problem::BlockShape::Block_N;
        index_t num_tile_iteration = integer_divide_ceil(per_channel_size, Block_N);

        // ==========================================
        // PHASE 1: WELFORD REDUCTION OVER ALL TILES
        // ==========================================
        ComputeDataType thread_mean = type_convert<ComputeDataType>(0);
        ComputeDataType thread_m2 = type_convert<ComputeDataType>(0);
        index_t thread_count = 0;

        // Iterate over tiles for Welford accumulation
        for(index_t tile_idx = 0; tile_idx < num_tile_iteration; ++tile_idx)
        {
            auto x = load_tile(x_window);
            
            sweep_tile(x, [&](auto idx) {
                ComputeDataType val = type_convert<ComputeDataType>(x[idx]);
                thread_count++;
                ComputeDataType delta = val - thread_mean;
                thread_mean += delta / type_convert<ComputeDataType>(thread_count);
                ComputeDataType delta2 = val - thread_mean;
                thread_m2 += delta * delta2;
            });
            
            // Move to next tile
            if(tile_idx < num_tile_iteration - 1)
            {
                move_tile_window(x_window, {0, Block_N});
            }
        }
        
        // Move x_window back to start
        move_tile_window(x_window, {0, -static_cast<int>(Block_N * (num_tile_iteration - 1))});

        // Block-level reduction
        ComputeDataType block_mean = thread_mean;
        ComputeDataType block_var = thread_m2;
        index_t block_count = thread_count;

        BlockWelford<ComputeDataType>::template Run<index_t, kBlockSize>(
            block_mean, block_var, block_count, smem);

        // ==========================================
        // PHASE 2: COMPUTE INVERSE STD
        // ==========================================
        ComputeDataType inv_std = type_convert<ComputeDataType>(1) / 
            ck_tile::sqrt(block_var + epsilon);

        // ==========================================
        // PHASE 3: NORMALIZE AND STORE (ITERATE OVER TILES)
        // ==========================================
        for(index_t tile_idx = 0; tile_idx < num_tile_iteration; ++tile_idx)
        {
            auto x = load_tile(x_window);
            auto y = make_static_distributed_tensor<YDataType>(x.get_tile_distribution());

            sweep_tile(y, [&](auto idx) {
                ComputeDataType x_val = type_convert<ComputeDataType>(x[idx]);
                
                // y = (x - mean) / std  (no gamma/beta for now)
                ComputeDataType normalized = (x_val - block_mean) * inv_std;
                y(idx) = type_convert<YDataType>(normalized);
            });

            store_tile(y_window, y);
            
            // Move to next tile
            if(tile_idx < num_tile_iteration - 1)
            {
                move_tile_window(x_window, {0, Block_N});
                move_tile_window(y_window, {0, Block_N});
            }
        }

        // ==========================================
        // PHASE 6: SAVE STATISTICS (Optional)
        // ==========================================
        if constexpr(Problem::Traits::kSaveMeanInvStd)
        {
            if(thread_id == 0)
            {
                p_save_mean[channel_idx] = type_convert<MeanVarDataType>(block_mean);
                p_save_inv_std[channel_idx] = type_convert<MeanVarDataType>(inv_std);
            }
        }

        // ==========================================
        // PHASE 7: UPDATE RUNNING STATISTICS (Optional)
        // ==========================================
        if constexpr(Problem::Traits::kUpdateMovingAverage)
        {
            if(thread_id == 0)
            {
                ComputeDataType one_minus_momentum = type_convert<ComputeDataType>(1) - momentum;
                
                ComputeDataType old_mean = type_convert<ComputeDataType>(p_running_mean[channel_idx]);
                ComputeDataType old_var = type_convert<ComputeDataType>(p_running_var[channel_idx]);
                
                p_running_mean[channel_idx] = type_convert<MeanVarDataType>(
                    one_minus_momentum * old_mean + momentum * block_mean);
                p_running_var[channel_idx] = type_convert<MeanVarDataType>(
                    one_minus_momentum * old_var + momentum * block_var);
            }
        }
    }
};

} // namespace ck_tile
