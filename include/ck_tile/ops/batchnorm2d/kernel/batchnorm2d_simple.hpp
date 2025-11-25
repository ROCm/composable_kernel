// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// Simple BatchNorm2d Forward - Single Block POC
// Each block processes one channel, reducing across all K elements
//
// Limitations for POC:
// - Single block per channel (M blocks total)
// - K must fit in one block (K <= BlockSize)
// - No scale/bias initially
// - fp32 only
//
template <index_t BlockSize_ = 256>
struct BatchNorm2dSimple
{
    static constexpr index_t BlockSize = BlockSize_;
    static constexpr index_t WarpSize = get_warp_size();
    static constexpr index_t NumWarps = BlockSize / WarpSize;
    
    // Host arguments
    struct HostArgs
    {
        const void* p_x;      // Input [M, K]
        void* p_y;            // Output [M, K]
        const void* p_scale;  // Scale [M] (optional, can be nullptr)
        const void* p_bias;   // Bias [M] (optional, can be nullptr)
        
        float epsilon;
        
        index_t m;  // Number of channels
        index_t k;  // Elements per channel
    };
    
    // Kernel arguments
    struct Kargs
    {
        const float* p_x;
        float* p_y;
        const float* p_scale;
        const float* p_bias;
        
        float epsilon;
        
        index_t m;
        index_t k;
    };
    
    CK_TILE_HOST static constexpr Kargs MakeKargs(const HostArgs& hargs)
    {
        return Kargs{
            static_cast<const float*>(hargs.p_x),
            static_cast<float*>(hargs.p_y),
            static_cast<const float*>(hargs.p_scale),
            static_cast<const float*>(hargs.p_bias),
            hargs.epsilon,
            hargs.m,
            hargs.k
        };
    }
    
    CK_TILE_HOST static constexpr auto GridSize(const HostArgs& hargs)
    {
        // One block per channel
        return dim3(hargs.m);
    }
    
    CK_TILE_HOST static constexpr auto BlockSize() { return BlockSize_; }
    
    CK_TILE_HOST static constexpr index_t GetSmemSize() 
    { 
        // LDS for cross-warp reduction: mean, M2, count
        return sizeof(float) * NumWarps * 3;
    }
    
    CK_TILE_HOST static std::string GetName()
    {
        return std::string("batchnorm2d_simple_") + std::to_string(BlockSize_);
    }
    
    // Welford merge operation
    CK_TILE_DEVICE static void WelfordMerge(float& mean_a,
                                            float& m2_a,
                                            index_t& count_a,
                                            float mean_b,
                                            float m2_b,
                                            index_t count_b)
    {
        const index_t count_total = count_a + count_b;
        
        if(count_total == 0)
            return;
            
        const float count_b_over_total = 
            static_cast<float>(count_b) / static_cast<float>(count_total);
        
        const float delta = mean_b - mean_a;
        
        mean_a = mean_a + delta * count_b_over_total;
        m2_a = m2_a + m2_b + delta * delta * 
               static_cast<float>(count_a) * count_b_over_total;
        count_a = count_total;
    }
    
    // Warp-level Welford reduction
    CK_TILE_DEVICE static void WarpWelfordReduce(float& mean, 
                                                 float& m2, 
                                                 index_t& count)
    {
        // Tree reduction using shuffle
        static_for<0, integer_log2_floor(WarpSize), 1>{}([&](auto I) {
            constexpr index_t Stride = WarpSize >> (I + 1);
            
            const float mean_other = warp_shuffle_down(mean, Stride);
            const float m2_other = warp_shuffle_down(m2, Stride);
            const index_t count_other = warp_shuffle_down(count, Stride);
            
            const index_t lane_id = get_lane_id();
            
            if(lane_id < Stride)
            {
                WelfordMerge(mean, m2, count, mean_other, m2_other, count_other);
            }
        });
    }
    
    // Block-level Welford reduction
    CK_TILE_DEVICE static void BlockWelfordReduce(float& mean,
                                                  float& m2,
                                                  index_t& count,
                                                  void* smem)
    {
        // Step 1: Warp-level reduction
        WarpWelfordReduce(mean, m2, count);
        
        // Step 2: Cross-warp reduction if needed
        if constexpr(NumWarps > 1)
        {
            float* smem_mean = static_cast<float*>(smem);
            float* smem_m2 = smem_mean + NumWarps;
            index_t* smem_count = reinterpret_cast<index_t*>(smem_m2 + NumWarps);
            
            const index_t warp_id = get_thread_local_1d_id() / WarpSize;
            const index_t lane_id = get_lane_id();
            
            // First thread in each warp writes to LDS
            if(lane_id == 0)
            {
                smem_mean[warp_id] = mean;
                smem_m2[warp_id] = m2;
                smem_count[warp_id] = count;
            }
            
            block_sync_lds();
            
            // First warp reduces across warp results
            if(warp_id == 0 && lane_id < NumWarps)
            {
                mean = smem_mean[lane_id];
                m2 = smem_m2[lane_id];
                count = smem_count[lane_id];
                
                // Reduce within first warp
                static_for<0, integer_log2_floor(NumWarps), 1>{}([&](auto I) {
                    constexpr index_t Stride = NumWarps >> (I + 1);
                    
                    if constexpr(Stride > 0)
                    {
                        const float mean_other = warp_shuffle_down(mean, Stride);
                        const float m2_other = warp_shuffle_down(m2, Stride);
                        const index_t count_other = warp_shuffle_down(count, Stride);
                        
                        if(lane_id < Stride)
                        {
                            WelfordMerge(mean, m2, count, mean_other, m2_other, count_other);
                        }
                    }
                });
                
                // Write final result
                if(lane_id == 0)
                {
                    smem_mean[0] = mean;
                    smem_m2[0] = m2;
                    smem_count[0] = count;
                }
            }
            
            block_sync_lds();
            
            // All threads read final result
            mean = smem_mean[0];
            m2 = smem_m2[0];
            count = smem_count[0];
        }
    }
    
    CK_TILE_DEVICE void operator()(Kargs kargs, void* smem) const
    {
        const index_t channel_id = get_block_1d_id();
        const index_t tid = get_thread_local_1d_id();
        
        // Each block processes one channel
        const float* p_x_channel = kargs.p_x + channel_id * kargs.k;
        float* p_y_channel = kargs.p_y + channel_id * kargs.k;
        
        // Load scale and bias for this channel (if provided)
        const float scale = (kargs.p_scale != nullptr) ? kargs.p_scale[channel_id] : 1.0f;
        const float bias = (kargs.p_bias != nullptr) ? kargs.p_bias[channel_id] : 0.0f;
        
        // Initialize Welford statistics
        float mean = 0.0f;
        float m2 = 0.0f;
        index_t count = 0;
        
        // Each thread loads one element and initializes Welford
        if(tid < kargs.k)
        {
            const float x = p_x_channel[tid];
            mean = x;
            m2 = 0.0f;
            count = 1;
        }
        
        // Block-level Welford reduction
        BlockWelfordReduce(mean, m2, count, smem);
        
        // Compute variance from M2
        const float variance = (count > 0) ? m2 / static_cast<float>(count) : 0.0f;
        
        // Compute inverse standard deviation
        const float inv_std = 1.0f / sqrtf(variance + kargs.epsilon);
        
        // Normalize and write output
        if(tid < kargs.k)
        {
            const float x = p_x_channel[tid];
            const float y = scale * (x - mean) * inv_std + bias;
            p_y_channel[tid] = y;
        }
    }
};

// Kernel wrapper
template <index_t BlockSize>
__global__ void kernel_batchnorm2d_simple(typename BatchNorm2dSimple<BlockSize>::Kargs kargs)
{
    __shared__ char smem[BatchNorm2dSimple<BlockSize>::GetSmemSize()];
    BatchNorm2dSimple<BlockSize>{}(kargs, smem);
}

} // namespace ck_tile
