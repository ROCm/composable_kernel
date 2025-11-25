// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// BlockWelford: Performs Welford's algorithm for online mean and variance computation
// This is adapted from the old CK implementation but using ck_tile patterns
// 
// Welford's algorithm allows computing mean and variance in a single pass:
// For combining two sets of statistics:
//   count_new = count_a + count_b
//   delta = mean_b - mean_a
//   mean_new = mean_a + delta * (count_b / count_new)
//   M2_new = M2_a + M2_b + delta^2 * count_a * count_b / count_new
//   variance = M2 / count
template <typename ComputeDataType_>
struct BlockWelford
{
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;

    // Merge two sets of Welford statistics
    // mean_a, var_a (actually M2_a), count_a will be updated
    template <typename CountDataType>
    CK_TILE_DEVICE static void Merge(ComputeDataType& mean_a,
                                     ComputeDataType& m2_a, // Note: This is M2, not variance yet
                                     CountDataType& count_a,
                                     ComputeDataType mean_b,
                                     ComputeDataType m2_b,
                                     CountDataType count_b)
    {
        CountDataType count_new = count_a + count_b;
        
        if(count_new == 0)
        {
            mean_a = type_convert<ComputeDataType>(0);
            m2_a = type_convert<ComputeDataType>(0);
            count_a = 0;
            return;
        }

        ComputeDataType count_b_over_count = 
            type_convert<ComputeDataType>(count_b) / type_convert<ComputeDataType>(count_new);
        ComputeDataType delta = mean_b - mean_a;
        
        mean_a = mean_a + delta * count_b_over_count;
        m2_a = m2_a + m2_b + delta * delta * 
               type_convert<ComputeDataType>(count_a) * count_b_over_count;
        count_a = count_new;
    }

    // Block-level reduction using Welford algorithm
    // Performs tree-reduction across threads within a block
    // Input: per-thread mean, m2 (sum of squared differences), and count
    // Output: reduced mean and variance for the block
    template <typename CountDataType, index_t BlockSize>
    CK_TILE_DEVICE static void Run(ComputeDataType& mean_value,
                                   ComputeDataType& var_value, // Will be converted from M2 to variance
                                   CountDataType& count,
                                   void* smem_ptr)
    {
        // Use shared memory for reduction
        auto* mean_smem = reinterpret_cast<ComputeDataType*>(smem_ptr);
        auto* m2_smem = reinterpret_cast<ComputeDataType*>(
            reinterpret_cast<char*>(smem_ptr) + BlockSize * sizeof(ComputeDataType));
        auto* count_smem = reinterpret_cast<CountDataType*>(
            reinterpret_cast<char*>(smem_ptr) + 2 * BlockSize * sizeof(ComputeDataType));

        const index_t thread_id = get_thread_id();

        // Store thread-local values to shared memory
        mean_smem[thread_id] = mean_value;
        m2_smem[thread_id] = var_value; // Note: This is M2, not variance yet
        count_smem[thread_id] = count;

        block_sync_lds();

        // Tree reduction
        index_t active_threads = BlockSize;
        while(active_threads > 1)
        {
            active_threads = (active_threads + 1) / 2;
            
            if(thread_id < active_threads)
            {
                index_t partner_id = thread_id + active_threads;
                
                if(partner_id < BlockSize)
                {
                    ComputeDataType mean_a = mean_smem[thread_id];
                    ComputeDataType m2_a = m2_smem[thread_id];
                    CountDataType count_a = count_smem[thread_id];
                    
                    ComputeDataType mean_b = mean_smem[partner_id];
                    ComputeDataType m2_b = m2_smem[partner_id];
                    CountDataType count_b = count_smem[partner_id];
                    
                    Merge(mean_a, m2_a, count_a, mean_b, m2_b, count_b);
                    
                    mean_smem[thread_id] = mean_a;
                    m2_smem[thread_id] = m2_a;
                    count_smem[thread_id] = count_a;
                }
            }
            
            block_sync_lds();
        }

        // Broadcast result to all threads
        mean_value = mean_smem[0];
        count = count_smem[0];
        
        // Convert M2 to variance: variance = M2 / count
        if(count > 0)
        {
            var_value = m2_smem[0] / type_convert<ComputeDataType>(count);
        }
        else
        {
            var_value = type_convert<ComputeDataType>(0);
        }
    }

    // Calculate required shared memory size
    template <typename CountDataType, index_t BlockSize>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // Need space for: mean_buffer + m2_buffer + count_buffer
        return BlockSize * sizeof(ComputeDataType) +  // mean
               BlockSize * sizeof(ComputeDataType) +  // m2
               BlockSize * sizeof(CountDataType);     // count
    }
};

} // namespace ck_tile
