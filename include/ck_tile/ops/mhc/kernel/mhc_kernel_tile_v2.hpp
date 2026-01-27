// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_default_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"

// Manifold Constrained Hyper Connection Kernel (True CK Tile Version):
// =====================================================================
// This implementation uses proper CK tile approach with:
// - Tile windows for input/output
// - load_tile/store_tile operations
// - Distributed tensors
// - Tiling across both batch and vector dimensions

namespace ck_tile {

template <typename Problem_,
          typename Policy_ = MHCDefaultPolicy,
          index_t N_ = 4>  // Template parameter for expansion factor (compile-time)
struct ManifoldConstrainedHyperConnectionTiled
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using PhiDataType     = ck_tile::remove_cvref_t<typename Problem::PhiDataType>;

    static constexpr index_t kN = N_;  // Expansion factor (compile-time)
    static constexpr index_t kOutputDim = 2 * kN + kN * kN;  // 2n + n² (compile-time)
    
    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // Need shared memory for reduction
        return 256 * sizeof(ComputeDataType);
    }

    CK_TILE_DEVICE void operator()(
        const XDataType* p_x,        // [B, nC] - input tensor
        const PhiDataType* p_phi,    // [nC, 2n+n²] - packed weight matrices
        YDataType* p_output,         // [B, 2n+n²] - output tensor
        [[maybe_unused]] int B,
        int C,                       // output layer dimension (runtime, potentially large)
        float r          = 1.0f,     // scaling factor
        float alpha_pre  = 1.0f,     // scaling for H^{pre}
        float alpha_post = 1.0f,     // scaling for H^{post}
        float alpha_res  = 1.0f,     // scaling for H^{res}
        float bias       = 0.0f) const // bias term
    {
        const index_t nC = kN * C;  // Use compile-time kN
        constexpr index_t output_dim = kOutputDim;  // Now compile-time!
        
        // NEW PARALLELIZATION STRATEGY:
        // Each block processes 16 output columns starting at stream_id * 16
        // block_id corresponds to which group of 16 columns we're computing
        const index_t block_id = get_block_id();
        const index_t stream_id = block_id * 16;  // Starting column for this block
        const index_t tid = get_thread_id();
        
        // Early exit if this block is beyond the output dimensions
        if (stream_id >= output_dim)
        {
            return;
        }
        
        // With expansion-parallel strategy:
        // - Grid size = output_dim (one block per output column)
        // - Each block computes output[:, stream_id] for ALL batches
        // - GEMM becomes: x[B, nC] * phi[nC, 1] = output[B, 1]
        // - This gives us M=B (e.g., 64), K=nC (e.g., 1024), N=1
        
        // Step 1: Allocate LDS for x - need to load all batches
        // For BlockGemm, we need x[B, nC] in LDS
        // Start with a tile: load 16 batches at a time (matching BlockGemmShape kM=16)
        constexpr index_t kBatchTile = 16;  // Process 16 batches per tile
        __shared__ XDataType x_lds[kBatchTile * 256];  // Allocate for 16 batches × 256 elements
        
        // Step 2: Load x from global to LDS
        // Load up to kBatchTile batches, each with nC elements (but limit to 256 per batch for now)
        for (index_t i = tid; i < kBatchTile * 256; i += get_block_size())
        {
            index_t batch_idx = i / 256;
            index_t elem_idx = i % 256;
            if (batch_idx < B && elem_idx < nC)
            {
                x_lds[i] = p_x[batch_idx * nC + elem_idx];
            }
        }
        block_sync_lds();
        
        // Step 3: Create tensor descriptor for x in LDS
        // x_lds contains [kBatchTile, 256] elements
        [[maybe_unused]] const auto x_lds_desc = make_naive_tensor_descriptor(
            make_tuple(number<kBatchTile>{}, number<256>{}),  // lengths - compile-time!
            make_tuple(number<256>{}, number<1>{}),           // strides - row-major
            number<1>{},                                       // vector length
            number<1>{});                                      // vector stride
        
        // Step 4: Create LDS tensor view for x
        [[maybe_unused]] const auto x_lds_tensor = make_naive_tensor_view<address_space_enum::lds>(
            x_lds,
            make_tuple(number<kBatchTile>{}, number<256>{}),
            make_tuple(number<256>{}, number<1>{}),
            number<1>{},
            number<1>{});
        
        // Step 5: Create tile window for x in LDS
        // BlockGemm expects [kM, kK] = [16, 16]
        // We'll iterate over the K dimension (256 / 16 = 16 iterations)
        [[maybe_unused]] auto x_lds_window = make_tile_window(
            x_lds_tensor,
            make_tuple(number<16>{}, number<16>{}),  // [M=16, K=16] to match BlockGemmShape
            {0, 0});  // origin
        
        // Step 6: Create phi infrastructure in LDS
        // For this stream, we need phi[:, stream_id:stream_id+16] which is [nC, 16]
        // IMPORTANT: BlockGemm expects B matrix in K-major (column-major) layout!
        // So phi_lds should be organized as [K_outer, N, K_inner] = [256/16, 16, 16]
        // with linear index: k_outer * (16 * 16) + n * 16 + k_inner
        // This way, elements in the same column are contiguous (or nearly so with padding)
        constexpr index_t kKPack = 16;  // Pack size for K dimension
        __shared__ PhiDataType phi_lds[256 * 16];
        
        // Load with K-major layout: iterate over K_outer, N, K_inner
        for (index_t i = tid; i < 256 * 16; i += get_block_size())
        {
            // Decode linear index for K-major layout
            index_t k_outer = i / (16 * kKPack);      // 0-15 (256/16)
            index_t remainder = i % (16 * kKPack);
            index_t n = remainder / kKPack;            // 0-15 (N dimension)
            index_t k_inner = remainder % kKPack;      // 0-15 (K pack)
            
            index_t global_k = k_outer * kKPack + k_inner;  // Actual K index (0-255)
            index_t global_n = stream_id + n;               // Actual N index
            
            if (global_k < nC && global_n < output_dim)
            {
                // Load phi[global_k, global_n] from global memory
                phi_lds[i] = p_phi[global_k * output_dim + global_n];
            }
            else
            {
                phi_lds[i] = 0;  // Pad
            }
        }
        block_sync_lds();
        
        // Create phi tensor view with K-major layout
        // Layout is [K_outer, N, K_inner] = [16, 16, 16] with appropriate strides
        constexpr index_t kKOuter = 256 / kKPack;  // 16
        const auto phi_lds_tensor_3d = make_naive_tensor_view<address_space_enum::lds>(
            phi_lds,
            make_tuple(number<kKOuter>{}, number<16>{}, number<kKPack>{}),
            make_tuple(number<16 * kKPack>{}, number<kKPack>{}, number<1>{}),
            number<kKPack>{},
            number<1>{});
        
        // Transform to 2D [K, N] by merging K_outer and K_inner
        const auto phi_lds_tensor = transform_tensor_view(
            phi_lds_tensor_3d,
            make_tuple(make_pass_through_transform(number<16>{}),
                       make_merge_transform(make_tuple(number<kKOuter>{}, number<kKPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
        
        // B window should be [kK, kN] = [16, 16]
        [[maybe_unused]] auto phi_lds_window = make_tile_window(
            phi_lds_tensor,
            make_tuple(number<16>{}, number<16>{}),  // [K=16, N=16] to match BlockGemmShape
            {0, 0});
        
        // Step 7: Call BlockGemm and iterate over K dimension
        // We need to accumulate over K: nC=256, but BlockGemm processes K=16 at a time
        // So we need 256/16 = 16 iterations
        using BlockGemm = BlockGemmASmemBSmemCRegV1<Problem, Policy>;
        
        // Initialize result tile to zero
        auto result_tile = BlockGemm::MakeCBlockTile();
        set_tile(result_tile, 0.0f);
        
        // Iterate over K dimension: nC=256, BlockGemm processes K=16 at a time
        constexpr index_t num_k_tiles = 256 / 16;  // 16 iterations for K=256
        for (index_t k_tile = 0; k_tile < num_k_tiles; k_tile++)
        {
            // Move windows to next K tile
            if (k_tile > 0)
            {
                move_tile_window(x_lds_window, {0, 16});    // Move K dimension
                move_tile_window(phi_lds_window, {16, 0});  // Move K dimension
            }
            
            // Accumulate: result_tile += x_lds_window * phi_lds_window
            BlockGemm{}(result_tile, x_lds_window, phi_lds_window);
        }
        
        // Step 8: Compute norm ||x_l||_2 / sqrt(nC) for each batch
        // We need this for potential normalization in step 3
        // Allocate shared memory for norm computation
        __shared__ ComputeDataType norm_shared[kBatchTile];
        
        // Compute norm for each batch in parallel
        // Each batch's norm is computed by all threads, then reduced
        for (index_t batch_idx = 0; batch_idx < kBatchTile && batch_idx < B; batch_idx++)
        {
            ComputeDataType local_sum = 0.0f;
            
            // Each thread accumulates part of the sum of squares
            for (index_t k = tid; k < nC && k < 256; k += get_block_size())
            {
                ComputeDataType val = type_convert<ComputeDataType>(x_lds[batch_idx * 256 + k]);
                local_sum += val * val;
            }
            
            // Simple reduction (can be optimized with block_reduce)
            // For now, use atomic add to shared memory
            if (tid == 0)
            {
                norm_shared[batch_idx] = 0;
            }
            block_sync_lds();
            
            // Accumulate (simplified - should use proper reduction)
            if (local_sum > 0)
            {
                atomicAdd(&norm_shared[batch_idx], local_sum);
            }
            block_sync_lds();
            
            // Compute final norm
            if (tid == 0)
            {
                norm_shared[batch_idx] = sqrt(norm_shared[batch_idx]) / 
                                        sqrt(type_convert<ComputeDataType>(nC));
            }
            block_sync_lds();
        }
        
        
        // Step 9: Apply elementwise operations to result_tile using tile_elementwise_inout
        // Determine which section this stream belongs to and get alpha
        float alpha = (stream_id < kN) ? alpha_pre : 
                      (stream_id < 2 * kN) ? alpha_post : alpha_res;
        
        // Apply scaling: result = (alpha / r) * result + bias
        tile_elementwise_inout([&](auto& val) {
            val = (alpha / r) * val + bias;
        }, result_tile);
        
        // Step 10: Create output tensor view and tile window, then store
        // Create full output tensor view [B, output_dim]
        auto output_tensor_view = make_naive_tensor_view<address_space_enum::global>(
            p_output,
            make_tuple(B, output_dim),     // Full output shape
            make_tuple(output_dim, 1),     // Row-major strides
            number<1>{},
            number<1>{});
        
        // Create tile window at the correct position for this stream
        // We want to write to output[:, stream_id:stream_id+16]
        auto output_window = make_tile_window(
            output_tensor_view,
            make_tuple(number<16>{}, number<16>{}),  // [16, 16] to match result_tile
            {0, stream_id},  // Start at row 0, column stream_id
            result_tile.get_tile_distribution());  // Use same distribution as result_tile
        
        // Store result_tile to output using store_tile!
        store_tile(output_window, cast_tile<YDataType>(result_tile));
    }
};

} // namespace ck_tile
