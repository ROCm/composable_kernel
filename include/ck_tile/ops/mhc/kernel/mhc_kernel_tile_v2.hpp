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
          index_t B_       = 16, // Batch size (compile-time)
          index_t N_       = 4,  // Expansion factor (compile-time)
          index_t C_       = 64, // Channels per stream (compile-time)
          index_t KTile_   = 256>  // K-tile size for shared memory (compile-time)
struct ManifoldConstrainedHyperConnectionTiled
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using PhiDataType     = ck_tile::remove_cvref_t<typename Problem::PhiDataType>;

    static constexpr index_t kB         = B_;               // Batch size (compile-time)
    static constexpr index_t kN         = N_;               // Expansion factor (compile-time)
    static constexpr index_t kC         = C_;               // Channels per stream (compile-time)
    static constexpr index_t kNC        = kN * kC;          // Input dimension (compile-time)
    static constexpr index_t kOutputDim = 2 * kN + kN * kN; // Output dimension (compile-time)
    static constexpr index_t kKTile     = KTile_;           // K-tile size (compile-time)

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // Shared memory is now bounded by kKTile instead of kNC
        // This allows handling arbitrary C values
        return kKTile * sizeof(ComputeDataType);
    }

    CK_TILE_DEVICE void operator()(const XDataType* p_x,     // [B, nC] - input tensor
                                   const PhiDataType* p_phi, // [nC, 2n+n²] - packed weight matrices
                                   YDataType* p_output,      // [B, 2n+n²] - output tensor
                                   float r          = 1.0f,  // scaling factor
                                   float alpha_pre  = 1.0f,  // scaling for H^{pre}
                                   float alpha_post = 1.0f,  // scaling for H^{post}
                                   float alpha_res  = 1.0f,  // scaling for H^{res}
                                   float bias       = 0.0f) const  // bias term
    {
        constexpr index_t nC         = kNC;        // Compile-time!
        constexpr index_t output_dim = kOutputDim; // Compile-time!
        constexpr index_t B          = kB;         // Compile-time!

        // NEW PARALLELIZATION STRATEGY:
        // Each block processes 16 output columns starting at stream_id * 16
        // block_id corresponds to which group of 16 columns we're computing
        const index_t block_id  = get_block_id();
        const index_t stream_id = block_id * 16; // Starting column for this block
        const index_t tid       = get_thread_id();

        // Early exit if this block is beyond the output dimensions
        if(stream_id >= output_dim)
        {
            return;
        }

        // Calculate number of batch tile iterations needed
        constexpr index_t kBatchTile  = 16; // Process 16 batches per tile
        const index_t num_batch_tiles = (B + kBatchTile - 1) / kBatchTile;

        // Calculate number of K-tile iterations needed for large C values
        // This allows us to handle arbitrary C by processing K in chunks
        constexpr index_t num_ktile_iterations = (kNC + kKTile - 1) / kKTile;

        // With expansion-parallel strategy + K-tiling:
        // - Grid size = output_dim (one block per output column)
        // - Each block computes output[:, stream_id] for ALL batches
        // - GEMM becomes: x[B, nC] * phi[nC, 1] = output[B, 1]
        // - K dimension is tiled to fit in shared memory

        // Step 1: Allocate LDS for x - bounded by kKTile instead of kNC
        // For BlockGemm, we need x[kBatchTile, kKTile] in LDS
        __shared__ XDataType
            x_lds[kBatchTile * kKTile]; // Allocate for 16 batches × kKTile elements

        // Step 2: Create phi infrastructure in LDS - bounded by kKTile
        // For this stream, we need phi[:, stream_id:stream_id+16] which is [kKTile, 16]
        // IMPORTANT: BlockGemm expects B matrix in K-major (column-major) layout!
        constexpr index_t kKPack = 16; // Pack size for K dimension
        __shared__ PhiDataType phi_lds[kKTile * 16];

        using BlockGemm = BlockGemmASmemBSmemCRegV1<Problem, Policy>;

        // Step 3: Iterate over batch tiles
        for(index_t batch_tile_idx = 0; batch_tile_idx < num_batch_tiles; batch_tile_idx++)
        {
            // Calculate batch range for this tile
            const index_t batch_start         = batch_tile_idx * kBatchTile;
            const index_t batch_end           = min(batch_start + kBatchTile, B);
            const index_t current_batch_count = batch_end - batch_start;

            // Step 3a: Initialize result tile to zero for this batch tile
            auto result_tile = BlockGemm::MakeCBlockTile();
            set_tile(result_tile, 0.0f);

            // Step 3b: Iterate over K-tiles (outer loop for large C values)
            for(index_t ktile_idx = 0; ktile_idx < num_ktile_iterations; ktile_idx++)
            {
                // Calculate K range for this tile
                const index_t k_start       = ktile_idx * kKTile;
                const index_t k_end         = min(k_start + kKTile, nC);
                const index_t current_k_len = k_end - k_start;

                // Step 3b-i: Load x from global to LDS for this batch tile and K-tile
                for(index_t i = tid; i < kBatchTile * kKTile; i += get_block_size())
                {
                    index_t local_batch_idx  = i / kKTile;
                    index_t local_k_idx      = i % kKTile;
                    index_t global_batch_idx = batch_start + local_batch_idx;
                    index_t global_k_idx     = k_start + local_k_idx;

                    if(local_batch_idx < current_batch_count && local_k_idx < current_k_len)
                    {
                        x_lds[i] = p_x[global_batch_idx * nC + global_k_idx];
                    }
                    else
                    {
                        x_lds[i] = 0; // Pad with zeros for out-of-bounds
                    }
                }

                // Step 3b-ii: Load phi from global to LDS for this K-tile
                // Load with K-major layout for optimal BlockGemm performance
                // Layout: [K_outer, N, K_inner] where K_outer * K_inner = kKTile
                for(index_t i = tid; i < kKTile * 16; i += get_block_size())
                {
                    // Decode linear index for K-major layout
                    index_t k_outer_local = i / (16 * kKPack);
                    index_t remainder     = i % (16 * kKPack);
                    index_t n_idx         = remainder / kKPack;
                    index_t k_inner       = remainder % kKPack;

                    index_t local_k  = k_outer_local * kKPack + k_inner;
                    index_t global_k = k_start + local_k;
                    index_t global_n = stream_id + n_idx;

                    if(local_k < current_k_len && global_n < output_dim)
                    {
                        phi_lds[i] = p_phi[global_k * output_dim + global_n];
                    }
                    else
                    {
                        phi_lds[i] = 0; // Pad
                    }
                }
                block_sync_lds();

                // Step 3b-iii: Create LDS tensor views for this K-tile
                const auto x_lds_tensor = make_naive_tensor_view<address_space_enum::lds>(
                    x_lds,
                    make_tuple(number<kBatchTile>{}, number<kKTile>{}),
                    make_tuple(number<kKTile>{}, number<1>{}),
                    number<1>{},
                    number<1>{});

                // Create phi tensor view with K-major layout
                constexpr index_t kKOuter_tile = (kKTile + kKPack - 1) / kKPack;
                const auto phi_lds_tensor_3d   = make_naive_tensor_view<address_space_enum::lds>(
                    phi_lds,
                    make_tuple(number<kKOuter_tile>{}, number<16>{}, number<kKPack>{}),
                    make_tuple(number<16 * kKPack>{}, number<kKPack>{}, number<1>{}),
                    number<kKPack>{},
                    number<1>{});

                const auto phi_lds_tensor = transform_tensor_view(
                    phi_lds_tensor_3d,
                    make_tuple(
                        make_pass_through_transform(number<16>{}),
                        make_merge_transform(make_tuple(number<kKOuter_tile>{}, number<kKPack>{}))),
                    make_tuple(sequence<1>{}, sequence<0, 2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                // Step 3b-iv: Create tile windows
                auto x_lds_window =
                    make_tile_window(x_lds_tensor, make_tuple(number<16>{}, number<16>{}), {0, 0});

                auto phi_lds_window = make_tile_window(
                    phi_lds_tensor, make_tuple(number<16>{}, number<16>{}), {0, 0});

                // Step 3b-v: Iterate over 16x16 tiles within this K-tile
                constexpr index_t num_inner_k_tiles = (kKTile + 15) / 16;
                for(index_t inner_k_tile = 0; inner_k_tile < num_inner_k_tiles; inner_k_tile++)
                {
                    if(inner_k_tile > 0)
                    {
                        move_tile_window(x_lds_window, {0, 16});
                        move_tile_window(phi_lds_window, {16, 0});
                    }

                    // Accumulate: result_tile += x_lds_window * phi_lds_window
                    BlockGemm{}(result_tile, x_lds_window, phi_lds_window);
                }

                block_sync_lds();
            } // End K-tile loop

            // Step 3h & 3i: Apply elementwise operations and store result_tile to output
            // We need to apply different alpha values based on which output column each element
            // belongs to Since result_tile contains columns [stream_id, stream_id+16), we apply
            // alpha during store
            constexpr auto result_spans = decltype(result_tile)::get_distributed_spans();

            sweep_tile_span(result_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(result_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        result_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                    const index_t i_idx = tile_idx.at(number<0>{});
                    const index_t j_idx = tile_idx.at(number<1>{});

                    const index_t global_batch = batch_start + i_idx;
                    const index_t global_col   = stream_id + j_idx;

                    if(global_batch < B && global_col < output_dim)
                    {
                        // Determine alpha based on the actual output column
                        float alpha = (global_col < kN)       ? alpha_pre
                                      : (global_col < 2 * kN) ? alpha_post
                                                              : alpha_res;

                        // Apply scaling and bias, then store: result = (alpha / r) * result + bias
                        constexpr auto i_j_idx   = make_tuple(idx0, idx1);
                        const index_t global_idx = global_batch * output_dim + global_col;
                        p_output[global_idx] =
                            type_convert<YDataType>((alpha / r) * result_tile[i_j_idx] + bias);
                    }
                });
            });

            // Synchronize before next batch tile iteration
            block_sync_lds();
        } // End batch tile loop
    }
};

} // namespace ck_tile
