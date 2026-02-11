// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_default_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

// Manifold Constrained Hyper Connection Kernel V4:
// =====================================================================
// Optimizations implemented:
// - Remove GEMM pipeline to avoid redundant global memory reads
// - Use BlockGemm directly with manual LDS management (like v2)
// - Compute normalization incrementally during GEMM loop
// - Single pass through input data: load once, compute norm and GEMM together

namespace ck_tile {

template <typename Problem_,
          typename Policy_     = MHCDefaultPolicy,
          typename Activation_ = element_wise::Sigmoid>
struct MHCKernelV4
{
    using Activation = ck_tile::remove_cvref_t<Activation_>;
    using Problem    = ck_tile::remove_cvref_t<Problem_>;
    using Policy     = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using PhiDataType     = ck_tile::remove_cvref_t<typename Problem::PhiDataType>;

    // Automatically derive tile sizes from BlockGemmShape (single source of truth!)
    static constexpr index_t kMTile = Problem::BlockGemmShape::kM; // Batch tile
    static constexpr index_t kNTile = Problem::BlockGemmShape::kN; // Output tile
    static constexpr index_t kKTile = Problem::BlockGemmShape::kK; // K tile for C dimension

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // LDS for BlockGemm: A[kMTile, kKTile] + B[kKTile, kNTile]
        constexpr index_t a_lds_size = kMTile * kKTile * sizeof(XDataType);
        constexpr index_t b_lds_size = kKTile * kNTile * sizeof(PhiDataType);
        return a_lds_size + b_lds_size;
    }

    // Grid configuration: 1D grid over batch dimension only
    // Each block processes full output dimension (kNTile=32 covers output_dim=24)
    CK_TILE_HOST static constexpr auto GetGridSize(index_t batch,
                                                   [[maybe_unused]] index_t output_dim)
    {
        const index_t grid_m = (batch + kMTile - 1) / kMTile;
        return make_tuple(grid_m, 1); // 1D grid: only tile in M dimension
    }

    CK_TILE_DEVICE void operator()(const XDataType* p_x,
                                   const PhiDataType* p_phi,
                                   YDataType* p_output,
                                   index_t batch,
                                   index_t nC,
                                   index_t output_dim,
                                   [[maybe_unused]] index_t n,
                                   [[maybe_unused]] float r          = 1.0f,
                                   [[maybe_unused]] float alpha_pre  = 1.0f,
                                   [[maybe_unused]] float alpha_post = 1.0f,
                                   [[maybe_unused]] float alpha_res  = 1.0f,
                                   [[maybe_unused]] float bias       = 0.0f) const
    {
        // 1D block indexing: only tile in M (batch) dimension
        const index_t block_id    = get_block_id();
        const index_t batch_start = block_id * kMTile;
        const index_t out_start   = 0; // Always process from output index 0

        if(batch_start >= batch)
            return;

        // Allocate shared memory for A and B tiles
        __shared__ char smem_ptr[GetSmemSize()];
        XDataType* x_lds = reinterpret_cast<XDataType*>(smem_ptr);
        PhiDataType* phi_lds =
            reinterpret_cast<PhiDataType*>(smem_ptr + kMTile * kKTile * sizeof(XDataType));

        // Create BlockGemm instance and result tile (distributed tensor in registers)
        using BlockGemm  = BlockGemmASmemBSmemCRegV1<Problem, Policy>;
        auto result_tile = BlockGemm::MakeCBlockTile();
        set_tile(result_tile, 0.0f);

        // Number of K-tile iterations
        const index_t num_k_tiles = (nC + kKTile - 1) / kKTile;

        // Create tensor views for X and Phi
        auto x_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(batch, nC), make_tuple(nC, 1), number<1>{}, number<1>{});

        auto x_tensor_padded = pad_tensor_view(x_tensor_full,
                                               make_tuple(number<kMTile>{}, number<kKTile>{}),
                                               sequence<false, Problem::kPadK>{});

        // Create X DRAM window with tile distribution for vectorized loading
        constexpr auto x_load_tile_dist = Problem::MakeXLoadTileDistribution();
        auto x_dram_window              = make_tile_window(x_tensor_padded,
                                              make_tuple(number<kMTile>{}, number<kKTile>{}),
                                                           {batch_start, 0},
                                              x_load_tile_dist);

        // Create X LDS tensor view and window
        auto x_lds_tensor = make_naive_tensor_view<address_space_enum::lds>(
            x_lds,
            make_tuple(number<kMTile>{}, number<kKTile>{}),
            make_tuple(number<kKTile>{}, number<1>{}),
            number<1>{},
            number<1>{});

        auto x_lds_window =
            make_tile_window(x_lds_tensor, make_tuple(number<kMTile>{}, number<kKTile>{}), {0, 0});

        // Create Phi tensor view and window with tile distribution
        auto phi_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_phi, make_tuple(output_dim, nC), make_tuple(1, output_dim), number<1>{}, number<1>{});

        auto phi_tensor_padded = pad_tensor_view(phi_tensor_full,
                                                 make_tuple(number<kNTile>{}, number<kKTile>{}),
                                                 sequence<false, Problem::kPadK>{});

        constexpr auto phi_load_tile_dist = Problem::MakePhiLoadTileDistribution();
        auto phi_dram_window              = make_tile_window(phi_tensor_padded,
                                                make_tuple(number<kNTile>{}, number<kKTile>{}),
                                                             {out_start, 0},
                                                phi_load_tile_dist);

        // Create Phi LDS tensor view and window
        auto phi_lds_tensor = make_naive_tensor_view<address_space_enum::lds>(
            phi_lds,
            make_tuple(number<kNTile>{}, number<kKTile>{}),
            make_tuple(number<kKTile>{}, number<1>{}),
            number<1>{},
            number<1>{});

        auto phi_lds_window = make_tile_window(
            phi_lds_tensor, make_tuple(number<kNTile>{}, number<kKTile>{}), {0, 0});

        // Compute norms BEFORE GEMM loop with ALL threads participating
        // With kMTile=16 and kBlockSize=64: 4 threads per batch element
        const index_t thread_id = get_thread_id();
        constexpr index_t threads_per_row =
            kBlockSize / kMTile; // 64/16 = 4 threads per batch element
        const index_t row_id        = thread_id / threads_per_row; // Which batch element (0-15)
        const index_t thread_in_row = thread_id % threads_per_row; // Which thread within row (0-3)

        __shared__ ComputeDataType norm_reduction[kMTile][threads_per_row];

        // Each thread computes partial sum for its assigned portion
        if(row_id < kMTile)
        {
            const index_t global_m      = batch_start + row_id;
            ComputeDataType partial_sum = 0.0f;

            if(global_m < batch)
            {
                const XDataType* row_ptr = p_x + global_m * nC;

                // Divide work: each thread processes every threads_per_row elements
                constexpr index_t kVecSize = 4;
                for(index_t k = thread_in_row * kVecSize; k < nC; k += threads_per_row * kVecSize)
                {
                    if(k + kVecSize <= nC)
                    {
                        using VecType = ext_vector_t<XDataType, kVecSize>;
                        VecType vec   = *c_style_pointer_cast<const VecType*>(row_ptr + k);

#pragma unroll
                        for(index_t i = 0; i < kVecSize; ++i)
                        {
                            ComputeDataType val = type_convert<ComputeDataType>(vec[i]);
                            partial_sum += val * val;
                        }
                    }
                    else
                    {
                        // Handle remainder elements
                        for(index_t i = 0; i < kVecSize && k + i < nC; ++i)
                        {
                            ComputeDataType val = type_convert<ComputeDataType>(row_ptr[k + i]);
                            partial_sum += val * val;
                        }
                    }
                }
            }

            norm_reduction[row_id][thread_in_row] = partial_sum;
        }

        block_sync_lds();

        // Reduce within each row and compute final norms
        ComputeDataType norms[kMTile];
        const ComputeDataType sqrt_nC = ck_tile::sqrt(static_cast<ComputeDataType>(nC));

        if(thread_in_row == 0 && row_id < kMTile)
        {
            const index_t global_m = batch_start + row_id;

            if(global_m < batch)
            {
                ComputeDataType sum_squares = 0.0f;
#pragma unroll
                for(index_t t = 0; t < threads_per_row; ++t)
                {
                    sum_squares += norm_reduction[row_id][t];
                }

                ComputeDataType norm = ck_tile::sqrt(sum_squares) / sqrt_nC;
                norms[row_id]        = (norm > 1e-12f) ? norm : 1.0f;
            }
            else
            {
                norms[row_id] = 1.0f;
            }
        }

        // Broadcast norms to all threads
        __shared__ ComputeDataType norms_shared[kMTile];
        if(thread_in_row == 0 && row_id < kMTile)
        {
            norms_shared[row_id] = norms[row_id];
        }
        block_sync_lds();

        for(index_t local_m = 0; local_m < kMTile; ++local_m)
        {
            norms[local_m] = norms_shared[local_m];
        }

        // Main GEMM loop: load tiles with vectorization and accumulate GEMM
        for(index_t k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx)
        {
            // Load X tile using vectorized load_tile
            auto x_tile = make_static_distributed_tensor<XDataType>(x_load_tile_dist);
            load_tile(x_tile, x_dram_window);

            // Store X tile to LDS
            store_tile(x_lds_window, x_tile);

            // Load Phi tile using vectorized load_tile
            auto phi_tile = make_static_distributed_tensor<PhiDataType>(phi_load_tile_dist);
            load_tile(phi_tile, phi_dram_window);

            // Store Phi tile to LDS
            store_tile(phi_lds_window, phi_tile);

            block_sync_lds();

            // Move windows for next iteration
            move_tile_window(x_dram_window, {0, kKTile});
            move_tile_window(phi_dram_window, {0, kKTile});

            // Perform GEMM using BlockGemm with MFMA: result_tile += x_lds * phi_lds^T
            BlockGemm{}(result_tile, x_lds_window, phi_lds_window);

            block_sync_lds();
        }

        // Apply normalization and activation in-place on result_tile
        constexpr auto result_spans = decltype(result_tile)::get_distributed_spans();
        sweep_tile_span(result_spans[number<0>{}], [&](auto idx0) {
            sweep_tile_span(result_spans[number<1>{}], [&](auto idx1) {
                const auto tile_idx = get_x_indices_from_distributed_indices(
                    result_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                const index_t local_m  = tile_idx.at(number<0>{});
                const index_t local_n  = tile_idx.at(number<1>{});
                const index_t global_m = batch_start + local_m;
                const index_t global_n = out_start + local_n;

                if(global_m < batch && global_n < output_dim)
                {
                    constexpr auto i_j_idx     = make_tuple(idx0, idx1);
                    const ComputeDataType norm = norms[local_m];
                    ComputeDataType value      = result_tile[i_j_idx];

                    // Apply normalization and activation based on output section
                    if(global_n < n)
                    {
                        ComputeDataType activated_value;
                        Activation{}(activated_value, value);
                        result_tile(i_j_idx) = (alpha_pre / norm) * activated_value + bias;
                    }
                    else if(global_n < 2 * n)
                    {
                        ComputeDataType activated_value;
                        Activation{}(activated_value, value);
                        result_tile(i_j_idx) = (alpha_post / norm) * 2.0f * activated_value + bias;
                    }
                    else
                    {
                        result_tile(i_j_idx) = (alpha_res / norm) * value + bias;
                    }
                }
            });
        });

        // Cast result to output data type
        auto result_output = cast_tile<YDataType>(result_tile);

        // Create output tensor view with vectorization for efficient writes
        constexpr index_t output_vector_size = 16 / sizeof(YDataType);

        auto output_tensor_full =
            make_naive_tensor_view<address_space_enum::global>(p_output,
                                                               make_tuple(batch, output_dim),
                                                               make_tuple(output_dim, 1),
                                                               number<output_vector_size>{},
                                                               number<1>{});

        // Pad output tensor for boundary handling
        auto output_tensor_padded = pad_tensor_view(output_tensor_full,
                                                    make_tuple(number<kMTile>{}, number<kNTile>{}),
                                                    sequence<false, Problem::kPadN>{});

        // Create tile window and store using vectorized store_tile
        auto output_window = make_tile_window(output_tensor_padded,
                                              make_tuple(number<kMTile>{}, number<kNTile>{}),
                                              {batch_start, out_start},
                                              result_output.get_tile_distribution());

        store_tile(output_window, result_output);
    }
};

} // namespace ck_tile
