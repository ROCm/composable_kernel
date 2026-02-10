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

    // Grid configuration: 2D grid over (batch, output_dim)
    CK_TILE_HOST static constexpr auto GetGridSize(index_t batch, index_t output_dim)
    {
        const index_t grid_m = (batch + kMTile - 1) / kMTile;
        const index_t grid_n = (output_dim + kNTile - 1) / kNTile;
        return make_tuple(grid_m, grid_n);
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
        // 2D block indexing
        const index_t grid_n_size = (output_dim + kNTile - 1) / kNTile;
        const index_t block_id    = get_block_id();
        const index_t block_m     = block_id / grid_n_size;
        const index_t block_n     = block_id % grid_n_size;

        const index_t batch_start = block_m * kMTile;
        const index_t out_start   = block_n * kNTile;

        if(batch_start >= batch || out_start >= output_dim)
            return;

        const index_t tid = get_thread_id();

        // Allocate shared memory for A and B tiles + norm accumulators + GEMM results
        __shared__ char smem_ptr[GetSmemSize()];
        XDataType* x_lds = reinterpret_cast<XDataType*>(smem_ptr);
        PhiDataType* phi_lds =
            reinterpret_cast<PhiDataType*>(smem_ptr + kMTile * kKTile * sizeof(XDataType));

        // Thread-local norm accumulation (one per batch element in tile)
        // Each thread accumulates for the elements it processes
        ComputeDataType thread_sum_squares[kMTile];
        for(index_t i = 0; i < kMTile; ++i)
        {
            thread_sum_squares[i] = 0.0f;
        }

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

        // Main loop: load tiles with vectorization, compute norms, and accumulate GEMM
        for(index_t k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx)
        {
            // Load X tile using vectorized load_tile
            auto x_tile = make_static_distributed_tensor<XDataType>(x_load_tile_dist);
            load_tile(x_tile, x_dram_window);

            // Accumulate norms from the loaded tile into thread-local storage
            constexpr auto x_tile_spans = decltype(x_tile)::get_distributed_spans();
            sweep_tile_span(x_tile_spans[number<0>{}], [&](auto idx0) {
                sweep_tile_span(x_tile_spans[number<1>{}], [&](auto idx1) {
                    const auto tile_idx = get_x_indices_from_distributed_indices(
                        x_tile.get_tile_distribution(), make_tuple(idx0, idx1));

                    const index_t local_m  = tile_idx.at(number<0>{});
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);

                    ComputeDataType x_val = type_convert<ComputeDataType>(x_tile[i_j_idx]);
                    thread_sum_squares[local_m] += x_val * x_val;
                });
            });

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

        // Reduce thread-local norm accumulators using warp shuffle + shared memory
        __shared__ ComputeDataType sum_squares_shared[kMTile];

        // Initialize shared memory
        if(tid < kMTile)
        {
            sum_squares_shared[tid] = 0.0f;
        }
        block_sync_lds();

        // Warp-level reduction for each batch element
        // Since we have 64 threads (1 warp) and kMTile=16, multiple threads contribute to each
        // element
        constexpr index_t threads_per_element =
            kBlockSize / kMTile; // 64/16 = 4 threads per batch element

        for(index_t local_m = 0; local_m < kMTile; ++local_m)
        {
            ComputeDataType my_sum = thread_sum_squares[local_m];

            // Warp shuffle reduction within threads handling this batch element
            // Threads [local_m*4, local_m*4+1, local_m*4+2, local_m*4+3] reduce together
            const index_t my_group      = tid / threads_per_element;
            const index_t lane_in_group = tid % threads_per_element;

            if(my_group == local_m)
            {
// Reduce within this group of 4 threads using warp shuffle
#pragma unroll
                for(index_t offset = threads_per_element / 2; offset > 0; offset /= 2)
                {
                    my_sum += __shfl_down(my_sum, offset);
                }

                // First thread in group writes to shared memory
                if(lane_in_group == 0)
                {
                    sum_squares_shared[local_m] = my_sum;
                }
            }
        }
        block_sync_lds();

        // Compute inverse norms after all K-tiles processed
        ComputeDataType inv_norms[kMTile];
        for(index_t local_m = 0; local_m < kMTile; ++local_m)
        {
            const index_t global_m = batch_start + local_m;
            if(global_m < batch)
            {
                const ComputeDataType norm = ck_tile::sqrt(sum_squares_shared[local_m]) /
                                             ck_tile::sqrt(static_cast<ComputeDataType>(nC));
                inv_norms[local_m] = 1.0f / norm;
            }
            else
            {
                inv_norms[local_m] = 1.0f;
            }
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
                    constexpr auto i_j_idx         = make_tuple(idx0, idx1);
                    const ComputeDataType inv_norm = inv_norms[local_m];
                    ComputeDataType value          = result_tile[i_j_idx];

                    // Apply normalization and activation based on output section
                    if(global_n < n)
                    {
                        ComputeDataType activated_value;
                        Activation{}(activated_value, value);
                        result_tile(i_j_idx) = alpha_pre * inv_norm * activated_value + bias;
                    }
                    else if(global_n < 2 * n)
                    {
                        ComputeDataType activated_value;
                        Activation{}(activated_value, value);
                        result_tile(i_j_idx) =
                            alpha_post * inv_norm * 2.0f * activated_value + bias;
                    }
                    else
                    {
                        result_tile(i_j_idx) = alpha_res * inv_norm * value + bias;
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
