// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

// Manifold Constrained Hyper Connection Kernel V3:
// =====================================================================
// Optimizations implemented:
// - Step 2.b: 2D tiling parallelization (batch × output_dim)
// - Step 3: No output_dim tiling (all 26 outputs in one block)
// - Step 4: Use CK-tile GEMM pipeline for proper memory handling

namespace ck_tile {

template <typename Problem_,
          typename Policy_     = MHCDefaultPolicy,
          index_t kMTile_      = 64, // Batch tile size
          index_t kNTile_      = 32, // Output dimension tile (can cover all 26 outputs)
          index_t kKTile_      = 8,  // K-tile for C dimension (must match BlockGemmShape::kK)
          typename Activation_ = element_wise::Sigmoid>
struct MHCKernelV3
{
    using Activation = ck_tile::remove_cvref_t<Activation_>;
    using Problem    = ck_tile::remove_cvref_t<Problem_>;
    using Policy     = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using PhiDataType     = ck_tile::remove_cvref_t<typename Problem::PhiDataType>;

    static constexpr index_t kMTile = kMTile_; // Batch tile
    static constexpr index_t kNTile = kNTile_; // Output tile
    static constexpr index_t kKTile = kKTile_; // K tile for C dimension

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // Calculate shared memory size based on BlockGemmShape
        // The pipeline needs LDS for A[kM, kK] and B[kK, kN]
        constexpr index_t kM = Problem::BlockGemmShape::kM;
        constexpr index_t kN = Problem::BlockGemmShape::kN;
        constexpr index_t kK = Problem::BlockGemmShape::kK;

        // Approximate LDS size (actual calculation is complex, but this is a safe upper bound)
        constexpr index_t a_lds_size = kM * kK * sizeof(XDataType) * 2;
        constexpr index_t b_lds_size = kN * kK * sizeof(PhiDataType) * 2;
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
        const index_t block_m     = get_block_id() / grid_n_size;
        const index_t block_n     = get_block_id() % grid_n_size;

        const index_t batch_start = block_m * kMTile;
        const index_t out_start   = block_n * kNTile;

        if(batch_start >= batch || out_start >= output_dim)
            return;

        // Create tensor views with adjusted pointers and dimensions
        // The GEMM pipeline expects windows with origin {0,0} relative to the tensor view
        const index_t remaining_batch  = batch - batch_start;
        const index_t remaining_output = output_dim - out_start;

        auto x_tensor_unpadded = make_naive_tensor_view<address_space_enum::global>(
            p_x + batch_start * nC,          // Adjust pointer to start at this block's batch range
            make_tuple(remaining_batch, nC), // Dimensions from this block's starting point
            make_tuple(nC, 1),
            number<1>{},
            number<1>{});

        auto phi_tensor_unpadded = make_naive_tensor_view<address_space_enum::global>(
            p_phi + out_start, // Adjust pointer to start at this block's output range
            make_tuple(nC, remaining_output), // Dimensions from this block's starting point
            make_tuple(remaining_output, 1),
            number<1>{},
            number<1>{});

        // Pad tensors to tile sizes to handle boundary conditions
        auto x_tensor = pad_tensor_view(
            x_tensor_unpadded, make_tuple(number<kMTile>{}, number<kKTile>{}), sequence<0, 1>{});

        auto phi_tensor = pad_tensor_view(
            phi_tensor_unpadded, make_tuple(number<kKTile>{}, number<kNTile>{}), sequence<0, 1>{});

        // Create DRAM tile windows with origin {0, 0} relative to the padded tensor views
        // The pipeline will internally manage K-dimension iteration
        auto x_dram_window =
            make_tile_window(x_tensor,
                             make_tuple(number<kMTile>{}, number<kKTile>{}),
                             {0, 0}); // Origin at {0, 0} relative to the padded tensor view

        auto phi_dram_window =
            make_tile_window(phi_tensor,
                             make_tuple(number<kKTile>{}, number<kNTile>{}),
                             {0, 0}); // Origin at {0, 0} relative to the padded tensor view

        // Use GEMM pipeline v3 to compute the full GEMM
        using GemmPipeline = GemmPipelineAgBgCrCompV3<Problem>;

        const index_t num_k_loops = (nC + kKTile - 1) / kKTile;

        extern __shared__ char smem[];
        auto gemm_pipeline = GemmPipeline{};

        // V3 pipeline expects non-tuple windows and uses identity functions internally
        auto result_tile = gemm_pipeline(x_dram_window, phi_dram_window, num_k_loops, smem);

        // Apply elementwise operations (currently commented out for GEMM testing)
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
                    constexpr auto i_j_idx                 = make_tuple(idx0, idx1);
                    [[maybe_unused]] ComputeDataType value = result_tile[i_j_idx];

                    // TESTING: Comment out post-GEMM operations to validate GEMM only
                    // // Apply activation based on output section
                    // if(global_n < n)
                    // {
                    //     ComputeDataType activated_value;
                    //     Activation{}(activated_value, value);
                    //     value = (alpha_pre / r) * activated_value + bias;
                    // }
                    // else if(global_n < 2 * n)
                    // {
                    //     ComputeDataType activated_value;
                    //     Activation{}(activated_value, value);
                    //     value = (alpha_post / r) * 2.0f * activated_value + bias;
                    // }
                    // else
                    // {
                    //     value = (alpha_res / r) * value + bias;
                    // }

                    // p_output[global_m * output_dim + global_n] = type_convert<YDataType>(value);
                }
            });
        });

        // Cast result to output data type
        auto result_output = cast_tile<YDataType>(result_tile);

        // Create output tensor view for efficient store_tile operation
        constexpr index_t output_vector_size = 16 / sizeof(YDataType);

        auto output_tensor_view_unpadded = make_naive_tensor_view<address_space_enum::global>(
            p_output + batch_start * output_dim +
                out_start, // Adjust pointer to this block's output region
            make_tuple(remaining_batch,
                       remaining_output), // Dimensions from this block's starting point
            make_tuple(output_dim, 1),    // Strides: row-major layout
            number<output_vector_size>{}, // Vector size for efficient memory access
            number<1>{});                 // Alignment

        // Pad output tensor view to match the tile size (for boundary handling)
        auto output_tensor_view = pad_tensor_view(output_tensor_view_unpadded,
                                                  make_tuple(number<kMTile>{}, number<kNTile>{}),
                                                  sequence<0, 1>{});

        // Create tile window for the output using result_output's distribution
        auto output_window = make_tile_window(
            output_tensor_view,
            make_tuple(number<kMTile>{}, number<kNTile>{}),
            {0, 0},                                 // Origin at {0, 0} relative to the padded view
            result_output.get_tile_distribution()); // Use distribution from result_output

        // Store the result using the tile window (padding will prevent out-of-bounds writes)
        store_tile(output_window, result_output);
    }
};

} // namespace ck_tile
