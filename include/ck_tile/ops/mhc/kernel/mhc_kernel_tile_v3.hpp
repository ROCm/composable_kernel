// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp"
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

    // Automatically derive tile sizes from BlockGemmShape (single source of truth!)
    static constexpr index_t kMTile = Problem::BlockGemmShape::kM; // Batch tile
    static constexpr index_t kNTile = Problem::BlockGemmShape::kN; // Output tile
    static constexpr index_t kKTile = Problem::BlockGemmShape::kK; // K tile for C dimension

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // Calculate LDS size for V1 pipeline
        // V1 uses single-buffered LDS for A and B tiles
        constexpr index_t kM = Problem::BlockGemmShape::kM;
        constexpr index_t kN = Problem::BlockGemmShape::kN;
        constexpr index_t kK = Problem::BlockGemmShape::kK;

        constexpr index_t kLdsAlignmentInBytes = 16;

        // A LDS: [kM, kK]
        constexpr index_t a_lds_size = kM * kK * sizeof(XDataType);
        constexpr index_t a_lds_size_aligned =
            ((a_lds_size + kLdsAlignmentInBytes - 1) / kLdsAlignmentInBytes) * kLdsAlignmentInBytes;

        // B LDS: [kN, kK] for column-major or [kK, kN] for row-major
        constexpr index_t b_lds_size = kN * kK * sizeof(PhiDataType);

        return a_lds_size_aligned + b_lds_size;
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

        // STEP 1: Compute norms BEFORE GEMM
        // We'll use these norms in an elementwise function during GEMM's data loading
        // This way we normalize X on-the-fly as it's loaded for GEMM
        constexpr index_t kVectorSize = 4; // Load 4 floats at a time

        ComputeDataType inv_norms[kMTile]; // Store inverse norms for efficiency

        for(index_t local_m = 0; local_m < kMTile; ++local_m)
        {
            const index_t global_m = batch_start + local_m;
            if(global_m < batch)
            {
                ComputeDataType sum_squares = 0.0f;
                const XDataType* row_ptr    = p_x + global_m * nC;

                // Vectorized loop: process kVectorSize elements at a time
                index_t k = 0;
                for(; k + kVectorSize <= nC; k += kVectorSize)
                {
                    // Load vector of elements
                    using VecType    = ext_vector_t<XDataType, kVectorSize>;
                    VecType vec_data = *c_style_pointer_cast<const VecType*>(row_ptr + k);

#pragma unroll
                    for(index_t i = 0; i < kVectorSize; ++i)
                    {
                        ComputeDataType val = type_convert<ComputeDataType>(vec_data[i]);
                        sum_squares += val * val;
                    }
                }

                // Handle remaining elements (scalar loop)
                for(; k < nC; ++k)
                {
                    ComputeDataType val = type_convert<ComputeDataType>(row_ptr[k]);
                    sum_squares += val * val;
                }

                const ComputeDataType norm =
                    ck_tile::sqrt(sum_squares) / ck_tile::sqrt(static_cast<ComputeDataType>(nC));
                inv_norms[local_m] = 1.0f / norm; // Store inverse for efficiency
            }
            else
            {
                inv_norms[local_m] = 1.0f; // Default for out-of-bounds
            }
        }

        // STEP 2: Setup GEMM with normalization applied during data loading
        // Create full tensor views
        auto x_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(batch, nC), make_tuple(nC, 1), number<1>{}, number<1>{});

        auto phi_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_phi, make_tuple(output_dim, nC), make_tuple(1, output_dim), number<1>{}, number<1>{});

        // Pad tensors
        auto x_tensor_padded = pad_tensor_view(x_tensor_full,
                                               make_tuple(number<kMTile>{}, number<kKTile>{}),
                                               sequence<false, Problem::kPadK>{});

        auto phi_tensor_padded = pad_tensor_view(phi_tensor_full,
                                                 make_tuple(number<kNTile>{}, number<kKTile>{}),
                                                 sequence<false, Problem::kPadK>{});

        // Create DRAM tile windows
        auto x_dram_window = make_tile_window(
            x_tensor_padded, make_tuple(number<kMTile>{}, number<kKTile>{}), {batch_start, 0});

        auto phi_dram_window = make_tile_window(
            phi_tensor_padded, make_tuple(number<kNTile>{}, number<kKTile>{}), {out_start, 0});

        // Use GEMM pipeline v1 with elementwise normalization function
        using GemmPipeline = GemmPipelineAGmemBGmemCRegV1<Problem>;

        const index_t num_k_loops = (nC + kKTile - 1) / kKTile;

        __shared__ char smem[GetSmemSize()];
        auto gemm_pipeline = GemmPipeline{};

        // Identity function for phi (no normalization needed)
        auto phi_identity_func = [](auto& e, const PhiDataType& phi_val) { e = phi_val; };

        auto result_tile = gemm_pipeline(make_tuple(x_dram_window),
                                        phi_identity_func,
                                        make_tuple(phi_dram_window),
                                        phi_identity_func,
                                        num_k_loops,
                                        smem);

        // Apply normalization and activation in post-processing
        // Now we divide by norm AFTER the GEMM, which means:
        // result = (x * phi) / norm = x/norm * phi (mathematically equivalent)
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

                    // Get the inverse norm for this batch element
                    const ComputeDataType inv_norm = inv_norms[local_m];

                    // Apply normalization and activation based on output section
                    // Formula: result = (1/norm) * value * alpha + bias
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
                        result_tile(i_j_idx) = alpha_post * inv_norm * 2.0f * activated_value + bias;
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

        // Create full output tensor view and use window origin
        constexpr index_t output_vector_size = 16 / sizeof(YDataType);

        auto output_tensor_full =
            make_naive_tensor_view<address_space_enum::global>(p_output,
                                                               make_tuple(batch, output_dim),
                                                               make_tuple(output_dim, 1),
                                                               number<output_vector_size>{},
                                                               number<1>{});

        // Pad output tensor view for boundary handling (row-major C: sequence<false, kPadN>)
        auto output_tensor_padded = pad_tensor_view(output_tensor_full,
                                                    make_tuple(number<kMTile>{}, number<kNTile>{}),
                                                    sequence<false, Problem::kPadN>{});

        // Create tile window with origin at this block's region
        auto output_window = make_tile_window(output_tensor_padded,
                                              make_tuple(number<kMTile>{}, number<kNTile>{}),
                                              {batch_start, out_start},
                                              result_output.get_tile_distribution());

        // Store the result
        store_tile(output_window, result_output);
    }
};

} // namespace ck_tile
