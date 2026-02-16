// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_default_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

// Manifold Constrained Hyper Connection Kernel V5:
// =====================================================================
// Split-K implementation with 2D grid (B, C):
// - Grid dimension 0: Batch tiles (B / kMTile)
// - Grid dimension 1: C tiles (nC / kKTile) - split-K dimension
// - Each block computes partial GEMM for its C-tile
// - Results stored to workspace buffer (no atomics!)
// - Separate reduction kernel combines partial results

namespace ck_tile {

template <typename Problem_,
          typename Policy_     = MHCDefaultPolicy,
          typename Activation_ = element_wise::Sigmoid>
struct MHCKernelV5
{
    using Activation = ck_tile::remove_cvref_t<Activation_>;
    using Problem    = ck_tile::remove_cvref_t<Problem_>;
    using Policy     = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using PhiDataType     = ck_tile::remove_cvref_t<typename Problem::PhiDataType>;

    // Tile sizes from BlockGemmShape
    static constexpr index_t kMTile = Problem::BlockGemmShape::kM; // Batch tile (16)
    static constexpr index_t kNTile = Problem::BlockGemmShape::kN; // Output tile (32)
    static constexpr index_t kKTile = Problem::BlockGemmShape::kK; // K tile for C dimension (64)

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    // Adaptive K-tiles per block based on C dimension
    CK_TILE_HOST_DEVICE static constexpr index_t GetKTilesPerBlock(index_t nC)
    {
        // Adaptive selection based on C size:
        // - Large C (≥4096): 8 tiles/block (512 elements) - maximize MFMA utilization
        // - Medium C (≥1024): 4 tiles/block (256 elements) - balance overhead and work
        // - Small C (≥256): 2 tiles/block (128 elements) - reduce overhead
        // - Tiny C (<256): 1 tile/block (64 elements) - minimize overhead
        return (nC >= 4096) ? 8 : (nC >= 1024) ? 4 : (nC >= 256) ? 2 : 1;
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    // Padding to avoid LDS bank conflicts
    // AMD GPUs have 32 LDS banks, 4-byte bank width
    // For bf16 (2 bytes), we need padding to avoid stride being multiple of 32
    static constexpr index_t kKTilePadded = kKTile + 8; // Add 8 elements padding

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        // LDS for BlockGemm with padding: A[kMTile, kKTile+2] + B[kNTile, kKTile+2]
        constexpr index_t a_lds_size = kMTile * kKTilePadded * sizeof(XDataType);
        constexpr index_t b_lds_size = kNTile * kKTilePadded * sizeof(PhiDataType);
        return a_lds_size + b_lds_size;
    }

    // Grid configuration: 2D grid (B, C) for split-K
    // Each block processes adaptive number of K-tiles (hierarchical split-K)
    CK_TILE_HOST static constexpr auto
    GetGridSize(index_t batch, [[maybe_unused]] index_t output_dim, index_t nC)
    {
        const index_t k_tiles_per_block = GetKTilesPerBlock(nC);
        const index_t k_per_block       = kKTile * k_tiles_per_block;
        const index_t grid_m            = (batch + kMTile - 1) / kMTile;
        const index_t grid_k            = (nC + k_per_block - 1) / k_per_block;
        return make_tuple(grid_m, grid_k);
    }

    CK_TILE_DEVICE void operator()(const XDataType* p_x,
                                   const PhiDataType* p_phi,
                                   ComputeDataType* p_workspace,     // [grid_k, batch, output_dim]
                                   ComputeDataType* p_partial_norms, // [grid_k, batch]
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
        // Determine adaptive K-tiles per block based on C dimension
        const index_t k_tiles_per_block = GetKTilesPerBlock(nC);
        const index_t k_per_block       = kKTile * k_tiles_per_block;

        // 2D block indexing
        const index_t grid_m  = (batch + kMTile - 1) / kMTile;
        const index_t block_m = get_block_id() % grid_m;
        const index_t block_k = get_block_id() / grid_m;

        const index_t batch_start = block_m * kMTile;
        const index_t k_start     = block_k * k_per_block; // Start of this block's K-range
        const index_t k_end       = ck_tile::min(k_start + k_per_block, nC);
        const index_t out_start   = 0;

        if(batch_start >= batch || k_start >= nC)
            return;

        // Allocate shared memory with padding
        __shared__ char smem_ptr[GetSmemSize()];
        XDataType* x_lds = reinterpret_cast<XDataType*>(smem_ptr);
        PhiDataType* phi_lds =
            reinterpret_cast<PhiDataType*>(smem_ptr + kMTile * kKTilePadded * sizeof(XDataType));

        // Create BlockGemm instance and result tile
        using BlockGemm  = BlockGemmASmemBSmemCRegV1<Problem, Policy>;
        auto result_tile = BlockGemm::MakeCBlockTile();
        set_tile(result_tile, 0.0f);

        // Create tensor views for X and Phi
        auto x_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_x, make_tuple(batch, nC), make_tuple(nC, 1), number<1>{}, number<1>{});

        auto x_tensor_padded = pad_tensor_view(x_tensor_full,
                                               make_tuple(number<kMTile>{}, number<kKTile>{}),
                                               sequence<false, Problem::kPadK>{});

        constexpr auto x_load_tile_dist = Problem::MakeXLoadTileDistribution();
        auto x_dram_window              = make_tile_window(x_tensor_padded,
                                              make_tuple(number<kMTile>{}, number<kKTile>{}),
                                                           {batch_start, k_start},
                                              x_load_tile_dist);

        auto x_lds_tensor = make_naive_tensor_view<address_space_enum::lds>(
            x_lds,
            make_tuple(number<kMTile>{}, number<kKTile>{}),
            make_tuple(number<kKTilePadded>{}, number<1>{}),
            number<1>{},
            number<1>{});

        auto x_lds_window =
            make_tile_window(x_lds_tensor, make_tuple(number<kMTile>{}, number<kKTile>{}), {0, 0});

        // Create Phi tensor view and window
        auto phi_tensor_full = make_naive_tensor_view<address_space_enum::global>(
            p_phi, make_tuple(output_dim, nC), make_tuple(1, output_dim), number<1>{}, number<1>{});

        auto phi_tensor_padded = pad_tensor_view(phi_tensor_full,
                                                 make_tuple(number<kNTile>{}, number<kKTile>{}),
                                                 sequence<false, Problem::kPadK>{});

        constexpr auto phi_load_tile_dist = Problem::MakePhiLoadTileDistribution();
        auto phi_dram_window              = make_tile_window(phi_tensor_padded,
                                                make_tuple(number<kNTile>{}, number<kKTile>{}),
                                                             {out_start, k_start},
                                                phi_load_tile_dist);

        auto phi_lds_tensor = make_naive_tensor_view<address_space_enum::lds>(
            phi_lds,
            make_tuple(number<kNTile>{}, number<kKTile>{}),
            make_tuple(number<kKTilePadded>{}, number<1>{}),
            number<1>{},
            number<1>{});

        auto phi_lds_window = make_tile_window(
            phi_lds_tensor, make_tuple(number<kNTile>{}, number<kKTile>{}), {0, 0});

        // Compute partial norms for this block's K-range (all K-tiles)
        const index_t thread_id           = get_thread_id();
        constexpr index_t threads_per_row = kBlockSize / kMTile;
        const index_t row_id              = thread_id / threads_per_row;
        const index_t thread_in_row       = thread_id % threads_per_row;

        // Allocate only what we need: kMTile rows × threads_per_row columns
        __shared__ ComputeDataType norm_reduction[kMTile][kBlockSize / kMTile];

        if(row_id < kMTile)
        {
            const index_t global_m      = batch_start + row_id;
            ComputeDataType partial_sum = 0.0f;

            if(global_m < batch)
            {
                const XDataType* row_ptr = p_x + global_m * nC + k_start;
                const index_t k_range    = k_end - k_start;

                constexpr index_t kVecSize = 4;
                for(index_t k = thread_in_row * kVecSize; k < k_range;
                    k += threads_per_row * kVecSize)
                {
                    if(k + kVecSize <= k_range)
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
                        for(index_t i = 0; i < kVecSize && k + i < k_range; ++i)
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

        // Reduce and store partial norms to global memory
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

                // Store to global memory: p_partial_norms[block_k, global_m]
                p_partial_norms[block_k * batch + global_m] = sum_squares;
            }
        }

        // Loop over K-tiles within this block's K-range (adaptive count)
        for(index_t k_tile_idx = 0; k_tile_idx < k_tiles_per_block; ++k_tile_idx)
        {
            const index_t k_current = k_start + k_tile_idx * kKTile;
            if(k_current >= k_end)
                break;

            // Load X tile for this K-slice
            auto x_tile = make_static_distributed_tensor<XDataType>(x_load_tile_dist);
            load_tile(x_tile, x_dram_window);
            store_tile(x_lds_window, x_tile);

            // Load Phi tile for this K-slice
            auto phi_tile = make_static_distributed_tensor<PhiDataType>(phi_load_tile_dist);
            load_tile(phi_tile, phi_dram_window);
            store_tile(phi_lds_window, phi_tile);

            block_sync_lds();

            // Accumulate GEMM: result_tile += x_lds * phi_lds^T
            BlockGemm{}(result_tile, x_lds_window, phi_lds_window);

            // Move windows to next K-tile
            move_tile_window(x_dram_window, {0, kKTile});
            move_tile_window(phi_dram_window, {0, kKTile});

            block_sync_lds();
        }

        // Store partial results to workspace buffer: p_workspace[block_k, batch, output_dim]
        // Layout: [grid_k][batch][output_dim]
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
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    ComputeDataType value  = result_tile[i_j_idx];

                    // Store to workspace: [block_k][global_m][global_n]
                    const index_t workspace_idx =
                        block_k * (batch * output_dim) + global_m * output_dim + global_n;
                    p_workspace[workspace_idx] = value;
                }
            });
        });
    }
};

// Optimized reduction kernel with block-level shared memory reduction
template <typename Problem_, typename Activation_ = element_wise::Sigmoid>
struct MHCReductionKernel
{
    using Activation      = ck_tile::remove_cvref_t<Activation_>;
    using Problem         = ck_tile::remove_cvref_t<Problem_>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;

    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kVecSize   = 4; // Vectorized loads

    CK_TILE_HOST static constexpr auto BlockSize() { return kBlockSize; }

    CK_TILE_DEVICE void operator()(const ComputeDataType* p_workspace,
                                   const ComputeDataType* p_partial_norms,
                                   YDataType* p_output,
                                   index_t batch,
                                   index_t nC,
                                   index_t output_dim,
                                   index_t n,
                                   index_t grid_k,
                                   float alpha_pre,
                                   float alpha_post,
                                   float alpha_res,
                                   float bias) const
    {
        const index_t tid        = get_thread_id();
        const index_t block_id   = get_block_id();
        const index_t block_size = get_block_size();

        // Each block processes multiple output elements
        // Use block-level reduction for better memory coalescing
        const index_t elements_per_block = block_size;
        const index_t global_start       = block_id * elements_per_block;
        const index_t total_elements     = batch * output_dim;

        const index_t global_idx = global_start + tid;

        if(global_idx >= total_elements)
            return;

        const index_t global_m = global_idx / output_dim;
        const index_t global_n = global_idx % output_dim;

        // Reduce partial norms with vectorized loads where possible
        ComputeDataType sum_squares = 0.0f;
        const index_t norm_base     = global_m;

        // Vectorized reduction for norms
        index_t k = 0;
        for(; k + kVecSize <= grid_k; k += kVecSize)
        {
            using VecType = ext_vector_t<ComputeDataType, kVecSize>;
            VecType vec_norms;

#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
            {
                vec_norms[i] = p_partial_norms[(k + i) * batch + norm_base];
            }

#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
            {
                sum_squares += vec_norms[i];
            }
        }

        // Handle remaining elements
        for(; k < grid_k; ++k)
        {
            sum_squares += p_partial_norms[k * batch + norm_base];
        }

        const ComputeDataType sqrt_nC = ck_tile::sqrt(static_cast<ComputeDataType>(nC));
        ComputeDataType norm          = ck_tile::sqrt(sum_squares) / sqrt_nC;
        norm                          = (norm > 1e-12f) ? norm : 1.0f;

        // Reduce partial GEMM results with improved memory access pattern
        // Reorganize to improve coalescing: threads in a warp access consecutive elements
        ComputeDataType value          = 0.0f;
        const index_t workspace_stride = batch * output_dim;

        // Vectorized reduction for workspace
        k = 0;
        for(; k + kVecSize <= grid_k; k += kVecSize)
        {
            using VecType = ext_vector_t<ComputeDataType, kVecSize>;
            VecType vec_values;

#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
            {
                const index_t workspace_idx = (k + i) * workspace_stride + global_idx;
                vec_values[i]               = p_workspace[workspace_idx];
            }

#pragma unroll
            for(index_t i = 0; i < kVecSize; ++i)
            {
                value += vec_values[i];
            }
        }

        // Handle remaining elements
        for(; k < grid_k; ++k)
        {
            const index_t workspace_idx = k * workspace_stride + global_idx;
            value += p_workspace[workspace_idx];
        }

        // Apply normalization and activation based on output section
        ComputeDataType final_value;
        if(global_n < n)
        {
            // Pre-activation section
            ComputeDataType activated_value;
            Activation{}(activated_value, value);
            final_value = (alpha_pre / norm) * activated_value + bias;
        }
        else if(global_n < 2 * n)
        {
            // Post-activation section
            ComputeDataType activated_value;
            Activation{}(activated_value, value);
            final_value = (alpha_post / norm) * 2.0f * activated_value + bias;
        }
        else
        {
            // Residual section
            final_value = (alpha_res / norm) * value + bias;
        }

        p_output[global_idx] = type_convert<YDataType>(final_value);
    }
};

} // namespace ck_tile
