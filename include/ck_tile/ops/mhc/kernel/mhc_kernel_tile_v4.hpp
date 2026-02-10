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

        // Shared memory for norm accumulation (one per batch element in tile)
        __shared__ ComputeDataType sum_squares_shared[kMTile];

        // Shared memory for GEMM result accumulation
        __shared__ ComputeDataType result_shared[kMTile * kNTile];

        // Initialize shared norm accumulators and result
        for(index_t i = tid; i < kMTile; i += get_block_size())
        {
            sum_squares_shared[i] = 0.0f;
        }
        for(index_t i = tid; i < kMTile * kNTile; i += get_block_size())
        {
            result_shared[i] = 0.0f;
        }
        block_sync_lds();

        // Number of K-tile iterations
        const index_t num_k_tiles = (nC + kKTile - 1) / kKTile;

        // Main loop: load tiles, compute norms incrementally, and accumulate GEMM
        for(index_t k_tile_idx = 0; k_tile_idx < num_k_tiles; ++k_tile_idx)
        {
            const index_t k_start = k_tile_idx * kKTile;
            const index_t k_end   = min(k_start + kKTile, nC);
            const index_t k_len   = k_end - k_start;

            // Load X tile from global to LDS and accumulate norm
            for(index_t i = tid; i < kMTile * kKTile; i += get_block_size())
            {
                const index_t local_m  = i / kKTile;
                const index_t local_k  = i % kKTile;
                const index_t global_m = batch_start + local_m;
                const index_t global_k = k_start + local_k;

                XDataType x_val = 0;
                if(global_m < batch && local_k < k_len)
                {
                    x_val = p_x[global_m * nC + global_k];

                    // Accumulate norm for this batch element using atomics
                    ComputeDataType x_compute = type_convert<ComputeDataType>(x_val);
                    ComputeDataType sq        = x_compute * x_compute;
                    atomicAdd(&sum_squares_shared[local_m], sq);
                }
                x_lds[i] = x_val;
            }

            // Load Phi tile from global to LDS in column-major format (K x N)
            // phi is stored in global memory as [nC, output_dim] row-major
            // We need to transpose it to [K, N] column-major for BlockGemm
            for(index_t i = tid; i < kKTile * kNTile; i += get_block_size())
            {
                const index_t local_k  = i / kNTile;
                const index_t local_n  = i % kNTile;
                const index_t global_k = k_start + local_k;
                const index_t global_n = out_start + local_n;

                PhiDataType phi_val = 0;
                if(local_k < k_len && global_n < output_dim)
                {
                    phi_val = p_phi[global_k * output_dim + global_n];
                }
                // Store in column-major: phi_lds[n * kKTile + k]
                phi_lds[local_n * kKTile + local_k] = phi_val;
            }

            block_sync_lds();

            // Perform manual GEMM: result_acc += x_lds * phi_lds^T
            // Distribute work: each thread computes a subset of output elements
            // With 64 threads and 16x16 output, each thread handles 4 elements
            const index_t total_elements = kMTile * kNTile;
            const index_t elements_per_thread =
                (total_elements + get_block_size() - 1) / get_block_size();

            for(index_t elem_idx = 0; elem_idx < elements_per_thread; ++elem_idx)
            {
                const index_t global_elem = tid * elements_per_thread + elem_idx;
                if(global_elem < total_elements)
                {
                    const index_t m_idx = global_elem / kNTile;
                    const index_t n_idx = global_elem % kNTile;

                    ComputeDataType acc = 0.0f;
                    for(index_t k_idx = 0; k_idx < kKTile; ++k_idx)
                    {
                        ComputeDataType x_val =
                            type_convert<ComputeDataType>(x_lds[m_idx * kKTile + k_idx]);
                        ComputeDataType phi_val =
                            type_convert<ComputeDataType>(phi_lds[n_idx * kKTile + k_idx]);
                        acc += x_val * phi_val;
                    }
                    // Accumulate to shared memory using atomics
                    atomicAdd(&result_shared[m_idx * kNTile + n_idx], acc);
                }
            }

            block_sync_lds();
        }

        // Ensure all norm accumulations are complete
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

        // Apply normalization, activation, and write output
        for(index_t i = tid; i < kMTile * kNTile; i += get_block_size())
        {
            const index_t local_m = i / kNTile;
            const index_t local_n = i % kNTile;

            const index_t global_m = batch_start + local_m;
            const index_t global_n = out_start + local_n;

            if(global_m >= batch || global_n >= output_dim)
                continue;

            const ComputeDataType inv_norm = inv_norms[local_m];
            ComputeDataType value          = result_shared[i];

            // Apply normalization and activation based on output section
            if(global_n < n)
            {
                ComputeDataType activated_value;
                Activation{}(activated_value, value);
                value = alpha_pre * inv_norm * activated_value + bias;
            }
            else if(global_n < 2 * n)
            {
                ComputeDataType activated_value;
                Activation{}(activated_value, value);
                value = alpha_post * inv_norm * 2.0f * activated_value + bias;
            }
            else
            {
                value = alpha_res * inv_norm * value + bias;
            }

            // Write to global memory
            p_output[global_m * output_dim + global_n] = type_convert<YDataType>(value);
        }
    }
};

} // namespace ck_tile
