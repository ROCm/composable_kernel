// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_default_policy.hpp"

// Manifold Constrained Hyper Connection Kernel (CK Tile Version):
// ================================================================
// This implementation uses CK tile primitives: tensor descriptors, buffer views, and tile windows

namespace ck_tile {

template <typename Problem_, typename Policy_ = MHCDefaultPolicy>
struct ManifoldConstrainedHyperConnectionCKTile
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using YDataType       = ck_tile::remove_cvref_t<typename Problem::YDataType>;
    using PhiDataType     = ck_tile::remove_cvref_t<typename Problem::PhiDataType>;

    static constexpr index_t kBlockSize = Problem::BlockShape::BlockSize;

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? kBlockSize / 2 : kBlockSize;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_DEVICE void operator()(const XDataType* p_x,     // [B, nC] - input tensor
                                   const PhiDataType* p_phi, // [nC, 2n+n²] - packed weight matrices
                                   YDataType* p_output,      // [B, 2n+n²] - output tensor
                                   int /*B*/,
                                   int n, // expansion factor (small, e.g., 4)
                                   int C, // output layer dimension (potentially large)
                                   float r          = 1.0f, // scaling factor
                                   float alpha_pre  = 1.0f, // scaling for H^{pre}
                                   float alpha_post = 1.0f, // scaling for H^{post}
                                   float alpha_res  = 1.0f, // scaling for H^{res}
                                   float bias       = 0.0f) const // bias term
    {
        // Each block processes one batch element
        const index_t batch_id   = get_block_id();
        const index_t nC         = n * C;
        const index_t output_dim = 2 * n + n * n; // 2n + n²
        const index_t tid        = get_thread_id();

        // Pointers to this batch's data
        const XDataType* x_batch = p_x + batch_id * nC;
        YDataType* output_batch  = p_output + batch_id * output_dim;

        // Step 1: Compute norm ||x||_2 / sqrt(nC) using shared memory reduction
        __shared__ ComputeDataType shared_norm[256];

        ComputeDataType local_sum = 0.0f;
        for(index_t i = tid; i < nC; i += get_block_size())
        {
            ComputeDataType val = type_convert<ComputeDataType>(x_batch[i]);
            local_sum += val * val;
        }
        shared_norm[tid] = local_sum;
        block_sync_lds();

        // Parallel reduction
        for(index_t stride = get_block_size() / 2; stride > 0; stride >>= 1)
        {
            if(tid < stride)
            {
                shared_norm[tid] += shared_norm[tid + stride];
            }
            block_sync_lds();
        }

        ComputeDataType norm = 0.0f;
        if(tid == 0)
        {
            norm           = sqrt(shared_norm[0]) / sqrt(type_convert<ComputeDataType>(nC));
            shared_norm[0] = norm;
        }
        block_sync_lds();
        norm = shared_norm[0];

        // Step 2: Perform GEMM operations for each phi section
        // Each thread processes a subset of output elements

        // Process H^{pre}: x * phi[:, 0:n] -> output[:, 0:n]
        for(index_t out_idx = tid; out_idx < n; out_idx += get_block_size())
        {
            ComputeDataType sum = 0.0f;
            for(index_t k = 0; k < nC; k++)
            {
                sum += type_convert<ComputeDataType>(x_batch[k]) *
                       type_convert<ComputeDataType>(p_phi[k * output_dim + out_idx]);
            }
            // Apply: 1/r * alpha_pre * sum + bias
            output_batch[out_idx] = type_convert<YDataType>((alpha_pre / r) * sum + bias);
        }

        // Process H^{post}: x * phi[:, n:2n] -> output[:, n:2n]
        for(index_t out_idx = tid; out_idx < n; out_idx += get_block_size())
        {
            ComputeDataType sum = 0.0f;
            for(index_t k = 0; k < nC; k++)
            {
                sum += type_convert<ComputeDataType>(x_batch[k]) *
                       type_convert<ComputeDataType>(p_phi[k * output_dim + n + out_idx]);
            }
            // Apply: 1/r * alpha_post * sum + bias
            output_batch[n + out_idx] = type_convert<YDataType>((alpha_post / r) * sum + bias);
        }

        // Process H^{res}: x * phi[:, 2n:2n+n²] -> output[:, 2n:2n+n²]
        const index_t n_squared = n * n;
        for(index_t out_idx = tid; out_idx < n_squared; out_idx += get_block_size())
        {
            ComputeDataType sum = 0.0f;
            for(index_t k = 0; k < nC; k++)
            {
                sum += type_convert<ComputeDataType>(x_batch[k]) *
                       type_convert<ComputeDataType>(p_phi[k * output_dim + 2 * n + out_idx]);
            }
            // Apply: 1/r * alpha_res * sum + bias
            output_batch[2 * n + out_idx] = type_convert<YDataType>((alpha_res / r) * sum + bias);
        }

        // Note: norm is computed but not currently used in the output
        (void)norm;
    }
};

} // namespace ck_tile
