// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_problem.hpp"
#include "ck_tile/ops/mhc/pipeline/mhc_default_policy.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"


// Manifold Constrained Hyper Connection  Kernel:
// =======================================
// TODO

namespace ck_tile {

template <typename Problem_,
          typename Policy_>
struct ManifoldConstrainedHyperConnection
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

    private:

    // Helper function to calculate optimal vector size for input tensor
    template <typename InputShape, typename ReduceDims>
    static constexpr index_t CalculateInputVectorSize()
    {
        using S                              = typename Problem::BlockShape;
        constexpr index_t memory_vector_size = 16 / sizeof(XDataType); // Vectorization
        constexpr index_t thread_tile_vector_size =
            S::ThreadTile_N; // In the continuous dimension, within the tile

        constexpr auto innermost_reduce_dim    = ReduceDims{}.at(number<ReduceDims{}.size() - 1>{});
        constexpr bool is_innermost_contiguous = (innermost_reduce_dim == InputShape{}.size() - 1);

        constexpr index_t stride_based_vector_size =
            is_innermost_contiguous
                ? ck_tile::min(memory_vector_size, thread_tile_vector_size)
                : 1; // Move at "vectorization" steps if continuous otherwise 1 step

        return stride_based_vector_size;
    }

    static constexpr index_t CalculateOutputVectorSize()
    {
        using S                                   = typename Problem::BlockShape;
        constexpr index_t memory_vector_size      = 16 / sizeof(YDataType);
        constexpr index_t thread_tile_vector_size = S::ThreadTile_M;
        constexpr index_t vector_size = ck_tile::min(memory_vector_size, thread_tile_vector_size);

        return vector_size;
    }

    public:

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_DEVICE void operator()(
        const XDataType* x,          // [B, nC] - input tensor
        const PhiDataType* phi,      // [nC, 2n+n²] - packed weight matrices
        YDataType* output,           // [B, 2n+n²] - output tensor
        int /*B*/, 
        int n,                       // expansion factor (small, e.g., 4)
        int C,                       // output layer dimension (potentially large)
        float r = 1.0f,              // scaling factor
        float alpha_pre = 1.0f,      // scaling for H^{pre}
        float alpha_post = 1.0f,     // scaling for H^{post}
        float alpha_res = 1.0f,      // scaling for H^{res}
        float bias = 0.0f) const     // bias term: TODO: make it a vector?
    {
        // Each block processes one batch element
        int batch_id = blockIdx.x;
        int nC = n * C;
        int output_dim = 2 * n + n * n;  // 2n + n²
        
        // Thread index within block
        int tid = threadIdx.x;
        
        // Pointer to this batch's input: x_l = [1, nC]
        const XDataType* x_batch = x + batch_id * nC;
        
        // Pointer to this batch's output: [1, 2n+n²]
        YDataType* output_batch = output + batch_id * output_dim;
        
        // Step 1: Compute norm ||x_l||_2 / sqrt(nC)
        // Use shared memory for reduction
        __shared__ ComputeDataType shared_norm[256];  // Assuming block size <= 256
        
        ComputeDataType local_sum = 0.0f;
        for (int i = tid; i < nC; i += blockDim.x)
        {
            ComputeDataType val = static_cast<ComputeDataType>(x_batch[i]);
            local_sum += val * val;
        }
        shared_norm[tid] = local_sum;
        __syncthreads();
        
        // Reduction to compute sum of squares
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                shared_norm[tid] += shared_norm[tid + stride];
            }
            __syncthreads();
        }
        
        // Compute norm: sqrt(sum) / sqrt(nC)
        ComputeDataType norm = 0.0f;
        if (tid == 0)
        {
            norm = sqrt(shared_norm[0]) / sqrt(static_cast<ComputeDataType>(nC));
            shared_norm[0] = norm;  // Store for all threads to use
        }
        __syncthreads();
        norm = shared_norm[0];
        
        // Step 2: Perform GEMM operations: x_l * phi_j for each part
        // We'll do a simple implementation where each thread computes some output elements
        
        // Process H^{pre}: x_l * phi[:, 0:n] -> output[:, 0:n]
        for (int out_idx = tid; out_idx < n; out_idx += blockDim.x)
        {
            ComputeDataType sum = 0.0f;
            for (int k = 0; k < nC; k++)
            {
                sum += static_cast<ComputeDataType>(x_batch[k]) * 
                       static_cast<ComputeDataType>(phi[k * output_dim + out_idx]);
            }
            // Apply: 1/r * alpha_pre * sum + bias
            output_batch[out_idx] = static_cast<YDataType>((alpha_pre / r) * sum + bias);
        }
        
        // Process H^{post}: x_l * phi[:, n:2n] -> output[:, n:2n]
        for (int out_idx = tid; out_idx < n; out_idx += blockDim.x)
        {
            ComputeDataType sum = 0.0f;
            for (int k = 0; k < nC; k++)
            {
                sum += static_cast<ComputeDataType>(x_batch[k]) * 
                       static_cast<ComputeDataType>(phi[k * output_dim + n + out_idx]);
            }
            // Apply: 1/r * alpha_post * sum + bias
            output_batch[n + out_idx] = static_cast<YDataType>((alpha_post / r) * sum + bias);
        }
        
        // Process H^{res}: x_l * phi[:, 2n:2n+n²] -> output[:, 2n:2n+n²]
        int n_squared = n * n;
        for (int out_idx = tid; out_idx < n_squared; out_idx += blockDim.x)
        {
            ComputeDataType sum = 0.0f;
            for (int k = 0; k < nC; k++)
            {
                sum += static_cast<ComputeDataType>(x_batch[k]) * 
                       static_cast<ComputeDataType>(phi[k * output_dim + 2 * n + out_idx]);
            }
            // Apply: 1/r * alpha_res * sum + bias
            output_batch[2 * n + out_idx] = static_cast<YDataType>((alpha_res / r) * sum + bias);
        }
        
        // Note: The norm computed above could be used for additional operations if needed
        // For now, it's computed but not applied to the output
        (void)norm;  // Suppress unused warning
    }
};

} // namespace ck_tile
