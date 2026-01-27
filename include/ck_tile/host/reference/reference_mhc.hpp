// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <thread>
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"

namespace ck_tile {

// Reference implementation for Manifold Constrained Hyper Connection
template <typename XDataType,
          typename PhiDataType,
          typename YDataType,
          typename ComputeDataType = float>
CK_TILE_HOST void reference_mhc(const HostTensor<XDataType>& x_b_nc,      // [B, nC]
                                 const HostTensor<PhiDataType>& phi_nc_out, // [nC, 2n+n²]
                                 HostTensor<YDataType>& output_b_out,       // [B, 2n+n²]
                                 int n,                                     // expansion factor
                                 int C,                                     // channels per stream
                                 [[maybe_unused]] float r          = 1.0f,
                                 [[maybe_unused]] float alpha_pre  = 1.0f,
                                 [[maybe_unused]] float alpha_post = 1.0f,
                                 [[maybe_unused]] float alpha_res  = 1.0f,
                                 [[maybe_unused]] float bias       = 0.0f)
{
    const int B  = x_b_nc.get_length(0);
    const int nC = n * C;
    (void)nC; // May not be used in all code paths
    
    // Process each batch element
    auto f_batch = [&](auto b) {
        // Step 1: Compute norm ||x_l||_2 / sqrt(nC)
        ComputeDataType sum_squares = 0.0f;
        for (int i = 0; i < nC; i++)
        {
            ComputeDataType val = type_convert<ComputeDataType>(x_b_nc(b, i));
            sum_squares += val * val;
        }
        ComputeDataType norm = std::sqrt(sum_squares) / std::sqrt(static_cast<ComputeDataType>(nC));
        
        // Step 2 & 3: Perform GEMM and apply elementwise operations
        
        // Process H^{pre}: x * phi[:, 0:n] -> output[:, 0:n]
        for (int out_idx = 0; out_idx < n; out_idx++)
        {
            ComputeDataType sum = 0.0f;
            for (int k = 0; k < nC; k++)
            {
                sum += type_convert<ComputeDataType>(x_b_nc(b, k)) * 
                       type_convert<ComputeDataType>(phi_nc_out(k, out_idx));
            }
            // Apply: 1/r * alpha_pre * sum + bias
            output_b_out(b, out_idx) = type_convert<YDataType>((alpha_pre / r) * sum + bias);
        }
        
        // Process H^{post}: x * phi[:, n:2n] -> output[:, n:2n]
        for (int out_idx = 0; out_idx < n; out_idx++)
        {
            ComputeDataType sum = 0.0f;
            for (int k = 0; k < nC; k++)
            {
                sum += type_convert<ComputeDataType>(x_b_nc(b, k)) * 
                       type_convert<ComputeDataType>(phi_nc_out(k, n + out_idx));
            }
            // Apply: 1/r * alpha_post * sum + bias
            output_b_out(b, n + out_idx) = type_convert<YDataType>((alpha_post / r) * sum + bias);
        }
        
        // Process H^{res}: x * phi[:, 2n:2n+n²] -> output[:, 2n:2n+n²]
        int n_squared = n * n;
        for (int out_idx = 0; out_idx < n_squared; out_idx++)
        {
            ComputeDataType sum = 0.0f;
            for (int k = 0; k < nC; k++)
            {
                sum += type_convert<ComputeDataType>(x_b_nc(b, k)) * 
                       type_convert<ComputeDataType>(phi_nc_out(k, 2 * n + out_idx));
            }
            // Apply: 1/r * alpha_res * sum + bias
            output_b_out(b, 2 * n + out_idx) = type_convert<YDataType>((alpha_res / r) * sum + bias);
        }
        
        // Note: norm is computed but not currently used in the output
        // It could be used for additional normalization if needed
        (void)norm;
    };
    
    make_ParallelTensorFunctor(f_batch, B)(std::thread::hardware_concurrency());
}

} // namespace ck_tile
