// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp"

namespace ck {

// Tensor Shape
// dy, x = [M, K], gamma = [1, K], x_mean, inv_std = [M, 1]

// Flow:
// def normalization_backward_x(dy, x, gamma, x_mean, inv_std, reduce_axis, reduce_size):
//     ds = np.sum(dy * gamma * x, axis=reduce_axis, keepdims=True)
//     db = np.sum(dy * gamma, axis=reduce_axis, keepdims=True)
//     b = (db * x_mean - ds) * inv_std ** (3) / reduce_size
//     c = -b * x_mean - db * inv_std / reduce_size
//     dx = inv_std * dy * gamma + b * x + c
//     return dx

template <typename DYDataType,
          typename XDataType,
          typename GammaDataType,
          typename MeanInvStdDataType,
          typename ComputeDataType,
          typename DXDataType,
          typename GridDesc_M_K,
          index_t BlockSize,
          index_t MThreadClusterSize,
          index_t KThreadClusterSize,
          index_t MThreadSliceSize,
          index_t KThreadSliceSize,
          index_t DYSrcVectorDim,
          index_t DYSrcVectorSize,
          index_t XSrcVectorDim,
          index_t XSrcVectorSize,
          index_t GammaSrcVectorDim,
          index_t GammaSrcVectorSize,
          index_t MeanInvStdSrcVectorDim,
          index_t MeanInvStdSrcVectorSize,
          index_t DXDstVectorDim,
          index_t DXDstVectorSize,
          bool SweepOnce>
struct GridwiseNormalizationBwdX_mk_to_mk
{

    __device__ static void Run(const GridDesc_M_K& dy_grid_desc_m_k,
                               const GridDesc_M_K& x_grid_desc_m_k,
                               const GridDesc_M_K& gamma_grid_desc_m_k,
                               const GridDesc_M_K& mean_grid_desc_m_k,
                               const GridDesc_M_K& inv_std_grid_desc_m_k,
                               const GridDesc_M_K& dx_grid_desc_m_k,
                               index_t num_k_block_tile_iteration,
                               const DYDataType* const __restrict__ p_dy_global,
                               const XDataType* const __restrict__ p_x_global,
                               const GammaDataType* const __restrict__ p_gamma_global,
                               const MeanInvStdDataType* const __restrict__ p_mean_global,
                               const MeanInvStdDataType* const __restrict__ p_inv_std_global,
                               DXDataType* const __restrict__ p_dx_global)
    {
        // TODO
    }
};

} // namespace ck
