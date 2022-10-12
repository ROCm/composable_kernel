// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v1.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_welford.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_welford.hpp"

namespace ck {

template <typename XDataType, typename YDataType, typename MeanDataType, typename VarDataType>
struct GridwiseWelfordSecondHalfLayernorm2d
{
    __device__ static void Run(const XDataType* __restrict__ p_x_grid,
                               const MeanDataType* __restrict__ p_mean_grid,
                               const VarDataType* __restrict__ p_var_grid,
                               YDataType* __restrict__ p_y_grid)
    {
        ignore = p_x_grid;
        ignore = p_mean_grid;
        ignore = p_var_grid;
        ignore = p_y_grid;
    } // run
};

} // namespace ck
