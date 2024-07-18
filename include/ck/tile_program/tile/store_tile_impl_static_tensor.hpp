// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"

namespace ck {
namespace tile_program {

template <typename SrcTileDistribution_, typename DstTileDistribution_, typename DataType_>
__device__ void
store_tile(StaticDistributedTensor<DataType_, DstTileDistribution_>& dst_dstr_tensor,
           const StaticDistributedTensor<DataType_, SrcTileDistribution_>& src_dstr_tensor)
{
    // static_assert(DstTileDistribution_==SrcTileDistribution_);
    dst_dstr_tensor.GetThreadBuffer() = src_dstr_tensor.GetThreadBuffer();
}

} // namespace tile_program
} // namespace ck
