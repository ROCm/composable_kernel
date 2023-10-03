// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/shuffle_distributed_tensor_impl_in_thread.hpp"

namespace ck {
namespace tile_program {

template <typename OutTensor, typename InTensor>
__device__ void shuffle_distributed_tensor(OutTensor& out, const InTensor& in)
{
    using InDataType  = typename InTensor::DataType;
    using OutDataType = typename OutTensor::DataType;

    using InDstrEncode  = typename InTensor::StaticTileDistribution::DstrEncode;
    using OutDstrEncode = typename OutTensor::StaticTileDistribution::DstrEncode;

    // type convert
    const auto in_tmp = tile_elementwise_in(type_convert<OutDataType, InDataType>, in);

    // shuffle
    if constexpr(InDstrEncode::rs_lengths_ == OutDstrEncode::rs_lengths_ &&
                 InDstrEncode::hs_lengthss_ == OutDstrEncode::hs_lengthss_ &&
                 InDstrEncode::ps_to_rhss_major_ == OutDstrEncode::ps_to_rhss_major_ &&
                 InDstrEncode::ps_to_rhss_minor_ == OutDstrEncode::ps_to_rhss_minor_ &&
                 InDstrEncode::NDimY == OutDstrEncode::NDimY)
    {
        detail::shuffle_distributed_tensor_impl_in_thread(out, in_tmp);
    }
    else
    {
        // NOT implemented
    }
}

} // namespace tile_program
} // namespace ck
