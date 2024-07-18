// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {

template <typename StaticDistributedTensor_, index_t... SliceBegins, index_t... SliceEnds>
__host__ __device__ constexpr auto get_slice_tile(const StaticDistributedTensor_& tile,
                                                  Sequence<SliceBegins...> slice_begins,
                                                  Sequence<SliceEnds...> slice_ends)
{
    using Distribution = decltype(StaticDistributedTensor_::GetTileDistribution());
    using DataType     = typename StaticDistributedTensor_::DataType;

    constexpr auto sliced_dstr_yidx_ylen =
        detail::slice_distribution_from_x(Distribution{}, slice_begins, slice_ends);

    constexpr auto sliced_dstr      = sliced_dstr_yidx_ylen.template At<0>();
    constexpr auto sliced_y_origins = sliced_dstr_yidx_ylen.template At<1>();
    constexpr auto sliced_y_lengths = sliced_dstr_yidx_ylen.template At<2>();

    auto sliced_tensor = make_static_distributed_tensor<DataType>(sliced_dstr);

    sliced_tensor.GetThreadBuffer() = tile.GetYSlicedThreadData(sliced_y_origins, sliced_y_lengths);

    return sliced_tensor;
}

template <typename DstStaticDistributedTensor_,
          typename SrcStaticDistributedTensor_,
          index_t... SliceBegins,
          index_t... SliceEnds>
__host__ __device__ constexpr auto set_slice_tile(DstStaticDistributedTensor_& dst_tile,
                                                  const SrcStaticDistributedTensor_& src_tile,
                                                  Sequence<SliceBegins...> slice_begins,
                                                  Sequence<SliceEnds...> slice_ends)
{
    using DstDistribution = decltype(DstStaticDistributedTensor_::GetTileDistribution());
    // using SrcDistribution = decltype(SrcStaticDistributedTensor_::GetTileDistribution());

    constexpr auto sliced_dstr_yidx_ylen =
        detail::slice_distribution_from_x(DstDistribution{}, slice_begins, slice_ends);

    // constexpr auto sliced_dstr      = sliced_dstr_yidx_ylen.template At<0>();
    constexpr auto sliced_y_origins = sliced_dstr_yidx_ylen.template At<1>();
    constexpr auto sliced_y_lengths = sliced_dstr_yidx_ylen.template At<2>();

    // static_assert(is_same_v<decltype(sliced_dstr), SrcDistribution>, "wrong!");

    dst_tile.SetYSlicedThreadData(sliced_y_origins, sliced_y_lengths, src_tile.GetThreadBuffer());
}

} // namespace tile_program
} // namespace ck
