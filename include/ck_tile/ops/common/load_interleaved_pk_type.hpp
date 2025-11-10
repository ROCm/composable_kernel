// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

template <class T>
struct is_pk_int4 : std::false_type
{
};
template <>
struct is_pk_int4<pk_int4_t> : std::true_type
{
};

template <typename ComputeDataType, index_t UnaryOpSize>
struct InterleavedPKTypeLoader
{
    template <typename WarpWindow, typename WarpTile>
    CK_TILE_DEVICE static void load_interleaved_pk_type(WarpTile& warp_tile,
                                                        const WarpWindow& warp_window)
    {
        const element_wise::PassThroughPack8 elementwise_op{};

        static_assert(WarpTile::get_thread_buffer_size() % UnaryOpSize == 0);
        constexpr index_t thread_buffer_size = WarpTile::get_thread_buffer_size() / UnaryOpSize;
        const auto in_dstr_tensors           = load_tile(warp_window);

        using ComputeVectorType = ComputeDataType __attribute__((ext_vector_type(UnaryOpSize)));
        static_for<0, thread_buffer_size, 1>{}([&](auto i) {
            elementwise_op(warp_tile.get_thread_buffer().template get_as<ComputeVectorType>()(i),
                           in_dstr_tensors.get_thread_buffer().template get_as<pk_int4x4_t>()[i]);
        });
    }
};

template <typename BDataType,
          typename ComputeDataType,
          index_t UnaryOpSize,
          typename WarpTile,
          typename WarpWindow>
CK_TILE_DEVICE void load_int4_tile(WarpTile& dst, const WarpWindow& src)
{
    if constexpr(is_pk_int4<std::remove_cv_t<BDataType>>::value)
    {
        InterleavedPKTypeLoader<ComputeDataType, UnaryOpSize>::load_interleaved_pk_type(dst, src);
    }
    else
    {
        dst = load_tile(src);
    }
}

} // namespace ck_tile
