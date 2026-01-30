// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

template <typename SrcDataType, typename DstDataType, index_t UnaryOpSize>
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

        // NOTE: we rely on types packing neatly here
        using RawSrcType          = typename SrcDataType::type;
        constexpr auto PackedSize = numeric_traits<SrcDataType>::PackedSize;

        using SrcVectorType = ext_vector_t<RawSrcType, UnaryOpSize / PackedSize>;
        using DstVectorType = ext_vector_t<DstDataType, UnaryOpSize>;
        static_for<0, thread_buffer_size, 1>{}([&](auto i) {
            elementwise_op(warp_tile.get_thread_buffer().template get_as<DstVectorType>()(i),
                           in_dstr_tensors.get_thread_buffer().template get_as<SrcVectorType>()[i]);
        });
    }
};

template <typename SrcDataType,
          typename DstDataType,
          index_t UnaryOpSize,
          bool LoadTranspose = false,
          typename WarpTile,
          typename WarpWindow>
CK_TILE_DEVICE void load_int4_tile(WarpTile& dst, const WarpWindow& src)
{
    if constexpr(is_packed_type_v<SrcDataType>)
    {
        static_assert(!LoadTranspose, "LoadTranspose not supported with pk_int4_t or pk_fp4_t");
        InterleavedPKTypeLoader<SrcDataType, DstDataType, UnaryOpSize>::load_interleaved_pk_type(
            dst, src);
    }
    else if constexpr(LoadTranspose)
    {
        dst = load_tile_transpose(src);
    }
    else
    {
        load_tile(dst, src);
    }
}

} // namespace ck_tile
