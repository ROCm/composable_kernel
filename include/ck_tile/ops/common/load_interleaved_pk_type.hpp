// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

template <typename DstDataType, index_t UnaryOpSize>
struct InterleavedPKTypeLoader
{
    template <typename WarpWindow, typename WarpTile>
    CK_TILE_DEVICE static void load_interleaved_pk_type(WarpTile& dst, const WarpWindow& src)
    {
        static_assert(WarpTile::get_thread_buffer_size() % UnaryOpSize == 0);
        constexpr index_t thread_buffer_size = WarpTile::get_thread_buffer_size() / UnaryOpSize;
        const auto tmp                       = load_tile(src);

        using DstVectorType = DstDataType __attribute__((ext_vector_type(UnaryOpSize)));
        static_for<0, thread_buffer_size, 1>{}([&](auto i) {
            const element_wise::PassThroughPack8 elementwise_op{};

            elementwise_op(dst.get_thread_buffer().template get_as<DstVectorType>()(i),
                           tmp.get_thread_buffer().template get_as<pk_int4x4_t>()[i]);
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
    if constexpr(std::is_same_v<SrcDataType, pk_int4_t>)
    {
        static_assert(!LoadTranspose, "LoadTranspose not supported with pk_int4_t");
        InterleavedPKTypeLoader<DstDataType, UnaryOpSize>::load_interleaved_pk_type(dst, src);
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
