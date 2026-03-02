// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

template <typename SrcDataType, typename DstDataType, index_t UnaryOpSize>
struct ConverterLoader
{
    template <typename WarpWindow, typename WarpTile>
    CK_TILE_DEVICE static void load_interleaved_pk_type(WarpTile& dst, const WarpWindow& src)
    {
        static_assert(WarpTile::get_thread_buffer_size() % UnaryOpSize == 0);
        constexpr index_t thread_buffer_size = WarpTile::get_thread_buffer_size() / UnaryOpSize;
        const auto tmp                       = load_tile(src);

        // NOTE: we rely on types packing neatly here
        using RawSrcType          = typename SrcDataType::type;
        constexpr auto PackedSize = numeric_traits<SrcDataType>::PackedSize;

        using SrcVectorType = ext_vector_t<RawSrcType, UnaryOpSize / PackedSize>;
        using DstVectorType = ext_vector_t<DstDataType, UnaryOpSize>;
        static_for<0, thread_buffer_size, 1>{}([&](auto i) {
            const element_wise::PassThroughPack8 elementwise_op{};

            elementwise_op(dst.get_thread_buffer().template get_as<DstVectorType>()(i),
                           tmp.get_thread_buffer().template get_as<SrcVectorType>()[i]);
        });
    }
};

template <index_t UnaryOpSize, bool LoadTranspose = false, typename WarpTile, typename WarpWindow>
CK_TILE_DEVICE void load_and_convert_tile(WarpTile& dst, const WarpWindow& src)
{
    using SrcDataType = typename WarpWindow::Base::DataType;
    using DstDataType = typename WarpTile::DataType;

    if constexpr(is_packed_type_v<SrcDataType> && !is_packed_type_v<DstDataType>)
    {
        static_assert(!LoadTranspose, "LoadTranspose not supported with pk_int4_t or pk_fp4_t");
        ConverterLoader<SrcDataType, DstDataType, UnaryOpSize>::load_interleaved_pk_type(dst, src);
    }
    else if constexpr(LoadTranspose)
    {
        load_tile_transpose(dst, src);
    }
    else
    {
        load_tile(dst, src);
    }
}

} // namespace ck_tile
