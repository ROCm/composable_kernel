// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"

namespace ck_tile {

/**
 * Helper class for pattern-matching supported unary op sizes for mixed-precision transpose
 * conversion. This is used to select the appropriate PassThroughPack (e.g., PassThroughPack8,
 * PassThroughPack2) based on the vector length of the transpose load. Unsupported sizes will result
 * in a compile-time error.
 */
template <index_t UnaryOpSize_>
struct PassThroughPackSelector;

template <>
struct PassThroughPackSelector<8>
{
    using type = element_wise::PassThroughPack8;
};

template <>
struct PassThroughPackSelector<2>
{
    using type = element_wise::PassThroughPack2;
};

template <typename SrcDataType,
          typename DstDataType,
          index_t UnaryOpSize,
          bool LoadTranspose = false>
struct ConverterLoader
{
    template <typename WarpWindow, typename WarpTile>
    CK_TILE_DEVICE static void load_interleaved_pk_type(WarpTile& dst, const WarpWindow& src_window)
    {
        static_assert(!LoadTranspose, "LoadTranspose not supported with pk_int4_t or pk_fp4_t");
        static_assert(WarpTile::get_thread_buffer_size() % UnaryOpSize == 0);
        constexpr index_t thread_buffer_size = WarpTile::get_thread_buffer_size() / UnaryOpSize;
        const auto src                       = load_tile(src_window);

        // NOTE: we rely on types packing neatly here
        using RawSrcType          = typename SrcDataType::type;
        constexpr auto PackedSize = numeric_traits<SrcDataType>::PackedSize;

        using SrcVectorType = ext_vector_t<RawSrcType, UnaryOpSize / PackedSize>;
        using DstVectorType = ext_vector_t<DstDataType, UnaryOpSize>;
        static_for<0, thread_buffer_size, 1>{}([&](auto i) {
            constexpr typename PassThroughPackSelector<UnaryOpSize>::type elementwise_op{};

            elementwise_op(dst.get_thread_buffer().template get_as<DstVectorType>()(i),
                           src.get_thread_buffer().template get_as<SrcVectorType>()[i]);
        });
    }

    template <typename WarpWindow, typename WarpTile>
    CK_TILE_DEVICE static void load_with_type_convert(WarpTile& dst, const WarpWindow& src_window)
    {
        if constexpr(LoadTranspose)
        {
            if constexpr(std::is_same_v<SrcDataType, DstDataType>)
            {
                load_tile_transpose(dst, src_window);
            }
            else
            {
                constexpr typename PassThroughPackSelector<UnaryOpSize>::type elementwise_op{};

                load_tile_transpose_convert(dst, src_window, number<UnaryOpSize>{}, elementwise_op);
            }
        }
        else
        {
            if constexpr(std::is_same_v<SrcDataType, DstDataType>)
            {
                load_tile(dst, src_window);
            }
            else
            {
                auto tmp = load_tile(src_window);
                constexpr index_t thread_buffer_size =
                    WarpTile::get_thread_buffer_size() / UnaryOpSize;
                using SrcVectorType = ext_vector_t<SrcDataType, UnaryOpSize>;
                using DstVectorType = ext_vector_t<DstDataType, UnaryOpSize>;
                static_for<0, thread_buffer_size, 1>{}([&](auto i) {
                    constexpr typename PassThroughPackSelector<UnaryOpSize>::type elementwise_op{};
                    elementwise_op(dst.get_thread_buffer().template get_as<DstVectorType>()(i),
                                   tmp.get_thread_buffer().template get_as<SrcVectorType>()[i]);
                });
            }
        }
    }
};

template <index_t UnaryOpSize, bool LoadTranspose = false, typename WarpTile, typename WarpWindow>
CK_TILE_DEVICE void load_and_convert_tile(WarpTile& dst, const WarpWindow& src_window)
{
    using SrcDataType = typename WarpWindow::Base::DataType;
    using DstDataType = typename WarpTile::DataType;

    if constexpr(is_packed_type_v<SrcDataType> && !is_packed_type_v<DstDataType>)
    {
        ConverterLoader<SrcDataType, DstDataType, UnaryOpSize, LoadTranspose>::
            load_interleaved_pk_type(dst, src_window);
    }
    else
    {
        ConverterLoader<SrcDataType, DstDataType, UnaryOpSize, LoadTranspose>::
            load_with_type_convert(dst, src_window);
    }
}
} // namespace ck_tile
