// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

struct FmhaTdmPreparedKRead
{
    struct Address
    {
        uint32_t byte_address = 0;
        bool valid            = false;
    };

    template <index_t IAccess, typename TileWindow>
    CK_TILE_DEVICE static Address Prepare(const TileWindow& window)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using Base     = typename TileWindow::Base;
        using Traits   = typename Base::Traits;
        using SFCYs    = typename Traits::SFC_Ys;
        using DataType = typename Base::DataType;
        static_assert(TileWindow::NumAccessPerCoord == 8 || TileWindow::NumAccessPerCoord == 16 ||
                      TileWindow::NumAccessPerCoord == 24);
        static_assert(IAccess >= 0 && IAccess < TileWindow::NumAccessPerCoord);
        static_assert(Traits::PackedSize == 1 && sizeof(DataType) == 2);

        constexpr auto distribution = typename Base::TileDstr{};
        constexpr auto input_index  = SFCYs::get_index(number<IAccess>{});
        const auto partition_index  = get_partition_index(distribution);
        const auto adaptor_coord    = make_tensor_adaptor_coordinate(
            distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(partition_index, to_array<index_t, input_index.size()>(input_index)));
        const auto& view = window.get_bottom_tensor_view();
        const auto coord =
            make_tensor_coordinate(view.get_tensor_descriptor(),
                                   window.get_window_origin() + adaptor_coord.get_bottom_index());
        const auto& buffer = view.get_buffer_view();
        using Buffer       = remove_cvref_t<decltype(buffer)>;
        static_assert(Buffer::get_address_space() == address_space_enum::lds);

        using LdsPointer   = const __attribute__((address_space(3))) DataType*;
        const auto pointer = c_style_pointer_cast<LdsPointer>(buffer.p_data_ + coord.get_offset());
        uint32_t address   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pointer));
        return {address,
                coordinate_has_valid_offset_assuming_top_index_is_valid(
                    view.get_tensor_descriptor(), coord)};
#else
        ignore = window;
        return {};
#endif
    }

    template <index_t IAccess, typename DistributedTensor, typename TileWindow>
    CK_TILE_DEVICE static void
    Load(DistributedTensor& dst, const TileWindow& window, const Address& address)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using Base             = typename TileWindow::Base;
        using Traits           = typename Base::Traits;
        using SFCYs            = typename Traits::SFC_Ys;
        using DataType         = typename Base::DataType;
        using VectorType       = typename Traits::vector_t;
        using NativeVector     = ext_vector_t<typename vector_traits<DataType>::scalar_type,
                                              vector_traits<VectorType>::vector_size>;
        using LdsVectorPointer = const __attribute__((address_space(3))) NativeVector*;
        static_assert(sizeof(VectorType) == 16 && sizeof(NativeVector) == 16);
        static_assert(
            std::is_same_v<remove_cvref_t<typename DistributedTensor::DataType>, DataType>);
        static_assert(std::is_same_v<typename DistributedTensor::StaticTileDistribution,
                                     typename Base::TileDstr>);

        const VectorType value =
            address.valid
                ? bit_cast<VectorType>(*reinterpret_cast<LdsVectorPointer>(address.byte_address))
                : window.get_bottom_tensor_view().get_buffer_view().template get<VectorType>(
                      index_t{0}, index_t{0}, false);
        constexpr auto distribution = typename Base::TileDstr{};
        constexpr auto input_index  = SFCYs::get_index(number<IAccess>{});
        static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
            constexpr auto destination_index = generate_tuple(
                [&](auto dim) {
                    return dim == Traits::VectorDimY ? (input_index[dim] + j) : input_index[dim];
                },
                number<Base::NDimY>{});
            constexpr index_t destination_offset =
                distribution.get_ys_to_d_descriptor().calculate_offset(destination_index) /
                Traits::PackedSize;
            dst.get_thread_buffer().template at<destination_offset>() =
                value.template get_as<DataType>()[j / Traits::PackedSize];
        });
#else
        ignore = dst;
        ignore = window;
        ignore = address;
#endif
    }
};

} // namespace ck_tile
