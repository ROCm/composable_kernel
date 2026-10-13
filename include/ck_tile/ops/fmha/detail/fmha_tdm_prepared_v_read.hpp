// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_tdm_prepared_k_read.hpp"

namespace ck_tile {

struct FmhaTdmPreparedVRead
{
    template <index_t IAccess, typename DistributedTensor, typename TileWindow>
    CK_TILE_DEVICE static void Load(DistributedTensor& dst,
                                    const TileWindow& window,
                                    const FmhaTdmPreparedKRead::Address& address)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using Base            = typename TileWindow::Base;
        using Traits          = typename Base::Traits;
        using SFCYs           = typename Traits::SFC_Ys;
        using DataType        = typename Base::DataType;
        using VectorType      = typename Traits::vector_t;
        using TransposePolicy = DefaultTranspose<DataType>;
        static_assert(TileWindow::NumAccessPerCoord == 8 || TileWindow::NumAccessPerCoord == 16);
        static_assert(IAccess >= 0 && IAccess < TileWindow::NumAccessPerCoord);
        static_assert(Traits::PackedSize == 1 && sizeof(DataType) == 2);
        static_assert(sizeof(VectorType) == 16);
        static_assert(std::is_same_v<typename DistributedTensor::DataType, DataType>);

        constexpr auto output_distribution = typename DistributedTensor::StaticTileDistribution{};
        using ExpectedEncoding =
            typename OutputTileDistributionTraits<typename Base::TileDstr::DstrEncode,
                                                  DataType>::TransposedDstrEncode;
        using ExpectedDistribution = decltype(make_static_tile_distribution(ExpectedEncoding{}));
        static_assert(
            detail::is_transpose_output_compatible_v<ExpectedDistribution,
                                                     remove_cvref_t<decltype(output_distribution)>,
                                                     DataType>);

        // Same transpose builtin and scatter as the original load; only the address is prepared.
        const VectorType value = [&]() -> VectorType {
            if(!address.valid)
                return window.get_bottom_tensor_view()
                    .get_buffer_view()
                    .template transpose_get<VectorType>(0, 0, false);
            if constexpr(std::is_same_v<DataType, bf16_t>)
            {
                using NativeVector = __bf16 __attribute__((vector_size(16)));
                using LdsPointer   = __attribute__((address_space(3))) NativeVector*;
                return bit_cast<VectorType>(__builtin_amdgcn_ds_load_tr16_b128_v8bf16(
                    reinterpret_cast<LdsPointer>(address.byte_address)));
            }
            else
            {
                static_assert(std::is_same_v<DataType, half_t>);
                using NativeVector = __fp16 __attribute__((vector_size(16)));
                using LdsPointer   = __attribute__((address_space(3))) NativeVector*;
                return bit_cast<VectorType>(__builtin_amdgcn_ds_load_tr16_b128_v8f16(
                    reinterpret_cast<LdsPointer>(address.byte_address)));
            }
        }();

        constexpr auto input_index = SFCYs::get_index(number<IAccess>{});
        constexpr auto group       = TransposePolicy::group_func;
        static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
            constexpr auto source_index = generate_tuple(
                [&](auto dim) {
                    return dim == Traits::VectorDimY ? (input_index[dim] + j) : input_index[dim];
                },
                number<Base::NDimY>{});
            constexpr auto output_index = group(source_index);
            constexpr index_t output_offset =
                output_distribution.get_ys_to_d_descriptor().calculate_offset(output_index) /
                Traits::PackedSize;
            dst.get_thread_buffer().template at<output_offset>() =
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
