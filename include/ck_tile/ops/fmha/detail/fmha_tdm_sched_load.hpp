// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <index_t ExpectedAccesses = -1>
struct FmhaTdmSchedTransposeLoad
{
    template <index_t IAccess, typename DistributedTensor, typename TileWindow_>
    CK_TILE_DEVICE static void LoadAccess(DistributedTensor& dst_tensor,
                                          const TileWindow_& tile_window)
    {
        using TileWindow      = remove_cvref_t<TileWindow_>;
        using Base            = typename TileWindow::Base;
        using Traits          = typename Base::Traits;
        using SFCYs           = typename Traits::SFC_Ys;
        using DataType        = typename Base::DataType;
        using TransposePolicy = DefaultTranspose<DataType>;
        using VectorType      = typename Traits::vector_t;

        static_assert(ExpectedAccesses == -1 || TileWindow::NumAccessPerCoord == ExpectedAccesses);
        static_assert(IAccess >= 0 && IAccess < TileWindow::NumAccessPerCoord);
        static_assert(
            std::is_same_v<remove_cvref_t<typename DistributedTensor::DataType>, DataType>);

        constexpr auto input_distribution  = typename Base::TileDstr{};
        constexpr auto output_distribution = typename DistributedTensor::StaticTileDistribution{};
        using ExpectedEncoding =
            typename OutputTileDistributionTraits<typename Base::TileDstr::DstrEncode,
                                                  DataType>::TransposedDstrEncode;
        using ExpectedDistribution = decltype(make_static_tile_distribution(ExpectedEncoding{}));
        static_assert(
            detail::is_transpose_output_compatible_v<ExpectedDistribution,
                                                     remove_cvref_t<decltype(output_distribution)>,
                                                     DataType>);

        constexpr auto access           = number<IAccess>{};
        constexpr auto input_index      = SFCYs::get_index(access);
        const auto partition_index      = get_partition_index(input_distribution);
        const auto window_adaptor_coord = make_tensor_adaptor_coordinate(
            input_distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(partition_index, to_array<index_t, input_index.size()>(input_index)));
        const auto bottom_tensor_index =
            tile_window.get_window_origin() + window_adaptor_coord.get_bottom_index();
        const auto bottom_tensor_coord = make_tensor_coordinate(
            tile_window.get_bottom_tensor_view().get_tensor_descriptor(), bottom_tensor_index);

        const VectorType value =
            tile_window.get_bottom_tensor_view()
                .template get_transpose_vectorized_elements<VectorType>(bottom_tensor_coord, 0);

        constexpr auto group = TransposePolicy::group_func;
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

            dst_tensor.get_thread_buffer().template at<output_offset>() =
                value.template get_as<DataType>()[j / Traits::PackedSize];
        });
    }
};

template <index_t ExpectedAccesses>
struct FmhaTdmSchedLoad
{
    template <index_t IInstruction, typename DistributedTensor, typename TileWindow_>
    CK_TILE_DEVICE static void LoadInstruction(DistributedTensor& dst_tensor,
                                               const TileWindow_& tile_window)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using TileWindow = remove_cvref_t<TileWindow_>;
        using Base       = typename TileWindow::Base;
        using Traits     = typename Base::Traits;
        using SFCYs      = typename Traits::SFC_Ys;
        using DataType   = typename Base::DataType;
        using VectorType = typename Traits::vector_t;

        static_assert(TileWindow::NumAccessPerCoord == ExpectedAccesses);
        static_assert(sizeof(VectorType) == 16, "Each scheduled K event must issue one b128 load");
        static_assert(IInstruction >= 0 && IInstruction < TileWindow::NumAccessPerCoord);
        static_assert(
            std::is_same_v<remove_cvref_t<typename DistributedTensor::DataType>, DataType>);
        static_assert(
            std::is_same_v<remove_cvref_t<typename DistributedTensor::StaticTileDistribution>,
                           remove_cvref_t<typename Base::TileDstr>>);

        constexpr auto distribution     = typename Base::TileDstr{};
        constexpr auto access           = number<IInstruction>{};
        constexpr auto input_index      = SFCYs::get_index(access);
        const auto partition_index      = get_partition_index(distribution);
        const auto window_adaptor_coord = make_tensor_adaptor_coordinate(
            distribution.get_ps_ys_to_xs_adaptor(),
            container_concat(partition_index, to_array<index_t, input_index.size()>(input_index)));
        const auto bottom_tensor_index =
            tile_window.get_window_origin() + window_adaptor_coord.get_bottom_index();
        const auto bottom_tensor_coord = make_tensor_coordinate(
            tile_window.get_bottom_tensor_view().get_tensor_descriptor(), bottom_tensor_index);

        const VectorType value =
            tile_window.get_bottom_tensor_view().template get_vectorized_elements<VectorType>(
                bottom_tensor_coord, 0, bool_constant<true>{});

        static_for<0, Traits::ScalarPerVector, Traits::PackedSize>{}([&](auto j) {
            constexpr auto destination_index = generate_tuple(
                [&](auto dim) {
                    return dim == Traits::VectorDimY ? (input_index[dim] + j) : input_index[dim];
                },
                number<Base::NDimY>{});
            constexpr index_t destination_offset =
                distribution.get_ys_to_d_descriptor().calculate_offset(destination_index) /
                Traits::PackedSize;

            dst_tensor.get_thread_buffer().template at<destination_offset>() =
                value.template get_as<DataType>()[j / Traits::PackedSize];
        });
#else
        ignore = number<IInstruction>{};
        ignore = dst_tensor;
        ignore = tile_window;
#endif
    }
};

} // namespace ck_tile
