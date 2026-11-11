// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/detail/fmha_tdm_sched_fp8_config.hpp"

namespace ck_tile {

// Byte offset of access IAccess from the lane's first access; both P/Y->X and LDS maps are affine.
template <index_t IAccess, typename TileWindow>
CK_TILE_DEVICE index_t fmha_tdm_sched_fp8_access_offset(const TileWindow& tile_window)
{
    using Base                 = typename TileWindow::Base;
    using TileDstr             = typename Base::TileDstr;
    constexpr auto input_index = Base::Traits::SFC_Ys::get_index(number<IAccess>{});
    const auto access_xs       = make_tensor_adaptor_coordinate(
                               TileDstr{}.get_ps_ys_to_xs_adaptor(),
                               container_concat(array<index_t, TileDstr::NDimP>{},
                                                to_array<index_t, input_index.size()>(input_index)))
                               .get_bottom_index();
    return tile_window.get_bottom_tensor_view().get_tensor_descriptor().calculate_offset(
               access_xs) *
           static_cast<index_t>(sizeof(typename Base::DataType));
}

// LDS byte address of this lane's first access.
template <typename TileWindow>
CK_TILE_DEVICE uint32_t fmha_tdm_sched_fp8_lane_address(const TileWindow& tile_window)
{
    using Base       = typename TileWindow::Base;
    using TileDstr   = typename Base::TileDstr;
    const auto& view = tile_window.get_bottom_tensor_view();
    const auto lane_xs =
        make_tensor_adaptor_coordinate(
            TileDstr{}.get_ps_ys_to_xs_adaptor(),
            container_concat(get_partition_index(TileDstr{}), array<index_t, TileDstr::NDimY>{}))
            .get_bottom_index();
    const index_t offset =
        view.get_tensor_descriptor().calculate_offset(tile_window.get_window_origin() + lane_xs);
    return fmha_tdm_sched_fp8_lds_offset(view.get_buffer_view().p_data_) +
           offset * static_cast<index_t>(sizeof(typename Base::DataType));
}

// LDS window whose NumBases equal access runs each read at a constant offset from a pinned base.
// SplitBytes > 0: two bases, accesses at or past SplitBytes from the first use the second.
template <typename Window, index_t NumBases, index_t SplitBytes = 0>
struct FmhaTdmSchedFp8PinnedWindow
{
    static constexpr index_t NumAccessPerCoord = Window::NumAccessPerCoord;
    static constexpr index_t kRun              = NumAccessPerCoord / NumBases;
    static_assert(NumAccessPerCoord % NumBases == 0);
    static_assert(SplitBytes == 0 || NumBases == 2);
    using Desc = remove_cvref_t<
        decltype(std::declval<const Window&>().get_bottom_tensor_view().get_tensor_descriptor())>;
    static_assert(Desc::ntransform_ == 1 &&
                      remove_cvref_t<decltype(std::declval<const Desc&>().get_transforms().at(
                              number<0>{}))>::get_type_enum() == coord_transform_enum::embed,
                  "constant offsets from a pinned base need an affine (naive) LDS descriptor");

    const Window& window;
    const int32_t* bases;

    // Unpinned bases of each run; the caller pins them so later reads need no address VALU.
    CK_TILE_DEVICE static void ComputeBases(const Window& w, int32_t* out)
    {
        const uint32_t lane = fmha_tdm_sched_fp8_lane_address(w);
        if constexpr(SplitBytes > 0)
        {
            out[0] = out[1] = static_cast<int32_t>(lane);
            return;
        }
        static_for<0, NumBases, 1>{}([&](auto i) {
            out[i] = static_cast<int32_t>(
                lane + fmha_tdm_sched_fp8_access_offset<decltype(i)::value * kRun>(w));
        });
    }

    template <index_t IAccess>
    CK_TILE_DEVICE uint32_t Address() const
    {
        if constexpr(SplitBytes > 0)
        {
            const index_t off = fmha_tdm_sched_fp8_access_offset<IAccess>(window);
            return static_cast<uint32_t>(off >= SplitBytes ? bases[1] : bases[0]) + off;
        }
        constexpr index_t run = IAccess / kRun;
        return static_cast<uint32_t>(bases[run]) +
               (fmha_tdm_sched_fp8_access_offset<IAccess>(window) -
                fmha_tdm_sched_fp8_access_offset<run * kRun>(window));
    }
};

// One scheduled non-transposed LDS read of ReadBytes per access (b64 for FP8 K128 lane runs).
template <index_t ExpectedAccesses, index_t ReadBytes>
struct FmhaTdmSchedFp8Load
{
    // Thread-buffer offset of element J of access IAccess.
    template <typename Base, index_t IAccess>
    CK_TILE_HOST_DEVICE static constexpr index_t DestinationOffset(index_t j)
    {
        using Traits               = typename Base::Traits;
        constexpr auto input_index = Traits::SFC_Ys::get_index(number<IAccess>{});
        array<index_t, Base::NDimY> idx{};
        static_for<0, Base::NDimY, 1>{}(
            [&](auto d) { idx(d) = input_index[d] + (d == Traits::VectorDimY ? j : 0); });
        return typename Base::TileDstr{}.get_ys_to_d_descriptor().calculate_offset(idx);
    }

    template <index_t IInstruction,
              bool DwordStore = false,
              typename DistributedTensor,
              typename TileWindow,
              index_t NumBases,
              index_t SplitBytes>
    CK_TILE_DEVICE static void
    LoadInstruction(DistributedTensor& dst_tensor,
                    const FmhaTdmSchedFp8PinnedWindow<TileWindow, NumBases, SplitBytes>& pinned)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using Base       = typename TileWindow::Base;
        using Traits     = typename Base::Traits;
        using DataType   = typename Base::DataType;
        using VectorType = typename Traits::vector_t;

        static_assert(TileWindow::NumAccessPerCoord == ExpectedAccesses);
        static_assert(sizeof(VectorType) == ReadBytes, "each scheduled event is one LDS read");
        static_assert(Traits::PackedSize == 1);
        static_assert(IInstruction >= 0 && IInstruction < TileWindow::NumAccessPerCoord);
        static_assert(
            std::is_same_v<remove_cvref_t<typename DistributedTensor::DataType>, DataType>);
        static_assert(
            std::is_same_v<remove_cvref_t<typename DistributedTensor::StaticTileDistribution>,
                           remove_cvref_t<typename Base::TileDstr>>);

        using raw_t = int32_t __attribute__((ext_vector_type(ReadBytes / 4)));
        // One vector store per access keeps a loop-carried tile in dwords instead of bytes.
        if constexpr(DwordStore)
        {
            constexpr index_t base = DestinationOffset<Base, IInstruction>(0);
            static_assert(base % ReadBytes == 0);
            static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                static_assert(DestinationOffset<Base, IInstruction>(decltype(j)::value) ==
                              base + j);
            });
            dst_tensor.get_thread_buffer().template get_as<raw_t>().at(number<base / ReadBytes>{}) =
                *fmha_tdm_sched_fp8_lds_ptr<const raw_t>(pinned.template Address<IInstruction>());
            return;
        }
        const VectorType value = bit_cast<VectorType>(
            *fmha_tdm_sched_fp8_lds_ptr<const raw_t>(pinned.template Address<IInstruction>()));
        static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
            constexpr index_t destination_offset =
                DestinationOffset<Base, IInstruction>(decltype(j)::value);
            dst_tensor.get_thread_buffer().template at<destination_offset>() =
                value.template get_as<DataType>()[j];
        });
#else
        ignore = number<IInstruction>{};
        ignore = dst_tensor;
        ignore = pinned;
#endif
    }
};

// Transposed V read at a constant offset from the pinned lane base.
template <index_t ExpectedAccesses>
struct FmhaTdmSchedFp8TransposeLoad
{
    template <index_t IAccess,
              typename DistributedTensor,
              typename TileWindow,
              index_t NumBases,
              index_t SplitBytes>
    CK_TILE_DEVICE static void
    LoadAccess(DistributedTensor& dst_tensor,
               const FmhaTdmSchedFp8PinnedWindow<TileWindow, NumBases, SplitBytes>& pinned)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx125__)
        using Base            = typename TileWindow::Base;
        using Traits          = typename Base::Traits;
        using SFCYs           = typename Traits::SFC_Ys;
        using DataType        = typename Base::DataType;
        using TransposePolicy = DefaultTranspose<DataType>;
        using VectorType      = typename Traits::vector_t;

        static_assert(TileWindow::NumAccessPerCoord == ExpectedAccesses);
        static_assert(IAccess >= 0 && IAccess < TileWindow::NumAccessPerCoord);
        static_assert(
            std::is_same_v<remove_cvref_t<typename DistributedTensor::DataType>, DataType>);

        constexpr auto output_distribution = typename DistributedTensor::StaticTileDistribution{};
        using ExpectedEncoding =
            typename OutputTileDistributionTraits<typename Base::TileDstr::DstrEncode,
                                                  DataType>::TransposedDstrEncode;
        using ExpectedDistribution = decltype(make_static_tile_distribution(ExpectedEncoding{}));
        static_assert(
            detail::is_transpose_output_compatible_v<ExpectedDistribution,
                                                     remove_cvref_t<decltype(output_distribution)>,
                                                     DataType>);

        constexpr auto input_index = SFCYs::get_index(number<IAccess>{});
        const VectorType value =
            bit_cast<VectorType>(amd_transpose_load_to_vgpr<DataType, Traits::ScalarPerVector>(
                reinterpret_cast<const DataType*>(
                    static_cast<uintptr_t>(pinned.template Address<IAccess>()))));

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
#else
        ignore = number<IAccess>{};
        ignore = dst_tensor;
        ignore = pinned;
#endif
    }
};

} // namespace ck_tile
