// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/ops/fmha/detail/fmha_tdm_prepared_k_read.hpp"

namespace ck_tile {

struct FmhaTdmAffineKRead
{
    template <index_t IAccess, index_t KPhysicalStride, index_t NumAccess>
    CK_TILE_HOST_DEVICE static constexpr index_t ByteDelta()
    {
        static_assert(NumAccess == 8 || NumAccess == 16 || NumAccess == 24);
        static_assert(IAccess >= 0 && IAccess < NumAccess);
        constexpr index_t kAccessesPerRowGroup = NumAccess / 2;
        return (IAccess / kAccessesPerRowGroup) * 16 * KPhysicalStride * 2 +
               (IAccess % kAccessesPerRowGroup) * 32;
    }

    template <index_t Stage, index_t IAccess, index_t KPhysicalStride, typename TileWindow>
    CK_TILE_HOST_DEVICE static constexpr bool ValidateStageDelta()
    {
        static_assert(Stage >= 0 && Stage < 4);
        using Base         = typename TileWindow::Base;
        using SFCYs        = typename Base::Traits::SFC_Ys;
        using Distribution = typename Base::TileDstr;
        using View = remove_cvref_t<decltype(std::declval<TileWindow>().get_bottom_tensor_view())>;
        using Descriptor = typename View::TensorDesc;
        static_assert(TileWindow::NumAccessPerCoord == 8 || TileWindow::NumAccessPerCoord == 16 ||
                      TileWindow::NumAccessPerCoord == 24);
        static_assert(IAccess >= 0 && IAccess < TileWindow::NumAccessPerCoord);
        static_assert(Distribution::NDimP == 2 && Distribution::NDimX == 2);
        static_assert(Descriptor::is_static());
        constexpr auto distribution = Distribution{};
        constexpr auto descriptor   = Descriptor{};
        constexpr auto zero_index   = SFCYs::get_index(number<0>{});
        constexpr auto input_index  = SFCYs::get_index(number<IAccess>{});

        for(index_t warp = 0; warp < 4; ++warp)
        {
            for(index_t lane = 0; lane < 32; ++lane)
            {
                const array<index_t, 2> partition{warp, lane};
                const auto zero = make_tensor_adaptor_coordinate(
                    distribution.get_ps_ys_to_xs_adaptor(),
                    container_concat(partition, to_array<index_t, zero_index.size()>(zero_index)));
                const auto next = make_tensor_adaptor_coordinate(
                    distribution.get_ps_ys_to_xs_adaptor(),
                    container_concat(partition,
                                     to_array<index_t, input_index.size()>(input_index)));
                // Only next_offset - zero_offset is checked: the affine descriptor
                // cancels the shared origin. This representative four-stage origin
                // proves the same access delta for N64's two-stage windows; it does
                // not select the production stage or its absolute read address.
                const auto origin = make_multi_index(((Stage + 1) % 4) * 32, 0);
                const auto zero_offset =
                    descriptor.calculate_offset(origin + zero.get_bottom_index());
                const auto next_offset =
                    descriptor.calculate_offset(origin + next.get_bottom_index());
                if((next_offset - zero_offset) * sizeof(typename Base::DataType) !=
                   ByteDelta<IAccess, KPhysicalStride, TileWindow::NumAccessPerCoord>())
                    return false;
            }
        }
        return true;
    }

    template <index_t Stage,
              index_t IAccess,
              index_t KPhysicalStride,
              typename DistributedTensor,
              typename TileWindow>
    CK_TILE_DEVICE static void Load(DistributedTensor& dst,
                                    const TileWindow& window,
                                    const FmhaTdmPreparedKRead::Address& base)
    {
        static_assert(ValidateStageDelta<Stage, IAccess, KPhysicalStride, TileWindow>());
        // Preserve each access's validity predicate and destination scatter.
        auto address         = FmhaTdmPreparedKRead::Prepare<IAccess>(window);
        address.byte_address = base.byte_address +
                               ByteDelta<IAccess, KPhysicalStride, TileWindow::NumAccessPerCoord>();
        FmhaTdmPreparedKRead::Load<IAccess>(dst, window, address);
    }
};

} // namespace ck_tile
