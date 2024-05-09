// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

template <typename AccDataType_, typename KGradDataType_, typename VGradDataType_>
struct FmhaBwdEpilogueProblem
{
    using AccDataType   = remove_cvref_t<AccDataType_>;
    using KGradDataType = remove_cvref_t<KGradDataType_>;
    using VGradDataType = remove_cvref_t<VGradDataType_>;
};

template <typename Problem_, typename Policy_ = void>
struct FmhaBwdEpilogue
{
    using Problem       = remove_cvref_t<Problem_>;
    using AccDataType   = remove_cvref_t<typename Problem::AccDataType>;
    using KGradDataType = remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType = remove_cvref_t<typename Problem::VGradDataType>;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    template <typename KGradDramWindowTmp,
              typename VGradDramWindowTmp,
              typename KGradAccTile,
              typename VGradAccTile>
    CK_TILE_DEVICE auto operator()(KGradDramWindowTmp& dk_dram_window_tmp,
                                   VGradDramWindowTmp& dv_dram_window_tmp,
                                   const KGradAccTile& dk_acc_tile,
                                   const VGradAccTile& dv_acc_tile)
    {
        store_tile(dk_dram_window_tmp, cast_tile<KGradDataType>(dk_acc_tile));
        store_tile(dv_dram_window_tmp, cast_tile<VGradDataType>(dv_acc_tile));
    }
};
} // namespace ck_tile
