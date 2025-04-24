// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/algorithm/coordinate_transform.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/container/thread_buffer.hpp"
#include "ck_tile/core/container/statically_indexed_array.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t NumCoord,
          index_t i_access           = -1,
          bool oob_conditional_check = true>
CK_TILE_DEVICE auto
load_tile_transpose(const tile_window_with_static_distribution<BottomTensorView_,
                                                               WindowLengths_,
                                                               TileDistribution_,
                                                               NumCoord>& tile_window,
                    number<i_access>                     = {},
                    bool_constant<oob_conditional_check> = {})
{
    return tile_window.load_transpose(number<i_access>{}, bool_constant<oob_conditional_check>{});
}

} // namespace ck_tile
