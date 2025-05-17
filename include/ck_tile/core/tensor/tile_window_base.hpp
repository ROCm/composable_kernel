// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/arch/utility.hpp"
#include "ck_tile/core/algorithm/space_filling_curve.hpp"
#include "ck_tile/core/config.hpp"
#include "ck_tile/core/container/array.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/container/container_helper.hpp"
#include "ck_tile/core/tensor/static_distributed_tensor.hpp"
#include "ck_tile/core/tensor/tensor_adaptor.hpp"
#include "ck_tile/core/tensor/tile_distribution.hpp"
#include "ck_tile/core/utility/functional.hpp"
#include "ck_tile/core/utility/type_traits.hpp"

namespace ck_tile {

template <typename BottomTensorView_>
struct tile_window_base
{
    using BottomTensorView                    = remove_reference_t<BottomTensorView_>;
    using BottomTensorDesc                    = typename BottomTensorView::TensorDesc;
    static constexpr index_t NDimBottomTensor = BottomTensorDesc::get_num_of_dimension();
    using BottomTensorIndex                   = array<index_t, NDimBottomTensor>;

    BottomTensorIndex window_origin_;

    CK_TILE_DEVICE constexpr auto get_window_origin() const { return window_origin_; }
};
} // namespace ck_tile
