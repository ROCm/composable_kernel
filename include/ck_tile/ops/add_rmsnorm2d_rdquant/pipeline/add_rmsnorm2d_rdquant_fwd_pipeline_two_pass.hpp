// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/rmsnorm2d/pipeline/rmsnorm2d_fwd_pipeline_default_policy.hpp"
#include <string>
#include <type_traits>

namespace ck_tile {

template <typename Problem_, typename Policy_ = AddRmsnorm2dRdquantFwdPipelineDefaultPolicy>
struct AddRmsnorm2dRdquantFwdPipelineTwoPass
{
    using Problem = ck_tile::remove_cvref_t<Problem_>;
    using Policy  = ck_tile::remove_cvref_t<Policy_>;

    using ADataType       = ck_tile::remove_cvref_t<typename Problem::ADataType>;
    using BDataType       = ck_tile::remove_cvref_t<typename Problem::BDataType>;
    using GammaDataType   = ck_tile::remove_cvref_t<typename Problem::GammaDataType>;
    using ComputeDataType = ck_tile::remove_cvref_t<typename Problem::ComputeDataType>;
    using XDataType       = ck_tile::remove_cvref_t<typename Problem::XDataType>;
    using YScaleDataType  = ck_tile::remove_cvref_t<typename Problem::YScaleDataType>;
    using QYDataType      = ck_tile::remove_cvref_t<typename Problem::QYDataType>;

    static constexpr bool kHasGamma = !std::is_same_v<GammaDataType, ck_tile::null_type>;
    static constexpr bool kSaveX    = Problem::kSaveX;

    static constexpr bool kNeedCrossWarpSync = Problem::kNeedCrossWarpSync;
    static constexpr bool kPadM = false; // TODO - BlockAddRmsnorm2dRdquantFwdProblem::kPadM
    static constexpr bool kPadN = Problem::kPadN;

    static constexpr const char* name = []() {
        if constexpr(kNeedCrossWarpSync)
            return "bpr_tp"; // block per row
        else
            return "wpr_tp"; // warp per row
    }();

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename AWindow,
              typename BWindow,
              typename GammaWindow,
              typename XWindow,
              typename YScaleWindow,
              typename QYWindow>
    CK_TILE_DEVICE auto operator()(const AWindow& a_window_,
                                   const BWindow& b_window_,
                                   const GammaWindow& gamma_window_,
                                   XWindow& x_window,
                                   YScaleWindow& yscale_window,
                                   QYWindow& y_window,
                                   ComputeDataType epsilon,
                                   ck_tile::index_t row_size,
                                   void* smem) const
    {
        auto a_window =
            make_tile_window(a_window_, Policy::template MakeABXBlockTileDistribution<Problem>());
        auto b_window =
            make_tile_window(b_window_, Policy::template MakeABXBlockTileDistribution<Problem>());
        const auto gamma_window = make_tile_window(
            gamma_window_, Policy::template MakeGammaBlockTileDistribution<Problem>());
    }
};
} // namespace ck_tile
