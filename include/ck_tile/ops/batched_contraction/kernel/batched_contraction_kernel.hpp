// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/batched_contraction/pipeline/batched_contraction_problem.hpp"

namespace ck_tile {

    template <typename Problem_, typename Policy_ = void>

    struct BatchedContractionKernel
    {
        using Problem = ck_tile::remove_cvref_t<Problem_>;
        using ADataType = ck_tile::remove_cvref_t<typename Problem::ADataType>;
        using BDataType = ck_tile::remove_cvref_t<typename Problem::BDataType>;
        using EDataType = ck_tile::remove_cvref_t<typename Problem::EDataType>;

        static constexpr ck_tile::index_t NumDimG = Problem::NumDimG;
        static constexpr ck_tile::index_t NumDimM = Problem::NumDimM;
        static constexpr ck_tile::index_t NumDimN = Problem::NumDimN;
        static constexpr ck_tile::index_t NumDimK = Problem::NumDimK;

        template <typename ATensor, typename BTensor, typename ETensor>
        CK_TILE_DEVICE void operator()(const ATensor& a_tensor,
                                       const BTensor& b_tensor,
                                       ETensor& e_tensor) const
        {
            (void)a_tensor;
            (void)b_tensor;
            (void)e_tensor;            
        }

    }

    CK_TILE_HOST static constexpr bool IsSupportedArguments()
    {
        return true;
    }
}