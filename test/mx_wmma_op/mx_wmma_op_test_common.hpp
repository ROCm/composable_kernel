// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "gtest/gtest.h"

#include "mx_wmma_op.hpp"

using ck::bf6_t;
using ck::bf8_t;
using ck::e4m3_scale_t;
using ck::e5m3_scale_t;
using ck::e8m0_bexp_t;
using ck::f4_t;
using ck::f6_t;
using ck::f8_t;
using ck::type_convert;

// Shared constant: default (random) initialisation mode used by most tests.
inline constexpr ck::index_t common_init = -1;

/**
 * @brief Run the test for the given WMMA scale instruction
 *
 * @param init - selects initialization algorithm for A and B tensors
 */
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AType,
          typename BType,
          typename CType,
          typename AScaleType,
          typename BScaleType,
          ck::WMMA_SCALE wmma>
bool run_mx_wmma_test(ck::index_t init)
{
    static_assert((wmma == ck::WMMA_SCALE::SCALE_F32_16x16x128 ||
                   wmma == ck::WMMA_SCALE::SCALE16_F32_16x16x128),
                  "Only SCALE_F32_16x16x128 and SCALE16_F32_16x16x128 are supported");

    using AccType = float; // only F32 instructions supported

    // WMMA scale instruction parameters
    ck::mfma_type<static_cast<ck::MfmaInstr>(wmma)> wmma_instr;
    constexpr auto BLOCK_M = wmma_instr.m_per_blk;
    constexpr auto BLOCK_N = wmma_instr.n_per_blk;
    constexpr auto BLOCK_K = wmma_instr.num_input_blks * wmma_instr.k_per_blk;
    constexpr auto BLOCK_X = wmma_instr.scale_blk_size; // scaling vector size

    const auto mx_wmma_kernel = ck::matmul<AType,
                                           BType,
                                           AScaleType,
                                           BScaleType,
                                           CType,
                                           AccType,
                                           BLOCK_M,
                                           BLOCK_N,
                                           BLOCK_K,
                                           BLOCK_X,
                                           ALayout,
                                           BLayout,
                                           CLayout>;

    bool pass = true;

    pass = ck::mx_wmma_test::TestMXWMMA<decltype(mx_wmma_kernel),
                                        AType,
                                        BType,
                                        AScaleType,
                                        BScaleType,
                                        CType,
                                        ALayout,
                                        BLayout,
                                        CLayout,
                                        BLOCK_M,
                                        BLOCK_N,
                                        BLOCK_K,
                                        BLOCK_X>{}(mx_wmma_kernel, init);

    return pass;
}

/**
 * @brief Run the test for the given unscaled WMMA instruction (no scale types)
 *
 * @param init - selects initialization algorithm for A and B tensors
 */
template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AType,
          typename BType,
          typename CType,
          typename AccType,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K>
bool run_mx_wmma_unscaled_test(ck::index_t init)
{
    // Unscaled WMMA kernel parameters
    const auto mx_wmma_kernel = ck::matmul_unscaled<AType,
                                                    BType,
                                                    CType,
                                                    AccType,
                                                    BLOCK_M,
                                                    BLOCK_N,
                                                    BLOCK_K,
                                                    ALayout,
                                                    BLayout,
                                                    CLayout>;

    bool pass         = true;
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    pass = ck::mx_wmma_test::TestMXWMMAUnscaled<decltype(mx_wmma_kernel),
                                                AType,
                                                BType,
                                                CType,
                                                ALayout,
                                                BLayout,
                                                CLayout,
                                                BLOCK_M,
                                                BLOCK_N,
                                                BLOCK_K,
                                                PassThrough,
                                                PassThrough,
                                                PassThrough>{}(mx_wmma_kernel, init);

    return pass;
}
