// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file kernel_template.hpp
 * @brief Template declaration only - no instantiation here!
 *
 * This header declares the kernel template but does NOT instantiate it.
 * Instantiation happens in separate .cpp files for parallel compilation.
 *
 * Compilation model:
 *   kernel_template.hpp  - Template declaration (this file)
 *   kernel_fp16_rcr_128x128x32.cpp - Explicit instantiation
 *   kernel_fp16_rcr_256x256x64.cpp - Explicit instantiation
 *   kernel_bf16_rcr_128x128x32.cpp - Explicit instantiation
 *   ...
 *
 * Each .cpp file is a separate compilation unit = parallel compilation!
 * `make -j16` will compile 16 kernels simultaneously.
 */

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace ck_tile {
namespace dispatcher {

// =============================================================================
// Kernel configuration struct - compile-time parameters
// =============================================================================

template <typename AType_,
          typename BType_,
          typename CType_,
          typename AccType_,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          index_t TileM_,
          index_t TileN_,
          index_t TileK_,
          index_t WaveM_,
          index_t WaveN_,
          index_t WaveK_,
          index_t WarpM_,
          index_t WarpN_,
          index_t WarpK_,
          bool PadM_,
          bool PadN_,
          bool PadK_,
          index_t BlockSize_ = 256>
struct GemmKernel
{
    // Types
    using ADataType   = AType_;
    using BDataType   = BType_;
    using CDataType   = CType_;
    using AccDataType = AccType_;
    using ALayout     = ALayout_;
    using BLayout     = BLayout_;
    using CLayout     = CLayout_;

    // Configuration
    static constexpr index_t BlockSize      = BlockSize_;
    static constexpr index_t TileM          = TileM_;
    static constexpr index_t TileN          = TileN_;
    static constexpr index_t TileK          = TileK_;
    static constexpr index_t WarpPerBlock_M = WaveM_;
    static constexpr index_t WarpPerBlock_N = WaveN_;
    static constexpr index_t WarpPerBlock_K = WaveK_;
    static constexpr index_t WarpTileM      = WarpM_;
    static constexpr index_t WarpTileN      = WarpN_;
    static constexpr index_t WarpTileK      = WarpK_;
    static constexpr bool kPadM             = PadM_;
    static constexpr bool kPadN             = PadN_;
    static constexpr bool kPadK             = PadK_;

    // Launch function - DECLARATION ONLY
    // Implementation is in separate .cpp files for parallel compilation
    static float launch(const GemmHostArgs& args, const stream_config& stream);

    // Support check
    static constexpr bool supports(index_t M, index_t N, index_t K)
    {
        if constexpr(kPadM && kPadN && kPadK)
            return true;
        return (kPadM || M % TileM == 0) && (kPadN || N % TileN == 0) && (kPadK || K % TileK == 0);
    }
};

// =============================================================================
// Common type aliases
// =============================================================================

using RowMajor = tensor_layout::gemm::RowMajor;
using ColMajor = tensor_layout::gemm::ColumnMajor;

// =============================================================================
// Kernel type declarations (no instantiation!)
// =============================================================================

// FP16 RCR variants
using Kernel_fp16_rcr_128x128x32 = GemmKernel<fp16_t,
                                              fp16_t,
                                              fp16_t,
                                              float,
                                              RowMajor,
                                              ColMajor,
                                              RowMajor,
                                              128,
                                              128,
                                              32,
                                              2,
                                              2,
                                              1,
                                              32,
                                              32,
                                              16,
                                              true,
                                              true,
                                              true>;

using Kernel_fp16_rcr_256x256x64 = GemmKernel<fp16_t,
                                              fp16_t,
                                              fp16_t,
                                              float,
                                              RowMajor,
                                              ColMajor,
                                              RowMajor,
                                              256,
                                              256,
                                              64,
                                              4,
                                              4,
                                              1,
                                              32,
                                              32,
                                              16,
                                              true,
                                              true,
                                              true>;

using Kernel_fp16_rcr_64x64x32 = GemmKernel<fp16_t,
                                            fp16_t,
                                            fp16_t,
                                            float,
                                            RowMajor,
                                            ColMajor,
                                            RowMajor,
                                            64,
                                            64,
                                            32,
                                            2,
                                            2,
                                            1,
                                            16,
                                            16,
                                            16,
                                            true,
                                            true,
                                            true>;

using Kernel_fp16_rcr_128x256x32 = GemmKernel<fp16_t,
                                              fp16_t,
                                              fp16_t,
                                              float,
                                              RowMajor,
                                              ColMajor,
                                              RowMajor,
                                              128,
                                              256,
                                              32,
                                              2,
                                              4,
                                              1,
                                              32,
                                              32,
                                              16,
                                              true,
                                              true,
                                              true>;

using Kernel_fp16_rcr_256x128x32 = GemmKernel<fp16_t,
                                              fp16_t,
                                              fp16_t,
                                              float,
                                              RowMajor,
                                              ColMajor,
                                              RowMajor,
                                              256,
                                              128,
                                              32,
                                              4,
                                              2,
                                              1,
                                              32,
                                              32,
                                              16,
                                              true,
                                              true,
                                              true>;

// BF16 RCR variants
using Kernel_bf16_rcr_128x128x32 = GemmKernel<bf16_t,
                                              bf16_t,
                                              bf16_t,
                                              float,
                                              RowMajor,
                                              ColMajor,
                                              RowMajor,
                                              128,
                                              128,
                                              32,
                                              2,
                                              2,
                                              1,
                                              32,
                                              32,
                                              16,
                                              true,
                                              true,
                                              true>;

using Kernel_bf16_rcr_256x256x64 = GemmKernel<bf16_t,
                                              bf16_t,
                                              bf16_t,
                                              float,
                                              RowMajor,
                                              ColMajor,
                                              RowMajor,
                                              256,
                                              256,
                                              64,
                                              4,
                                              4,
                                              1,
                                              32,
                                              32,
                                              16,
                                              true,
                                              true,
                                              true>;

// FP16 RRR variants
using Kernel_fp16_rrr_128x128x32 = GemmKernel<fp16_t,
                                              fp16_t,
                                              fp16_t,
                                              float,
                                              RowMajor,
                                              RowMajor,
                                              RowMajor,
                                              128,
                                              128,
                                              32,
                                              2,
                                              2,
                                              1,
                                              32,
                                              32,
                                              16,
                                              true,
                                              true,
                                              true>;

} // namespace dispatcher
} // namespace ck_tile
