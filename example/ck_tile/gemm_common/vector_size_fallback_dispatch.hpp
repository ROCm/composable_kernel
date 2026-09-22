// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

#include <iostream>

// Shared by 03_gemm and 16_batched_gemm: picks the GemmConfig variant for a given shape. It
// dispatches to natural vector sizes, a reduced-but-legal vector size, and/or tile-size
// padding
namespace ck_tile_example {

template <typename GemmConfig,
          ck_tile::index_t VectorSizeA_,
          ck_tile::index_t VectorSizeB_,
          ck_tile::index_t VectorSizeC_>
struct GemmConfigVectorSizeFallback : public GemmConfig
{
    // Split-K partitions are aligned to K_Warp_Tile: preserve split-K remainder
    static_assert(GemmConfig::K_Warp_Tile % VectorSizeA_ == 0 &&
                      GemmConfig::K_Warp_Tile % VectorSizeB_ == 0,
                  "A/B vector width must divide K_Warp_Tile");

    // Enable K/M/N padding
    static constexpr bool kPadM = true;
    static constexpr bool kPadN = true;
    static constexpr bool kPadK = true;

    static constexpr bool FixedVectorSize         = true;
    static constexpr ck_tile::index_t VectorSizeA = VectorSizeA_;
    static constexpr ck_tile::index_t VectorSizeB = VectorSizeB_;
    static constexpr ck_tile::index_t VectorSizeC = VectorSizeC_;
};

// The three helpers below walk a vector-size ladder (Max, Max/2, Max/4, ..., 1), picking the
// largest width that evenly divides the runtime extent(s) instead of always collapsing straight
// to scalar (1) the moment the natural width doesn't fit.
template <typename GemmConfig, ck_tile::index_t Max, ck_tile::index_t VC, typename InvokeFn>
float invoke_with_best_ab_vector_size(ck_tile::index_t a_extent,
                                      ck_tile::index_t b_extent,
                                      InvokeFn&& invoke)
{
    if constexpr(Max >= 16)
    {
        if(a_extent % 16 == 0 && b_extent % 16 == 0)
            return invoke
                .template operator()<GemmConfigVectorSizeFallback<GemmConfig, 16, 16, VC>>();
    }
    if constexpr(Max >= 8)
    {
        if(a_extent % 8 == 0 && b_extent % 8 == 0)
            return invoke.template operator()<GemmConfigVectorSizeFallback<GemmConfig, 8, 8, VC>>();
    }
    if constexpr(Max >= 4)
    {
        if(a_extent % 4 == 0 && b_extent % 4 == 0)
            return invoke.template operator()<GemmConfigVectorSizeFallback<GemmConfig, 4, 4, VC>>();
    }
    if constexpr(Max >= 2)
    {
        if(a_extent % 2 == 0 && b_extent % 2 == 0)
            return invoke.template operator()<GemmConfigVectorSizeFallback<GemmConfig, 2, 2, VC>>();
    }
    return invoke.template operator()<GemmConfigVectorSizeFallback<GemmConfig, 1, 1, VC>>();
}

template <typename GemmConfig,
          ck_tile::index_t VA,
          ck_tile::index_t VB,
          ck_tile::index_t Max,
          typename InvokeFn>
float invoke_with_best_c_vector_size(ck_tile::index_t c_extent, InvokeFn&& invoke)
{
    if constexpr(Max >= 16)
    {
        if(c_extent % 16 == 0)
            return invoke
                .template operator()<GemmConfigVectorSizeFallback<GemmConfig, VA, VB, 16>>();
    }
    if constexpr(Max >= 8)
    {
        if(c_extent % 8 == 0)
            return invoke
                .template operator()<GemmConfigVectorSizeFallback<GemmConfig, VA, VB, 8>>();
    }
    if constexpr(Max >= 4)
    {
        if(c_extent % 4 == 0)
            return invoke
                .template operator()<GemmConfigVectorSizeFallback<GemmConfig, VA, VB, 4>>();
    }
    if constexpr(Max >= 2)
    {
        if(c_extent % 2 == 0)
            return invoke
                .template operator()<GemmConfigVectorSizeFallback<GemmConfig, VA, VB, 2>>();
    }
    return invoke.template operator()<GemmConfigVectorSizeFallback<GemmConfig, VA, VB, 1>>();
}

template <typename GemmConfig, ck_tile::index_t ABMax, ck_tile::index_t CMax, typename InvokeFn>
float invoke_with_best_vector_sizes(ck_tile::index_t a_extent,
                                    ck_tile::index_t b_extent,
                                    ck_tile::index_t c_extent,
                                    InvokeFn&& invoke)
{
    if constexpr(ABMax >= 16)
    {
        if(a_extent % 16 == 0 && b_extent % 16 == 0)
            return invoke_with_best_c_vector_size<GemmConfig, 16, 16, CMax>(c_extent, invoke);
    }
    if constexpr(ABMax >= 8)
    {
        if(a_extent % 8 == 0 && b_extent % 8 == 0)
            return invoke_with_best_c_vector_size<GemmConfig, 8, 8, CMax>(c_extent, invoke);
    }
    if constexpr(ABMax >= 4)
    {
        if(a_extent % 4 == 0 && b_extent % 4 == 0)
            return invoke_with_best_c_vector_size<GemmConfig, 4, 4, CMax>(c_extent, invoke);
    }
    if constexpr(ABMax >= 2)
    {
        if(a_extent % 2 == 0 && b_extent % 2 == 0)
            return invoke_with_best_c_vector_size<GemmConfig, 2, 2, CMax>(c_extent, invoke);
    }
    return invoke_with_best_c_vector_size<GemmConfig, 1, 1, CMax>(c_extent, invoke);
}

// Picks, for the given shape, which GemmConfig variant to instantiate `invoke` with: the bare
// GemmConfig, a natural-vector-size-but-padded variant, or a reduced-vector-size variant (via
// the ladder walk above). GemmConfig::EnableSmallerVectorLoadFallback gates whether A/B and C
// are allowed to take the reduced-vector-size/padded path at all.
template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename InvokeFn>
float dispatch_vector_size_fallback(ck_tile::index_t M,
                                    ck_tile::index_t N,
                                    ck_tile::index_t K,
                                    ck_tile::index_t stride_A,
                                    ck_tile::index_t stride_B,
                                    ck_tile::index_t stride_C,
                                    ck_tile::index_t kbatch,
                                    InvokeFn&& invoke,
                                    ck_tile::index_t batch_stride_A = 0,
                                    ck_tile::index_t batch_stride_B = 0,
                                    ck_tile::index_t batch_stride_C = 0)
{
    // A: M x K, B: K x N, C: M x N
    // When RowMajor: the second dimension is packed continuously in memory; when ColumnMajor,
    // the first dimension is.
    constexpr bool a_is_row_major = std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::RowMajor>;
    constexpr bool b_is_row_major = std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>;
    constexpr bool c_is_row_major = std::is_same_v<CLayout, ck_tile::tensor_layout::gemm::RowMajor>;

    // Fallback is only enabled if the GemmConfig allows smaller vector loads, the vector size is
    // not fixed, and A and B have a packed size of 1.
    constexpr bool fallback_enabled = GemmConfig::EnableSmallerVectorLoadFallback &&
                                      !GemmConfig::FixedVectorSize &&
                                      std::is_same_v<ADataType, BDataType> &&
                                      ck_tile::numeric_traits<ADataType>::PackedSize == 1 &&
                                      ck_tile::numeric_traits<BDataType>::PackedSize == 1;

    // C additionally needs its own check
    constexpr bool ab_reducible = fallback_enabled;
    constexpr bool c_reducible =
        fallback_enabled && ck_tile::numeric_traits<CDataType>::PackedSize == 1;

    constexpr ck_tile::index_t ABVectorSize = 16 / sizeof(ADataType);
    constexpr ck_tile::index_t CVectorSize  = 16 / sizeof(CDataType);

    // Extent of the tensor's continuous dimension (fastest varying), which is the one that
    // must be a multiple of the vector-load/store width for A/B/C
    const ck_tile::index_t a_contiguous_extent = a_is_row_major ? K : M;
    const ck_tile::index_t b_contiguous_extent = b_is_row_major ? N : K;
    const ck_tile::index_t c_contiguous_extent = c_is_row_major ? N : M;

    // The vector load's base address is row/batch * stride, not just the logical extent, so an
    // odd stride (or batch stride, for batched GEMM) can make a load misaligned even when the
    // extent itself divides evenly.
    const bool needsScalarAB =
        ab_reducible &&
        ((a_contiguous_extent % ABVectorSize != 0) || (b_contiguous_extent % ABVectorSize != 0) ||
         (stride_A % ABVectorSize != 0) || (stride_B % ABVectorSize != 0) ||
         (batch_stride_A % ABVectorSize != 0) || (batch_stride_B % ABVectorSize != 0));
    const bool needsScalarC =
        c_reducible && ((c_contiguous_extent % CVectorSize != 0) || (stride_C % CVectorSize != 0) ||
                        (batch_stride_C % CVectorSize != 0));

    // Tile-size alignment is a separate, coarser boundary than the vector-size one above: a
    // shape can be vector-aligned yet still not be a multiple of the tile size. K's alignment
    // only matters when AB can fall back; M/N's only when C can. GemmConfig may not even define
    // M_Tile/N_Tile/K_Tile when fallback is disabled, so these must stay inside if constexpr.
    const bool needsPadK = [&] {
        if constexpr(ab_reducible)
            return K % (GemmConfig::K_Tile * kbatch) != 0;
        else
            return false;
    }();
    const bool needsPadMN = [&] {
        if constexpr(c_reducible)
            return (M % GemmConfig::M_Tile != 0) || (N % GemmConfig::N_Tile != 0);
        else
            return false;
    }();
    const bool needsPad = needsPadK || needsPadMN;

    // Report details about the GemmConfig this shape ended up dispatching
    if(needsScalarAB || needsScalarC)
    {
        std::cout << "Run with GemmConfig fallback for reduced-width loads/stores" << std::endl;
    }
    else if(needsPad)
    {
        std::cout << "Run with GemmConfig fallback for tile-size padding" << std::endl;
    }
    else
    {
        std::cout << "Run with GemmConfig: natural vector sizes" << std::endl;
    }

    if constexpr(ab_reducible)
    {
        if(needsScalarAB)
        {
            if constexpr(c_reducible)
            {
                if(needsScalarC)
                    return invoke_with_best_vector_sizes<GemmConfig, ABVectorSize, CVectorSize>(
                        a_contiguous_extent, b_contiguous_extent, c_contiguous_extent, invoke);
            }
            return invoke_with_best_ab_vector_size<GemmConfig, ABVectorSize, CVectorSize>(
                a_contiguous_extent, b_contiguous_extent, invoke);
        }
    }
    if constexpr(c_reducible)
    {
        if(needsScalarC)
            return invoke_with_best_c_vector_size<GemmConfig,
                                                  ABVectorSize,
                                                  ABVectorSize,
                                                  CVectorSize>(c_contiguous_extent, invoke);
    }
    if constexpr(ab_reducible)
    {
        if(needsPad)
            return invoke.template operator()<GemmConfigVectorSizeFallback<GemmConfig,
                                                                           ABVectorSize,
                                                                           ABVectorSize,
                                                                           CVectorSize>>();
    }
    return invoke.template operator()<GemmConfig>();
}

} // namespace ck_tile_example
