// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp"
// Add smfmac attribute impl for structured sparsity support
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_smfmac_impl.hpp"

namespace ck_tile {

// WarpGemmCoreDispatcher: choose the underlying MFMA/SMFMAC attribute Impl
// based on A/B/Acc types and (M,N,K) per wave.
template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool UseStructuredSparsity = false>
struct WarpGemmCoreDispatcher;

// Generic specialization for MFMA
template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave>
struct WarpGemmCoreDispatcher<AType, BType, AccType, MPerWave, NPerWave, KPerWave, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<AType,
                                           BType,
                                           AccType,
                                           MPerWave,
                                           NPerWave,
                                           KPerWave,
                                           WGAttrCtlEnum::Default_>;
};

// Generic specialization for SMFMAC
// TODO: we also need to support smfmac for FP8/BF8 and I8 format
template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave>
struct WarpGemmCoreDispatcher<AType, BType, AccType, MPerWave, NPerWave, KPerWave, true>
{
    using Impl = WarpGemmAttributeSmfmacImpl<AType,
                                             BType,
                                             AccType,
                                             MPerWave,
                                             NPerWave,
                                             KPerWave,
                                             WGAttrCtlEnum::Default_>;
};

// Specialization for special cases
template <>
struct WarpGemmCoreDispatcher<ck_tile::half_t, ck_tile::half_t, float, 32, 32, 16, false>
{
#if defined(__gfx950__)
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::half_t,
                                           ck_tile::half_t,
                                           float,
                                           32,
                                           32,
                                           16,
                                           WGAttrCtlEnum::Default_>;
#else
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::half_t,
                                           ck_tile::half_t,
                                           float,
                                           32,
                                           32,
                                           8,
                                           WGAttrCtlEnum::Default_>;
#endif
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::half_t, ck_tile::half_t, float, 16, 16, 32, false>
{
#if defined(__gfx950__)
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::half_t,
                                           ck_tile::half_t,
                                           float,
                                           16,
                                           16,
                                           32,
                                           WGAttrCtlEnum::Default_>;
#else
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::half_t,
                                           ck_tile::half_t,
                                           float,
                                           16,
                                           16,
                                           16,
                                           WGAttrCtlEnum::Default_>;
#endif
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::half_t, ck_tile::half_t, float, 4, 64, 16, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::half_t,
                                           ck_tile::half_t,
                                           float,
                                           4,
                                           64,
                                           4,
                                           WGAttrCtlEnum::Default_>;
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::half_t, ck_tile::half_t, float, 64, 4, 16, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::half_t,
                                           ck_tile::half_t,
                                           float,
                                           64,
                                           4,
                                           4,
                                           WGAttrCtlEnum::Default_>;
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 32, 32, 32, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::fp8_t,
                                           ck_tile::fp8_t,
                                           float,
                                           32,
                                           32,
                                           16,
                                           WGAttrCtlEnum::Default_>;
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 32, 32, 32, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::bf8_t,
                                           ck_tile::bf8_t,
                                           float,
                                           32,
                                           32,
                                           16,
                                           WGAttrCtlEnum::Default_>;
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16, 64, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::fp8_t,
                                           ck_tile::fp8_t,
                                           float,
                                           16,
                                           16,
                                           32,
                                           WGAttrCtlEnum::Default_>;
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16, 64, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::bf8_t,
                                           ck_tile::bf8_t,
                                           float,
                                           16,
                                           16,
                                           32,
                                           WGAttrCtlEnum::Default_>;
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 32, 32, 16, false>
{
#if defined(__gfx950__)
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::bf16_t,
                                           ck_tile::bf16_t,
                                           float,
                                           32,
                                           32,
                                           16,
                                           WGAttrCtlEnum::Default_>;
#else
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::bf16_t,
                                           ck_tile::bf16_t,
                                           float,
                                           32,
                                           32,
                                           8,
                                           WGAttrCtlEnum::Default_>;
#endif
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 16, 16, 32, false>
{
#if defined(__gfx950__)
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::bf16_t,
                                           ck_tile::bf16_t,
                                           float,
                                           16,
                                           16,
                                           32,
                                           WGAttrCtlEnum::Default_>;
#else
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::bf16_t,
                                           ck_tile::bf16_t,
                                           float,
                                           16,
                                           16,
                                           16,
                                           WGAttrCtlEnum::Default_>;
#endif
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 4, 64, 16, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::bf16_t,
                                           ck_tile::bf16_t,
                                           float,
                                           4,
                                           64,
                                           4,
                                           WGAttrCtlEnum::Default_>;
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::bf16_t, ck_tile::bf16_t, float, 64, 4, 16, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::bf16_t,
                                           ck_tile::bf16_t,
                                           float,
                                           64,
                                           4,
                                           4,
                                           WGAttrCtlEnum::Default_>;
};

// Iterate-K variants built from the direct impls
template <>
struct WarpGemmCoreDispatcher<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 32, 32, 32, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::int8_t,
                                           ck_tile::int8_t,
                                           ck_tile::int32_t,
                                           32,
                                           32,
                                           16,
                                           WGAttrCtlEnum::Default_>;
};

template <>
struct WarpGemmCoreDispatcher<ck_tile::int8_t, ck_tile::int8_t, ck_tile::int32_t, 16, 16, 64, false>
{
    using Impl = WarpGemmAttributeMfmaImpl<ck_tile::int8_t,
                                           ck_tile::int8_t,
                                           ck_tile::int32_t,
                                           16,
                                           16,
                                           32,
                                           WGAttrCtlEnum::Default_>;
};

} // namespace ck_tile
