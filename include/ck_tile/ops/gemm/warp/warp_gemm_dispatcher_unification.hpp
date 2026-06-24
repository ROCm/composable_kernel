// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/numeric/pk_fp4.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp"
#include "ck_tile/core/arch/mma/scale/scale_mma_pipeline.hpp"
#include "ck_tile/core/arch/mma/mma_wavewise.hpp"

#if USE_NEW_UNIFIED_FRAMEWORK
namespace ck_tile {
namespace impl {
namespace warp_gemm_dispatcher {

using namespace ck_tile::core::arch;
using namespace mma;

// This is a bit awkward but we need to be able to select the appropriate Mma Pipeline (dense,
// sparse, scale) based on some constexpr calculations in the UnificationDispatcher, without
// exposing the wrong path to the compiler, which may end up being ill-formed (if we were to use a
// simple "if constexpr" instead of TMP).
template <bool IsMx,
          typename AType,
          typename BType,
          typename AccType,
          index_t M,
          index_t N,
          index_t K,
          MmaAccumPolicy AccumPolicy,
          bool TransposeC,
          index_t SwizzleFactor,
          index_t AttrNumAccessAV,
          index_t AttrNumAccessBV>
struct MmaPipelineSelector;

template <typename AType,
          typename BType,
          typename AccType,
          index_t M,
          index_t N,
          index_t K,
          MmaAccumPolicy AccumPolicy,
          bool TransposeC,
          index_t SwizzleFactor,
          index_t AttrNumAccessAV,
          index_t AttrNumAccessBV>
struct MmaPipelineSelector<true,
                           AType,
                           BType,
                           AccType,
                           M,
                           N,
                           K,
                           AccumPolicy,
                           TransposeC,
                           SwizzleFactor,
                           AttrNumAccessAV,
                           AttrNumAccessBV>
{
    using Type = ScaleMmaPipeline<AType,
                                  BType,
                                  AccType,
                                  M,
                                  N,
                                  K,
                                  AccumPolicy,
                                  TransposeC,
                                  SwizzleFactor,
                                  AttrNumAccessAV,
                                  AttrNumAccessBV>;
};

template <typename AType,
          typename BType,
          typename AccType,
          index_t M,
          index_t N,
          index_t K,
          MmaAccumPolicy AccumPolicy,
          bool TransposeC,
          index_t SwizzleFactor,
          index_t AttrNumAccessAV,
          index_t AttrNumAccessBV>
struct MmaPipelineSelector<false,
                           AType,
                           BType,
                           AccType,
                           M,
                           N,
                           K,
                           AccumPolicy,
                           TransposeC,
                           SwizzleFactor,
                           AttrNumAccessAV,
                           AttrNumAccessBV>
{
    using Type = WaveWiseMmaPipeline<AType,
                                     BType,
                                     AccType,
                                     M,
                                     N,
                                     K,
                                     AccumPolicy,
                                     TransposeC,
                                     SwizzleFactor,
                                     AttrNumAccessAV,
                                     AttrNumAccessBV>;
};

// TODO: Figure out how to deal with the "packed" version of AttrNumAccess. In the unification
// framework there is no reason to combine packedness with AttrNumAccess but in CK Tile they did,
// alongside the refactor introducing gfx1250.
template <WGAttrNumAccessEnum AttrNumAccess>
struct get_wgattr_num_access_safe_v
{
    static constexpr int32_t value = get_wgattr_num_access<AttrNumAccess>::value;
};
template <>
struct get_wgattr_num_access_safe_v<WGAttrNumAccessEnum::Default>
{
    static constexpr int32_t value = 1;
};

template <typename AType,
          typename BType,
          typename AccType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC,
          bool SwizzleA                      = false,
          bool UseStructuredSparsity         = false,
          WGAttrNumAccessEnum AttrNumAccessA = WGAttrNumAccessEnum::Single,
          WGAttrNumAccessEnum AttrNumAccessB = AttrNumAccessA,
          bool IsScale16                     = false>
struct UnificationDispatcher
{
    static_assert(!IsScale16); // TODO: We can't deal with scale16 yet.

    // TODO: The dispatcher currently determines whether microscaling intrinsics are requested based
    // on the WaveTile sizes and types. This is potentially dangerous and we should add a dedicated
    // parameter instead.
    static constexpr bool IsMxSized = (MPerWave == 16 && NPerWave == 16 && KPerWave == 128) ||
                                      (MPerWave == 32 && NPerWave == 32 && KPerWave == 64);
    static constexpr bool IsMx =
        (IsMxSized && std::is_same_v<AccType, float> && UseStructuredSparsity == false);

    // General checks. Swizzle not supported yet. Structured sparsity Mma pipeline not adapted to
    // UnificationDispatcher yet since we have no sparse tests or examples in CK Tile.
    static_assert(SwizzleA == false);
    static_assert(UseStructuredSparsity == false);

    // Scale checks.
    // TODO: Add the tiny types after those are merged.
    static_assert(!IsMx || (std::is_same_v<AType, fp8_t> || std::is_same_v<AType, bf8_t> ||
                            std::is_same_v<AType, pk_fp4_t>));
    static_assert(!IsMx || (std::is_same_v<BType, fp8_t> || std::is_same_v<BType, bf8_t> ||
                            std::is_same_v<BType, pk_fp4_t>));

    // Convert SwizzleA bool to SwizzleFactor. This used to be hardcoded in a number of places in
    // the original dispatcher / warpgemms, generally using a factor of 2 if swizzling was
    // requested but not always. TODO: Check original usage for correct swizzle factors.
    static constexpr index_t SwizzleFactor = SwizzleA ? 2 : 1;

    // Convert WGAttrNumAccessEnums to index_t values. Default value sent to 1 for now, but needs a
    // better implementation TODO.
    static constexpr index_t AttrNumAccessAV = get_wgattr_num_access_safe_v<AttrNumAccessA>::value;
    static constexpr index_t AttrNumAccessBV = get_wgattr_num_access_safe_v<AttrNumAccessB>::value;

    using Type =
        typename MmaPipelineSelector<IsMx,
                                     AType,
                                     BType,
                                     AccType,
                                     MPerWave,
                                     NPerWave,
                                     KPerWave,
                                     MmaAccumPolicy::ROW_MAJOR, // Always ROW_MAJOR for now, we
                                                                // don't allow MN composition.
                                     TransposeC,
                                     SwizzleFactor,
                                     AttrNumAccessAV,
                                     AttrNumAccessBV>::Type;
};
} // namespace warp_gemm_dispatcher
} // namespace impl
} // namespace ck_tile
#endif // #if USE_NEW_UNIFIED_FRAMEWORK
