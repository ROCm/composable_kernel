// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/core/numeric/integer.hpp>

// can be used on both host and kernel building
#if defined(BUILD_HSTU_FOR_GFX125)
#define __hstu_gfx125__ 1
#endif

// can be used on both host and kernel building
#if defined(BUILD_HSTU_FOR_GFX95)
#define __hstu_gfx95__ 1
#endif

// can be used on both host and kernel building
#if defined(BUILD_HSTU_FOR_GFX94)
#define __hstu_gfx94__ 1
#endif

#if defined(BUILD_HSTU_FOR_GFX95) || defined(BUILD_HSTU_FOR_GFX125)
#define HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE 1
#else
#define HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE 0
#endif

#if defined(BUILD_HSTU_FOR_GFX125)
#define HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE 1
#else
#define HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE 0
#endif

// Which forward pipeline a dispatch instance instantiates.
//
// This replaces the per-dispatch "static constexpr bool use_trload_pipeline", which was
// spelled out identically in all six *_forward{,_splitkv}_dispatch.hpp files. Keeping the
// rule in one place means a dispatch site cannot pick up part of the gating and miss the
// rest, and it leaves room for a third pipeline without turning one bool into two.
enum class HstuFwdPipelineKind
{
    // Only instantiable where HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE is 0. The Default
    // pipelines call Policy::MakeShuffledVRegTileDistribution() unconditionally, and that
    // member is itself declared under #if !HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE in
    // hstu_attention_fwd_pipeline_policy.hpp -- so returning Default on an arch that has
    // ds_read_tr is a hard template error several headers down, not a fallback.
    Default,         // HstuAttention{With,No}SoftmaxFwdPipelineQRKSVS
    TrLoad,          // HstuAttention{With,No}SoftmaxFwdPipelineQRKSVSTrLoad
    DirectLdsTrLoad, // // HstuAttention{With,No}SoftmaxFwdPipelineQRKSVSDirectLdsTrLoad, gfx950
                     // only
    TdmTrLoad,       // HstuAttention{With,No}SoftmaxFwdPipelineQRKSVSTdm, gfx1250 only
};

// The choice depends only on the build arch and on values known before the tile setting is
// selected, so a dispatch site is free to query it first and feed the answer into the tile
// setting. Keep it that way.
template <bool kUseSoftmax, ck_tile::index_t MaxK, bool kHdimPadded>
constexpr HstuFwdPipelineKind get_hstu_fwd_pipeline_kind()
{
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
    // Tdm is selected per MaxK, for the buckets that have Tdm tile rows in the gfx125 tile
    // setting (the Tdm pipelines need kN0 == kN0Sub == kK1). Every other MaxK keeps trload.
    if constexpr(MaxK == 128)
        return HstuFwdPipelineKind::TdmTrLoad;
    else if constexpr(MaxK == 64)
        return HstuFwdPipelineKind::TdmTrLoad;
    else if constexpr(MaxK == 256)
        return HstuFwdPipelineKind::TdmTrLoad;
    else if constexpr(MaxK == 96)
        return HstuFwdPipelineKind::TdmTrLoad;
    else
        return HstuFwdPipelineKind::TrLoad;
#elif HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE
    if constexpr(!kHdimPadded && MaxK != 96)
        return HstuFwdPipelineKind::DirectLdsTrLoad;
    else
        return HstuFwdPipelineKind::TrLoad;
#else
    return HstuFwdPipelineKind::Default;
#endif
}

using HstuBwdPipelineKind = HstuFwdPipelineKind;

template <ck_tile::index_t MaxK, bool kHdimPadded>
constexpr HstuBwdPipelineKind get_hstu_bwd_pipeline_kind()
{
#if HSTU_LDS_STAGING_THROUGH_TDM_AVAILABLE
    return HstuBwdPipelineKind::TdmTrLoad;
#elif HSTU_LDS_READ_WITH_TRANSPOSE_AVAILABLE
    if constexpr(!kHdimPadded && MaxK != 96)
        return HstuBwdPipelineKind::DirectLdsTrLoad;
    else
        return HstuBwdPipelineKind::TrLoad;
#else
    return HstuBwdPipelineKind::Default;
#endif
}
