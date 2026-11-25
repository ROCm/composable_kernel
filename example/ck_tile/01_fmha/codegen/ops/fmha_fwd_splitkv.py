# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
# generate kernel instances to speed up compilation
import copy
import dataclasses
import fnmatch
import itertools
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from codegen.arch import ArchTrait, get_factories_for_targets
from codegen.cmake_config import GEN_DIR
from codegen.cpp_symbol_map import (
    PIPELINE_ENUM_MAP,
    get_mask_check_map,
    LAYOUT_MAP,
    BIAS_CHECK_MAP,
    MODE_MAP,
    FWD_DTYPE_MAP,
    BIAS_MAP,
    get_mask_map,
    BOOL_MAP,
    QSCALE_MAP,
    QSCALE_CHECK_MAP,
)
from codegen.utils import check_duplicates_and_paddings, if_, indent, update_file

from codegen.ops.fmha_fwd import (
    FmhaFwdPipeline,
    CppConstraint,
    FmhaFwdTileSize,
    GFX125_QR_TDM_PINGPONG_HDIMS,
    _gfx125_qr_tdm_pingpong_block_scales,
    qr_tdm_pingpong_lds_read_schedule,
    _gfx125_qr_tdm_pingpong_tiles,
    qr_tdm_pingpong_effective_schedule,
    DTYPE_BITS,
    K0_MAX_SUBMAX_MAP,
    TDM_K0_MAX_SUBMAX_MAP,
    FMHA_FWD_KERNEL_HEADER,
    FMHA_FWD_API_PER_ARCH,
    FMHA_FWD_API_PER_DTYPE,
    FMHA_FWD_API_PER_HDIM_CASE,
)


SPLITKV_PIPELINES_TAG = ["qr", "qr_nwarp_sshuffle", "qr_tdm", "qr_tdm_pingpong"]

# TDM pipelines merge the q heads of a kv head with a mask too, except with a sink, and are the
# only ones that merge with a sink (each merged row reads the sink of its q head).
# qr_tdm_pingpong's split-KV is the forward kernel (FMHA_FWD_SPLITKV_PINGPONG_KERNEL_BODY), which
# merges heads itself and has its own head-dim checks.
TDM_PIPELINES_TAG = ["qr_tdm"]

FMHA_FWD_SPLITKV_PIPELINE_MAP = {
    "qr": "ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS",
    "qr_nwarp_sshuffle": "ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS",
    "qr_tdm": "ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVSTdm",
    "qr_tdm_pingpong": "ck_tile::BlockFmhaPipelineQRKSVSTdmPingPong",
}

# qr_tdm_pingpong as the split-KV main kernel: the forward kernel (FmhaFwdKernel<..., kSplitKV>)
# on its 16-row decode tile, writing each split's normalized O and LSE to the fp32 accumulators.
FMHA_FWD_SPLITKV_PINGPONG_KERNEL_BODY = """
#include <iostream>
#include <type_traits>

#if !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})

namespace {{
template <bool kMergeHeads>
struct instance_{F_idx} {{
using fmha_dtype = {F_dtype};

using fmha_block_tile = ck_tile::sequence<{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}>;

using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                          ck_tile::sequence<{F_rm0}, {F_rn0}, {F_rk0}>,
                                          ck_tile::sequence<{F_wm0}, {F_wn0}, {F_wk0}>,
                                          ck_tile::sequence<{F_rm1}, {F_rn1}, {F_rk1}>,
                                          ck_tile::sequence<{F_wm1}, {F_wn1}, {F_wk1}>,
                                          {F_vlayout},
                                          true>;

using fmha_traits_base = ck_tile::TileFmhaTraits<{F_spad},
                                            {F_skpad},
                                            {F_dpad},
                                            {F_dvpad},
                                            {F_logits},
                                            {F_bias},
                                            false,
                                            true,
                                            false,
                                            {F_qscale},
                                            {F_occupancy},
                                            false,
                                            {F_sink}>;
// merged: a KV head's query heads (one query each) as the tile's rows
using fmha_traits = std::conditional_t<kMergeHeads,
                                       ck_tile::TileFmhaMergeHeadsTraits<fmha_traits_base>,
                                       fmha_traits_base>;

using fmha_variant = ck_tile::ComposedAttention<{F_logits} * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>;

using fmha_mask = {F_mask};

using fmha_pipeline_problem = ck_tile::BlockFmhaPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::RandValOutputDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
    fmha_shape,
    {F_mode},
    fmha_variant,
    fmha_mask,
    false,
    fmha_traits,
    false,
    false>;

using fmha_pipeline = ck_tile::BlockFmhaPipelineQRKSVSTdmPingPong<
    fmha_pipeline_problem, ck_tile::BlockFmhaPipelineQRKSVSTdmPingPongPolicy,
    ck_tile::QrTdmPingPongLdsReadSchedule<{F_pp_slice_prefetch}, {F_pp_v_before_softmax}>>;

using fmha_epilogue = ck_tile::LdsShuffle2DEpilogue<
    ck_tile::LdsShuffle2DEpilogueProblem<
        typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
        typename FmhaFwdTypeConfig<fmha_dtype>::OaccDataType,
        fmha_shape::NumWarps,
        {F_bm0},
        {F_bn1},
        {F_spad},
        {F_dvpad},
        1>>;

using fmha_kernel = ck_tile::FmhaFwdKernel<fmha_pipeline, fmha_epilogue, true>;
}};
}} // anonymous namespace

using trait_{F_idx} = fmha_fwd_splitkv_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}, {F_vlayout},
                        {F_pipeline_enum}, {F_logits}, {F_mask}, {F_bias}, true, false, false, {F_sink}, {F_spad}, {F_skpad}, {F_dpad},
                        {F_dvpad}, {F_occupancy}, {F_qscale}>;

namespace {{
template <bool kMergeHeads>
void run_{F_idx}(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    using k_ = typename instance_{F_idx}<kMergeHeads>::fmha_kernel;
    auto [kargs, grids] = fmha_fwd_splitkv_pingpong_create_kargs_and_grids<k_>(a);
    const dim3 blocks                      = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu, {F_arch.tag}>(k_{{}}, grids, blocks, 0, kargs)(ck_tile::stream_config{{s.stream_id_}});
}}

// GQA decode: a KV head's single-query heads fold into one tile's rows (K/V read once); batch
// mode without a bias (a merged row's index is its head's)
constexpr bool kCanMergeHeads_{F_idx} = !{F_mode} && {F_bias} == ck_tile::BlockAttentionBiasEnum::NO_BIAS;
bool merge_heads_{F_idx}(const fmha_fwd_splitkv_args& a)
{{
    return kCanMergeHeads_{F_idx} && a.seqlen_q == 1 && a.max_seqlen_q == 1 &&
           a.nhead_q > a.nhead_k && a.nhead_q / a.nhead_k <= {F_bm0};
}}
}} // anonymous namespace

template<>
void fmha_fwd_splitkv_oneshot_<trait_{F_idx}, {F_arch.tag}>(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    if constexpr(kCanMergeHeads_{F_idx})
    {{
        if(merge_heads_{F_idx}(a))
            return run_{F_idx}<true>(s, a);
    }}
    run_{F_idx}<false>(s, a);
}}

// the forward kernel's own checks (descale layouts, merged rows' strides), ahead of the launch
template<>
bool fmha_fwd_splitkv_supported_<trait_{F_idx}, {F_arch.tag}>(const fmha_fwd_splitkv_args& a)
{{
    const auto check = [&](auto merge) {{
        using k_ = typename instance_{F_idx}<decltype(merge)::value>::fmha_kernel;
        const auto [kargs, grids] = fmha_fwd_splitkv_pingpong_create_kargs_and_grids<k_>(a);
        (void)grids;
        return k_::IsSupportedArgument(kargs);
    }};
    if constexpr(kCanMergeHeads_{F_idx})
    {{
        if(merge_heads_{F_idx}(a))
            return check(std::true_type{{}});
    }}
    return check(std::false_type{{}});
}}

template<>
std::string fmha_fwd_splitkv_get_name_<trait_{F_idx}, {F_arch.tag}>()
{{
    return "{F_kname}";
}}

#endif // !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})
"""

FMHA_FWD_SPLITKV_KERNEL_BODY = """
#include <iostream>

#if !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})

using fmha_dtype_{F_idx} = {F_dtype};
using fmha_variant_{F_idx} = ck_tile::ComposedAttention<{F_logits} * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP2>;
using fmha_mask_{F_idx} = {F_mask};

namespace {{
template <bool kHasUnevenSplits, bool kMergeNumHeadGroupsSeqLenQ = false>
struct instance {{
using fmha_block_tile = ck_tile::sequence<{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}>;

using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                          ck_tile::sequence<{F_rm0}, {F_rn0}, {F_rk0}>,
                                          ck_tile::sequence<{F_wm0}, {F_wn0}, {F_wk0}>,
                                          ck_tile::sequence<{F_rm1}, {F_rn1}, {F_rk1}>,
                                          ck_tile::sequence<{F_wm1}, {F_wn1}, {F_wk1}>,
                                          {F_vlayout}>;

using fmha_trait = ck_tile::TileFmhaFwdSplitKVTraits<{F_spad},
                                                     {F_skpad},
                                                     {F_dpad},
                                                     {F_dvpad},
                                                     {F_logits},
                                                     {F_bias},
                                                     /*kHasBiasGrad=*/false,
                                                     {F_lse},
                                                     {F_squant},
                                                     {F_pagedkv},
                                                     kHasUnevenSplits,
                                                     kMergeNumHeadGroupsSeqLenQ,
                                                     {F_occupancy},
                                                     {F_sink}>;

using fmha_pipeline_problem = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::OaccDataType,
    fmha_shape,
    {F_mode},
    fmha_variant_{F_idx},
    fmha_mask_{F_idx},
    fmha_trait>;

using fmha_pipeline = {F_pipeline}<
    fmha_pipeline_problem>;

/// FIXME: use {F_spad}/{F_dvpad} as kPadM/kPadN parameters after solving
///        store_tile_raw() data corruption issue
using fmha_epilogue =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType,
                                           typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType,
                                           false, false>>;

using fmha_kernel =
    ck_tile::FmhaFwdSplitKVKernel<fmha_pipeline, fmha_epilogue>;

static void run(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    using k_ = fmha_kernel;
    auto [kargs, grids] = fmha_fwd_splitkv_create_kargs_and_grids<k_>(a);
    const dim3 blocks                      = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    using attr_ = std::conditional_t<k_::kMaxWavesPerEu == 0, {F_arch.tag},
        ck_tile::kernel_attr_for<{F_arch.tag}, ck_tile::kernel_attr_max_waves_per_eu<k_::kMaxWavesPerEu>>>;
    ck_tile::make_kernel<kBlockPerCu, attr_>(k_{{}}, grids, blocks, 0, kargs)(ck_tile::stream_config{{s.stream_id_}});
}}
}}; // struct instance
}} // anonymous namespace

using trait_{F_idx} = fmha_fwd_splitkv_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}, {F_vlayout},
                        {F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_squant}, {F_pagedkv}, {F_sink}, {F_spad}, {F_skpad}, {F_dpad},
                        {F_dvpad}, {F_occupancy}>;

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wtautological-compare"
#endif
namespace {{
template <bool kHasUnevenSplits>
void run_instance(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a) {{
    if constexpr ({F_bias} == ck_tile::BlockAttentionBiasEnum::NO_BIAS && (!{F_sink} || {F_merge_sink})
                  && (std::is_same_v<{F_mask}, ck_tile::SimplifiedGenericAttentionMask<false>>
                      || std::is_same_v<{F_mask}, FmhaMasks::NoMask> || {F_merge_masked})) {{
        if (a.max_seqlen_q == 1 && a.nhead_k < a.nhead_q) {{
            instance<kHasUnevenSplits, /*kMergeNumHeadGroupsSeqLenQ=*/true>::run(s, a);
        }} else {{
            instance<kHasUnevenSplits>::run(s, a);
        }}
    }} else {{
        instance<kHasUnevenSplits>::run(s, a);
    }}
}}
}} // anonymous namespace

#ifdef __clang__
#pragma clang diagnostic pop
#endif

template<>
void fmha_fwd_splitkv_oneshot_<trait_{F_idx}, {F_arch.tag}>(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    if constexpr({F_mode} == false) {{ // batch mode
        // we don't check every seqlen_k values for kvcache
        if (a.seqlen_k_ptr != nullptr) {{
            run_instance</*kHasUnevenSplits=*/true>(s, a);
        // make sure F_bn0 is divisible by F_bk1
        }} else if (a.seqlen_k % (a.num_splits * {F_bn0}) == 0) {{
            run_instance</*kHasUnevenSplits=*/false>(s, a);
        }} else {{
            run_instance</*kHasUnevenSplits=*/true>(s, a);
        }}
    }} else {{
        run_instance</*kHasUnevenSplits=*/true>(s, a);
    }}
}}

template<>
std::string fmha_fwd_splitkv_get_name_<trait_{F_idx}, {F_arch.tag}>()
{{
    using k_ = instance<true>::fmha_kernel; /// FIXME: choose real kernel type
    return k_::GetName();
}}

#endif // !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})
"""

FMHA_FWD_SPLITKV_COMBINE_KERNEL_BODY = """
#include <iostream>

#if !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})

using fmha_dtype_{F_idx} = {F_dtype};

namespace {{
template <ck_tile::index_t kLogMaxSplits>
struct instance {{
using fmha_trait = ck_tile::TileFmhaFwdSplitKVCombineTraits<{F_spad},
                                                    {F_dvpad},
                                                    {F_lse},
                                                    {F_squant},
                                                    kLogMaxSplits,
                                                    {F_occupancy}>;

using fmha_pipeline_problem = ck_tile::BlockFmhaSplitKVCombinePipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::ODataType,
    {F_hdim},
    {F_mode},
    {F_bn1},
    fmha_trait>;

using fmha_pipeline = ck_tile::BlockFmhaFwdSplitKVCombinePipeline<
    fmha_pipeline_problem>;

/// FIXME: use {F_spad}/{F_dvpad} as kPadM/kPadN parameters after solving
///        store_tile_raw() data corruption issue
using fmha_epilogue =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType,
                                           typename FmhaFwdTypeConfig<{F_dtype}>::ODataType,
                                           false, false>>;

using fmha_kernel =
    ck_tile::FmhaFwdSplitKVCombineKernel<fmha_pipeline, fmha_epilogue>;

static void run(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    using k_ = fmha_kernel;
    auto [kargs, grids] = fmha_fwd_splitkv_combine_create_kargs_and_grids<k_>(a);
    const dim3 blocks                      = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<kBlockPerCu, {F_arch.tag}>(k_{{}}, grids, blocks, 0, kargs)(ck_tile::stream_config{{s.stream_id_}});
}}
}}; // struct instance
}} // anonymous namespace

using trait_{F_idx} = fmha_fwd_splitkv_combine_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bn1},
                        {F_lse}, {F_squant}, {F_spad}, {F_dvpad}>;

template<>
void fmha_fwd_splitkv_combine_oneshot_<trait_{F_idx}, {F_arch.tag}>(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    if (a.num_splits <= 8) {{
        instance<3>::run(s, a);
    }} else if (a.num_splits <= 16) {{
        instance<4>::run(s, a);
    }} else if (a.num_splits <= 32) {{
        instance<5>::run(s, a);
    }} else if (a.num_splits <= 64) {{
        instance<6>::run(s, a);
    }} else if (a.num_splits <= 128) {{
        instance<7>::run(s, a);
    }}
}}

template<>
std::string fmha_fwd_splitkv_combine_get_name_<trait_{F_idx}, {F_arch.tag}>()
{{
    using k_ = instance<6>::fmha_kernel; /// FIXME: choose real kernel type
    return k_::GetName();
}}

#endif // !defined(__HIP_DEVICE_COMPILE__) || ({F_arch.preprocessor_check})
"""

FMHA_FWD_SPLITKV_API_FILENAME = "fmha_fwd_splitkv_api.cpp"
FMHA_FWD_SPLITKV_API = """
#include <iostream>

template<typename fmha_fwd_splitkv_traits_, typename fmha_fwd_splitkv_combine_traits_, typename Arch>
float fmha_fwd_splitkv_(const ck_tile::stream_config& s, fmha_fwd_splitkv_args a)
{{
    if(s.log_level_ > 0)
        std::cout
            << ", " << fmha_fwd_splitkv_get_name_<fmha_fwd_splitkv_traits_, Arch>()
            << ", " << fmha_fwd_splitkv_combine_get_name_<fmha_fwd_splitkv_combine_traits_, Arch>()
            << std::flush;

    // qr_tdm_pingpong's split kernel is the forward kernel: what it refuses, the call does
    if constexpr(fmha_fwd_splitkv_traits_::FmhaPipelineEnum ==
                 ck_tile::BlockFmhaPipelineEnum::QRKSVS_TDM_PINGPONG)
    {{
        if(!fmha_fwd_splitkv_supported_<fmha_fwd_splitkv_traits_, Arch>(a))
            return -1;
    }}
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){{ fmha_fwd_splitkv_oneshot_<fmha_fwd_splitkv_traits_, Arch>(s_, a); }},
        [=](const ck_tile::stream_config& s_){{ fmha_fwd_splitkv_combine_oneshot_<fmha_fwd_splitkv_combine_traits_, Arch>(s_, a); }}
    );
}}

float fmha_fwd_splitkv(fmha_fwd_splitkv_traits t, fmha_fwd_splitkv_args a, const ck_tile::stream_config& s) {{
    float r = -1;

    [[maybe_unused]] const std::string device_name = ck_tile::get_device_name();

{F_dispatch}
    return r;
}}
"""

FMHA_FWD_SPLITKV_API_INNER_DISPATCH = """{F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && (t.has_logits_soft_cap == {F_logits}) && ({F_mask_check}) && (t.bias_type == {F_bias_check}) && (t.do_fp8_static_quant == {F_squant}) && (t.qscale_type == {F_qscale_check}) &&
        ((a.block_table_ptr != nullptr) == {F_pagedkv}) && ({F_sink_check}) && ({F_scheck}) && ({F_seqtune}) && ({F_skcheck}) && ({F_dcheck}) && ({F_dvcheck}) && ({F_constraint})) {{
    using traits_ = fmha_fwd_splitkv_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}, {F_vlayout}, {F_pipeline_enum}, {F_logits}, {F_mask}, {F_bias}, true, {F_squant}, {F_pagedkv},{F_sink}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}, {F_occupancy}, {F_qscale_enum}>;

    // get combine kernel tile sizes
    using OaccDataType = typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType;
    constexpr ck_tile::index_t kM0 = ck_tile::BlockFmhaSplitKVCombinePipelineTileSizes<OaccDataType, {F_bn1comb}>::kM0;

    // make sure we can reuse the padding flags in combine kernels
    static_assert({F_bm0} % kM0 == 0);
    static_assert({F_bn1} % {F_bn1comb} == 0);

    if (t.has_lse) {{
        using traits2_ = fmha_fwd_splitkv_combine_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bn1comb}, true, {F_squant}, {F_spad}, {F_dvpad}>;

        return fmha_fwd_splitkv_<traits_, traits2_, {F_arch.tag}>(s, a);
    }} else {{
        using traits2_ = fmha_fwd_splitkv_combine_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bn1comb}, false, {F_squant}, {F_spad}, {F_dvpad}>;

        return fmha_fwd_splitkv_<traits_, traits2_, {F_arch.tag}>(s, a);
    }}
}}
"""


@dataclass
class FmhaFwdSplitKVApiTrait:
    arch: ArchTrait
    pipeline_tag: str
    # sync with fmha_fwd_traits<>, to generate fallback calls
    hdim: int
    dtype: str  # data type
    mode: str  # value from MODE_MAP
    bm0: int  # tile size along q seqlen (block size)
    bn0: int  # tile size along qk seqlen
    bk0: int  # tile size along qk gemm unroll
    bn1: int  # tile size along v head_dim
    bk1: int  # tile size along kv gemm unroll
    bk0max: int
    vlayout: str
    mask: str
    logits: str
    bias: str  #
    lse: str  #
    squant: str  #
    spad: str
    skpad: str
    dpad: str
    dvpad: str
    pagedkv: str
    sink: str  # sink or not
    bn1comb: int  # tile size along v head_dim of combine kernel
    qscale: str = "no"  # value from QSCALE_CHECK_MAP
    # the head-dim bucket dispatching the instance, where it is not its own (an alias of
    # another bucket's instance, GFX125_SPLITKV_FALLBACK_BUCKETS)
    bucket: Optional[int] = None
    # Metadata-only hint (occupancy of the splitkv kernel), see
    # fmha_fwd_traits_::kOccupancy for rationale.
    occupancy: int = -1
    constraint: CppConstraint = field(default_factory=lambda: CppConstraint())

    @property
    def name(self) -> str:
        return (
            f"{self.hdim}-{self.dtype}-{self.mode}-{self.bm0}-{self.bn0}-{self.bk0}-{self.bn0}-{self.bk1}-{self.bk0max}-"
            + f"{self.vlayout}-{self.logits}-{self.mask}-{self.bias}-{self.lse}-{self.squant}-{self.spad}-{self.skpad}-{self.dpad}-"
            + f"{self.dvpad}-{self.pagedkv}-{self.sink}"
        )

    @property
    def scheck(self) -> str:
        if self.mode == "group":
            return "true/*group mode spad always true*/"  # group mode only generate spad/skpad == true
        if self.pipeline_tag == "qr_async":
            if self.spad == "t":
                return "true"  # always support
            else:
                return "true"
        elif self.pipeline_tag in SPLITKV_PIPELINES_TAG:
            if self.spad == "t":
                return f"true /*a.seqlen_q % {self.bm0} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
            else:
                return f"a.seqlen_q % {self.bm0} == 0"
        else:
            assert False

    def seqtune(self, max_bm0: int) -> str:
        if self.bm0 == max_bm0:
            return "true/*fall back to largest tile*/"
        elif self.pipeline_tag == "qr_tdm_pingpong" and self.bm0 == 16:
            rows = qr_tdm_pingpong_splitkv_decode_rows(self.dtype, self.hdim)
            if self.mode == "group":
                return f"a.max_seqlen_q <= {rows}"
            return f"a.seqlen_q <= {rows}"
        else:
            if self.mode == "group":
                return f"a.max_seqlen_q <= {self.bm0}"
            return f"a.seqlen_q <= {self.bm0}"

    @property
    def skcheck(self) -> str:
        if self.mode == "group":
            return "true/*group mode skpad always true*/"  # group mode only generate spad/skpad == true
        if self.pipeline_tag == "qr_async":
            if self.skpad == "t":
                return f"a.seqlen_k == 0 || a.seqlen_k % {self.bn0} != 0"
            else:
                return f"a.seqlen_k != 0 && a.seqlen_k % {self.bn0} == 0"
        elif self.pipeline_tag == "qr_tdm_pingpong":
            # the forward kernel reads one physical K length and each batch's own K/V: no
            # per-batch K lengths (KV cache) or cache batch index; fp8 block scales of whole
            # 32-key chunks (as the forward instances)
            check = "a.seqlen_k_ptr == nullptr && a.cache_batch_idx == nullptr"
            if self.qscale == "blockscale":
                check += " && " + _gfx125_qr_tdm_pingpong_block_scales()
            return check
        elif self.pipeline_tag in SPLITKV_PIPELINES_TAG:
            if self.skpad == "t":
                return f"true /*a.seqlen_k_ptr != nullptr || a.seqlen_k % {self.bn0} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
            else:
                return f"a.seqlen_k_ptr == nullptr && a.seqlen_k % {self.bn0} == 0"
        else:
            assert False

    @property
    def dcheck(self) -> str:
        if self.pipeline_tag == "qr_async":
            vec = int((32 * 4) / DTYPE_BITS[self.dtype])
            if self.dpad == "t":
                return f"a.hdim_q % {vec} == 0"
            else:
                assert False
        elif self.pipeline_tag in TDM_PIPELINES_TAG:
            if self.dpad == "t":
                return "a.hdim_q % 8 == 0"
            return f"a.hdim_q % {TDM_K0_MAX_SUBMAX_MAP[self.bk0max]} == 0"
        elif self.pipeline_tag == "qr_tdm_pingpong":
            # padded: any head dim of the bucket; unpadded: exactly the instance's
            return "true" if self.dpad == "t" else f"a.hdim_q == {self.hdim}"
        elif self.pipeline_tag in SPLITKV_PIPELINES_TAG:
            bk0submax = K0_MAX_SUBMAX_MAP[self.bk0max]
            if self.dpad == "t":
                return f"true /*a.hdim_q % {bk0submax} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
            else:
                return f"a.hdim_q % {bk0submax} == 0"
        else:
            assert False

    @property
    def dvcheck(self) -> str:
        if self.pipeline_tag == "qr_async":
            vec = int((32 * 4) / DTYPE_BITS[self.dtype])
            if self.dvpad == "t":
                return f"a.hdim_v % {vec} == 0"
            else:
                assert False
        elif self.pipeline_tag in TDM_PIPELINES_TAG:
            if self.dvpad == "t":
                return "a.hdim_v % 8 == 0"
            return f"a.hdim_v % {TDM_K0_MAX_SUBMAX_MAP[self.bk0max]} == 0"
        elif self.pipeline_tag == "qr_tdm_pingpong":
            # the tile's V columns (bn1): (192, 128) takes no wider hdim_v of its bucket
            return (
                f"a.hdim_v <= {self.bn1}"
                if self.dvpad == "t"
                else f"a.hdim_v == {self.bn1}"
            )
        elif self.pipeline_tag in SPLITKV_PIPELINES_TAG:
            bk0submax = K0_MAX_SUBMAX_MAP[self.bk0max]
            if self.dvpad == "t":
                return f"true /*a.hdim_v % {bk0submax} != 0*/"  # TODO: order of get_pipelines() matters! (ugly)
            else:
                return f"a.hdim_v % {bk0submax} == 0"
        else:
            assert False


@dataclass
class FmhaFwdSplitKVPipeline:
    tag: str

    F_vlayout: str  # row/col
    F_spad: str  # true/false
    F_skpad: str  #
    F_dpad: str  #
    F_dvpad: str  #
    F_logits: str  # t/f
    F_bias: str  # true/false
    F_lse: str  #
    F_squant: str  #
    F_pagedkv: str  # t/f
    F_sink: str  # t/f
    F_mask: str  # value from MASK_MAP
    F_qscale: str = "no"  # value from QSCALE_MAP (qr_tdm_pingpong's fp8 / MX split-KV)

    @property
    def name(self) -> str:
        def pad_name() -> str:
            n = ""
            if self.F_spad == "t":
                n += "s"
            if self.F_skpad == "t":
                n += "sk"
            if self.F_dpad == "t":
                n += "d"
            if self.F_dvpad == "t":
                n += "dv"
            if n != "":
                n = "p" + n
            return n

        pn = pad_name()
        n = f"{self.tag}_v{self.F_vlayout[0]}"
        if pn != "":
            n += f"_{pn}"
        else:
            n += "_npad"

        if self.F_logits == "t":
            n += "_logits"
        else:
            n += "_nlogits"

        if self.F_bias != "no":
            n += f"_{self.F_bias}"
        else:
            n += "_nbias"

        if self.F_mask[0:2] == "s_":
            if self.F_mask == "s_mask":
                n += "_mask"
            else:
                n += "_nmask"
        else:
            if self.F_mask != "no":
                n += f"_m{self.F_mask[0]}"
            else:
                n += "_nmask"

        if self.F_lse == "t":
            n += "_lse"
        else:
            n += "_nlse"

        if self.F_squant == "t":
            n += "_squant"
        else:
            n += "_nsquant"

        if self.F_pagedkv == "t":
            n += "_pagedkv"
        else:
            n += "_npagedkv"
        if self.F_sink == "t":
            n += "_sink"
        else:
            n += "_nsink"
        if self.F_qscale != "no":
            n += f"_{self.F_qscale}"
        return n


@dataclass
class FmhaFwdSplitKVCombinePipeline:
    tag: str

    F_spad: str  # true/false
    F_dvpad: str  #
    F_lse: str  #
    F_squant: str  #

    @property
    def name(self) -> str:
        def pad_name() -> str:
            n = ""
            if self.F_spad == "t":
                n += "s"
            if self.F_dvpad == "t":
                n += "dv"
            if n != "":
                n = "p" + n
            return n

        pn = pad_name()
        n = f"{self.tag}"
        if pn != "":
            n += f"_{pn}"
        else:
            n += "_npad"

        if self.F_lse == "t":
            n += "_lse"
        else:
            n += "_nlse"

        if self.F_squant == "t":
            n += "_squant"
        else:
            n += "_nsquant"
        return n


class FmhaFwdSplitKVApiPool:
    def __init__(self, mask_impl):
        self.pool = OrderedDict()
        self.mask_impl = mask_impl

    def register_traits(self, trait: FmhaFwdSplitKVApiTrait) -> None:
        hdim = trait.hdim if trait.bucket is None else trait.bucket
        ts = (
            self.pool.setdefault(trait.arch, OrderedDict())
            .setdefault(trait.dtype, OrderedDict())
            .setdefault(hdim, [])
        )
        check_duplicates_and_paddings(ts, trait)
        ts.append(copy.copy(trait))

    @property
    def api(self) -> str:
        per_arch = str()
        for i_arch, (arch, pool_by_arch) in enumerate(self.pool.items()):
            per_dtypes = str()
            for i_dtype, (dtype, pool_by_dtype) in enumerate(pool_by_arch.items()):
                per_hdim_case = str()
                for i_hdim, (hdim, pool_by_hdim) in enumerate(pool_by_dtype.items()):
                    max_bm0 = max((t.bm0 for t in pool_by_hdim), default=0)
                    inners = str()
                    for i_trait, trait in enumerate(pool_by_hdim):
                        inners += FMHA_FWD_SPLITKV_API_INNER_DISPATCH.format(
                            F_if=if_(i_trait),
                            F_arch=arch,
                            F_mode=MODE_MAP[trait.mode],
                            F_vlayout=LAYOUT_MAP[trait.vlayout],
                            F_pipeline_enum=PIPELINE_ENUM_MAP[trait.pipeline_tag],
                            F_logits=BOOL_MAP[trait.logits],
                            F_mask=get_mask_map(self.mask_impl)[trait.mask],
                            F_mask_check="true"
                            if qr_tdm_pingpong_any_mask(trait, self.mask_impl)
                            else get_mask_check_map(self.mask_impl)[trait.mask],
                            F_bias_check=BIAS_CHECK_MAP[trait.bias],
                            F_bias=BIAS_MAP[trait.bias],
                            F_lse=BOOL_MAP[trait.lse],
                            F_squant=BOOL_MAP[trait.squant],
                            F_qscale_check=QSCALE_CHECK_MAP[trait.qscale],
                            F_qscale_enum=QSCALE_MAP[trait.qscale],
                            F_pagedkv=BOOL_MAP[trait.pagedkv],
                            F_sink=BOOL_MAP[trait.sink],
                            F_sink_check=f"t.has_sink == {BOOL_MAP[trait.sink]}",
                            F_scheck=trait.scheck,
                            F_seqtune=trait.seqtune(max_bm0),
                            F_skcheck=trait.skcheck,
                            F_dcheck=trait.dcheck,
                            F_dvcheck=trait.dvcheck,
                            F_constraint=trait.constraint,
                            F_spad=BOOL_MAP[trait.spad],
                            F_skpad=BOOL_MAP[trait.skpad],
                            F_dpad=BOOL_MAP[trait.dpad],
                            F_dvpad=BOOL_MAP[trait.dvpad],
                            F_bm0=trait.bm0,
                            F_bn0=trait.bn0,
                            F_bk0=trait.bk0,
                            F_bn1=trait.bn1,
                            F_bk1=trait.bk1,
                            F_bk0max=trait.bk0max,
                            F_occupancy=trait.occupancy,
                            F_hdim=trait.hdim,
                            F_dtype=FWD_DTYPE_MAP[dtype],
                            F_bn1comb=trait.bn1comb,
                        )
                    per_hdim_case += FMHA_FWD_API_PER_HDIM_CASE.format(
                        F_if=if_(i_hdim),
                        F_hdim=hdim,
                        F_hdim_v=hdim,
                        F_inner_dispatch=indent(inners),
                    )
                per_dtypes += FMHA_FWD_API_PER_DTYPE.format(
                    F_if=if_(i_dtype), F_dtype=dtype, F_hdim_case=indent(per_hdim_case)
                )
            per_arch += FMHA_FWD_API_PER_ARCH.format(
                F_if=if_(i_arch),
                F_arch=arch,
                F_dtype_case=indent(per_dtypes),
            )
        if not per_arch:
            # empty string we add some ignore to suppress warning in api
            per_arch = "(void)t; (void)s; (void)a;"
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_SPLITKV_API.format(
            F_dispatch=indent(per_arch)
        )


@dataclass
class FmhaFwdSplitKVCombineTileSize:
    F_bn1: int  # tile size along v head_dim
    F_occupancy: int  # occupancy, -1 will let pipeline decide the occupancy, other value will overwrite occupancy

    @property
    def name(self) -> str:
        return f"b{self.F_bn1}" + (
            "" if self.F_occupancy == -1 else f"_o{self.F_occupancy}"
        )


@dataclass
class FmhaFwdSplitKVKernel:
    F_arch: ArchTrait
    F_idx: int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim: int  # hdim
    F_dtype: str  # data type
    F_mode: str  # value from MODE_MAP
    F_tile: FmhaFwdTileSize
    F_pipeline: FmhaFwdSplitKVPipeline
    mask_impl: str

    def _pingpong_template(self) -> str:
        # the LDS read schedule the forward kernel runs on this tile
        sp, vbs = qr_tdm_pingpong_lds_read_schedule(
            (self.F_hdim, self.F_tile.F_bn1), self.F_pipeline.F_qscale
        )
        fwd_pipeline = qr_tdm_pingpong_effective_schedule(
            self.F_tile,
            FmhaFwdPipeline(
                "qr_tdm_pingpong", self.F_pipeline.F_vlayout, self.F_pipeline.F_spad,
                self.F_pipeline.F_skpad, self.F_pipeline.F_dpad, self.F_pipeline.F_dvpad,
                self.F_pipeline.F_logits, self.F_pipeline.F_bias, "t", "f",
                self.F_pipeline.F_qscale, self.F_pipeline.F_mask, "f", "f", "f",
                F_pp_slice_prefetch=sp, F_pp_v_before_softmax=vbs,
            ),
        )  # fmt: skip
        t = self.F_tile
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_SPLITKV_PINGPONG_KERNEL_BODY.format(
            F_idx=self.F_idx,
            F_kname=self.name,
            F_arch=self.F_arch,
            F_hdim=self.F_hdim,
            F_dtype=FWD_DTYPE_MAP[self.F_dtype],
            F_bm0=t.F_bm0, F_bn0=t.F_bn0, F_bk0=t.F_bk0, F_bn1=t.F_bn1, F_bk1=t.F_bk1,
            F_bk0max=t.F_bk0max,
            F_rm0=t.F_rm0, F_rn0=t.F_rn0, F_rk0=t.F_rk0,
            F_rm1=t.F_rm1, F_rn1=t.F_rn1, F_rk1=t.F_rk1,
            F_wm0=t.F_wm0, F_wn0=t.F_wn0, F_wk0=t.F_wk0,
            F_wm1=t.F_wm1, F_wn1=t.F_wn1, F_wk1=t.F_wk1,
            F_vlayout=LAYOUT_MAP[self.F_pipeline.F_vlayout],
            F_spad=BOOL_MAP[self.F_pipeline.F_spad],
            F_skpad=BOOL_MAP[self.F_pipeline.F_skpad],
            F_dpad=BOOL_MAP[self.F_pipeline.F_dpad],
            F_dvpad=BOOL_MAP[self.F_pipeline.F_dvpad],
            F_logits=BOOL_MAP[self.F_pipeline.F_logits],
            F_bias=BIAS_MAP[self.F_pipeline.F_bias],
            F_sink=BOOL_MAP[self.F_pipeline.F_sink],
            F_occupancy=t.F_occupancy,
            F_pipeline_enum=PIPELINE_ENUM_MAP[self.F_pipeline.tag],
            F_mask=get_mask_map(self.mask_impl)[self.F_pipeline.F_mask],
            F_mode=MODE_MAP[self.F_mode],
            F_pp_slice_prefetch=fwd_pipeline.F_pp_slice_prefetch,
            F_pp_v_before_softmax=fwd_pipeline.F_pp_v_before_softmax,
            F_qscale=QSCALE_MAP[self.F_pipeline.F_qscale],
        )  # fmt: skip

    @property
    def template(self) -> str:
        assert self.F_pipeline.F_lse == "t"
        if self.F_pipeline.tag == "qr_tdm_pingpong":
            return self._pingpong_template()
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_SPLITKV_KERNEL_BODY.format(
            F_idx=self.F_idx,
            F_arch=self.F_arch,
            F_hdim=self.F_hdim,
            F_dtype=FWD_DTYPE_MAP[self.F_dtype],
            F_bm0=self.F_tile.F_bm0,
            F_bn0=self.F_tile.F_bn0,
            F_bk0=self.F_tile.F_bk0,
            F_bn1=self.F_tile.F_bn1,
            F_bk1=self.F_tile.F_bk1,
            F_bk0max=self.F_tile.F_bk0max,
            F_rm0=self.F_tile.F_rm0,
            F_rn0=self.F_tile.F_rn0,
            F_rk0=self.F_tile.F_rk0,
            F_rm1=self.F_tile.F_rm1,
            F_rn1=self.F_tile.F_rn1,
            F_rk1=self.F_tile.F_rk1,
            F_wm0=self.F_tile.F_wm0,
            F_wn0=self.F_tile.F_wn0,
            F_wk0=self.F_tile.F_wk0,
            F_wm1=self.F_tile.F_wm1,
            F_wn1=self.F_tile.F_wn1,
            F_wk1=self.F_tile.F_wk1,
            F_vlayout=LAYOUT_MAP[self.F_pipeline.F_vlayout],
            F_spad=BOOL_MAP[self.F_pipeline.F_spad],
            F_skpad=BOOL_MAP[self.F_pipeline.F_skpad],
            F_dpad=BOOL_MAP[self.F_pipeline.F_dpad],
            F_dvpad=BOOL_MAP[self.F_pipeline.F_dvpad],
            F_logits=BOOL_MAP[self.F_pipeline.F_logits],
            F_bias=BIAS_MAP[self.F_pipeline.F_bias],
            F_lse=BOOL_MAP[self.F_pipeline.F_lse],
            F_squant=BOOL_MAP[self.F_pipeline.F_squant],
            F_pagedkv=BOOL_MAP[self.F_pipeline.F_pagedkv],
            F_occupancy=self.F_tile.F_occupancy,
            F_sink=BOOL_MAP[self.F_pipeline.F_sink],
            F_pipeline_enum=PIPELINE_ENUM_MAP[self.F_pipeline.tag],
            F_mask=get_mask_map(self.mask_impl)[self.F_pipeline.F_mask],
            F_mode=MODE_MAP[self.F_mode],
            F_pipeline=FMHA_FWD_SPLITKV_PIPELINE_MAP[self.F_pipeline.tag],
            F_merge_masked=BOOL_MAP[
                self.F_pipeline.tag in TDM_PIPELINES_TAG
                and self.F_pipeline.F_sink == "f"
            ],
            F_merge_sink=BOOL_MAP[self.F_pipeline.tag in TDM_PIPELINES_TAG],
        )

    @property
    def name(self) -> str:
        # TODO: we don't encode idx here
        return (
            f"fmha_fwd_splitkv_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_"
            + self.F_tile.name
            + "_"
            + self.F_pipeline.name
        )

    @property
    def filename(self) -> str:
        return f"{self.name}{self.F_arch.filename_suffix}.cpp"


@dataclass
class FmhaFwdSplitKVCombineKernel:
    F_arch: ArchTrait
    F_idx: int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim: int  # hdim
    F_dtype: str  # data type
    F_mode: str  # value from MODE_MAP
    F_tile: FmhaFwdSplitKVCombineTileSize
    F_pipeline: FmhaFwdSplitKVCombinePipeline

    @property
    def template(self) -> str:
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_SPLITKV_COMBINE_KERNEL_BODY.format(
            F_idx=self.F_idx,
            F_arch=self.F_arch,
            F_hdim=self.F_hdim,
            F_dtype=FWD_DTYPE_MAP[self.F_dtype],
            F_bn1=self.F_tile.F_bn1,
            F_spad=BOOL_MAP[self.F_pipeline.F_spad],
            F_dvpad=BOOL_MAP[self.F_pipeline.F_dvpad],
            F_lse=BOOL_MAP[self.F_pipeline.F_lse],
            F_squant=BOOL_MAP[self.F_pipeline.F_squant],
            F_occupancy=self.F_tile.F_occupancy,
            F_mode=MODE_MAP[self.F_mode],
        )

    @property
    def name(self) -> str:
        # TODO: we don't encode idx here
        return (
            f"fmha_fwd_splitkv_combine_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_"
            + self.F_tile.name
            + "_"
            + self.F_pipeline.name
        )

    @property
    def filename(self) -> str:
        return f"{self.name}{self.F_arch.filename_suffix}.cpp"


class KernelComponentFactoryBase:
    @staticmethod
    def get_pipelines(dtype, hdim, mask_impl) -> List[FmhaFwdSplitKVPipeline]:
        # this function will populate a list possible pipelines
        # TODO: the order of List matters! the later in this list will be also be checked later
        # TODO: currently for qr pipeline, let "t" padding to appear later!!
        # TODO: how to design this more generic?
        Pipeline = FmhaFwdSplitKVPipeline
        squant = "f"
        pipelines = []
        if dtype in ["fp16", "bf16"]:
            for logits, mask, bias, pagedkv, sink in itertools.product(
                ["t", "f"],
                get_mask_map(mask_impl).keys(),
                BIAS_MAP.keys(),
                ["t", "f"],
                ["t", "f"],
            ):
                pipelines.append(Pipeline("qr", "row", "f", "t", "f", "f", logits, bias, "t", squant, pagedkv, sink, mask))  # fmt: skip
                pipelines.append(Pipeline("qr", "row", "t", "f", "f", "f", logits, bias, "t", squant, pagedkv, sink, mask))  # fmt: skip
                pipelines.append(Pipeline("qr", "row", "t", "t", "f", "f", logits, bias, "t", squant, pagedkv, sink, mask))  # fmt: skip
                pipelines.append(Pipeline("qr", "row", "t", "t", "t", "t", logits, bias, "t", squant, pagedkv, sink, mask))  # fmt: skip
                pipelines.append(Pipeline("qr_nwarp_sshuffle", "row", "t", "t", "f", "f", logits, bias, "t", squant, pagedkv, sink, mask))  # fmt: skip
        elif dtype in ["bf8"]:
            for logits, mask, bias in itertools.product(
                ["t", "f"], get_mask_map(mask_impl).keys(), BIAS_MAP.keys()
            ):
                pipelines.append(Pipeline("qr", "row", "f", "f", "f", "f", logits, bias, "t", squant, "f", "f", mask))  # fmt: skip
                pipelines.append(Pipeline("qr", "row", "t", "t", "f", "f", logits, bias, "t", squant, "f", "f", mask))  # fmt: skip
        elif dtype in ["fp8fp16", "fp8bf16"]:
            # TODO
            None
        else:
            assert False
        return pipelines

    @staticmethod
    def get_combine_pipelines(dtype, hdim) -> List[FmhaFwdSplitKVCombinePipeline]:
        Pipeline = FmhaFwdSplitKVCombinePipeline
        squant = "f"
        pipelines = []
        if dtype in ["fp16", "bf16"]:
            for spad, dvpad, lse in itertools.product(
                ["t", "f"], ["t", "f"], ["t", "f"]
            ):
                pipelines.append(Pipeline("unused", spad, dvpad, lse, squant))
        elif dtype in ["bf8"]:
            # no need lse kernels
            for spad, dvpad in itertools.product(["t", "f"], ["t", "f"]):
                pipelines.append(Pipeline("unused", spad, dvpad, "f", squant))
        else:
            assert False
        return pipelines

    @staticmethod
    def get_combine_hdim_tile_size_dict(dtype: str) -> Optional[dict]:
        # Possible values of F_bn1: 8, 16, 32
        if dtype in ["fp16", "bf16"]:
            return {
                "32": FmhaFwdSplitKVCombineTileSize(32, -1),
                "64": FmhaFwdSplitKVCombineTileSize(32, -1),
                "96": FmhaFwdSplitKVCombineTileSize(32, -1),
                "128": FmhaFwdSplitKVCombineTileSize(32, -1),
                # "160" : FmhaFwdSplitKVCombineTileSize(32, -1),
                "256": FmhaFwdSplitKVCombineTileSize(32, -1),
            }
        elif dtype in ["bf8"]:
            return {
                "64": FmhaFwdSplitKVCombineTileSize(32, -1),
                "128": FmhaFwdSplitKVCombineTileSize(32, -1),
                "256": FmhaFwdSplitKVCombineTileSize(32, -1),
            }
        else:
            return None


class KernelComponentFactoryGfx9(KernelComponentFactoryBase):
    arch = ArchTrait("gfx9")

    @staticmethod
    def get_hdim_tile_size_dict(dtype: str) -> Optional[dict]:
        if dtype in ["fp16", "bf16"]:
            return {
                "32" : FmhaFwdTileSize( 32,  64, 16,  32, 32,  32, 2, 1, 1, 2, 1, 1, 16, 16, 16, 16, 16, 16, -1),
                "64" : FmhaFwdTileSize( 64,  64, 32,  64, 32,  64, 4, 1, 1, 4, 1, 1, 16, 16, 16, 16, 16, 16, -1),
                "96" : FmhaFwdTileSize( 64, 128, 32, 128, 32,  96, 4, 1, 1, 4, 1, 1, 16, 16, 16, 16, 16, 16, -1),
                "128": FmhaFwdTileSize( 64, 128, 32, 128, 32, 128, 4, 1, 1, 4, 1, 1, 16, 16, 16, 16, 16, 16, -1),
                # "160" : FmhaFwdTileSize(64, 128, 32, 160, 32, 160, 4, 1, 1, 4, 1, 1, 16, 16, 16, 16, 16, 16, -1),
                "256": FmhaFwdTileSize( 64, 128, 32, 256, 32, 256, 4, 1, 1, 4, 1, 1, 16, 16, 16, 16, 16, 16, -1),
            }  # fmt: skip
        elif dtype in ["bf8"]:
            return {
                "64" : FmhaFwdTileSize(128,  64, 32,  64, 32,  64, 2, 1, 1, 2, 1, 1, 32, 32, 32, 32, 32, 32, -1),
                "128": FmhaFwdTileSize(128, 128, 32, 128, 32, 128, 4, 1, 1, 4, 1, 1, 32, 32, 32, 32, 32, 32, -1),
            }  # fmt: skip
        else:
            return None


class KernelComponentFactoryGfx11(KernelComponentFactoryBase):
    arch = ArchTrait("gfx11")

    @staticmethod
    def get_hdim_tile_size_dict(dtype: str) -> Optional[dict]:
        if dtype in ["fp16", "bf16"]:
            return {
                #                       bm0, bn0, bk0, bn1, bk1,
                "32" : [FmhaFwdTileSize( 16,  64,  16,  32,  32,   32,  1, 2, 1,  1, 2, 1,  16, 16, 16,  16, 16, 16,  -1),
                        FmhaFwdTileSize( 64,  64,  16,  32,  32,   32,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1)],
                "64" : [FmhaFwdTileSize( 16,  64,  32,  64,  32,   64,  1, 4, 1,  1, 4, 1,  16, 16, 16,  16, 16, 16,  -1),
                        FmhaFwdTileSize( 64,  64,  32,  64,  32,   64,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1)],
                "128": [FmhaFwdTileSize( 16,  64,  32, 128,  32,  128,  1, 4, 1,  1, 4, 1,  16, 16, 16,  16, 16, 16,  -1),
                        FmhaFwdTileSize( 64,  64,  32, 128,  32,  128,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1)],
                "256": [FmhaFwdTileSize( 16,  64,  32, 256,  32,  256,  1, 4, 1,  1, 4, 1,  16, 16, 16,  16, 16, 16,  -1),
                        FmhaFwdTileSize( 64,  64,  32, 256,  32,  256,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1)],
            }  # fmt: skip
        else:
            return None


class KernelComponentFactoryGfx12(KernelComponentFactoryBase):
    arch = ArchTrait("gfx12")

    @staticmethod
    def get_hdim_tile_size_dict(dtype: str) -> Optional[dict]:
        if dtype in ["fp16", "bf16"]:
            return {
                #                       bm0, bn0, bk0, bn1, bk1,
                "32" : [FmhaFwdTileSize( 16,  64,  16,  32,  32,   32,  1, 2, 1,  1, 2, 1,  16, 16, 16,  16, 16, 16,  -1),
                        FmhaFwdTileSize( 64,  64,  16,  32,  32,   32,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1)],
                "64" : [FmhaFwdTileSize( 16,  64,  32,  64,  32,   64,  1, 4, 1,  1, 4, 1,  16, 16, 16,  16, 16, 16,  -1),
                        FmhaFwdTileSize( 64,  64,  32,  64,  32,   64,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1)],
                "128": [FmhaFwdTileSize( 16, 128,  32, 128,  32,  128,  1, 8, 1,  1, 8, 1,  16, 16, 16,  16, 16, 16,  -1),
                        FmhaFwdTileSize( 64,  64,  32, 128,  32,  128,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1)],
                "256": [FmhaFwdTileSize( 16, 128,  32, 256,  32,  256,  1, 8, 1,  1, 8, 1,  16, 16, 16,  16, 16, 16,  -1),
                        FmhaFwdTileSize( 64,  64,  32, 256,  32,  256,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1)],
            }  # fmt: skip
        elif dtype in ["bf8"]:
            return {
                #                      bm0, bn0, bk0, bn1, bk1,
                "64" : FmhaFwdTileSize(128,  64,  32,  64,  32,   64,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
                "128": FmhaFwdTileSize( 64,  64,  32, 128,  32,  128,  4, 1, 1,  4, 1, 1,  16, 16, 16,  16, 16, 16,  -1),
            }  # fmt: skip
        else:
            return None


class KernelComponentFactoryGfx117(KernelComponentFactoryGfx12):
    arch = ArchTrait("gfx117")


# qr_tdm_pingpong's split-KV head dims (hdim, hdim_v): the forward kernel's
GFX125_QR_TDM_PINGPONG_SPLITKV_HDIMS = GFX125_QR_TDM_PINGPONG_HDIMS


# Split-KV buckets that only qr_tdm_pingpong's instances were made for: the calls it refuses
# (group mode, bias, soft cap, paged KV, sinks, K lengths a batch) dispatch the next qr_tdm
# bucket up's instances there (its padded ones), as before these buckets existed.
GFX125_SPLITKV_FALLBACK_BUCKETS = {96: 128, 160: 256, 192: 256}

# qr_tdm_pingpong's split-KV features: every (logits soft cap, bias, sink) combination. They
# take the padded head-dim instances (any head dim of the bucket) in batch and group mode; with
# the simplified masks only the masking kernel (no mask is its unbounded window,
# qr_tdm_pingpong_any_mask). Sinks stay compiled in or out: compiled in, the kernel gives up
# its no-sink trip schedule (gfx1250 decode, soft cap and group mode 5..28% slower).
GFX125_QR_TDM_PINGPONG_SPLITKV_FEATURES = tuple(
    itertools.product(("f", "t"), ("no", "bias", "alibi"), ("f", "t"))
)


def qr_tdm_pingpong_splitkv_decode_rows(dtype, hdim):
    """The query rows qr_tdm_pingpong's 16-row split tile takes; more take the 64-row one.
    fp8 / MX up to d64 keep it to 32 rows (gfx1250, 32k keys, 32/8 heads: d64 17..32 rows
    fp8 19..33%, MX 9..17% faster than the 64-row tile; d128 and 16-bit inputs no gain)."""
    return 32 if dtype in GFX125_QR_TDM_PINGPONG_SPLITKV_QSCALES and hdim <= 64 else 16


def _gfx125_qr_tdm_pingpong_splitkv_tiles(hdim_pair, fp8=False, mx=False):
    """qr_tdm_pingpong's split-KV tiles: the 16-row decode tile with the longest trips (a
    split's keys are long), then the 64-row one for longer queries (small-query prefill)."""
    tiles = _gfx125_qr_tdm_pingpong_tiles(hdim_pair, fp8=fp8, mx=mx)
    return [tiles[0], next(t for t in tiles if t.F_bm0 == 64)]


def qr_tdm_pingpong_any_mask(trait, mask_impl) -> bool:
    """qr_tdm_pingpong's padded split-KV instances (the features') take every mask type with
    the simplified masks: their masking kernel reads no mask as an unbounded window."""
    return (
        trait.pipeline_tag == "qr_tdm_pingpong"
        and trait.dpad == "t"
        and mask_impl == "simplified"
        and trait.mask == "s_mask"
    )


def _qr_tdm_pingpong_splitkv_pipelines(mask_impl, squant, qscale):
    """qr_tdm_pingpong's split-KV pipelines at one input width: the plain decode on whole head
    dims (batch mode), then every feature on padded head dims (batch and group mode; with the
    simplified masks, on the masking kernel alone: qr_tdm_pingpong_any_mask)."""
    Pipeline = FmhaFwdSplitKVPipeline
    masks = get_mask_map(mask_impl).keys()
    feature_masks = ["s_mask"] if mask_impl == "simplified" else masks

    def masks_of(logits, bias, sink):
        # sinks alone keep a no-mask kernel too, ahead of the masking one (gfx1250 d64 decode,
        # 32k keys, sink logits: the masking kernel 18% slower; the other features within 5%)
        if mask_impl == "simplified" and (logits, bias, sink) == ("f", "no", "t"):
            return ["s_no", "s_mask"]
        return feature_masks

    #                    vlayout spad skpad dpad dvpad logits bias lse squant pagedkv sink mask
    pipelines = [
        Pipeline(
            "qr_tdm_pingpong",
            "row",
            "t",
            "t",
            "f",
            "f",
            "f",
            "no",
            "t",
            squant,
            "f",
            "f",
            mask,
            qscale,
        )  # fmt: skip
        for mask in masks
    ]
    pipelines += [
        Pipeline(
            "qr_tdm_pingpong",
            "row",
            "t",
            "t",
            "t",
            "t",
            logits,
            bias,
            "t",
            squant,
            "f",
            sink,
            mask,
            qscale,
        )  # fmt: skip
        for logits, bias, sink in GFX125_QR_TDM_PINGPONG_SPLITKV_FEATURES
        for mask in masks_of(logits, bias, sink)
    ]
    return pipelines


# qr_tdm_pingpong's fp8 / MX split-KV: the input types and their quantization scales
GFX125_QR_TDM_PINGPONG_SPLITKV_QSCALES = {
    "fp8bf16": ("pertensor", "perhead", "blockscale"),
    "mxfp8bf16": ("mx",),
}


class KernelComponentFactoryGfx125(KernelComponentFactoryBase):
    arch = ArchTrait("gfx125")

    @staticmethod
    def get_combine_pipelines(dtype, hdim) -> List[FmhaFwdSplitKVCombinePipeline]:
        if dtype in GFX125_QR_TDM_PINGPONG_SPLITKV_QSCALES:
            # what qr_tdm_pingpong's split kernels pair with (spad; whole and padded head
            # dims; with and without the LSE)
            return [
                FmhaFwdSplitKVCombinePipeline("unused", "t", dvpad, lse, "f")
                for dvpad, lse in itertools.product(("f", "t"), ("t", "f"))
            ]
        return KernelComponentFactoryBase.get_combine_pipelines(dtype, hdim)

    @staticmethod
    def get_combine_hdim_tile_size_dict(dtype: str) -> Optional[dict]:
        if dtype in GFX125_QR_TDM_PINGPONG_SPLITKV_QSCALES:
            # fp8 / MX: qr_tdm_pingpong's split-KV only (fp32 partials, bf16 output)
            d = {}
        else:
            d = KernelComponentFactoryBase.get_combine_hdim_tile_size_dict(dtype)
        if d is not None and dtype in [
            "fp16",
            "bf16",
            *GFX125_QR_TDM_PINGPONG_SPLITKV_QSCALES,
        ]:
            # the head dims qr_tdm_pingpong splits
            for h, _ in GFX125_QR_TDM_PINGPONG_SPLITKV_HDIMS:
                d.setdefault(str(h), FmhaFwdSplitKVCombineTileSize(32, -1))
        return d

    @staticmethod
    def get_pipelines(dtype, hdim, mask_impl) -> List[FmhaFwdSplitKVPipeline]:
        # Dispatch returns on the first hit, so emission order is dispatch order.
        Pipeline = FmhaFwdSplitKVPipeline
        pipelines = []
        if dtype in ["fp16", "bf16"]:
            squant = "f"
            # qr_tdm_pingpong's decode tile first (its seqtune takes up to 16 query rows); what
            # it does not take (paged KV, KV cache, soft cap or sinks with a bias) is qr_tdm's
            pipelines += _qr_tdm_pingpong_splitkv_pipelines(mask_impl, squant, "no")
            # Entries with an unpadded seqlen go first: spad/skpad == t predicates
            # always hold, so the fully padded (t, t) entries must come last.
            # Soft cap with bias is dropped in get_fwd_splitkv_blobs.
            for logits, mask, bias, pagedkv, sink in itertools.product(
                ["t", "f"],
                get_mask_map(mask_impl).keys(),
                BIAS_MAP.keys(),
                ["t", "f"],
                ["t", "f"],
            ):
                #                             vlayout spad skpad dpad dvpad logits bias lse squant pagedkv sink mask
                pipelines.append(Pipeline("qr_tdm", "row", "f", "t", "f", "f", logits, bias, "t", squant, pagedkv, sink, mask))  # fmt: skip
                pipelines.append(Pipeline("qr_tdm", "row", "t", "f", "f", "f", logits, bias, "t", squant, pagedkv, sink, mask))  # fmt: skip
                pipelines.append(Pipeline("qr_tdm", "row", "t", "t", "f", "f", logits, bias, "t", squant, pagedkv, sink, mask))  # fmt: skip
                pipelines.append(Pipeline("qr_tdm", "row", "t", "t", "t", "t", logits, bias, "t", squant, pagedkv, sink, mask))  # fmt: skip
            # No qr fallback: every entry needs hdim % 8 == 0, so other head
            # dims are unsupported here (the prefill api still serves them).
        elif dtype in ["fp8", "bf8"]:
            squant = "t" if dtype == "fp8" else "f"
            for logits, mask, bias in itertools.product(
                ["t", "f"], get_mask_map(mask_impl).keys(), BIAS_MAP.keys()
            ):
                pipelines.append(Pipeline("qr", "row", "f", "f", "f", "f", logits, bias, "t", squant, "f", "f", mask))  # fmt: skip
                pipelines.append(Pipeline("qr", "row", "t", "t", "f", "f", logits, bias, "t", squant, "f", "f", mask))  # fmt: skip
        elif dtype in GFX125_QR_TDM_PINGPONG_SPLITKV_QSCALES:
            # fp8 / MX: qr_tdm_pingpong's decode tile only, at each quantization scale
            for qscale in GFX125_QR_TDM_PINGPONG_SPLITKV_QSCALES[dtype]:
                pipelines += _qr_tdm_pingpong_splitkv_pipelines(mask_impl, "f", qscale)
        elif dtype in ["fp8fp16", "fp8bf16", "mxfp8", "mxfp4"]:
            pass
        else:
            assert False
        return pipelines

    @staticmethod
    def get_hdim_tile_size_dict(dtype: str) -> Optional[dict]:
        if dtype in GFX125_QR_TDM_PINGPONG_SPLITKV_QSCALES:
            mx = dtype.startswith("mx")
            return {
                str(h): _gfx125_qr_tdm_pingpong_splitkv_tiles((h, h_v), fp8=True, mx=mx)
                for h, h_v in GFX125_QR_TDM_PINGPONG_SPLITKV_HDIMS
            }
        if dtype in ["fp16", "bf16"]:
            tiles = {
                #                      bm0, bn0, bk0, bn1, bk1,
                "32" : FmhaFwdTileSize( 64,  64,  32,  32,  32,   64,  4, 1, 1,  4, 1, 1,  16, 16, 32,  16, 16, 32,  -1),
                "64" : FmhaFwdTileSize( 64,  64,  32,  64,  32,   64,  4, 1, 1,  4, 1, 1,  16, 16, 32,  16, 16, 32,  -1),
                "128": FmhaFwdTileSize( 64,  64,  32, 128,  32,  128,  4, 1, 1,  4, 1, 1,  16, 16, 32,  16, 16, 32,  -1),
                "256": FmhaFwdTileSize( 64,  64,  32, 256,  32,  256,  4, 1, 1,  4, 1, 1,  16, 16, 32,  16, 16, 32,  -1),
            }  # fmt: skip
            # qr_tdm_pingpong's decode tile, the longest-trip one (a split's keys are long), at
            # each of its head dims (hdim_v: 128 at d192), ahead of qr_tdm's
            # (the buckets past qr_tdm's own, 96 / 160 / 192, dispatch the next qr_tdm bucket
            # up's instances behind pingpong's: GFX125_SPLITKV_FALLBACK_BUCKETS)
            result = {}
            for h, h_v in GFX125_QR_TDM_PINGPONG_SPLITKV_HDIMS:
                pp = _gfx125_qr_tdm_pingpong_splitkv_tiles((h, h_v))
                result[str(h)] = pp + ([tiles[str(h)]] if str(h) in tiles else [])
            return result
        elif dtype in ["bf8"]:
            return {
                #                      bm0, bn0, bk0, bn1, bk1,
                "64" : FmhaFwdTileSize(128,  64,  64,  64,  64,  128,  4, 1, 1,  4, 1, 1,  16, 16, 64,  16, 16, 64,  -1),
                "128": FmhaFwdTileSize( 64,  64,  64, 128,  64,  128,  4, 1, 1,  4, 1, 1,  16, 16, 64,  16, 16, 64,  -1),
            }  # fmt: skip
        else:
            return None


def get_factory(target: str):
    # Place more specific architectures first

    if target.startswith("gfx9"):
        return KernelComponentFactoryGfx9
    if target.startswith(("gfx117", "gfx11-7-generic")):
        return KernelComponentFactoryGfx117
    if target.startswith("gfx11"):
        return KernelComponentFactoryGfx11
    if target.startswith("gfx125"):
        return KernelComponentFactoryGfx125
    if target.startswith("gfx12"):
        return KernelComponentFactoryGfx12

    raise Exception(f"Unsupported device target {target}")


def get_fwd_splitkv_blobs(
    targets: List[str], kernel_filter: Optional[str], receipt, mask_impl, optdim_list
) -> List[FmhaFwdSplitKVKernel]:
    Kernel = FmhaFwdSplitKVKernel

    gen = list()

    factories = get_factories_for_targets(targets, get_factory)

    for factory, dtype in itertools.product(factories, FWD_DTYPE_MAP.keys()):
        d = factory.get_hdim_tile_size_dict(dtype)
        if d is None:
            continue
        for hdim_str, mode in itertools.product(d.keys(), MODE_MAP.keys()):
            tiles = d[hdim_str]
            if not isinstance(tiles, list):
                tiles = [tiles]
            hdim = int(hdim_str)
            for tile, pipeline in itertools.product(
                tiles, factory.get_pipelines(dtype, hdim, mask_impl)
            ):
                # qr_tdm_pingpong takes its own tiles (F_pipeline_tag), the others the rest
                is_pingpong_tile = tile.F_pipeline_tag == "qr_tdm_pingpong"
                if is_pingpong_tile != (pipeline.tag == "qr_tdm_pingpong"):
                    continue
                # (qr_tdm_pingpong's split-KV in group mode: its padded instances)
                if is_pingpong_tile and mode == "group" and pipeline.F_dpad == "f":
                    continue
                # Use qr_nwarp_sshuffle with multiple N warps and qr otherwise
                if not is_pingpong_tile and (tile.F_rn0 != 1) != (
                    pipeline.tag == "qr_nwarp_sshuffle"
                ):
                    continue
                # qr_tdm's unpadded head-dim check reads the TDM length map. Where
                # that exceeds the head dim this bucket dispatches on, the
                # predicate can never hold and the instance is unreachable.
                if (
                    pipeline.tag == "qr_tdm"
                    and pipeline.F_dpad == "f"
                    and TDM_K0_MAX_SUBMAX_MAP[tile.F_bk0max] > hdim
                ):
                    continue
                if mode == "group":
                    if pipeline.F_spad != "t" or pipeline.F_skpad != "t":
                        # in group mode, spad/skpad must be true, since we can't predict if seqlen of current batch need pad or not
                        continue
                # logits_soft_cap is only allowed if no bias (qr_tdm_pingpong: either)
                if not (
                    (pipeline.F_logits == "t" and pipeline.F_bias == "no")
                    or pipeline.F_logits == "f"
                    or pipeline.tag == "qr_tdm_pingpong"
                ):
                    continue
                k = Kernel(
                    F_arch=factory.arch,
                    F_idx=0,
                    F_hdim=hdim,
                    F_dtype=dtype,
                    F_mode=mode,
                    F_tile=tile,
                    F_pipeline=pipeline,
                    mask_impl=mask_impl,
                )
                if kernel_filter != "":
                    if not fnmatch.fnmatch(k.name, kernel_filter):
                        continue
                if optdim_list != [-1]:
                    if hdim not in optdim_list:
                        continue
                # Flash attention integration
                if receipt == 2:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_bias in ["no", "alibi"]
                    # FlashAttention splitkv paths use softcap-disabled kernels only.
                    cond &= pipeline.F_logits == "f"
                    cond &= pipeline.F_squant == "f"
                    cond &= pipeline.F_sink == "f"
                    if not cond:
                        continue
                # PyTorch integration
                elif receipt == 4:
                    cond = dtype in ["fp16, bf16"]
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_bias in ["no", "bias"]
                    cond &= pipeline.F_squant == "f"
                    cond &= mode == "batch"
                    cond &= pipeline.F_sink == "f"
                    if not cond:
                        continue
                # CK Tile Python API
                elif receipt == 5:
                    cond = dtype in ["fp16", "bf16"] and hdim <= 256
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_bias in ["no", "alibi"]
                    cond &= pipeline.F_logits == "f"
                    cond &= pipeline.F_squant == "f"
                    cond &= pipeline.F_sink == "f"
                    if not cond:
                        continue
                # Aiter(mha_varlen_fwd) integration
                elif receipt == 200:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= mode == "group"
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_squant == "f"
                    if not cond:
                        continue
                # aiter::mha_fwd_splikv C++ api integration
                elif receipt == 600:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= pipeline.F_vlayout == "row"
                    cond &= pipeline.F_squant == "f"
                    if not cond:
                        continue
                elif receipt == 700:
                    continue  # TE does not use this API

                # fp32 only
                if receipt == 800 or receipt == 801:
                    cond = dtype == "fp32"
                    if not cond:
                        continue

                gen.append(k)

    return gen


def get_fwd_splitkv_combine_blobs(
    targets: List[str], kernel_filter: Optional[str], receipt, optdim_list
) -> List[FmhaFwdSplitKVCombineKernel]:
    Kernel = FmhaFwdSplitKVCombineKernel

    gen = list()

    factories = get_factories_for_targets(targets, get_factory)

    for factory, dtype in itertools.product(factories, FWD_DTYPE_MAP.keys()):
        d = factory.get_combine_hdim_tile_size_dict(dtype)
        if d is None:
            continue
        for hdim_str, mode in itertools.product(d.keys(), MODE_MAP.keys()):
            tile = d[hdim_str]
            hdim = int(hdim_str)
            for pipeline in factory.get_combine_pipelines(dtype, hdim):
                if mode == "group":
                    if pipeline.F_spad != "t":
                        # in group mode, spad/skpad must be true, since we can't predict if seqlen of current batch need pad or not
                        continue
                k = Kernel(
                    F_arch=factory.arch,
                    F_idx=0,
                    F_hdim=hdim,
                    F_dtype=dtype,
                    F_mode=mode,
                    F_tile=tile,
                    F_pipeline=pipeline,
                )
                if kernel_filter != "":
                    if not fnmatch.fnmatch(k.name, kernel_filter):
                        continue
                if optdim_list != [-1]:
                    if hdim not in optdim_list:
                        continue
                # CK Tile Python API. The shared dispatcher references both LSE variants.
                if receipt == 5:
                    cond = dtype in ["fp16", "bf16"] and hdim <= 256
                    cond &= pipeline.F_squant == "f"
                    if not cond:
                        continue
                # Aiter(mha_varlen_fwd) integration
                elif receipt == 200:
                    cond = dtype in ["fp16", "bf16"]
                    cond &= mode == "group"
                    if not cond:
                        continue
                # aiter::mha_fwd_splikv C++ api integration
                elif receipt == 600:
                    cond = dtype in ["fp16", "bf16"]
                    if not cond:
                        continue

                # fp32 only
                if receipt == 800 or receipt == 801:
                    cond = dtype == "fp32"
                    if not cond:
                        continue

                gen.append(k)

    return gen


def write_single_kernel(
    kernel: Union[FmhaFwdSplitKVKernel, FmhaFwdSplitKVCombineKernel], autogen_dir: Path
) -> None:
    update_file(autogen_dir / kernel.filename, kernel.template)


def write_fwd_splitkv_api(api_pool: FmhaFwdSplitKVApiPool, autogen_dir: Path) -> None:
    update_file(autogen_dir / FMHA_FWD_SPLITKV_API_FILENAME, api_pool.api)


def write_blobs(
    targets: List[str],
    output_dir: Path,
    filter_list: str,
    receipt,
    optdim_list,
    mask_impl,
) -> None:
    filter_list = filter_list.split("@")
    filter_list.extend([""] * (2 - len(filter_list)))

    combine_kernels = get_fwd_splitkv_combine_blobs(
        targets, filter_list[0], receipt, optdim_list
    )
    for kernel in combine_kernels:
        write_single_kernel(kernel, output_dir)
    kernels = get_fwd_splitkv_blobs(
        targets, filter_list[1], receipt, mask_impl, optdim_list
    )
    for kernel in kernels:
        write_single_kernel(kernel, output_dir)

    api_pool = FmhaFwdSplitKVApiPool(mask_impl)
    for kernel in kernels:
        combine_ks = [
            k
            for k in combine_kernels
            if k.F_arch == kernel.F_arch
            and k.F_hdim == kernel.F_hdim
            and k.F_dtype == kernel.F_dtype
            and k.F_mode == kernel.F_mode
            and k.F_pipeline.F_spad == kernel.F_pipeline.F_spad
            and k.F_pipeline.F_dvpad == kernel.F_pipeline.F_dvpad
            and k.F_pipeline.F_lse == "f"
            and k.F_pipeline.F_squant == kernel.F_pipeline.F_squant
        ]
        assert len(combine_ks) == 1, (
            f"{len(combine_ks)} matching FmhaFwdSplitKVCombineKernel for {kernel}"
        )
        combine_kernel = combine_ks[0]
        api_pool.register_traits(
            FmhaFwdSplitKVApiTrait(
                arch=kernel.F_arch,
                pipeline_tag=kernel.F_pipeline.tag,
                hdim=kernel.F_hdim,
                dtype=kernel.F_dtype,
                mode=kernel.F_mode,
                bm0=kernel.F_tile.F_bm0,
                bn0=kernel.F_tile.F_bn0,
                bk0=kernel.F_tile.F_bk0,
                bn1=kernel.F_tile.F_bn1,
                bk1=kernel.F_tile.F_bk1,
                bk0max=kernel.F_tile.F_bk0max,
                vlayout=kernel.F_pipeline.F_vlayout,
                logits=kernel.F_pipeline.F_logits,
                mask=kernel.F_pipeline.F_mask,
                bias=kernel.F_pipeline.F_bias,
                lse=kernel.F_pipeline.F_lse,
                squant=kernel.F_pipeline.F_squant,
                pagedkv=kernel.F_pipeline.F_pagedkv,
                sink=kernel.F_pipeline.F_sink,
                spad=kernel.F_pipeline.F_spad,
                skpad=kernel.F_pipeline.F_skpad,
                dpad=kernel.F_pipeline.F_dpad,
                dvpad=kernel.F_pipeline.F_dvpad,
                bn1comb=combine_kernel.F_tile.F_bn1,
                occupancy=int(kernel.F_tile.F_occupancy),
                qscale=kernel.F_pipeline.F_qscale,
                # qr_tdm_pingpong's split tiles come from the forward tile table: its
                # constraints are the forward dispatch's (seqtune/skcheck decide here)
                constraint=CppConstraint()
                if kernel.F_pipeline.tag == "qr_tdm_pingpong"
                else kernel.F_tile.F_constraint,
            )
        )
    # behind pingpong's in its own buckets, the next bucket up's padded instances
    # (GFX125_SPLITKV_FALLBACK_BUCKETS): dispatched there, not compiled again. qr_tdm's take
    # what pingpong does not (paged KV, KV cache); pingpong's own come first where the bucket's
    # pingpong tile is narrower than the bucket (d192: its 192/128 tile, so d192/192 takes
    # d256's padded pingpong instances)
    narrow_pingpong = {h for h, h_v in GFX125_QR_TDM_PINGPONG_SPLITKV_HDIMS if h_v < h}
    for arch, by_dtype in api_pool.pool.items():
        if not arch.name.startswith("gfx125"):
            continue
        for by_hdim in by_dtype.values():
            for bucket, up in GFX125_SPLITKV_FALLBACK_BUCKETS.items():
                if bucket not in by_hdim or up not in by_hdim:
                    continue
                tags = (
                    ("qr_tdm_pingpong", "qr_tdm")
                    if bucket in narrow_pingpong
                    else ("qr_tdm",)
                )
                for tag in tags:
                    for t in list(by_hdim[up]):
                        if t.pipeline_tag == tag and (t.dpad, t.dvpad) == ("t", "t"):
                            api_pool.register_traits(
                                dataclasses.replace(t, bucket=bucket)
                            )
    write_fwd_splitkv_api(api_pool, output_dir)


def list_blobs(
    targets: List[str],
    file_path: Path,
    filter_list: str,
    receipt,
    optdim_list,
    mask_impl,
) -> None:
    filter_list = filter_list.split("@")
    filter_list.extend([""] * (2 - len(filter_list)))

    with file_path.open("a") as f:
        kernels = get_fwd_splitkv_combine_blobs(
            targets, filter_list[0], receipt, optdim_list
        )
        for kernel in kernels:
            f.write((file_path.parent / GEN_DIR / kernel.filename).as_posix() + "\n")
        kernels = get_fwd_splitkv_blobs(
            targets, filter_list[1], receipt, mask_impl, optdim_list
        )
        for kernel in kernels:
            f.write((file_path.parent / GEN_DIR / kernel.filename).as_posix() + "\n")
        f.write(
            (file_path.parent / GEN_DIR / FMHA_FWD_SPLITKV_API_FILENAME).as_posix()
            + "\n"
        )
