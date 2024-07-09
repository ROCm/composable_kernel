# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import copy
from dataclasses import dataclass
import fnmatch
import itertools
from pathlib import Path
from typing import List, Optional, Tuple, Union

from codegen.cmake_config import *
from codegen.cpp_symbol_map import *

from codegen.ops.fmha_fwd import (
    FmhaFwdTileSize,
    FmhaFwdApiTrait,
    FMHA_FWD_KERNEL_HEADER,
    FMHA_FWD_API_PER_DTYPE,
    FMHA_FWD_API_PER_HDIM_CASE,
)


FMHA_FWD_SPLITKV_PIPELINE_MAP = {
    "qr" : "ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVS",
    "qr_async" : "ck_tile::BlockFmhaFwdSplitKVPipelineQRKSVSAsync",
}

FMHA_FWD_SPLITKV_KERNEL_BODY="""
using fmha_dtype_{F_idx} = {F_dtype};
using fmha_mask_{F_idx} = {F_mask};

namespace {{
template <bool kHasUnevenSplits>
struct kernel_runner {{
using fmha_block_tile = ck_tile::sequence<{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0blen}>;
using fmha_block_warps = ck_tile::sequence<{F_rm}, {F_rn}, {F_rk}>;
using fmha_warp_tile = ck_tile::sequence<{F_wm}, {F_wn}, {F_wk}>;

using fmha_shape = ck_tile::TileFmhaShape<fmha_block_tile,
                                          fmha_block_warps,
                                          fmha_warp_tile,
                                          fmha_block_warps,
                                          fmha_warp_tile,
                                          {F_vlayout}>;

using fmha_trait = ck_tile::TileFmhaFwdSplitKVTraits<{F_spad},
                                                     {F_skpad},
                                                     {F_dpad},
                                                     {F_dvpad},
                                                     {F_bias},
                                                     false,
                                                     {F_lse},
                                                     {F_dropout},
                                                     {F_squant},
                                                     kHasUnevenSplits,
                                                     {F_occupancy}>;

using fmha_pipeline_problem = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::RandValOutputDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::OaccDataType,
    fmha_shape,
    {F_mode},
    fmha_mask_{F_idx},
    fmha_trait>;

using fmha_pipeline = {F_pipeline}<
    fmha_pipeline_problem>;

using fmha_epilogue =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType,
                                           typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType,
                                           {F_spad}, {F_dvpad}>>;

using fmha_kernel =
    ck_tile::FmhaFwdSplitKVKernel<ck_tile::FmhaFwdSplitKVTilePartitioner<fmha_shape>,
                  fmha_pipeline,
                  fmha_epilogue>;

static void run(const ck_tile::stream_config& s, fmha_fwd_args a)
{{
    using k_ = fmha_kernel;
    auto [kargs, grids] = fmha_fwd_splitkv_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks             = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs)(ck_tile::stream_config{{s.stream_id_}});
}}
}};
}}

using trait_{F_idx} = fmha_fwd_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0blen}, {F_vlayout},
                        {F_pipeline_enum}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}>;

#include <iostream>

template<>
void fmha_fwd_splitkv_oneshot_<trait_{F_idx}>(const ck_tile::stream_config& s, fmha_fwd_args a)
{{
    if constexpr({F_mode} == false) {{ // batch mode
        if (a.seqlen_k % (a.num_splits * {F_bn0}) == 0) {{
            kernel_runner<false>::run(s, a);
        }} else {{
            kernel_runner<true>::run(s, a);
        }}
    }} else {{
        kernel_runner<true>::run(s, a);
    }}
}}

template<>
std::string fmha_fwd_splitkv_get_name_<trait_{F_idx}>()
{{
    using k_ = kernel_runner<true>::fmha_kernel; /// FIXME: choose real kernel type
    return k_::GetName();
}}
"""

FMHA_FWD_SPLITKV_COMBINE_KERNEL_BODY="""
using fmha_dtype_{F_idx} = {F_dtype};

namespace {{
template <ck_tile::index_t kLogMaxSplits>
struct kernel_runner {{
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
    {F_bm0}, 
    {F_bn1},
    {F_mode},
    fmha_trait>;

using fmha_pipeline = ck_tile::BlockFmhaFwdSplitKVCombinePipeline<
    fmha_pipeline_problem>;

using fmha_epilogue =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType,
                                           typename FmhaFwdTypeConfig<{F_dtype}>::ODataType,
                                           {F_spad}, {F_dvpad}>>;

using fmha_kernel =
    ck_tile::FmhaFwdSplitKVCombineKernel<ck_tile::FmhaFwdSplitKVCombineTilePartitioner<{F_bm0}, {F_bn1}>,
                  fmha_pipeline,
                  fmha_epilogue>;

static void run(const ck_tile::stream_config& s, fmha_fwd_args a)
{{
    using k_ = fmha_kernel;
    auto [kargs, grids] = fmha_fwd_splitkv_combine_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks             = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    ck_tile::make_kernel<blocks.x, kBlockPerCu>(k_{{}}, grids, blocks, 0, kargs)(ck_tile::stream_config{{s.stream_id_}});
}}
}};
}}

using trait_{F_idx} = fmha_fwd_splitkv_combine_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn1},
                        {F_lse}, {F_squant}, {F_spad}, {F_dvpad}>;

#include <iostream>

template<>
void fmha_fwd_splitkv_combine_oneshot_<trait_{F_idx}>(const ck_tile::stream_config& s, fmha_fwd_args a)
{{
    if (a.num_splits <= 16) {{
        kernel_runner<4>::run(s, a);
    }} else if (a.num_splits <= 32) {{
        kernel_runner<5>::run(s, a);
    }} else if (a.num_splits <= 64) {{
        kernel_runner<6>::run(s, a);
    }} else if (a.num_splits <= 128) {{
        kernel_runner<7>::run(s, a);
    }}
}}

template<>
std::string fmha_fwd_splitkv_combine_get_name_<trait_{F_idx}>()
{{
    using k_ = kernel_runner<6>::fmha_kernel; /// FIXME: choose real kernel type
    return k_::GetName();
}}
"""

FMHA_FWD_SPLITKV_API_FILENAME="fmha_fwd_splitkv_api.cpp"
FMHA_FWD_SPLITKV_API="""
#include <iostream>

template<typename fmha_fwd_splitkv_traits_, typename fmha_fwd_splitkv_combine_traits_>
float fmha_fwd_splitkv_(const ck_tile::stream_config& s, fmha_fwd_args a)
{{
    if(s.log_level_ > 0)
    std::cout
    << ", " << fmha_fwd_splitkv_get_name_<fmha_fwd_splitkv_traits_>()
    << ", " << fmha_fwd_splitkv_combine_get_name_<fmha_fwd_splitkv_combine_traits_>() 
    << std::flush;

    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){{ fmha_fwd_splitkv_oneshot_<fmha_fwd_splitkv_traits_>(s_, a); }}, 
        [=](const ck_tile::stream_config& s_){{ fmha_fwd_splitkv_combine_oneshot_<fmha_fwd_splitkv_combine_traits_>(s_, a); }}
    );
}}

float fmha_fwd_splitkv(fmha_fwd_traits t, fmha_fwd_args a, const ck_tile::stream_config& s){{
    float r = -1;
{F_dispatch}
    return r;
}}
"""

FMHA_FWD_SPLITKV_API_INNER_DISPATCH="""            {F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && ({F_mask_check}) && (t.bias_type == {F_bias_check}) && (t.has_lse == {F_lse})  && (t.has_dropout == {F_dropout}) && (t.do_fp8_static_quant == {F_squant}) &&
                        ({F_scheck}) && ({F_skcheck}) && ({F_dcheck}) && ({F_dvcheck})) {{
                using traits_ = fmha_fwd_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0blen}, {F_vlayout}, {F_pipeline_enum}, {F_mask}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}>;
                using traits2_ = fmha_fwd_splitkv_combine_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}/2, {F_bn1}, {F_lse}, {F_squant}, {F_spad}, {F_dvpad}>;

                return fmha_fwd_splitkv_<traits_, traits2_>(s, a);
            }}
"""

@dataclass
class FmhaFwdSplitKVPipeline:
    tag : str

    F_vlayout   : str  # row/col
    F_spad      : str  # true/false
    F_skpad     : str  #
    F_dpad      : str  #
    F_dvpad     : str  #
    F_bias      : str  # true/false
    F_lse       : str  #
    F_dropout   : str  #
    F_squant    : str  #
    F_mask      : str  # value from MASK_MAP

    @property
    def name(self) -> str:
        def pad_name() -> str:
            n = ''
            if self.F_spad == 't': n += 's'
            if self.F_skpad == 't' : n += 'sk'
            if self.F_dpad == 't' : n += 'd'
            if self.F_dvpad == 't' : n += 'dv'
            if n != '' : n = 'p' + n
            return n
        pn = pad_name()
        n = f'{self.tag}_v{self.F_vlayout[0]}'
        if pn != '' : n += f'_{pn}'
        if self.F_bias != 'no' : n += f'_{self.F_bias}'
        if self.F_mask[0:2] == 's_':
            if self.F_mask == 's_mask': n += f'_mask'
        else:
            if self.F_mask != 'no' : n += f'_m{self.F_mask[0]}'
        if self.F_lse == 't' : n += '_lse'
        if self.F_dropout == 't' : n += '_dropout'
        if self.F_squant == 't' : n += '_squant'
        return n

@dataclass
class FmhaFwdSplitKVCombinePipeline:
    tag : str

    F_spad      : str  # true/false
    F_dvpad     : str  #
    F_lse       : str  #
    F_squant    : str  #

    @property
    def name(self) -> str:
        def pad_name() -> str:
            n = ''
            if self.F_spad == 't': n += 's'
            if self.F_dvpad == 't' : n += 'dv'
            if n != '' : n = 'p' + n
            return n
        pn = pad_name()
        n = f'{self.tag}'
        if pn != '' : n += f'_{pn}'
        if self.F_lse == 't' : n += '_lse'
        if self.F_squant == 't' : n += '_squant'
        return n

class FmhaFwdSplitKVApiPool:
    def __init__(self, mask_impl):
        self.pool = dict()
        self.mask_impl = mask_impl

    def register_traits(self, trait : FmhaFwdApiTrait) -> None:
        # TODO: do we need to check duplication?
        if trait.dtype not in self.pool.keys():
            self.pool[trait.dtype] = dict()
        if trait.hdim not in self.pool[trait.dtype].keys():
            self.pool[trait.dtype][trait.hdim] = list()

        self.pool[trait.dtype][trait.hdim].append(copy.copy(trait))

    @property
    def api(self) -> str:
        per_dtypes=str()
        for i, dtype in enumerate(self.pool.keys()):
            per_hdim_case=str()
            for j, hdim in enumerate(self.pool[dtype].keys()):
                traits=self.pool[dtype][hdim]
                inners=str()
                for k, trait in enumerate(traits):
                    if_k = 'if' if k == 0 else 'else if'
                    inners = inners + FMHA_FWD_SPLITKV_API_INNER_DISPATCH.format(F_if=if_k, F_mode=MODE_MAP[trait.mode], F_vlayout=LAYOUT_MAP[trait.vlayout],
                                   F_pipeline_enum=PIPELINE_ENUM_MAP[trait.pipeline_tag], F_mask=get_mask_map(self.mask_impl)[trait.mask],
                                   F_mask_check=get_mask_check_map(self.mask_impl)[trait.mask], F_bias_check=BIAS_CHECK_MAP[trait.bias], F_bias=BIAS_MAP[trait.bias],
                                   F_lse=BOOL_MAP[trait.lse], F_dropout=BOOL_MAP[trait.dropout] ,
                                   F_squant=BOOL_MAP[trait.squant], F_scheck=trait.scheck, F_skcheck=trait.skcheck, F_dcheck=trait.dcheck, F_dvcheck=trait.dvcheck,
                                   F_spad=BOOL_MAP[trait.spad], F_skpad=BOOL_MAP[trait.skpad], F_dpad=BOOL_MAP[trait.dpad], F_dvpad=BOOL_MAP[trait.dvpad],
                                   F_bm0=trait.bm0, F_bn0=trait.bn0, F_bk0=trait.bk0, F_bn1=trait.bn1, F_bk1=trait.bk1, F_bk0blen=trait.bk0blen,
                                   F_hdim=hdim, F_dtype=DTYPE_MAP[dtype])
                if_j = 'if' if j == 0 else 'else if'
                per_hdim_case = per_hdim_case + FMHA_FWD_API_PER_HDIM_CASE.format(F_if=if_j, F_hdim=hdim, F_inner_dispatch=inners)
            if_i = 'if' if i == 0 else 'else if'
            per_dtypes = per_dtypes + FMHA_FWD_API_PER_DTYPE.format(F_if=if_i, F_dtype=dtype, F_hdim_case=per_hdim_case)
        if not per_dtypes:
            # empty string we add some ignore to suppress warning in api
            per_dtypes += '    (void)t ; (void)s ; (void)a;'
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_SPLITKV_API.format(F_dispatch = per_dtypes)

@dataclass
class FmhaFwdSplitKVCombineTileSize:
    F_bm0       : int  # tile size along q seqlen
    F_bn1       : int  # tile size along v head_dim
    F_occupancy : int  # occupancy, -1 will let pipeline decide the occupancy, other value will overwrite occupancy
    @property
    def name(self) -> str:
        return f"b{self.F_bm0}x{self.F_bn1}" +\
            ("" if self.F_occupancy == -1 else f"_o{self.F_occupancy}")

@dataclass
class FmhaFwdSplitKVKernel:
    F_idx           : int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim          : int  # hdim
    F_dtype         : str  # data type
    F_mode          : str  # value from MODE_MAP
    F_tile          : FmhaFwdTileSize
    F_pipeline      : FmhaFwdSplitKVPipeline
    mask_impl       : str

    @property
    def template(self) -> str:
        kernel_body = str()
        return FMHA_FWD_KERNEL_HEADER + \
            FMHA_FWD_SPLITKV_KERNEL_BODY.format(
                F_idx           = self.F_idx,
                F_hdim          = self.F_hdim,
                F_dtype         = DTYPE_MAP[self.F_dtype],
                F_bm0           = self.F_tile.F_bm0,
                F_bn0           = self.F_tile.F_bn0,
                F_bk0           = self.F_tile.F_bk0,
                F_bn1           = self.F_tile.F_bn1,
                F_bk1           = self.F_tile.F_bk1,
                F_bk0blen       = self.F_tile.F_bk0blen,
                F_rm            = self.F_tile.F_rm,
                F_rn            = self.F_tile.F_rn,
                F_rk            = self.F_tile.F_rk,
                F_wm            = self.F_tile.F_wm,
                F_wn            = self.F_tile.F_wn,
                F_wk            = self.F_tile.F_wk,
                F_vlayout       = LAYOUT_MAP[self.F_pipeline.F_vlayout],
                F_spad          = BOOL_MAP[self.F_pipeline.F_spad],
                F_skpad         = BOOL_MAP[self.F_pipeline.F_skpad],
                F_dpad          = BOOL_MAP[self.F_pipeline.F_dpad],
                F_dvpad         = BOOL_MAP[self.F_pipeline.F_dvpad],    
                F_bias          = BIAS_MAP[self.F_pipeline.F_bias],
                F_lse           = BOOL_MAP[self.F_pipeline.F_lse],
                F_dropout       = BOOL_MAP[self.F_pipeline.F_dropout],
                F_squant        = BOOL_MAP[self.F_pipeline.F_squant],
                F_occupancy     = self.F_tile.F_occupancy,
                F_pipeline_enum = PIPELINE_ENUM_MAP[self.F_pipeline.tag],
                F_mask          = get_mask_map(self.mask_impl)[self.F_pipeline.F_mask],
                F_mode          = MODE_MAP[self.F_mode],
                F_pipeline      = FMHA_FWD_SPLITKV_PIPELINE_MAP[self.F_pipeline.tag])

    @property
    def name(self) -> str:
        # TODO: we don't encode idx here
        return f"fmha_fwd_splitkv_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_" + \
                self.F_tile.name + '_' + self.F_pipeline.name

    @property
    def filename(self) -> str:
        return self.name + ".cpp"

    def api_trait(self) -> FmhaFwdApiTrait:
        return FmhaFwdApiTrait(
                pipeline_tag=self.F_pipeline.tag,
                hdim=str(self.F_hdim),
                dtype=self.F_dtype,
                mode=self.F_mode,
                bm0=self.F_tile.F_bm0,
                bn0=self.F_tile.F_bn0,
                bk0=self.F_tile.F_bk0,
                bn1=self.F_tile.F_bn1,
                bk1=self.F_tile.F_bk1,
                bk0blen=self.F_tile.F_bk0blen,
                vlayout=self.F_pipeline.F_vlayout,
                mask=self.F_pipeline.F_mask,
                bias=self.F_pipeline.F_bias,
                lse=self.F_pipeline.F_lse,
                dropout=self.F_pipeline.F_dropout,
                squant=self.F_pipeline.F_squant,
                spad=self.F_pipeline.F_spad,
                skpad=self.F_pipeline.F_skpad,
                dpad=self.F_pipeline.F_dpad,
                dvpad=self.F_pipeline.F_dvpad)

@dataclass
class FmhaFwdSplitKVCombineKernel:
    F_idx           : int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim          : int  # hdim
    F_dtype         : str  # data type
    F_mode          : str  # value from MODE_MAP
    F_tile          : FmhaFwdSplitKVCombineTileSize
    F_pipeline      : FmhaFwdSplitKVCombinePipeline

    @property
    def template(self) -> str:
        kernel_body = str()
        return FMHA_FWD_KERNEL_HEADER + \
            FMHA_FWD_SPLITKV_COMBINE_KERNEL_BODY.format(
                F_idx           = self.F_idx,
                F_hdim          = self.F_hdim,
                F_dtype         = DTYPE_MAP[self.F_dtype],
                F_bm0           = self.F_tile.F_bm0,
                F_bn1           = self.F_tile.F_bn1,
                F_spad          = BOOL_MAP[self.F_pipeline.F_spad],
                F_dvpad         = BOOL_MAP[self.F_pipeline.F_dvpad],
                F_lse           = BOOL_MAP[self.F_pipeline.F_lse],
                F_squant        = BOOL_MAP[self.F_pipeline.F_squant],
                F_occupancy     = self.F_tile.F_occupancy,
                F_mode          = MODE_MAP[self.F_mode])

    @property
    def name(self) -> str:
        # TODO: we don't encode idx here
        return f"fmha_fwd_splitkv_combine_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_" + \
                self.F_tile.name + '_' + self.F_pipeline.name

    @property
    def filename(self) -> str:
        return self.name + ".cpp"

    def api_trait(self) -> FmhaFwdApiTrait:
        return FmhaFwdApiTrait(
                pipeline_tag=self.F_pipeline.tag,
                hdim=str(self.F_hdim),
                dtype=self.F_dtype,
                mode=self.F_mode,
                bm0=self.F_tile.F_bm0,
                bn0=self.F_tile.F_bn0,
                bk0=self.F_tile.F_bk0,
                bn1=self.F_tile.F_bn1,
                bk1=self.F_tile.F_bk1,
                bk0blen=self.F_tile.F_bk0blen,
                vlayout=self.F_pipeline.F_vlayout,
                mask=self.F_pipeline.F_mask,
                bias=self.F_pipeline.F_bias,
                lse=self.F_pipeline.F_lse,
                dropout=self.F_pipeline.F_dropout,
                squant=self.F_pipeline.F_squant,
                spad=self.F_pipeline.F_spad,
                skpad=self.F_pipeline.F_skpad,
                dpad=self.F_pipeline.F_dpad,
                dvpad=self.F_pipeline.F_dvpad)

# TODO: design a more practical way to do it
# this is current supported tile size per hdim
def get_fmha_fwd_tile_dict_from_dtype(dtype : str) -> Optional[dict]:
    if dtype == 'fp16' or dtype == 'bf16':
        return {
                '32'  : FmhaFwdTileSize(128, 64, 16, 32, 32, 32,     2, 1, 1, 32, 32, 16, -1),
                '64'  : FmhaFwdTileSize(128, 64, 32, 64, 32, 64,     4, 1, 1, 32, 32, 16, -1),
                '128' : FmhaFwdTileSize(128, 128, 32, 128, 32, 128,  4, 1, 1, 32, 32, 16, -1),
                '256' : FmhaFwdTileSize(128, 128, 32, 256, 32, 256,  4, 1, 1, 32, 32, 16, -1),
        }
    elif dtype == 'fp8' or dtype == 'bf8':
        return {
            '64'  : FmhaFwdTileSize(128, 64, 32, 64, 32, 64,     2, 1, 1, 32, 32, 32, -1),
            '128' : FmhaFwdTileSize(128, 128, 32, 128, 32, 128,  4, 1, 1, 32, 32, 32, -1),
            '256' : FmhaFwdTileSize(128, 128, 32, 256, 32, 256,  4, 1, 1, 32, 32, 32, -1)
        }
    else:
        return None

def get_fmha_fwd_splitkv_combine_tile_dict_from_dtype(dtype : str) -> Optional[dict]:
    if dtype == 'fp16' or dtype == 'bf16':
        return {
                '32'  : FmhaFwdSplitKVCombineTileSize(64, 32, -1),
                '64'  : FmhaFwdSplitKVCombineTileSize(64, 64, -1),
                '128' : FmhaFwdSplitKVCombineTileSize(64, 128, -1),
                '256' : FmhaFwdSplitKVCombineTileSize(64, 256, -1),
    }
    elif dtype == 'fp8' or dtype == 'bf8':
        return {
                '64'  : FmhaFwdSplitKVCombineTileSize(64, 64, -1),
                '128' : FmhaFwdSplitKVCombineTileSize(64, 128, -1),
                '256' : FmhaFwdSplitKVCombineTileSize(64, 256, -1),
        }
    else:
        return None

def get_fwd_splitkv_blobs(kernel_filter : Optional[str], receipt, mask_impl) -> Tuple[FmhaFwdSplitKVApiPool, List[FmhaFwdSplitKVKernel]]:
    Pipeline = FmhaFwdSplitKVPipeline
    Kernel = FmhaFwdSplitKVKernel

    # TODO: we don't support tuning yet, so pick up one value for vlayout/pipeline/pad
    #       support this in future
    def get_pipelines(dtype, hdim) -> List[FmhaFwdSplitKVPipeline]:
        # this function will populate a list possible pipelines
        # TODO: the order of List matters! the later in this list will be also be checked later
        # TODO: currently for qr pipeline, let 't' padding to appear later!!
        # TODO: how to design this more generic?
        squant = 't' if dtype == 'fp8' else 'f'
        pipelines = []
        if dtype in ['fp16', 'bf16']:
            # splitkv kernel donot support dropout
            for mask, bias, lse, dropout in itertools.product(get_mask_map(mask_impl).keys(), BIAS_MAP.keys(), ["t", "f"], ["f"]):
                if hdim == 256:
                # if True:
                    pipelines.append(Pipeline('qr', 'row', 'f', 'f', 'f', 'f', bias, lse, dropout, squant, mask))
                    pipelines.append(Pipeline('qr', 'col', 'f', 'f', 'f', 'f', bias, lse, dropout, squant, mask))

                    pipelines.append(Pipeline('qr', 'row', 't', 't', 't', 't', bias, lse, dropout, squant, mask))
                    pipelines.append(Pipeline('qr', 'col', 't', 't', 't', 't', bias, lse, dropout, squant, mask))
                else:
                    pipelines.append(Pipeline('qr_async', 'row', 't', 'f', 't', 't', bias, lse, dropout, squant, mask))
                    pipelines.append(Pipeline('qr_async', 'row', 't', 't', 't', 't', bias, lse, dropout, squant, mask))
                    pipelines.append(Pipeline('qr_async', 'col', 't', 'f', 't', 't', bias, lse, dropout, squant, mask))
                    pipelines.append(Pipeline('qr_async', 'col', 't', 't', 't', 't', bias, lse, dropout, squant, mask))
                    if receipt == 1:
                        pipelines.append(Pipeline('qr', 'row', 't', 't', 't', 't', bias, lse, dropout, squant, mask)) # TODO: cover arbitraty hdim
                        pipelines.append(Pipeline('qr', 'col', 't', 'f', 't', 't', bias, lse, dropout, squant, mask)) # TODO: cover arbitraty hdim
        elif dtype in ['fp8', 'bf8']:
            # no need lse/dropout kernels
            for mask, bias in itertools.product(get_mask_map(mask_impl).keys(), BIAS_MAP.keys()):
                pipelines.append(Pipeline('qr', 'col', 'f', 'f', 'f', 'f', bias, 'f', 'f', squant, mask))
        else:
            assert False
        return pipelines

    gen = list()
    api_pool = FmhaFwdSplitKVApiPool(mask_impl)

    for dtype in DTYPE_MAP.keys():
        d = get_fmha_fwd_tile_dict_from_dtype(dtype)
        if d == None:
            continue
        #for hdim_str, mode, mask, bias, lse in itertools.product(d.keys(), MODE_MAP.keys(), MASK_MAP.keys(), ["t", "f"], ["t", "f"]):
        for hdim_str, mode in itertools.product(d.keys(), MODE_MAP.keys()):
            tile = d[hdim_str]
            hdim = int(hdim_str)
            for pipeline in get_pipelines(dtype, hdim):
                if mode == "group":
                    if pipeline.F_spad != 't' or pipeline.F_skpad != 't':
                        # in group mode, spad/skpad must be true, since we can't predict if seqlen of current batch need pad or not
                        continue
                k = Kernel(F_idx=0,
                           F_hdim=hdim,
                           F_dtype=dtype,
                           F_mode=mode,
                           F_tile=tile,
                           F_pipeline=pipeline,
                           mask_impl=mask_impl)
                if kernel_filter != None:
                    if not fnmatch.fnmatch(k.name, kernel_filter):
                        continue
                if receipt == 2:
                    cond = dtype in ['fp16', 'bf16']
                    cond &= pipeline.F_vlayout == 'row'
                    cond &= pipeline.F_bias in ['no', 'alibi']
                    cond &= pipeline.F_squant == 'f'
                    if not cond:
                        continue
                api_pool.register_traits(k.api_trait())
                gen.append(k)

    return (api_pool, gen)

def get_fwd_splitkv_combine_blobs(kernel_filter : Optional[str], receipt) -> List[FmhaFwdSplitKVCombineKernel]:
    Pipeline = FmhaFwdSplitKVCombinePipeline
    Kernel = FmhaFwdSplitKVCombineKernel

    # TODO: we don't support tuning yet, so pick up one value for vlayout/pipeline/pad
    #       support this in future
    def get_pipelines(dtype, hdim) -> List[FmhaFwdSplitKVCombinePipeline]:
        # this function will populate a list possible pipelines
        # TODO: the order of List matters! the later in this list will be also be checked later
        # TODO: currently for qr pipeline, let 't' padding to appear later!!
        # TODO: how to design this more generic?
        squant = 't' if dtype == 'fp8' else 'f'
        pipelines = []
        if dtype in ['fp16', 'bf16']:
            for spad, dvpad, lse in itertools.product(["t", "f"], ["t", "f"], ["t", "f"]):
                pipelines.append(Pipeline('unused', spad, dvpad, lse, squant))
        elif dtype in ['fp8', 'bf8']:
            # no need lse kernels
            pipelines.append(Pipeline('unused', 'f', 'f', 'f', squant))
        else:
            assert False
        return pipelines

    gen = list()

    for dtype in DTYPE_MAP.keys():
        d = get_fmha_fwd_splitkv_combine_tile_dict_from_dtype(dtype)
        if d == None:
            continue
        #for hdim_str, mode, mask, bias, lse in itertools.product(d.keys(), MODE_MAP.keys(), MASK_MAP.keys(), ["t", "f"], ["t", "f"]):
        for hdim_str, mode in itertools.product(d.keys(), MODE_MAP.keys()):
            tile = d[hdim_str]
            hdim = int(hdim_str)
            for pipeline in get_pipelines(dtype, hdim):
                if mode == "group":
                    if pipeline.F_spad != 't':
                        # in group mode, spad/skpad must be true, since we can't predict if seqlen of current batch need pad or not
                        continue
                k = Kernel(F_idx=0,
                           F_hdim=hdim,
                           F_dtype=dtype,
                           F_mode=mode,
                           F_tile=tile,
                           F_pipeline=pipeline)
                if kernel_filter != None:
                    if not fnmatch.fnmatch(k.name, kernel_filter):
                        continue
                gen.append(k)

    return gen

def write_single_kernel(kernel: Union[FmhaFwdSplitKVKernel, FmhaFwdSplitKVCombineKernel], autogen_dir: Path) -> None:
    (autogen_dir / kernel.filename).write_text(kernel.template)

def write_fwd_splitkv_api(api_pool : FmhaFwdSplitKVApiPool, autogen_dir: Path) -> None:
    file_path = autogen_dir / FMHA_FWD_SPLITKV_API_FILENAME
    file_path.write_text(api_pool.api)

def write_blobs(output_dir : Path, kernel_filter : Optional[str], receipt, mask_impl) -> None:
    kernels = get_fwd_splitkv_combine_blobs(kernel_filter, receipt)
    for kernel in kernels:
        write_single_kernel(kernel, output_dir)
    api_pool, kernels = get_fwd_splitkv_blobs(kernel_filter, receipt, mask_impl)
    for kernel in kernels:
        write_single_kernel(kernel, output_dir)
    write_fwd_splitkv_api(api_pool, output_dir)

def list_blobs(file_path : Path, kernel_filter : Optional[str], receipt, mask_impl) -> None:
    with file_path.open('a') as f:
        kernels = get_fwd_splitkv_combine_blobs(kernel_filter, receipt)
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        _, kernels = get_fwd_splitkv_blobs(kernel_filter, receipt, mask_impl)
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        f.write(str(file_path.parent / GEN_DIR / FMHA_FWD_SPLITKV_API_FILENAME) + "\n")