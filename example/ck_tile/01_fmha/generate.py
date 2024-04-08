# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
import itertools
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import copy
import fnmatch

DTYPE_MAP = {
    "fp16": "ck_tile::fp16_t",
    "bf16": "ck_tile::bf16_t",
    "fp8" : "ck_tile::fp8_t"
}

DTYPE_BITS = {
    "fp32": 32,
    "fp16": 16,
    "bf16": 16,
    "fp8" : 8,
    "bf8" : 8
}

MASK_MAP = {
    "no" : "FmhaMasks::NoMask",
    "causal" : "FmhaMasks::CausalMask",
    "generic" : "FmhaMasks::GenericMask"
}

MODE_MAP = {
    "batch" : "false",
    "group" : "true"
}

LAYOUT_MAP = {
    "row" : "true",
    "col" : "false"
}

PIPELINE_MAP = {
    "qr" : "ck_tile::BlockFmhaPipelineQRKSVS",
    "qr_fp8" : "ck_tile::BlockFmhaPipelineQRKSVSFp8",
    "qr_async" : "ck_tile::BlockFmhaPipelineQRKSVSAsync",
}

ELEMENT_FUNC_MAP = {
    "no" : "FmhaDefaultElementFunctions",
    "f8_static_quant" : "FmhaF8StaticQuantizationElementFunctions",
}

BOOL_MAP = {
    "t" : "true",
    "f" : "false"
}

MASKS = ["no", "causal", "generic"]
DIRECTIONS = ["fwd"]
GEN_DIR = ""    # in Cmake, have to generate files in same folder

FMHA_FWD_KERNEL_HEADER = """// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.\n
// auto generated by generate.py
#include "fmha_fwd.hpp"
"""

FMHA_FWD_KERNEL_BODY="""
using fmha_dtype_{F_idx} = {F_dtype};

using fmha_block_tile_{F_idx} = ck_tile::sequence<{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0blen}>;
using fmha_block_warps_{F_idx} = ck_tile::sequence<{F_rm}, {F_rn}, {F_rk}>;
using fmha_warp_tile_{F_idx} = ck_tile::sequence<{F_wm}, {F_wn}, {F_wk}>;

using fmha_shape_{F_idx} = ck_tile::TileFmhaShape<fmha_block_tile_{F_idx},
                                      fmha_block_warps_{F_idx},
                                      fmha_warp_tile_{F_idx},
                                      fmha_block_warps_{F_idx},
                                      fmha_warp_tile_{F_idx},
                                      {F_vlayout}>;

using fmha_trait_{F_idx} = ck_tile::TileFmhaTraits<{F_spad},
                                                    {F_skpad},
                                                    {F_dpad},
                                                    {F_dvpad},
                                                    {F_bias},
                                                    {F_lse},
                                                    {F_occupancy}>;
using fmha_mask_{F_idx} = {F_mask};

using fmha_element_function_{F_idx} = ck_tile::FmhaElementFunctions<
    // typename {F_element_func}::QElementFunction,
    // typename {F_element_func}::KElementFunction,
    // typename {F_element_func}::VElementFunction,
    // typename {F_element_func}::BiasElementFunction,
    // typename {F_element_func}::LSEElementFunction,
    // typename {F_element_func}::SAccElementFunction,
    typename {F_element_func}::PComputeElementFunction,
    typename {F_element_func}::OAccElementFunction>;

using fmha_fwd_args_{F_idx} = fmha_fwd_args<{F_element_func}>;

using fmha_pipeline_problem_{F_idx} = ck_tile::BlockFmhaPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::ODataType,
    fmha_element_function_{F_idx},
    fmha_shape_{F_idx},
    {F_mode},
    fmha_mask_{F_idx},
    fmha_trait_{F_idx}>;

using fmha_pipeline_{F_idx} = {F_pipeline}<
    fmha_pipeline_problem_{F_idx}>;

using fmha_epilogue_{F_idx} =
    ck_tile::Default2DEpilogue<ck_tile::Default2DEpilogueProblem<typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType,
                                           typename FmhaFwdTypeConfig<{F_dtype}>::ODataType,
                                           {F_spad}, {F_dvpad}>>;

using fmha_kernel_{F_idx} =
    ck_tile::FmhaFwdKernel<ck_tile::FmhaFwdTilePartitioner<fmha_shape_{F_idx}>,
                  fmha_pipeline_{F_idx},
                  fmha_epilogue_{F_idx}>;

using trait_{F_idx} = fmha_fwd_traits_<{F_hdim}, {F_dtype}, {F_mode},{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0blen}, {F_vlayout}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}>;

#include <iostream>

template<>
float fmha_fwd_<trait_{F_idx}, fmha_fwd_args_{F_idx}>(const ck_tile::stream_config& s, fmha_fwd_args_{F_idx} a)
{{
    using k_ = fmha_kernel_{F_idx};
    if(s.log_level_ > 0)
        std::cout << ", " << k_::GetName() << std::flush;
    auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks             = k_::BlockSize();
    constexpr ck_tile::index_t kBlockPerCu = k_::kBlockPerCu;
    return ck_tile::launch_kernel<blocks.x, kBlockPerCu>(s, k_{{}}, grids, blocks, 0, kargs);
}}
"""

FMHA_FWD_API_FILENAME="fmha_fwd_api.cpp"
FMHA_FWD_API="""
using fmha_fwd_args_ = fmha_fwd_args<{F_element_func}>;
template<>
float fmha_fwd<fmha_fwd_args_>(fmha_fwd_traits t, fmha_fwd_args_ a, const ck_tile::stream_config& s){{
    float r = -1;
{F_dispatch}
    return r;
}}
"""

FMHA_FWD_API_PER_DTYPE="""    {F_if}(t.data_type.compare(\"{F_dtype}\") == 0){{
{F_hdim_case}
    }}
"""
FMHA_FWD_API_PER_HDIM_CASE="""        {F_if} (t.hdim_q <= {F_hdim} && t.hdim_v <= {F_hdim}) {{
{F_inner_dispatch}
        }}
"""
MASK_CHECK_MAP = {
    "no" : "t.mask_type == mask_enum::no_mask",
    "causal" : "t.mask_type == mask_enum::causal_top_left || t.mask_type == mask_enum::causal_bottom_right",
    "generic" : "t.mask_type == mask_enum::window_generic",
}

FMHA_FWD_API_INNER_DISPATCH="""            {F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && ({F_mask_check}) && (t.has_bias == {F_bias}) && (t.has_lse == {F_lse}) &&
                        ({F_scheck}) && ({F_skcheck}) && ({F_dcheck}) && ({F_dvcheck})) {{
                using trait_ = fmha_fwd_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0blen}, {F_vlayout}, {F_mask}, {F_bias}, {F_lse}, {F_spad}, {F_skpad}, {F_dpad}, {F_dvpad}>;
                return fmha_fwd_<trait_>(s, a);
            }}
"""

@dataclass
class FmhaFwdApiTrait:
    pipeline_tag : str
    # sync with fmha_fwd_traits<>, to generate fallback calls
    hdim      : str
    dtype     : str  # data type
    mode      : str  # value from MODE_MAP
    bm0       : int  # tile size along q seqlen (block size)
    bn0       : int  # tile size along qk seqlen
    bk0       : int  # tile size along qk gemm unroll
    bn1       : int  # tile size along v head_dim
    bk1       : int  # tile size along kv gemm unroll
    bk0blen   : int
    vlayout   : str
    mask      : str
    bias      : str  # true/false
    lse       : str  #
    spad      : str
    skpad     : str
    dpad      : str
    dvpad     : str

    @property
    def name(self) -> str:
        return f'{self.hdim}-{self.dtype}-{self.mode}-{self.bm0}-{self.bn0}-{self.bk0}-{self.bn0}-{self.bk1}-{self.bk0blen}-'+\
                    f'{self.vlayout}-{self.mask}-{self.bias}-{self.lse}-{self.spad}-{self.skpad}-{self.dpad}-{self.dvpad}'

    @property
    def scheck(self) -> str:
        if self.pipeline_tag == 'qr_async':
            if self.spad == 't' : return 'true' # always support
            else :                return 'true'
        elif self.pipeline_tag in ['qr', 'qr_fp8']:
            if self.spad == 't' : return f'a.seqlen_q % {self.bm0} != 0'
            else :                return f'a.seqlen_q % {self.bm0} == 0'
        else: assert False

    @property
    def skcheck(self) -> str:
        if self.skpad == 't' : return f'a.seqlen_k % {self.bn0} != 0'
        else :                 return f'a.seqlen_k % {self.bn0} == 0'

    @property
    def dcheck(self) -> str:
        if self.pipeline_tag == 'qr_async':
            vec = int((32 * 4) / DTYPE_BITS[self.dtype])
            if self.dpad == 't': return f'a.hdim_q % {vec} == 0'
            else :               assert False
        elif self.pipeline_tag in ['qr', 'qr_fp8']:
            if self.dpad == 't': return f'a.hdim_q % {self.bk0blen} != 0'
            else :               return f'a.hdim_q % {self.bk0blen} == 0'
        else:   assert False

    @property
    def dvcheck(self) -> str:
        if self.pipeline_tag == 'qr_async':
            vec = int((32 * 4) / DTYPE_BITS[self.dtype])
            if self.dvpad == 't': return f'a.hdim_v % {vec} == 0'
            else :                assert False
        elif self.pipeline_tag in ['qr', 'qr_fp8']:
            if self.dvpad == 't': return f'a.hdim_v % {self.bk0blen} != 0'
            else :                return f'a.hdim_v % {self.bk0blen} == 0'
        else:   assert False

@dataclass
class FmhaFwdPipeline:
    tag : str

    F_vlayout   : str  # row/col
    F_spad      : str  # true/false
    F_skpad     : str  #
    F_dpad      : str  #
    F_dvpad     : str  #
    F_bias      : str  # true/false
    F_lse       : str  #
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
        if self.F_bias == 't' : n += '_bias'
        if self.F_mask != 'no' : n += f'_m{self.F_mask[0]}'
        if self.F_lse == 't' : n += '_lse'
        return n

class FmhaFwdApiPool:
    def __init__(self):
        self.pool = dict()

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
            element_func='no'
            per_hdim_case=str()
            for j, hdim in enumerate(self.pool[dtype].keys()):
                traits=self.pool[dtype][hdim]
                inners=str()
                for k, trait in enumerate(traits):
                    if_k = 'if' if k == 0 else 'else if'
                    inners = inners + FMHA_FWD_API_INNER_DISPATCH.format(F_if=if_k, F_mode=MODE_MAP[trait.mode], F_vlayout=LAYOUT_MAP[trait.vlayout], F_mask=MASK_MAP[trait.mask],
                                   F_mask_check=MASK_CHECK_MAP[trait.mask], F_bias=BOOL_MAP[trait.bias], F_lse=BOOL_MAP[trait.lse],
                                   F_scheck=trait.scheck, F_skcheck=trait.skcheck, F_dcheck=trait.dcheck, F_dvcheck=trait.dvcheck,
                                   F_spad=BOOL_MAP[trait.spad], F_skpad=BOOL_MAP[trait.skpad], F_dpad=BOOL_MAP[trait.dpad], F_dvpad=BOOL_MAP[trait.dvpad],
                                   F_bm0=trait.bm0, F_bn0=trait.bn0, F_bk0=trait.bk0, F_bn1=trait.bn1, F_bk1=trait.bk1, F_bk0blen=trait.bk0blen,
                                   F_hdim=hdim, F_dtype=DTYPE_MAP[dtype])
                if_j = 'if' if j == 0 else 'else if'
                per_hdim_case = per_hdim_case + FMHA_FWD_API_PER_HDIM_CASE.format(F_if=if_j, F_hdim=hdim, F_inner_dispatch=inners)
            if_i = 'if' if i == 0 else 'else if'
            per_dtypes = per_dtypes + FMHA_FWD_API_PER_DTYPE.format(F_if=if_i, F_dtype=dtype, F_hdim_case=per_hdim_case)
        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_API.format(F_element_func = ELEMENT_FUNC_MAP[element_func],
                                                            F_dispatch = per_dtypes)

@dataclass
class FmhaFwdTileSize:
    F_bm0       : int  # tile size along q seqlen (block size)
    F_bn0       : int  # tile size along qk seqlen
    F_bk0       : int  # tile size along qk gemm unroll
    F_bn1       : int  # tile size along v head_dim
    F_bk1       : int  # tile size along kv gemm unroll
    F_bk0blen   : int  # total length of K0, used for pipeline that need load Q at once (or repeately load Q as a whole tile)
    F_rm        : int  # number of warps along q seqlen (block warps)
    F_rn        : int  # number of warps along k seqlen(not used)
    F_rk        : int  # number of warps along gemm-k(not used)
    F_wm        : int  # warp size along m (warp size)
    F_wn        : int  # warp size along n
    F_wk        : int  # warp size along k
    F_occupancy : int  # occupancy, -1 will let pipeline decide the occupancy, other value will overwrite occupancy
    @property
    def name(self) -> str:
        return f"b{self.F_bm0}x{self.F_bn0}x{self.F_bk0}x{self.F_bn1}x{self.F_bk1}x{self.F_bk0blen}" +\
        f"_r{self.F_rm}x{self.F_rn}x{self.F_rk}_w{self.F_wm}x{self.F_wn}x{self.F_wk}" +\
            ("" if self.F_occupancy == -1 else f"_o{self.F_occupancy}")

@dataclass
class FmhaFwdKernel:
    direction       : str
    F_idx           : int  # this is not a tunable, but a counter to differentiate symbol
    F_hdim          : int  # hdim
    F_dtype         : str  # data type
    F_mode          : str  # value from MODE_MAP
    F_tile          : FmhaFwdTileSize
    F_pipeline      : FmhaFwdPipeline
    F_element_func  : str

    @property
    def template(self) -> str:
        return FMHA_FWD_KERNEL_HEADER + \
            FMHA_FWD_KERNEL_BODY.format(
                F_idx          = self.F_idx,
                F_hdim         = self.F_hdim,
                F_dtype        = DTYPE_MAP[self.F_dtype],
                F_bm0          = self.F_tile.F_bm0,
                F_bn0          = self.F_tile.F_bn0,
                F_bk0          = self.F_tile.F_bk0,
                F_bn1          = self.F_tile.F_bn1,
                F_bk1          = self.F_tile.F_bk1,
                F_bk0blen      = self.F_tile.F_bk0blen,
                F_rm           = self.F_tile.F_rm,
                F_rn           = self.F_tile.F_rn,
                F_rk           = self.F_tile.F_rk,
                F_wm           = self.F_tile.F_wm,
                F_wn           = self.F_tile.F_wn,
                F_wk           = self.F_tile.F_wk,
                F_vlayout      = LAYOUT_MAP[self.F_pipeline.F_vlayout],
                F_spad         = BOOL_MAP[self.F_pipeline.F_spad],
                F_skpad        = BOOL_MAP[self.F_pipeline.F_skpad],
                F_dpad         = BOOL_MAP[self.F_pipeline.F_dpad],
                F_dvpad        = BOOL_MAP[self.F_pipeline.F_dvpad],
                F_bias         = BOOL_MAP[self.F_pipeline.F_bias],
                F_lse          = BOOL_MAP[self.F_pipeline.F_lse],
                F_occupancy    = self.F_tile.F_occupancy ,
                F_mask         = MASK_MAP[self.F_pipeline.F_mask],
                F_mode         = MODE_MAP[self.F_mode],
                F_pipeline     = PIPELINE_MAP[self.F_pipeline.tag],
                F_element_func = ELEMENT_FUNC_MAP[self.F_element_func])

    @property
    def name(self) -> str:
        # TODO: we don't encode idx here
        return f"fmha_{self.direction}_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_" +\
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
                spad=self.F_pipeline.F_spad,
                skpad=self.F_pipeline.F_skpad,
                dpad=self.F_pipeline.F_dpad,
                dvpad=self.F_pipeline.F_dvpad)

# TODO: design a more practical way to do it
# this is current supported tile size per hdim
def get_fmha_fwd_tile_dict_from_dtype(direction : str, dtype : str) -> Optional[dict]:
    if direction == 'fwd':
        if dtype == 'fp16' or dtype == 'bf16':
            return {
                 '32'  : FmhaFwdTileSize(128, 64, 16, 32, 32, 32,     2, 1, 1, 32, 32, 16, -1),
                 '64'  : FmhaFwdTileSize(128, 64, 32, 64, 32, 64,     4, 1, 1, 32, 32, 16, -1),
                 '128' : FmhaFwdTileSize(128, 128, 32, 128, 32, 128,  4, 1, 1, 32, 32, 16, -1),
                 '256' : FmhaFwdTileSize(128, 128, 32, 256, 32, 256,  4, 1, 1, 32, 32, 16, -1),
            }
        elif dtype == 'fp8' or dtype == 'bf8':
            return {
                '128' : FmhaFwdTileSize(128, 128, 32, 128, 32, 128,  4, 1, 1, 32, 32, 32, -1)
            }
        else:
            return None
    else:
        return None

def get_blobs(kernel_filter : Optional[str]) -> Tuple[FmhaFwdApiPool, List[FmhaFwdKernel]]:
    # TODO: we don't support tuning yet, so pick up one value for vlayout/pipeline/pad
    #       support this in future
    def get_pipelines(dtype, hdim) -> List[FmhaFwdPipeline]:
        # this function will populate a list possible pipelines
        pipelines = []
        if dtype in ['fp16', 'bf16']:
            for mask, bias, lse in itertools.product(MASK_MAP.keys(), ["t", "f"], ["t", "f"]):
                if hdim == 256:
                # if True:
                    pipelines.append(FmhaFwdPipeline('qr', 'row', 'f', 'f', 'f', 'f', bias, lse, mask))
                    pipelines.append(FmhaFwdPipeline('qr', 'col', 'f', 'f', 'f', 'f', bias, lse, mask))

                    pipelines.append(FmhaFwdPipeline('qr', 'row', 't', 't', 't', 't', bias, lse, mask))
                    pipelines.append(FmhaFwdPipeline('qr', 'col', 't', 't', 't', 't', bias, lse, mask))
                else:
                    pipelines.append(FmhaFwdPipeline('qr_async', 'row', 't', 'f', 't', 't', bias, lse, mask))
                    pipelines.append(FmhaFwdPipeline('qr_async', 'row', 't', 't', 't', 't', bias, lse, mask))
                    pipelines.append(FmhaFwdPipeline('qr_async', 'col', 't', 'f', 't', 't', bias, lse, mask))
                    pipelines.append(FmhaFwdPipeline('qr_async', 'col', 't', 't', 't', 't', bias, lse, mask))
        elif dtype in ['fp8', 'bf8']:
            # no need lse kernels
            for mask, bias in itertools.product(MASK_MAP.keys(), ["t", "f"]):
                pipelines.append(FmhaFwdPipeline('qr_fp8', 'col', 'f', 'f', 'f', 'f', bias, 'f', mask))
        else:
            assert False
        return pipelines

    gen = list()
    api_pool = FmhaFwdApiPool()

    for direction, dtype in itertools.product(DIRECTIONS, DTYPE_MAP.keys()):
        d = get_fmha_fwd_tile_dict_from_dtype(direction, dtype)
        if d == None:
            continue
        #for hdim_str, mode, mask, bias, lse in itertools.product(d.keys(), MODE_MAP.keys(), MASK_MAP.keys(), ["t", "f"], ["t", "f"]):
        for hdim_str, mode in itertools.product(d.keys(), MODE_MAP.keys()):
            tile = d[hdim_str]
            hdim = int(hdim_str)
            for pipeline in get_pipelines(dtype, hdim):
                element_func='no'
                k = FmhaFwdKernel(direction=direction,
                                  F_idx=0,
                                  F_hdim=hdim,
                                  F_dtype=dtype,
                                  F_mode=mode,
                                  F_tile=tile,
                                  F_pipeline=pipeline,
                                  F_element_func=element_func)
                if kernel_filter != None:
                    if not fnmatch.fnmatch(k.name, kernel_filter):
                        continue
                api_pool.register_traits(k.api_trait())
                gen.append(k)

    return (api_pool, gen)

def write_single_kernel(kernel: FmhaFwdKernel, autogen_dir: Path) -> None:
    (autogen_dir / kernel.filename).write_text(kernel.template)

def write_api(api_pool : FmhaFwdApiPool, autogen_dir: Path) -> None:
    (autogen_dir / FMHA_FWD_API_FILENAME).write_text(api_pool.api)

def write_blobs(output_dir : Optional[str], kernel_filter : Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    api_pool, kernels = get_blobs(kernel_filter)
    for kernel in kernels:
        write_single_kernel(kernel, output_dir)
    write_api(api_pool, output_dir)

# list all the files that will be generated
def list_blobs(output_file : Optional[str], kernel_filter : Optional[str]) -> None:
    assert output_file is not None
    file_path = Path(output_file)
    with file_path.open('a') as f:
        _, kernels = get_blobs(kernel_filter)
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        f.write(str(file_path.parent / GEN_DIR / FMHA_FWD_API_FILENAME) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen api for CK fmha kernel",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory"
    )
    parser.add_argument(
        "-l",
        "--list_blobs",
        required=False,
        help="list all the kernels to a file"
    )
    # TODO: if using filter, must apply same value to output_dir and list_blobs
    parser.add_argument(
        "-f",
        "--filter",
        required=False,
        help="filter out kernels that need to generate, using fnmatch module"
    )
    args = parser.parse_args()
    if args.list_blobs is not None:
        list_blobs(args.list_blobs, args.filter)
    else:
        write_blobs(args.output_dir, args.filter)
