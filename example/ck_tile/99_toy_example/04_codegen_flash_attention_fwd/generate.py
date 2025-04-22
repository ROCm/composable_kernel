# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
from enum import IntEnum
from pathlib import Path
import sys
from typing import List, Optional, Any
import functools
import itertools
import copy
from dataclasses import dataclass

# def get_if_str(idx, total, last_else=True):
#     if idx == 0:
#         return 'if'
#     elif idx < total - 1:
#         return 'else if'
#     else:
#         return 'else' if last_else else 'else if'

def get_if_str(size_, total, last_else=True):
    if size_ == "small":
        return 'if'
    else:
        return 'else if'

DATA_TYPE_MAP = {'fp32': 'float',
                 'fp16': 'ck_tile::half_t',
                 'bf16': 'ck_tile::bf16_t'}

def BOOL_MAP(b_) -> str:
    return 'true' if b_ else 'false'

class FlashAttentionFwdCodegen:
    API_TRAITS_DEFINE = """

template <typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename PDataType_,
          typename OaccDataType_,
          index_t kBlockSize_,
          index_t kHeadDim_,
          index_t kM0PerBlock_,
          index_t kN0PerBlock_,
          index_t kK0PerBlock_,
          index_t kN1PerBlock_,
          index_t kK1PerBlock_>
struct flash_attention_fwd_traits_
{
    using SaccDataType = ck_tile::remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType = ck_tile::remove_cvref_t<SMPLComputeDataType_>;
    using PDataType = ck_tile::remove_cvref_t<PDataType_>;
    using OaccDataType = ck_tile::remove_cvref_t<OaccDataType_>;

    static constexpr index_t kBlockSize  = kBlockSize_;
    static constexpr index_t kHeadDim    = kHeadDim_;
    static constexpr index_t kM0PerBlock = kM0PerBlock_;
    static constexpr index_t kN0PerBlock = kN0PerBlock_;
    static constexpr index_t kK0PerBlock = kK0PerBlock_;
    static constexpr index_t kN1PerBlock = kN1PerBlock_;
    static constexpr index_t kK1PerBlock = kK1PerBlock_;

    static constexpr ck_tile::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    static constexpr ck_tile::index_t kWarpPerBlock = kBlockSize / warpSize;
    static constexpr ck_tile::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;
};

template <typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          ck_tile::index_t kBlockSize,
          ck_tile::index_t kHeadDim,
          ck_tile::index_t kM0PerBlock,
          ck_tile::index_t kN0PerBlock,
          ck_tile::index_t kK0PerBlock,
          ck_tile::index_t kN1PerBlock,
          ck_tile::index_t kK1PerBlock>
using traits_ = flash_attention_fwd_traits_<SaccDataType,
                                          SMPLComputeDataType,
                                          PDataType,
                                          OaccDataType,
                                          kBlockSize,
                                          kHeadDim,
                                          kM0PerBlock,
                                          kN0PerBlock,
                                          kK0PerBlock,
                                          kN1PerBlock,
                                          kK1PerBlock>;
"""

#     API_COMMON_HEADER = """
# // SPDX-License-Identifier: MIT
# // Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

# #include <ck_tile/core.hpp>
# #include "flash_attention_fwd.hpp"
# #include <iostream>

# #pragma once

# using S = ck_tile::stream_config;
# using A = FlashAttnArgs;

# {F_traits_define}

# template <typename QDataType,
#           typename KDataType,
#           typename VDataType,
#           typename ODataType,
#           typename Traits_>
# float flash_attention_fwd_(const FlashAttnArgs<QDataType, KDataType, VDataType, ODataType>& a,
#                           const ck_tile::stream_config& stream_config) {{
#     using SaccDataType        = typename Traits_::SaccDataType;
#     using SMPLComputeDataType = typename Traits_::SMPLComputeDataType;
#     using PDataType           = typename Traits_::PDataType;
#     using OaccDataType        = typename Traits_::OaccDataType;

#     index_t kGridSize = a.Batch * (a.M0 / Traits_::kM0PerBlock) * (a.N1 / Traits_::kN1PerBlock);

#     if(stream_config.log_level_ > 0)
#         std::cout << ", " << "FlashAttentionFwd<" << Traits_::kBlockSize << "," << Traits_::kHeadDim << ">" << std::flush;

#     return ck_tile::launch_kernel(stream_config,
#         ck_tile::make_kernel<Traits_::kBlockSize, Traits_::kBlockPerCu>(
#         ck_tile::FlashAttentionFwd<QDataType,
#                                    KDataType,
#                                    VDataType,
#                                    SaccDataType,
#                                    SMPLComputeDataType,
#                                    PDataType,
#                                    OaccDataType,
#                                    ODataType,
#                                    Traits_::kBlockSize,
#                                    Traits_::kHeadDim,
#                                    Traits_::kM0PerBlock,
#                                    Traits_::kN0PerBlock,
#                                    Traits_::kK0PerBlock,
#                                    Traits_::kN1PerBlock,
#                                    Traits_::kK1PerBlock>{{}},
#         kGridSize,
#         Traits_::kBlockSize,
#         0,
#         a.q_ptr,
#         a.k_ptr,
#         a.v_ptr,
#         a.o_ptr,
#         a.M0,
#         a.N0,
#         a.K0,
#         a.N1,
#         a.Batch,
#         a.strideQ,        // StrideQ
#         a.strideK,        // StrideK
#         a.strideV,        // StrideV
#         a.strideO,        // StrideO
#         a.batchStrideQ,   // BatchStrideQ
#         a.batchStrideK,   // BatchStrideK
#         a.batchStrideV,   // BatchStrideV
#         a.batchStrideO)); // BatchStrideO
# }}
# """

    API_BASE = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck_tile/core.hpp>
#include "flash_attention_fwd.hpp"

namespace ck_tile {{

{F_traits_define}

// Note: this internal API only declare, not define here, otherwise will block `make -j`
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename Traits_>
float flash_attention_fwd_(const FlashAttnArgs<QDataType, KDataType, VDataType, ODataType>& a, 
                          const ck_tile::stream_config& stream_config);

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          typename ODataType>
float flash_attention_fwd(const FlashAttnArgs<QDataType, KDataType, VDataType, ODataType>& a, 
                          const ck_tile::stream_config& stream_config) {{
    float r = -1;
{F_dispatch}
    return r;
}}

template float flash_attention_fwd<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, float, float, ck_tile::half_t, float, ck_tile::half_t>(
    const FlashAttnArgs<ck_tile::half_t, ck_tile::half_t, ck_tile::half_t, ck_tile::half_t>&,
    const ck_tile::stream_config&);

}}
"""

#     API_PER_DTYPE = """    {F_if}(std::is_same_v<QDataType, {F_q_type}> && std::is_same_v<KDataType, {F_k_type}> && std::is_same_v<VDataType, {F_v_type}> && std::is_same_v<ODataType, {F_o_type}>) {{
# {F_per_size_case}
#     }}
# """
#     API_PER_SIZE_CASE = """        {F_if} {F_SIZE_COND} {{
# {F_inner_dispatch}
#         }}
# """
    API_INNER_CASE = """            {F_if} {F_VEC_COND}
                r = flash_attention_fwd_<QDataType, KDataType, VDataType, ODataType, traits_<{F_trait_name}>>(a, stream_config);
"""

    INSTANCE_BASE = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "flash_attention_fwd_api_common.hpp"

namespace ck_tile {
// clang-format off
//
{F_instance_def}
// clang-format on

}
"""

    def __init__(self, working_path, kernel_filter):
        self.working_path = working_path
        self.kernel_filter = kernel_filter

    @dataclass
    class h_traits:
        F_SaccDataType: str
        F_SMPLComputeDataType: str
        F_PDataType: str
        F_OaccDataType: str
        F_kBlockSize: int
        F_kHeadDim: int
        F_kM0PerBlock: int
        F_kN0PerBlock: int
        F_kK0PerBlock: int
        F_kN1PerBlock: int
        F_kK1PerBlock: int

        @property
        def trait_name(self) -> str:
            return (f"{DATA_TYPE_MAP[self.F_SaccDataType]}, "
                    f"{DATA_TYPE_MAP[self.F_SMPLComputeDataType]}, "
                    f"{DATA_TYPE_MAP[self.F_PDataType]}, "
                    f"{DATA_TYPE_MAP[self.F_OaccDataType]}, "
                    f"{self.F_kBlockSize}, {self.F_kHeadDim}, "
                    f"{self.F_kM0PerBlock}, {self.F_kN0PerBlock}, {self.F_kK0PerBlock}, "
                    f"{self.F_kN1PerBlock}, {self.F_kK1PerBlock}")

        @property
        def def_name(self) -> str:
            return (f"template float flash_attention_fwd_<{DATA_TYPE_MAP['fp16']}, "
                    f"{DATA_TYPE_MAP['fp16']}, {DATA_TYPE_MAP['fp16']}, {DATA_TYPE_MAP['fp16']}, "
                    f"traits_<{self.trait_name}>>(const FlashAttnArgs<{DATA_TYPE_MAP['fp16']}, "
                    f"{DATA_TYPE_MAP['fp16']}, {DATA_TYPE_MAP['fp16']}, {DATA_TYPE_MAP['fp16']}>&, "
                    "const ck_tile::stream_config&);")

    @dataclass
    class h_instance:
        F_DataTypePair: str  # "q,k,v,o"
        F_SizeCategory: str  # "small", "medium", "large"
        instance_list: List[Any]  # List[h_traits]

        INSTANCE_BASE = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "flash_attention_fwd_api_common.hpp"

namespace ck_tile {{
// clang-format off
//
{F_instance_def}
// clang-format on
}}
"""

        @property
        def name(self) -> str:
            q_type, k_type, v_type, o_type = self.F_DataTypePair.split(',')
            dtype_str = f"{q_type}_{k_type}_{v_type}_{o_type}"
            return f"flash_attention_fwd_{dtype_str}_{self.F_SizeCategory}"

        @property
        def content(self) -> str:
            instance_defs = '\n'.join(ins.def_name for ins in self.instance_list)
            return self.INSTANCE_BASE.format(F_instance_def=instance_defs)

    @property
    def name_api(self) -> str:
        return "flash_attention_fwd_api"

    @property
    def name_common_header(self) -> str:
        return "flash_attention_fwd_api_common"

    @property
    def content_common_header(self) -> str:
        return f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "flash_attention_fwd.hpp"

namespace ck_tile {{

template <typename SaccDataType_,
          typename SMPLComputeDataType_,
          typename PDataType_,
          typename OaccDataType_,
          index_t kBlockSize_,
          index_t kHeadDim_,
          index_t kM0PerBlock_,
          index_t kN0PerBlock_,
          index_t kK0PerBlock_,
          index_t kN1PerBlock_,
          index_t kK1PerBlock_>
struct flash_attention_fwd_traits_
{{
    using SaccDataType = ck_tile::remove_cvref_t<SaccDataType_>;
    using SMPLComputeDataType = ck_tile::remove_cvref_t<SMPLComputeDataType_>;
    using PDataType = ck_tile::remove_cvref_t<PDataType_>;
    using OaccDataType = ck_tile::remove_cvref_t<OaccDataType_>;

    static constexpr index_t kBlockSize  = kBlockSize_;
    static constexpr index_t kHeadDim    = kHeadDim_;
    static constexpr index_t kM0PerBlock = kM0PerBlock_;
    static constexpr index_t kN0PerBlock = kN0PerBlock_;
    static constexpr index_t kK0PerBlock = kK0PerBlock_;
    static constexpr index_t kN1PerBlock = kN1PerBlock_;
    static constexpr index_t kK1PerBlock = kK1PerBlock_;

    static constexpr ck_tile::index_t kWarpPerCu    = 8; // 2 warps per SIMD
    static constexpr ck_tile::index_t kWarpPerBlock = kBlockSize / warpSize;
    static constexpr ck_tile::index_t kBlockPerCu   = kWarpPerCu / kWarpPerBlock;
}};


template <typename SaccDataType,
          typename SMPLComputeDataType,
          typename PDataType,
          typename OaccDataType,
          ck_tile::index_t kBlockSize,
          ck_tile::index_t kHeadDim,
          ck_tile::index_t kM0PerBlock,
          ck_tile::index_t kN0PerBlock,
          ck_tile::index_t kK0PerBlock,
          ck_tile::index_t kN1PerBlock,
          ck_tile::index_t kK1PerBlock>
using traits_ = flash_attention_fwd_traits_<SaccDataType,
                                          SMPLComputeDataType,
                                          PDataType,
                                          OaccDataType,
                                          kBlockSize,
                                          kHeadDim,
                                          kM0PerBlock,
                                          kN0PerBlock,
                                          kK0PerBlock,
                                          kN1PerBlock,
                                          kK1PerBlock>;


template <typename QDataType,
        typename KDataType,
        typename VDataType,
        typename ODataType,
        typename Traits_>
float flash_attention_fwd_(const FlashAttnArgs<QDataType, KDataType, VDataType, ODataType>& a, 
                        const ck_tile::stream_config& stream_config) {{
    using SaccDataType        = typename Traits_::SaccDataType;
    using SMPLComputeDataType = typename Traits_::SMPLComputeDataType;
    using PDataType           = typename Traits_::PDataType;
    using OaccDataType        = typename Traits_::OaccDataType;

    index_t kGridSize = a.Batch * (a.M0 / Traits_::kM0PerBlock) * (a.N1 / Traits_::kN1PerBlock);

    if(stream_config.log_level_ > 0)
        std::cout << ", " << "FlashAttentionFwd<" << Traits_::kBlockSize << "," << Traits_::kHeadDim << ">" << std::flush;

    return ck_tile::launch_kernel(stream_config,
        ck_tile::make_kernel<Traits_::kBlockSize, Traits_::kBlockPerCu>(
        ck_tile::FlashAttentionFwd<QDataType,
                                KDataType,
                                VDataType,
                                SaccDataType,
                                SMPLComputeDataType,
                                PDataType,
                                OaccDataType,
                                ODataType,
                                Traits_::kBlockSize,
                                Traits_::kHeadDim,
                                Traits_::kM0PerBlock,
                                Traits_::kN0PerBlock,
                                Traits_::kK0PerBlock,
                                Traits_::kN1PerBlock,
                                Traits_::kK1PerBlock>{{}},
        kGridSize,
        Traits_::kBlockSize,
        0,
        a.q_ptr,
        a.k_ptr,
        a.v_ptr,
        a.o_ptr,
        a.M0,
        a.N0,
        a.K0,
        a.N1,
        a.Batch,
        a.strideQ,        // StrideQ
        a.strideK,        // StrideK
        a.strideV,        // StrideV
        a.strideO,        // StrideO
        a.batchStrideQ,   // BatchStrideQ
        a.batchStrideK,   // BatchStrideK
        a.batchStrideV,   // BatchStrideV
        a.batchStrideO)); // BatchStrideO
}}
}}
"""
    def content_api(self, args) -> str:
        # Sort based on dtype
        t_dtype_dict = {}
        blobs = self.get_blobs(args)

        for blob in blobs:
            if blob.F_DataTypePair not in t_dtype_dict:
                t_dtype_dict[blob.F_DataTypePair] = {}
            if blob.F_SizeCategory not in t_dtype_dict[blob.F_DataTypePair]:
                t_dtype_dict[blob.F_DataTypePair][blob.F_SizeCategory] = []
            t_dtype_dict[blob.F_DataTypePair][blob.F_SizeCategory].append(blob)

        d_str = ''
        for i_d, dtype_ in enumerate(t_dtype_dict):
            blob_per_t = t_dtype_dict[dtype_]
            size_str = ''

            for i_size, size_ in enumerate(blob_per_t):
                blob_per_size = blob_per_t[size_]
                inner_str = ""

                for i_b, b_ in enumerate(blob_per_size):
                    for i_ins, ins in enumerate(b_.instance_list):
                        idx_in_size = i_b * len(b_.instance_list) + i_ins
                        len_in_size = sum(len(b.instance_list) for b in blob_per_size)

                        size_cond = ""
                        if size_ == "small":
                            size_cond = "(a.M0 < 2048 && a.N0 < 2048)"
                        elif size_ == "medium":
                            size_cond = "(a.M0 >= 2048 && a.N0 >= 2048 && a.M0 < 4096 && a.N0 < 4096)"
                        else:  # large
                            size_cond = "(a.M0 >= 4096 || a.N0 >= 4096)"

                        inner_str += self.API_INNER_CASE.format(
                            # F_if=get_if_str(idx_in_size, len_in_size, False),
                            F_if=get_if_str(size_, len_in_size, False),
                            F_VEC_COND=size_cond,
                            F_trait_name=ins.trait_name
                        )

                # size_str += self.API_PER_SIZE_CASE.format(
                #     F_if=get_if_str(i_size, len(blob_per_t)),
                #     F_SIZE_COND=size_cond,
                #     F_inner_dispatch=inner_str
                # )
                size_str += inner_str

            # q_type, k_type, v_type, o_type = dtype_.split(',')
            # d_str += self.API_PER_DTYPE.format(
            #     F_if=get_if_str(i_d, len(t_dtype_dict)),
            #     F_q_type=DATA_TYPE_MAP[q_type],
            #     F_k_type=DATA_TYPE_MAP[k_type],
            #     F_v_type=DATA_TYPE_MAP[v_type],
            #     F_o_type=DATA_TYPE_MAP[o_type],
            #     F_per_size_case=size_str
            # )
            d_str += size_str

        api_base = self.API_BASE.format(
            F_traits_define=self.API_TRAITS_DEFINE,
            F_dispatch=d_str
        )
        return api_base

    def get_blobs(self, args):
        h_traits = self.h_traits
        h_instance = self.h_instance

        # Define kernel configurations for different size categories
        trait_dict = {
            "small": [
                h_traits('fp32', 'fp32', 'fp32', 'fp32', 128, 128, 128, 128, 32, 128, 32),
                # h_traits('fp32', 'fp32', 'fp32', 'fp32', 256, 128, 128, 64, 32, 64, 32),
            ],
            "medium": [
                h_traits('fp32', 'fp32', 'fp32', 'fp32', 128, 128, 128, 128, 32, 128, 32),
                # h_traits('fp32', 'fp32', 'fp32', 'fp32', 256, 128, 256, 128, 32, 128, 32),
            ],
            "large": [
                h_traits('fp32', 'fp32', 'fp32', 'fp32', 256, 128, 128, 128, 32, 128, 32),
                # h_traits('fp32', 'fp32', 'fp32', 'fp32', 512, 128, 256, 256, 32, 256, 32),
            ]
        }

        # Toy example only support fp16
        dtype_combinations = [
            "fp16,fp16,fp16,fp16"
        #    "bf16,bf16,bf16,bf16"
        ]

        total_blob = []
        for dtype_pair in dtype_combinations:
            for size_category in trait_dict:
                traits = trait_dict[size_category]
                # Convert data types for the current dtype_pair
                q_type, k_type, v_type, o_type = dtype_pair.split(',')
                current_traits = []
                for t in traits:
                    new_t = copy.copy(t)
                    new_t.F_SaccDataType = 'fp32'  # accumulation in fp32
                    new_t.F_SMPLComputeDataType = 'fp32'  # softmax compute in fp32
                    new_t.F_PDataType = q_type
                    new_t.F_OaccDataType = 'fp32'  # output accumulation in fp32
                    current_traits.append(new_t)

                total_blob.append(h_instance(dtype_pair, size_category, current_traits))

        return total_blob

    def list_blobs(self, args) -> None:
        w_p = Path(self.working_path)
        list_p = w_p / 'flash_attention_fwd_blobs.txt'
        blobs = self.get_blobs(args)

        with list_p.open('w') as list_f:
            # API related files
            list_f.write(str(w_p / (self.name_api + ".cpp")) + "\n")
            list_f.write(str(w_p / (self.name_common_header + ".hpp")) + "\n")
            # Kernel instance files
            for b in blobs:
                list_f.write(str(w_p / (b.name + ".cpp")) + "\n")

    def gen_blobs(self, args) -> None:
        w_p = Path(self.working_path)
        w_str = self.content_api(args)
        (w_p / (self.name_api + ".cpp")).write_text(w_str)
        (w_p / (self.name_common_header + ".hpp")).write_text(self.content_common_header)

        blobs = self.get_blobs(args)
        for b in blobs:
            (w_p / (b.name + ".cpp")).write_text(b.content)

def list_blobs(args):
    api_list = args.api.split(',')
    for api in api_list:
        if api == 'fwd':
            FlashAttentionFwdCodegen(args.working_path, args.filter).list_blobs(args)

def gen_blobs(args):
    api_list = args.api.split(',')
    for api in api_list:
        if api == 'fwd':
            FlashAttentionFwdCodegen(args.working_path, args.filter).gen_blobs(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for Flash Attention kernel",
    )
    parser.add_argument(
        "-a",
        "--api",
        default='fwd',
        required=False,
        help="supply API(s) to generate (default: fwd). separated by comma."
    )
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated"
    )
    parser.add_argument(
        "-l",
        "--list_blobs",
        action='store_true',
        help="list all the kernels to a file"
    )
    parser.add_argument(
        "-g",
        "--gen_blobs",
        action='store_true',
        help="generate all kernels into different tile"
    )
    parser.add_argument(
        "-f",
        "--filter",
        required=False,
        help="filter out kernels that need to generate"
    )

    args = parser.parse_args()

    if (args.gen_blobs and args.list_blobs) or ((not args.gen_blobs) and (not args.list_blobs)):
        print('gen_blobs/list_blobs must specify only one option')
        sys.exit()

    p = Path(args.working_path)
    if not p.exists():
        p.mkdir()

    if args.list_blobs:
        list_blobs(args)
    else:
        gen_blobs(args)
