# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from enum import IntEnum
from pathlib import Path
import sys
from typing import List, Optional, Any
import functools
import itertools
import copy
import json
from dataclasses import dataclass

def get_if_str(idx, total, lase_else = True):
    if idx == 0:
        return 'if'
    elif idx < total - 1:
        return 'else if'
    else:
        if lase_else:
            return 'else'
        else:
            return 'else if'

 
DATA_TYPE_MAP = {'fp32'  : 'float',
                 'fp16'  : 'ck_tile::half_t',
                 'bf16'  : 'ck_tile::bf16_t',
                 'int8'  : 'ck_tile::int8_t',
                 'fp8'   : 'ck_tile::fp8_t',
                 'bf8'   :  'ck_tile::bf8_t',
                 'int4'  : 'ck_tile::pk_int4_t'
                }

LAYOUT_MAP = {'R' : 'ck_tile::tensor_layout::gemm::RowMajor',
              'C' : 'ck_tile::tensor_layout::gemm::ColumnMajor'}

def sizeOf(data_type):
    if data_type == 'fp16' or data_type == 'bf16':
        return 2
    elif data_type == 'int8' or data_type == 'fp8' or data_type == 'bf8':
        return 1
    elif data_type == 'int4': ## TODO:: needs to confirm
        return 0.5
    else:
        return 4

def BOOL_MAP(b_) -> str:
    if b_:
        return 'true'
    else:
        return 'false'

class gemm_instance_codegen:

    COMMON_HEADER = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/core.hpp"
#pragma once

using ADataType = {ADataTypeDefine};
using BDataType = {BDataTypeDefine};
using AccDataType = {AccDataTypeDefine};
using CDataType = {ODataTypeDefine};

using ALayout = {ALayoutDefine};
using BLayout = {BLayoutDefine};
using CLayout = {CLayoutDefine};

struct GemmConfig
{{
    static constexpr ck_tile::index_t M_Tile = {mTileDefine};
    static constexpr ck_tile::index_t N_Tile = {nTileDefine};
    static constexpr ck_tile::index_t K_Tile = {kTileDefine};

    static constexpr ck_tile::index_t M_Warp = {mWarpDefine};
    static constexpr ck_tile::index_t N_Warp = {nWarpDefine};
    static constexpr ck_tile::index_t K_Warp = {kWarpDefine};

    static constexpr ck_tile::index_t M_Warp_Tile = {mWarpTileDefine};
    static constexpr ck_tile::index_t N_Warp_Tile = {nWarpTileDefine};
    static constexpr ck_tile::index_t K_Warp_Tile = {kWarpTileDefine};

    static constexpr bool DoubleSmemBuffer = {doubleSmemBufferDefine};


    static constexpr bool kPadM = {kPadMDefine};
    static constexpr bool kPadN = {kPadNDefine};
    static constexpr bool kPadK = {kPadKDefine};

    static constexpr bool PermuteA = false;  //TODO:: still deciding
    static constexpr bool PermuteB = false;

    static constexpr bool TransposeC = false;

    static constexpr int kBlockPerCu                         = 1;
    static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
    static constexpr ck_tile::index_t TileParitionerM01      = 4;
}};

"""
    DEFAULT_EPILOGUE = """
        using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
                                ck_tile::DefaultGemm2DEpilogueProblem<AccDataType, 
                                                                      CDatatType, 
                                                                      CLayout, 
                                                                      GemmConfig::kPadM,
                                                                      GemmConfig::kPadN,
                                                                      GemmConfig::M_Warp_Tile,
                                                                      GemmConfig::N_Warp_Tile,
                                                                      GemmConfig::K_Warp_Tile,
                                                                      UniversalGemmProblem::TransposeC>>
"""

    CSHUFFLE_EPILOGUE = """
        using GemmEpilogue = ck_tile::CShuffleEpilogue<
                            ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             CDataType,
                                                             CLayout,
                                                             GemmPipelineProblem::kBlockSize,
                                                             TilePartitioner::MPerBlock,
                                                             TilePartitioner::NPerBlock,
                                                             GemmConfig::M_Warp,
                                                             GemmConfig::N_Warp,
                                                             GemmConfig::M_Warp_Tile,
                                                             GemmConfig::N_Warp_Tile,
                                                             GemmConfig::K_Warp_Tile,
                                                             UniversalGemmProblem::TransposeC>>;
"""
    HOT_LOOP_FALSE = """
        if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<false>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }
        else if(tail_num == ck_tile::TailNumber::Odd)
        {
            Run(ck_tile::bool_constant<false>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
        }
        else if(tail_num == ck_tile::TailNumber::Even)
        {
            Run(ck_tile::bool_constant<false>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
        }
        else
        {
            throw std::runtime_error("Num K loop must be larger than number of prefetech stages.");
        }  
"""
    RUN_MEM = """
        if(tail_num == ck_tile::TailNumber::One)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
        }
        else if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }

        if constexpr(BaseGemmPipeline::PrefetchStages > 2)
        {
            if(tail_num == ck_tile::TailNumber::Two)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
            }
       
            if(tail_num == ck_tile::TailNumber::Three)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
            }
            if(tail_num == ck_tile::TailNumber::Four)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{});
            }
            if(tail_num == ck_tile::TailNumber::Five)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{});
            }
            if(tail_num == ck_tile::TailNumber::Six)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{});
            }
            if(tail_num == ck_tile::TailNumber::Seven)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{});
            }
            throw std::runtime_error("The tile number is wrong! It should not exceed the prefetch stage numbers");
        }
"""

    RUN_COMPV3 = """
        if(tail_num == ck_tile::TailNumber::Full)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
        }
        else if(tail_num == ck_tile::TailNumber::Odd)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
        }
        else if(tail_num == ck_tile::TailNumber::Even)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
        }
        else
        {
            throw std::runtime_error("The tail number is wrong. It should be Full, Odd, or Even.");
        }
"""

    RUN_COMPV4 = """
        if(tail_num == ck_tile::TailNumber::Three)
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
        }
        else
        {
            Run(ck_tile::bool_constant<true>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
        }
"""

    GEMM_KERNEL_HEADER = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "tensor_configuration.hpp"
#include "gemm_host.hpp"


template <typename Tensor>
void permute_tensor_b(Tensor& tensor)
{{
    const ck_tile::index_t K  = tensor.get_length(0);
    const ck_tile::index_t N  = tensor.get_length(1);
    const ck_tile::index_t K1 = {KPerThread};
    const ck_tile::index_t K0 = K / K1;

    Tensor tensor_copy = tensor;

    // int K0, N, K1
    for(int j = 0; j < K0; j++)
    {{
        for(int i = 0; i < N; i++)
        {{
            for(int jj = 0; jj < K1; jj++)
            {{
                tensor(j * N * K1 + i * K1 + jj) = tensor_copy(i * K + (j * K1 + jj));
            }}
        }}
    }}
}}




template<typename ADataType, 
         typename BDataType, 
         typename AccDataType,
         typename CDataType,
         typename ALayout,         
         typename BLayout,
         typename CLayout>
float gemm_kernel_launch(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
{{
    using GemmShape = 
        ck_tile::TileGemmShape<ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
                               ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
                               ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
                               GemmConfig::PermuteA,
                               GemmConfig::PermuteB>;

    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using Traits  = 
        ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                GemmConfig::kPadN,
                                GemmConfig::kPadK,
                                ALayout,
                                BLayout,
                                CLayout>; 

    using GemmUniversalTraits = 
        ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                         GemmConfig::kPadN,
                                         GemmConfig::kPadK,
                                         GemmConfig::DoubleSmemBuffer,
                                         ALayout,
                                         BLayout,
                                         CLayout,
                                         GemmConfig::TransposeC>;     

    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, 
                                     BDataType, 
                                     AccDataType, 
                                     GemmShape, 
                                     Traits>;      

    using BaseGemmPipeline = {universal_gemm_pipeline}<GemmPipelineProblem>; 

    const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{{0}};    
    const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
        constexpr bool has_hot_loop_v = has_hot_loop_.value;
        constexpr auto tail_number_v  = tail_number_.value;
        constexpr auto scheduler      = {gemm_pipeline_scheduler}; 

        using UniversalGemmProblem = 
            ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                  BDataType,
                                                  AccDataType,
                                                  GemmShape,
                                                  GemmUniversalTraits,
                                                  scheduler,
                                                  has_hot_loop_v,
                                                  tail_number_v>;  

        using GemmPipeline = {gemm_pipeline}<UniversalGemmProblem>;    

        {epilogue_define};

        using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        auto kargs   = Kernel::MakeKernelArgs(args);

        const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
        constexpr dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {{
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
        }}

        if(s.log_level_ > 0)
        {{
            std::cout << "Launching kernel with args:"
                      << " grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                      << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                      << std::endl;
        }}

        ave_time = ck_tile::launch_kernel(s,
                                          ck_tile::make_kernel<blocks.x, GemmConfig::kBlockPerCu>(
                                              Kernel{{}}, grids, blocks, 0, kargs));
        return ave_time;
    }};     

    if(has_hot_loop){{
        {run_func}
    }}else{{
        {hot_loop_false}
    }}
   
                                                                                                                    
    return ave_time;
}}
"""

    @dataclass
    class datatype_configuration:
        F_ADataType: str
        F_BDataType: str
        F_AccDataType: str
        F_ODataType: str
    
    @dataclass
    class gemm_traits:
        F_kPadM: bool
        F_kPadN: bool
        F_kPadK: bool
        F_A_Layout: str
        F_B_Layout: str
        F_C_Layout: str


    @dataclass
    class tile_shapes:
        F_BlockTile      : List[int]
        F_WarpPerBlock   : List[int]
        F_WarpTile       : List[int]
    
    @dataclass
    class gemm_kernel_method:
        F_pipeline        : Any
        F_epilogue        : Any
        F_scheduler       : Any


    def __init__(self, working_path, kernel_filter, json_data):
        self.working_path = working_path
        self.kernel_filter = kernel_filter
        self.data = json_data
        self.init()

    def init(self):
        #TODO:: pass one datatype or multiple datatypes; o_type could be different
        ctype = self.data['Prec_datatype']
        atype = self.data['Prec_datatype']
        btype = self.data['Prec_datatype']
        if self.data['Prec_datatype'] in ['fp8', 'bf8']:
            ctype = 'fp16'
        elif self.data['Prec_datatype'] in ['int4']:
            atype = 'fp16'
            ctype = 'fp16'

        self.datatype_config = gemm_instance_codegen.datatype_configuration(
            DATA_TYPE_MAP[atype],
            DATA_TYPE_MAP[btype],
            DATA_TYPE_MAP['fp32'],
            DATA_TYPE_MAP[ctype] 
        )

        self.trait = gemm_instance_codegen.gemm_traits(
            self.data['kPadM'], self.data['kPadN'], self.data['kPadK'], 
            self.data['A_layout'], self.data['B_layout'], self.data['C_layout']
        )
        
        self.tile_shape = gemm_instance_codegen.tile_shapes(
            [self.data['M_tile'], self.data['N_tile'], self.data['K_tile']], 
            [self.data['M_warp'], self.data['N_warp'], self.data['K_warp']], 
            [self.data['M_warp_tile'], self.data['N_warp_tile'], self.data['K_warp_tile']]
        )

        self.kernel_method = gemm_instance_codegen.gemm_kernel_method(
            self.data['Pipeline_type'], 
            self.data['Epilogue_type'], 
            self.data['Scheduler_type']
        )

        if self.data['Prec_datatype'] in ['fp16', 'bf16']:
            self.KPerThread = 4
        elif self.data['Prec_datatype'] in ['fp8', 'bf8']:
            self.KPerThread = 8
        else:
            # fallback or int4 or int8 or fp32 – up to you:
            self.KPerThread = 1

    @property
    def name_common_header(self) -> str:
        return 'tensor_configuration'
    
    @property
    def name_kernel_file(self) -> str:
        return 'launch_kernel'

    @property
    def common_header(self) -> str:
        str1 = self.COMMON_HEADER.format(
            ADataTypeDefine = self.datatype_config.F_ADataType,
            BDataTypeDefine = self.datatype_config.F_BDataType,
            AccDataTypeDefine = self.datatype_config.F_AccDataType,
            ODataTypeDefine = self.datatype_config.F_ODataType,
            ALayoutDefine = LAYOUT_MAP[self.trait.F_A_Layout],  
            BLayoutDefine = LAYOUT_MAP[self.trait.F_B_Layout],
            CLayoutDefine = LAYOUT_MAP[self.trait.F_C_Layout],  
            mTileDefine = self.tile_shape.F_BlockTile[0],
            nTileDefine = self.tile_shape.F_BlockTile[1],
            kTileDefine = self.tile_shape.F_BlockTile[2],
            mWarpDefine = self.tile_shape.F_WarpPerBlock[0],
            nWarpDefine = self.tile_shape.F_WarpPerBlock[1],
            kWarpDefine = self.tile_shape.F_WarpPerBlock[2],
            mWarpTileDefine = self.tile_shape.F_WarpTile[0],
            nWarpTileDefine = self.tile_shape.F_WarpTile[1],
            kWarpTileDefine = self.tile_shape.F_WarpTile[2],
            doubleSmemBufferDefine = BOOL_MAP(self.kernel_method.F_pipeline == 'ComputeV4'),
            kPadMDefine = BOOL_MAP(self.trait.F_kPadM),
            kPadNDefine = BOOL_MAP(self.trait.F_kPadN), 
            kPadKDefine = BOOL_MAP(self.trait.F_kPadK),        
        )
        return str1

    def gemm_pipeline_func(self):
        list_f = []
        if self.kernel_method.F_pipeline == 'Memory':
            list_f.append('ck_tile::BaseGemmPipelineAgBgCrMem')
            list_f.append('ck_tile::GemmPipelineAgBgCrMem')
        elif self.kernel_method.F_pipeline == 'ComputeV3':
            list_f.append('ck_tile::BaseGemmPipelineAgBgCrCompV3')
            list_f.append('ck_tile::GemmPipelineAgBgCrCompV3')
        else:
            list_f.append('ck_tile::BaseGemmPipelineAgBgCrCompV4')
            list_f.append('ck_tile::GemmPipelineAgBgCrCompV4')
        return list_f
        
    def gemm_scheduler(self):
        if self.kernel_method.F_scheduler == 'Interwave':
            return 'ck_tile::GemmPipelineScheduler::Interwave'
        else:
            return 'ck_tile::GemmPipelineScheduler::Intrawave'
            
    def content_api(self, args) -> str:
        list_f = self.gemm_pipeline_func()
        run_f = self.RUN_MEM if self.kernel_method.F_pipeline == 'Memory' else self.RUN_COMPV3 if self.kernel_method.F_pipeline == 'ComputeV3' else self.RUN_COMPV4
        str1 = self.GEMM_KERNEL_HEADER.format(
                universal_gemm_pipeline = list_f[0], 
                gemm_pipeline_scheduler = self.gemm_scheduler(), 
                gemm_pipeline = list_f[1],
                epilogue_define = self.DEFAULT_EPILOGUE if self.kernel_method.F_epilogue == "Default" else self.CSHUFFLE_EPILOGUE,
                run_func = run_f.replace('{HOP_LOOP_FALSE}', self.HOT_LOOP_FALSE),
                hot_loop_false = self.HOT_LOOP_FALSE,
                KPerThread = self.KPerThread          
                )
        return str1

    def get_blobs(self, args) -> List[str]:
        return []
    
    def list_blobs(self, args) -> None:
        w_p = Path(self.working_path)
        list_p = w_p / 'gemm_instance_blobs.txt'
        blobs = self.get_blobs(args)
        with list_p.open('w') as list_f:
            # api related file
            list_f.write(str(w_p / (self.name_common_header + ".hpp"))  + "\n")
            list_f.write(str(w_p / (self.name_kernel_file + ".hpp"))  + "\n")   #TODO:: define name_api
            # kernel instance file
            #for b in blobs:
            #    list_f.write(str(w_p / (b.name + ".cpp")) + "\n") 
        

    def gen_blobs(self, args) -> None:
        w_p = Path(self.working_path)
        w_str = self.content_api(args)
        (w_p / (self.name_common_header + ".hpp")).write_text(self.common_header)
        (w_p / (self.name_kernel_file + ".hpp")).write_text(w_str)

        
        

        
def do_list_blobs(args, data):
    api_list = args.api.split(',')
    for api in api_list:
        if api == 'single':
            generator = gemm_instance_codegen(args.working_path, args.filter, data)
            generator.list_blobs(args)

def do_gen_blobs(args, data):
    api_list = args.api.split(',')
    for api in api_list:
        if api == 'single':
            generator = gemm_instance_codegen(args.working_path, args.filter, data)
            generator.gen_blobs(args)


def validate_json_data(json_data):
    '''
        Validate the json data for the kernel configurations
        For missing parameter: Assigned default values, 
        For invalid values: Raise an error
        TODO:: check case sensitivity for parameters names.
    '''
   
    string_values = {
        #           [possible values, last entry is default]
        "A_layout": ["R", "C", "R"],
        "B_layout": ["R", "C", "C"],
        "C_layout": ["R", "C", "R"], 
        "Prec_datatype": ["fp16", "bf16", "fp8", "bf8", "fp16", "int4", "fp16"],
        "Pipeline_type": ["Memory", "ComputeV3", "ComputeV4", "ComputeV3"],
        "Scheduler_type": ["Interwave", "Intrawave", "Interwave"],
        "Epilogue_type": ["CShuffle", "Default", "CShuffle"]
    }

    datatype_to_warp_tile_map = {
        'fp16' : [(32,32,8), (32,32,16), (16,16,16), (4,64,4), (64,4,4)],
        'bf16' : [(32,32,8), (32,32,16), (16,16,16), (4,64,4), (64,4,4)],
        'int8' : [(32,32,16)],
        'fp8' : [(32,32,16)],
        'bf8' : [(32,32,16)] , 
        'int4' : [(32,32,8), (32,32,16), (16,16,16), (4,64,4), (64,4,4)]
    }

    # Validate String values
    for key, value in string_values.items():
        if key in json_data:
            if not isinstance(json_data[key], str) or json_data[key] not in value:
                raise ValueError(f'Invalid value for {key}: {json_data[key]}. Must be one of {value[:-1]}. ')
        else:
            json_data[key] = value[-1]

    # LDS size validation
    if isinstance(json_data['M_tile'], int) and isinstance(json_data['N_tile'], int) and isinstance(json_data['K_tile'], int):
        total_tiles = json_data['M_tile'] * json_data['K_tile'] +  json_data['N_tile'] * json_data['K_tile']
        if json_data['Pipeline_type'] != "ComputeV4":
            if total_tiles * sizeOf(json_data['Prec_datatype']) > pow(2, 16):
                raise ValueError(f'Total tile size should not exceed 64KB of LDS. Current size: {total_tiles * sizeOf(json_data["Prec_datatype"])} bytes. ')
        else:
            if total_tiles * sizeOf(json_data['Prec_datatype']) > pow(2, 15):
                raise ValueError(f'Total tile size should not exceed 32KB of LDS. Current size: {total_tiles * sizeOf(json_data["Prec_datatype"])} bytes. ')
    else:
        raise ValueError(f'Invalid value for tile sizes. Must be integers. ')
    
    # Warp tile validation
    if json_data['Prec_datatype'] in datatype_to_warp_tile_map:
        if isinstance(json_data['M_warp_tile'], int) and isinstance(json_data['N_warp_tile'], int) and isinstance(json_data['K_warp_tile'], int):
            if (json_data['M_warp_tile'], json_data['N_warp_tile'], json_data['K_warp_tile']) not in datatype_to_warp_tile_map[json_data['Prec_datatype']]:
                raise ValueError(f'Invalid warp tile sizes for datatype {json_data["Prec_datatype"]}. Must be one of {datatype_to_warp_tile_map[json_data["Prec_datatype"]][:]}')
        else:
            raise ValueError(f'Invalid value for warp tile sizes. Must be integers. ')
    else:
        raise ValueError(f'Invalid datatype {json_data["Prec_datatype"]}. ')

    # Warp validation  
    possible_warp_m = json_data['M_tile'] / json_data['M_warp_tile']
    possible_warp_n = json_data['N_tile'] / json_data['N_warp_tile']
    possible_warp_k = json_data['K_tile'] / json_data['K_warp_tile']

    if possible_warp_m % json_data['M_warp'] != 0 or possible_warp_n % json_data['N_warp'] != 0 or possible_warp_k % json_data['K_warp'] != 0:
        raise ValueError(f'Invalid warp sizes. M_tile, N_tile, K_tile should be divisible by M_warp, N_warp, K_warp. ')

    # pad values must be bool
    if not isinstance(json_data['kPadM'], bool) and not isinstance(json_data['kPadN'], bool) and not isinstance(json_data['kPadK'], bool):
        raise ValueError(f'Invalid value for padding. Must be boolean. ')

    # Intrawave scheduler support for Compute pipeline
    if json_data['Pipeline_type'] == 'ComputeV3' or json_data['Pipeline_type'] == 'ComputeV4':
        if json_data['Scheduler_type'] == 'Interwave':
            raise ValueError(f'Invalid scheduler type for Compute pipeline. Must be Intrawave. ')

    # int4 datatype supported only for ComputeV3 pipeline
    if json_data['Prec_datatype'] == 'int4' and json_data['Pipeline_type'] != 'ComputeV3':
        raise ValueError(f'Invalid pipeline type for int4 datatype. Must be ComputeV3 only')
   
    return json_data
     

def main(args):
    # Read and validate json file
    with open(args.json, 'r') as json_file:
        data = json.load(json_file)
    data = validate_json_data(data)
    if args.list_blobs:
        do_list_blobs(args, data)
    elif args.gen_blobs:
        do_gen_blobs(args, data)
    else:
        # If neither was specified, either do nothing or default to gen_blobs
        print("No mode specified (use --list_blobs or --gen_blobs). Generating by default...")
        do_gen_blobs(args, data)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm kernel",
    )
    parser.add_argument(
        "-a", "--api", default='single', required=False, help="supply API(s) to generate (default: single). separated by comma."
    )
    parser.add_argument(
        "-w", "--working_path", default="./", required=False, help="the path where all the blobs are going to be generated"
    )
    #TODO:: Not needed; just added for completeness
    parser.add_argument(
        "-f", "--filter", required=False, help="filter out kernels that need to generate, using fnmatch module"
    )

    parser.add_argument(
        "-j", "--json", required=True, help="Path to the json which contains the kernel configurations"
    )

    parser.add_argument(
        "-l",
        "--list_blobs",
        action='store_true',
        help="list all the kernels to a file, "
    )

    parser.add_argument(
        "-g",
        "--gen_blobs",
        action='store_true',
        help="generate all kernels into different tile"
    )

    args = parser.parse_args()

    if (args.gen_blobs and args.list_blobs) or ((not args.gen_blobs) and (not args.list_blobs)):
        print('gen_blobs/list_blobs must specify only one option')
        sys.exit()

    p = Path(args.working_path)
    if not p.exists():
        p.mkdir()

    main(args)
