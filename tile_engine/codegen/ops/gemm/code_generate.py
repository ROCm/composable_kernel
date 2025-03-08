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


DATA_TYPE_MAP = {'fp32' : 'float',
                 'fp16' : 'ck_tile::fp16_t',
                 'bf16' : 'ck_tile::bf16_t',
                 'int8' : 'ck_tile::int8_t',
                 'fp8'  : 'ck_tile::fp8_t',
                 'bf8'  :  'ck_tile::bf8_t'
                }

LAYOUT_MAP = {'R' : 'ck_tile::tensor_layout::gemm::RowMajor',
              'C' : 'ck_tile::tensor_layout::gemm::ColumnMajor'}

def sizeOf(data_type):
    if data_type == 'fp16' or data_type == 'bf16':
        return 2
    elif data_type == 'int8' or data_type == 'fp8':
        return 1
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

using ADataType = {ADataTypeDefine};
using BDataType = {BDataTypeDefine};
using AccDataType = {AccDataTypeDefine};
using CDataType = {ODataTypeDefine};
using ALayout = {ALayoutDefine};
using BLayout = {BLayoutDefine};
using CLayout = {CLayoutDefine};

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
        if self.data['Prec_datatype'] in ['fp8', 'bf8']:
            ctype = 'fp16'
        self.datatype_config = gemm_instance_codegen.datatype_configuration(
            DATA_TYPE_MAP[self.data['Prec_datatype']],
            DATA_TYPE_MAP[self.data['Prec_datatype']],
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
            self.data['Scheduler_type'], 
            self.data['Epilogue_type']
        )

    @property
    def name_common_header(self) -> str:
        return 'tensor_configuration'

    @property
    def common_header(self) -> str:
        str1 = self.COMMON_HEADER.format(
            ADataTypeDefine = self.datatype_config.F_ADataType,
            BDataTypeDefine = self.datatype_config.F_BDataType,
            AccDataTypeDefine = self.datatype_config.F_AccDataType,
            ODataTypeDefine = self.datatype_config.F_ODataType,
            ALayoutDefine = LAYOUT_MAP[self.trait.F_A_Layout],  
            BLayoutDefine = LAYOUT_MAP[self.trait.F_B_Layout],
            CLayoutDefine = LAYOUT_MAP[self.trait.F_C_Layout]            
        )
        return str1

    def content_api(self, args) -> str:
        return ""

    def gen_blobs(self, args) -> None:
        w_p = Path(self.working_path)
        #w_str = self.content_api(args)
        file_name = self.name_common_header + ".hpp"
        header = self.common_header
        (w_p / (file_name)).write_text(self.common_header)
        
def gen_blobs(args, data):
    api_list = args.api.split(',')
    for api in api_list:
        if api == 'single':
            gemm_instance_codegen(args.working_path, args.filter, data).gen_blobs(args)


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
        "Prec_datatype": ["fp16", "bf16", "fp8", "bf8", "fp16"],
        "Pipeline_type": ["Memory", "ComputeV3", "ComputeV4", "ComputeV3"],
        "Scheduler_type": ["Interwave", "Intrawave", "Interwave"],
        "Epilogue_type": ["CShuffleEpilogue", "DefaultGemm2DEpilogue", "CShuffleEpilogue"]
    }

    datatype_to_warp_tile_map = {
        'fp16' : [(32,32,8), (16,16,16), (4,64,4), (64,4,4)],
        'bf16' : [(32,32,8), (16,16,16), (4,64,4), (64,4,4)],
        'int8' : [(32,32,16)],
        'fp8' : [(32,32,16)]
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

   
    return json_data
     

def main(args):
    # Read and validate json file
    with open(args.json, 'r') as json_file:
        data = json.load(json_file)
    data = validate_json_data(data)
    gen_blobs(args, data)


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
    #TODO:: Not needed as of now just added for completeness
    parser.add_argument(
        "-f", "--filter", required=False, help="filter out kernels that need to generate, using fnmatch module"
    )

    parser.add_argument(
        "-j", "--json", required=True, help="Path to the json which contains the kernel configurations"
    )

    args = parser.parse_args()
    main(args)
