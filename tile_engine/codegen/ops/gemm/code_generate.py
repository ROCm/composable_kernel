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
                 'fp8'  : 'ck_tile::fp8_t'}

def BOOL_MAP(b_) -> str:
    if b_:
        return 'true'
    else:
        return 'false'

class gemm_instance_codegen:

    GEMM_KERNEL_RUN_BASE = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
        F_pipeline_policy : Any  
        F_epilogue        : Any
        F_scheduler       : Any


    def __init__(self, working_path, kernel_filter, json_data):
        self.working_path = working_path
        self.kernel_filter = kernel_filter
        self.data = json_data
        init()

    def init(self):
        #TODO:: pass one datatype or multiple datatypes
       datatype_configuration = datatype_configuration(
            DATA_TYPE_MAP[self.data['Prec_datatype']],
            DATA_TYPE_MAP[self.data['Prec_datatype']],
            DATA_TYPE_MAP['fp32'],
            DATA_TYPE_MAP[self.data['Prec_datatype']]
        )

        gemm_trait = gemm_traits(
            self.data['kPadM'], self.data['kPadN'], self.data['kPadK'], 
            self.data['A_layout'], self.data['B_layout'], self.data['C_layout']
        )
        
        tile_shape = tile_shapes(
            [self.data['M_tile'], self.data['N_tile'], self.data['K_tile']], 
            [self.data['M_warp'], self.data['N_warp'], self.data['K_warp']], 
            [self.data['M_warp_tile'], self.data['N_warp_tile'], self.data['K_warp_tile']]
        )

        gemm_kernel_method = gemm_kernel_method(
            self.data['Pipeline_type'], 
            self.data['Scheduler_type'], #TODO:: what to do with pipeline-policy
            self.data['Epilogue_type']
        )


    def content_api(self, args) -> str:

    def gen_blobs(self, args) -> None:
        w_p = Path(self.working_path)
        w_str = self.content_api(args)
        (w_p / (self.name_api + ".cpp")).write_text(w_str)
        
def gen_blobs(args):
    api_list = args.api.split(',')
    for api in api_list:
        if api == 'single':
            gemm_instance_codegen(args.working_path, args.filter).gen_blobs(args)


def validate_json_data(json_data):
    '''
        Validate the json data for the kernel configurations
        For missing parameter: Assigned default values, 
        For invalid values: Raise an error
        TODO:: check case sensitivity for parameters names, default values confirmed.
        TODO:: check for valid values of tile sizes, warp and warp tiles
    '''
    int_ranges = {
        #       (min, max, default)
        "M_tile": (1, 256, 128),
        "N_tile": (1, 256, 128),
        "K_tile": (1, 512, 128),
        "M_warp": (1, 16, 4),
        "N_warp": (1, 16, 1),
        "K_warp": (1, 16, 1),
        "M_warp_tile": (1, 32, 32),
        "N_warp_tile": (1, 32, 32),
        "K_warp_tile": (1, 32, 32),
        "kPadM": (0, 1, 0),
        "kPadN": (0, 1, 0),
        "kPadK": (0, 1, 0)
    }
    
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

    for key, value in int_ranges.items():
        if key in json_data:
            if not isinstance(json_data[key], int) or not (value[0] <= json_data[key] <= value[1]):
                raise ValueError(f'Invalid value for {key}: {json_data[key]}. Must be an integer between {value[0]} and {value[1]}. ')
        else:
            json_data[key] = value[-1]
    
    for key, value in string_values.items():
        if key in json_data:
            if not isinstance(json_data[key], str) or json_data[key] not in value:
                raise ValueError(f'Invalid value for {key}: {json_data[key]}. Must be one of {value[:-1]}. ')
        else:
            json_data[key] = value[-1]

    print("Valid json data")
    print(json_data)
    return json_data
     


def main(args):
    # Read and validate json file
    with open(args.json, 'r') as json_file:
        data = json.load(json_file)
    data = validate_json_data(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm kernel",
    )
    parser.add_argument(
        "-a", "--api", default='single', required=False, help="supply API(s) to generate (default: single). separated by comma."
    )
    parser.add_argument(
        "-j", "--json", required=True, help="Path to the json which contains the kernel configurations"
    )

    args = parser.parse_args()
    main(args)
