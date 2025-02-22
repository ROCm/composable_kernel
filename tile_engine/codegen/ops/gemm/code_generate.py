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
    class tile_shape:
        F_BlockTile      : List[int]
        F_WarpPerBlock   : List[int]
        F_WarpTile       : List[int]
    
    @dataclass
    class gemm_kernel_method:
        F_pipeline        : Any
        F_pipeline_policy : Any
        F_epilogue        : Any
    def __init__(self, working_path, kernel_filter):
        self.working_path = working_path
        self.kernel_filter = kernel_filter
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm kernel",
    )
    parser.add_argument(
        "-a",
        "--api",
        default='single_instance',
        required=False,
        help="supply API(s) to generate (default: single_instance). separated by comma."
    )

    args = parser.parse_args()
