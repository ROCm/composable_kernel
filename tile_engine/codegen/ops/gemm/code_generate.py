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

def BOOL_MAP(b_) -> str:
    if b_:
        return 'true'
    else:
        return 'false'

class gemm_instance_codegen:
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

    def get_blobs(self, args):
        