# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from pathlib import Path
from typing import List, Optional

from codegen.cmake_config import *
from codegen.ops import (
    fmha_fwd,
    fmha_bwd
)

def write_blobs(output_dir: Optional[str], api_list : List[str], kernel_filter : Optional[str], receipt, mask_impl) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    write_blobs_iml = {
        'fwd': fmha_fwd.write_blobs,
        'bwd': fmha_bwd.write_blobs,
    }
    for api in api_list:
        write_blobs_iml[api](output_dir, kernel_filter, receipt, mask_impl)

# list all the files that will be generated
def list_blobs(output_file : Optional[str], api_list : List[str], kernel_filter : Optional[str], receipt, mask_impl) -> None:
    assert output_file is not None
    file_path = Path(output_file)

    list_blobs_iml = {
        'fwd': fmha_fwd.list_blobs,
        'bwd': fmha_bwd.list_blobs,
    }
    for api in api_list:
        list_blobs_iml[api](file_path, kernel_filter, receipt, mask_impl)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK fmha kernel",
    )
    parser.add_argument(
        "-d",
        "--direction", # we keep 'direction' option for backward compatibility
        "-a",
        "--api",
        default='fwd',
        required=False,
        help="supply API(s) to generate (default: fwd). separated by comma."
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

    parser.add_argument(
        "-m",
        "--mask",
        default="simplified",
        required=False,
        help="mask implementation, simplified/generic"
    )

    parser.add_argument(
        "-r",
        "--receipt",
        default=0,
        required=False,
        help="codegen receipt. 0: generate only 8xhdim coverage\n"  + \
             "  1: generate more instance to cover all hdim\n"  + \
             "  2: Only generate instance for Flash attention integration"
    )

    args = parser.parse_args()
    api_list = args.direction.split(',')
    if args.list_blobs is not None:
        list_blobs(args.list_blobs, api_list, args.filter, int(args.receipt), mask_impl=args.mask)
    else:
        write_blobs(args.output_dir, api_list, args.filter, int(args.receipt), mask_impl=args.mask)
