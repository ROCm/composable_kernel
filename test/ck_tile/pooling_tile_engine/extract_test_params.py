#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Extract pooling test parameters from config JSON and write to C++ header.
Generates test_params.hpp with problem sizes for parameterized GTest.
"""

import json
import argparse
import os
from pathlib import Path


def extract_test_params(config_file, output_file, pooling_dim="2d"):
    """Extract test parameters from config JSON and write to output file"""

    with open(config_file, "r") as f:
        config = json.load(f)

    # Extract test parameters based on pooling dimension
    test_params = []
    if pooling_dim == "2d":
        if "test_params" in config and "problem_sizes_2d" in config["test_params"]:
            test_params = config["test_params"]["problem_sizes_2d"]
        else:
            # Default 2D test parameters
            test_params = [
                {
                    "N": 1,
                    "H": 8,
                    "W": 8,
                    "C": 32,
                    "Y": 2,
                    "X": 2,
                    "stride_h": 2,
                    "stride_w": 2,
                    "dilation_h": 1,
                    "dilation_w": 1,
                    "pad_h_left": 0,
                    "pad_h_right": 0,
                    "pad_w_left": 0,
                    "pad_w_right": 0,
                },
                {
                    "N": 2,
                    "H": 16,
                    "W": 16,
                    "C": 32,
                    "Y": 3,
                    "X": 3,
                    "stride_h": 2,
                    "stride_w": 2,
                    "dilation_h": 1,
                    "dilation_w": 1,
                    "pad_h_left": 1,
                    "pad_h_right": 1,
                    "pad_w_left": 1,
                    "pad_w_right": 1,
                },
            ]
    else:  # 3d
        if "test_params" in config and "problem_sizes_3d" in config["test_params"]:
            test_params = config["test_params"]["problem_sizes_3d"]
        else:
            # Default 3D test parameters
            test_params = [
                {
                    "N": 1,
                    "D": 4,
                    "H": 4,
                    "W": 4,
                    "C": 32,
                    "Z": 2,
                    "Y": 2,
                    "X": 2,
                    "stride_d": 2,
                    "stride_h": 2,
                    "stride_w": 2,
                    "dilation_d": 1,
                    "dilation_h": 1,
                    "dilation_w": 1,
                    "pad_d_left": 0,
                    "pad_d_right": 0,
                    "pad_h_left": 0,
                    "pad_h_right": 0,
                    "pad_w_left": 0,
                    "pad_w_right": 0,
                },
            ]

    # Write to output file in C++ format
    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        f.write("// Generated test parameters for pooling tile_engine tests\n")
        f.write("// This file is auto-generated during CMake configuration\n\n")

        if pooling_dim == "2d":
            f.write(
                "static const std::vector<PoolTestParams2D> CONFIG_TEST_PARAMS = {\n"
            )
            for i, params in enumerate(test_params):
                comma = "," if i < len(test_params) - 1 else ""
                f.write(
                    f"    {{"
                    f"{params['N']}, {params['H']}, {params['W']}, {params['C']}, "
                    f"{params['Y']}, {params['X']}, "
                    f"{params['stride_h']}, {params['stride_w']}, "
                    f"{params['dilation_h']}, {params['dilation_w']}, "
                    f"{params['pad_h_left']}, {params['pad_h_right']}, "
                    f"{params['pad_w_left']}, {params['pad_w_right']}"
                    f"}}{comma}\n"
                )
            f.write("};\n")
        else:  # 3d
            f.write(
                "static const std::vector<PoolTestParams3D> CONFIG_TEST_PARAMS = {\n"
            )
            for i, params in enumerate(test_params):
                comma = "," if i < len(test_params) - 1 else ""
                f.write(
                    f"    {{"
                    f"{params['N']}, {params['D']}, {params['H']}, {params['W']}, {params['C']}, "
                    f"{params['Z']}, {params['Y']}, {params['X']}, "
                    f"{params['stride_d']}, {params['stride_h']}, {params['stride_w']}, "
                    f"{params['dilation_d']}, {params['dilation_h']}, {params['dilation_w']}, "
                    f"{params['pad_d_left']}, {params['pad_d_right']}, "
                    f"{params['pad_h_left']}, {params['pad_h_right']}, "
                    f"{params['pad_w_left']}, {params['pad_w_right']}"
                    f"}}{comma}\n"
                )
            f.write("};\n")

    print(
        f"Extracted {len(test_params)} {pooling_dim} test parameters from {config_file} -> {output_file}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract pooling test parameters from config JSON"
    )
    parser.add_argument("--config_file", required=True, help="Input config JSON file")
    parser.add_argument(
        "--output_file", required=True, help="Output test parameters file"
    )
    parser.add_argument(
        "--pooling_dim",
        default="2d",
        choices=["2d", "3d"],
        help="Pooling dimension (2d or 3d)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        return 1

    extract_test_params(args.config_file, args.output_file, args.pooling_dim)
    return 0


if __name__ == "__main__":
    exit(main())
