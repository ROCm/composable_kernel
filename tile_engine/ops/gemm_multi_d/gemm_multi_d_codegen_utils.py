# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-

"""
Mappings and utility functions for kernel code generation.
"""

DATA_TYPE_MAP = {
    "fp32": "float",
    "fp16": "ck_tile::half_t",
    "bf16": "ck_tile::bf16_t",
    "int8": "ck_tile::int8_t",
    "fp8": "ck_tile::fp8_t",
    "bf8": "ck_tile::bf8_t",
    "int4": "ck_tile::pk_int4_t",
    "int32": "ck_tile::int32_t",
}

LAYOUT_MAP = {
    "r": "ck_tile::tensor_layout::gemm::RowMajor",
    "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
}


# TODO THIS IS NOT SUPPORTED FOR MULTI D AS OF NOW
# DEFAULT_EPILOGUE = """
#             using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
#                                 ck_tile::DefaultGemm2DEpilogueProblem<ADataType,
#                                                                       BDataType,
#                                                                       AccDataType,
#                                                                       CDataType,
#                                                                       CLayout,
#                                                                       kPadM,
#                                                                       kPadN,
#                                                                       WarpTileM,
#                                                                       WarpTileN,
#                                                                       WarpTileK,
#                                                                       UniversalGemmProblem::TransposeC,
#                                                                       true,
#                                                                       memory_operation>>;
# """

CSHUFFLE_EPILOGUE = """
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                            ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             DsDataType,
                                                             AccDataType,
                                                             EDataType,
                                                             DsLayout,
                                                             ELayout,
                                                             CDEElementWise,
                                                             TilePartitioner::MPerBlock,
                                                             TilePartitioner::NPerBlock,
                                                             WarpM,
                                                             WarpN,
                                                             WarpTileM,
                                                             WarpTileN,
                                                             WarpTileK,
                                                             UniversalGemmProblem::TransposeC,
                                                             memory_operation>>;
"""

PIPELINE_MAP = {
    "mem": ["ck_tile::BaseGemmPipelineAgBgCrMem", "ck_tile::GemmPipelineAgBgCrMem"],
    "compv3": [
        "ck_tile::BaseGemmPipelineAgBgCrCompV3",
        "ck_tile::GemmPipelineAgBgCrCompV3",
    ],
    "compv4": [
        "ck_tile::BaseGemmPipelineAgBgCrCompV4",
        "ck_tile::GemmPipelineAgBgCrCompV4",
    ],
}

SCHEDULER_MAP = {
    "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
    "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
}

# EPILOGUE_MAP = {"default": DEFAULT_EPILOGUE, "cshuffle": CSHUFFLE_EPILOGUE}

EPILOGUE_MAP = {"cshuffle": CSHUFFLE_EPILOGUE}


def BOOL_MAP(b_):
    return {True: "true", False: "false"}[bool(b_)]


# Can add some more supported combinations
warp_tile_supported_combinations = {
    "gfx90a": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "fp8_fp8_fp16": [[32, 32, 16], [32, 32, 32]],
        "bf8_bf8_fp16": [[32, 32, 16], [32, 32, 32]],
    },
    "gfx942": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "fp8_fp8_fp16": [[32, 32, 16], [32, 32, 32], [16, 16, 32], [16, 16, 64]],
        "bf8_bf8_fp16": [[32, 32, 16], [32, 32, 32], [16, 16, 64], [16, 16, 32]],
        "int8_int8_int32": [[16, 16, 32], [32, 32, 16]],
    },
    "gfx950": {
        "fp16_fp16_fp16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "bf16_bf16_bf16": [
            [32, 32, 8],
            [16, 16, 16],
            [32, 32, 16],
            [16, 16, 32],
            [4, 64, 16],
            [64, 4, 16],
        ],
        "fp8_fp8_fp16": [
            [32, 32, 16],
            [32, 32, 32],
            [16, 16, 32],
            [16, 16, 64],
            [16, 16, 128],
            [32, 32, 64],
        ],
        "bf8_bf8_fp16": [
            [32, 32, 16],
            [32, 32, 32],
            [16, 16, 64],
            [16, 16, 32],
            [16, 16, 128],
            [32, 32, 64],
        ],
    },
}

# Remove some unsupported combinations
trait_unsupported_combinations = {
    ("compv3", "cshuffle", "interwave"),
    ("compv3", "default", "interwave"),
    ("compv4", "cshuffle", "interwave"),
    ("compv4", "default", "interwave"),
}


ELEMENT_SIZE_MAP = {
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "fp8": 1,
    "bf8": 1,
    "int4": 0.5,
    "int32": 4,
}


def element_size(data_type: str) -> float:
    """Calculate the size (in bytes) of a single element for given data type."""
    data_type = data_type.lower()
    if data_type not in ELEMENT_SIZE_MAP:
        raise ValueError(f"Unsupported data type: {data_type}")
    return ELEMENT_SIZE_MAP[data_type]
