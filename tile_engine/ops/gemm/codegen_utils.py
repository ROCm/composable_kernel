# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-

"""
Mappings and utility functions for kernel code generation.
"""

import subprocess
import re
from functools import lru_cache

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

DEFAULT_EPILOGUE = """
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
                                ck_tile::DefaultGemm2DEpilogueProblem<ADataType,
                                                                      BDataType,
                                                                      AccDataType,
                                                                      CDataType,
                                                                      CLayout,
                                                                      kPadM,
                                                                      kPadN,
                                                                      WarpTileM,
                                                                      WarpTileN,
                                                                      WarpTileK,
                                                                      UniversalGemmProblem::TransposeC,
                                                                      true,
                                                                      memory_operation>>;
"""

CSHUFFLE_EPILOGUE = """
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                            ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             ck_tile::tuple<>,
                                                             AccDataType,
                                                             CDataType,
                                                             ck_tile::tuple<>,
                                                             CLayout,
                                                             ck_tile::element_wise::PassThrough,
                                                             GemmPipelineProblem::kBlockSize,
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
HOT_LOOP_FALSE = """
            if(tail_num == ck_tile::TailNumber::Full)
            {
                RunSplitk(ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_num == ck_tile::TailNumber::Odd)
            {
                RunSplitk(ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_num == ck_tile::TailNumber::Even)
            {
                RunSplitk(ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Even>{});
            }
            else
            {
                throw std::runtime_error("Num K loop must be larger than number of prefetech stages.");
            }
"""
RUN_MEM = """
            // Handle One and Full cases directly
            if (tail_num == ck_tile::TailNumber::One) {
                RunSplitk(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
            } else if (tail_num == ck_tile::TailNumber::Full) {
                RunSplitk(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            
            auto check_tail = [&](auto... TNs) {
                ([&]{
                    if constexpr(BaseGemmPipeline::PrefetchStages > static_cast<int>(decltype(TNs)::value)) {
                        if(tail_num == decltype(TNs)::value) {
                            RunSplitk(ck_tile::bool_constant<true>{},
                                    ck_tile::integral_constant<ck_tile::TailNumber, decltype(TNs)::value>{});
                        }
                    }
                }(), ...);
            };

            check_tail(
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{},
                ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{}
            );
"""

RUN_COMPV3 = """
            if(tail_num == ck_tile::TailNumber::Full)
            {
                RunSplitk(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_num == ck_tile::TailNumber::Odd)
            {
                RunSplitk(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_num == ck_tile::TailNumber::Even)
            {
                RunSplitk(ck_tile::bool_constant<true>{},
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
                RunSplitk(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
            }
            else
            {
                RunSplitk(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
            }
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

EPILOGUE_MAP = {"default": DEFAULT_EPILOGUE, "cshuffle": CSHUFFLE_EPILOGUE}

HOT_LOOP_TRUE = {"mem": RUN_MEM, "compv3": RUN_COMPV3, "compv4": RUN_COMPV4}


def BOOL_MAP(b_):
    return {True: "true", False: "false"}[bool(b_)]


# To Do: add some more supported combinations
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

# To Do: remove some unsupported combinations
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


GPU_NAME_PATTERN = re.compile(r"Name:\s*(gfx\d+\w*)")


@lru_cache(maxsize=1)
def get_gpu_name_by_id(gpu_id: int = 0) -> str:
    """Retrieve GPU name (e.g. gfx90a) by device ID"""
    try:
        output = subprocess.check_output(
            ["rocminfo"], text=True, stderr=subprocess.PIPE, timeout=5
        )
        if matches := GPU_NAME_PATTERN.finditer(output):
            gpu_list = [m.group(1) for m in matches]
            return gpu_list[gpu_id] if gpu_id < len(gpu_list) else ""

        return ""

    except subprocess.CalledProcessError as e:
        print(f"GPU query failed (exit {e.returncode}): {e.stderr.strip()}")
    except FileNotFoundError:
        print("ROCm tools not installed (requires rocminfo)")
    except subprocess.TimeoutExpired:
        print("GPU query timeout (5s)")
    except Exception as e:
        print(f"GPU detection error: {str(e)}")

    return ""
