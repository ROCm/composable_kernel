# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-

"""
generate kernel instances to speed up compilation
"""

import argparse
import itertools
from pathlib import Path
from typing import List, Optional
from json_config import GemmConfig, RangeConfigParam
from codegen_utils import (
    DATA_TYPE_MAP,
    LAYOUT_MAP,
    DEFAULT_EPILOGUE,
    CSHUFFLE_EPILOGUE,
    HOT_LOOP_FALSE,
    RUN_MEM,
    RUN_COMPV3,
    RUN_COMPV4,
    PIPELINE_MAP,
    SCHEDULER_MAP,
    EPILOGUE_MAP,
    HOT_LOOP_TRUE,
    BOOL_MAP,
    warp_tile_supported_combinations,
    trait_unsupported_combinations,
    element_size,
    get_gpu_name_by_id,
)
import logging

logging.basicConfig(level=logging.INFO)


class GemmCodeGenerator:
    """GEMM (General Matrix Multiplication) code generator."""

    def __init__(
        self, output_dir: str, user_provided_config: Optional[GemmConfig] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if user_provided_config is not None:
            self.config = user_provided_config
        else:
            config_path = (
                Path(__file__).resolve().parent / "configs" / "default_config.json"
            )
            self.config = GemmConfig.from_json(config_path)

        self.valid_trait_names: List[str] = []
        self.valid_trait_tile_combinations: map[str, list[tuple[int]]] = {}

    def list_all_trait_names(self):
        """List all possible kernel trait names into file."""
        w_p = Path(self.output_dir)
        file_path = w_p / "gemm_instance_blobs.txt"
        self._generate_all_traits()
        self._get_valid_trait_tile_combinations()
        file_range_map = {}
        # Write all file paths to the header file
        files_listed = 0
        with file_path.open("w") as f:
            # Core files
            core_files = [
                "gemm_common.hpp",
                "gemm_instances.hpp",
                "gemm_dispatcher.hpp",
            ]
            for core_file in core_files:
                f.write(str(w_p / core_file) + "\n")
                files_listed += 1

            # Trait header files
            for trait in self.valid_trait_names:
                trait_file = f"gemm_{trait}.hpp"
                f.write(str(w_p / trait_file) + "\n")
                files_listed += 1
            file_name = set()
            # Instance source files
            for trait, tile_valid_params in self.valid_trait_tile_combinations.items():
                start_idx = files_listed
                for tile in tile_valid_params:
                    for (
                        tile_m,
                        tile_n,
                        tile_k,
                        warp_m,
                        warp_n,
                        warp_k,
                        _,
                        _,
                        _,
                    ) in tile:
                        instance_name = f"gemm_{trait}_{tile_m}x{tile_n}x{tile_k}_{warp_m}x{warp_n}x{warp_k}.cpp"

                        if instance_name not in file_name:
                            file_name.add(instance_name)
                            f.write(str(w_p / instance_name) + "\n")
                            files_listed += 1

                file_range_map[trait] = (start_idx, files_listed)

        file_path = w_p / "gemm_instance_blobs_range.txt"
        with file_path.open("w") as f:
            for name, ranges in file_range_map.items():
                s, l = ranges
                f.write(name + " " + f"{s}" + " " + f"{l}" + "\n")

    def _generate_all_traits(self):
        """Generate all possible kernel traits names."""
        params = ["pipeline", "epilogue", "scheduler", "pad_m", "pad_n", "pad_k"]

        # Generate all unique_combinations
        _unique = set(
            itertools.product(
                *[getattr(self.config.trait_config, param).values for param in params]
            )
        )

        for combo in _unique:
            pipeline, epilogue, scheduler, pad_m, pad_n, pad_k = combo
            current_combination = (pipeline, epilogue, scheduler)

            if current_combination not in trait_unsupported_combinations:
                trait_name = (
                    f"{pipeline}_{epilogue}_{scheduler}_"
                    f"{BOOL_MAP(pad_m)}_{BOOL_MAP(pad_n)}_{BOOL_MAP(pad_k)}"
                )
                self.valid_trait_names.append(trait_name)
            else:
                logging.debug(f"Invalid combination: {pipeline}-{epilogue}-{scheduler}")

    def generate_all_instance_files(self):
        """Generate all kernel instances files."""
        self._generate_common_header_file()
        self._generate_all_trait_files()
        self._generate_dispatcher_file()

    def _generate_common_header_file(self):
        """Generate common header file with datatypes and layout."""

        # Determine appropriate accumulation type based on input types
        a_type = self.config.problem.datatype_map["matrix_a"]
        b_type = self.config.problem.datatype_map["matrix_b"]
        c_type = self.config.problem.datatype_map["matrix_c"]

        if a_type in ["int8", "int4"] and b_type in ["int8", "int4"]:
            acc_type = "ck_tile::int32_t"
        else:
            acc_type = "float"

        content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"

// Data types
using ADataType = {DATA_TYPE_MAP[self.config.problem.datatype_map["matrix_a"]]};
using BDataType = {DATA_TYPE_MAP[self.config.problem.datatype_map["matrix_b"]]};
using AccDataType = {acc_type};
using CDataType = {DATA_TYPE_MAP[self.config.problem.datatype_map["matrix_c"]]};

// Layout configurations
using ALayout = {LAYOUT_MAP[self.config.problem.layout_map["matrix_a"]]};
using BLayout = {LAYOUT_MAP[self.config.problem.layout_map["matrix_b"]]};
using CLayout = {LAYOUT_MAP[self.config.problem.layout_map["matrix_c"]]};
"""

        (self.output_dir / "gemm_common.hpp").write_text(content)

    def _generate_all_trait_files(self):
        """Generate all kernel traits into files."""
        if not self.valid_trait_names:
            self._generate_all_traits()
            self._get_valid_trait_tile_combinations()
        for trait in self.valid_trait_names:
            self._generate_trait_file(trait)
        self._generate_instantiation_source_files()
        self._generate_common_instance_header_file()

    def _generate_trait_file(self, trait: str):
        """Generate a trait with all tile/warp combinations."""
        pipeline, epilogue, scheduler, pad_m, pad_n, pad_k = trait.split("_")
        filename = f"gemm_{trait}.hpp"

        content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "gemm_common.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/host.hpp"

namespace {trait} {{
"""
        # Add template struct with configuration
        content += self._generate_kernel_struct(
            pipeline, epilogue, scheduler, pad_m, pad_n, pad_k
        )

        content += f"\n}} // namespace {trait}\n"
        (self.output_dir / filename).write_text(content)

    def _generate_kernel_struct(
        self,
        pipeline: str,
        epilogue: str,
        scheduler: str,
        pad_m: str,
        pad_n: str,
        pad_k: str,
    ) -> str:
        """Generate the code block of kernel struct"""
        return f"""

template <int TileM, int TileN, int TileK,
          int WarpM, int WarpN, int WarpK,
          int WarpTileM, int WarpTileN, int WarpTileK,
          bool structured_sparsity>
struct GemmKernel {{
    static constexpr bool kPadM = {pad_m};
    static constexpr bool kPadN = {pad_n};
    static constexpr bool kPadK = {pad_k};

    static float launch(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& stream) {{
        static constexpr bool permuteA = false;
        static constexpr bool permuteB = false;
        static constexpr bool DoubleSmemBuffer ={"true" if pipeline == "compv4" else "false"};
        static constexpr bool TransposeC = false;

        static constexpr int kBlockPerCu                         = 1;
        static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
        static constexpr ck_tile::index_t TileParitionerM01      = 4;

        using GemmShape =
            ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
                                   ck_tile::sequence<WarpM, WarpN, WarpK>,
                                   ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
                                   permuteA,
                                   permuteB>;


        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                      TileParitionerGroupNum,
                                                      TileParitionerM01>;

        using Traits  =
            ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                             ALayout, BLayout, CLayout, TransposeC, structured_sparsity>;

        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        using BaseGemmPipeline = {PIPELINE_MAP[pipeline][0]}<GemmPipelineProblem>;

        const ck_tile::index_t k_grain     = args.k_batch * TileK;
        const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * TileK;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

        float ave_time{{0}};

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr auto scheduler      = {SCHEDULER_MAP[scheduler]};
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem =
                ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                      BDataType,
                                                      AccDataType,
                                                      GemmShape,
                                                      GemmUniversalTraits,
                                                      scheduler,
                                                      has_hot_loop_v,
                                                      tail_number_v>;

            using GemmPipeline = {PIPELINE_MAP[pipeline][1]}<UniversalGemmProblem>;
            {EPILOGUE_MAP[epilogue]}
            using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
            auto kargs   = Kernel::MakeKernelArgs(args);

            const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
            constexpr dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {{
                throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
            }}

            if(stream.log_level_ > 0)
            {{
                std::cout << "Launching kernel with args:"
                      << " grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                      << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                      << std::endl;
            }}

            if(stream.flush_cache_)
            {{
                std::cout << "Flushing cache..." << std::endl;
                static constexpr ck_tile::index_t APackedSize =
                    std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
                static constexpr ck_tile::index_t BPackedSize =
                    std::is_same_v<BDataType, ck_tile::pk_int4_t> ? 2 : 1;
                
                auto is_row_major = [](auto layout_) {{
                    return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
                                                 ck_tile::tensor_layout::gemm::RowMajor>>{{}};
                }};

                ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                    args.M, args.K, args.stride_A, is_row_major(ALayout{{}})));
                ck_tile::HostTensor<BDataType> b_n(ck_tile::host_tensor_descriptor(
                    args.K, args.N, args.stride_B, is_row_major(BLayout{{}})));

                auto size_a_buffer = a_m.get_element_space_size_in_bytes() / APackedSize;
                auto size_b_buffer = b_n.get_element_space_size_in_bytes() / BPackedSize;

                ck_tile::RotatingMemWrapper<ADataType, BDataType> rotating_mem(
                    kargs.as_ptr[0], kargs.bs_ptr[0], stream.rotating_count_, size_a_buffer, size_b_buffer);
                rotating_mem.Print();

                auto run_flush_cache = [&]() {{
                    // flush icache
                    ck_tile::flush_icache();
                    // rotating mem
                    rotating_mem.Next();
                    // clear c mem
                    if(args.k_batch > 1)
                        hipGetErrorString(hipMemsetAsync(
                            args.e_ptr, 0, args.M * args.N * sizeof(CDataType), stream.stream_id_));
                }};
                ave_time = ck_tile::launch_kernel_time_mask(
                    stream,
                    run_flush_cache,
                    ck_tile::make_kernel<blocks.x, kBlockPerCu>(
                        Kernel{{}}, grids, blocks, 0, kargs));
            }}
            else{{
                ave_time = ck_tile::launch_kernel(stream,
                                          ck_tile::make_kernel<blocks.x, kBlockPerCu>(
                                              Kernel{{}}, grids, blocks, 0, kargs));
            }}
            return ave_time;

        }};

        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {{
            if(args.k_batch == 1) {{
                Run(has_hot_loop_,
                    tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                            ck_tile::memory_operation_enum::set>{{}});
            }} else {{
                Run(has_hot_loop_,
                    tail_number_,
                    ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                            ck_tile::memory_operation_enum::atomic_add>{{}});
            }}
        }};

        if(has_hot_loop) {{
            {HOT_LOOP_TRUE[pipeline]}
        }} else {{
            {HOT_LOOP_FALSE}
        }}

        return ave_time;
    }}

    static std::string get_name() {{
        return std::string("gemm_") + std::to_string(TileM) + "x" + std::to_string(TileN) + "x" + std::to_string(TileK) +
                "_" + std::to_string(WarpM) + "x" + std::to_string(WarpN) + "x" + std::to_string(WarpK) + "_" +
                std::to_string(WarpTileM) + "x" + std::to_string(WarpTileN) + "x" + std::to_string(WarpTileK) + "_" +
                "{pad_m}" + "_" +
                "{pad_n}" + "_" +
                "{pad_k}" + "_" +
                "{pipeline}" + "_" +
                "{epilogue}" + "_" +
                "{scheduler}";
    }}
}};
"""

    def _generate_common_instance_header_file(self):
        """Generate common instance header into file."""
        content = """// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
"""
        for trait in self.valid_trait_names:
            content += f'#include "gemm_{trait}.hpp"\n'
        (self.output_dir / "gemm_instances.hpp").write_text(content)

    def is_tile_valid(self, tile: tuple, trait: str) -> bool:
        """Check if the tile configuration is valid for the given trait."""
        (
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            warp_k,
            warp_tile_m,
            warp_tile_n,
            warp_tile_k,
        ) = tile
        pipeline, *_ = trait.split("_")

        # Parameter validity check
        invalid_params = []
        if (warp_m, warp_n, warp_k) not in [(1, 4, 1), (2, 2, 1), (4, 1, 1)]:
            invalid_params.append(
                f"warp_m({warp_m}) * warp_n({warp_n}) * warp_k({warp_k})"
            )
        if (warp_m * warp_tile_m) == 0:
            invalid_params.append(f"warp_m({warp_m}) * warp_tile_m({warp_tile_m})")
        if (warp_n * warp_tile_n) == 0:
            invalid_params.append(f"warp_n({warp_n}) * warp_tile_n({warp_tile_n})")
        if (warp_k * warp_tile_k) == 0:
            invalid_params.append(f"warp_k({warp_k}) * warp_tile_k({warp_tile_k})")

        if invalid_params:
            logging.debug(
                f"Trait: [{trait}], Invalid warp configuration: {', '.join(invalid_params)}. "
                f"Parameter combination: warp=({warp_m},{warp_n},{warp_k}), "
                f"warp_tile=({warp_tile_m},{warp_tile_n},{warp_tile_k})"
            )
            return False
        # Dimension alignment check
        alignment_issues = []
        if tile_m % (warp_m * warp_tile_m) != 0:
            alignment_issues.append(
                f"tile_m({tile_m}) % [{warp_m}x{warp_tile_m}] = {tile_m % (warp_m * warp_tile_m)}"
            )
        if tile_n % (warp_n * warp_tile_n) != 0:
            alignment_issues.append(
                f"tile_n({tile_n}) % [{warp_n}x{warp_tile_n}] = {tile_n % (warp_n * warp_tile_n)}"
            )
        if tile_k % (warp_k * warp_tile_k) != 0:
            alignment_issues.append(
                f"tile_k({tile_k}) % [{warp_k}x{warp_tile_k}] = {tile_k % (warp_k * warp_tile_k)}"
            )

        if alignment_issues:
            logging.debug(
                f"Trait: [{trait}], Dimension alignment failed: {', '.join(alignment_issues)}. "
                f"Tile dimensions {tile_m}x{tile_n}x{tile_k} must be divisible by "
                f"[warp]: {warp_m}x{warp_n}x{warp_k} x [warp_tile]: {warp_tile_m}x{warp_tile_n}x{warp_tile_k}"
            )
            return False

        # LDS capacity verification
        matrix_a_size = (tile_m * tile_k) * element_size(
            self.config.problem.datatype_map["matrix_a"]
        )
        matrix_b_size = (tile_n * tile_k) * element_size(
            self.config.problem.datatype_map["matrix_b"]
        )
        total_tile_in_lds = matrix_a_size + matrix_b_size

        max_tile_size = 2**15 if pipeline == "compv4" else 2**16

        if total_tile_in_lds > max_tile_size:
            logging.debug(
                f"LDS capacity exceeded [{trait}]: Total required {total_tile_in_lds:,}B ({total_tile_in_lds / 1024:.1f}KB) > "
                f"maximum allowed {max_tile_size:,}B ({max_tile_size / 1024}KB). Breakdown:\n"
                f"- Matrix A ({self.config.problem.datatype_map['matrix_a']}): {tile_m}x{tile_k} = {matrix_a_size:,}B\n"
                f"- Matrix B ({self.config.problem.datatype_map['matrix_b']}): {tile_n}x{tile_k} = {matrix_b_size:,}B"
            )
            return False

        # Warp combination validation
        warp_tile_key = f"{self.config.problem.datatype_map['matrix_a']}_{self.config.problem.datatype_map['matrix_b']}_{self.config.problem.datatype_map['matrix_c']}"
        current_combination = [warp_tile_m, warp_tile_n, warp_tile_k]

        gpu_name = get_gpu_name_by_id(0)

        gpu_warp_tile_key = warp_tile_supported_combinations.get(gpu_name, {})
        if not gpu_warp_tile_key:
            logging.debug(
                f"Trait: [{trait}], No valid warp tile combinations found for {gpu_name}/{warp_tile_key}, skip this check."
            )
            return False

        allowed_combinations = gpu_warp_tile_key.get(warp_tile_key, [])
        if not allowed_combinations:
            logging.debug(
                f"Trait: [{trait}], No valid warp tile combinations found for {gpu_name}/{warp_tile_key}, skip this check."
            )
            return False

        if current_combination not in allowed_combinations:
            logging.debug(
                f"Trait: [{trait}], Invalid warp combination: {current_combination} not in allowed list. "
                f"Valid combinations for data type '{warp_tile_key}': {allowed_combinations}"
            )
            return False

        return True

    def _get_valid_trait_tile_combinations(self):
        def get_tile_value(tile_param):
            return (
                tile_param.generate_candidates()
                if isinstance(tile_param, RangeConfigParam)
                else tile_param.values
            )

        tile_group = list(
            itertools.product(
                get_tile_value(self.config.tile_config.tile_m),
                get_tile_value(self.config.tile_config.tile_n),
                get_tile_value(self.config.tile_config.tile_k),
            )
        )

        warp_group = list(
            itertools.product(
                get_tile_value(self.config.tile_config.warp_m),
                get_tile_value(self.config.tile_config.warp_n),
                get_tile_value(self.config.tile_config.warp_k),
            )
        )

        warp_tile_group = list(
            itertools.product(
                get_tile_value(self.config.tile_config.warp_tile_m),
                get_tile_value(self.config.tile_config.warp_tile_n),
                get_tile_value(self.config.tile_config.warp_tile_k),
            )
        )

        tile_params = {
            t + w + wt for t in tile_group for w in warp_group for wt in warp_tile_group
        }

        for trait in self.valid_trait_names:
            tile_valid_params = [
                tile for tile in tile_params if self.is_tile_valid(tile, trait)
            ]

            if trait not in self.valid_trait_tile_combinations:
                self.valid_trait_tile_combinations[trait] = []
            self.valid_trait_tile_combinations[trait].append(tile_valid_params)

    def _generate_instantiation_source_files(self):
        """Generate kernel instance instantiation source files"""
        tile_map = {}
        for trait, tile_valid_params in self.valid_trait_tile_combinations.items():
            for tile in tile_valid_params:
                for (
                    tile_m,
                    tile_n,
                    tile_k,
                    warp_m,
                    warp_n,
                    warp_k,
                    warp_tile_m,
                    warp_tile_n,
                    warp_tile_k,
                ) in tile:
                    key = f"{tile_m}x{tile_n}x{tile_k}x{warp_m}x{warp_n}x{warp_k}"
                    value = f"{warp_tile_m}x{warp_tile_n}x{warp_tile_k}"
                    if key not in tile_map:
                        tile_map[key] = set()
                    tile_map[key].add(value)

        files_listed = 0
        for trait, _ in self.valid_trait_tile_combinations.items():
            for block_tile, warp_tiles in tile_map.items():
                tile_m, tile_n, tile_k, warp_m, warp_n, warp_k = map(
                    int, block_tile.split("x")
                )

                content = f"""
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.


#include "gemm_{trait}.hpp" 

"""
                for warp_tile in warp_tiles:
                    warp_tile_m, warp_tile_n, warp_tile_k = map(
                        int, warp_tile.split("x")
                    )

                    sparse = (
                        self.config.problem.datatype_map["matrix_a"] == "fp16"
                        and self.config.problem.datatype_map["matrix_b"] == "fp16"
                        and self.config.problem.datatype_map["matrix_c"] == "fp16"
                        and (
                            (
                                warp_tile_m == 32
                                and warp_tile_n == 32
                                and warp_tile_k == 16
                            )
                            or (
                                warp_tile_m == 16
                                and warp_tile_n == 16
                                and warp_tile_k == 32
                            )
                        )
                    )
                    if sparse:
                        files_listed = files_listed + 1
                        content = (
                            content
                            + f"""
template struct {trait}::GemmKernel<{tile_m}, {tile_n}, {tile_k}, {warp_m}, {warp_n}, {warp_k}, {warp_tile_m}, {warp_tile_n}, {warp_tile_k}, true>;"""
                        )
                    files_listed = files_listed + 1
                    content = (
                        content
                        + f"""
template struct {trait}::GemmKernel<{tile_m}, {tile_n}, {tile_k}, {warp_m}, {warp_n}, {warp_k}, {warp_tile_m}, {warp_tile_n}, {warp_tile_k}, false>;"""
                    )
                content += f"""
"""
                (
                    self.output_dir
                    / f"gemm_{trait}_{tile_m}x{tile_n}x{tile_k}_{warp_m}x{warp_n}x{warp_k}.cpp"
                ).write_text(content)
        print(f"Generated {files_listed} kernel instances in total.")

    def _generate_dispatcher_file(self):
        """Generate the code block of dispatch mechanism."""
        content = """
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <unordered_map>
#include <functional>
#include <vector>

#include "gemm_common.hpp"
#include "gemm_instances.hpp"

/// @brief Defines the configuration parameters for a GEMM operation, enabling the selection of a
/// specific kernel instance based on the provided settings.
struct KernelTraits
{
    /// @brief The name of the pipeline.
    std::string pipeline;
    /// @brief The name of the scheduler (e.g., "intrawave", "interwave").
    std::string scheduler;
    /// @brief The name of the epilogue (e.g., "cshuffle", "default").
    std::string epilogue;
    /// @brief Indicates whether padding is applied to the M dimension.
    bool pad_m;
    /// @brief Indicates whether padding is applied to the N dimension.
    bool pad_n;
    /// @brief Indicates whether padding is applied to the K dimension.
    bool pad_k;
};

struct GemmDispatcher {
    static auto& get_kernel_map() {
        // Use a static local variable
        static std::unordered_map<
            std::string,
            std::vector<std::function<std::tuple<std::string, float>(ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>>>
            kernel_map;
        return kernel_map;
    }

    static void init([[maybe_unused]]bool structured_sparsity) {
        auto& kernel_map = get_kernel_map();
        if(!kernel_map.empty()) return;
        \n"""

        for trait, tile_valid_params in self.valid_trait_tile_combinations.items():
            content += f"""         kernel_map["{trait}"] = {{"""
            for _, tile in enumerate(tile_valid_params):
                for j in range(len(tile)):
                    (
                        tile_m,
                        tile_n,
                        tile_k,
                        warp_m,
                        warp_n,
                        warp_k,
                        warp_tile_m,
                        warp_tile_n,
                        warp_tile_k,
                    ) = tile[j]
                    content += f"""[=](ck_tile::GemmHostArgs& args, const ck_tile::stream_config& stream) {{ """
                    content += f""" 
                                    if(structured_sparsity){{  // SMFMA"""
                    sparse = (
                        self.config.problem.datatype_map["matrix_a"] == "fp16"
                        and self.config.problem.datatype_map["matrix_b"] == "fp16"
                        and self.config.problem.datatype_map["matrix_c"] == "fp16"
                        and (
                            (
                                warp_tile_m == 32
                                and warp_tile_n == 32
                                and warp_tile_k == 16
                            )
                            or (
                                warp_tile_m == 16
                                and warp_tile_n == 16
                                and warp_tile_k == 32
                            )
                        )
                    )
                    content += f"""
                                        return run_kernel<{trait}::GemmKernel<{tile_m}, {tile_n}, {tile_k}, {warp_m}, {warp_n}, {warp_k}, {warp_tile_m}, {warp_tile_n}, {warp_tile_k}, {BOOL_MAP(sparse)}>>(args, stream);"""
                    content += f"""
                                    }} else {{"""
                    content += f"""
                                        return run_kernel<{trait}::GemmKernel<{tile_m}, {tile_n}, {tile_k}, {warp_m}, {warp_n}, {warp_k}, {warp_tile_m}, {warp_tile_n}, {warp_tile_k}, {BOOL_MAP(False)}>>(args, stream);"""
                    content += f"""
                                    }} """

                    if j == len(tile) - 1:
                        content += f"""
                                }} """
                    else:
                        content += f"""
                                }}, """
            content += f"""
            }};\n """

        content += """    }

    template <typename Kernel>
    static std::tuple<std::string, float> run_kernel(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& stream)
    {
        std::string name = Kernel::get_name();
        float avg_time = Kernel::launch(args, stream);
        
        return std::make_tuple(name, avg_time);
    }
    
    
    static auto dispatch(bool structured_sparsity, const KernelTraits& trait) {
        init(structured_sparsity);
        const std::string key = assemble_key(trait);
        auto& kernel_map = get_kernel_map();
        if(auto it = kernel_map.find(key); it != kernel_map.end())
        {
            return it->second;
        }
        throw std::runtime_error("No suitable kernel found: " + key);
    }

private:
    static std::string assemble_key(const KernelTraits &trait) {
        return std::string(trait.pipeline) + "_" +
               trait.epilogue + "_" +
               trait.scheduler + "_" +
               (trait.pad_m ? "true" : "false") + "_" +
               (trait.pad_n ? "true" : "false") + "_" +
               (trait.pad_k ? "true" : "false");
    }
};

"""
        (self.output_dir / "gemm_dispatcher.hpp").write_text(content)


def do_list_blobs(
    args: argparse.Namespace, user_provide_config: Optional[GemmConfig] = None
):
    generator = GemmCodeGenerator(args.working_path, user_provide_config)
    generator.list_all_trait_names()


def do_gen_blobs(
    args: argparse.Namespace, user_provide_config: Optional[GemmConfig] = None
):
    generator = GemmCodeGenerator(args.working_path, user_provide_config)
    generator.generate_all_instance_files()


def main(args):
    gemm_config = (
        GemmConfig.from_json(args.config_json, args.datatype, args.layout)
        if args.config_json is not None
        else args.config_json
    )

    if args.list_blobs:
        do_list_blobs(args, gemm_config)
    elif args.gen_blobs:
        do_gen_blobs(args, gemm_config)
    else:
        logging.warning(
            "No mode specified (use --list_blobs or --gen_blobs). Generating by default..."
        )
        do_gen_blobs(args, gemm_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm kernel",
    )
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="The path where all the blobs are going to be generated",
    )
    parser.add_argument(
        "-j",
        "--config_json",
        required=False,
        help="Path to the json which contains the configurations that user provide",
    )
    parser.add_argument(
        "-d",
        "--datatype",
        required=True,
        help="Specify what datatype to use for the kernel generation, e.g. fp16, bf16, int8, fp8, bf8",
    )
    parser.add_argument(
        "-ly",
        "--layout",
        required=True,
        help="Specify what layout to use for the kernel generation, e.g. rcr, rrr",
    )
    parser.add_argument(
        "-l",
        "--list_blobs",
        action="store_true",
        help="List all kernel instances to file",
    )
    parser.add_argument(
        "-g",
        "--gen_blobs",
        action="store_true",
        help="Generate all kernel instances into different files",
    )

    args = parser.parse_args()

    main(args)
