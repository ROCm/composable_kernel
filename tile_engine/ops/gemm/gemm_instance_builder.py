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
    get_gpu_name_by_id
)
import logging


class GemmCodeGenerator:
    """GEMM (General Matrix Multiplication) code generator."""

    def __init__(self, output_dir: str,
                 user_provided_config: Optional[GemmConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if user_provided_config is not None:
            self.config = user_provided_config
        else:
            config_path = Path(__file__).resolve().parent / \
                "configs" / "default_config.json"
            self.config = GemmConfig.from_json(config_path)

        self.all_trait_names: List[str] = []

    def list_all_trait_names(self):
        """List all possible kernel trait names into file."""
        w_p = Path(self.output_dir)
        list_p = w_p / 'gemm_instance_blobs.txt'
        self._generate_all_traits()

        # Write all file paths to the list file
        with list_p.open('w') as list_f:
            list_f.write(str(w_p / "gemm_common.hpp") + "\n")
            list_f.write(str(w_p / "gemm_instances.hpp") + "\n")
            list_f.write(str(w_p / "gemm_dispatcher.hpp") + "\n")
            for trait in self.all_trait_names:
                list_f.write(str(w_p / f"gemm_{trait}.hpp") + "\n")

    def _generate_all_traits(self):
        """Generate all possible kernel traits names."""
        params = [
            "pipeline",
            "epilogue",
            "scheduler",
            "pad_m",
            "pad_n",
            "pad_k"]

        # Generate all unique_combinations
        _unique = set(itertools.product(*[
            getattr(self.config.trait_config, param).values
            for param in params
        ]))

        for combo in _unique:
            pipeline, epilogue, scheduler, pad_m, pad_n, pad_k = combo
            current_combination = (pipeline, epilogue, scheduler)

            if current_combination not in trait_unsupported_combinations:
                trait_name = (
                    f"{pipeline}_{epilogue}_{scheduler}_"
                    f"{BOOL_MAP(pad_m)}_{BOOL_MAP(pad_n)}_{BOOL_MAP(pad_k)}"
                )
                self.all_trait_names.append(trait_name)
            else:
                logging.warning(
                    f"Invalid combination: {pipeline}-{epilogue}-{scheduler}"
                )

    def generate_all_instance_files(self):
        """Generate all kernel instances files."""
        self._generate_common_header_file()
        self._generate_all_trait_files()
        self._generate_dispatcher_file()

    def _generate_common_header_file(self):
        """Generate common header file with datatypes and layout."""

        content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core.hpp"

// Data types
using ADataType = {DATA_TYPE_MAP[self.config.problem.datatype_map['matrix_a']]};
using BDataType = {DATA_TYPE_MAP[self.config.problem.datatype_map['matrix_b']]};
using AccDataType = float;
using CDataType = {DATA_TYPE_MAP[self.config.problem.datatype_map['matrix_c']]};

// Layout configurations
using ALayout = {LAYOUT_MAP[self.config.problem.layout_map['matrix_a']]};
using BLayout = {LAYOUT_MAP[self.config.problem.layout_map['matrix_b']]};
using CLayout = {LAYOUT_MAP[self.config.problem.layout_map['matrix_c']]};
"""

        (self.output_dir / "gemm_common.hpp").write_text(content)

    def _generate_all_trait_files(self):
        """Generate all kernel traits into files."""
        if not self.all_trait_names:
            self._generate_all_traits()
        for trait in self.all_trait_names:
            self._generate_trait_file(trait)
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
            pipeline, epilogue, scheduler, pad_m, pad_n, pad_k)

        content += f"\n}} // namespace {trait}\n"
        (self.output_dir / filename).write_text(content)

    def _generate_kernel_struct(self, pipeline: str, epilogue: str, scheduler: str,
                                pad_m: str, pad_n: str, pad_k: str) -> str:
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

            ave_time = ck_tile::launch_kernel(stream,
                                          ck_tile::make_kernel<blocks.x, kBlockPerCu>(
                                              Kernel{{}}, grids, blocks, 0, kargs));
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
        for trait in self.all_trait_names:
            content += f"#include \"gemm_{trait}.hpp\"\n"
        (self.output_dir / "gemm_instances.hpp").write_text(content)

    def is_tile_valid(self, tile: tuple, trait: str) -> bool:
        """Check if the tile configuration is valid for the given trait."""
        tile_m, tile_n, tile_k, warp_m, warp_n, warp_k, warp_tile_m, warp_tile_n, warp_tile_k = tile
        pipeline, *_ = trait.split("_")

        # Parameter validity check
        invalid_params = []
        if (warp_m * warp_tile_m) == 0:
            invalid_params.append(
                f"warp_m({warp_m}) * warp_tile_m({warp_tile_m})")
        if (warp_n * warp_tile_n) == 0:
            invalid_params.append(
                f"warp_n({warp_n}) * warp_tile_n({warp_tile_n})")
        if (warp_k * warp_tile_k) == 0:
            invalid_params.append(
                f"warp_k({warp_k}) * warp_tile_k({warp_tile_k})")

        if invalid_params:
            logging.warning(
                f"Trait: [{trait}], Invalid warp configuratio: {', '.join(invalid_params)}. "
                f"Parameter combination: warp=({warp_m},{warp_n},{warp_k}), "
                f"warp_tile=({warp_tile_m},{warp_tile_n},{warp_tile_k})"
            )
            return False

        # Dimension alignment check
        alignment_issues = []
        if tile_m % (warp_m * warp_tile_m) != 0:
            alignment_issues.append(
                f"tile_m({tile_m}) % [{warp_m}x{warp_tile_m}] = {tile_m % (warp_m * warp_tile_m)}")
        if tile_n % (warp_n * warp_tile_n) != 0:
            alignment_issues.append(
                f"tile_n({tile_n}) % [{warp_n}x{warp_tile_n}] = {tile_n % (warp_n * warp_tile_n)}")
        if tile_k % (warp_k * warp_tile_k) != 0:
            alignment_issues.append(
                f"tile_k({tile_k}) % [{warp_k}x{warp_tile_k}] = {tile_k % (warp_k * warp_tile_k)}")

        if alignment_issues:
            logging.warning(
                f"Trait: [{trait}], Dimension alignment failed: {', '.join(alignment_issues)}. "
                f"Tile dimensions {tile_m}x{tile_n}x{tile_k} must be divisible by "
                f"[warp]: {warp_m}x{warp_n}x{warp_k} x [warp_tile]: {warp_tile_m}x{warp_tile_n}x{warp_tile_k}"
            )
            return False

        # LDS capacity verification
        matrix_a_size = (tile_m * tile_k) * \
            element_size(self.config.problem.datatype_map['matrix_a'])
        matrix_b_size = (tile_n * tile_k) * \
            element_size(self.config.problem.datatype_map['matrix_b'])
        total_tile_in_lds = matrix_a_size + matrix_b_size

        max_tile_size = 2**16 if pipeline == "compv4" else 2**15
        if total_tile_in_lds > max_tile_size:
            logging.warning(
                f"LDS capacity exceeded [{trait}]: Total required {total_tile_in_lds:,}B ({total_tile_in_lds/1024:.1f}KB) > "
                f"maximum allowed {max_tile_size:,}B ({max_tile_size/1024}KB). Breakdown:\n"
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
            logging.warning(
                f"Trait: [{trait}], No valid warp tile combinations found for {gpu_name}/{warp_tile_key}, skip this check.")
            return True

        allowed_combinations = gpu_warp_tile_key.get(warp_tile_key, [])
        if not allowed_combinations:
            logging.warning(
                f"Trait: [{trait}], No valid warp tile combinations found for {gpu_name}/{warp_tile_key}, skip this check.")
            return True

        if current_combination not in allowed_combinations:
            logging.warning(
                f"Trait: [{trait}], Invalid warp combination: {current_combination} not in allowed list. "
                f"Valid combinations for data type '{warp_tile_key}': {allowed_combinations}"
            )
            return False

        return True

    def _generate_dispatcher_file(self):
        """Generate the code block of dispatch mechanism."""
        content = """// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <unordered_map>
#include <functional>
#include <vector>

#include "gemm_common.hpp"
#include "gemm_instances.hpp"
#include "gemm_host_api.hpp"

struct GemmDispatcher {
    static auto& get_kernel_map() {
        // Use a static local variable
        static std::unordered_map<
            std::string,
            std::vector<std::function<std::tuple<std::string, float>(ck_tile::GemmHostArgs&, const ck_tile::stream_config&)>>>
            kernel_map;
        return kernel_map;
    }

    static void init(bool structured_sparsity) {
        auto& kernel_map = get_kernel_map();
        if(!kernel_map.empty()) return;
        \n"""

        def get_tile_value(tile_param): return tile_param.generate_candidates(
        ) if isinstance(tile_param, RangeConfigParam) else tile_param.values

        tile_params = set(itertools.product(
            get_tile_value(self.config.tile_config.tile_m),
            get_tile_value(self.config.tile_config.tile_n),
            get_tile_value(self.config.tile_config.tile_k),
            get_tile_value(self.config.tile_config.warp_m),
            get_tile_value(self.config.tile_config.warp_n),
            get_tile_value(self.config.tile_config.warp_k),
            get_tile_value(self.config.tile_config.warp_tile_m),
            get_tile_value(self.config.tile_config.warp_tile_n),
            get_tile_value(self.config.tile_config.warp_tile_k),
        ))

        for trait in self.all_trait_names:
            content += f"""         kernel_map["{trait}"] = {{"""
            for tile in tile_params:
                if self.is_tile_valid(tile, trait):
                    content += f"""[&](ck_tile::GemmHostArgs& args, const ck_tile::stream_config& stream) {{ """
                    content += f""" if(structured_sparsity){{  // SMFMA"""
                    sparse = self.config.problem.datatype_map['matrix_a'] == 'fp16' and \
                        self.config.problem.datatype_map['matrix_b'] == 'fp16' and \
                        self.config.problem.datatype_map['matrix_c'] == 'fp16' and \
                        ((tile[6] == 32 and tile[7] == 32 and tile[8] == 16) or
                         (tile[6] == 16 and tile[7] == 16 and tile[8] == 32))
                    content += f"""
                        return run_kernel<{trait}::GemmKernel<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}, {tile[6]}, {tile[7]}, {tile[8]}, {BOOL_MAP(sparse)}>>(args, stream);"""
                    content += f"""
                            }} else {{"""
                    content += f"""
                        return run_kernel<{trait}::GemmKernel<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}, {tile[6]}, {tile[7]}, {tile[8]}, {BOOL_MAP(False)}>>(args, stream);"""
                    content += f"""
                            }} """
                    content += f"""
                        }} """
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


def do_list_blobs(args: argparse.Namespace,
                  user_provide_config: Optional[GemmConfig] = None):
    generator = GemmCodeGenerator(args.working_path, user_provide_config)
    generator.list_all_trait_names()


def do_gen_blobs(args: argparse.Namespace,
                 user_provide_config: Optional[GemmConfig] = None):
    generator = GemmCodeGenerator(args.working_path, user_provide_config)
    generator.generate_all_instance_files()


def main(args):

    gemm_config = GemmConfig.from_json(
        args.config_json) if args.config_json is not None else args.config_json

    if args.list_blobs:
        do_list_blobs(args, gemm_config)
    elif args.gen_blobs:
        do_gen_blobs(args, gemm_config)
    else:
        logging.warning(
            "No mode specified (use --list_blobs or --gen_blobs). Generating by default...")
        do_gen_blobs(args, gemm_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm kernel",
    )
    parser.add_argument(
        "-w", "--working_path", default="./", required=False, help="The path where all the blobs are going to be generated"
    )
    parser.add_argument(
        "-j", "--config_json", required=False, help="Path to the json which contains the configurations that user provide"
    )
    parser.add_argument(
        "-l", "--list_blobs", action='store_true', help="List all kernel instances to file"
    )
    parser.add_argument(
        "-g", "--gen_blobs", action='store_true', help="Generate all kernel instances into different files"
    )

    args = parser.parse_args()

    main(args)
