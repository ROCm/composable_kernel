# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-
"""
generate kernel instances to speed up compilation
"""

import argparse
import os
import sys
import itertools
import copy
import logging
from json_utils import *
from codegen_utils import *

class GemmCodeGenerator:
    def __init__(self, output_dir: str, user_provided_config: Optional[GemmConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if user_provided_config is not None:
            self.config = user_provided_config 
        else:
            config_path = Path(__file__).resolve().parent / "configs" / "default_config.json"
            self.config = GemmConfig.from_json(config_path)

        self.all_trait_names: List[str] = []
        self.all_trait_configs: List[dict[str, Union[str, bool]]] = []

    def list_all_trait_names(self):
        """List all possible kernel trait names"""
        w_p = Path(self.output_dir)
        list_p = w_p / 'gemm_instance_blobs.txt'
        self._generate_all_traits()

        # Write all file paths to the list file
        with list_p.open('w') as list_f:
            list_f.write(str(w_p / "gemm_common.hpp") + "\n")
            list_f.write(str(w_p / "gemm_instances.hpp") + "\n")
            list_f.write(str(w_p / "gemm_dispatcher.hpp") + "\n")
            for trait in sorted(self.all_trait_names):
                list_f.write(str(w_p / f"gemm_{trait}.hpp") + "\n")

    def _generate_all_traits(self):
        params = ["pipeline", "epilogue", "scheduler", "pad_m", "pad_n", "pad_k"]

        # To remove some unsupported combinations
        unsupported_combinations = {
            ("compv3", "cshuffle", "interwave"),
            ("compv3", "default", "interwave"),
            ("compv4", "cshuffle", "interwave"),
            ("compv4", "default", "interwave")
        }
    
        # Generate all unique_combinations
        _unique = set(itertools.product(*[
            getattr(self.config.trait_config, param).values 
            for param in params
        ]))

        for combo in _unique:
            pipeline, epilogue, scheduler, pad_m, pad_n, pad_k = combo
            current_combination = (pipeline, epilogue, scheduler)
            
            if current_combination not in unsupported_combinations:
                trait_name = (
                    f"{pipeline}_{epilogue}_{scheduler}_"
                    f"pad_{BOOL_MAP(pad_m)}_{BOOL_MAP(pad_n)}_{BOOL_MAP(pad_k)}"
                )
                self.all_trait_names.append(trait_name)
                self.all_trait_configs.append({
                    "pipeline": pipeline,
                    "epilogue": epilogue,
                    "scheduler": scheduler,
                    "pad_m": pad_m,
                    "pad_n": pad_m,
                    "pad_k": pad_k
                })
            else:
                logging.warning(
                    f"Invalid combination: {pipeline}-{epilogue}-{scheduler}"
                )

    def generate_all_instance_files(self):
        self._generate_common_header_files()
        self._generate_all_trait_files()
        self._generate_dispatcher_files()
       

    def _generate_common_header_files(self):
        """Generate common header with datatypes and layout"""

        content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"

// Data types
using ADataType = {DATA_TYPE_MAP[self.config.problem.datatype_values[0]]};
using BDataType = {DATA_TYPE_MAP[self.config.problem.datatype_values[1]]};
using AccDataType = float;
using CDataType = {DATA_TYPE_MAP[self.config.problem.datatype_values[2]]};

// Layout configurations
using ALayout = {LAYOUT_MAP[self.config.problem.layout_values[0]]};
using BLayout = {LAYOUT_MAP[self.config.problem.layout_values[1]]};
using CLayout = {LAYOUT_MAP[self.config.problem.layout_values[2]]};
"""
        

        (self.output_dir / "gemm_common.hpp").write_text(content)

    def _generate_all_trait_files(self):
        """Generate implementation """
        if not self.all_trait_configs:  # Check if the list is empty
            self._generate_all_traits()
        for trait_config in self.all_trait_configs:
            self._generate_trait_files(**trait_config)
        self._generate_common_instance_header_files()

    
    def _generate_trait_files(self, pipeline: str, epilogue: str, scheduler: str,
                              pad_m: bool, pad_n: bool, pad_k: bool):
        """Generate a configuration group with all tile/warp combinations"""
        trait_name = f"{pipeline}_{epilogue}_{scheduler}_pad_{BOOL_MAP(pad_m)}_{BOOL_MAP(pad_n)}_{BOOL_MAP(pad_k)}"
        filename = f"gemm_{trait_name}.hpp"

        content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_common.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/host.hpp"

namespace {trait_name} {{
"""
        # Add template struct with configuration
        content += self._generate_kernel_struct(pipeline, epilogue, scheduler, pad_m, pad_n, pad_k)

        content += f"\n}} // namespace {trait_name}\n"
        (self.output_dir / filename).write_text(content)

    def _generate_kernel_struct(self, pipeline: str, epilogue: str, scheduler: str,
                               pad_m: bool, pad_n: bool, pad_k: bool) -> str:
        """Generate kernel struct template"""
        return f"""
template <typename Pipeline, ck_tile::TailNumber TN>
void try_run(ck_tile::TailNumber tn) {{
    if constexpr (Pipeline::PrefetchStages > static_cast<int>(TN)) {{
        if (tn == TN) {{
            RunSplitk(ck_tile::bool_constant<true>{{}},
                ck_tile::integral_constant<ck_tile::TailNumber, TN>{{}});
        }}
    }}
}}
template <int TileM, int TileN, int TileK,
          int WarpM, int WarpN, int WarpK,
          int WarpTileM, int WarpTileN, int WarpTileK,
          bool structured_sparsity>
struct GemmKernel {{
    static constexpr bool kPadM = {BOOL_MAP(pad_m)};
    static constexpr bool kPadN = {BOOL_MAP(pad_n)};
    static constexpr bool kPadK = {BOOL_MAP(pad_k)};

    static float launch(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s) {{
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

            if(s.log_level_ > 0)
            {{
                std::cout << "Launching kernel with args:"
                      << " grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                      << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                      << std::endl;
            }}

            ave_time = ck_tile::launch_kernel(s,
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
        return std::string("GemmKernel<Bllktile: ") + std::to_string(TileM) + "x" + std::to_string(TileN) + "x" + std::to_string(TileK) + ", " +
                "WaveMap: " + std::to_string(WarpM) + "x" + std::to_string(WarpN) + "x" + std::to_string(WarpK) + ", " +
                "WarpTile: " + std::to_string(WarpTileM) + "x" + std::to_string(WarpTileN) + "x" + std::to_string(WarpTileK) + ", " +
                "PadidngM: " + "{pad_m}" + ", " +
                "PaddingN: " + "{pad_n}" + ", " +
                "PaddingK: " + "{pad_k}" + ", " +
                "Pipeline: " + "{pipeline}" + ", " +
                "Epilogue: " + "{epilogue}" + ", " +
                "Scheduler: " + "{scheduler}";
                }}
}};
"""

    def _generate_common_instance_header_files(self):
        """Generate common instances header"""
        content = """// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
"""   
        for trait in self.all_trait_names:
            content += f"#include \"gemm_{trait}.hpp\"\n"
        (self.output_dir / "gemm_instances.hpp").write_text(content)

    def _generate_dispatcher_files(self):
        """Generate dispatch mechanism"""
        content = """// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <unordered_map>
#include <functional>
#include <vector>

#include "gemm_common.hpp"
#include "gemm_instances.hpp"
#include "gemm_host_api.hpp"
#include "benchmark_gemm.hpp"

struct GemmDispatcher {
    static auto& get_kernel_map() {
        // Use a static local variable
        static std::unordered_map<std::string,
                                  std::function<void(GemmProfiler&,
                                                     ck_tile::DeviceMem&,
                                                     ck_tile::HostTensor<CDataType>&,
                                                     ck_tile::HostTensor<CDataType>&,
                                                     int,
                                                     ck_tile::GemmHostArgs&,
                                                     const ck_tile::stream_config&)>>
            kernel_map;
        return kernel_map;
    }

    static void init(bool structured_sparsity) {
        auto& kernel_map = get_kernel_map();    
        if(!kernel_map.empty()) return;            
        \n"""

        def get_tile_value(tile_param):
            if isinstance(tile_param, RangeConfigParam):
                tile_param_values = list(range(tile_param.min, tile_param.max, tile_param.step))
                if tile_param.exclude is not None:
                    exclude_set = set(tile_param.exclude)
                    tile_param_values = [v for v in tile_param_values if v not in exclude_set]
            else:
                tile_param_values = tile_param.values
            return tile_param_values
        
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
            content += f"""        kernel_map["{trait}"] = [=]( GemmProfiler& profiler,
                                                                ck_tile::DeviceMem& c_m_n_dev_buf,
                                                                ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                                                                ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                                                                int verify, 
                                                                ck_tile::GemmHostArgs& args,
                                                                const ck_tile::stream_config& stream) {{
            if(structured_sparsity){{  // SMFMA"""
            for tile in tile_params:
                # Check if we have valid tile/warp combinations 
                # (tile_m/(warp_m*warp_tile_m)) * warp_m * warp_tile_m == tile_m
                if ((tile[0]/(tile[3] * tile[7]) * tile[3] * tile[7]) != tile[0]) or \
                   ((tile[1]/(tile[4] * tile[8]) * tile[4] * tile[8]) != tile[1]):
                    continue
                sparse = self.config.problem.datatype_values[0] == 'fp16' and \
                    ((tile[6] == 32 and tile[7] == 32 and tile[8] == 16) or
                     (tile[6] == 16 and tile[7] == 16 and tile[8] == 32))
                content += f"""
                profiler.benchmark_kernel<{trait}::GemmKernel<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}, {tile[6]}, {tile[7]}, {tile[8]}, {BOOL_MAP(sparse)}>>(c_m_n_dev_buf, c_m_n_host_result, c_m_n_dev_result, verify, args, stream);"""
            content += f"""
            }} else {{"""
            for tile in tile_params:
                # Check if we have valid tile/warp combinations 
                # (tile_m/(warp_m*warp_tile_m)) * warp_m * warp_tile_m == tile_m
                if ((tile[0]/(tile[3] * tile[7]) * tile[3] * tile[7]) != tile[0]) or \
                ((tile[1]/(tile[4] * tile[8]) * tile[4] * tile[8]) != tile[1]):
                    continue
                content += f"""
                profiler.benchmark_kernel<{trait}::GemmKernel<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}, {tile[6]}, {tile[7]}, {tile[8]}, {BOOL_MAP(False)}>>(c_m_n_dev_buf, c_m_n_host_result, c_m_n_dev_result, verify, args, stream);"""
            content += f"""
            }}
        }};\n"""

        content += """    }
    
    static auto dispatch(ck_tile::DeviceMem& c_m_n_dev_buf,
                         ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                         ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                         int verify,
                         int metric,
                         bool structured_sparsity, 
                         const KernelTraits& trait,
                         ck_tile::GemmHostArgs& gemm_args,
                         const ck_tile::stream_config& stream) {
        init(structured_sparsity);
        const std::string key = assemble_key(trait);
        auto& kernel_map = get_kernel_map(); 
        auto& profiler        = GemmProfiler::instance();
        if(auto it = kernel_map.find(key); it != kernel_map.end()) {
            it->second(
                profiler, c_m_n_dev_buf, c_m_n_host_result, c_m_n_dev_result, verify, gemm_args, stream);
            profiler.select_best_instance(static_cast<Metric>(metric));
            return;
        }
        throw std::runtime_error("No suitable kernel found: " + key);
    }

private:
    static std::string assemble_key(const KernelTraits &trait) {
        return std::string(trait.pipeline) + "_" + 
               trait.epilogue + "_" + 
               trait.scheduler + "_" +
               "pad_" + 
               (trait.pad_m ? "true" : "false") + "_" +
               (trait.pad_n ? "true" : "false") + "_" +
               (trait.pad_k ? "true" : "false");
    }
};

"""
        (self.output_dir / "gemm_dispatcher.hpp").write_text(content)

        
def do_list_blobs(args: argparse.Namespace, user_provide_config: Optional[GemmConfig] = None):
    generator = GemmCodeGenerator(args.working_path, user_provide_config)
    generator.list_all_trait_names()

def do_gen_blobs(args: argparse.Namespace, user_provide_config: Optional[GemmConfig] = None):
    generator = GemmCodeGenerator(args.working_path, user_provide_config)
    generator.generate_all_instance_files()

     

def main(args):

    # Read user provide json file
    if args.config_json is not None:
        gemm_config = GemmConfig.from_json(args.config_json)

        if args.list_blobs:
            do_list_blobs(args, gemm_config)
        elif args.gen_blobs:
            do_gen_blobs(args, gemm_config)
        else:
            # If neither was specified, either do nothing or default to gen_blobs
            print("No mode specified (use --list_blobs or --gen_blobs). Generating by default...")
            do_gen_blobs(args, gemm_config)
    else:
        if args.list_blobs:
            do_list_blobs(args)
        elif args.gen_blobs:
            do_gen_blobs(args)
        else:
            # If neither was specified, either do nothing or default to gen_blobs
            print("No mode specified (use --list_blobs or --gen_blobs). Generating by default...")
            do_gen_blobs(args)
   


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
        "-l", "--list_blobs", action = 'store_true', help="List all kernel instances to file"
    )
    parser.add_argument(
        "-g", "--gen_blobs", action = 'store_true', help="Generate all kernel instances into different files"
    )
    
    args = parser.parse_args()
    
    main(args)
