# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from enum import IntEnum
from pathlib import Path
import sys
from typing import List, Optional, Dict, Any
import functools
import itertools
import copy
import json
from dataclasses import dataclass
 
DATA_TYPE_MAP = {'fp32'  : 'float',
                 'fp16'  : 'ck_tile::half_t',
                 'bf16'  : 'ck_tile::bf16_t',
                 'int8'  : 'ck_tile::int8_t',
                 'fp8'   : 'ck_tile::fp8_t',
                 'bf8'   : 'ck_tile::bf8_t',
                 'int4'  : 'ck_tile::pk_int4_t'
                }

LAYOUT_MAP = {'r' : 'ck_tile::tensor_layout::gemm::RowMajor',
              'c' : 'ck_tile::tensor_layout::gemm::ColumnMajor'}                                       

DEFAULT_EPILOGUE = """
            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<
                                ck_tile::DefaultGemm2DEpilogueProblem<AccDataType, 
                                                                      CDataType, 
                                                                      CLayout, 
                                                                      kPadM,
                                                                      kPadN,
                                                                      WarpTileM,
                                                                      WarpTileN,
                                                                      WarpTileK,
                                                                      UniversalGemmProblem::TransposeC>>;
"""

CSHUFFLE_EPILOGUE = """
            using GemmEpilogue = ck_tile::CShuffleEpilogue<
                            ck_tile::CShuffleEpilogueProblem<ADataType,
                                                             BDataType,
                                                             AccDataType,
                                                             CDataType,
                                                             CLayout,
                                                             GemmPipelineProblem::kBlockSize,
                                                             TilePartitioner::MPerBlock,
                                                             TilePartitioner::NPerBlock,
                                                             WarpM,
                                                             WarpN,
                                                             WarpTileM,
                                                             WarpTileN,
                                                             WarpTileK,
                                                             UniversalGemmProblem::TransposeC>>;
"""
HOT_LOOP_FALSE = """
            if(tail_num == ck_tile::TailNumber::Full)
            {
                Run(ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_num == ck_tile::TailNumber::Odd)
            {
                Run(ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_num == ck_tile::TailNumber::Even)
            {
                Run(ck_tile::bool_constant<false>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else
            {
                throw std::runtime_error("Num K loop must be larger than number of prefetech stages.");
            }  
"""
RUN_MEM = """
            if(tail_num == ck_tile::TailNumber::One)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::One>{});
            }
            else if(tail_num == ck_tile::TailNumber::Full)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }

            if constexpr(BaseGemmPipeline::PrefetchStages > 2)
            {
                if(tail_num == ck_tile::TailNumber::Two)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
                }
        
                if(tail_num == ck_tile::TailNumber::Three)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
                }
                if(tail_num == ck_tile::TailNumber::Four)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Four>{});
                }
                if(tail_num == ck_tile::TailNumber::Five)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Five>{});
                }
                if(tail_num == ck_tile::TailNumber::Six)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Six>{});
                }
                if(tail_num == ck_tile::TailNumber::Seven)
                {
                    Run(ck_tile::bool_constant<true>{},
                        ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Seven>{});
                }
                throw std::runtime_error("The tile number is wrong! It should not exceed the prefetch stage numbers");
            }
"""

RUN_COMPV3 = """
            if(tail_num == ck_tile::TailNumber::Full)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Full>{});
            }
            else if(tail_num == ck_tile::TailNumber::Odd)
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Odd>{});
            }
            else if(tail_num == ck_tile::TailNumber::Even)
            {
                Run(ck_tile::bool_constant<true>{},
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
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Three>{});
            }
            else
            {
                Run(ck_tile::bool_constant<true>{},
                    ck_tile::integral_constant<ck_tile::TailNumber, ck_tile::TailNumber::Two>{});
            }
"""


PIPELINE_MAP = {'mem' : ['ck_tile::BaseGemmPipelineAgBgCrMem', 'ck_tile::GemmPipelineAgBgCrMem'],
                'compv3' : ['ck_tile::BaseGemmPipelineAgBgCrCompV3', 'ck_tile::GemmPipelineAgBgCrCompV3'],
                'compv4' : ['ck_tile::BaseGemmPipelineAgBgCrCompV4', 'ck_tile::GemmPipelineAgBgCrCompV4']}

SCHEDULER_MAP = {'interwave' : 'ck_tile::GemmPipelineScheduler::Interwave',
                 'intrawave' : 'ck_tile::GemmPipelineScheduler::Intrawave'}

EPILOGUE_MAP = {'default' :DEFAULT_EPILOGUE,
                'cshuffle' : CSHUFFLE_EPILOGUE}      

HOT_LOOP_TRUE = {'mem' : RUN_MEM,
                 'compv3' : RUN_COMPV3,
                 'compv4' : RUN_COMPV4}    


def BOOL_MAP(b_) -> str:
    if b_:
        return 'true'
    else:
        return 'false'

@dataclass
class GemmProblem:
    def __init__(self, config_data):
        self.data : Dict[str, Any] = {}
        for key, value in config_data.items():
            self.data[key] = value

    @property
    def datatype(self) -> str:
        return self.data["datatype"]["values"][0]
    
    @property
    def layouts(self) -> List[str]:
        return [
            self.data["layout_a"]["values"][0],
            self.data["layout_b"]["values"][0],
            self.data["layout_c"]["values"][0]
        ]


@dataclass
class GemmConfig:
    def __init__(self, config_data):
        self.data : Dict[str, Any] = {}
        for key, value in config_data.items():
            self.data[key] = value


class GemmCodeGenerator:
    def __init__(self, output_dir: str, problem: GemmProblem, use_default_config: bool, user_provide_config: Optional[GemmConfig] = None):
        self.output_dir = Path(output_dir)
        self.problem = problem

        self.config = []
        if use_default_config:
            config_path = Path(__file__).resolve().parent / "configs" / "default_config.json"
            with open(config_path, 'r') as json_file:
                config_data = json.load(json_file)
            default_config = GemmConfig(config_data)
            self.config.append(default_config)
            if user_provide_config is not None:
                if not isinstance(user_provide_config, GemmConfig):
                    raise TypeError("user_provide_config must be a GemmConfig instance")
                self.config.append(user_provide_config)
        else:
            if user_provide_config is None:
                raise ValueError("user_provide_config must be provided when use_default_config=False")
            if not isinstance(user_provide_config, GemmConfig):
                raise TypeError("user_provide_config must be a GemmConfig instance")
            self.config.append(user_provide_config)

        self.all_kernels = []
        self.unique_configs = [] 
        # Validate configurations
        self._check_validate()

    def _check_validate(self):
        """Validate matrix problem and kernel configurations"""
        # Matrix problem validation
        for param in ["datatype", "layout_a", "layout_b", "layout_c"]:
            if len(self.problem.data[param]["values"]) != 1:
                raise ValueError(f"Matrix problem {param} must have exactly one value")
        
        # kernel config validation
        required_params = ["tile_m", "tile_n", "tile_k", "warp_m", "warp_n", "warp_k",
                          "warp_tile_m", "warp_tile_n", "warp_tile_k", "pipeline",
                          "epilogue", "scheduler", "kPadM", "kPadN", "kPadK"]
        for config in self.config:
            for param in required_params:
                if not config.data.get(param, {}).get("values"):
                    raise ValueError(f"Missing kernel parameter: {param}")

    def list_all(self):
        """List all possible kernel configurations"""
        w_p = Path(self.output_dir)
        list_p = w_p / 'gemm_instance_blobs.txt'
        self._list_config_groups()
        with list_p.open('w') as list_f:
            list_f.write(str(w_p / ("gemm_common.hpp"))  + "\n")
            list_f.write(str(w_p / ("gemm_instances.hpp"))  + "\n")
            list_f.write(str(w_p / ("gemm_dispatcher.hpp"))  + "\n")  
            for group in self.all_kernels:
                list_f.write(str(w_p / ("gemm_" + group + ".hpp")) + "\n")
            


    def _list_config_groups(self):
        params = [
            ("pipeline", "pipeline"),
            ("epilogue", "epilogue"),
            ("scheduler", "scheduler"),
            ("kPadM", "kPadM"),
            ("kPadN", "kPadN"), 
            ("kPadK", "kPadK")
        ]
        
        # Generate all unique_combinations
        _unique = set(
            itertools.product(*[
                [value for config in self.config for value in config.data[p]["values"]]
                for (p, _) in params
            ])
        )
        for combo in _unique:
            config = {name: value for (_, name), value in zip(params, combo)}
            pipeline, epilogue, scheduler, kPadM, kPadN, kPadK = config.values()
            # To remove some unsupported combinations
            unsupported_combination = [("compv3", "cshuffle", "interwave"),
                                       ("compv3", "default", "interwave"),
                                       ("compv4", "cshuffle", "interwave"),
                                       ("compv4", "default", "interwave")]
            if (pipeline, epilogue, scheduler) not in unsupported_combination:
                group_name = f"{pipeline}_{epilogue}_{scheduler}_pad_{BOOL_MAP(kPadM)}_{BOOL_MAP(kPadN)}_{BOOL_MAP(kPadK)}"
                self.all_kernels.append(group_name)
                self.unique_configs.append(config)

    def generate_all(self):
        self._generate_common_header()
        self._generate_config_groups()
        self._generate_dispatcher()
       

    def _generate_common_header(self):
        """Generate common header with datatypes and layout"""
        ctype = self.problem.datatype
        atype = self.problem.datatype
        btype = self.problem.datatype
        if self.problem.datatype in ['fp8', 'bf8']:
            ctype = 'fp16'
        elif self.problem.datatype in ['int4']:
            atype = 'fp16'
            ctype = 'fp16'

        content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"

// Data types
using ADataType = {DATA_TYPE_MAP[atype]};
using BDataType = {DATA_TYPE_MAP[btype]};
using AccDataType = float;
using CDataType = {DATA_TYPE_MAP[ctype]};

// Layout configurations
using ALayout = {LAYOUT_MAP[self.problem.layouts[0]]};
using BLayout = {LAYOUT_MAP[self.problem.layouts[1]]};
using CLayout = {LAYOUT_MAP[self.problem.layouts[2]]};
"""
        

        (self.output_dir / "gemm_common.hpp").write_text(content)

    def _generate_config_groups(self):
        """Generate implementation configuration groups"""
        if not self.unique_configs:  # Check if the list is empty
            self._list_config_groups()
        for config in self.unique_configs:
            self._generate_config_group(**config)
        self.generate_common_instances_header()

    
    def _generate_config_group(self, pipeline: str, epilogue: str, scheduler: str,
                              kPadM: bool, kPadN: bool, kPadK: bool):
        """Generate a configuration group with all tile/warp combinations"""
        group_name = f"{pipeline}_{epilogue}_{scheduler}_pad_{BOOL_MAP(kPadM)}_{BOOL_MAP(kPadN)}_{BOOL_MAP(kPadK)}"
        filename = f"gemm_{group_name}.hpp"

        content = f"""// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_common.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/host.hpp"

namespace {group_name} {{
"""
        # Add template struct with configuration
        content += self._generate_kernel_struct(pipeline, epilogue, scheduler, kPadM, kPadN, kPadK)

        content += f"\n}} // namespace {group_name}\n"
        (self.output_dir / filename).write_text(content)

    def _generate_kernel_struct(self, pipeline: str, epilogue: str, scheduler: str,
                               kPadM: bool, kPadN: bool, kPadK: bool) -> str:
        """Generate kernel struct template"""
        return f"""
template <int TileM, int TileN, int TileK,
          int WarpM, int WarpN, int WarpK,
          int WarpTileM, int WarpTileN, int WarpTileK>
struct GemmKernel {{
    static constexpr bool kPadM = {BOOL_MAP(kPadM)};
    static constexpr bool kPadN = {BOOL_MAP(kPadN)};
    static constexpr bool kPadK = {BOOL_MAP(kPadK)};

    static float launch(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s) {{
        static constexpr bool permuteA = false;
        static constexpr bool permuteB = false;
        static constexpr bool DoubleSmemBuffer = false;
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
                                             ALayout, BLayout, CLayout, TransposeC>;    

        using GemmPipelineProblem =
            ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

        using BaseGemmPipeline = {PIPELINE_MAP[pipeline][0]}<GemmPipelineProblem>;  

        const ck_tile::index_t k_grain     = args.k_batch * TileK;
        const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * TileK;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);                                                                                                             

        float ave_time{{0}};

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v  = tail_number_.value;
            constexpr auto scheduler      = {SCHEDULER_MAP[scheduler]};

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
                "PadidngM: " + "{kPadM}" + ", " +
                "PaddingN: " + "{kPadN}" + ", " +
                "PaddingK: " + "{kPadK}" + ", " +
                "Pipeline: " + "{pipeline}" + ", " +
                "Epilogue: " + "{epilogue}" + ", " +
                "Scheduler: " + "{scheduler}";
                }}
}};
"""

    def generate_common_instances_header(self):
        """Generate common instances header"""
        content = """// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
"""
        for group in self.all_kernels:
            content += f"#include \"gemm_{group}.hpp\"\n"
        (self.output_dir / "gemm_instances.hpp").write_text(content)

    def _generate_dispatcher(self):
        """Generate dispatch mechanism"""
        content = """// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
                                  std::function<void(Profiler&,
                                                     ck_tile::DeviceMem&,
                                                     ck_tile::HostTensor<CDataType>&,
                                                     ck_tile::HostTensor<CDataType>&,
                                                     int,
                                                     ck_tile::GemmHostArgs&,
                                                     const ck_tile::stream_config&)>>
            kernel_map;
        return kernel_map;
    }

    static void init() {
        auto& kernel_map = get_kernel_map();    
        if(!kernel_map.empty()) return;
        \n"""
         # Add tile/warp instantiations        
        tile_params = set(
            itertools.product(*[
                [value for config in self.config 
                for value in config.data[param]["values"]]
                for param in [
                    "tile_m", "tile_n", "tile_k",
                    "warp_m", "warp_n", "warp_k",
                    "warp_tile_m", "warp_tile_n", "warp_tile_k"
                ]
            ])
        )
        
        for group in self.all_kernels:
            content += f"""            kernel_map["{group}"] = [](Profiler& profiler,
                                                                  ck_tile::DeviceMem& c_m_n_dev_buf,
                                                                  ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                                                                  ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                                                                  int verify, ck_tile::GemmHostArgs& args,
                                                                  const ck_tile::stream_config& s) {{
                        """
            for tile in tile_params:
                # Check if we have valid tile/warp combinations 
                # (tile_m/(warp_m*warp_tile_m)) * warp_m * warp_tile_m == tile_m
                if ((tile[0]/(tile[3] * tile[7]) * tile[3] * tile[7]) != tile[0]) or \
                   ((tile[1]/(tile[4] * tile[8]) * tile[4] * tile[8]) != tile[1]):
                    continue
                content += f"""
                profiler.benchmark_kernel<{group}::GemmKernel<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}, {tile[6]}, {tile[7]}, {tile[8]}>>(c_m_n_dev_buf, c_m_n_host_result, c_m_n_dev_result, verify, args, s);"""
            content += f"""
            }};\n"""

        content += """    }
    
    static auto dispatch(ck_tile::DeviceMem& c_m_n_dev_buf,
                         ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                         ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                         int verify,
                         int metric,
                         const KernelTraits& trait,
                         ck_tile::GemmHostArgs& gemm_args,
                         const ck_tile::stream_config& s) {
        init();
        const std::string key = assemble_key(trait);
        auto& kernel_map = get_kernel_map(); 
        auto& profiler        = Profiler::instance();
        if(auto it = kernel_map.find(key); it != kernel_map.end()) {
            it->second(
                profiler, c_m_n_dev_buf, c_m_n_host_result, c_m_n_dev_result, verify, gemm_args, s);
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
               (trait.kPadM ? "true" : "false") + "_" +
               (trait.kPadN ? "true" : "false") + "_" +
               (trait.kPadK ? "true" : "false");
    }
};

"""
        (self.output_dir / "gemm_dispatcher.hpp").write_text(content)

        
def do_list_blobs(args, gemm_problem, user_provide_config):
    generator = GemmCodeGenerator(args.working_path, gemm_problem, args.use_default_config, user_provide_config)
    generator.list_all()

def do_gen_blobs(args, gemm_problem, user_provide_config):
    generator = GemmCodeGenerator(args.working_path, gemm_problem, args.use_default_config, user_provide_config)
    generator.generate_all()

     

def main(args):
    # Read problem json file
    with open(args.problem_json, 'r') as json_file:
        config_data = json.load(json_file)
    gemm_problem = GemmProblem(config_data)

    # Read user provide json file
    with open(args.config_json, 'r') as json_file:
        config_data = json.load(json_file)
    gemm_config = GemmConfig(config_data)

    if args.list_blobs:
        do_list_blobs(args, gemm_problem, gemm_config)
    elif args.gen_blobs:
        do_gen_blobs(args, gemm_problem, gemm_config)
    else:
        # If neither was specified, either do nothing or default to gen_blobs
        print("No mode specified (use --list_blobs or --gen_blobs). Generating by default...")
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
        "-pj", "--problem_json", required=True, help="Path to the json which defines gemm problem"
    )
    parser.add_argument(
        "-u", "--use_default_config", action = 'store_true', help="Wether use default config json file to generate kernel instance or not"
    )
    parser.add_argument(
        "-cj", "--config_json", required=True, help="Path to the json which contains the kernel configurations that user provide"
    )
    parser.add_argument(
        "-l", "--list_blobs", action = 'store_true', help="List all kernel instance to file"
    )
    parser.add_argument(
        "-g", "--gen_blobs", action = 'store_true', help="Generate all kernels instance into different files"
    )
    
    args = parser.parse_args()
    
    main(args)
