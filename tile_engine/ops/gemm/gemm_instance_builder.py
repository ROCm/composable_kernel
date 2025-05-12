# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# -*- coding: utf-8 -*-
"""
generate kernel instances to speed up compilation
"""

import argparse
from enum import IntEnum
from pathlib import Path
import os
import sys
from typing import List, Optional, Dict, Any, Union, Tuple, Type
import functools
import itertools
import copy
import json
from dataclasses import dataclass
from pydantic import BaseModel, model_validator, field_validator, ValidationInfo, Field, ValidationError
from codegen_utils import *

class BaseConfigParam(BaseModel):
    """Base configuration parameter model"""
    
    @model_validator(mode='before')
    def validate_mode_exclusivity(cls, data: Dict) -> Dict:
        mode_requirements = {
            'enum': {'required': ['values'], 'optional': []},
            'range': {'required': ['min', 'max'], 'optional': ['step']}
        }

        active_modes = []
        for mode, reqs in mode_requirements.items():
            required_fields = reqs['required']
            if all(field in data for field in required_fields):
                active_modes.append(mode)

        if len(active_modes) > 1:
            raise ValidationError(
                f"Configuration conflict: Multiple active modes detected {active_modes}",
                [{'type': 'mode_conflict', 'ctx': {'modes': active_modes}}]
            )

        if not active_modes:
            raise ValidationError(
                "No valid configuration mode detected. Must provide either: "
                "- enum: 'values' list\n"
                "- range: 'min'/'max' with optional 'step'",
                [{'type': 'mode_required'}]
            )

        current_mode = active_modes[0]
        if current_mode == 'enum':
            if not isinstance(data['values'], list) or len(data['values']) == 0:
                raise ValueError("Enum mode requires non-empty 'values' list")
        elif current_mode == 'range':
            min_val = data['min']
            max_val = data['max']
            if min_val > max_val:
                raise ValueError(f"Invalid range: {min_val} > {max_val}")
            if 'step' in data and data['step'] <= 0:
                raise ValueError(f"Invalid step: {data['step']} (must be positive)")

        return data

class EnumConfigParam(BaseConfigParam):
    """Enum-type configuration parameter that enforces explicit values mode"""
    # name: str = Field(..., description="Parameter name for semantic checks")
    values: List[Union[int, str, bool]] = Field(
        ...,
        min_items=1,
        description="Allowed values for enum selection"
    )

    @field_validator("values")
    def validate_enum_values(cls, v, info: ValidationInfo)-> Any: 
        # param_name = info.data.get('name', 'unknown')
        # 1. bool type validation
        # BOOLEAN_PARAMS = {'pad_m', 'pad_n', 'pad_k'}
        # if param_name in BOOLEAN_PARAMS:
        #     for item in v:
        #         if not isinstance(item, bool):
        #             invalid_type = type(item).__name__
        #             raise ValueError(
        #                 f"Parameter '{param_name}' requires boolean values only. "
        #                 f"Found invalid type: {invalid_type}"
        #             )

        # 2. General type validation (int/str/bool)
        valid_types = (int, str, bool) 
        for item in v:
            if not isinstance(item, valid_types):
                invalid_type = type(item).__name__
                allowed = [t.__name__ for t in valid_types]
                raise TypeError(
                    f"Invalid type '{invalid_type}' in enum values. "
                    f"Allowed types: {allowed}"
                )
            
            # 3. String content validation
            if isinstance(item, str) and len(item.strip()) == 0:
                raise ValueError("Empty string not allowed in enum values")
        
        if len(v) != len(set(v)):
            raise ValueError("Duplicate values in enum list")

        return v

class RangeConfigParam(BaseConfigParam):
    """Range-type parameter with min/max/step and exclusion support"""
    min: int = Field( 
        ...,
        json_schema_extra={
            "mode": "range",
            "description": "Lower boundary for range mode"
        }
    )
    max: int = Field(
        ...,
        json_schema_extra={
            "mode": "range",
            "description": "Upper boundary for range mode"
        }
    )
    step: int = Field(
        default=1,
        ge=1,
        json_schema_extra={
            "description": "Increment step between values"
        }
    )
    exclude: Optional[List[int]] = Field(
        default=None,
        json_schema_extra={
            "validation": "Values must be within [min, max] range"
        }
    )

    @model_validator(mode='before')
    def validate_min_max_relationship(cls, data: dict) -> dict:
        """Validates range boundaries and step compatibility"""
        min_val = data.get('min')
        max_val = data.get('max')
        if min_val is not None and max_val is not None and min_val > max_val:
            raise ValueError("`min` must be less than `max`")
        # Pre-validate candidate generation to catch empty ranges
        if all(key in data for key in ('min', 'max', 'step')):
            try:
                candidates = list(range(data['min'], data['max'] + 1, data['step']))
                if not candidates:
                    raise ValueError("Empty candidate list with current step")
            except ValueError as e:
                raise ValueError(f"Invalid step configuration: {str(e)}")
                
        return data

    @field_validator('step')
    def validate_step_value(cls, v: int) -> int:
        """Ensures step is a valid positive integer"""
        if v <= 0:
            raise ValueError("Step must be a positive integer")
        return v

    @field_validator('exclude')
    def validate_exclusion_range(cls, v: list, values: ValidationInfo) -> list:
        """Validates exclusion list against range constraints"""
        if not v:
            return v

        data = values.data
        if 'min' not in data or 'max' not in data:
            raise ValueError("Missing min/max for exclusion validation")

        min_val = data['min']
        max_val = data['max']
        step_val = data.get('step', 1)

        # Check for duplicate exclusions
        if len(v) != len(set(v)):
            raise ValueError("Exclude list contains duplicate values")

        # Validate value boundaries
        out_of_bounds = [x for x in v if not (min_val <= x <= max_val)]
        if out_of_bounds:
            raise ValueError(f"Excluded values {out_of_bounds} out of bounds")

        # Verify step alignment
        misaligned = [x for x in v if (x - min_val) % step_val != 0]
        if misaligned:
            raise ValueError(f"Misaligned exclude values {misaligned} with step {step_val}")

        # Detect non-existent candidates in exclusion list
        try:
            candidates = list(range(min_val, max_val + 1, step_val))
            ghost_excludes = [x for x in v if x not in candidates]
            if ghost_excludes:
                raise ValueError(f"Excludes {ghost_excludes} not in candidate list")
        except ValueError as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

        return v

    def generate_candidates(self) -> List[int]:
        """Generates valid candidates after applying range constraints"""
        candidates = list(range(self.min, self.max + 1, self.step))
        
        if self.exclude:
            exclude_set = set(self.exclude)
            candidates = [x for x in candidates if x not in exclude_set]
        
        if not candidates:
            raise ValueError(
                f"No valid candidates for range [{self.min}-{self.max}] "
                f"with step {self.step} and excludes {self.exclude}"
            )
        
        return candidates


@dataclass
class ProblemConfig:
    """configuration class for managing problem parameter groups."""

    datatypes: Tuple[EnumConfigParam, ...] = Field(
        default_factory=lambda: (
            EnumConfigParam(name='datatype_a', values=["fp16"], metadata={'group': 'datatype'}),
            EnumConfigParam(name='datatype_b', values=["fp16"], metadata={'group': 'datatype'}),
            EnumConfigParam(name='datatype_c', values=["fp16"], metadata={'group': 'datatype'})
        )
    )

    layouts: Tuple[EnumConfigParam, ...] = Field(
        default_factory=lambda: (
            EnumConfigParam(name='layout_a', values=["r"], metadata={'group': 'layout'}),
            EnumConfigParam(name='layout_b', values=["c"], metadata={'group': 'layout'}),
            EnumConfigParam(name='layout_c', values=["r"], metadata={'group': 'layout'})
        )
    )


@dataclass
class TileConfig:
    # Core tile dimensions
    tile_m: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            name="tile_m",
            values=[256],
            metadata={'category': 'tile', 'doc': "M-dimension base tiling"}
        )
    )
    tile_n: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            name="tile_n",
            values=[256],
            metadata={'category': 'tile', 'doc': "N-dimension base tiling"}
        )
    )
    tile_k: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            name="tile_k",
            values=[256],
            metadata={'category': 'tile', 'doc': "K-dimension base tiling"}
        )
    )

    # Warp-level configurations
    warp_m: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            name="warp_m",
            values=[256],
            metadata={'category': 'warp', 'doc': "K-dimension base tiling"}
        )
    )
    warp_n: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            name="warp_n",
            values=[256],
            metadata={'category': 'warp', 'doc': "N-dimension base tiling"}
        )
    )
    warp_k: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            name="warp_k",
            values=[256],
            metadata={'category': 'warp', 'doc': "K-dimension base tiling"}
        )
    )

    # Warp tile subdivision
    warp_tile_m: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            name="warp_tile_m",
            values=[256],
            metadata={'category': 'warp_tile', 'doc': "K-dimension base tiling"}
        )
    )
    warp_tile_n: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            name="warp_tile_n",
            values=[256],
            metadata={'category': 'warp_tile', 'doc': "K-dimension base tiling"}
        )
    )
    warp_tile_k: Union[EnumConfigParam, RangeConfigParam] = Field(
        default_factory=lambda: EnumConfigParam(
            name="warp_tile_k",
            values=[256],
            metadata={'category': 'warp_tile', 'doc': "K-dimension base tiling"}
        )
    )


@dataclass
class TraitConfig:
    """Configuration container for architecture-specific traits and optimizations."""

    pipeline: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=['compv3']),
        metadata={'category': 'execution', 'doc': "Data processing pipeline strategy"}
    )

    scheduler: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=['intrawave']),
        metadata={'category': 'execution', 'doc': "Task scheduling methodology"}
    )

    epilogue: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=['default']),
        metadata={'category': 'execution', 'doc': "Post-processing stage configuration"}
    )

    pad_m: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=[False]),
        metadata={'category': 'padding', 'doc': "M-dimension padding strategy"}
    )

    pad_n: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=[False]),
        metadata={'category': 'padding', 'doc': "N-dimension parallelization approach"}
    )

    pad_k: EnumConfigParam = Field(
        default_factory=lambda: EnumConfigParam(values=[False]),
        metadata={'category': 'padding', 'doc': "K-dimension padding configuration"}
    )

class GemmConfig(BaseModel):
    """Main configuration class for GEMM operations """
    problem: ProblemConfig
    tile_config: TileConfig
    trait_config: TraitConfig

    @classmethod
    def from_json(cls:Type["GemmConfig"], filepath: str, validate_nested: bool = True) -> "GemmConfig":
        """JSON configuration loader with validation controls"""
        
        config_path = Path(filepath)
        
        try:
            # Validate file existence and accessibility
            if not config_path.exists():
                raise FileNotFoundError(f"Config file {filepath} not found")
            config_path.stat()  # Verify file accessibility
            
            # Parse JSON content
            with open(filepath, 'r') as f:
                try:
                    config_dict = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"JSON parsing failed in {filepath}\n"
                        f"Error at line {e.lineno}: {e.msg}"
                    ) from e

            # Configuration construction logic
            if validate_nested:
                return cls.model_validate(
                    config_dict,
                    context={'validating': True}
                )
            else:
                # Verify required fields in construct mode
                required_fields = {'problem', 'tile_config', 'trait_config'}
                if missing := required_fields - config_dict.keys():
                    raise ValueError(
                        f"Missing required fields: {missing}"
                    )
                return cls.model_construct(**config_dict)

        except ValidationError as ve:
            # Format validation errors
            error_msgs = [
                f"[{'->'.join(map(str, err['loc']))}] "
                f"{err['msg']} (received: {err['input']!r})"
                for err in ve.errors()
            ]
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(error_msgs)
            ) from ve
            
        except PermissionError as pe:
            raise RuntimeError(
                f"Permission denied accessing {filepath}"
            )
                
    
class GemmCodeGenerator:
    def __init__(self, output_dir: str, use_default_config: bool, user_provided_config: Optional[GemmConfig] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = {"default": None, "user": None} 
        if use_default_config:
            config_path = Path(__file__).resolve().parent / "configs" / "default_config.json"
            self.config["default"] = GemmConfig.from_json(config_path)

        if user_provided_config is not None:
            self.config["user"] = user_provided_config 
        else:
            if not use_default_config:
                raise ValueError("user_provided_config must be provided when use_default_config=False") 

        self.all_trait_name: Dict[str, List[Dict]] = {"default": [], "user": []}
        self.all_trait_config: Dict[str, List[Dict]] = {"default": [], "user": []}

    def list_all(self):
        """List all possible kernel configurations"""
        w_p = Path(self.output_dir)
        list_p = w_p / 'gemm_instance_blobs.txt'
        self._list_config_groups()
        
        # Collect all unique trait names from both default and user configs
        all_traits = []
        for config_type in ["default", "user"]:
            all_traits.extend(self.all_trait_name.get(config_type, []))
        unique_traits = sorted(set(all_traits))  # Sort for consistent order

        # Write all file paths to the list file
        with list_p.open('w') as list_f:
            # Write fixed files
            list_f.write(str(w_p / "gemm_common.hpp") + "\n")
            list_f.write(str(w_p / "gemm_instances.hpp") + "\n")
            list_f.write(str(w_p / "gemm_dispatcher.hpp") + "\n")
            # Write each unique trait file
            for trait in unique_traits:
                list_f.write(str(w_p / f"gemm_{trait}.hpp") + "\n")

    def _list_config_groups(self):
        params = [
            ("pipeline", "pipeline"),
            ("epilogue", "epilogue"),
            ("scheduler", "scheduler"),
            ("pad_m", "pad_m"),
            ("pad_n", "pad_n"), 
            ("pad_k", "pad_k")
        ]

        # To remove some unsupported combinations
        unsupported_combinations = {
            ("compv3", "cshuffle", "interwave"),
            ("compv3", "default", "interwave"),
            ("compv4", "cshuffle", "interwave"),
            ("compv4", "default", "interwave")
        }
        
        for key, gemm_config in self.config.items():
            if gemm_config is None:
                continue
            trait_config = gemm_config.trait_config
            param_values = [
                getattr(trait_config, p).values 
                for (p, _) in params
            ]
            # Generate all unique_combinations
            _unique = set(itertools.product(*param_values))

            for combo in _unique:
                pipeline, epilogue, scheduler, pad_m, pad_n, pad_k = combo
                current_combination = (pipeline, epilogue, scheduler)
                
                if current_combination in unsupported_combinations:
                    raise ValueError(
                        f"Invalid combination: {pipeline}-{epilogue}-{scheduler} "
                        f"in config '{key}'"
                    )
                
                trait_name = (
                    f"{pipeline}_{epilogue}_{scheduler}_"
                    f"pad_{BOOL_MAP(pad_m)}_{BOOL_MAP(pad_n)}_{BOOL_MAP(pad_k)}"
                )
                self.all_trait_name[key].append(trait_name)
                self.all_trait_config[key].append({
                    "pipeline": pipeline,
                    "epilogue": epilogue,
                    "scheduler": scheduler,
                    "pad_m": pad_m,
                    "pad_n": pad_n,
                    "pad_k": pad_k
                })

#     def generate_all(self):
#         self._generate_common_header()
#         self._generate_config_groups()
#         self._generate_dispatcher()
       

#     def _generate_common_header(self):
#         """Generate common header with datatypes and layout"""
#         ctype = self.config["user"].datatypes[0]
#         atype = self.config["user"].datatypes[1]
#         btype = self.config["user"].datatypes[2]

#         content = f"""// SPDX-License-Identifier: MIT
# // Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# #pragma once
# #include "ck_tile/core.hpp"

# // Data types
# using ADataType = {DATA_TYPE_MAP[atype]};
# using BDataType = {DATA_TYPE_MAP[btype]};
# using AccDataType = float;
# using CDataType = {DATA_TYPE_MAP[ctype]};

# // Layout configurations
# using ALayout = {LAYOUT_MAP[self.config["user"].layouts[0]]};
# using BLayout = {LAYOUT_MAP[self.config["user"].layouts[1]]};
# using CLayout = {LAYOUT_MAP[self.config["user"].layouts[2]]};
# """
        

#         (self.output_dir / "gemm_common.hpp").write_text(content)

#     def _generate_config_groups(self):
#         """Generate implementation configuration groups"""
#         self._list_config_groups()
#         for category, configs in self.unique_configs.items():
#             for config in configs:
#                 self._generate_config_group(**config)
#         self.generate_common_instances_header()

    
#     def _generate_config_group(self, pipeline: str, epilogue: str, scheduler: str,
#                               pad_m: bool, pad_n: bool, pad_k: bool):
#         """Generate a configuration group with all tile/warp combinations"""
#         group_name = f"{pipeline}_{epilogue}_{scheduler}_pad_{BOOL_MAP(pad_m)}_{BOOL_MAP(pad_n)}_{BOOL_MAP(pad_k)}"
#         filename = f"gemm_{group_name}.hpp"

#         content = f"""// SPDX-License-Identifier: MIT
# // Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# #include "gemm_common.hpp"
# #include "ck_tile/ops/gemm.hpp"
# #include "ck_tile/ops/epilogue.hpp"
# #include "ck_tile/host.hpp"

# namespace {group_name} {{
# """
#         # Add template struct with configuration
#         content += self._generate_kernel_struct(pipeline, epilogue, scheduler, pad_m, pad_n, pad_k)

#         content += f"\n}} // namespace {group_name}\n"
#         (self.output_dir / filename).write_text(content)

#     def _generate_kernel_struct(self, pipeline: str, epilogue: str, scheduler: str,
#                                pad_m: bool, pad_n: bool, pad_k: bool) -> str:
#         """Generate kernel struct template"""
#         return f"""
# template <int TileM, int TileN, int TileK,
#           int WarpM, int WarpN, int WarpK,
#           int WarpTileM, int WarpTileN, int WarpTileK>
# struct GemmKernel {{
#     static constexpr bool pad_m = {BOOL_MAP(pad_m)};
#     static constexpr bool pad_n = {BOOL_MAP(pad_n)};
#     static constexpr bool pad_k = {BOOL_MAP(pad_k)};

#     static float launch(ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s) {{
#         static constexpr bool permuteA = false;
#         static constexpr bool permuteB = false;
#         static constexpr bool DoubleSmemBuffer = false;
#         static constexpr bool TransposeC = false;

#         static constexpr int kBlockPerCu                         = 1;
#         static constexpr ck_tile::index_t TileParitionerGroupNum = 8;
#         static constexpr ck_tile::index_t TileParitionerM01      = 4;

#         using GemmShape = 
#             ck_tile::TileGemmShape<ck_tile::sequence<TileM, TileN, TileK>,
#                                    ck_tile::sequence<WarpM, WarpN, WarpK>,
#                                    ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>,
#                                    permuteA,
#                                    permuteB>;


#         using TilePartitioner =
#             ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
#                                                       TileParitionerGroupNum,
#                                                       TileParitionerM01>;

#         using Traits  =
#             ck_tile::TileGemmTraits<pad_m, pad_n, pad_k, ALayout, BLayout, CLayout>;        

#         using GemmUniversalTraits =
#             ck_tile::TileGemmUniversalTraits<pad_m, pad_n, pad_k, DoubleSmemBuffer,
#                                              ALayout, BLayout, CLayout, TransposeC>;    

#         using GemmPipelineProblem =
#             ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

#         using BaseGemmPipeline = {PIPELINE_MAP[pipeline][0]}<GemmPipelineProblem>;  

#         const ck_tile::index_t k_grain     = args.k_batch * TileK;
#         const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * TileK;
#         const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
#         const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
#         const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);                                                                                                             

#         float ave_time{{0}};

#         const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
#             constexpr bool has_hot_loop_v = has_hot_loop_.value;
#             constexpr auto tail_number_v  = tail_number_.value;
#             constexpr auto scheduler      = {SCHEDULER_MAP[scheduler]};

#             using UniversalGemmProblem = 
#                 ck_tile::UniversalGemmPipelineProblem<ADataType,
#                                                       BDataType,
#                                                       AccDataType,
#                                                       GemmShape,
#                                                       GemmUniversalTraits,
#                                                       scheduler,
#                                                       has_hot_loop_v,
#                                                       tail_number_v>;

#             using GemmPipeline = {PIPELINE_MAP[pipeline][1]}<UniversalGemmProblem>; 
#             {EPILOGUE_MAP[epilogue]}
#             using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
#             auto kargs   = Kernel::MakeKernelArgs(args);

#             const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
#             constexpr dim3 blocks = Kernel::BlockSize();

#             if(!Kernel::IsSupportedArgument(kargs))
#             {{
#                 throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
#             }}

#             if(s.log_level_ > 0)
#             {{
#                 std::cout << "Launching kernel with args:"
#                       << " grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
#                       << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
#                       << std::endl;
#             }}

#             ave_time = ck_tile::launch_kernel(s,
#                                           ck_tile::make_kernel<blocks.x, kBlockPerCu>(
#                                               Kernel{{}}, grids, blocks, 0, kargs));
#             return ave_time;

#         }};

#         if(has_hot_loop) {{
#             {HOT_LOOP_TRUE[pipeline]}
#         }} else {{
#             {HOT_LOOP_FALSE}
#         }}

#         return ave_time;
#     }}
#     static std::string get_name() {{
#         return std::string("GemmKernel<Bllktile: ") + std::to_string(TileM) + "x" + std::to_string(TileN) + "x" + std::to_string(TileK) + ", " +
#                 "WaveMap: " + std::to_string(WarpM) + "x" + std::to_string(WarpN) + "x" + std::to_string(WarpK) + ", " +
#                 "WarpTile: " + std::to_string(WarpTileM) + "x" + std::to_string(WarpTileN) + "x" + std::to_string(WarpTileK) + ", " +
#                 "PadidngM: " + "{pad_m}" + ", " +
#                 "PaddingN: " + "{pad_n}" + ", " +
#                 "PaddingK: " + "{pad_k}" + ", " +
#                 "Pipeline: " + "{pipeline}" + ", " +
#                 "Epilogue: " + "{epilogue}" + ", " +
#                 "Scheduler: " + "{scheduler}";
#                 }}
# }};
# """

#     def generate_common_instances_header(self):
#         """Generate common instances header"""
#         content = """// SPDX-License-Identifier: MIT
# // Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# #pragma once
# """   
#         generated_groups = set()
#         for _, kernels in self.all_kernels.items():
#             for group in kernels:
#                 if group not in generated_groups:
#                     generated_groups.add(group)
#                     content += f"#include \"gemm_{group}.hpp\"\n"
#         (self.output_dir / "gemm_instances.hpp").write_text(content)

#     def _generate_dispatcher(self):
#         """Generate dispatch mechanism"""
#         content = """// SPDX-License-Identifier: MIT
# // Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

# #pragma once

# #include <unordered_map>
# #include <functional>
# #include <vector>

# #include "gemm_common.hpp"
# #include "gemm_instances.hpp"
# #include "gemm_host_api.hpp"
# #include "benchmark_gemm.hpp"

# struct GemmDispatcher {
#     static auto& get_kernel_map() {
#         // Use a static local variable
#         static std::unordered_map<std::string,
#                                   std::function<void(Profiler&,
#                                                      ck_tile::DeviceMem&,
#                                                      ck_tile::HostTensor<CDataType>&,
#                                                      ck_tile::HostTensor<CDataType>&,
#                                                      int,
#                                                      ck_tile::GemmHostArgs&,
#                                                      const ck_tile::stream_config&)>>
#             kernel_map;
#         return kernel_map;
#     }

#     static void init() {
#         auto& kernel_map = get_kernel_map();    
#         if(!kernel_map.empty()) return;
#         \n"""

#         for category, gemm_config in self.config.items():
#             # Add tile/warp instantiations
#             tile_params = set(itertools.product(
#                 gemm_config.data["tile_m"]["values"],
#                 gemm_config.data["tile_n"]["values"],
#                 gemm_config.data["tile_k"]["values"],
#                 gemm_config.data["warp_m"]["values"],
#                 gemm_config.data["warp_n"]["values"],
#                 gemm_config.data["warp_k"]["values"],
#                 gemm_config.data["warp_tile_m"]["values"],
#                 gemm_config.data["warp_tile_n"]["values"],
#                 gemm_config.data["warp_tile_k"]["values"]
#             ))
#             generated_groups = set()
#             for group in self.all_kernels[category]:
#                 if group not in generated_groups:
#                     generated_groups.add(group)
#                     content += f"""            kernel_map["{group}"] = [](Profiler& profiler,
#                                                                         ck_tile::DeviceMem& c_m_n_dev_buf,
#                                                                         ck_tile::HostTensor<CDataType>& c_m_n_host_result,
#                                                                         ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
#                                                                         int verify, ck_tile::GemmHostArgs& args,
#                                                                         const ck_tile::stream_config& s) {{
#                                 """
#                     for tile in tile_params:
#                         # Check if we have valid tile/warp combinations 
#                         # (tile_m/(warp_m*warp_tile_m)) * warp_m * warp_tile_m == tile_m
#                         if ((tile[0]/(tile[3] * tile[7]) * tile[3] * tile[7]) != tile[0]) or \
#                         ((tile[1]/(tile[4] * tile[8]) * tile[4] * tile[8]) != tile[1]):
#                             continue
#                         content += f"""
#                         profiler.benchmark_kernel<{group}::GemmKernel<{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}, {tile[4]}, {tile[5]}, {tile[6]}, {tile[7]}, {tile[8]}>>(c_m_n_dev_buf, c_m_n_host_result, c_m_n_dev_result, verify, args, s);"""
#                     content += f"""
#                     }};\n"""

#         content += """    }
    
#     static auto dispatch(ck_tile::DeviceMem& c_m_n_dev_buf,
#                          ck_tile::HostTensor<CDataType>& c_m_n_host_result,
#                          ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
#                          int verify,
#                          int metric,
#                          const KernelTraits& trait,
#                          ck_tile::GemmHostArgs& gemm_args,
#                          const ck_tile::stream_config& s) {
#         init();
#         const std::string key = assemble_key(trait);
#         auto& kernel_map = get_kernel_map(); 
#         auto& profiler        = Profiler::instance();
#         if(auto it = kernel_map.find(key); it != kernel_map.end()) {
#             it->second(
#                 profiler, c_m_n_dev_buf, c_m_n_host_result, c_m_n_dev_result, verify, gemm_args, s);
#             profiler.select_best_instance(static_cast<Metric>(metric));
#             return;
#         }
#         throw std::runtime_error("No suitable kernel found: " + key);
#     }

# private:
#     static std::string assemble_key(const KernelTraits &trait) {
#         return std::string(trait.pipeline) + "_" + 
#                trait.epilogue + "_" + 
#                trait.scheduler + "_" +
#                "pad_" + 
#                (trait.pad_m ? "true" : "false") + "_" +
#                (trait.pad_n ? "true" : "false") + "_" +
#                (trait.pad_k ? "true" : "false");
#     }
# };

# """
#         (self.output_dir / "gemm_dispatcher.hpp").write_text(content)

        
def do_list_blobs(args: argparse.Namespace, user_provide_config: Optional[GemmConfig] = None):
    generator = GemmCodeGenerator(args.working_path, args.use_default_config, user_provide_config)
    generator.list_all()

# def do_gen_blobs(args: argparse.Namespace, gemm_problem: GemmProblem, user_provide_config: Optional[GemmConfig] = None):
#     generator = GemmCodeGenerator(args.working_path, gemm_problem, args.use_default_config, user_provide_config)
#     generator.generate_all()

     

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
        "-u", "--use_default_config", action = 'store_true', help="Wether use default config json file to generate kernel instance or not"
    )
    parser.add_argument(
        "-j", "--config_json", required=False, help="Path to the json which contains the configurations that user provide"
    )
    parser.add_argument(
        "-l", "--list_blobs", action = 'store_true', help="List all kernel instances to file"
    )
    parser.add_argument(
        "-g", "--gen_blobs", action = 'store_true', help="Generate all kernels instances into different files"
    )
    
    args = parser.parse_args()
    
    main(args)
