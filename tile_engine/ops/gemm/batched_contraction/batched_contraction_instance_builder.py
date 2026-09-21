# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import json
import argparse
import importlib.util
import multiprocessing
import concurrent.futures


def _import_gemm_kernel_builder():
    """Import validation utilities from commons directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(parent_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder


GemmKernelBuilder = _import_gemm_kernel_builder()


def _import_validation_utils():
    """Import validation utilities from commons directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    spec = importlib.util.spec_from_file_location(
        "gemm_validation_utils",
        os.path.join(parent_dir, "gemm_validation_utils.py"),
    )
    validation_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_module)

    return validation_module


_validation_utils = _import_validation_utils()
get_dtype_string = _validation_utils.get_dtype_string


# Layout mapping for batched contraction (only A and B have independent layouts)
def get_contraction_layouts(layout_str):
    """
    Parse layout string for batched contraction.
    layout_str is 3 characters: a_layout + b_layout + e_layout
    e.g., 'rcr' means A=RowMajor, B=ColumnMajor, E=RowMajor
    """
    layout_map = {
        "r": "ck_tile::tensor_layout::gemm::RowMajor",
        "c": "ck_tile::tensor_layout::gemm::ColumnMajor",
    }
    assert len(layout_str) == 3, (
        f"Layout string must be 3 characters (a, b, e), got: {layout_str}"
    )
    a_layout = layout_map[layout_str[0]]
    b_layout = layout_map[layout_str[1]]
    e_layout = layout_map[layout_str[2]]
    return a_layout, b_layout, e_layout


class BatchedContractionKernelBuilder(GemmKernelBuilder):
    def __init__(
        self,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        num_dim_g,
        num_dim_m,
        num_dim_n,
        num_dim_k,
        num_d_tensors,
        elementwise_function,
        config_json=None,
        max_instances=None,
        seed=None,
        tier=None,
        manifest_path=None,
    ):
        super().__init__(
            kernel_name_prefix,
            working_path,
            gpu_target,
            datatype,
            layout,
            config_json,
            max_instances=max_instances,
            seed=seed,
            tier=tier,
            manifest_path=manifest_path,
        )
        self.num_dim_g = num_dim_g
        self.num_dim_m = num_dim_m
        self.num_dim_n = num_dim_n
        self.num_dim_k = num_dim_k
        self.num_d_tensors = num_d_tensors
        self.elementwise_function = elementwise_function

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation with parallel processing"""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)

        sampled_kernels = self._get_sampled_kernel_list()

        work_items = []
        for kernel in sampled_kernels:
            tile_config = kernel["tile_config"]
            trait_combo = kernel["trait_combo"]
            work_items.append(
                (
                    tile_config,
                    trait_combo,
                    self.kernel_name_prefix,
                    self.working_path,
                    self.gpu_target,
                    self.datatype,
                    self.layout,
                    self.num_dim_g,
                    self.num_dim_m,
                    self.num_dim_n,
                    self.num_dim_k,
                    self.num_d_tensors,
                    self.elementwise_function,
                    self.config_json,
                )
            )

        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )
        print(f"  Total kernels: {len(work_items)}")

        if work_items:
            print("  First work item example:")
            tile_config, trait_combo = work_items[0][:2]
            print(f"    Tile config: {tile_config}")
            print(f"    Trait combo: {trait_combo[:3]}")

        # Process work items in parallel
        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            print(f"  Submitting {len(work_items)} tasks to executor...")
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }
            print("  All tasks submitted, waiting for completion...")

            for future in concurrent.futures.as_completed(future_to_item):
                completed += 1
                if completed % 100 == 0 or completed == len(work_items):
                    print(
                        f"  Progress: {completed}/{len(work_items)} kernels generated"
                    )

                try:
                    result = future.result()
                    if result:
                        kernel_list.append(result)
                except Exception as exc:
                    item = future_to_item[future]
                    print(f"Kernel generation failed for {item}: {exc}")

        kernel_list.sort(key=lambda x: x[0])

        self._generate_cmake_individual_targets(kernel_list)

        print(
            f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
        )

    def populate_kernel_header(self, kernel_name):
        """Override to include batched contraction headers"""
        instance_code = f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <cstdint>
#include <utility>
#include <tuple>
#include <array>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
"""
        return instance_code

    def populate_kernel_dtype_layout(self):
        """Override to emit batched contraction types"""
        acc_type = "float"
        e_type = self.datatype
        if self.datatype in ["fp8", "bf8"]:
            e_type = "fp16"

        a_layout, b_layout, e_layout = get_contraction_layouts(self.layout)

        instance_code = f"""
using ADataType = {get_dtype_string(self.datatype)};
using BDataType = {get_dtype_string(self.datatype)};
using AccDataType = {acc_type};
using EDataType = {get_dtype_string(e_type)};
using DBaseDataType = {get_dtype_string(self.datatype)};
"""
        # Generate DsDataType tuple based on num_d_tensors
        if self.num_d_tensors == 0:
            instance_code += "using DsDataType = ck_tile::tuple<>;\n"
        else:
            d_types = ", ".join(["DBaseDataType"] * self.num_d_tensors)
            instance_code += f"using DsDataType = ck_tile::tuple<{d_types}>;\n"

        instance_code += f"""
using ALayout = {a_layout};
using BLayout = {b_layout};
using ELayout = {e_layout};
"""
        # Generate DsLayout tuple
        if self.num_d_tensors == 0:
            instance_code += "using DsLayout = ck_tile::tuple<>;\n"
        else:
            d_layouts = ", ".join(["ELayout"] * self.num_d_tensors)
            instance_code += f"using DsLayout = ck_tile::tuple<{d_layouts}>;\n"

        # CDEElementWise function
        instance_code += f"""
using CDEElementWise = ck_tile::element_wise::{self.elementwise_function};

static constexpr ck_tile::index_t NUM_D_TENSORS = {self.num_d_tensors};
static constexpr ck_tile::index_t NUM_DIM_G = {self.num_dim_g};
static constexpr ck_tile::index_t NUM_DIM_M = {self.num_dim_m};
static constexpr ck_tile::index_t NUM_DIM_N = {self.num_dim_n};
static constexpr ck_tile::index_t NUM_DIM_K = {self.num_dim_k};
"""
        return instance_code

    def populate_initialization(self, base_pipeline_map, pipeline):
        """Override to use batched contraction problem and kernel types"""
        instance_code = """

    // Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    // Tile partitioner
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

    // Traits
    using Traits = ck_tile::TileGemmTraits<kPadM, kPadN, kPadK, ALayout, BLayout, ELayout>;

    // Pipeline problem
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        ADataType,
        BDataType,
        AccDataType,
        TileShape,
        Traits>;
"""
        instance_code += f"""
    // Base pipeline for hot loop detection
    using BaseGemmPipeline = {base_pipeline_map.get(pipeline)}<GemmPipelineProblem>;"""

        return instance_code

    def populate_launch(
        self,
        scheduler_type_map,
        scheduler,
        pipeline_impl_map,
        pipeline,
        epilogue,
        k_block_per_cu,
        persistent,
    ):
        """Override to launch BatchedContractionKernel"""
        instance_code = f"""

    // Launch function
    static float launch(const ck_tile::BatchedContractionHostArgs<NUM_D_TENSORS>& args,
                        const ck_tile::stream_config& stream) {{

        constexpr auto scheduler = {scheduler_type_map.get(scheduler)};

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                ADataType,
                BDataType,
                AccDataType,
                TileShape,
                ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                                ALayout, BLayout, ELayout, TransposeC>,
                scheduler>;

        using GemmPipeline = {pipeline_impl_map.get(pipeline)}<UniversalGemmProblem>;
"""
        # Epilogue
        instance_code += self.populate_epilogue(epilogue)

        # Kernel type - BatchedContractionKernel
        instance_code += f"""

        // Batched Contraction Problem
        using ContractionProblem = ck_tile::BatchedContractionProblem<
            ADataType,
            BDataType,
            DsDataType,
            EDataType,
            NUM_DIM_G,
            NUM_DIM_M,
            NUM_DIM_N,
            NUM_DIM_K,
            NUM_D_TENSORS>;

        // Kernel type
        using ContractionKernel = ck_tile::BatchedContractionKernel<
            ContractionProblem, TilePartitioner, GemmPipeline, GemmEpilogue>;

        // Kernel arguments
        auto kargs = ContractionKernel::MakeKernelArgs(args);

        if (!ContractionKernel::IsSupportedArguments(kargs)) {{
            throw std::runtime_error("Wrong! Arguments not supported! Skipping contraction!");
        }}

        // Get grid and block sizes
        const dim3 grids  = ContractionKernel::GridSize(kargs);
        const dim3 blocks = ContractionKernel::GetBlockSize();

        if(stream.log_level_ > 0) {{
            std::cout << "Launching kernel with args: " << ContractionKernel::GetKernelName() << '\\n'
                      << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                      << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                      << std::endl;
        }}

        // Launch kernel
        constexpr int kBlockPerCu = {k_block_per_cu};
        float ave_time = ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel<kBlockPerCu>(ContractionKernel{{}}, grids, blocks, 0, kargs));

        return ave_time;
    }}
}};
"""
        return instance_code

    def populate_epilogue(self, epilogue):
        """Override epilogue to use batched contraction types"""
        instance_code = """

        // Epilogue
        """
        if epilogue == "cshuffle":
            instance_code += """
        using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
            ADataType,
            BDataType,
            DsDataType,
            AccDataType,
            EDataType,
            DsLayout,
            ELayout,
            CDEElementWise,
            TileM,  // kM_
            TileN,  // kN_
            WarpPerBlock_M,              // MWave_
            WarpPerBlock_N,              // NWave_
            WarpTileM,                   // MPerXdl_
            WarpTileN,                   // NPerXdl_
            WarpTileK,                   // KPerXdl_
            TransposeC>;                 // isCTransposed_

        using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        else:  # default epilogue
            instance_code += """
        using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
            ADataType,
            BDataType,
            DsDataType,
            AccDataType,
            EDataType,
            DsLayout,
            ELayout,
            CDEElementWise,
            TileM,  // kM_
            TileN,  // kN_
            kPadM,
            kPadN,
            WarpTileM,  // kMPerXdl_
            WarpTileN,  // kNPerXdl_
            WarpTileK,  // kKPerXdl_
            TransposeC>;  // isCTransposed_

        using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;"""

        return instance_code


def _generate_single_kernel_individual(work_item):
    """Worker function to generate a single individual kernel file"""
    (
        tile_config,
        trait_combo,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        num_dim_g,
        num_dim_m,
        num_dim_n,
        num_dim_k,
        num_d_tensors,
        elementwise_function,
        config_json,
    ) = work_item

    builder = BatchedContractionKernelBuilder(
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        num_dim_g,
        num_dim_m,
        num_dim_n,
        num_dim_k,
        num_d_tensors,
        elementwise_function,
        config_json,
    )

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        simplified_name = kernel_name
        if simplified_name.startswith(f"{kernel_name_prefix}_"):
            simplified_name = simplified_name[len(kernel_name_prefix) + 1 :]

        header_file = working_path / f"batched_contraction_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Batched Contraction kernel instance builder with parallel support"
    )
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument("--gpu_target", required=True, help="GPU target architecture")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16", "bf16", "fp32"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        help="Layout string (3 chars: a_layout + b_layout + e_layout, e.g. 'rcr')",
    )
    parser.add_argument(
        "--num_dim_g",
        type=int,
        required=True,
        help="Number of G (batch) dimensions",
    )
    parser.add_argument(
        "--num_dim_m",
        type=int,
        required=True,
        help="Number of M dimensions",
    )
    parser.add_argument(
        "--num_dim_n",
        type=int,
        required=True,
        help="Number of N dimensions",
    )
    parser.add_argument(
        "--num_dim_k",
        type=int,
        required=True,
        help="Number of K dimensions",
    )
    parser.add_argument(
        "--num_d_tensors",
        type=int,
        default=0,
        help="Number of D (auxiliary) tensors (default: 0)",
    )
    parser.add_argument(
        "--elementwise_function",
        default="PassThrough",
        help="Elementwise function for CDE epilogue (e.g. PassThrough, Add, Mul)",
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument(
        "--num_workers", type=int, help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--gen_all_individual",
        action="store_true",
        help="Generate individual kernel files",
    )
    parser.add_argument(
        "--gen_single", action="store_true", help="Generate a single kernel file"
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument(
        "--tile_config", help="Tile configuration string for single generation"
    )
    parser.add_argument(
        "--trait_combo", help="Trait combination string for single generation"
    )
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Cap on number of kernel instances per (dtype, layout) combo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for deterministic sampling; if omitted, derived from today's date",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Sampling tier (daily/weekly)",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Directory for chosen_instances.json",
    )

    args = parser.parse_args()

    layout_parts = args.layout.lower()
    assert len(layout_parts) == 3, (
        f"Invalid layout string: {args.layout} (must be 3 characters: a_layout + b_layout + e_layout)"
    )
    for ch in layout_parts:
        assert ch in ["r", "c"], f"Invalid layout character: {ch} (must be 'r' or 'c')"

    # Elementwise function name mapping
    elementwise_function = args.elementwise_function
    valid_functions = ["PassThrough", "Add", "Mul"]
    if elementwise_function.lower() not in [v.lower() for v in valid_functions]:
        raise ValueError(
            f"Invalid elementwise function: {elementwise_function}. "
            f"Valid options are: PassThrough, Add, Mul"
        )

    # Normalize function name
    func_map = {
        "passthrough": "PassThrough",
        "add": "MultiDAdd",
        "mul": "MultiDMultiply",
    }
    elementwise_function = func_map.get(
        elementwise_function.lower(), elementwise_function
    )

    # CLI args take precedence; fall back to config JSON values when CLI
    # args were not explicitly provided (i.e. still at their defaults).
    num_dim_g = args.num_dim_g
    num_dim_m = args.num_dim_m
    num_dim_n = args.num_dim_n
    num_dim_k = args.num_dim_k
    num_d_tensors = args.num_d_tensors

    if args.config_json and os.path.exists(args.config_json):
        with open(args.config_json, "r") as f:
            cfg = json.load(f)
        # Only use config JSON as fallback for args that weren't explicitly
        # provided on the command line (num_dim_g/m/n/k are required so they
        # are always explicit; num_d_tensors has a default of 0).
        if "num_d_tensors" in cfg and args.num_d_tensors == 0:
            num_d_tensors = cfg["num_d_tensors"]

    # Create builder
    kernel_name_prefix = "batched_contraction"
    builder = BatchedContractionKernelBuilder(
        kernel_name_prefix,
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        num_dim_g,
        num_dim_m,
        num_dim_n,
        num_dim_k,
        num_d_tensors,
        elementwise_function,
        args.config_json,
        max_instances=args.max_instances,
        seed=args.seed,
        tier=args.tier,
        manifest_path=args.manifest_path,
    )

    if args.list_kernels:
        builder._list_kernels()
    elif args.gen_single:
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )

        # Parse tile config
        tile_parts = args.tile_config.split("_")
        tile_dims = tile_parts[0].split("x")
        warp_dims = tile_parts[1].split("x")
        warp_tile_dims = tile_parts[2].split("x")

        tile_config = {
            "tile_m": int(tile_dims[0]),
            "tile_n": int(tile_dims[1]),
            "tile_k": int(tile_dims[2]),
            "warp_m": int(warp_dims[0]),
            "warp_n": int(warp_dims[1]),
            "warp_k": int(warp_dims[2]),
            "warp_tile_m": int(warp_tile_dims[0]),
            "warp_tile_n": int(warp_tile_dims[1]),
            "warp_tile_k": int(warp_tile_dims[2]),
        }

        # Parse trait combo
        trait_parts = args.trait_combo.split("_")
        trait_combo = (
            trait_parts[0],  # pipeline
            trait_parts[1],  # epilogue
            trait_parts[2],  # scheduler
            trait_parts[3] == "True",  # pad_m
            trait_parts[4] == "True",  # pad_n
            trait_parts[5] == "True",  # pad_k
            trait_parts[6] == "True",  # persistent
        )

        builder._generate_kernel_instance(tile_config, trait_combo)
    elif args.gen_all_individual:
        builder._generate_all_individual(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_all_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
