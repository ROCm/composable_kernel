# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import argparse
import importlib.util
import multiprocessing
import concurrent.futures


def _import_gemm_kernel_builder():
    """Import validation utilities from commons directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(parent_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder, gemm_builder_module.get_dtype_string


GemmKernelBuilder, get_dtype_string = _import_gemm_kernel_builder()


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
LAYOUT_MAP = _validation_utils.LAYOUT_MAP


def get_multi_abd_base_layouts(layout_code):
    """
    Return base layouts for multi_abd from a 4-letter code like 'rcrr'.
    Characters map to: A, B, E, D (all tensors of same group share the layout).
    Returns: (a_layout, b_layout, e_layout, d_layout)
    """
    code = str(layout_code).strip().lower()
    a_layout = LAYOUT_MAP[code[0]]
    b_layout = LAYOUT_MAP[code[1]]
    e_layout = LAYOUT_MAP[code[2]]
    d_layout = LAYOUT_MAP[code[3]]
    return a_layout, b_layout, e_layout, d_layout


class GemmMultiABDKernelBuilder(GemmKernelBuilder):
    def __init__(
        self,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        a_elementwise_function,
        b_elementwise_function,
        cde_elementwise_function,
        num_a_tensors=2,
        num_b_tensors=2,
        num_d_tensors=2,
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
        self.a_elementwise_function = a_elementwise_function
        self.b_elementwise_function = b_elementwise_function
        self.cde_elementwise_function = cde_elementwise_function
        self.num_a_tensors = num_a_tensors
        self.num_b_tensors = num_b_tensors
        self.num_d_tensors = num_d_tensors

    def populate_kernel_header(self, kernel_name):
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
#include "ck_tile/ops/gemm/kernel/gemm_multi_abd_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
"""
        return instance_code

    def populate_kernel_dtype_layout(self):
        acc_type = "float"
        e_type = self.datatype
        if self.datatype in ["fp8", "bf8"]:
            e_type = "fp16"

        a_layout, b_layout, e_layout, d_layout = get_multi_abd_base_layouts(self.layout)
        dtype_str = get_dtype_string(self.datatype)
        e_dtype_str = get_dtype_string(e_type)

        lines = []

        # Individual A types
        for i in range(self.num_a_tensors):
            lines.append(f"using A{i}DataType = {dtype_str};")
        # Individual B types
        for i in range(self.num_b_tensors):
            lines.append(f"using B{i}DataType = {dtype_str};")
        # Individual D types
        for i in range(self.num_d_tensors):
            lines.append(f"using D{i}DataType = {dtype_str};")
        lines.append(f"using EDataType = {e_dtype_str};")
        lines.append(
            "using CDataType = EDataType;"
        )  # alias required by GemmProfiler base
        lines.append(f"using AccDataType = {acc_type};")
        lines.append("")

        # Tuple types
        a_types = ", ".join(f"A{i}DataType" for i in range(self.num_a_tensors))
        b_types = ", ".join(f"B{i}DataType" for i in range(self.num_b_tensors))
        d_types = ", ".join(f"D{i}DataType" for i in range(self.num_d_tensors))
        lines.append(f"using AsDataType = ck_tile::tuple<{a_types}>;")
        lines.append(f"using BsDataType = ck_tile::tuple<{b_types}>;")
        lines.append(f"using DsDataType = ck_tile::tuple<{d_types}>;")
        lines.append("")

        # Base types (convenience aliases for the common type within each group)
        lines.append("using ADataType = A0DataType;")
        lines.append("using BDataType = B0DataType;")
        lines.append("using DBaseDataType = D0DataType;")
        lines.append("")

        # Individual layouts
        for i in range(self.num_a_tensors):
            lines.append(f"using A{i}Layout = {a_layout};")
        for i in range(self.num_b_tensors):
            lines.append(f"using B{i}Layout = {b_layout};")
        for i in range(self.num_d_tensors):
            lines.append(f"using D{i}Layout = {d_layout};")
        lines.append(f"using ELayout = {e_layout};")
        lines.append("")

        # Tuple layouts
        a_layouts = ", ".join(f"A{i}Layout" for i in range(self.num_a_tensors))
        b_layouts = ", ".join(f"B{i}Layout" for i in range(self.num_b_tensors))
        d_layouts = ", ".join(f"D{i}Layout" for i in range(self.num_d_tensors))
        lines.append(f"using AsLayout = ck_tile::tuple<{a_layouts}>;")
        lines.append(f"using BsLayout = ck_tile::tuple<{b_layouts}>;")
        lines.append(f"using DsLayout = ck_tile::tuple<{d_layouts}>;")
        lines.append("")

        # Base layouts
        lines.append("using ALayout = A0Layout;")
        lines.append("using BLayout = B0Layout;")
        lines.append("using DLayout = D0Layout;")
        lines.append("")

        # Tensor count constants
        lines.append(
            f"static constexpr std::size_t NumATensors = {self.num_a_tensors};"
        )
        lines.append(
            f"static constexpr std::size_t NumBTensors = {self.num_b_tensors};"
        )
        lines.append(
            f"static constexpr std::size_t NumDTensors = {self.num_d_tensors};"
        )
        lines.append("")

        # Element-wise function types
        lines.append(
            f"using AElementWiseFn = ck_tile::element_wise::{self.a_elementwise_function};"
        )
        lines.append(
            f"using BElementWiseFn = ck_tile::element_wise::{self.b_elementwise_function};"
        )
        lines.append(
            f"using CDEElementWiseFn = ck_tile::element_wise::{self.cde_elementwise_function};"
        )

        return "\n" + "\n".join(lines) + "\n"

    def populate_initialization(self, base_pipeline_map, pipeline):
        instance_code = f"""  

    // Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    // Tile partitioner
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

    // Traits - use tuple layouts for multi_abd
    using Traits = ck_tile::TileGemmUniversalTraits<kPadM, kPadN, kPadK, DoubleSmemBuffer,
                                                    AsLayout, BsLayout, ELayout, TransposeC>;

    // Pipeline problem
    using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
        AsDataType,
        BsDataType,
        AccDataType,
        TileShape,
        Traits,
        AElementWiseFn,
        BElementWiseFn>;

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
        instance_code = """

    // Launch function
    static float launch(const ck_tile::GemmMultiABDHostArgs<NumATensors,
                                                             NumBTensors,
                                                             NumDTensors>& args,
                        const ck_tile::stream_config& stream) {"""

        instance_code += f"""

        constexpr auto scheduler = {scheduler_type_map.get(scheduler)};

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                AsDataType,
                BsDataType,
                AccDataType,
                TileShape,
                Traits,
                scheduler,
                AElementWiseFn,
                BElementWiseFn>;
            
        using GemmPipeline = {pipeline_impl_map.get(pipeline)}<UniversalGemmProblem>;"""

        instance_code += self.populate_epilogue(epilogue)

        instance_code += """
            
        // Kernel type
        using GemmKernelType = ck_tile::GemmKernelMultiABD<TilePartitioner, GemmPipeline, GemmEpilogue>;
        
        // Kernel arguments
        auto kargs = GemmKernelType::MakeKernelArgs(args);
        
        if (!GemmKernelType::IsSupportedArgument(kargs)) {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!");
        }

        // Get grid and block sizes
        const dim3 grids = GemmKernelType::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = GemmKernelType::BlockSize();
        
        if(stream.log_level_ > 0) {
            std::cout << "Launching kernel with args: " << GemmKernelType::GetName() << '\\n'
                        << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                        << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                        << std::endl;
        }"""

        instance_code += f"""    
        // Launch kernel
        constexpr int kBlockPerCu = {k_block_per_cu};
        float ave_time = ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel<kBlockPerCu>(GemmKernelType{{}}, grids, blocks, 0, kargs));
        
        return ave_time;
    }}
}};
"""
        return instance_code

    def populate_epilogue(self, epilogue):
        instance_code = """

        // Epilogue
        """
        if epilogue == "cshuffle":
            instance_code += self.populate_cshuffle_gemm_multi_abd()
        else:
            instance_code += self.populate_default_gemm_multi_abd()
        return instance_code

    def populate_cshuffle_gemm_multi_abd(self):
        instance_code = """            
        using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
            AsDataType,
            BsDataType,
            DsDataType,
            AccDataType,
            EDataType,
            DsLayout,
            ELayout,
            CDEElementWiseFn,
            TileM,  // kM_
            TileN,  // kN_
            WarpPerBlock_M,              // MWave_
            WarpPerBlock_N,              // NWave_
            WarpTileM,                   // MPerXdl_
            WarpTileN,                   // NPerXdl_
            WarpTileK,                   // KPerXdl_
            TransposeC>;                 // isCTransposed_
    
        using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        return instance_code

    def populate_default_gemm_multi_abd(self):
        instance_code = """            
        using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
            AsDataType,
            BsDataType,
            DsDataType,
            AccDataType,
            EDataType,
            DsLayout,
            ELayout,
            CDEElementWiseFn,
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

    def _generate_all_individual(self, num_workers=None):
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
                    self.a_elementwise_function,
                    self.b_elementwise_function,
                    self.cde_elementwise_function,
                    self.num_a_tensors,
                    self.num_b_tensors,
                    self.num_d_tensors,
                    self.config_json,
                )
            )

        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )
        print(f"  Total kernels: {len(work_items)}")
        print(
            f"  NumA={self.num_a_tensors}, NumB={self.num_b_tensors}, NumD={self.num_d_tensors}"
        )

        if work_items:
            print("  First work item example:")
            tile_config, trait_combo = work_items[0][:2]
            print(f"    Tile config: {tile_config}")
            print(f"    Trait combo: {trait_combo[:3]}")

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


def _generate_single_kernel_individual(work_item):
    (
        tile_config,
        trait_combo,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        a_elementwise_function,
        b_elementwise_function,
        cde_elementwise_function,
        num_a_tensors,
        num_b_tensors,
        num_d_tensors,
        config_json,
    ) = work_item

    builder = GemmMultiABDKernelBuilder(
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        a_elementwise_function,
        b_elementwise_function,
        cde_elementwise_function,
        num_a_tensors,
        num_b_tensors,
        num_d_tensors,
        config_json,
    )

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )
        simplified_name = kernel_name
        if simplified_name.startswith("gemm_multi_abd_"):
            simplified_name = simplified_name[len(kernel_name_prefix) + 1 :]

        header_file = working_path / f"gemm_multi_abd_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)
        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Multi ABD kernel instance builder with parallel support"
    )
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument("--gpu_target", required=True, help="GPU target architecture")
    parser.add_argument("--datatype", required=True, choices=["fp16"], help="Data type")
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcrr"],
        help="Matrix layout (4 chars: A, B, E, D)",
    )
    parser.add_argument(
        "--a_elementwise_function",
        default="PassThrough",
        help="Element-wise function for A tensors",
    )
    parser.add_argument(
        "--b_elementwise_function",
        default="PassThrough",
        help="Element-wise function for B tensors",
    )
    parser.add_argument(
        "--cde_elementwise_function",
        default="PassThrough",
        help="Element-wise function for CDE",
    )
    parser.add_argument(
        "--num_a_tensors", type=int, default=2, help="Number of A tensors (>=1)"
    )
    parser.add_argument(
        "--num_b_tensors", type=int, default=2, help="Number of B tensors (>=1)"
    )
    parser.add_argument(
        "--num_d_tensors", type=int, default=2, help="Number of D tensors (>=1)"
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument("--num_workers", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--gen_all_individual",
        action="store_true",
        help="Generate individual kernel files",
    )
    parser.add_argument(
        "--gen_single", action="store_true", help="Generate a single kernel file"
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument("--tile_config", help="Tile configuration string")
    parser.add_argument("--trait_combo", help="Trait combination string")
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
    parser.add_argument("--tier", default=None, help="Sampling tier (daily/weekly)")
    parser.add_argument(
        "--manifest-path", default=None, help="Directory for chosen_instances.json"
    )

    args = parser.parse_args()

    assert args.datatype in ["fp16"], f"Invalid datatype: {args.datatype}"

    layout_parts = args.layout.lower()
    assert len(layout_parts) == 4, (
        f"Invalid layout string: {args.layout} (must be 4 characters)"
    )
    assert layout_parts[0] in ["r", "c"] and layout_parts[1] in ["r", "c"]
    assert layout_parts[2] == "r" and layout_parts[3] == "r"

    assert args.num_a_tensors >= 1, "num_a_tensors must be >= 1"
    assert args.num_b_tensors >= 1, "num_b_tensors must be >= 1"
    assert args.num_d_tensors >= 1, "num_d_tensors must be >= 1"

    valid_functions = ["PassThrough", "AddScale", "MultiDMultiply", "MultiDAdd"]
    for fn_name, fn_val in [
        ("a_elementwise_function", args.a_elementwise_function),
        ("b_elementwise_function", args.b_elementwise_function),
        ("cde_elementwise_function", args.cde_elementwise_function),
    ]:
        if fn_val not in valid_functions:
            raise ValueError(
                f"Invalid {fn_name}: {fn_val}. Valid: {', '.join(valid_functions)}"
            )

    kernel_name_prefix = "gemm_multi_abd"
    builder = GemmMultiABDKernelBuilder(
        kernel_name_prefix,
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        args.a_elementwise_function,
        args.b_elementwise_function,
        args.cde_elementwise_function,
        args.num_a_tensors,
        args.num_b_tensors,
        args.num_d_tensors,
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
