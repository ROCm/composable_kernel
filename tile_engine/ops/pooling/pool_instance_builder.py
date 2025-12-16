#!/usr/bin/env python
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import json
import argparse
import itertools
import multiprocessing
import concurrent.futures
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


def get_dtype_string(dtype):
    """Convert dtype name to C++ type string"""
    dtype_map = {
        "fp16": "ck_tile::half_t",
        "fp32": "float",
        "bf16": "ck_tile::bf16_t",
        "fp8": "ck_tile::fp8_t",
        "bf8": "ck_tile::bf8_t",
        "int8": "ck_tile::int8_t",
        "int32": "ck_tile::int32_t",
        "index_t": "ck_tile::index_t",
    }
    return dtype_map.get(dtype, dtype)


def get_reduce_op_string(reduce_op):
    """Convert reduce op name to C++ type string"""
    reduce_op_map = {
        "max": "ck_tile::ReduceOp::Max",
        "min": "ck_tile::ReduceOp::Min",
        "add": "ck_tile::ReduceOp::Add",
        "avg": "ck_tile::ReduceOp::Add",  # Average uses Add and divides later
    }
    return reduce_op_map.get(reduce_op.lower(), "ck_tile::ReduceOp::Max")


class PoolKernelBuilder:
    def __init__(self, working_path, gpu_target, datatype, reduce_op, config_json=None):
        self.working_path = Path(working_path)
        self.gpu_target = gpu_target
        self.datatype = datatype
        self.reduce_op = reduce_op
        self.config_json = config_json

        # Create working directory if it doesn't exist
        self.working_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_json and os.path.exists(config_json):
            with open(config_json, "r") as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = self._get_default_config()

    def _get_default_config(self):
        """Return default configuration for pooling kernels"""
        return {
            "block_config": {
                "block_m": {"values": [64, 128, 256]},
                "block_n": {"values": [1]},
                "warp_m": {"values": [1, 2]},
                "warp_n": {"values": [1]},
                "thread_tile_m": {"values": [1, 2, 4]},
                "thread_tile_n": {"values": [1]},
            },
            "trait_config": {
                "output_index": {"values": [True, False]},
                "propagate_nan": {"values": [False]},
                "pool_dim": {"values": [2, 3]},
            },
            "k_block_per_cu": 1,
        }

    def write_kernel_list(self):
        """Write kernel list to file for CMake to read"""
        block_configs = self._get_block_configs()
        trait_combos = self._generate_trait_combinations()

        kernel_list = []
        for block_config in block_configs:
            for trait_combo in trait_combos:
                output_index, propagate_nan, pool_dim = trait_combo

                # Create kernel name
                kernel_name = f"pool{pool_dim}d_{self.datatype}_{self.reduce_op}"
                kernel_name += f"_{str(output_index).capitalize()}_{str(propagate_nan).capitalize()}"

                # Create block configuration string
                block_str = f"{block_config['block_m']}x{block_config['block_n']}_"
                block_str += f"{block_config['warp_m']}x{block_config['warp_n']}_"
                block_str += (
                    f"{block_config['thread_tile_m']}x{block_config['thread_tile_n']}"
                )

                kernel_name += f"_{block_str}"

                kernel_list.append(
                    {
                        "name": kernel_name,
                        "block_config": block_config,
                        "trait_combo": trait_combo,
                    }
                )

        # Write kernel count
        with open(self.working_path / "pool_kernel_count.txt", "w") as f:
            f.write(str(len(kernel_list)))

        # Write kernel list
        with open(self.working_path / "pool_kernel_list.txt", "w") as f:
            for kernel in kernel_list:
                block_config = kernel["block_config"]
                trait_combo = kernel["trait_combo"]

                block_str = f"{block_config['block_m']}x{block_config['block_n']}_"
                block_str += f"{block_config['warp_m']}x{block_config['warp_n']}_"
                block_str += (
                    f"{block_config['thread_tile_m']}x{block_config['thread_tile_n']}"
                )

                trait_str = "_".join(str(x) for x in trait_combo)

                f.write(f"{kernel['name']}|{block_str}|{trait_str}\n")

        print(f"Listed {len(kernel_list)} kernel configurations")

    def _get_block_configs(self):
        """Get block configurations for the current datatype"""
        block_config = self.config["block_config"]

        block_m_values = block_config.get("block_m").get("values")
        block_n_values = block_config.get("block_n").get("values")
        warp_m_values = block_config.get("warp_m").get("values")
        warp_n_values = block_config.get("warp_n").get("values")
        thread_tile_m_values = block_config.get("thread_tile_m").get("values")
        thread_tile_n_values = block_config.get("thread_tile_n").get("values")

        configs = []
        for block_m in block_m_values:
            for block_n in block_n_values:
                for warp_m in warp_m_values:
                    for warp_n in warp_n_values:
                        for thread_tile_m in thread_tile_m_values:
                            for thread_tile_n in thread_tile_n_values:
                                if self._validate_block_config(
                                    block_m,
                                    block_n,
                                    warp_m,
                                    warp_n,
                                    thread_tile_m,
                                    thread_tile_n,
                                ):
                                    configs.append(
                                        {
                                            "block_m": block_m,
                                            "block_n": block_n,
                                            "warp_m": warp_m,
                                            "warp_n": warp_n,
                                            "thread_tile_m": thread_tile_m,
                                            "thread_tile_n": thread_tile_n,
                                        }
                                    )
        return configs

    def _validate_block_config(
        self, block_m, block_n, warp_m, warp_n, thread_tile_m, thread_tile_n
    ):
        """Validate that block configuration is reasonable"""
        if block_m <= 0 or block_n <= 0:
            return False
        if warp_m <= 0 or warp_n <= 0:
            return False
        if thread_tile_m <= 0 or thread_tile_n <= 0:
            return False

        # Warp size is 64 for AMD GPUs
        warp_size = 64

        # Calculate warp tile sizes
        warp_tile_m = block_m // warp_m
        warp_tile_n = block_n // warp_n

        if warp_tile_m <= 0 or warp_tile_n <= 0:
            return False

        # Check block_m is divisible by warp_m
        if block_m % warp_m != 0:
            return False
        if block_n % warp_n != 0:
            return False

        # Check thread tile fits in warp tile
        if warp_tile_m % thread_tile_m != 0:
            return False
        if warp_tile_n % thread_tile_n != 0:
            return False

        # Critical constraint from pool_shape.hpp:
        # (Warp_M * Warp_N / ThreadTile_M / ThreadTile_N) % warp_size == 0
        # This means threads_per_warp must be a multiple of warp_size (typically equal to it)
        threads_per_warp = (warp_tile_m * warp_tile_n) // (
            thread_tile_m * thread_tile_n
        )
        if threads_per_warp % warp_size != 0:
            return False

        # threads_per_warp should not be too large (usually exactly warp_size)
        if threads_per_warp > warp_size * 4:
            return False

        return True

    def _generate_trait_combinations(self):
        """Generate all combinations of traits"""
        trait_config = self.config["trait_config"]

        output_index_values = trait_config.get("output_index").get("values")
        propagate_nan_values = trait_config.get("propagate_nan").get("values")
        pool_dim_values = trait_config.get("pool_dim").get("values")

        all_combinations = list(
            itertools.product(
                output_index_values,
                propagate_nan_values,
                pool_dim_values,
            )
        )

        return all_combinations

    def _generate_kernel_instance(
        self, block_config, trait_combo, k_block_per_cu, is_header=True
    ):
        """Generate a single kernel instance"""
        output_index, propagate_nan, pool_dim = trait_combo

        # Create kernel name
        kernel_name = f"pool{pool_dim}d_{self.datatype}_{self.reduce_op}"
        kernel_name += (
            f"_{str(output_index).capitalize()}_{str(propagate_nan).capitalize()}"
        )

        # Create block configuration string
        block_str = f"{block_config['block_m']}x{block_config['block_n']}_"
        block_str += f"{block_config['warp_m']}x{block_config['warp_n']}_"
        block_str += f"{block_config['thread_tile_m']}x{block_config['thread_tile_n']}"

        kernel_name += f"_{block_str}"

        # Determine output type (same as input for pooling)
        out_type = self.datatype
        compute_type = "fp32"  # Always use fp32 for compute
        index_type = "index_t"

        # Calculate warp tile sizes
        warp_tile_m = block_config["block_m"] // block_config["warp_m"]
        warp_tile_n = block_config["block_n"] // block_config["warp_n"]

        # Generate kernel instance code
        pragma_line = "#pragma once\n" if is_header else ""
        instance_code = f"""// Generated kernel instance for {kernel_name}
{pragma_line}
#include <cstdint>
#include <utility>
#include <tuple>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/pooling.hpp"
#include "ck_tile/ops/pooling/kernel/pool_kernel.hpp"
#include "ck_tile/ops/pooling/pipeline/pool_problem.hpp"
#include "ck_tile/ops/pooling/pipeline/pool_shape.hpp"

using InDataType = {get_dtype_string(self.datatype)};
using OutDataType = {get_dtype_string(out_type)};
using ComputeDataType = {get_dtype_string(compute_type)};
using IndexDataType = {get_dtype_string(index_type)};

// Reduce operation
using ReduceOpType = {get_reduce_op_string(self.reduce_op)};

// Kernel name for display
constexpr const char* KERNEL_NAME = "{kernel_name}";
constexpr const char* BLOCK_SHAPE_NAME = "{block_str}";
constexpr const char* REDUCE_OP_NAME = "{self.reduce_op}";

// Flags and dimensions
constexpr bool OUTPUT_INDEX = {"true" if output_index else "false"};
constexpr bool PROPAGATE_NAN = {"true" if propagate_nan else "false"};
constexpr int POOL_DIM = {pool_dim};

// Block configuration
using BlockWarps = ck_tile::sequence<{block_config["warp_m"]}, {block_config["warp_n"]}>;
using BlockTile = ck_tile::sequence<{block_config["block_m"]}, {block_config["block_n"]}>;
using WarpTile = ck_tile::sequence<{warp_tile_m}, {warp_tile_n}>;
using ThreadTile = ck_tile::sequence<{block_config["thread_tile_m"]}, {block_config["thread_tile_n"]}>;

using PoolBlockShape = ck_tile::PoolShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;

// Pool problem definition
using Problem = ck_tile::PoolProblem<InDataType,
                                     OutDataType,
                                     ComputeDataType,
                                     IndexDataType,
                                     ReduceOpType,
                                     OUTPUT_INDEX,
                                     PROPAGATE_NAN,
                                     PoolBlockShape>;

// Pool kernel type
using Kernel = ck_tile::PoolKernel<Problem>;

// Shape types for {pool_dim}D pooling
"""
        if pool_dim == 3:
            instance_code += """// 3D pooling shapes (N, D, H, W, C)
using TensorShapeType = decltype(ck_tile::make_tuple(
    ck_tile::index_t{}, ck_tile::index_t{}, ck_tile::index_t{}, 
    ck_tile::index_t{}, ck_tile::index_t{}));
// Window shape (Z, Y, X)
using WindowShapeType = decltype(ck_tile::make_tuple(
    ck_tile::index_t{}, ck_tile::index_t{}, ck_tile::index_t{}));
"""
        else:
            instance_code += """// 2D pooling shapes (N, H, W, C)
using TensorShapeType = decltype(ck_tile::make_tuple(
    ck_tile::index_t{}, ck_tile::index_t{}, 
    ck_tile::index_t{}, ck_tile::index_t{}));
// Window shape (Y, X)
using WindowShapeType = decltype(ck_tile::make_tuple(
    ck_tile::index_t{}, ck_tile::index_t{}));
"""

        instance_code += f"""
// Wrapper for simplified launch interface
struct SelectedKernel {{
    template <typename TensorShape, typename WindowShape>
    static float launch(const ck_tile::PoolHostArgs<TensorShape, WindowShape>& args,
                        const ck_tile::stream_config& stream) {{
        auto kernel_args = Kernel::MakeKernelArgs(
            const_cast<ck_tile::PoolHostArgs<TensorShape, WindowShape>&>(args));
        
        if (!Kernel::IsSupportedArgument(kernel_args)) {{
            throw std::runtime_error("Wrong! Arguments not supported! Skipping pooling kernel!");
        }}
        
        constexpr ck_tile::index_t kBlockPerCu = {k_block_per_cu};
        const ck_tile::index_t kBlockSize = Kernel::BlockSize();
        const ck_tile::index_t kGridSize = Kernel::CalculateGridSize(kernel_args);
        
        if(stream.log_level_ > 0) {{
            std::cout << "Launching kernel: " << KERNEL_NAME << '\\n'
                      << "grid: " << kGridSize
                      << ", blocks: " << kBlockSize
                      << std::endl;
        }}
        
        // Launch kernel
        float ave_time = ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, kGridSize, kBlockSize, 0, kernel_args));
        
        return ave_time;
    }}
}};
"""
        return kernel_name, instance_code

    def run(self, num_workers=None):
        """Run the builder to generate individual kernel files"""
        self.generate_individual(num_workers)

    def generate_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation"""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)

        block_configs = self._get_block_configs()
        trait_combos = self._generate_trait_combinations()
        k_block_per_cu = self.config.get("k_block_per_cu", 1)

        # Prepare work items
        work_items = []
        for block_config in block_configs:
            for trait_combo in trait_combos:
                work_items.append(
                    (
                        block_config,
                        trait_combo,
                        k_block_per_cu,
                        self.working_path,
                        self.gpu_target,
                        self.datatype,
                        self.reduce_op,
                        self.config_json,
                    )
                )

        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )
        print(f"  Block configs: {len(block_configs)}")
        print(f"  Trait combinations: {len(trait_combos)}")
        print(f"  Total kernels: {len(work_items)}")

        # Process work items
        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }

            for future in concurrent.futures.as_completed(future_to_item):
                completed += 1
                if completed % 10 == 0 or completed == len(work_items):
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

        # Sort kernel list
        kernel_list.sort(key=lambda x: x[0])

        # Generate CMake include file
        self._generate_cmake_individual_targets(kernel_list)

        print(
            f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
        )

    def _generate_cmake_individual_targets(self, kernel_list):
        """Generate CMake include file that creates individual targets"""
        cmake_code = f"""# Generated CMake file for individual Pool targets
# Datatype: {self.datatype}, ReduceOp: {self.reduce_op}

"""
        for kernel_name, trait_combo, block_config in kernel_list:
            block_str = f"{block_config['block_m']}x{block_config['block_n']}_"
            block_str += f"{block_config['warp_m']}x{block_config['warp_n']}_"
            block_str += (
                f"{block_config['thread_tile_m']}x{block_config['thread_tile_n']}"
            )

            trait_str = "_".join(str(x) for x in trait_combo)

            cmake_code += f'create_individual_pool_target("{self.datatype}" "{self.reduce_op}" "{trait_str}" "{block_str}")\n'

        with open(self.working_path / "pool_individual_targets.cmake", "w") as f:
            f.write(cmake_code)


def _generate_single_kernel_individual(work_item):
    """Worker function to generate a single individual kernel file"""
    (
        block_config,
        trait_combo,
        k_block_per_cu,
        working_path,
        gpu_target,
        datatype,
        reduce_op,
        config_json,
    ) = work_item

    # Create a temporary builder instance
    builder = PoolKernelBuilder(
        working_path, gpu_target, datatype, reduce_op, config_json
    )

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            block_config, trait_combo, k_block_per_cu
        )

        # Create simplified filename
        simplified_name = kernel_name
        if simplified_name.startswith("pool"):
            simplified_name = simplified_name[4:]  # Remove "pool" prefix

        # Write individual header file
        header_file = working_path / f"pool_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, block_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Pool kernel instance builder with parallel support"
    )
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument(
        "--gpu_target",
        required=True,
        help="GPU target architecture",
    )
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16", "fp32", "bf16"],
        help="Data type",
    )
    parser.add_argument(
        "--reduce_op",
        required=True,
        choices=["max", "min", "avg"],
        help="Reduce operation",
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
        "--block_config", help="Block configuration string for single generation"
    )
    parser.add_argument(
        "--trait_combo", help="Trait combination string for single generation"
    )
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )

    args = parser.parse_args()

    # Create builder
    builder = PoolKernelBuilder(
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.reduce_op,
        args.config_json,
    )

    if args.list_kernels:
        builder.write_kernel_list()
    elif args.gen_single:
        # Generate a single kernel file
        if not args.kernel_name or not args.block_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --block_config, and --trait_combo"
            )

        # Parse block config
        block_parts = args.block_config.split("_")
        block_dims = block_parts[0].split("x")
        warp_dims = block_parts[1].split("x")
        thread_tile_dims = block_parts[2].split("x")

        block_config = {
            "block_m": int(block_dims[0]),
            "block_n": int(block_dims[1]),
            "warp_m": int(warp_dims[0]),
            "warp_n": int(warp_dims[1]),
            "thread_tile_m": int(thread_tile_dims[0]),
            "thread_tile_n": int(thread_tile_dims[1]),
        }

        # Parse trait combo
        trait_parts = args.trait_combo.split("_")
        trait_combo = (
            trait_parts[0] == "True",  # output_index
            trait_parts[1] == "True",  # propagate_nan
            int(trait_parts[2]),  # pool_dim
        )

        k_block_per_cu = builder.config.get("k_block_per_cu", 1)

        # Generate the kernel
        kernel_name, instance_code = builder._generate_kernel_instance(
            block_config, trait_combo, k_block_per_cu
        )

        # Write the file
        simplified_name = kernel_name
        if simplified_name.startswith("pool"):
            simplified_name = simplified_name[4:]

        header_file = builder.working_path / f"pool_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")

    elif args.gen_all_individual:
        builder.run(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_all_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
