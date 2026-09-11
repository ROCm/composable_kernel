#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Pooling kernel instance builder for tile_engine.

Generates C++ kernel headers for pooling operations with specific tile
configurations and trait combinations.

Usage:
    --list_kernels: List valid kernel configurations
    --gen_single:   Generate a single kernel header
    --gen_individual: Generate all kernel headers
"""

import os
import json
import argparse
import itertools
import multiprocessing
import concurrent.futures
from pathlib import Path
import logging

from pooling_validation_utils import (
    is_tile_config_valid,
    is_trait_combination_valid,
    get_dtype_string,
    get_reduce_op_string,
)

logger = logging.getLogger(__name__)


class PoolingKernelBuilder:
    def __init__(self, working_path, datatype, config_json=None):
        self.working_path = Path(working_path)
        self.datatype = datatype
        self.config_json = config_json

        # Create working directory if it doesn't exist
        self.working_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_json and os.path.exists(config_json):
            with open(config_json, "r") as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()

    def _get_default_config(self):
        """Return default configuration if no config file is provided"""
        return {
            "tile_config": {
                "block_m": {"values": [64,128,256]},
                "block_n": {"values": [1,2]},
                "warp_m": {"values": [1]},
                "warp_n": {"values": [1]},
                "warp_tile_m": {"values": [128]},
                "warp_tile_n": {"values": [1]},
                "thread_tile_m": {"values": [1,2,4]},
                "thread_tile_n": {"values": [1]},
            },
            "trait_config": {
                "reduce_op": {"values": ["max", "min", "avg"]},
                "output_index": {"values": [True, False]},
                "propagate_nan": {"values": [True, False]},
                "pooling_dim": {"values": ["2d", "3d"]},
            },
        }

    def _get_tile_configs(self, fast_mode=False):
        """Get tile configurations from config"""
        if "tile_config" not in self.config:
            return []

        tile_config = self.config["tile_config"]

        block_m_values = tile_config.get("block_m", {}).get("values", [64,128,256])
        block_n_values = tile_config.get("block_n", {}).get("values", [1,2])
        warp_m_values = tile_config.get("warp_m", {}).get("values", [1])
        warp_n_values = tile_config.get("warp_n", {}).get("values", [1])
        warp_tile_m_values = tile_config.get("warp_tile_m", {}).get("values", [128])
        warp_tile_n_values = tile_config.get("warp_tile_n", {}).get("values", [1])
        thread_tile_m_values = tile_config.get("thread_tile_m", {}).get("values", [1,2,4])
        thread_tile_n_values = tile_config.get("thread_tile_n", {}).get("values", [1])

        configs = []
        for block_m in block_m_values:
            for block_n in block_n_values:
                for warp_m in warp_m_values:
                    for warp_n in warp_n_values:
                        for warp_tile_m in warp_tile_m_values:
                            for warp_tile_n in warp_tile_n_values:
                                for thread_tile_m in thread_tile_m_values:
                                    for thread_tile_n in thread_tile_n_values:
                                        if self._validate_tile_config(
                                            block_m,
                                            block_n,
                                            warp_m,
                                            warp_n,
                                            warp_tile_m,
                                            warp_tile_n,
                                            thread_tile_m,
                                            thread_tile_n,
                                            fast_mode=fast_mode,
                                        ):
                                            configs.append(
                                                {
                                                    "block_m": block_m,
                                                    "block_n": block_n,
                                                    "warp_m": warp_m,
                                                    "warp_n": warp_n,
                                                    "warp_tile_m": warp_tile_m,
                                                    "warp_tile_n": warp_tile_n,
                                                    "thread_tile_m": thread_tile_m,
                                                    "thread_tile_n": thread_tile_n,
                                                }
                                            )
        return configs

    def _validate_tile_config(
        self,
        block_m,
        block_n,
        warp_m,
        warp_n,
        warp_tile_m,
        warp_tile_n,
        thread_tile_m,
        thread_tile_n,
        fast_mode=False,
    ):
        """Validate tile configuration via pooling_validation_utils."""
        return is_tile_config_valid(
            block_m,
            block_n,
            warp_m,
            warp_n,
            warp_tile_m,
            warp_tile_n,
            thread_tile_m,
            thread_tile_n,
            self.datatype,
            self.datatype,
            fast_mode=fast_mode,
        )

    def _generate_trait_combinations(self):
        """Generate all combinations of traits"""
        if "trait_config" not in self.config:
            return [("max", True, False, "2d")]

        trait_config = self.config["trait_config"]

        reduce_ops = trait_config.get("reduce_op", {}).get("values", ["min","max","avg"])
        output_indices = trait_config.get("output_index", {}).get("values", [True, False])
        propagate_nans = trait_config.get("propagate_nan", {}).get("values", [True, False])
        pooling_dims = trait_config.get("pooling_dim", {}).get("values", ["2d", "3d"])

        all_combinations = list(
            itertools.product(reduce_ops, output_indices, propagate_nans, pooling_dims)
        )

        # Filter valid combinations
        combinations = []
        for combo in all_combinations:
            reduce_op, output_index, propagate_nan, pooling_dim = combo
            if is_trait_combination_valid(
                reduce_op, output_index, propagate_nan, pooling_dim
            ):
                combinations.append(combo)
            else:
                logger.debug(
                    f"Skipping unsupported trait combination: {reduce_op}-{output_index}-{propagate_nan}-{pooling_dim}"
                )

        return combinations

    def _get_dtype_string(self):
        """Get C++ type string for datatype."""
        return get_dtype_string(self.datatype)

    def _get_reduce_op_string(self, reduce_op):
        """Get C++ reduce op type string."""
        return get_reduce_op_string(reduce_op)

    def _generate_kernel_instance(self, tile_config, trait_combo, is_header=True):
        """Generate a single kernel instance header"""
        reduce_op, output_index, propagate_nan, pooling_dim = trait_combo

        # Create kernel name
        kernel_name = (
            f"pool_{self.datatype}_{pooling_dim}_{reduce_op}_"
            f"{'idx' if output_index else 'noidx'}_"
            f"{'nan' if propagate_nan else 'nonan'}"
        )

        # Create tile configuration string
        tile_str = (
            f"{tile_config['block_m']}x{tile_config['block_n']}_"
            f"{tile_config['warp_m']}x{tile_config['warp_n']}_"
            f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}_"
            f"{tile_config['thread_tile_m']}x{tile_config['thread_tile_n']}"
        )

        kernel_name += f"_{tile_str}"

        # Determine types
        in_type = self._get_dtype_string()
        out_type = in_type
        compute_type = "float"  # Always use float for computation
        index_type = "ck_tile::index_t"
        reduce_op_type = self._get_reduce_op_string(reduce_op)

        output_index_str = "true" if output_index else "false"
        propagate_nan_str = "true" if propagate_nan else "false"

        # Generate 2D or 3D specific code
        if pooling_dim == "2d":
            tensor_shape_type = "ck_tile::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t, ck_tile::index_t>"
            window_shape_type = "ck_tile::tuple<ck_tile::index_t, ck_tile::index_t>"
            window_rank = 2
        else:
            tensor_shape_type = "ck_tile::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t, ck_tile::index_t, ck_tile::index_t>"
            window_shape_type = (
                "ck_tile::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t>"
            )
            window_rank = 3

        pragma_line = "#pragma once\n" if is_header else ""
        instance_code = f"""// Generated kernel instance for {kernel_name}
{pragma_line}
#include <cstdint>
#include <utility>
#include <tuple>
#include <iostream>
#include <stdexcept>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/pooling.hpp"

using InDataType = {in_type};
using OutDataType = {out_type};
using ComputeDataType = {compute_type};
using IndexDataType = {index_type};
using ReduceOpType = {reduce_op_type};

using TensorShape = {tensor_shape_type};
using WindowShape = {window_shape_type};

// Kernel name for display
constexpr const char* KERNEL_NAME = "{kernel_name}";
constexpr int POOLING_DIM = {window_rank};

// Wrapper for simplified launch interface
struct SelectedKernel {{
    // Tile configuration - PoolShape parameters
    static constexpr ck_tile::index_t Block_M = {tile_config["block_m"]};
    static constexpr ck_tile::index_t Block_N = {tile_config["block_n"]};
    static constexpr ck_tile::index_t WarpPerBlock_M = {tile_config["warp_m"]};
    static constexpr ck_tile::index_t WarpPerBlock_N = {tile_config["warp_n"]};
    static constexpr ck_tile::index_t WarpTile_M = {tile_config["warp_tile_m"]};
    static constexpr ck_tile::index_t WarpTile_N = {tile_config["warp_tile_n"]};
    static constexpr ck_tile::index_t ThreadTile_M = {tile_config["thread_tile_m"]};
    static constexpr ck_tile::index_t ThreadTile_N = {tile_config["thread_tile_n"]};

    // Traits
    static constexpr bool kOutputIndex = {output_index_str};
    static constexpr bool kPropagateNan = {propagate_nan_str};

    // Pool shape
    using BlockWarps = ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N>;
    using BlockTile = ck_tile::sequence<Block_M, Block_N>;
    using WarpTile = ck_tile::sequence<WarpTile_M, WarpTile_N>;
    using ThreadTile = ck_tile::sequence<ThreadTile_M, ThreadTile_N>;

    using PoolShapeType = ck_tile::PoolShape<BlockWarps, BlockTile, WarpTile, ThreadTile>;

    // Problem and kernel types
    using Problem = ck_tile::PoolProblem<InDataType,
                                         OutDataType,
                                         ComputeDataType,
                                         IndexDataType,
                                         ReduceOpType,
                                         kOutputIndex,
                                         kPropagateNan,
                                         PoolShapeType>;
    using Kernel = ck_tile::PoolKernel<Problem>;

    static float launch(ck_tile::PoolHostArgs<TensorShape, WindowShape>& args,
                        const ck_tile::stream_config& stream) {{

        constexpr ck_tile::index_t kBlockPerCu = 1;
        const ck_tile::index_t kBlockSize = Kernel::BlockSize();

        auto kernel_args = Kernel::MakeKernelArgs(args);

        if (!Kernel::IsSupportedArgument(kernel_args)) {{
            throw std::runtime_error(
                std::string("Unsupported arguments for pooling kernel: ") + KERNEL_NAME);
        }}

        const ck_tile::index_t kGridSize = Kernel::CalculateGridSize(kernel_args);

        if(stream.log_level_ > 0) {{
            std::cout << "Launching pooling kernel: " << KERNEL_NAME << "\\n"
                      << "  grid_size: " << kGridSize << ", block_size: " << kBlockSize
                      << std::endl;
        }}

        return ck_tile::launch_kernel(
            stream,
            ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, kGridSize, kBlockSize, 0, kernel_args));
    }}
}};
"""
        return kernel_name, instance_code

    def write_kernel_list(self):
        """Write kernel list to file for CMake to read"""
        tile_configs = self._get_tile_configs(fast_mode=False)
        trait_combos = self._generate_trait_combinations()

        kernel_list = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                reduce_op, output_index, propagate_nan, pooling_dim = trait_combo

                kernel_name = (
                    f"pool_{self.datatype}_{pooling_dim}_{reduce_op}_"
                    f"{'idx' if output_index else 'noidx'}_"
                    f"{'nan' if propagate_nan else 'nonan'}"
                )

                tile_str = (
                    f"{tile_config['block_m']}x{tile_config['block_n']}_"
                    f"{tile_config['warp_m']}x{tile_config['warp_n']}_"
                    f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}_"
                    f"{tile_config['thread_tile_m']}x{tile_config['thread_tile_n']}"
                )

                kernel_name += f"_{tile_str}"

                trait_str = (
                    f"{reduce_op}_"
                    f"{'true' if output_index else 'false'}_"
                    f"{'true' if propagate_nan else 'false'}_"
                    f"{pooling_dim}"
                )

                kernel_list.append(
                    {
                        "name": kernel_name,
                        "tile_config": tile_config,
                        "trait_combo": trait_combo,
                        "tile_str": tile_str,
                        "trait_str": trait_str,
                    }
                )

        # Write kernel count
        with open(self.working_path / "pool_kernel_count.txt", "w") as f:
            f.write(str(len(kernel_list)))

        # Write kernel list
        with open(self.working_path / "pool_kernel_list.txt", "w") as f:
            for kernel in kernel_list:
                f.write(
                    f"{kernel['name']}|{kernel['tile_str']}|{kernel['trait_str']}\n"
                )

        print(f"Listed {len(kernel_list)} kernel configurations")

    def generate_individual(self, num_workers=None):
        """Generate individual kernel files with parallel processing"""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        work_items = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                work_items.append(
                    (
                        tile_config,
                        trait_combo,
                        self.working_path,
                        self.datatype,
                    )
                )

        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )

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

        kernel_list.sort(key=lambda x: x[0])
        print(
            f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
        )

    def run(self, num_workers=None):
        """Run the builder to generate individual kernel files"""
        self.generate_individual(num_workers)


def _generate_single_kernel_individual(work_item):
    """Worker function to generate a single individual kernel file"""
    tile_config, trait_combo, working_path, datatype = work_item

    builder = PoolingKernelBuilder(working_path, datatype)

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        header_file = working_path / f"pooling_single_{kernel_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Pooling kernel instance builder for tile_engine"
    )
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp8", "fp16", "bf16", "fp32"],
        help="Data type",
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument(
        "--num_workers", type=int, help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--gen_individual", action="store_true", help="Generate individual kernel files"
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

    args = parser.parse_args()

    builder = PoolingKernelBuilder(args.working_path, args.datatype, args.config_json)

    if args.list_kernels:
        builder.write_kernel_list()
    elif args.gen_single:
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )

        # Parse tile config: "block_mx block_n_warp_mxwarp_n_warp_tile_mxwarp_tile_n_thread_tile_mxthread_tile_n"
        tile_parts = args.tile_config.split("_")
        block_dims = tile_parts[0].split("x")
        warp_dims = tile_parts[1].split("x")
        warp_tile_dims = tile_parts[2].split("x")
        thread_tile_dims = tile_parts[3].split("x")

        tile_config = {
            "block_m": int(block_dims[0]),
            "block_n": int(block_dims[1]),
            "warp_m": int(warp_dims[0]),
            "warp_n": int(warp_dims[1]),
            "warp_tile_m": int(warp_tile_dims[0]),
            "warp_tile_n": int(warp_tile_dims[1]),
            "thread_tile_m": int(thread_tile_dims[0]),
            "thread_tile_n": int(thread_tile_dims[1]),
        }

        # Parse trait combo: "reduce_op_output_index_propagate_nan_pooling_dim"
        trait_parts = args.trait_combo.split("_")
        trait_combo = (
            trait_parts[0],  # reduce_op
            trait_parts[1].lower() == "true",  # output_index
            trait_parts[2].lower() == "true",  # propagate_nan
            trait_parts[3],  # pooling_dim
        )

        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        header_file = builder.working_path / f"pooling_single_{kernel_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")

    elif args.gen_individual:
        builder.run(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
