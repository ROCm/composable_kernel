# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import argparse
import importlib.util
import multiprocessing
import concurrent.futures


def _import_gemm_kernel_builder():
    """Import the base GemmKernelBuilder from the gemm directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))

    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(gemm_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder


GemmKernelBuilder = _import_gemm_kernel_builder()


class GemmBQuantKernelBuilder(GemmKernelBuilder):
    def __init__(
        self,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
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
        self.group_size_k = self.config.get("group_size_k", 128)

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation with parallel processing"""
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
                        self.kernel_name_prefix,
                        self.working_path,
                        self.gpu_target,
                        self.datatype,
                        self.layout,
                        self.config_json,
                    )
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
    """Worker function to generate a single individual kernel file"""
    (
        tile_config,
        trait_combo,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json,
    ) = work_item

    builder = GemmBQuantKernelBuilder(
        kernel_name_prefix, working_path, gpu_target, datatype, layout, config_json
    )

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        simplified_name = kernel_name.removeprefix(kernel_name_prefix + "_")

        header_file = working_path / f"gemm_bquant_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="GEMM BQuant kernel instance builder")
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument("--gpu_target", required=True, help="GPU target architecture")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp8", "bf8"],
        help="Data type (fp8 or bf8)",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr", "rrr", "ccr", "crr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument("--num_workers", type=int, help="Number of parallel workers")
    parser.add_argument(
        "--gen_all_individual",
        action="store_true",
        help="Generate individual kernel files",
    )
    parser.add_argument(
        "--gen_single",
        action="store_true",
        help="Generate a single kernel file",
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
        help="Maximum number of kernel instances to select via sampling",
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
        help="Sampling tier name (e.g., 'daily')",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Directory to write chosen_instances manifest JSON",
    )

    args = parser.parse_args()

    layout_parts = args.layout.lower()
    assert len(layout_parts) == 3, (
        f"Invalid layout string: {args.layout} (must be 3 characters like 'rcr')"
    )
    assert layout_parts[0] in ["r", "c"] and layout_parts[1] in ["r", "c"], (
        f"Invalid matrix_a layout: {layout_parts[0]} or matrix_b layout: {layout_parts[1]}"
    )
    assert layout_parts[2] == "r", (
        f"Invalid matrix_c layout: {layout_parts[2]} (must be 'r' for row major)"
    )

    kernel_name_prefix = "gemm_bquant"
    builder = GemmBQuantKernelBuilder(
        kernel_name_prefix,
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
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

        # Parse trait combo:
        # pipeline_epilogue_scheduler_padM_padN_padK_bPreshuffle
        trait_parts = args.trait_combo.split("_")
        if len(trait_parts) < 7:
            parser.error(
                f"--trait_combo must have 7 underscore-separated fields "
                f"(e.g. 'compv3_default_intrawave_True_True_True_False'), "
                f"got {len(trait_parts)} field(s): '{args.trait_combo}'"
            )
        trait_combo = (
            trait_parts[0],  # pipeline
            trait_parts[1],  # epilogue
            trait_parts[2],  # scheduler
            trait_parts[3] == "True",  # pad_m
            trait_parts[4] == "True",  # pad_n
            trait_parts[5] == "True",  # pad_k
            trait_parts[6] == "True",  # b_preshuffle_quant
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
