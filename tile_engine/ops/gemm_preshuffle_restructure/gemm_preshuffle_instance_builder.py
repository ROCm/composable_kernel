import argparse
import os
import json
import itertools
import logging

from pathlib import Path

from commons.validation_utils import is_tile_config_valid, is_trait_combination_valid


class GemmPreshuffleKernelBuilder:
    def __init__(self, working_path, datatype, layout, config_json=None):
        self.working_path = Path(working_path)
        self.datatype = datatype
        self.layout = layout
        self.config_json = config_json

        # Create working directory if it doesn't exist
        self.working_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_json and os.path.exists(config_json):
            with open(config_json, "r") as f:
                self.config = json.load(f)
        else:
            print(
                "No config JSON provided or file does not exist. Using default configuration."
            )
            self.config = self._get_default_config()
        print("[DELETE] Printing", self.config)

    # [DELETE] Currently only support fp16 and fp8 needs to add more
    def _get_default_config(self):
        """Return default configuration if no config file is provided"""
        # Define base tile configurations that work for all layouts
        base_fp16_configs = [
            {
                "tile_m": 256,
                "tile_n": 256,
                "tile_k": 32,
                "warp_m": 2,
                "warp_n": 2,
                "warp_k": 1,
                "warp_tile_m": 32,
                "warp_tile_n": 32,
                "warp_tile_k": 32,
            },
            {
                "tile_m": 256,
                "tile_n": 128,
                "tile_k": 32,
                "warp_m": 2,
                "warp_n": 2,
                "warp_k": 1,
                "warp_tile_m": 32,
                "warp_tile_n": 32,
                "warp_tile_k": 16,
            },
        ]

        base_fp8_configs = [
            {
                "tile_m": 256,
                "tile_n": 256,
                "tile_k": 32,
                "warp_m": 4,
                "warp_n": 1,
                "warp_k": 1,
                "warp_tile_m": 32,
                "warp_tile_n": 32,
                "warp_tile_k": 32,
            },
            {
                "tile_m": 256,
                "tile_n": 128,
                "tile_k": 32,
                "warp_m": 1,
                "warp_n": 4,
                "warp_k": 1,
                "warp_tile_m": 16,
                "warp_tile_n": 16,
                "warp_tile_k": 32,
            },
        ]

        # Create configurations for all supported layouts
        all_layouts = ["rcr", "rrr", "ccr", "crr"]
        tile_configs = {}

        for datatype, base_configs in [
            ("fp16", base_fp16_configs),
            ("fp8", base_fp8_configs),
        ]:
            tile_configs[datatype] = {}
            for layout in all_layouts:
                tile_configs[datatype][layout] = base_configs

        return {
            "tile_configs": tile_configs,
            "traits": {
                "pipelines": ["mem", "compv3", "compv4"],
                "epilogues": ["default", "cshuffle"],
                "schedulers": ["intrawave", "interwave"],
            },
            "structured_sparsity": ["false"],
            "padding": {"pad_m": ["false"], "pad_n": ["false"], "pad_k": ["false"]},
            "persistent": ["false"],
            "tunable_params": {
                "kBlockPerCu": 2,  # [DELETE] Address this later
            },
        }

    def write_kernel_list(self):
        """Write kernel list to file for CMake to read (with comprehensive validation)"""
        # Get configurations using comprehensive validation
        tile_configs = self._get_tile_configs(fast_mode=False)
        trait_combos = self._generate_trait_combinations()

        kernel_list = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                (
                    pipeline,
                    epilogue,
                    scheduler,
                    pad_m,
                    pad_n,
                    pad_k,
                    persistent,
                ) = trait_combo

                # Create kernel name with proper boolean capitalization
                kernel_name = f"gemm_preshuffle_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_{str(persistent).capitalize()}"

                # Create tile configuration string
                tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
                tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
                tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

                kernel_name += f"_{tile_str}"

                kernel_list.append(
                    {
                        "name": kernel_name,
                        "tile_config": tile_config,
                        "trait_combo": trait_combo,
                    }
                )

        # Write kernel count
        with open(self.working_path / "gemm_preshuffle_kernel_count.txt", "w") as f:
            f.write(str(len(kernel_list)))

        # Write kernel list
        with open(self.working_path / "gemm_preshuffle_kernel_list.txt", "w") as f:
            for kernel in kernel_list:
                # Format: kernel_name|tile_config|trait_combo
                tile_config = kernel["tile_config"]
                trait_combo = kernel["trait_combo"]

                tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
                tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
                tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

                trait_str = (
                    f"{trait_combo[0]}_{trait_combo[1]}_{trait_combo[2]}_"
                    + "_".join(str(x) for x in trait_combo[3:])
                )

                f.write(f"{kernel['name']}|{tile_str}|{trait_str}\n")

        print(f"Listed {len(kernel_list)} kernel configurations")

    def _get_tile_configs(self, fast_mode=False):
        """Get tile configurations for the current datatype and layout"""
        if "tile_configs" in self.config:
            # Old format
            return (
                self.config["tile_configs"].get(self.datatype, {}).get(self.layout, [])
            )
        elif "tile_config" in self.config:
            # New format - generate combinations from individual parameter values
            tile_config = self.config["tile_config"]

            # Get all possible values for each parameter
            tile_m_values = tile_config.get("tile_m", {}).get("values", [256])
            tile_n_values = tile_config.get("tile_n", {}).get("values", [256])
            tile_k_values = tile_config.get("tile_k", {}).get("values", [32])
            warp_m_values = tile_config.get("warp_m", {}).get("values", [2])
            warp_n_values = tile_config.get("warp_n", {}).get("values", [2])
            warp_k_values = tile_config.get("warp_k", {}).get("values", [1])
            warp_tile_m_values = tile_config.get("warp_tile_m", {}).get("values", [32])
            warp_tile_n_values = tile_config.get("warp_tile_n", {}).get("values", [32])
            warp_tile_k_values = tile_config.get("warp_tile_k", {}).get("values", [32])

            # Generate all combinations
            configs = []
            for tile_m in tile_m_values:
                for tile_n in tile_n_values:
                    for tile_k in tile_k_values:
                        for warp_m in warp_m_values:
                            for warp_n in warp_n_values:
                                for warp_k in warp_k_values:
                                    for warp_tile_m in warp_tile_m_values:
                                        for warp_tile_n in warp_tile_n_values:
                                            for warp_tile_k in warp_tile_k_values:
                                                # Validate configuration
                                                if self._validate_tile_config(
                                                    tile_m,
                                                    tile_n,
                                                    tile_k,
                                                    warp_m,
                                                    warp_n,
                                                    warp_k,
                                                    warp_tile_m,
                                                    warp_tile_n,
                                                    warp_tile_k,
                                                    fast_mode=fast_mode,
                                                ):
                                                    configs.append(
                                                        {
                                                            "tile_m": tile_m,
                                                            "tile_n": tile_n,
                                                            "tile_k": tile_k,
                                                            "warp_m": warp_m,
                                                            "warp_n": warp_n,
                                                            "warp_k": warp_k,
                                                            "warp_tile_m": warp_tile_m,
                                                            "warp_tile_n": warp_tile_n,
                                                            "warp_tile_k": warp_tile_k,
                                                        }
                                                    )
            return configs
        else:
            # Fallback to default
            return []

    def _generate_trait_combinations(self):  # [DELETE] Look into the function name
        """Generate all combinations of traits"""
        if "traits" in self.config:
            # Old format
            traits = self.config["traits"]
            pipelines = traits["pipelines"]
            epilogues = traits["epilogues"]
            schedulers = traits["schedulers"]

            padding = self.config["padding"]
            persistent = self.config["persistent"]

            all_combinations = list(
                itertools.product(
                    pipelines,
                    epilogues,
                    schedulers,
                    padding["pad_m"],
                    padding["pad_n"],
                    padding["pad_k"],
                    persistent,
                )
            )

            # Filter out unsupported trait combinations
            combinations = []
            for combo in all_combinations:
                pipeline, epilogue, scheduler = combo[:3]
                if is_trait_combination_valid(pipeline, epilogue, scheduler):
                    combinations.append(combo)
                else:
                    logging.debug(
                        f"Skipping unsupported trait combination: {pipeline}-{epilogue}-{scheduler}"
                    )

        elif "trait_config" in self.config:
            # New format
            trait_config = self.config["trait_config"]

            pipelines = trait_config.get("pipeline", {}).get("values", ["mem"])
            epilogues = trait_config.get("epilogue", {}).get("values", ["default"])
            schedulers = trait_config.get("scheduler", {}).get("values", ["intrawave"])
            pad_m_values = trait_config.get("pad_m", {}).get("values", [False])
            pad_n_values = trait_config.get("pad_n", {}).get("values", [False])
            pad_k_values = trait_config.get("pad_k", {}).get("values", [False])
            persistent_values = trait_config.get("persistent", {}).get(
                "values", [False]
            )

            all_combinations = list(
                itertools.product(
                    pipelines,
                    epilogues,
                    schedulers,
                    pad_m_values,
                    pad_n_values,
                    pad_k_values,
                    persistent_values,
                )
            )

            # Filter out unsupported trait combinations
            combinations = []
            for combo in all_combinations:
                pipeline, epilogue, scheduler = combo[:3]
                if is_trait_combination_valid(pipeline, epilogue, scheduler):
                    combinations.append(combo)
                else:
                    logging.debug(
                        f"Skipping unsupported trait combination: {pipeline}-{epilogue}-{scheduler}"
                    )
        else:
            # Fallback to minimal default
            combinations = [
                ("preshufflev2", "default", "intrawave", False, False, False, False)
            ]

        return combinations

    def _validate_tile_config(
        self,
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
        pipeline="mem",  # Default pipeline for validation
        fast_mode=False,  # Add fast mode option
    ):
        """Validate that tile configuration is reasonable"""
        if fast_mode:
            # Fast validation for listing - only basic sanity checks
            if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
                return False
            if warp_m <= 0 or warp_n <= 0 or warp_k <= 0:
                return False
            if warp_tile_m <= 0 or warp_tile_n <= 0 or warp_tile_k <= 0:
                return False

            # Basic divisibility check
            if tile_m % (warp_m * warp_tile_m) != 0:
                return False
            if tile_n % (warp_n * warp_tile_n) != 0:
                return False
            if tile_k % (warp_k * warp_tile_k) != 0:
                return False

            return True
        else:
            # Full validation for generation
            # Determine data types for validation
            a_datatype = self.datatype
            b_datatype = self.datatype
            c_datatype = self.datatype

            # Special handling for certain data types
            if self.datatype in ["fp8", "bf8"]:
                c_datatype = "fp16"

            # Use the comprehensive validation function
            return is_tile_config_valid(
                tile_m,
                tile_n,
                tile_k,
                warp_m,
                warp_n,
                warp_k,
                warp_tile_m,
                warp_tile_n,
                warp_tile_k,
                a_datatype,
                b_datatype,
                c_datatype,
                pipeline,
            )


def main():
    parser = argparse.ArgumentParser(
        description="GEMM kernel instance builder with parallel support"
    )
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16", "fp8", "bf16", "bf8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr", "rrr", "ccr", "crr"],
        help="Matrix layout",
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

    # [DELETE] Add Validation for datatype and layout here
    # validation()

    # Create builder
    builder = GemmPreshuffleKernelBuilder(
        args.working_path, args.datatype, args.layout, args.config_json
    )

    if args.list_kernels:
        # Fast listing mode - just write kernel list without generating files
        builder.write_kernel_list()
        pass
    elif args.gen_single:
        """
        # Generate a single kernel file
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

        # Generate the kernel
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        # Write the file
        simplified_name = kernel_name
        if simplified_name.startswith("gemm_"):
            simplified_name = simplified_name[5:]

        header_file = builder.working_path / f"gemm_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}") """
        pass

    elif args.gen_individual:
        # Generate all individual kernel files
        # builder.run(args.num_workers)
        pass
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
