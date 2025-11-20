import os
import json
from pathlib import Path
import importlib.util
import itertools
import logging


def _import_validation_utils():
    """Import validation utilities from commons directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(
        "validation_utils",
        os.path.join(parent_dir, "gemm", "gemm_validation_utils.py"),
    )
    validation_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_utils)

    return validation_utils


# Import validation functions
_validation_utils = _import_validation_utils()
is_tile_config_valid = _validation_utils.is_tile_config_valid
is_trait_combination_valid = _validation_utils.is_trait_combination_valid


class GemmKernelBuilder:
    def __init__(self, working_path, gpu_target, datatype, layout, config_json=None):
        self.working_path = Path(working_path)
        self.gpu_target = gpu_target
        self.datatype = datatype
        self.layout = layout
        self.config_json = config_json

        # Create working directory if it doesn't exist
        self.working_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        if config_json and os.path.exists(config_json):
            with open(config_json, "r") as f:
                self.config = json.load(f)

    def list_kernels(self, kernel_name_prefix):
        """Write kernel list to file for CMake to read (with comprehensive validation)"""
        # Get configurations using comprehensive validation
        tile_configs = self._get_tile_configs(kernel_name_prefix)
        trait_combos = self._generate_trait_combinations()

        print(f"[NEW] Generated {len(tile_configs)} tile configurations")
        print(f"[NEW] Generated {len(trait_combos)} trait combinations")

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
                kernel_name = f"{kernel_name_prefix}_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_{str(persistent).capitalize()}"

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
        with open(
            self.working_path / "{kernel_name_prefix}_kernel_count.txt", "w"
        ) as f:
            f.write(str(len(kernel_list)))

        # Write kernel list
        with open(self.working_path / "{kernel_name_prefix}_kernel_list.txt", "w") as f:
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

    def _get_tile_configs(self, kernel_name_prefix, fast_mode=False):
        """Get tile configurations for the current datatype and layout"""

        tile_config = self.config["tile_config"]

        # Generate values in the config if default range is given
        if tile_config.get("tile_m").get("values") is None:
            tile_config.get("tile_m")["values"] = self._generate_values(
                tile_config.get("tile_m").get("min"),
                tile_config.get("tile_m").get("max"),
                tile_config.get("tile_m").get("step"),
            )
        if tile_config.get("tile_n").get("values") is None:
            tile_config.get("tile_n")["values"] = self._generate_values(
                tile_config.get("tile_n").get("min"),
                tile_config.get("tile_n").get("max"),
                tile_config.get("tile_n").get("step"),
            )
        if tile_config.get("tile_k").get("values") is None:
            tile_config.get("tile_k")["values"] = self._generate_values(
                tile_config.get("tile_k").get("min"),
                tile_config.get("tile_k").get("max"),
                tile_config.get("tile_k").get("step"),
            )

        # Get all possible values for each parameter
        tile_m_values = tile_config.get("tile_m").get("values")
        tile_n_values = tile_config.get("tile_n").get("values")
        tile_k_values = tile_config.get("tile_k").get("values")
        warp_m_values = tile_config.get("warp_m").get("values")
        warp_n_values = tile_config.get("warp_n").get("values")
        warp_k_values = tile_config.get("warp_k").get("values")
        warp_tile_m_values = tile_config.get("warp_tile_m").get("values")
        warp_tile_n_values = tile_config.get("warp_tile_n").get("values")
        warp_tile_k_values = tile_config.get("warp_tile_k").get("values")

        # Generate all combinations
        default_pipeline = ""
        if kernel_name_prefix == "gemm_universal":
            default_pipeline = "compv4"
        elif kernel_name_prefix == "gemm_multi_d":
            default_pipeline = "compv4"
        elif kernel_name_prefix == "gemm_preshuffle":
            default_pipeline = "preshufflev2"

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
                                                default_pipeline,
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

    def _generate_values(self, min_val, max_val, step):
        """Generate a list of values from min to max with the given step"""
        values = []
        val = min_val
        while val <= max_val:
            values.append(val)
            val += step
        return values

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
        pipeline,
        fast_mode=False,
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
            # Validate preshuffle specific constraints
            if (
                self.config.get("permute_n") is not None
                and self.config.get("permute_n") is True
            ):
                valid = (tile_n / warp_tile_n / warp_n) % 2 == 0
                if not valid:
                    return False

            # Full validation for generation
            # Determine data types for validation
            a_datatype = self.datatype
            b_datatype = self.datatype
            c_datatype = self.datatype

            layout = self.layout

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
                layout,
                self.gpu_target,
            )

    def _generate_trait_combinations(self):
        """Generate all combinations of traits"""

        trait_config = self.config["trait_config"]

        pipelines = trait_config.get("pipeline").get("values")
        epilogues = trait_config.get("epilogue").get("values")
        schedulers = trait_config.get("scheduler").get("values")
        pad_m_values = trait_config.get("pad_m").get("values")
        pad_n_values = trait_config.get("pad_n").get("values")
        pad_k_values = trait_config.get("pad_k").get("values")
        persistent_values = trait_config.get("persistent").get("values")

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
        return combinations
