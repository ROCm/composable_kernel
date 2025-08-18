import argparse
import logging
import itertools
from pathlib import Path
from typing import List, Optional
from gemm_preshuffle_config import JsonConfig, ArgumentConfig, RangeConfigParam
from commons.common_utils import (
    BOOL_MAP,
    element_size,
    get_gpu_name_by_id,
    trait_unsupported_combinations,
    warp_tile_supported_combinations,
)


class GemmPreshuffleCodeGenerator:
    def __init__(
        self,
        args: argparse.Namespace,
        user_provided_config: Optional[JsonConfig] = None,
    ):
        self.output_dir = Path(args.working_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if user_provided_config is not None:
            self.config = user_provided_config
        else:
            config_path = (
                Path(__file__).resolve().parent / "configs" / "default_config.json"
            )
            self.config = JsonConfig.from_json(config_path)

        self.args = ArgumentConfig.from_args(args.datatype, args.layout)

        self.valid_trait_names: List[str] = []
        self.valid_trait_tile_combinations: map[str, list[tuple[int]]] = {}

    def list_all_trait_names(self):
        """List all possible kernel trait names into file."""
        w_p = Path(self.output_dir)
        file_path = w_p / "gemm_preshuffle_instance_blobs.txt"
        self._generate_all_traits()
        self._get_valid_trait_tile_combinations()
        file_range_map = {}
        # Write all file paths to the header file
        files_listed = 0
        with file_path.open("w") as f:
            # Core files
            core_files = [
                "gemm_preshuffle_common.hpp",
                "gemm_preshuffle_instances.hpp",
                "gemm_preshuffle_dispatcher.hpp",
            ]
            for core_file in core_files:
                f.write(str(w_p / core_file) + "\n")
                files_listed += 1

            # Trait header files
            for trait in self.valid_trait_names:
                trait_file = f"gemm_preshuffle_{trait}.hpp"
                f.write(str(w_p / trait_file) + "\n")
                files_listed += 1
            file_name = set()
            # Instance source files
            for trait, tile_valid_params in self.valid_trait_tile_combinations.items():
                start_idx = files_listed
                for tile in tile_valid_params:
                    for (
                        tile_m,
                        tile_n,
                        tile_k,
                        warp_m,
                        warp_n,
                        warp_k,
                        _,
                        _,
                        _,
                    ) in tile:
                        instance_name = f"gemm_preshuffle_{trait}_{tile_m}x{tile_n}x{tile_k}_{warp_m}x{warp_n}x{warp_k}.cpp"

                        if instance_name not in file_name:
                            file_name.add(instance_name)
                            f.write(str(w_p / instance_name) + "\n")
                            files_listed += 1

                file_range_map[trait] = (start_idx, files_listed)

        file_path = w_p / "gemm_preshuffle_instance_blobs_range.txt"
        with file_path.open("w") as f:
            for name, ranges in file_range_map.items():
                start, last = ranges
                f.write(name + " " + f"{start}" + " " + f"{last}" + "\n")

    def _generate_all_traits(self):
        params = [
            "pipeline",
            "epilogue",
            "scheduler",
            "pad_m",
            "pad_n",
            "pad_k",
            "persistent",
        ]

        # Generate all unique_combinations
        _unique = set(
            itertools.product(
                *[getattr(self.config.trait_config, param).values for param in params]
            )
        )

        for combo in _unique:
            pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent = combo
            current_combination = (pipeline, epilogue, scheduler)

            if current_combination not in trait_unsupported_combinations:
                trait_name = (
                    f"{pipeline}_{epilogue}_{scheduler}_"
                    f"{BOOL_MAP(pad_m)}_{BOOL_MAP(pad_n)}_{BOOL_MAP(pad_k)}_"
                    f"{BOOL_MAP(persistent)}"
                )
                self.valid_trait_names.append(trait_name)
            else:
                logging.debug(f"Invalid combination: {pipeline}-{epilogue}-{scheduler}")

    def _get_valid_trait_tile_combinations(self):
        def get_tile_value(tile_param):
            return (
                tile_param.generate_candidates()
                if isinstance(tile_param, RangeConfigParam)
                else tile_param.values
            )

        tile_group = list(
            itertools.product(
                get_tile_value(self.config.tile_config.tile_m),
                get_tile_value(self.config.tile_config.tile_n),
                get_tile_value(self.config.tile_config.tile_k),
            )
        )

        warp_group = list(
            itertools.product(
                get_tile_value(self.config.tile_config.warp_m),
                get_tile_value(self.config.tile_config.warp_n),
                get_tile_value(self.config.tile_config.warp_k),
            )
        )

        warp_tile_group = list(
            itertools.product(
                get_tile_value(self.config.tile_config.warp_tile_m),
                get_tile_value(self.config.tile_config.warp_tile_n),
                get_tile_value(self.config.tile_config.warp_tile_k),
            )
        )

        tile_params = {
            t + w + wt for t in tile_group for w in warp_group for wt in warp_tile_group
        }

        print("[DELETE] Tile params:", tile_params)
        print("[DELETE] valid_trait_names:", self.valid_trait_names)

        for trait in self.valid_trait_names:
            tile_valid_params = [
                tile for tile in tile_params if self.is_tile_valid(tile, trait)
            ]

            if trait not in self.valid_trait_tile_combinations:
                self.valid_trait_tile_combinations[trait] = []
            self.valid_trait_tile_combinations[trait].append(tile_valid_params)

        print("[DELETE] tile_valid_params:", tile_valid_params)

    def is_tile_valid(self, tile: tuple, trait: str) -> bool:
        """Check if the tile configuration is valid for the given trait."""
        (
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            warp_k,
            warp_tile_m,
            warp_tile_n,
            warp_tile_k,
        ) = tile
        pipeline, *_ = trait.split("_")

        # Parameter validity check #DELETE WHY IS THIS EXACTLY THIS?
        invalid_params = []
        if (warp_m, warp_n, warp_k) not in [(1, 4, 1), (2, 2, 1), (4, 1, 1)]:
            invalid_params.append(
                f"warp_m({warp_m}) * warp_n({warp_n}) * warp_k({warp_k})"
            )
        if (warp_m * warp_tile_m) == 0:
            invalid_params.append(f"warp_m({warp_m}) * warp_tile_m({warp_tile_m})")
        if (warp_n * warp_tile_n) == 0:
            invalid_params.append(f"warp_n({warp_n}) * warp_tile_n({warp_tile_n})")
        if (warp_k * warp_tile_k) == 0:
            invalid_params.append(f"warp_k({warp_k}) * warp_tile_k({warp_tile_k})")

        if invalid_params:
            logging.debug(
                f"Trait: [{trait}], Invalid warp configuration: {', '.join(invalid_params)}. "
                f"Parameter combination: warp=({warp_m},{warp_n},{warp_k}), "
                f"warp_tile=({warp_tile_m},{warp_tile_n},{warp_tile_k})"
            )
            return False

        # Dimension alignment check #DELETE WHY IS THIS EXACTLY THIS? I think it is because of dividing equally
        alignment_issues = []
        if tile_m % (warp_m * warp_tile_m) != 0:
            alignment_issues.append(
                f"tile_m({tile_m}) % [{warp_m}x{warp_tile_m}] = {tile_m % (warp_m * warp_tile_m)}"
            )
        if tile_n % (warp_n * warp_tile_n) != 0:
            alignment_issues.append(
                f"tile_n({tile_n}) % [{warp_n}x{warp_tile_n}] = {tile_n % (warp_n * warp_tile_n)}"
            )
        if tile_k % (warp_k * warp_tile_k) != 0:
            alignment_issues.append(
                f"tile_k({tile_k}) % [{warp_k}x{warp_tile_k}] = {tile_k % (warp_k * warp_tile_k)}"
            )

        if alignment_issues:
            logging.debug(
                f"Trait: [{trait}], Dimension alignment failed: {', '.join(alignment_issues)}. "
                f"Tile dimensions {tile_m}x{tile_n}x{tile_k} must be divisible by "
                f"[warp]: {warp_m}x{warp_n}x{warp_k} x [warp_tile]: {warp_tile_m}x{warp_tile_n}x{warp_tile_k}"
            )
            return False

        # LDS capacity verification
        matrix_a_size = (tile_m * tile_k) * element_size(self.args.datatypes.a_datatype)
        matrix_b_size = (tile_n * tile_k) * element_size(self.args.datatypes.b_datatype)
        total_tile_in_lds = matrix_a_size + matrix_b_size

        max_tile_size = (
            2**15 if pipeline == "compv4" else 2**16
        )  # DELETE WHY IS THIS EXACTLY THIS? I think it is because of dividing equally

        if total_tile_in_lds > max_tile_size:
            logging.debug(
                f"LDS capacity exceeded [{trait}]: Total required {total_tile_in_lds:,}B ({total_tile_in_lds / 1024:.1f}KB) > "
                f"maximum allowed {max_tile_size:,}B ({max_tile_size / 1024}KB). Breakdown:\n"
                f"- Matrix A ({self.args.datatypes.a_datatype}): {tile_m}x{tile_k} = {matrix_a_size:,}B\n"
                f"- Matrix B ({self.args.datatypes.b_datatype}): {tile_n}x{tile_k} = {matrix_b_size:,}B"
            )
            return False

        # Warp combination validation
        warp_tile_key = f"{self.args.datatypes.a_datatype}_{self.args.datatypes.b_datatype}_{self.args.datatypes.c_datatype}"
        current_combination = [warp_tile_m, warp_tile_n, warp_tile_k]

        gpu_name = get_gpu_name_by_id(0)
        gpu_warp_tile_key = warp_tile_supported_combinations.get(gpu_name, {})

        # print("[DELETE] warp_tile_key:", warp_tile_key)
        # print("[DELETE] current_combination:", current_combination)
        # print("[DELETE] gpu_name:", gpu_name)
        # print("[DELETE] gpu_warp_tile_key:", gpu_warp_tile_key)

        if not gpu_warp_tile_key:
            logging.debug(
                f"Trait: [{trait}], No valid warp tile combinations found for {gpu_name}/{warp_tile_key}, skip this check."
            )
            return False

        allowed_combinations = gpu_warp_tile_key.get(warp_tile_key, [])
        # print("[DELETE] I am here")
        # print("[DELETE] allowed_combinations:", allowed_combinations)

        if not allowed_combinations:
            logging.debug(
                f"Trait: [{trait}], No valid warp tile combinations found for {gpu_name}/{warp_tile_key}, skip this check."
            )
            return False

        if current_combination not in allowed_combinations:
            logging.debug(
                f"Trait: [{trait}], Invalid warp combination: {current_combination} not in allowed list. "
                f"Valid combinations for data type '{warp_tile_key}': {allowed_combinations}"
            )
            return False

        return True


def do_list_blobs(
    args: argparse.Namespace, user_provide_config: Optional[JsonConfig] = None
):
    print("I am at this point")
    generator = GemmPreshuffleCodeGenerator(args, user_provide_config)
    generator.list_all_trait_names()


def do_gen_blobs(
    args: argparse.Namespace, user_provide_config: Optional[JsonConfig] = None
):
    print("In do_gen_blobs")
    # generator = GemmCodeGenerator(args, user_provide_config)
    # generator.generate_all_instance_files()


def main(args):
    gemm_preshuffle_config = JsonConfig.from_json(args.config_json)

    if args.list_blobs:
        do_list_blobs(args, gemm_preshuffle_config)
    elif args.gen_blobs:
        do_gen_blobs(args, gemm_preshuffle_config)
    else:
        logging.warning(
            "No mode specified (use --list_blobs or --gen_blobs). Generating by default..."
        )
        do_gen_blobs(args, gemm_preshuffle_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK GEMM Preshuffle kernel",
    )
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="The path where all the blobs are going to be generated",
    )
    parser.add_argument(
        "-j",
        "--config_json",
        required=False,
        help="Path to JSON file containing user-specified kernel configurations",
    )
    parser.add_argument(
        "-d",
        "--datatype",
        required=True,
        help="Data type for kernel generation (supported: fp16, bf16, int8, fp8, bf8)",
    )
    parser.add_argument(
        "-ly",
        "--layout",
        required=True,
        help="Matrix layout configuration for kernel generation (e.g., rcr, rrr)",
    )
    parser.add_argument(
        "-l",
        "--list_blobs",
        action="store_true",
        help="List all available kernel instances to a file",
    )
    parser.add_argument(
        "-g",
        "--gen_blobs",
        action="store_true",
        help="Generate all kernel instances into different files",
    )

    args = parser.parse_args()

    main(args)
