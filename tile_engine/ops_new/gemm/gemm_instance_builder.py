import os
import json
from pathlib import Path


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

    def write_kernel_list(self, kernel_name_prefix):
        """Write kernel list to file for CMake to read (with comprehensive validation)"""
        # Get configurations using comprehensive validation
        # tile_configs = self._get_tile_configs(fast_mode=False)
        # trait_combos = self._generate_trait_combinations()

        # kernel_list = []
        # for tile_config in tile_configs:
        #     for trait_combo in trait_combos:
        #         (
        #             pipeline,
        #             epilogue,
        #             scheduler,
        #             pad_m,
        #             pad_n,
        #             pad_k,
        #             persistent,
        #         ) = trait_combo

        #         # Create kernel name with proper boolean capitalization
        #         kernel_name = f"{kernel_name_prefix}_{self.datatype}_{self.layout}_{pipeline}_{epilogue}_{scheduler}_{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_{str(persistent).capitalize()}"

        #         # Create tile configuration string
        #         tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
        #         tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
        #         tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

        #         kernel_name += f"_{tile_str}"

        #         kernel_list.append(
        #             {
        #                 "name": kernel_name,
        #                 "tile_config": tile_config,
        #                 "trait_combo": trait_combo,
        #             }
        #         )

        # # Write kernel count
        # with open(self.working_path / "{kernel_name_prefix}_kernel_count.txt", "w") as f:
        #     f.write(str(len(kernel_list)))

        # # Write kernel list
        # with open(self.working_path / "{kernel_name_prefix}_kernel_list.txt", "w") as f:
        #     for kernel in kernel_list:
        #         # Format: kernel_name|tile_config|trait_combo
        #         tile_config = kernel["tile_config"]
        #         trait_combo = kernel["trait_combo"]

        #         tile_str = f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
        #         tile_str += f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
        #         tile_str += f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"

        #         trait_str = (
        #             f"{trait_combo[0]}_{trait_combo[1]}_{trait_combo[2]}_"
        #             + "_".join(str(x) for x in trait_combo[3:])
        #         )

        #         f.write(f"{kernel['name']}|{tile_str}|{trait_str}\n")

        # print(f"Listed {len(kernel_list)} kernel configurations")
