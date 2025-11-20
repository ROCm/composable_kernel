import os
import argparse
import importlib.util


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


class GemmUniversalKernelBuilder(GemmKernelBuilder):
    def __init__(self, working_path, gpu_target, datatype, layout, config_json=None):
        super().__init__(working_path, gpu_target, datatype, layout, config_json)


def main():
    parser = argparse.ArgumentParser(
        description="GEMM Universal kernel instance builder with parallel support"
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

    args = parser.parse_args()

    assert args.datatype in ["fp16", "bf16", "fp8", "bf8"], (
        f"Invalid datatype string: {args.datatype} (supported datatypes are [fp16, bf16, fp8, and bf8])"
    )

    layout_parts = args.layout.lower()
    assert len(layout_parts) == 3, (
        f"Invalid layout string: {args.layout} (must be 3 characters like 'rcr' where r stands for row major and c stands for column major)"
    )
    assert layout_parts[0] in ["r", "c"] and layout_parts[1] in ["r", "c"], (
        f"Invalid matrix_a layout : {layout_parts[0]} or matrix_b layout: {layout_parts[1]} (matrix_a and matrix_b must be either 'r' for row major or 'c' for column major)"
    )
    assert layout_parts[2] == "r", (
        f"Invalid matrix_c layout: {layout_parts[2]} (must be 'r' only as currently we are supporting only row major)"
    )

    builder = GemmUniversalKernelBuilder(
        args.working_path, args.gpu_target, args.datatype, args.layout, args.config_json
    )

    if args.list_kernels:
        builder.list_kernels("gemm_universal")
    elif args.gen_single:
        # # Generate a single kernel file
        # if not args.kernel_name or not args.tile_config or not args.trait_combo:
        #     parser.error(
        #         "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
        #     )

        # # Parse tile config
        # tile_parts = args.tile_config.split("_")
        # tile_dims = tile_parts[0].split("x")
        # warp_dims = tile_parts[1].split("x")
        # warp_tile_dims = tile_parts[2].split("x")

        # tile_config = {
        #     "tile_m": int(tile_dims[0]),
        #     "tile_n": int(tile_dims[1]),
        #     "tile_k": int(tile_dims[2]),
        #     "warp_m": int(warp_dims[0]),
        #     "warp_n": int(warp_dims[1]),
        #     "warp_k": int(warp_dims[2]),
        #     "warp_tile_m": int(warp_tile_dims[0]),
        #     "warp_tile_n": int(warp_tile_dims[1]),
        #     "warp_tile_k": int(warp_tile_dims[2]),
        # }

        # # Parse trait combo
        # trait_parts = args.trait_combo.split("_")
        # trait_combo = (
        #     trait_parts[0],  # pipeline
        #     trait_parts[1],  # epilogue
        #     trait_parts[2],  # scheduler
        #     trait_parts[3] == "True",  # pad_m
        #     trait_parts[4] == "True",  # pad_n
        #     trait_parts[5] == "True",  # pad_k
        #     trait_parts[6] == "True",  # persistent
        # )

        # k_block_per_cu = builder.config.get("k_block_per_cu")
        # if k_block_per_cu is None:
        #     k_block_per_cu = 1

        # # Generate the kernel
        # kernel_name, instance_code = builder._generate_kernel_instance(
        #     tile_config, trait_combo, k_block_per_cu
        # )

        # # Write the file
        # simplified_name = kernel_name
        # if simplified_name.startswith("gemm_"):
        #     simplified_name = simplified_name[5:]

        # header_file = builder.working_path / f"gemm_single_{simplified_name}.hpp"
        # with open(header_file, "w") as f:
        #     f.write(instance_code)

        # print(f"Generated {header_file}")
        pass
    elif args.gen_all_individual:
        # Generate all individual kernel files
        # builder.run(args.num_workers)
        pass
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_all_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
