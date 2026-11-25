# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import json
import argparse
import importlib.util
import multiprocessing
import concurrent.futures


def _import_gemm_kernel_builder():
    """Load the parent builder and reuse its architecture validation table."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(parent_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder, gemm_builder_module._validation_utils


GemmKernelBuilder, _validation_utils = _import_gemm_kernel_builder()
GEMM_MX_PIPELINES_BY_ARCH = _validation_utils.GEMM_MX_PIPELINES_BY_ARCH
TDM_SCHEDULED_PIPELINE = _validation_utils.MX_TDM_SCHEDULED_PIPELINE
mx_tdm_scheduled = _validation_utils.load_mx_tdm_scheduled()


class MxGemmKernelBuilder(GemmKernelBuilder):
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

    def _generate_trait_combinations(self):
        arch = self.gpu_target.split(":", 1)[0]
        expected = GEMM_MX_PIPELINES_BY_ARCH.get(arch, ())
        return [
            combo
            for combo in super()._generate_trait_combinations()
            if combo[0] in expected
            and not (arch == "gfx1250" and (combo[5] or combo[6]))  # pad_k, persistent
        ]

    # comp_tdm_scheduled carries a tuning besides its tile and traits: the cluster, the policy's
    # knobs and the shape traits of the instance, from the optional "tuning_config" section (the
    # product of its values; the defaults when absent). Other pipelines take no tuning. It is built
    # only through the dispatcher (mx_gemm_full_benchmark.py); main() rejects it.

    def _scheduled_tunings(self):
        return mx_tdm_scheduled.expand_tuning_config(self.config.get("tuning_config"))

    def _scheduled_spec(self, tile_config, tuning):
        out = mx_tdm_scheduled.TE_OUT_DTYPES[self._c_datatype()]
        return mx_tdm_scheduled.spec_from_tile(
            tile_config, tuning, self.datatype, self.datatype, out
        )

    def _scheduled_reject_reason(self, tile_config, tuning):
        keys = ("tile", "warp", "warp_tile")
        dims = [tile_config[f"{k}_{d}"] for k in keys for d in "mnk"]
        return _validation_utils.mx_tdm_scheduled_reject_reason(
            *dims,
            self.datatype,
            self.datatype,
            self._c_datatype(),
            self.gpu_target,
            tuning,
        )

    def scheduled_name_suffix(self, tile_config, tuning):
        return mx_tdm_scheduled.instance_name_suffix(
            self._scheduled_spec(tile_config, tuning), tuning
        )

    def _get_sampled_kernel_list(self):
        """The parent's enumeration with every comp_tdm_scheduled entry expanded over the valid
        tunings (key "tuning"; None for other pipelines), then sampled. The dispatcher's
        expand_mx_sweep calls it."""
        max_instances, self.max_instances = self.max_instances, None
        try:
            kernels = super()._get_sampled_kernel_list()
        finally:
            self.max_instances = max_instances
        expanded = []
        for kernel in kernels:
            if kernel["trait_combo"][0] != TDM_SCHEDULED_PIPELINE:
                expanded.append({**kernel, "tuning": None})
                continue
            for tuning in self._scheduled_tunings():
                if self._scheduled_reject_reason(kernel["tile_config"], tuning):
                    continue
                suffix = self.scheduled_name_suffix(kernel["tile_config"], tuning)
                expanded.append(
                    {**kernel, "name": kernel["name"] + suffix, "tuning": tuning}
                )
        return self._apply_sampling(expanded)

    def _generate_kernel_instance(
        self, tile_config, trait_combo, validate=False, tuning=None
    ):
        if self._normalize_trait_combo(trait_combo)[0] == TDM_SCHEDULED_PIPELINE:
            return self._generate_scheduled_instance(tile_config, trait_combo, tuning)
        if tuning is not None and not mx_tdm_scheduled.is_default_tuning(tuning):
            raise ValueError(f"only {TDM_SCHEDULED_PIPELINE} takes a tuning")
        return super()._generate_kernel_instance(tile_config, trait_combo, validate)

    def _generate_scheduled_instance(self, tile_config, trait_combo, tuning):
        tuning = mx_tdm_scheduled.normalize_tuning(tuning)
        pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent = (
            self._normalize_trait_combo(trait_combo)
        )
        if not _validation_utils.is_trait_combination_valid(
            pipeline,
            epilogue,
            scheduler,
            persistent,
            self.kernel_name_prefix,
            self.layout,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            gpu_target=self.gpu_target,
        ):
            raise ValueError(
                f"{TDM_SCHEDULED_PIPELINE} needs an epilogue of "
                f"{_validation_utils.MX_TDM_SCHEDULED_EPILOGUES}, "
                "the intrawave scheduler, no padding and no persistence"
            )
        reason = self._scheduled_reject_reason(tile_config, tuning)
        if reason:
            raise ValueError(f"tile {tile_config} with tuning {tuning}: {reason}")
        spec = self._scheduled_spec(tile_config, tuning)
        kernel_name = self._format_kernel_name(
            trait_combo, tile_config
        ) + mx_tdm_scheduled.instance_name_suffix(spec, tuning)
        instance_code = self._scheduled_instance_code(
            kernel_name, tile_config, spec, tuning
        )

        simplified_name = kernel_name[len(self.kernel_name_prefix) + 1 :]
        header_file = (
            self.working_path
            / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)
        print(f"Generated {header_file}")
        return kernel_name, instance_code

    def _scheduled_instance_code(self, kernel_name, tile_config, spec, tuning):
        t = tile_config
        return f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <stdexcept>
#include <string>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
{self.populate_kernel_dtype_layout()}
// Kernel name for display
constexpr const char* KERNEL_NAME = "{kernel_name}";

// comp_tdm_scheduled (MxGemmPipelineAgBgCrCompTDMScheduled): K-major scales, E = A * B^T.
struct SelectedKernel {{
    static constexpr ck_tile::index_t BlockSize = 128;
    static constexpr ck_tile::index_t TileM = {t["tile_m"]};
    static constexpr ck_tile::index_t TileN = {t["tile_n"]};
    static constexpr ck_tile::index_t TileK = {t["tile_k"]};
    static constexpr ck_tile::index_t WarpPerBlock_M = {t["warp_m"]};
    static constexpr ck_tile::index_t WarpPerBlock_N = {t["warp_n"]};
    static constexpr ck_tile::index_t WarpPerBlock_K = 1;
    static constexpr ck_tile::index_t WarpTileM = {t["warp_tile_m"]};
    static constexpr ck_tile::index_t WarpTileN = 16;
    static constexpr ck_tile::index_t WarpTileK = 128;
    static constexpr bool Preshuffle = false;
    static constexpr bool KMajorScales = true;
    // The B reshuffle interface the profiler names (unused: B is not preshuffled).
    static constexpr ck_tile::index_t N_Warp_Tile = WarpTileN;
    static constexpr ck_tile::index_t K_Warp_Tile = WarpTileK;
    static constexpr ck_tile::index_t BContiguousItemsPerAccess =
        std::is_same_v<BDataType, ck_tile::pk_fp4_t> ? 32 : 16;
    static constexpr bool BPreshuffleLaneInterleavedK = true;

    using Cluster = {mx_tdm_scheduled.cpp_cluster_sequence(tuning)};
    using Policy = {mx_tdm_scheduled.cpp_policy_type(tuning)};
    using Traits = {mx_tdm_scheduled.cpp_kernel_traits(spec, tuning)};
    using Pipeline = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduled<
        ck_tile::MxGemmTdmScheduledProblem<
            ADataType,
            BDataType,
            CDataType,
            ck_tile::ClusterTileGemmShape<Cluster,
                                          ck_tile::sequence<TileM, TileN, TileK>,
                                          ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, 1>,
                                          ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>,
            Traits>,
        Policy>;
    // One kernel per TailNumber of the pipeline (Pipeline::WithTailNum).
    template <typename P>
    using KernelOf = ck_tile::MxGemmKernel<
        ck_tile::GemmTile2DNMajorPartitioner<typename P::BlockGemmShape>,
        P,
        ck_tile::MxTdmStoreEpilogue<P>>;

    // The pipeline's TailHandler picks the kernel of K's whole-tile count on the host.
    static float launch(const MxGemmHostArgs& args, const ck_tile::stream_config& stream)
    {{
        const ck_tile::index_t full_tiles = args.K / TileK;
        return Pipeline::TailHandler(
            [&](auto, auto tail_num) {{
                return launch_kernel<
                    KernelOf<typename Pipeline::template WithTailNum<decltype(tail_num)::value>>>(
                    args, stream);
            }},
            Pipeline::BlockHasHotloop(full_tiles),
            Pipeline::GetBlockLoopTailNum(full_tiles));
    }}

    template <typename Kernel>
    static float launch_kernel(const MxGemmHostArgs& args, const ck_tile::stream_config& stream)
    {{
        const auto kargs = Kernel::MakeKernelArgs(args);
        if(!Kernel::IsSupportedArgument(kargs))
        {{
            const char* why = Kernel::GetUnsupportedReason(kargs);
            throw std::runtime_error(std::string("Arguments not supported: ") +
                                     (why ? why : "see CK_TILE_LOGGING"));
        }}
        const dim3 grids  = Kernel::GridSize(args.M, args.N, args.k_batch);
        const dim3 blocks = Kernel::BlockSize();
        if constexpr(Kernel::ClusterLaunch)
        {{
#if CK_TILE_ENABLE_CLUSTER_LAUNCH
            return ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<1>(Kernel{{}}, Kernel::ClusterSize(), grids, blocks, 0, kargs));
#else
            throw std::runtime_error(Kernel::GetName() + ": this build has no cluster launch");
#endif
        }}
        else
        {{
            return ck_tile::launch_kernel(
                stream, ck_tile::make_kernel<1>(Kernel{{}}, grids, blocks, 0, kargs));
        }}
    }}
}};
"""

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation with parallel processing"""
        if num_workers is None:
            num_workers = min(
                multiprocessing.cpu_count(), 8
            )  # Limit to avoid memory issues

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        # Prepare work items for parallel processing
        work_items = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                if not self._validate_tile_config(
                    tile_config["tile_m"],
                    tile_config["tile_n"],
                    tile_config["tile_k"],
                    tile_config["warp_m"],
                    tile_config["warp_n"],
                    tile_config["warp_k"],
                    tile_config["warp_tile_m"],
                    tile_config["warp_tile_n"],
                    tile_config["warp_tile_k"],
                    trait_combo[0],
                ):
                    continue
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

        # Apply RFC-compliant sampling (Sobol + LHS + maximin)
        if self.max_instances is not None and len(work_items) > self.max_instances:
            kernel_dicts = [
                {"tile_config": item[0], "trait_combo": item[1], "_work_item": item}
                for item in work_items
            ]
            sampled = self._apply_sampling(kernel_dicts)
            work_items = [k["_work_item"] for k in sampled]

        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )
        print(f"  Tile configs: {len(tile_configs)}")
        print(f"  Trait combinations: {len(trait_combos)}")
        print(f"  Total kernels: {len(work_items)}")

        # Show first few work items for debugging
        if work_items:
            print("  First work item example:")
            tile_config, trait_combo = work_items[0][:2]
            print(f"    Tile config: {tile_config}")
            print(f"    Trait combo: {trait_combo[:3]}")  # Show first 3 traits

        # Process work items in parallel
        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # Submit all work items
            print(f"  Submitting {len(work_items)} tasks to executor...")
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }
            print("  All tasks submitted, waiting for completion...")

            # Collect results with progress reporting
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

        # Sort kernel list for consistent ordering
        kernel_list.sort(key=lambda x: x[0])  # Sort by kernel name

        # Generate CMake include file for individual targets
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

    # Create a temporary builder instance for this worker
    builder = MxGemmKernelBuilder(
        kernel_name_prefix, working_path, gpu_target, datatype, layout, config_json
    )

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        # Create simplified filename without the "mx_gemm_" prefix
        # Remove "mx_gemm_" from the beginning of kernel_name for the filename
        simplified_name = kernel_name
        if simplified_name.startswith("mx_gemm_"):
            simplified_name = simplified_name[
                len(kernel_name_prefix) + 1 :
            ]  # Remove "mx_gemm" prefix

        # Write individual header file
        header_file = working_path / f"mx_gemm_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="MX GEMM kernel instance builder with parallel support"
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
        choices=["fp4", "fp8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr"],
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

    kernel_name_prefix = "mx_gemm"
    if args.config_json:
        with open(args.config_json) as f:
            pipelines = (
                json.load(f)
                .get("trait_config", {})
                .get("pipeline", {})
                .get("values", [])
            )
        if TDM_SCHEDULED_PIPELINE in pipelines:
            sys.exit(
                f"{TDM_SCHEDULED_PIPELINE} runs through the dispatcher (mx_gemm_full_benchmark.py)"
            )
    builder = MxGemmKernelBuilder(
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
        # Generate a single kernel file input validation
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

        # Parse trait combo from the right so pipeline names may contain underscores.
        trait_parts = args.trait_combo.split("_")
        trait_combo = (
            "_".join(trait_parts[:-6]),  # pipeline
            trait_parts[-6],  # epilogue
            trait_parts[-5],  # scheduler
            trait_parts[-4] == "True",  # pad_m
            trait_parts[-3] == "True",  # pad_n
            trait_parts[-2] == "True",  # pad_k
            trait_parts[-1] == "True",  # persistent
        )

        # Generate the kernel
        builder._generate_kernel_instance(
            tile_config,
            trait_combo,
        )
    elif args.gen_all_individual:
        # Generate all individual kernel files
        builder._generate_all_individual(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_all_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
