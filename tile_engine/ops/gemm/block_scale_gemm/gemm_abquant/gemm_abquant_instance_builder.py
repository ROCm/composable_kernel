# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import argparse
import importlib.util
import multiprocessing
import concurrent.futures
import itertools
import logging


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


def _import_validation_utils():
    """Import validation utilities."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gemm_dir = os.path.dirname(os.path.dirname(current_dir))
    parent_dir = os.path.dirname(gemm_dir)

    spec = importlib.util.spec_from_file_location(
        "validation_utils",
        os.path.join(parent_dir, "gemm", "gemm_validation_utils.py"),
    )
    validation_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation_utils)

    return validation_utils


GemmKernelBuilder = _import_gemm_kernel_builder()
_validation_utils = _import_validation_utils()
is_trait_combination_valid = _validation_utils.is_trait_combination_valid


class GemmABQuantKernelBuilder(GemmKernelBuilder):
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
        # group_size_n can be an int or a dict with "values" key
        group_size_n_cfg = self.config.get("group_size_n", 1)
        if isinstance(group_size_n_cfg, dict):
            self.group_size_n_values = group_size_n_cfg.get("values", [1])
        else:
            self.group_size_n_values = [group_size_n_cfg]

    @staticmethod
    def _tile_config_to_str(tile_config):
        return (
            f"{tile_config['tile_m']}x{tile_config['tile_n']}x{tile_config['tile_k']}_"
            f"{tile_config['warp_m']}x{tile_config['warp_n']}x{tile_config['warp_k']}_"
            f"{tile_config['warp_tile_m']}x{tile_config['warp_tile_n']}x{tile_config['warp_tile_k']}"
        )

    def _apply_sampling(self, kernel_list):
        """Apply RFC Sobol+LHS+maximin sampling for ABQuant kernels.

        Overrides base class to handle ABQuant's 8-tuple trait_combo
        and group_size_n field.
        """
        if self.max_instances is None or len(kernel_list) <= self.max_instances:
            return kernel_list

        import sys

        sampling_parent = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", ".."
        )
        if sampling_parent not in sys.path:
            sys.path.insert(0, sampling_parent)

        from sampling.sampler import sample_feasible_set
        from sampling.seed import make_seed
        from sampling.feasible_set import GEMM_ABQUANT_AXES

        effective_seed = make_seed(
            self.seed, self.gpu_target, self.datatype, self.layout
        )

        flat_items = []
        for k in kernel_list:
            flat = dict(k["tile_config"])
            (
                pipeline,
                epilogue,
                scheduler,
                pad_m,
                pad_n,
                pad_k,
                a_preshuffle_quant,
                b_preshuffle_quant,
            ) = k["trait_combo"]
            flat.update(
                {
                    "pipeline": pipeline,
                    "epilogue": epilogue,
                    "scheduler": scheduler,
                    "pad_m": pad_m,
                    "pad_n": pad_n,
                    "pad_k": pad_k,
                    "a_preshuffle_quant": a_preshuffle_quant,
                    "b_preshuffle_quant": b_preshuffle_quant,
                    "group_size_n": k.get("group_size_n", 1),
                }
            )
            flat_items.append(flat)

        selected, method, selected_indices = sample_feasible_set(
            flat_items,
            self.max_instances,
            effective_seed,
            GEMM_ABQUANT_AXES,
        )

        kernel_list = [kernel_list[i] for i in selected_indices]

        if self.manifest_path:
            from sampling.manifest import write_manifest

            write_manifest(
                selected,
                self.manifest_path,
                self.kernel_name_prefix,
                self.datatype,
                self.layout,
                self.gpu_target,
                effective_seed,
                self.tier or "daily",
                method,
            )

        print(
            f"Sampled {len(kernel_list)} from feasible set "
            f"(budget={self.max_instances}, seed={effective_seed}, method={method})"
        )
        return kernel_list

    def _get_group_size_n_values(self):
        """Return the list of group_size_n values to generate kernels for."""
        return self.group_size_n_values

    def _generate_trait_combinations(self):
        """Generate all combinations of traits for ABQuant.

        ABQuant uses 8-tuples:
        (pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, a_preshuffle_quant, b_preshuffle_quant)
        """
        trait_config = self.config["trait_config"]

        pipelines = trait_config.get("pipeline").get("values")
        epilogues = trait_config.get("epilogue").get("values")
        schedulers = trait_config.get("scheduler").get("values")
        pad_m_values = trait_config.get("pad_m").get("values")
        pad_n_values = trait_config.get("pad_n").get("values")
        pad_k_values = trait_config.get("pad_k").get("values")
        a_preshuffle_values = trait_config.get("a_preshuffle_quant").get("values")
        b_preshuffle_values = trait_config.get("b_preshuffle_quant").get("values")

        all_combinations = list(
            itertools.product(
                pipelines,
                epilogues,
                schedulers,
                pad_m_values,
                pad_n_values,
                pad_k_values,
                a_preshuffle_values,
                b_preshuffle_values,
            )
        )

        # Filter out unsupported trait combinations
        combinations = []
        for combo in all_combinations:
            pipeline, epilogue, scheduler = combo[:3]
            a_preshuffle = combo[6]
            b_preshuffle = combo[7]
            if is_trait_combination_valid(
                pipeline,
                epilogue,
                scheduler,
                b_preshuffle,
                self.kernel_name_prefix,
                self.layout,
            ):
                combinations.append(combo)
            else:
                logging.debug(
                    f"Skipping unsupported ABQuant trait combination: "
                    f"{pipeline}-{epilogue}-{scheduler}-a_pre={a_preshuffle}-b_pre={b_preshuffle}"
                )
        return combinations

    def _list_kernels(self):
        """Write kernel list to file for CMake to read."""
        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()
        group_size_n_values = self._get_group_size_n_values()

        kernel_list = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                for group_size_n in group_size_n_values:
                    (
                        pipeline,
                        epilogue,
                        scheduler,
                        pad_m,
                        pad_n,
                        pad_k,
                        a_preshuffle,
                        b_preshuffle,
                    ) = trait_combo

                    # Create kernel name with proper boolean capitalization
                    kernel_name = (
                        f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}_"
                        f"{pipeline}_{epilogue}_{scheduler}_"
                        f"{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_"
                        f"{str(a_preshuffle).capitalize()}_{str(b_preshuffle).capitalize()}_"
                        f"gsn{group_size_n}"
                    )

                    tile_str = self._tile_config_to_str(tile_config)
                    kernel_name += f"_{tile_str}"

                    kernel_list.append(
                        {
                            "name": kernel_name,
                            "tile_config": tile_config,
                            "trait_combo": trait_combo,
                            "group_size_n": group_size_n,
                        }
                    )

        # Apply RFC-compliant sampling (Sobol + LHS + maximin)
        kernel_list = self._apply_sampling(kernel_list)

        # Write kernel count
        with open(
            self.working_path / f"{self.kernel_name_prefix}_kernel_count.txt", "w"
        ) as f:
            f.write(str(len(kernel_list)))

        # Write kernel list
        with open(
            self.working_path / f"{self.kernel_name_prefix}_kernel_list.txt", "w"
        ) as f:
            for kernel in kernel_list:
                tile_config = kernel["tile_config"]
                trait_combo = kernel["trait_combo"]
                group_size_n = kernel["group_size_n"]

                tile_str = self._tile_config_to_str(tile_config)

                trait_str = (
                    f"{trait_combo[0]}_{trait_combo[1]}_{trait_combo[2]}_"
                    + "_".join(str(x) for x in trait_combo[3:])
                    + f"_gsn{group_size_n}"
                )

                f.write(f"{kernel['name']}|{tile_str}|{trait_str}\n")

        print(f"Listed {len(kernel_list)} kernel configurations")

    def _generate_kernel_instance(self, tile_config, trait_combo, group_size_n=None):
        """Generate a single kernel instance for ABQuant.

        trait_combo is an 8-tuple for ABQuant.
        group_size_n is the BQ group size along N dimension.
        """
        if group_size_n is None:
            group_size_n = self.group_size_n_values[0]

        k_block_per_cu = self.config.get("k_block_per_cu", 1)

        (
            pipeline,
            epilogue,
            scheduler,
            pad_m,
            pad_n,
            pad_k,
            a_preshuffle,
            b_preshuffle,
        ) = trait_combo

        # Create kernel name
        kernel_name = (
            f"{self.kernel_name_prefix}_{self.datatype}_{self.layout}_"
            f"{pipeline}_{epilogue}_{scheduler}_"
            f"{str(pad_m).capitalize()}_{str(pad_n).capitalize()}_{str(pad_k).capitalize()}_"
            f"{str(a_preshuffle).capitalize()}_{str(b_preshuffle).capitalize()}_"
            f"gsn{group_size_n}"
        )

        tile_str = self._tile_config_to_str(tile_config)
        kernel_name += f"_{tile_str}"

        # Pipeline maps
        pipeline_impl_map = {
            "compv3": "ck_tile::ABQuantGemmPipelineAgBgCrCompV3",
        }
        base_pipeline_map = {
            "compv3": "ck_tile::BaseGemmPipelineAgBgCrCompV3",
        }
        scheduler_type_map = {
            "intrawave": "ck_tile::GemmPipelineScheduler::Intrawave",
            "interwave": "ck_tile::GemmPipelineScheduler::Interwave",
            "default": "ck_tile::GemmPipelineScheduler::Default",
        }

        instance_code = self.populate_kernel_header(kernel_name)
        instance_code += self.populate_kernel_dtype_layout()
        instance_code += self.populate_strut_begin(kernel_name)
        instance_code += self.populate_tile_config(tile_config)
        instance_code += self._populate_abquant_trait_config(trait_combo, group_size_n)
        instance_code += self._populate_abquant_initialization(
            base_pipeline_map, pipeline
        )
        instance_code += self._populate_launch_abquant(
            scheduler_type_map,
            scheduler,
            pipeline_impl_map,
            pipeline,
            epilogue,
            k_block_per_cu,
        )

        # Write into a file
        simplified_name = kernel_name
        if simplified_name.startswith(f"{self.kernel_name_prefix}_"):
            simplified_name = simplified_name[len(self.kernel_name_prefix) + 1 :]

        header_file = (
            self.working_path
            / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)

        print(f"Generated {header_file}")

        return kernel_name, instance_code

    def populate_kernel_header(self, kernel_name):
        instance_code = f"""// Generated kernel instance for {kernel_name}
#pragma once

#include <cstdint>
#include <utility>
#include <tuple>
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_kernel.hpp"
#include "ck_tile/ops/common/tensor_layout.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/gemm_quant.hpp"
#include "ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp"
"""
        return instance_code

    def populate_kernel_dtype_layout(self):
        a_layout, b_layout, c_layout = _validation_utils.get_abc_layouts(self.layout)
        dtype_str = _validation_utils.get_dtype_string(self.datatype)
        c_type_str = _validation_utils.get_dtype_string(
            "fp16" if self.datatype in ["fp8", "bf8"] else self.datatype
        )

        instance_code = f"""
using ADataType = {dtype_str};
using BDataType = {dtype_str};
using AccDataType = float;
using CDataType = {c_type_str};
using AQDataType = float;
using BQDataType = float;

using ALayout = {a_layout};
using BLayout = {b_layout};
using CLayout = {c_layout};
using AQLayout = ck_tile::tensor_layout::gemm::RowMajor;
using BQLayout = ck_tile::tensor_layout::gemm::ColumnMajor;
"""
        return instance_code

    def _populate_abquant_trait_config(self, trait_combo, group_size_n):
        (
            pipeline,
            epilogue,
            scheduler,
            pad_m,
            pad_n,
            pad_k,
            a_preshuffle,
            b_preshuffle,
        ) = trait_combo

        instance_code = f"""

    // Traits configurations
    static constexpr bool kPadM = {"true" if pad_m in [True, "true"] else "false"};
    static constexpr bool kPadN = {"true" if pad_n in [True, "true"] else "false"};
    static constexpr bool kPadK = {"true" if pad_k in [True, "true"] else "false"};
    static constexpr bool TransposeC = false;
    static constexpr bool DoubleSmemBuffer = false;
    static constexpr bool APreshuffleQuant = {"true" if a_preshuffle in [True, "true"] else "false"};
    static constexpr bool BPreshuffleQuant = {"true" if b_preshuffle in [True, "true"] else "false"};
    static constexpr bool PreshuffleB = false;
    static constexpr ck_tile::index_t GroupSizeK = {self.group_size_k};
    static constexpr ck_tile::index_t GroupSizeN = {group_size_n};"""

        return instance_code

    def _populate_abquant_initialization(self, base_pipeline_map, pipeline):
        instance_code = """

    // Tile shape
    using TileShape = ck_tile::TileGemmShape<
        ck_tile::sequence<TileM, TileN, TileK>,
        ck_tile::sequence<WarpPerBlock_M, WarpPerBlock_N, WarpPerBlock_K>,
        ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;

    // Tile partitioner
    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<TileShape, 8, 4>;

    // Quantization group sizes
    using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, GroupSizeK>>;
    using BQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, GroupSizeN, GroupSizeK>>;

    // Traits
    using Traits = ck_tile::TileGemmQuantTraits<
        kPadM, kPadN, kPadK,
        APreshuffleQuant,
        BPreshuffleQuant,
        PreshuffleB,
        ALayout, BLayout, CLayout,
        ck_tile::QuantType::ABQuantGrouped,
        AQLayout, BQLayout>;

    // Pipeline problem (base, for hot loop detection)
    using GemmPipelineProblem = ck_tile::GemmPipelineProblemBase<
        ADataType,
        BDataType,
        AccDataType,
        TileShape,
        Traits,
        BDataType>;"""

        instance_code += f"""

    // Base pipeline for hot loop detection
    using BaseGemmPipeline = {base_pipeline_map.get(pipeline)}<GemmPipelineProblem>;"""

        return instance_code

    def _populate_launch_abquant(
        self,
        scheduler_type_map,
        scheduler,
        pipeline_impl_map,
        pipeline,
        epilogue,
        k_block_per_cu,
    ):
        """Generate the complete launch function for ABQuant kernels."""
        instance_code = """

    // Launch function
    static float launch(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& stream) {

        // Hot loop detection
        const ck_tile::index_t K_split = ck_tile::integer_least_multiple(args.K, TileShape::kK);
        const ck_tile::index_t num_loop = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);"""

        instance_code += f"""

        const auto Run = [&](const auto has_hot_loop_, const auto tail_number_) {{
            constexpr bool has_hot_loop_v = has_hot_loop_.value;
            constexpr auto tail_number_v = tail_number_.value;

            // Full pipeline problem with hot loop and tail number
            using PipelineProblem = ck_tile::GemmABQuantPipelineProblem<
                ADataType,
                AQDataType,
                BDataType,
                BQDataType,
                AccDataType,
                TileShape,
                Traits,
                AQuantGroupSize,
                BQuantGroupSize,
                false,
                ADataType,
                {scheduler_type_map.get(scheduler)},
                has_hot_loop_v,
                tail_number_v>;

            using GemmPipeline = {pipeline_impl_map.get(pipeline)}<PipelineProblem>;"""

        instance_code += """

            // Epilogue"""
        if epilogue == "cshuffle":
            instance_code += """
            using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                TileM,  // kM_
                TileN,  // kN_
                WarpPerBlock_M,  // MWave_
                WarpPerBlock_N,  // NWave_
                WarpTileM,  // kMPerXdl_
                WarpTileN,  // kNPerXdl_
                WarpTileK,  // kKPerXdl_
                TransposeC>;  // isCTransposed_

            using GemmEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;"""
        else:
            instance_code += """
            using EpilogueProblem = ck_tile::DefaultGemm2DEpilogueProblem<
                ADataType,
                BDataType,
                ck_tile::tuple<>,  // DsDataType
                AccDataType,
                CDataType,
                ck_tile::tuple<>,  // DsLayout
                CLayout,
                ck_tile::element_wise::PassThrough,
                TileM,  // kM_
                TileN,  // kN_
                kPadM,
                kPadN,
                WarpTileM,  // kMPerXdl_
                WarpTileN,  // kNPerXdl_
                WarpTileK,  // kKPerXdl_
                TransposeC>;  // isCTransposed_

            using GemmEpilogue = ck_tile::DefaultGemm2DEpilogue<EpilogueProblem>;"""

        instance_code += f"""

            // Kernel type
            using Kernel = ck_tile::QuantGemmKernel<
                TilePartitioner, GemmPipeline, GemmEpilogue,
                ck_tile::QuantType::ABQuantGrouped>;

            // Kernel arguments
            auto kargs = Kernel::MakeKernelArgs(args);

            if (!Kernel::IsSupportedArgument(kargs)) {{
                throw std::runtime_error("Unsupported kernel arguments; skipping GEMM launch.");
            }}

            // Get grid and block sizes
            const dim3 grids = Kernel::GridSize(args.M, args.N, args.k_batch);
            const dim3 blocks = Kernel::BlockSize();

            if(stream.log_level_ > 0) {{
                std::cout << "Launching kernel: " << Kernel::GetName() << '\\n'
                          << "grid: {{" << grids.x << ", " << grids.y << ", " << grids.z << "}}"
                          << ", blocks: {{" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}}"
                          << std::endl;
            }}

            // Launch kernel
            constexpr int kBlockPerCu = {k_block_per_cu};
            float ave_time = ck_tile::launch_kernel(
                stream,
                ck_tile::make_kernel<kBlockPerCu>(Kernel{{}}, grids, blocks, 0, kargs));

            return ave_time;
        }};
        return BaseGemmPipeline::TailHandler(Run, has_hot_loop, tail_num);
    }}
}};
"""
        return instance_code

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation with parallel processing"""
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()
        group_size_n_values = self._get_group_size_n_values()

        work_items = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                for group_size_n in group_size_n_values:
                    work_items.append(
                        (
                            tile_config,
                            trait_combo,
                            group_size_n,
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
                {
                    "tile_config": item[0],
                    "trait_combo": item[1],
                    "group_size_n": item[2],
                    "_work_item": item,
                }
                for item in work_items
            ]
            sampled = self._apply_sampling(kernel_dicts)
            work_items = [k["_work_item"] for k in sampled]

        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )
        print(f"  Tile configs: {len(tile_configs)}")
        print(f"  Trait combinations: {len(trait_combos)}")
        print(f"  Group size N values: {group_size_n_values}")
        print(f"  Total kernels: {len(work_items)}")

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
    """Worker function to generate a single individual kernel file"""
    (
        tile_config,
        trait_combo,
        group_size_n,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json,
    ) = work_item

    builder = GemmABQuantKernelBuilder(
        kernel_name_prefix, working_path, gpu_target, datatype, layout, config_json
    )

    try:
        kernel_name, _ = builder._generate_kernel_instance(
            tile_config, trait_combo, group_size_n
        )
        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="GEMM ABQuant kernel instance builder")
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

    assert args.datatype in ["fp16", "bf16", "fp8", "bf8"], (
        f"Invalid datatype string: {args.datatype} (supported datatypes are [fp16, bf16, fp8, and bf8])"
    )

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

    kernel_name_prefix = "gemm_abquant"
    builder = GemmABQuantKernelBuilder(
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

        # Parse trait combo for ABQuant:
        # pipeline_epilogue_scheduler_padM_padN_padK_aPreshuffle_bPreshuffle_gsnN
        trait_parts = args.trait_combo.split("_")
        # Find the gsn part
        group_size_n = 1
        gsn_idx = None
        for i, part in enumerate(trait_parts):
            if part.startswith("gsn"):
                group_size_n = int(part[3:])
                gsn_idx = i
                break

        if gsn_idx is not None:
            # Remove gsn from trait_parts for parsing
            trait_parts_no_gsn = trait_parts[:gsn_idx] + trait_parts[gsn_idx + 1 :]
        else:
            trait_parts_no_gsn = trait_parts

        if len(trait_parts_no_gsn) < 8:
            parser.error(
                f"--trait_combo must have 8 underscore-separated fields + gsn suffix "
                f"(e.g. 'compv3_default_intrawave_False_False_False_False_False_gsn1'), "
                f"got {len(trait_parts_no_gsn)} field(s): '{args.trait_combo}'"
            )
        trait_combo = (
            trait_parts_no_gsn[0],  # pipeline
            trait_parts_no_gsn[1],  # epilogue
            trait_parts_no_gsn[2],  # scheduler
            trait_parts_no_gsn[3] == "True",  # pad_m
            trait_parts_no_gsn[4] == "True",  # pad_n
            trait_parts_no_gsn[5] == "True",  # pad_k
            trait_parts_no_gsn[6] == "True",  # a_preshuffle_quant
            trait_parts_no_gsn[7] == "True",  # b_preshuffle_quant
        )

        builder._generate_kernel_instance(tile_config, trait_combo, group_size_n)
    elif args.gen_all_individual:
        builder._generate_all_individual(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_all_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
