# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import shutil
import sys
from pathlib import Path

# Add dispatcher/codegen/grouped_conv to path for shared validation rules
_THIS_DIR = Path(__file__).resolve().parent
_DISPATCHER_CODEGEN = _THIS_DIR.parents[1] / "dispatcher" / "codegen" / "grouped_conv"
if str(_DISPATCHER_CODEGEN) not in sys.path:
    sys.path.insert(0, str(_DISPATCHER_CODEGEN))

from grouped_config_rules_default import (  # noqa E402
    check_vectors as _shared_check_vectors,
    check_warp_coverage,
    check_bwd_data_vec_coverage,
    check_wmma_instance,
    check_wmma_native_warp_tile,
    get_warp_size,
)


class ConvInstanceTemplateParams:
    def __init__(
        self,
        specialization,
        tile_size,
        warps,
        warp_tile,
        double_smem_buffer,
        num_wave_groups,
        is_two_stage_instance,
        pipeline_version,
        scheduler,
        scalar_per_vector,
        num_groups_to_merge,
        split_image,
        explicit_gemm,
        id,
        streamk_enabled=False,
        streamk_reduction_strategy=None,
        streamk_persistent=False,
    ):
        self.specialization = specialization
        self.tile_size = tile_size
        self.warps = warps
        self.warp_tile = warp_tile
        self.double_smem_buffer = double_smem_buffer
        self.num_wave_groups = num_wave_groups
        self.is_two_stage_instance = is_two_stage_instance
        self.pipeline_version = pipeline_version
        self.scheduler = scheduler
        self.scalar_per_vector = scalar_per_vector
        self.num_groups_to_merge = num_groups_to_merge
        self.split_image = split_image
        self.explicit_gemm = explicit_gemm
        self.id = id
        self.streamk_enabled = streamk_enabled
        self.streamk_reduction_strategy = streamk_reduction_strategy
        self.streamk_persistent = streamk_persistent

    def get_optimizations(self):
        explicit_gemm = "true" if self.explicit_gemm else "false"
        split_image = "true" if self.split_image else "false"
        num_groups_to_merge = str(self.num_groups_to_merge)
        two_stage_instance = "true" if self.is_two_stage_instance else "false"
        if self.streamk_enabled:
            streamk_str = (
                f"{{true, ckb::StreamKReductionStrategy::{self.streamk_reduction_strategy}, "
                f"{'true' if self.streamk_persistent else 'false'}}}"
            )
        else:
            streamk_str = "ckb::StreamKConfig::disabled()"
        return (
            f"ckt::TileOptimizations{{.num_groups_to_merge = {num_groups_to_merge}, "
            f".split_image = {split_image}, .explicit_gemm = {explicit_gemm}, "
            f".two_stage = {two_stage_instance}, .streamk = {streamk_str}}}"
        )

    def get_specialization(self):
        namespace = "ckb::TileConvSpecialization::"
        if self.specialization == "Default" or self.specialization == "OddC":
            return namespace + "DEFAULT"
        if self.specialization == "Filter1x1Pad0":
            return namespace + "FILTER_1X1_PAD0"
        if self.specialization == "Filter1x1Stride1Pad0":
            return namespace + "FILTER_1X1_STRIDE1_PAD0"
        if self.specialization == "Filter3x3":
            return namespace + "FILTER_3x3"
        else:
            raise RuntimeError("not supported specialization")

    def get_thread_block(self):
        return f"ckt::TileThreadBlock{{.tile_size = {{.m = {self.tile_size[0]}, .n = {self.tile_size[1]}, .k = {self.tile_size[2]}}}}}"

    def get_block_gemm_desc(self):
        double_smem_buffer = "true" if self.double_smem_buffer else "false"
        scheduler = (
            "INTRAWAVE" if self.scheduler.find("Intrawave") != -1 else "INTERWAVE"
        )
        return f"""ckt::TileBlockGemm{{
                    .warps              = {{.m = {self.warps[0]}, .n = {self.warps[1]}, .k = {self.warps[2]}}},
                    .warp_tile          = {{.m = {self.warp_tile[0]}, .n = {self.warp_tile[1]}, .k = {self.warp_tile[2]}}},
                    .double_smem_buffer = {double_smem_buffer},
                    .num_wave_groups    = {self.num_wave_groups},
                    .pipeline_version   = ckb::PipelineVersion::{self.pipeline_version},
                    .scheduler          = ckb::PipelineScheduler::{scheduler}}}"""

    def get_block_transfer(self):
        return f"""ckt::TileTransfer{{.a_scalar_per_vector = {self.scalar_per_vector[0]}, 
                .b_scalar_per_vector = {self.scalar_per_vector[1]}, .c_scalar_per_vector = {self.scalar_per_vector[2]}}}"""


def get_dtype(problem_name):
    if problem_name.find("fp32") != -1:
        return "float"
    if problem_name.find("fp16") != -1:
        return "ck_tile::half_t"
    if problem_name.find("bf16") != -1:
        return "ck_tile::bf16_t"
    else:
        raise RuntimeError("Cannot parse data type from problem name: " + problem_name)


def get_k_mfma(dtype, m_per_xdl, n_per_xdl):
    if m_per_xdl != n_per_xdl:
        raise RuntimeError("Not supported")
    if dtype == "float":
        if m_per_xdl == 32:
            return 2
        else:
            return 4
    else:
        if m_per_xdl == 32:
            return 16
        else:
            return 32


def check_vectors(a_scalar_per_vector, b_scalar_per_vector, c_scalar_per_vector):
    """Reject odd vector sizes (except 1).

    Delegates to the shared rule in grouped_config_rules_default.py.
    """
    return _shared_check_vectors(
        a_scalar_per_vector, b_scalar_per_vector, c_scalar_per_vector
    )


def parse_instance_string(instance_string):
    """Parse instance string, treating Seq(...) as a single parameter."""
    params = []
    current_param = ""
    paren_depth = 0

    for char in instance_string:
        if char == "(":
            paren_depth += 1
            current_param += char
        elif char == ")":
            paren_depth -= 1
            current_param += char
        elif char == "," and paren_depth == 0:
            # Only split on comma if we're not inside parentheses
            params.append(current_param.strip())
            current_param = ""
        else:
            current_param += char

    # Add the last parameter
    if current_param.strip():
        params.append(current_param.strip())

    return params


def copy_includes(instances_path):
    inc_dir = Path(__file__).resolve().parent
    output_dir = Path(instances_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(f"{inc_dir}/include/instance_includes.inc", instances_path)
    shutil.copy(f"{inc_dir}/include/instance_run.inc", instances_path)
    shutil.copy(f"{inc_dir}/include/signatures.hpp", instances_path)


def generate_calls_inc(instances, problem_name, direction, filter_pattern):
    generate_dir = Path(__file__).resolve().parent
    output_dir = Path(f"{generate_dir}/instances/{direction}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        f"{generate_dir}/instances/{direction}/{problem_name}_calls.inc", "w"
    ) as f:
        if problem_name.find(filter_pattern) == -1:
            return
        for instance in instances:
            instance_name = problem_name + "_" + str(instance.id)
            f.write(f"run_alg(run_{instance_name});\n")


def generate_defs_inc(instances, problem_name, signature, direction, filter_pattern):
    generate_dir = Path(__file__).resolve().parent
    with open(f"{generate_dir}/instances/{direction}/{problem_name}.inc", "w") as f:
        if problem_name.find(filter_pattern) == -1:
            return
        for instance in instances:
            instance_name = problem_name + "_" + str(instance.id)
            f.write(
                f"std::tuple<bool, float, std::string> run_{instance_name}(\n"
                f"   const ckt::Args<{signature}>& args,\n"
                f"   const ckt::Inputs<{signature}>& inputs,\n"
                f"   const ckt::Outputs<{signature}>& outputs,\n"
                f"   const ck_tile::stream_config& s_conf);\n"
            )


def generate_conv_cpp(
    instances,
    problem_name,
    config,
    direction,
    signature_name,
    filter_pattern,
    instances_path,
):
    for instance in instances:
        if problem_name.find(filter_pattern) == -1:
            break
        instance_name = problem_name + "_" + str(instance.id)
        directory_path = Path(f"{instances_path}/{direction}/{config}")
        directory_path.mkdir(parents=True, exist_ok=True)
        parent_dir = Path(__file__).resolve().parent
        template_file = "include/grouped_convolution_tile.cpp.in"

        with open(
            f"{parent_dir}/{template_file}",
            "r",
        ) as f:
            content = f.read()

            content = content.replace("gen_signature", signature_name)
            content = content.replace("gen_instance_name", instance_name)
            content = content.replace(
                "gen_specialization", instance.get_specialization()
            )
            content = content.replace("gen_thread_block", instance.get_thread_block())
            content = content.replace(
                "gen_block_gemm_desc", instance.get_block_gemm_desc()
            )
            content = content.replace(
                "gen_block_transfer", instance.get_block_transfer()
            )
            content = content.replace("gen_optimizations", instance.get_optimizations())

        with open(
            f"{instances_path}/{direction}/{config}/{instance_name}.cpp",
            "w",
        ) as f:
            f.write(content)


# Maps ck_tile pipeline names (from GetPipelineName()) to builder PipelineVersion enum names.
PIPELINE_NAME_TO_VERSION = {
    "BASIC_V1": "V1",
    "MEMORY": "V2",
    "COMPUTE_V3": "V3",
    "COMPUTE_V4": "V4",
    "COMPUTE_V5": "V5",
    "COMPUTE_V6": "V6",
    "BASIC_ASYNC_V1": "ASYNC_V1",
    "COMPUTE_ASYNC": "ASYNC_V4",
    "WAVELET": "WAVELET",
}

# Maps ck_tile StreamKReductionStrategy int values (from static_cast<int> in instance string)
# to builder enum names. ck_tile enum: Atomic=0, Linear=1, Tree=2.
# Atomic=0 is omitted: it is not expected in generated instances. If encountered, .get()
# falls back to str(reduction_int) ("0"), which will cause a downstream build error.
STREAMK_REDUCTION_STRATEGY = {
    1: "LINEAR",
    2: "TREE",
}


def parse_native_instance(args, instance_id, problem_name, has_streamk, has_two_stage):
    """Parse a native CK Tile grouped-conv instance string for any direction
    (GroupedConvolution{Forward,BackwardData,BackwardWeight}Kernel<...>).

    Fields (0-indexed after splitting on commas inside <>), shared by all directions:
    0: NDimSpatial, 1: ConvSpec, 2: InLayout, 3: WeiLayout, 4: DsLayout, 5: OutLayout,
    6: VecA, 7: VecB, 8: VecC, 9: NumGroupsToMerge, 10: SplitImage, 11: ExplicitGemm,
    12: MPerBlock, 13: NPerBlock, 14: KPerBlock, 15: MWarp, 16: NWarp, 17: KWarp,
    18: MWarpTile, 19: NWarpTile, 20: KWarpTile, 21: ADataType, 22: BDataType,
    23: PipelineName, 24: Scheduler, 25: DoubleSmemBuffer, 26: NumWaveGroups,
    27: AccDataType, 28: EDataType, 29: DsDataType, 30: CDEElementwiseOp,
    [31: IsStreamK, 32: ReductionStrategy, 33: PersistentDP]  (backward_weight only)

    has_streamk: direction carries the trailing StreamK fields (backward_weight only).
    has_two_stage: direction has a two-stage path (backward_weight only); else False.
    """
    spec = args[1]
    tile_size = [int(args[12]), int(args[13]), int(args[14])]
    warps = [int(args[15]), int(args[16]), int(args[17])]
    warp_tile = [int(args[18]), int(args[19]), int(args[20])]

    pipeline_name = args[23]
    if pipeline_name not in PIPELINE_NAME_TO_VERSION:
        raise RuntimeError(
            f"Unknown pipeline name '{pipeline_name}' in native instance {instance_id}"
        )
    pipeline_version = PIPELINE_NAME_TO_VERSION[pipeline_name]

    scheduler = args[24]
    double_smem_buffer = int(args[25]) != 0
    num_wave_groups = int(args[26])

    scalar_per_vector = [int(args[6]), int(args[7]), int(args[8])]
    num_groups_to_merge = int(args[9])
    split_image = int(args[10]) != 0
    explicit_gemm = int(args[11]) != 0

    is_two_stage = (
        has_two_stage
        and get_dtype(problem_name) != "float"
        and scalar_per_vector[2] == 1
    )
    is_streamk = has_streamk and int(args[31]) != 0
    streamk_reduction_strategy = None
    streamk_persistent = False
    if is_streamk:
        is_two_stage = False
        reduction_int = int(args[32])
        streamk_reduction_strategy = STREAMK_REDUCTION_STRATEGY.get(
            reduction_int, str(reduction_int)
        )
        streamk_persistent = int(args[33]) != 0

    return ConvInstanceTemplateParams(
        spec,
        tile_size,
        warps,
        warp_tile,
        double_smem_buffer,
        num_wave_groups,
        is_two_stage,
        pipeline_version,
        scheduler,
        scalar_per_vector,
        num_groups_to_merge,
        split_image,
        explicit_gemm,
        instance_id,
        streamk_enabled=is_streamk,
        streamk_reduction_strategy=streamk_reduction_strategy,
        streamk_persistent=streamk_persistent,
    )


def parse_native_bwd_weight_instance(args, instance_id, problem_name):
    return parse_native_instance(
        args, instance_id, problem_name, has_streamk=True, has_two_stage=True
    )


def parse_native_fwd_instance(args, instance_id, problem_name):
    return parse_native_instance(
        args, instance_id, problem_name, has_streamk=False, has_two_stage=False
    )


def parse_native_bwd_data_instance(args, instance_id, problem_name):
    return parse_native_instance(
        args, instance_id, problem_name, has_streamk=False, has_two_stage=False
    )


# Maps kernel type prefix to native parser function.
NATIVE_PARSERS = {
    "GroupedConvolutionBackwardWeightKernel": parse_native_bwd_weight_instance,
    "GroupedConvolutionForwardKernel": parse_native_fwd_instance,
    "GroupedConvolutionBackwardDataKernel": parse_native_bwd_data_instance,
}


def try_parse_native_instance(instance, instance_id, problem_name):
    """Try to parse an instance line as a native CK Tile instance string.

    Returns a ConvInstanceTemplateParams if the line matches a native format,
    or None if it doesn't match (so the caller can fall through to old CK parsing).
    """
    stripped = instance.strip()
    for prefix, parser in NATIVE_PARSERS.items():
        if stripped.startswith(prefix + "<"):
            start = stripped.index("<") + 1
            end = stripped.rindex(">")
            params_str = stripped[start:end]
            args = parse_instance_string(params_str)
            return parser(args, instance_id, problem_name)
    return None



def parse_fwd_instances(instances, problem_name, warp_size=32, verbose=True):
    convs = []
    for instance_id, instance in enumerate(instances):
        if instance.find("#") != -1 or instance.find(";") != -1:
            continue
        native = try_parse_native_instance(instance, instance_id, problem_name)
        if native is not None:
            convs.append(native)
            continue
        start = instance.index("<") + 1
        end = instance.rindex(">")
        params_str = instance[start:end]
        args = parse_instance_string(params_str)

        is_v3_instance = instance.find("Xdl_CShuffle_V3") != -1
        split_image = instance.find("Large_Tensor") != -1

        if is_v3_instance:
            spec = args[14]
            block_size = int(args[16])
            m_per_block = int(args[17])
            n_per_block = int(args[18])
            k_per_block = int(args[19])
            k1 = int(args[20])
            m_per_xdl = int(args[22])
            n_per_xdl = int(args[23])
            m_xdl_per_wave = int(args[24])
            n_xdl_per_wave = int(args[25])
            a_scalar_per_vector = int(args[30])
            b_scalar_per_vector = int(args[37])
            c_scalar_per_vector = int(args[43])
            scheduler = args[44]
            pipeline_version = args[45]
            direct_load = args[48] == "true"
            num_groups_to_merge = int(args[49])
        else:
            spec = args[14]
            block_size = int(args[17])
            m_per_block = int(args[18])
            n_per_block = int(args[19])
            k_per_block = int(args[20])
            k1 = int(args[21])
            m_per_xdl = int(args[23])
            n_per_xdl = int(args[24])
            m_xdl_per_wave = int(args[25])
            n_xdl_per_wave = int(args[26])
            a_scalar_per_vector = int(args[31])
            b_scalar_per_vector = int(args[38])
            c_scalar_per_vector = int(args[44])
            scheduler = "Intrawave"
            pipeline_version = "v1"
            direct_load = 0
            num_groups_to_merge = 1 if split_image else int(args[48])

        double_smem_buffer = pipeline_version == "v4"
        num_wave_groups = 1
        # Replace pipeline if Direct Load
        if direct_load:
            if pipeline_version == "v1":
                pipeline_version = "ASYNC_V1"
            elif pipeline_version == "v4":
                pipeline_version = "ASYNC_V4"
            else:
                raise RuntimeError(
                    f"{pipeline_version} not supported pipeline for direct load"
                )
        else:
            pipeline_version = pipeline_version.upper()

        # Old CK pipeline version V5 maps to V6 for CK Tile
        if pipeline_version == "V5":
            pipeline_version = "V6"

        # WMMA
        dtype = get_dtype(problem_name)

        m_warp = int(m_per_block / (m_per_xdl * m_xdl_per_wave))
        n_warp = int(n_per_block / (n_per_xdl * n_xdl_per_wave))
        k_warp = int(block_size / (warp_size * m_warp * n_warp))
        k_per_xdl = min(max(k1, get_k_mfma(dtype, m_per_xdl, n_per_xdl)), k_per_block)

        is_two_stage = False
        if not check_wmma_instance(warp_size, k_per_block, k_warp, k_per_xdl, m_per_xdl, dtype):
            continue
        conv = ConvInstanceTemplateParams(
            spec,
            [m_per_block, n_per_block, k_per_block],
            [m_warp, n_warp, k_warp],
            [m_per_xdl, n_per_xdl, k_per_xdl],
            double_smem_buffer,
            num_wave_groups,
            is_two_stage,
            pipeline_version,
            scheduler,
            [a_scalar_per_vector, b_scalar_per_vector, c_scalar_per_vector],
            num_groups_to_merge,
            split_image,
            False,
            instance_id,
        )
        convs.append(conv)
    return convs


def parse_bwd_weight_instances(instances, problem_name, warp_size=32, verbose=True):
    convs = []

    for instance_id, instance in enumerate(instances):
        if instance.find("#") != -1 or instance.find(";") != -1:
            continue
        native = try_parse_native_instance(instance, instance_id, problem_name)
        if native is not None:
            if (
                native.streamk_enabled
                and get_dtype(problem_name) == "float"
                and native.pipeline_version.find("ASYNC") != -1
            ):
                if verbose:
                    print(
                        f"Skipping instance {instance_id} with streamk, async, float since it's not supported yet."
                    )
                continue
            if not check_wmma_native_warp_tile(warp_size, native.streamk_enabled):
                continue
            if not check_wmma_instance(warp_size, native.tile_size[2], native.warps[2], native.warp_tile[2], native.warp_tile[0], get_dtype(problem_name)):
                continue
            convs.append(native)
            continue

        device_op_name = instance.split("<")[0]
        start = instance.index("<") + 1
        end = instance.rindex(">")
        params_str = instance[start:end]
        args = parse_instance_string(params_str)

        direct_load = False

        is_v3_instance = instance.find("Xdl_CShuffleV3") != -1
        is_two_stage_instance = instance.find("TwoStage") != -1
        is_explicit_gemm = device_op_name.find("Explicit") != -1

        if is_explicit_gemm:
            gemm_params = device_op_name = (
                instance.split("<")[2].split(">")[1].split(",")
            )
            args = [param.split(":")[1].strip() for param in gemm_params]

            spec = "Filter1x1Stride1Pad0"
            block_size = int(args[0])

            mnk_per_block = args[1].split("x")
            m_per_block = int(mnk_per_block[0])
            n_per_block = int(mnk_per_block[1])
            k_per_block = int(mnk_per_block[2])

            wave_tile = args[2].split("x")
            m_per_xdl = int(wave_tile[0])
            n_per_xdl = int(wave_tile[1])

            k1_values = args[3].split("x")
            ak1 = int(k1_values[0])
            bk1 = int(k1_values[1])
            k1 = min(ak1, bk1)

            wave_map = args[4].split("x")
            m_xdl_per_wave = int(wave_map[0])
            n_xdl_per_wave = int(wave_map[1])

            vector_read = args[5].split("x")
            a_scalar_per_vector = int(vector_read[0])
            b_scalar_per_vector = int(vector_read[1])
            c_scalar_per_vector_seq = [
                int(x)
                for x in vector_read[2].strip("Seq").strip("(").strip(")").split(",")
            ]

            if len(set(c_scalar_per_vector_seq)) != 1:
                raise RuntimeError(
                    f"c_scalar_per_vector must be the same across all waves for instance {instance_id} with device op {device_op_name}. Found values: {c_scalar_per_vector_seq}"
                )

            c_scalar_per_vector = c_scalar_per_vector_seq[0]

            num_groups_to_merge = 1

            # Block GEMM pipeline parameters
            block_gemm_pipeline_scheduler = args[6]
            blk_gemm_pipeline_version = args[7]
        else:
            spec = args[11]
            block_size = int(args[12])
            m_per_block = int(args[13])
            n_per_block = int(args[14])
            k1 = int(args[16])
            m_per_xdl = int(args[17])
            n_per_xdl = int(args[18])
            m_xdl_per_wave = int(args[19])
            n_xdl_per_wave = int(args[20])
            a_scalar_per_vector = int(args[25])
            b_scalar_per_vector = int(args[32])
            c_scalar_per_vector = int(args[38])

            if is_v3_instance or is_two_stage_instance:
                k_per_block = int(args[15])
            else:
                k0_per_block = int(args[15])
                k_per_block = k0_per_block * k1

            if is_v3_instance:
                if len(args) != 45:
                    raise RuntimeError(
                        f"Wrong number of parameters in the V3 XDL CShuffle instance string: {instance}"
                    )

                direct_load = int(args[43]) == 1
                num_groups_to_merge = int(args[44])

                # Block GEMM pipeline parameters
                block_gemm_pipeline_scheduler = args[39]
                blk_gemm_pipeline_version = args[40]
            elif is_two_stage_instance:
                if len(args) != 46:
                    raise RuntimeError(
                        f"Wrong number of parameters in the TwoStage instance string: {instance}\n"
                        + f"Expected 46 parameters for TwoStage instance. Found {len(args)} parameters."
                    )

                num_groups_to_merge = int(args[41])

                # Block GEMM pipeline parameters
                block_gemm_pipeline_scheduler = args[39]
                blk_gemm_pipeline_version = args[40]

            else:
                # Regular V1 XDL CShuffle instance
                if len(args) != 43:
                    raise RuntimeError(
                        f"Wrong number of parameters in the XDL CShuffle instance string: {instance}\n"
                        + f"Expected 43 parameters for V1 instance. Found {len(args)} parameters."
                    )

                num_groups_to_merge = 1

                # Block GEMM pipeline parameters
                block_gemm_pipeline_scheduler = "Intrawave"
                blk_gemm_pipeline_version = "v1"

        # Common part to all solvers.

        # Sanity check for Block GEMM pipeline parameters
        # Scheduler must be either Intrawave or Interwave.
        # Version must be from v1 to v5
        if block_gemm_pipeline_scheduler not in ["Intrawave", "Interwave"]:
            raise RuntimeError(
                f"Invalid Block GEMM pipeline scheduler: {block_gemm_pipeline_scheduler} in instance: {instance}"
            )
        if blk_gemm_pipeline_version not in ["v1", "v2", "v3", "v4", "v5"]:
            raise RuntimeError(
                f"Invalid Block GEMM pipeline version: {blk_gemm_pipeline_version} in instance: {instance}"
            )

        split_image = instance.find("Large") != -1
        double_smem_buffer = blk_gemm_pipeline_version == "v4"
        num_wave_groups = 1
        scheduler = block_gemm_pipeline_scheduler
        pipeline_version = blk_gemm_pipeline_version.upper()

        # Old CK pipeline version V5 maps to V6 for CK Tile
        if pipeline_version == "V5":
            pipeline_version = "V6"

        if direct_load:
            if pipeline_version == "V1":
                pipeline_version = "ASYNC_V1"
            elif pipeline_version == "V4":
                pipeline_version = "ASYNC_V4"
            else:
                raise RuntimeError(
                    f"Not supported pipeline for direct load: pipeline_version={pipeline_version} in instance: {instance}"
                )

        # WMMA
        dtype = get_dtype(problem_name)

        m_warp = int(m_per_block / (m_per_xdl * m_xdl_per_wave))
        n_warp = int(n_per_block / (n_per_xdl * n_xdl_per_wave))
        k_warp = int(block_size / (warp_size * m_warp * n_warp))

        k_per_xdl = min(max(k1, get_k_mfma(dtype, m_per_xdl, n_per_xdl)), k_per_block)

        if not check_wmma_instance(warp_size, k_per_block, k_warp, k_per_xdl, m_per_xdl, dtype):
            continue
        if not check_vectors(
            a_scalar_per_vector, b_scalar_per_vector, c_scalar_per_vector
        ):
            if verbose:
                print(
                    f"Skipping instance {instance_id} with irregular load since it's not supported yet."
                )
            continue
        if not check_warp_coverage(
            m_per_block,
            n_per_block,
            k_per_block,
            a_scalar_per_vector,
            b_scalar_per_vector,
            variant="bwd_weight",
            warp_size=warp_size,
        ):
            if verbose:
                print(
                    f"Skipping instance {instance_id} with multiple warps per continous tile dim since it's not supported yet."
                )
            continue

        if is_explicit_gemm:
            if dtype != "float" and c_scalar_per_vector % 2 != 0:
                is_two_stage_instance = True

        conv = ConvInstanceTemplateParams(
            spec,
            [m_per_block, n_per_block, k_per_block],
            [m_warp, n_warp, k_warp],
            [m_per_xdl, n_per_xdl, k_per_xdl],
            double_smem_buffer,
            num_wave_groups,
            is_two_stage_instance,
            pipeline_version,
            scheduler,
            [a_scalar_per_vector, b_scalar_per_vector, c_scalar_per_vector],
            num_groups_to_merge,
            split_image,
            is_explicit_gemm,
            instance_id,
        )
        convs.append(conv)

    return convs


def parse_bwd_data_instances(instances, problem_name, warp_size=32, verbose=True):
    convs = []

    for instance_id, instance in enumerate(instances):
        if instance.find("#") != -1 or instance.find(";") != -1:
            continue
        native = try_parse_native_instance(instance, instance_id, problem_name)
        if native is not None:
            convs.append(native)
            continue

        start = instance.index("<") + 1
        end = instance.rindex(">")
        params_str = instance[start:end]
        args = parse_instance_string(params_str)

        is_v1_instance = instance.find("Xdl_CShuffle<") != -1

        if is_v1_instance:
            if len(args) != 51:
                raise RuntimeError(
                    f"Wrong number of parameters in the V1 XDL CShuffle instance string: {instance}\n"
                    + f"Expected 51 parameters for V1 instance. Found {len(args)} parameters."
                )
        else:
            raise RuntimeError(
                f"Only V1 XDL CShuffle instances are supported for backward data. Found instance: {instance}"
            )

        spec = args[13]
        block_size = int(args[17])
        m_per_block = int(args[18])
        n_per_block = int(args[19])
        k_per_block = int(args[20])
        ak1 = int(args[21])
        bk1 = int(args[22])
        m_per_xdl = int(args[23])
        n_per_xdl = int(args[24])
        m_xdl_per_wave = int(args[25])
        n_xdl_per_wave = int(args[26])
        a_scalar_per_vector = int(args[31])
        b_scalar_per_vector = int(args[38])
        c_scalar_per_vector = int(args[44])

        if ak1 != bk1:
            raise RuntimeError(
                f"Not supported instance {instance_id} since ak1 != bk1. ak1: {ak1}, bk1: {bk1} in instance: {instance}"
            )

        k1 = min(ak1, bk1)

        # TODO: Do we need split image for 3D bwd data convs?
        split_image = False

        # Default optimization parameters
        num_groups_to_merge = 1
        is_two_stage_instance = False
        is_explicit_gemm = False
        num_wave_groups = 1
        direct_load = False

        # Block GEMM pipeline parameters
        block_gemm_pipeline_scheduler = args[46]
        if block_gemm_pipeline_scheduler == "Default":
            block_gemm_pipeline_scheduler = "Intrawave"

        blk_gemm_pipeline_version = "v1"
        if block_gemm_pipeline_scheduler == "Interwave":
            blk_gemm_pipeline_version = "v1"

        # Sanity check for Block GEMM pipeline parameters
        # Scheduler must be either Intrawave or Interwave.
        # Version must be from v1 to v5
        if block_gemm_pipeline_scheduler not in ["Intrawave", "Interwave"]:
            raise RuntimeError(
                f"Invalid Block GEMM pipeline scheduler: {block_gemm_pipeline_scheduler} in instance: {instance}"
            )
        if blk_gemm_pipeline_version not in ["v1", "v2", "v3", "v4", "v5"]:
            raise RuntimeError(
                f"Invalid Block GEMM pipeline version: {blk_gemm_pipeline_version} in instance: {instance}"
            )

        double_smem_buffer = blk_gemm_pipeline_version == "v4"
        scheduler = block_gemm_pipeline_scheduler
        pipeline_version = blk_gemm_pipeline_version.upper()

        # Old CK pipeline version V5 maps to V6 for CK Tile
        if pipeline_version == "V5":
            pipeline_version = "V6"

        if direct_load:
            if pipeline_version == "V1":
                pipeline_version = "ASYNC_V1"
            elif pipeline_version == "V4":
                pipeline_version = "ASYNC_V4"
            else:
                raise RuntimeError(
                    f"Not supported pipeline for direct load: pipeline_version={pipeline_version} in instance: {instance}"
                )

        # WMMA
        dtype = get_dtype(problem_name)

        m_warp = int(m_per_block / (m_per_xdl * m_xdl_per_wave))
        n_warp = int(n_per_block / (n_per_xdl * n_xdl_per_wave))
        k_warp = int(block_size / (warp_size * m_warp * n_warp))
        k_per_xdl = min(max(k1, get_k_mfma(dtype, m_per_xdl, n_per_xdl)), k_per_block)
        if not check_wmma_instance(warp_size, k_per_block, k_warp, k_per_xdl, m_per_xdl, dtype):
            continue
        # Skip irregular vector sizes -- no HW vector load instructions for odd widths
        if not check_vectors(
            a_scalar_per_vector, b_scalar_per_vector, c_scalar_per_vector
        ):
            if verbose:
                print(
                    f"Skipping instance {instance_id} with irregular load since it's not supported yet."
                )
            continue

        # Skip multi-warp: single warp can't cover tile dim when it exceeds warp_size * vec
        if not check_warp_coverage(
            m_per_block,
            n_per_block,
            k_per_block,
            a_scalar_per_vector,
            b_scalar_per_vector,
            variant="bwd_data",
            warp_size=warp_size,
        ):
            if verbose:
                print(
                    f"Skipping instance {instance_id} with multiple warps per continous tile dim since it's not supported yet."
                )
            continue
        if not check_bwd_data_vec_coverage(
            m_per_block,
            n_per_block,
            k_per_block,
            m_warp,
            n_warp,
            k_warp,
            a_scalar_per_vector,
            b_scalar_per_vector,
            warp_size=warp_size,
        ):
            if verbose:
                print(
                    f"Skipping instance {instance_id} because current scalar per vector exceedes tile size"
                )
            continue

        conv = ConvInstanceTemplateParams(
            spec,
            [m_per_block, n_per_block, k_per_block],
            [m_warp, n_warp, k_warp],
            [m_per_xdl, n_per_xdl, k_per_xdl],
            double_smem_buffer,
            num_wave_groups,
            is_two_stage_instance,
            pipeline_version,
            scheduler,
            [a_scalar_per_vector, b_scalar_per_vector, c_scalar_per_vector],
            num_groups_to_merge,
            split_image,
            is_explicit_gemm,
            instance_id,
        )
        convs.append(conv)

    return convs


def get_signature_base(config):
    """Extract layout_dtype from config name, stripping variant suffixes.

    Config names follow {layout}_{dtype}[_{variant}], e.g. nhwgc_fp16_streamk.
    The signature is determined by layout and dtype only.
    """
    parts = config.split("_")
    return f"{parts[0]}_{parts[1]}"


def generate_instances_fwd(
    instances, problem_name, config, filter_pattern, instances_path, warp_size=32
):
    direction = "forward"
    signature_name = f"SIGNATURE_{get_signature_base(config).upper()}_FWD"
    instances = parse_fwd_instances(instances, problem_name, warp_size)
    generate_calls_inc(instances, problem_name, direction, filter_pattern)
    generate_defs_inc(
        instances, problem_name, signature_name, direction, filter_pattern
    )
    generate_conv_cpp(
        instances,
        problem_name,
        config,
        direction,
        signature_name,
        filter_pattern,
        instances_path,
    )


def generate_instances_bwd_weight(
    instances, problem_name, config, filter_pattern, instances_path, warp_size=32
):
    direction = "backward_weight"
    signature_name = f"SIGNATURE_{get_signature_base(config).upper()}_BWD_WEIGHT"
    instances = parse_bwd_weight_instances(instances, problem_name, warp_size)
    generate_calls_inc(instances, problem_name, direction, filter_pattern)
    generate_defs_inc(
        instances, problem_name, signature_name, direction, filter_pattern
    )
    generate_conv_cpp(
        instances,
        problem_name,
        config,
        direction,
        signature_name,
        filter_pattern,
        instances_path,
    )


def generate_instances_bwd_data(
    instances, problem_name, config, filter_pattern, instances_path, warp_size=32
):
    direction = "backward_data"
    signature_name = f"SIGNATURE_{get_signature_base(config).upper()}_BWD_DATA"
    instances = parse_bwd_data_instances(instances, problem_name, warp_size)
    generate_calls_inc(instances, problem_name, direction, filter_pattern)
    generate_defs_inc(
        instances, problem_name, signature_name, direction, filter_pattern
    )
    generate_conv_cpp(
        instances,
        problem_name,
        config,
        direction,
        signature_name,
        filter_pattern,
        instances_path,
    )


def process_direction(
    configs, direction, generate_func, configs_prefix, filter_pattern, instances_path, warp_size=32
):
    """Helper function to process a single direction."""
    for config in configs:
        instances = []
        generate_dir = Path(__file__).resolve().parent
        config_path = (
            f"{generate_dir}/configs/{direction}/{configs_prefix}/{config}.conf"
        )
        with open(config_path, "r") as file:
            instances = file.readlines()

        # Determine problem name based on direction
        if direction == "forward":
            problem_name = f"grouped_convolution_forward_tile_{config}"
        elif direction == "backward_weight":
            problem_name = f"grouped_convolution_backward_weight_tile_{config}"
        elif direction == "backward_data":
            problem_name = f"grouped_convolution_backward_data_tile_{config}"
        else:
            raise RuntimeError(f"Unknown direction: {direction}")

        generate_func(instances, problem_name, config, filter_pattern, instances_path, warp_size)


# ---------------------------------------------------------------------------
# Depthwise forward generation
# ---------------------------------------------------------------------------

DEPTHWISE_CONFIGS = [
    {
        "name": "ngchw_depthwise_fp32",
        "conf": "ngchw_depthwise.conf",
        "signature": "SIGNATURE_NGCHW_FP32_FWD",
    },
    {
        "name": "ngchw_depthwise_fp16",
        "conf": "ngchw_depthwise.conf",
        "signature": "SIGNATURE_NGCHW_FP16_FWD",
    },
    {
        "name": "ngchw_depthwise_bf16",
        "conf": "ngchw_depthwise.conf",
        "signature": "SIGNATURE_NGCHW_BF16_FWD",
    },
]


def parse_depthwise_config(conf_path: Path, verbose=True) -> list:
    """Parse a depthwise config file.

    Accepts the ``GroupedConvolutionForwardDepthwise<...>`` format.

    Returns a list of 12-element integer lists:
        [TileH, TileW, Filter, StrH, StrW, PadH, PadW,
         NBatch, SubTileH, SubTileW, InVecSize, OutVecSize]
    """
    instances = []
    for raw in conf_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "<" in line and ">" in line:
            start = line.index("<") + 1
            end = line.rindex(">")
            line = line[start:end]
        params = [int(x.strip()) for x in line.split(",")]
        if len(params) != 12:
            raise ValueError(
                f"Expected 12 parameters per depthwise instance, got {len(params)}: {raw!r}"
            )
        instances.append(params)
    return instances


def generate_depthwise_cpp(
    params: list, instance_name: str, signature: str, cpp_out: Path
) -> None:
    (
        tile_h,
        tile_w,
        filt,
        str_h,
        str_w,
        pad_h,
        pad_w,
        nbatch,
        sub_h,
        sub_w,
        in_vec,
        out_vec,
    ) = params

    parent_dir = Path(__file__).resolve().parent
    template_file = parent_dir / "include/grouped_convolution_depthwise_tile.cpp.in"
    content = template_file.read_text()

    content = content.replace("gen_signature", signature)
    content = content.replace("gen_instance_name", instance_name)
    content = content.replace("gen_block_size", "64")
    content = content.replace("gen_tile_h", str(tile_h))
    content = content.replace("gen_tile_w", str(tile_w))
    content = content.replace("gen_filter_h", str(filt))
    content = content.replace("gen_filter_w", str(filt))
    content = content.replace("gen_stride_h", str(str_h))
    content = content.replace("gen_stride_w", str(str_w))
    content = content.replace("gen_dilation_h", "1")
    content = content.replace("gen_dilation_w", "1")
    content = content.replace("gen_pad_h", str(pad_h))
    content = content.replace("gen_pad_w", str(pad_w))
    content = content.replace("gen_nbatch", str(nbatch))
    content = content.replace("gen_subtile_h", str(sub_h))
    content = content.replace("gen_subtile_w", str(sub_w))
    content = content.replace("gen_in_vec", str(in_vec))
    content = content.replace("gen_out_vec", str(out_vec))

    cpp_out.write_text(content)


def generate_depthwise_defs_inc(
    instances: list, config_name: str, signature: str, inc_path: Path
) -> None:
    lines = []
    for i in range(len(instances)):
        name = f"grouped_convolution_forward_tile_{config_name}_{i}"
        lines.append(
            f"std::tuple<bool, float, std::string> run_{name}(\n"
            f"    const ckt::Args<{signature}>& args,\n"
            f"    const ckt::Inputs<{signature}>& inputs,\n"
            f"    const ckt::Outputs<{signature}>& outputs,\n"
            f"    const ck_tile::stream_config& s_conf);"
        )
    inc_path.write_text("\n".join(lines) + "\n")


def generate_depthwise_calls_inc(
    instances: list, config_name: str, calls_path: Path
) -> None:
    lines = []
    for i in range(len(instances)):
        name = f"grouped_convolution_forward_tile_{config_name}_{i}"
        lines.append(f"run_alg(run_{name});")
    calls_path.write_text("\n".join(lines) + "\n")


def process_depthwise_forward(configs_prefix: str, instances_path: str) -> None:
    """Generate all depthwise forward instances."""
    generate_dir = Path(__file__).resolve().parent
    conf_dir = generate_dir / "configs/forward" / configs_prefix
    inc_dir = generate_dir / "instances" / "forward"
    cpp_base = Path(instances_path) / "forward"

    for cfg in DEPTHWISE_CONFIGS:
        name = cfg["name"]
        conf_path = conf_dir / cfg["conf"]
        signature = cfg["signature"]

        if not conf_path.exists():
            print(f"  Skipping {name}: config not found at {conf_path}")
            continue

        instances = parse_depthwise_config(conf_path)
        print(f"Processing {name}: {len(instances)} instances ...")

        cpp_dir = cpp_base / name
        cpp_dir.mkdir(parents=True, exist_ok=True)

        for i, params in enumerate(instances):
            instance_name = f"grouped_convolution_forward_tile_{name}_{i}"
            generate_depthwise_cpp(
                params, instance_name, signature, cpp_dir / f"{instance_name}.cpp"
            )

        generate_depthwise_defs_inc(
            instances,
            name,
            signature,
            inc_dir / f"grouped_convolution_forward_tile_{name}.inc",
        )
        generate_depthwise_calls_inc(
            instances,
            name,
            inc_dir / f"grouped_convolution_forward_tile_{name}_calls.inc",
        )

        print(f"  -> {cpp_dir}  ({len(instances)} .cpp files)")


fwd_configs = [
    "nhwgc_fp32",
    "nhwgc_fp16",
    "nhwgc_bf16",
    "ndhwgc_fp32",
    "ndhwgc_fp16",
    "ndhwgc_bf16",
]

bwd_weight_configs = [
    "nhwgc_fp32",
    "nhwgc_fp16",
    "nhwgc_bf16",
    "ndhwgc_fp32",
    "ndhwgc_fp16",
    "ndhwgc_bf16",
    "nhwgc_fp32_streamk",
    "nhwgc_fp16_streamk",
    "nhwgc_bf16_streamk",
    "ndhwgc_fp32_streamk",
    "ndhwgc_fp16_streamk",
    "ndhwgc_bf16_streamk",
]

bwd_data_configs = [
    "nhwgc_fp32",
    "nhwgc_fp16",
    "nhwgc_bf16",
    "ndhwgc_fp32",
    "ndhwgc_fp16",
    "ndhwgc_bf16",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate grouped conv CK Tile instances."
    )
    parser.add_argument(
        "--filter_pattern",
        type=str,
        default="convolution",
        help="Filter pattern for configs.",
    )
    parser.add_argument(
        "--mode",
        choices=["compilation", "tests", "profiler"],
        type=str,
        default="profiler",
        help="Generator modes. compilation - empty instance list, tests - limited instance list, profiler - generate all instances",
    )
    parser.add_argument(
        "--direction",
        choices=["forward", "backward_weight", "backward_data", "all"],
        type=str,
        default="all",
        help="Convolution direction for which to generate instances.",
    )
    parser.add_argument(
        "--instances_dir",
        type=str,
        default="../build/experimental/grouped_convolution_tile_instances",
        help="Directory store generated instances.",
    )
    parser.add_argument(
        "--gpu_target",
        choices=["gfx9", "gfx11", "gfx12"],
        type=str,
        default="gfx9",
        help="GPU target architecture. gfx9 uses warp_size=64, gfx11/gfx12 use warp_size=32.",
    )
    args = parser.parse_args()

    warp_size = get_warp_size(args.gpu_target)

    # apply empty filter
    if args.mode == "compilation":
        args.filter_pattern = "empty"
        configs_prefix = "profiler"
    elif args.mode == "tests":
        configs_prefix = "tests"
    elif args.mode == "profiler":
        configs_prefix = "profiler"
    else:
        raise RuntimeError("wrong mode")

    copy_includes(args.instances_dir)
    match args.direction:
        case "forward":
            process_direction(
                fwd_configs,
                args.direction,
                generate_instances_fwd,
                configs_prefix,
                args.filter_pattern,
                args.instances_dir,
                warp_size,
            )
            process_depthwise_forward(configs_prefix, args.instances_dir)
        case "backward_weight":
            process_direction(
                bwd_weight_configs,
                args.direction,
                generate_instances_bwd_weight,
                configs_prefix,
                args.filter_pattern,
                args.instances_dir,
                warp_size,
            )
        case "backward_data":
            process_direction(
                bwd_data_configs,
                args.direction,
                generate_instances_bwd_data,
                configs_prefix,
                args.filter_pattern,
                args.instances_dir,
                warp_size,
            )
        case "all":
            process_direction(
                fwd_configs,
                "forward",
                generate_instances_fwd,
                configs_prefix,
                args.filter_pattern,
                args.instances_dir,
                warp_size,
            )
            process_depthwise_forward(configs_prefix, args.instances_dir)
            process_direction(
                bwd_weight_configs,
                "backward_weight",
                generate_instances_bwd_weight,
                configs_prefix,
                args.filter_pattern,
                args.instances_dir,
                warp_size,
            )
            process_direction(
                bwd_data_configs,
                "backward_data",
                generate_instances_bwd_data,
                configs_prefix,
                args.filter_pattern,
                args.instances_dir,
                warp_size,
            )
