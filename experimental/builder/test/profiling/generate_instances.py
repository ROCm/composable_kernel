# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path


class ConvInstanceTemplateParams:
    def __init__(
        self,
        specialization,
        tile_size,
        warps,
        warp_tile,
        double_smem_buffer,
        num_wave_groups,
        pipeline_version,
        scheduler,
        scalar_per_vector,
        num_groups_to_merge,
        split_image,
        explicit_gemm,
        id,
    ):
        self.specialization = specialization
        self.tile_size = tile_size
        self.warps = warps
        self.warp_tile = warp_tile
        self.double_smem_buffer = double_smem_buffer
        self.num_wave_groups = num_wave_groups
        self.pipeline_version = pipeline_version
        self.scheduler = scheduler
        self.scalar_per_vector = scalar_per_vector
        self.num_groups_to_merge = num_groups_to_merge
        self.split_image = split_image
        self.explicit_gemm = explicit_gemm
        self.id = id

    def get_optimizations(self):
        explicit_gemm = "true" if self.explicit_gemm else "false"
        split_image = "true" if self.split_image else "false"
        num_groups_to_merge = str(self.num_groups_to_merge)
        return f"ckt::TileOptimizations{{.num_groups_to_merge = {num_groups_to_merge}, .split_image = {split_image}, .explicit_gemm = {explicit_gemm}}}"

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
        pipeline_version = self.pipeline_version[-1:]
        scheduler = (
            "INTRAWAVE" if self.scheduler.find("Intrawave") != -1 else "INTERWAVE"
        )
        return f"""ckt::TileBlockGemm{{
                    .warps              = {{.m = {self.warps[0]}, .n = {self.warps[1]}, .k = {self.warps[2]}}},
                    .warp_tile          = {{.m = {self.warp_tile[0]}, .n = {self.warp_tile[1]}, .k = {self.warp_tile[2]}}},
                    .double_smem_buffer = {double_smem_buffer},
                    .num_wave_groups    = {self.num_wave_groups},
                    .pipeline_version   = ckb::PipelineVersion::V{pipeline_version},
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
        raise RuntimeError("wrong dtype")


def generate_calls_inc(instances, problem_name, direction):
    with open(
        f"../experimental/builder/test/profiling/{problem_name}_calls.inc", "w"
    ) as f:
        for instance in instances:
            instance_name = problem_name + "_" + str(instance.id)
            f.write(f"run_alg(run_{instance_name});\n")


def generate_defs_inc(instances, problem_name, signature, direction):
    with open(f"../experimental/builder/test/profiling/{problem_name}.inc", "w") as f:
        for instance in instances:
            instance_name = problem_name + "_" + str(instance.id)
            f.write(
                f"std::tuple<float, std::string> run_{instance_name}(\n"
                f"   const ckt::Args<{signature}>& args,\n"
                f"   const ckt::Inputs<{signature}>& inputs,\n"
                f"   const ckt::Outputs<{signature}>& outputs,\n"
                f"   const ck_tile::stream_config& s_conf);\n"
            )


def generate_fwd_cpp(instances, problem_name, config, direction):
    for instance in instances:
        instance_name = problem_name + "_" + str(instance.id)
        directory_path = Path(f"../experimental/builder/test/profiling/src/{config}")
        directory_path.mkdir(parents=True, exist_ok=True)
        with open(
            f"../experimental/builder/test/profiling/src/{config}/{instance_name}.cpp",
            "w",
        ) as f:
            f.write(
                f"// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.\n"
                f"// SPDX-License-Identifier: MIT\n"
                f'#include "../instance_includes.inc"\n'
                f'#include "../{problem_name}_signature.inc"\n'
                f"namespace ck_tile::builder::profiling {{\n"
                f"std::tuple<float, std::string> run_{instance_name}(\n"
                f"   const ckt::Args<SIGNATURE>& args,\n"
                f"   const ckt::Inputs<SIGNATURE>& inputs,\n"
                f"   const ckt::Outputs<SIGNATURE>& outputs,\n"
                f"   const ck_tile::stream_config& s_conf) {{\n"
            )

            f.write(
                f"constexpr auto ALGORITHM = cku::ConvAlgorithm_Tile_GroupedConvolutionKernel{{}}\n"
                f"   .with_tile_specializations({instance.get_specialization()})\n"
                f"   .with_tile_thread_block({instance.get_thread_block()})\n"
                f"   .with_tile_block_gemm({instance.get_block_gemm_desc()})\n"
                f"   .with_tile_transfer({instance.get_block_transfer()})\n"
                f"   .with_tile_optimizations(\n"
                f"      {instance.get_optimizations()});\n"
            )

            f.write(
                '#include "../instance_run.inc"\n'
                "}\n"
                "} // namespace ck_tile::builder::profiling\n"
            )


def parse_fwd_instances(instances, problem_name):
    convs = []
    for instance_id, instance in enumerate(instances):
        if instance.find("#") != -1 or instance.find(";") != -1:
            continue
        instance_args_list = instance[instance.find("<") + 1 : instance.find(">")]
        args = instance_args_list.split(", ")

        block_size = int(args[0])
        m_per_block = int(args[1])
        n_per_block = int(args[2])
        k_per_block = int(args[3])
        spec = args[4]
        m_per_xdl = int(args[5])
        n_per_xdl = int(args[6])
        m_xdl_per_wave = int(args[7])
        n_xdl_per_wave = int(args[8])
        a_scalar_per_vector = int(args[9])
        b_scalar_per_vector = int(args[10])
        c_scalar_per_vector = int(args[11])
        if len(args) == 15:
            num_groups_to_merge = int(args[14])
        elif len(args) != 16 and len(args) != 14:
            raise RuntimeError("wrong number of parameters")
        else:
            num_groups_to_merge = 1
        split_image = instance.find("Large") != -1
        double_smem_buffer = instance.find("BlkGemmPipelineVersion: v4") != -1
        num_wave_groups = 2 if instance.find("BlkGemmPipelineVersion: v5") != -1 else 1
        scheduler = (
            "Intrawave" if instance.find("BlkGemmPipelineScheduler") == -1 else args[14]
        )
        pipeline_version = (
            "v1" if instance.find("BlkGemmPipelineVersion") == -1 else args[15]
        )

        m_warp = int(m_per_block / (m_per_xdl * m_xdl_per_wave))
        n_warp = int(n_per_block / (n_per_xdl * n_xdl_per_wave))
        warp_size = 64
        k_warp = int(block_size / (warp_size * m_warp * n_warp))
        dtype = get_dtype(problem_name)
        # TODO: Make it more flexible
        # k_per_xdl = f"ck_tile::get_k_warp_tile<{dtype}, {m_per_xdl}>()"
        k_per_xdl = 8 if dtype == "float" else 16

        conv = ConvInstanceTemplateParams(
            spec,
            [m_per_block, n_per_block, k_per_block],
            [m_warp, n_warp, k_warp],
            [m_per_xdl, n_per_xdl, k_per_xdl],
            double_smem_buffer,
            num_wave_groups,
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


def generate_instances_fwd(instances, problem_name, config):
    direction = "forward"
    instances = parse_fwd_instances(instances, problem_name)
    generate_calls_inc(instances, problem_name, direction)
    generate_defs_inc(
        instances, problem_name, f"SIGNATURE_{config.upper()}_FWD", direction
    )
    generate_fwd_cpp(instances, problem_name, config, direction)


if __name__ == "__main__":
    fwd_configs = [
        "nhwgc_fp32",
        "nhwgc_fp16",
        "nhwgc_bf16",
        "ndhwgc_fp32",
        "ndhwgc_fp16",
        "ndhwgc_bf16",
    ]

    for config in fwd_configs:
        instances = []
        config_path = f"../experimental/builder/test/profiling/configs/{config}.conf"
        with open(config_path, "r") as file:
            instances = file.readlines()
        problem_name = f"grouped_convolution_forward_tile_{config}"
        generate_instances_fwd(instances, problem_name, config)
