// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>
#include <memory>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/host.hpp"
#include "grouped_gemm.hpp"

template <typename GemmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float grouped_gemm(const std::vector<grouped_gemm_kargs>& gemm_descs,
                   const ck_tile::stream_config& s,
                   void* kargs_ptr)
{

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 GemmConfig::TransposeC>;

    constexpr auto scheduler = GemmConfig::Scheduler;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                       BDataType,
                                                                       AccDataType,
                                                                       GemmShape,
                                                                       GemmUniversalTraits,
                                                                       scheduler>;

    using GemmPipeline = typename PipelineTypeTraits<GemmConfig::Pipeline>::template GemmPipeline<
        UniversalGemmProblem>;

    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ADataType,
                                         BDataType,
                                         DsDataType,
                                         AccDataType,
                                         CDataType,
                                         DsLayout,
                                         CLayout,
                                         CDEElementWise,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC>>;
    using Kernel = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    auto kargs   = Kernel::MakeKargs(gemm_descs);
    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("Kernel arguments not supported!");
    }

    const dim3 blocks = Kernel::BlockSize();
    const dim3 grids  = Kernel::GridSize(gemm_descs);

    HIP_CHECK_ERROR(hipMemcpyWithStream(kargs_ptr,
                                        kargs.data(),
                                        get_workspace_size(gemm_descs),
                                        hipMemcpyHostToDevice,
                                        s.stream_id_));

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                  << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                  << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
    }

    return ck_tile::launch_kernel(s,
                                  ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                                      Kernel{},
                                      grids,
                                      blocks,
                                      0,
                                      ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                      gemm_descs.size()));
}

template <typename GemmConfig,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType>
float grouped_gemm_tileloop(const ck_tile::stream_config& s,
                            const ck_tile::index_t num_groups,
                            void* kargs_ptr)
{
    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
        ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
        ck_tile::
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>>;
    using TilePartitioner =
        ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                   GemmConfig::TileParitionerGroupNum,
                                                   GemmConfig::TileParitionerM01>;

    using GemmUniversalTraits =
        ck_tile::PersistentTileGemmUniversalTraits<GemmConfig::kPadM,
                                                   GemmConfig::kPadN,
                                                   GemmConfig::kPadK,
                                                   GemmConfig::DoubleSmemBuffer,
                                                   ALayout,
                                                   BLayout,
                                                   CLayout>;

    constexpr auto scheduler = GemmConfig::Scheduler;

    // We create the GEMM pipeline without specifying hotloop or tailnumber.
    // These are automatically run inside the kernel based on the given input data.
    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                       BDataType,
                                                                       AccDataType,
                                                                       GemmShape,
                                                                       GemmUniversalTraits,
                                                                       scheduler>;

    using GemmPipeline = typename PipelineTypeTraits<GemmConfig::Pipeline>::template GemmPipeline<
        UniversalGemmProblem>;
    using GemmEpilogue = ck_tile::CShuffleEpilogue<
        ck_tile::CShuffleEpilogueProblem<ADataType,
                                         BDataType,
                                         ck_tile::tuple<>,
                                         AccDataType,
                                         CDataType,
                                         ck_tile::tuple<>,
                                         CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         GemmConfig::M_Warp,
                                         GemmConfig::N_Warp,
                                         GemmConfig::M_Warp_Tile,
                                         GemmConfig::N_Warp_Tile,
                                         GemmConfig::K_Warp_Tile,
                                         UniversalGemmProblem::TransposeC>>;
    using Kernel      = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
    const dim3 blocks = Kernel::BlockSize();
    const dim3 grids  = Kernel::MaxOccupancyGridSize(s);

    if(s.log_level_ > 0)
    {
        std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                  << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                  << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
    }

    return ck_tile::launch_kernel(s,
                                  ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                                      Kernel{},
                                      grids,
                                      blocks,
                                      0,
                                      ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                                      num_groups));
}

#include "run_grouped_gemm_example.inc"

template <typename GemmConfig, typename PrecType>
int run_gemm_example_prec_type(std::string a_layout, std::string b_layout, int argc, char* argv[])
{
    using Row   = ck_tile::tensor_layout::gemm::RowMajor;
    using Col   = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Types = GemmTypeConfig<PrecType>;
    // Specific type aliases for easy access
    using ADataType   = typename Types::ADataType;
    using BDataType   = typename Types::BDataType;
    using AccDataType = typename Types::AccDataType;
    using CDataType   = typename Types::CDataType;

    if(a_layout == "R" && b_layout == "C")
    {
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Row{}, Col{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "R")
    {
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Row{}, Row{}, Row{});
    }
    else if(a_layout == "C" && b_layout == "R")
    {
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Col{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A and B tensors!");
    }
}

template <template <typename PrecType> typename GemmConfig>
int run_grouped_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");
    const std::string data_type = arg_parser.get_str("prec");

    if(data_type == "bf16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::bf16_t>, ck_tile::bf16_t>(
            a_layout, b_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type configuration.");
    }
}

// Determine appropriate tile config based on N alignment and config selection
int run_grouped_gemm_example_with_n_check(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout  = arg_parser.get_str("a_layout");
    const std::string b_layout  = arg_parser.get_str("b_layout");
    const std::string data_type = arg_parser.get_str("prec");
    const int group_count       = arg_parser.get_int("group_count");
    const std::string config    = arg_parser.get_str("config");
    std::vector<ck_tile::index_t> Ns = arg_parser.get_int_vec("Ns");

    // Check N alignment for all groups
    bool all_n_mod_256 = true;
    bool all_n_mod_128 = true;
    bool all_n_mod_64  = true;

    
    if(Ns.size() == static_cast<size_t>(group_count))
    {
        for(const auto& n : Ns)
        {
            if(n % 256 != 0)
                all_n_mod_256 = false;
            if(n % 128 != 0)
                all_n_mod_128 = false;
            if(n % 64 != 0)
                all_n_mod_64 = false;
        }
    }

    if(data_type == "bf16")
    {
        // Memory pipeline configs
        if(config == "memory_interwave")
        {
            std::cout << "[Config] Memory Interwave 128x32" << std::endl;
            return run_gemm_example_prec_type<GemmConfigMemoryInterwave<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "memory_intrawave")
        {
            std::cout << "[Config] Memory Intrawave 128x32" << std::endl;
            return run_gemm_example_prec_type<GemmConfigMemoryIntrawave<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "mem_inter_128x128")
        {
            if(!all_n_mod_128)
                throw std::runtime_error("N must be multiple of 128");
            std::cout << "[Config] Memory Interwave 128x128" << std::endl;
            return run_gemm_example_prec_type<GemmConfigMemoryInterwave_128x128<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "mem_intra_128x128")
        {
            if(!all_n_mod_128)
                throw std::runtime_error("N must be multiple of 128");
            std::cout << "[Config] Memory Intrawave 128x128" << std::endl;
            return run_gemm_example_prec_type<GemmConfigMemoryIntrawave_128x128<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "mem_inter_256x128")
        {
            if(!all_n_mod_128)
                throw std::runtime_error("N must be multiple of 128");
            std::cout << "[Config] Memory Interwave 256x128" << std::endl;
            return run_gemm_example_prec_type<GemmConfigMemoryInterwave_256x128<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "mem_inter_256x256")
        {
            if(!all_n_mod_256)
                throw std::runtime_error("N must be multiple of 256");
            std::cout << "[Config] Memory Interwave 256x256" << std::endl;
            return run_gemm_example_prec_type<GemmConfigMemoryInterwave_256x256<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        // 128x128 configs
        else if(config == "v3_128x128_16_k1" || config == "128w16k1")
        {
            if(!all_n_mod_128)
                throw std::runtime_error("N must be multiple of 128");
            std::cout << "[Config] 128x128, warp=16, kBlockPerCu=1" << std::endl;
            return run_gemm_example_prec_type<GemmConfigComputeV3_128x128_16_k1<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "v3_128x128_16_k2" || config == "128w16k2")
        {
            if(!all_n_mod_128)
                throw std::runtime_error("N must be multiple of 128");
            std::cout << "[Config] 128x128, warp=16, kBlockPerCu=2" << std::endl;
            return run_gemm_example_prec_type<GemmConfigComputeV3_128x128_16_k2<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        // 64x64 config
        else if(config == "v3_64x64_16_k1" || config == "64w16k1")
        {
            if(!all_n_mod_64)
                throw std::runtime_error("N must be multiple of 64");
            std::cout << "[Config] 64x64, warp=16, kBlockPerCu=1" << std::endl;
            return run_gemm_example_prec_type<GemmConfigComputeV3_64x64_16_k1<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        // grad_B optimized configs
        else if(config == "256x256_k2")
        {
            if(!all_n_mod_256)
                throw std::runtime_error("N must be multiple of 256");
            std::cout << "[Config] 256x256, kBlockPerCu=2" << std::endl;
            return run_gemm_example_prec_type<GemmConfigComputeV3_256x256_k2<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "128x256")
        {
            if(!all_n_mod_256)
                throw std::runtime_error("N must be multiple of 256");
            std::cout << "[Config] 128x256" << std::endl;
            return run_gemm_example_prec_type<GemmConfigComputeV3_128x256<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "256x128_k2")
        {
            if(!all_n_mod_128)
                throw std::runtime_error("N must be multiple of 128");
            std::cout << "[Config] 256x128, kBlockPerCu=2" << std::endl;
            return run_gemm_example_prec_type<GemmConfigComputeV3_256x128_k2<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "128x256_k2")
        {
            if(!all_n_mod_256)
                throw std::runtime_error("N must be multiple of 256");
            std::cout << "[Config] 128x256, kBlockPerCu=2" << std::endl;
            return run_gemm_example_prec_type<GemmConfigComputeV3_128x256_k2<ck_tile::bf16_t>, ck_tile::bf16_t>(
                a_layout, b_layout, argc, argv);
        }
        else if(config == "compute_v3" || config == "")
        {
            // Default: auto-select based on N alignment
            if(all_n_mod_256)
            {
                std::cout << "[Config] Using 256x256 tile (N % 256 == 0)" << std::endl;
                return run_gemm_example_prec_type<GemmConfigComputeV3_2<ck_tile::bf16_t>, ck_tile::bf16_t>(
                    a_layout, b_layout, argc, argv);
            }
            else if(all_n_mod_128)
            {
                std::cout << "[Config] Using 256x128 tile (N % 128 == 0, N % 256 != 0)" << std::endl;
                return run_gemm_example_prec_type<GemmConfigComputeV3_256x128<ck_tile::bf16_t>, ck_tile::bf16_t>(
                    a_layout, b_layout, argc, argv);
            }
            else
            {
                throw std::runtime_error("Unsupported N alignment for compute_v3 config.");
            }
        }
        else
        {
            throw std::runtime_error("Unknown config: " + config + ". Use: compute_v3, compute_v3_32x128, compute_v3_128x128, memory_interwave, memory_intrawave");
        }
    }
    else
    {
        throw std::runtime_error("Unsupported data type configuration.");
    }
}

int main(int argc, char* argv[])
{
    return !run_grouped_gemm_example_with_n_check(argc, argv);
}
