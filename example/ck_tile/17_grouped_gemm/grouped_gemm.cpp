// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

    using Traits              = ck_tile::TileGemmTraits<GemmConfig::kPadM,
                                                        GemmConfig::kPadN,
                                                        GemmConfig::kPadK,
                                                        ALayout,
                                                        BLayout,
                                                        CLayout>;
    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                                                 GemmConfig::kPadN,
                                                                 GemmConfig::kPadK,
                                                                 GemmConfig::DoubleSmemBuffer,
                                                                 ALayout,
                                                                 BLayout,
                                                                 CLayout,
                                                                 GemmConfig::TransposeC>;
    using GemmPipelineProblem =
        ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType, GemmShape, Traits>;

    using BaseGemmPipeline = typename PipelineTypeTraits<
        GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

    const ck_tile::index_t k_grain = gemm_descs[0].k_batch * GemmConfig::K_Tile;
    const ck_tile::index_t K_split = (gemm_descs[0].K + k_grain - 1) / k_grain * GemmConfig::K_Tile;
    const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
    const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
    const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);

    float ave_time{0};

    const auto Run =
        [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
            constexpr bool has_hot_loop_v   = has_hot_loop_.value;
            constexpr auto tail_number_v    = tail_number_.value;
            constexpr auto scheduler        = GemmConfig::Scheduler;
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                               BDataType,
                                                                               AccDataType,
                                                                               GemmShape,
                                                                               GemmUniversalTraits,
                                                                               scheduler,
                                                                               has_hot_loop_v,
                                                                               tail_number_v>;

            using GemmPipeline = typename PipelineTypeTraits<
                GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;
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
                                                 UniversalGemmProblem::TransposeC,
                                                 memory_operation>>;
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
                std::cout << "Launching kernel: " << Kernel::GetName()
                          << " with args:" << " grid: {" << grids.x << ", " << grids.y << ", "
                          << grids.z << "}" << ", blocks: {" << blocks.x << ", " << blocks.y << ", "
                          << blocks.z << "}" << std::endl;
            }

            return ave_time = ck_tile::launch_kernel(
                       s,
                       ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                           Kernel{},
                           grids,
                           blocks,
                           0,
                           ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                           gemm_descs.size()));
        };

    const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
        if(gemm_descs[0].k_batch == 1)
        {
            return Run(has_hot_loop_,
                       tail_number_,
                       ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                  ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            return Run(has_hot_loop_,
                       tail_number_,
                       ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                  ck_tile::memory_operation_enum::atomic_add>{});
        }
    };

    return ave_time = BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
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
                            void* kargs_ptr,
                            bool splitk)
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

    float ave_time{0};

    const auto Run = [&](const auto memory_operation_) {
        constexpr auto scheduler        = GemmConfig::Scheduler;
        constexpr auto memory_operation = memory_operation_.value;

        // We create the GEMM pipeline without specifying hotloop or tailnumber.
        // These are automatically run inside the kernel based on the given input data.
        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;
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
                                             UniversalGemmProblem::TransposeC,
                                             memory_operation>>;
        using Kernel      = ck_tile::GroupedGemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;
        const dim3 blocks = Kernel::BlockSize();
        const dim3 grids  = Kernel::MaxOccupancyGridSize(s);

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel: " << Kernel::GetName() << " with args:" << " grid: {"
                      << grids.x << ", " << grids.y << ", " << grids.z << "}" << ", blocks: {"
                      << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << std::endl;
        }

        return ave_time = ck_tile::launch_kernel(
                   s,
                   ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
                       Kernel{},
                       grids,
                       blocks,
                       0,
                       ck_tile::cast_pointer_to_constant_address_space(kargs_ptr),
                       num_groups));
    };

    if(!splitk)
    {
        return ave_time = Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                         ck_tile::memory_operation_enum::set>{});
    }
    else
    {
        return ave_time =
                   Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                  ck_tile::memory_operation_enum::atomic_add>{});
    }
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
    else if(a_layout == "C" && b_layout == "C")
    {
        return run_grouped_gemm_example_with_layouts<GemmConfig,
                                                     ADataType,
                                                     BDataType,
                                                     CDataType,
                                                     AccDataType>(argc, argv, Col{}, Col{}, Row{});
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

    if(data_type == "fp16")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::half_t>, ck_tile::half_t>(
            a_layout, b_layout, argc, argv);
    }
    else if(data_type == "fp8")
    {
        return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>, ck_tile::fp8_t>(
            a_layout, b_layout, argc, argv);
    }
    else
    {
        throw std::runtime_error("Unsupported data type configuration.");
    }
}

int main(int argc, char* argv[])
{
#if CK_TILE_USE_WMMA
    return !run_grouped_gemm_example<GemmConfigComputeV4_Wmma>(argc, argv);
#else
    return !run_grouped_gemm_example<GemmConfigComputeV4>(argc, argv) ||
           !run_grouped_gemm_example<GemmConfigComputeV3_2>(argc, argv) ||
           !run_grouped_gemm_example<GemmConfigComputeV4_V2>(argc, argv);
#endif
}
