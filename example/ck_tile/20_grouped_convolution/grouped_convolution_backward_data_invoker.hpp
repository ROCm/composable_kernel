// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "grouped_convolution_utils.hpp"

struct GroupedConvolutionBackwardDataInvoker
{

    template <ck_tile::index_t NDimSpatial,
              typename GemmConfig,
              typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType,
              typename InLayout,
              typename WeiLayout,
              typename OutLayout,
              typename DsDataType     = ck_tile::tuple<>,
              typename DsLayout       = ck_tile::tuple<>,
              typename CDEElementWise = ck_tile::element_wise::PassThrough>
    static float grouped_conv_bwd_data(const ck_tile::GroupedConvBwdDataHostArgs& args,
                                       const ck_tile::stream_config& s)
    {
        constexpr int kBlockPerCu = 1;

        // Implicit GEMM Traits
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::
                sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
            GemmConfig::PermuteA,
            GemmConfig::PermuteB>;

        constexpr ck_tile::index_t VectorSizeA = 8;
        constexpr ck_tile::index_t VectorSizeB = 8;
        constexpr ck_tile::index_t VectorSizeC = 8;

        constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;
        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       GemmConfig::TileParitionerGroupNum,
                                                       GemmConfig::TileParitionerM01>;
        using GroupedConvTraitsType = ck_tile::GroupedConvTraits<NDimSpatial,
                                                                 ConvSpec,
                                                                 InLayout,
                                                                 WeiLayout,
                                                                 DsLayout,
                                                                 OutLayout,
                                                                 VectorSizeA,
                                                                 VectorSizeB,
                                                                 VectorSizeC>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
            GemmConfig::kPadM,
            GemmConfig::kPadN,
            GemmConfig::kPadK,
            GemmConfig::DoubleSmemBuffer,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdData::AsLayout,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdData::BsLayout,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdData::CLayout,
            GemmConfig::TransposeC,
            GemmConfig::UseStructuredSparsity,
            false, // Persistent,
            GemmConfig::NumWaveGroups>;

        using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
            OutDataType,
            WeiDataType,
            AccDataType,
            GemmShape,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdData,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            InDataType,
            true,
            VectorSizeA,
            VectorSizeB>;

        using BaseGemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

        const ck_tile::index_t gemm_k =
            args.K_ * std::accumulate(args.filter_spatial_lengths_.begin(),
                                      args.filter_spatial_lengths_.end(),
                                      1,
                                      std::multiplies<ck_tile::index_t>());

        const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
        const ck_tile::index_t K_split     = (gemm_k + k_grain - 1) / k_grain * GemmConfig::K_Tile;
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

                using UniversalGemmProblem =
                    ck_tile::UniversalGemmPipelineProblem<OutDataType,
                                                          WeiDataType,
                                                          AccDataType,
                                                          GemmShape,
                                                          GemmUniversalTraits,
                                                          scheduler,
                                                          has_hot_loop_v,
                                                          tail_number_v,
                                                          ck_tile::element_wise::PassThrough,
                                                          ck_tile::element_wise::PassThrough,
                                                          InDataType,
                                                          true,
                                                          VectorSizeA,
                                                          VectorSizeB>;

                using GemmPipeline = typename PipelineTypeTraits<
                    GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

                using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                    OutDataType,
                    WeiDataType,
                    DsDataType,
                    AccDataType,
                    InDataType,
                    typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                    ck_tile::tensor_layout::gemm::RowMajor,
                    CDEElementWise,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    GemmConfig::M_Warp,
                    GemmConfig::N_Warp,
                    GemmConfig::M_Warp_Tile,
                    GemmConfig::N_Warp_Tile,
                    GemmConfig::K_Warp_Tile,
                    GemmConfig::TransposeC,
                    memory_operation,
                    1,
                    true,
                    GroupedConvTraitsType::VectorSizeC>>;

                using Kernel = ck_tile::GroupedConvolutionBackwardDataKernel<GroupedConvTraitsType,
                                                                             TilePartitioner,
                                                                             GemmPipeline,
                                                                             ConvEpilogue>;
                auto kargs   = Kernel::MakeKernelArgs(args);

                const dim3 grids  = Kernel::GridSize(args);
                const dim3 blocks = Kernel::BlockSize();

                if(!Kernel::IsSupportedArgument(kargs))
                {
                    throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
                }

                if(s.log_level_ > 0)
                {
                    std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                              << "shape: " << GemmShape::GetName() << '\n'
                              << "problem: " << UniversalGemmProblem::GetName() << '\n'
                              << "pipeline: " << GemmPipeline::GetName() << '\n'
                              << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                              << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                              << "}" << '\n'
                              << "Vector size A: " << GemmPipeline::GetVectorSizeA()
                              << ", Vector size B: " << GemmPipeline::GetVectorSizeB()
                              << ", Vector size C: " << ConvEpilogue::GetVectorSizeC() << std::endl;
                }

                ave_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

                return ave_time;
            };

        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
            if(args.k_batch == 1)
            {
                Run(has_hot_loop_, tail_number_, MemoryOpSet{});
            }
            else
            {
                Run(has_hot_loop_, tail_number_, MemoryOpAtomicAdd{});
            }
        };

        BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
        return ave_time;
    }
};
