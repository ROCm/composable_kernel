// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// Regular grouped convolution invoker (no split-image)
// This invoker demonstrates regular convolution without split-image.
// It always uses Kernel<false> (split-image disabled).
// For large images that require split-image, use
// grouped_convolution_forward_split_image_invoker.hpp

#pragma once

#include "grouped_convolution_utils.hpp"

struct GroupedConvolutionForwardInvoker
{
    template <ck_tile::index_t NDimSpatial,
              typename ConvConfig,
              typename InDataType,
              typename WeiDataType,
              typename AccDataType,
              typename OutDataType,
              typename InLayout,
              typename WeiLayout,
              typename OutLayout,
              typename DsDataType    = ck_tile::tuple<>,
              typename DsLayout      = ck_tile::tuple<>,
              typename CDElementWise = ck_tile::element_wise::PassThrough>
    static float grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs<CDElementWise>& args,
                                  const ck_tile::stream_config& s)
    {
        if(s.log_level_ > 0)
        {
            std::cout << "[INVOKER] grouped_conv_fwd called, NDimSpatial=" << NDimSpatial << "\n";
        }
        // Implicit GEMM Traits
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
            ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
            ck_tile::sequence<ConvConfig::M_Warp_Tile,
                              ConvConfig::N_Warp_Tile,
                              ConvConfig::K_Warp_Tile>>;

        constexpr auto ConvSpec     = ck_tile::ConvolutionSpecialization::Default;
        using GroupedConvTraitsType = ck_tile::GroupedConvTraits<NDimSpatial,
                                                                 ConvSpec,
                                                                 InLayout,
                                                                 WeiLayout,
                                                                 DsLayout,
                                                                 OutLayout,
                                                                 ConvConfig::VectorSizeA,
                                                                 ConvConfig::VectorSizeB,
                                                                 ConvConfig::VectorSizeC,
                                                                 ConvConfig::NumGroupsToMerge>;

        using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
            GemmShape,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
            GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
            GroupedConvTraitsType::FixedGemmParams::kPadM,
            GroupedConvTraitsType::FixedGemmParams::kPadN,
            GroupedConvTraitsType::FixedGemmParams::kPadK,
            ConvConfig::DoubleSmemBuffer,
            typename GroupedConvTraitsType::AsLayoutFwd,
            typename GroupedConvTraitsType::BsLayoutFwd,
            typename GroupedConvTraitsType::CLayoutFwd,
            GroupedConvTraitsType::FixedGemmParams::TransposeC,
            GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
            GroupedConvTraitsType::FixedGemmParams::Persistent,
            ConvConfig::NumWaveGroups>;

        using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
            InDataType,
            WeiDataType,
            AccDataType,
            GemmShape,
            typename GroupedConvTraitsType::template GroupedConvImplicitGemmTraitsFwd<
                ConvConfig::NumWaveGroups>,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            OutDataType,
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            GroupedConvTraitsType::VectorSizeA,
            GroupedConvTraitsType::VectorSizeB>;

        using BaseGemmPipeline = typename PipelineTypeTraits<
            ConvConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

        const ck_tile::index_t gemm_k =
            args.C_ * std::accumulate(args.filter_spatial_lengths_.begin(),
                                      args.filter_spatial_lengths_.end(),
                                      1,
                                      std::multiplies<ck_tile::index_t>());

        // Split-K parameters
        const ck_tile::index_t k_grain     = args.k_batch * ConvConfig::K_Tile;
        const ck_tile::index_t K_split     = (gemm_k + k_grain - 1) / k_grain * ConvConfig::K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        float ave_time{0};

        // =====================================================================
        // Regular Convolution: Simple, no split-image
        // =====================================================================
        const auto Run =
            [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
                constexpr bool has_hot_loop_v   = has_hot_loop_.value;
                constexpr auto tail_number_v    = tail_number_.value;
                constexpr auto scheduler        = ConvConfig::Scheduler;
                constexpr auto memory_operation = memory_operation_.value;

                using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
                    InDataType,
                    WeiDataType,
                    AccDataType,
                    GemmShape,
                    GemmUniversalTraits,
                    scheduler,
                    has_hot_loop_v,
                    tail_number_v,
                    ck_tile::element_wise::PassThrough,
                    ck_tile::element_wise::PassThrough,
                    OutDataType,
                    GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
                    GroupedConvTraitsType::VectorSizeA,
                    GroupedConvTraitsType::VectorSizeB>;

                using GemmPipeline = typename PipelineTypeTraits<
                    ConvConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

                using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                    InDataType,
                    WeiDataType,
                    DsDataType,
                    AccDataType,
                    OutDataType,
                    typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                    typename GroupedConvTraitsType::FixedGemmParams::ELayout,
                    CDElementWise,
                    TilePartitioner::MPerBlock,
                    TilePartitioner::NPerBlock,
                    ConvConfig::M_Warp,
                    ConvConfig::N_Warp,
                    ConvConfig::M_Warp_Tile,
                    ConvConfig::N_Warp_Tile,
                    ConvConfig::K_Warp_Tile,
                    GroupedConvTraitsType::FixedGemmParams::TransposeC,
                    memory_operation,
                    ConvConfig::NumWaveGroups,
                    GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
                    GroupedConvTraitsType::VectorSizeC>>;

                using Kernel = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                                        TilePartitioner,
                                                                        GemmPipeline,
                                                                        ConvEpilogue>;
                auto kargs   = Kernel::MakeKernelArgs(args);

                const dim3 grids  = Kernel::GridSize(kargs);
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

                ave_time = ck_tile::launch_kernel(s,
                                                  ck_tile::make_kernel<ConvConfig::kBlockPerCu>(
                                                      Kernel{}, grids, blocks, 0, kargs));

                return ave_time;
            };

        // =====================================================================
        // Split-K lambda
        // =====================================================================
        const auto RunSplitk = [&](const auto has_hot_loop_, const auto tail_number_) {
            if(args.k_batch == 1)
            {
                Run.template operator()(has_hot_loop_, tail_number_, MemoryOpSet{});
            }
            else
            {
                Run.template operator()(has_hot_loop_, tail_number_, MemoryOpAtomicAdd{});
            }
        };

        // =====================================================================
        // Regular Convolution Example: ALWAYS uses regular path (Kernel<false>)
        // =====================================================================
        // This example demonstrates regular convolution without split-image.
        // For large images that don't fit in memory, use
        // grouped_convolution_forward_split_image.cpp

        // Launch kernel using regular path (no split-image)
        BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);

        return ave_time;
    }
};
