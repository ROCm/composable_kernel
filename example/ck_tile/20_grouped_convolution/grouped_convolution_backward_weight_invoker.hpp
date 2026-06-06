// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_convolution_utils.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp"

/// @brief Partitioner policies for the conv bwd weight invoker.
/// SplitKPartitionerPolicy is the default (data-parallel + split-K).
/// StreamKPartitionerPolicy selects StreamK work distribution.
struct SplitKPartitionerPolicy
{
    template <typename GemmShape, typename GroupedConvTraitsType>
    using type = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;
};

template <ck_tile::StreamKReductionStrategy ReductionStrategy, bool Persistent = false>
struct StreamKPartitionerPolicy
{
    template <typename GemmShape, typename>
    using type = ck_tile::StreamKTilePartitioner<GemmShape, ReductionStrategy, Persistent>;
};

template <typename PartitionerPolicy = SplitKPartitionerPolicy>
struct GroupedConvolutionBackwardWeightInvoker
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
              typename DsDataType     = ck_tile::tuple<>,
              typename DsLayout       = ck_tile::tuple<>,
              typename CDEElementWise = ck_tile::element_wise::PassThrough>
    static InvokerResult grouped_conv_bwd_weight(const ck_tile::GroupedConvBwdWeightHostArgs& args,
                                                 const ck_tile::stream_config& s)
    {
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

        using TilePartitioner =
            typename PartitionerPolicy::template type<GemmShape, GroupedConvTraitsType>;

        constexpr bool LargeTensors = false;

        using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
            GroupedConvTraitsType::FixedGemmParams::kPadM,
            GroupedConvTraitsType::FixedGemmParams::kPadN,
            GroupedConvTraitsType::FixedGemmParams::kPadK,
            ConvConfig::DoubleSmemBuffer,
            typename GroupedConvTraitsType::AsLayoutBwdWeight,
            typename GroupedConvTraitsType::BsLayoutBwdWeight,
            typename GroupedConvTraitsType::CLayoutBwdWeight,
            GroupedConvTraitsType::FixedGemmParams::TransposeC,
            GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
            GroupedConvTraitsType::FixedGemmParams::Persistent,
            ConvConfig::NumWaveGroups,
            GroupedConvTraitsType::FixedGemmParams::Preshuffle,
            GroupedConvTraitsType::FixedGemmParams::LDSVectorSize,
            ck_tile::DataCachePrefetchKind::None,
            ck_tile::DataCachePrefetchKind::None,
            false, /*Async*/
            LargeTensors>;
        constexpr auto scheduler = ConvConfig::Scheduler;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
            OutDataType,
            InDataType,
            AccDataType,
            GemmShape,
            GemmUniversalTraits,
            scheduler,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            WeiDataType,
            WeiDataType, // TODO: need to double check
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            GroupedConvTraitsType::VectorSizeA,
            GroupedConvTraitsType::VectorSizeB>;

        using GemmPipeline = typename PipelineTypeTraits<
            ConvConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
            OutDataType,
            InDataType,
            DsDataType,
            AccDataType,
            WeiDataType,
            typename GroupedConvTraitsType::ImplicitGemmDsLayout,
            typename GroupedConvTraitsType::FixedGemmParams::ELayout,
            CDEElementWise,
            TilePartitioner::MPerBlock,
            TilePartitioner::NPerBlock,
            ConvConfig::M_Warp,
            ConvConfig::N_Warp,
            ConvConfig::M_Warp_Tile,
            ConvConfig::N_Warp_Tile,
            ConvConfig::K_Warp_Tile,
            GroupedConvTraitsType::FixedGemmParams::TransposeC,
            ConvConfig::NumWaveGroups,
            GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
            GroupedConvTraitsType::VectorSizeC>;

        using ConvEpilogue =
            std::conditional_t<ConvConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V1 ||
                                   ConvConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_TDM_V2,
                               ck_tile::TdmEpilogue<EpilogueProblem>,
                               ck_tile::CShuffleEpilogue<EpilogueProblem>>;

        using Kernel = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType,
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

        // Workspace: may be non-zero for StreamK (depends on SK/DP tile split),
        // always zero for Split-K.
        auto ws_size = Kernel::GetWorkSpaceSize(kargs);
        ck_tile::DeviceMem workspace_dev(ws_size);
        Kernel::SetWorkSpacePointer(kargs, workspace_dev.GetDeviceBuffer());

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << UniversalGemmProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << '\n'
                      << "workspace: " << ws_size << " bytes" << '\n'
                      << "Vector size A: " << GemmPipeline::GetVectorSizeA()
                      << ", Vector size B: " << GemmPipeline::GetVectorSizeB()
                      << ", Vector size C: " << ConvEpilogue::GetVectorSizeC() << std::endl;
        }

        auto preprocess = [&]() {
            if constexpr(Kernel::IsStreamK)
            {
                // StreamK: zero workspace flags before each kernel launch
                if(ws_size > 0)
                {
                    ck_tile::hip_check_error(
                        hipMemsetAsync(workspace_dev.GetDeviceBuffer(), 0, ws_size, s.stream_id_));
                }
            }
            else if(kargs.k_batch > 1)
            {
                // Split-K: zero weight buffer for atomic accumulation
                ck_tile::hip_check_error(hipMemsetAsync(
                    kargs.wei_ptr, 0, args.template GetWeightByte<WeiDataType>(), s.stream_id_));
            }
        };

        float ave_time = ck_tile::launch_kernel_time_mask(
            s,
            preprocess,
            ck_tile::make_kernel<ConvConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

        return InvokerResult{ave_time, args.k_batch};
    }
};
