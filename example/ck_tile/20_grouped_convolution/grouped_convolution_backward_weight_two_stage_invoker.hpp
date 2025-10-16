// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "grouped_convolution_utils.hpp"

struct GroupedConvolutionBackwardWeightTwoStageInvoker
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
    static float grouped_conv_bwd_weight(const ck_tile::GroupedConvBwdWeightHostArgs& args,
                                         const ck_tile::stream_config& s)
    {
        using WorkspaceDataType = float;

        constexpr int kBlockPerCu = 1;

        // Implicit GEMM Traits
        using GemmShape = ck_tile::TileGemmShape<
            ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            ck_tile::
                sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
            GemmConfig::PermuteA,
            GemmConfig::PermuteB>;

        constexpr ck_tile::index_t VectorSizeA = 4;
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
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdWeight::AsLayout,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdWeight::BsLayout,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdWeight::CLayout,
            GemmConfig::TransposeC,
            GemmConfig::UseStructuredSparsity,
            false, // Persistent,
            GemmConfig::NumWaveGroups>;

        using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
            OutDataType,
            InDataType,
            AccDataType,
            GemmShape,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdWeight,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            WeiDataType,
            true,
            VectorSizeA,
            VectorSizeB>;

        using BaseGemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

        const ck_tile::index_t gemm_k =
            args.N_ * std::accumulate(args.output_spatial_lengths_.begin(),
                                      args.output_spatial_lengths_.end(),
                                      1,
                                      std::multiplies<ck_tile::index_t>());

        const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
        const ck_tile::index_t K_split     = (gemm_k + k_grain - 1) / k_grain * GemmConfig::K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        float ave_time{0};

        const auto Run = [&](const auto has_hot_loop_,
                             const auto tail_number_,
                             const auto memory_operation_) {
            constexpr bool has_hot_loop_v   = has_hot_loop_.value;
            constexpr auto tail_number_v    = tail_number_.value;
            constexpr auto scheduler        = GemmConfig::Scheduler;
            constexpr auto memory_operation = memory_operation_.value;

            using UniversalGemmProblem =
                ck_tile::UniversalGemmPipelineProblem<OutDataType,
                                                      InDataType,
                                                      AccDataType,
                                                      GemmShape,
                                                      GemmUniversalTraits,
                                                      scheduler,
                                                      has_hot_loop_v,
                                                      tail_number_v,
                                                      ck_tile::element_wise::PassThrough,
                                                      ck_tile::element_wise::PassThrough,
                                                      WeiDataType,
                                                      true,
                                                      VectorSizeA,
                                                      VectorSizeB>;

            using GemmPipeline = typename PipelineTypeTraits<
                GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

            using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                OutDataType, // A: Out
                InDataType,  // B: In
                DsDataType,
                AccDataType,
                WorkspaceDataType, // C: Workspace  normally Out
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
                GemmPipelineProblem::TransposeC,
                memory_operation,
                1,
                true,
                GroupedConvTraitsType::VectorSizeC>>;

            using Kernel = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType,
                                                                           TilePartitioner,
                                                                           GemmPipeline,
                                                                           ConvEpilogue>;

            const ck_tile::index_t spatial_lengths_accum =
                std::accumulate(args.filter_spatial_lengths_.begin(),
                                args.filter_spatial_lengths_.end(),
                                1,
                                std::multiplies<ck_tile::index_t>());
            ck_tile::DeviceMem ws_m_n_dev_buf(args.G_ * args.K_ * args.C_ * spatial_lengths_accum *
                                              sizeof(WorkspaceDataType));
            ck_tile::GroupedConvBwdWeightHostArgs ws_args =
                ck_tile::GroupedConvBwdWeightHostArgs(args);
            auto c_ptr      = ws_args.wei_ptr;
            ws_args.wei_ptr = ws_m_n_dev_buf.GetDeviceBuffer();
            auto kargs      = Kernel::MakeKernelArgs(ws_args);

            const dim3 grids  = Kernel::GridSize(kargs);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs))
            {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
            }

            using XElementwiseOperation = ck_tile::element_wise::UnaryConvert;
            using BlockTile             = ck_tile::sequence<2048>;
            using BlockWarps            = ck_tile::sequence<8>;
            using WarpTile              = ck_tile::sequence<64>;

            using ElementwiseShape =
                ck_tile::ElementWiseShape<BlockWarps, BlockTile, WarpTile, WorkspaceDataType>;
            using Problem = ck_tile::ElementWisePipelineProblem<WorkspaceDataType,
                                                                WorkspaceDataType,
                                                                WeiDataType,
                                                                ElementwiseShape,
                                                                XElementwiseOperation>;
            using ElementwiseKernel =
                ck_tile::ElementWiseKernel<Problem, ck_tile::ElementWiseDefaultPolicy>;

            ck_tile::index_t total_elements     = 1;
            std::vector<ck_tile::index_t> shape = {
                static_cast<ck_tile::index_t>(args.G_ * args.K_),
                static_cast<ck_tile::index_t>(args.C_ * spatial_lengths_accum)};

            for(auto d : shape)
                total_elements *= d;

            const ck_tile::index_t kBlockSize = ElementwiseKernel::BlockSize();

            constexpr ck_tile::index_t elements_per_block = BlockTile::at(ck_tile::number<0>{});
            ck_tile::index_t kGridSize =
                (total_elements + elements_per_block - 1) / elements_per_block;

            auto input_tensors =
                ck_tile::make_tuple(static_cast<WorkspaceDataType*>(ws_args.wei_ptr));
            auto input_size = ck_tile::make_tuple(shape[0], shape[1]);

            // Check if the kernel configuration is supported
            if(!ElementwiseKernel::IsSupportedArgument(input_size))
            {
                throw std::runtime_error(
                    "Wrong! Elementwise arguments not supported! Skipping gemm!\n");
            }

            if(s.log_level_ > 0)
            {
                std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                          << "shape: " << GemmShape::GetName() << '\n'
                          << "problem: " << GemmPipelineProblem::GetName() << '\n'
                          << "pipeline: " << GemmPipeline::GetName() << '\n'
                          << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << '\n'
                          << "Vector size A: " << GemmPipeline::GetVectorSizeA()
                          << ", Vector size B: " << GemmPipeline::GetVectorSizeB()
                          << ", Vector size C: " << ConvEpilogue::GetVectorSizeC() << std::endl;
            }

            auto preprocess = [&]() {
                if(args.k_batch > 1)
                    ck_tile::hip_check_error(
                        hipMemsetAsync(ws_args.wei_ptr,
                                       0,
                                       shape[0] * shape[1] * sizeof(WorkspaceDataType),
                                       s.stream_id_));
            };

            ave_time = ck_tile::launch_kernel_time_mask(
                s,
                preprocess,
                ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs),
                ck_tile::make_kernel<kBlockPerCu>(ElementwiseKernel{},
                                                  kGridSize,
                                                  kBlockSize,
                                                  0,
                                                  input_size,
                                                  ck_tile::make_tuple(shape[1], 1), // Input Stride
                                                  ck_tile::make_tuple(shape[1], 1), // Output Stride
                                                  input_tensors,
                                                  static_cast<WeiDataType*>(c_ptr)));

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
