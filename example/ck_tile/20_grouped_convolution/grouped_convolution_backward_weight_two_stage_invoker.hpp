// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "grouped_convolution_utils.hpp"

struct GroupedConvolutionBackwardWeightTwoStageInvoker
{
    template <ck_tile::index_t NDimSpatial,
              typename GemmWarpConfig,
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

        constexpr ck_tile::index_t M_Tile = 64;
        constexpr ck_tile::index_t N_Tile = 64;
        constexpr ck_tile::index_t K_Tile = 64;

        constexpr ck_tile::index_t M_Warp = 2;
        constexpr ck_tile::index_t N_Warp = 2;
        constexpr ck_tile::index_t K_Warp = 1;

        constexpr ck_tile::index_t M_Warp_Tile = GemmWarpConfig::M_Warp_Tile;
        constexpr ck_tile::index_t N_Warp_Tile = GemmWarpConfig::N_Warp_Tile;
        constexpr ck_tile::index_t K_Warp_Tile = GemmWarpConfig::K_Warp_Tile;

        constexpr ck_tile::index_t VectorSizeA = 1;
        constexpr ck_tile::index_t VectorSizeB = 1;
        constexpr ck_tile::index_t VectorSizeC = 1;

        // Implicit GEMM Traits
        using CodegenShape =
            ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                   ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                   ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

        constexpr auto ConvSpec      = ck_tile::ConvolutionSpecialization::Default;
        using TilePartitioner        = ck_tile::GemmTile1DPartitioner<CodegenShape>;
        using GroupedConvTraitsType  = ck_tile::GroupedConvTraits<NDimSpatial,
                                                                  ConvSpec,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  DsLayout,
                                                                  OutLayout,
                                                                  VectorSizeA,
                                                                  VectorSizeB,
                                                                  VectorSizeC>;
        using CodegenPipelineProblem = ck_tile::GemmPipelineProblem<
            OutDataType, // A: Out
            InDataType,  // B: In
            AccDataType,
            CodegenShape,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsBwdWeight,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            InDataType,
            true,
            GroupedConvTraitsType::VectorSizeA,
            GroupedConvTraitsType::VectorSizeB>;
        using CodegenPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

        const auto Run = [&](const auto memory_operation_) {
            constexpr auto memory_operation = memory_operation_.value;

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
                M_Warp,
                N_Warp,
                M_Warp_Tile,
                N_Warp_Tile,
                K_Warp_Tile,
                CodegenPipelineProblem::TransposeC,
                memory_operation,
                1,
                true,
                GroupedConvTraitsType::VectorSizeC>>;

            using Kernel = ck_tile::GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType,
                                                                           TilePartitioner,
                                                                           CodegenPipeline,
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
                          << "shape: " << CodegenShape::GetName() << '\n'
                          << "problem: " << CodegenPipelineProblem::GetName() << '\n'
                          << "pipeline: " << CodegenPipeline::GetName() << '\n'
                          << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << '\n'
                          << "Vector size A: " << CodegenPipeline::GetVectorSizeA()
                          << ", Vector size B: " << CodegenPipeline::GetVectorSizeB()
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

            return ck_tile::launch_kernel_time_mask(
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
        };

        if(args.k_batch == 1)
        {
            return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                  ck_tile::memory_operation_enum::set>{});
        }
        else
        {
            return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                  ck_tile::memory_operation_enum::atomic_add>{});
        }
    }
};
