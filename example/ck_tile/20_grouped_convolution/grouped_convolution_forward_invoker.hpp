// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "grouped_convolution_utils.hpp"

struct GroupedConvolutionForwardInvoker
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
    static float grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs& args,
                                  const ck_tile::stream_config& s)
    {
        if(s.log_level_ > 0)
        {
            std::cout << "[INVOKER] grouped_conv_fwd called, NDimSpatial=" << NDimSpatial << "\n";
        }
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
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsFwd::AsLayout,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsFwd::BsLayout,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsFwd::CLayout,
            GemmConfig::TransposeC,
            GemmConfig::UseStructuredSparsity,
            false, // Persistent,
            GemmConfig::NumWaveGroups,
            GemmConfig::Preshuffle>;

        using GemmPipelineProblem = ck_tile::GemmPipelineProblem<
            InDataType,
            WeiDataType,
            AccDataType,
            GemmShape,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsFwd,
            ck_tile::element_wise::PassThrough,
            ck_tile::element_wise::PassThrough,
            OutDataType,
            true,
            VectorSizeA,
            VectorSizeB>;

        using BaseGemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template UniversalGemmPipeline<GemmPipelineProblem>;

        const ck_tile::index_t gemm_k =
            args.C_ * std::accumulate(args.filter_spatial_lengths_.begin(),
                                      args.filter_spatial_lengths_.end(),
                                      1,
                                      std::multiplies<ck_tile::index_t>());

        // Split-K parameters
        const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
        const ck_tile::index_t K_split     = (gemm_k + k_grain - 1) / k_grain * GemmConfig::K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        float ave_time{0};

        // Split-Image: Use transform helper to calculate split factors
        // Extract output spatial dimensions
        const ck_tile::index_t total_d =
            (NDimSpatial == 3) ? args.output_spatial_lengths_[NDimSpatial - 3] : 1;
        const ck_tile::index_t total_h =
            (NDimSpatial >= 2) ? args.output_spatial_lengths_[NDimSpatial - 2] : 1;
        const ck_tile::index_t total_w = args.output_spatial_lengths_[NDimSpatial - 1];

        // Use transform helper to calculate split-image info
        // This considers both split-N threshold and optimal spatial splitting
        using TransformType =
            ck_tile::TransformConvFwdToGemm<NDimSpatial,
                                            ck_tile::ConvolutionSpecialization::Default,
                                            VectorSizeA,
                                            VectorSizeB,
                                            VectorSizeC,
                                            false, // SplitN handled separately
                                            InDataType,
                                            OutDataType>;

        // Check if layout is supported for split-image
        // Split-image requires specific memory layouts:
        // 1D: NWGC (input), GKXC (weight), NWGK (output)
        // 2D: NHWGC (input), GKYXC (weight), NHWGK (output)
        // 3D: NDHWGC (input), GKZYXC (weight), NDHWGK (output)
        constexpr bool is_supported_layout =
            std::is_same<InLayout, ck_tile::tensor_layout::convolution::NWGC>::value ||
            std::is_same<InLayout, ck_tile::tensor_layout::convolution::NHWGC>::value ||
            std::is_same<InLayout, ck_tile::tensor_layout::convolution::NDHWGC>::value;

        auto split_info = TransformType::GetSplitImageInfo(
            args.G_, args.N_, args.C_, args.K_, total_d, total_h, total_w);

        const ck_tile::index_t num_d_pieces = split_info.num_d_pieces;
        const ck_tile::index_t num_h_pieces = split_info.num_h_pieces;
        const ck_tile::index_t num_w_pieces = split_info.num_w_pieces;
        const ck_tile::index_t total_pieces = num_d_pieces * num_h_pieces * num_w_pieces;

        // Enable split-image only when:
        // 1. GetSplitImageInfo says we should split (memory-based decision)
        // 2. Layout is supported
        const bool enable_split_image = split_info.should_split && is_supported_layout;

        if(split_info.should_split && !is_supported_layout && s.log_level_ > 0)
        {
            std::cout << "[INVOKER] WARNING: Split-image requested but layout not supported. "
                      << "Falling back to non-split mode.\n";
        }

        if(s.log_level_ > 0)
        {
            std::cout << "[INVOKER] Split-image: Independent split factors per dimension\n";
            if(NDimSpatial == 3)
            {
                std::cout << "[INVOKER] Dimensions: D=" << total_d << " H=" << total_h
                          << " W=" << total_w << "\n";
                std::cout << "[INVOKER] Pieces: D=" << num_d_pieces << " × H=" << num_h_pieces
                          << " × W=" << num_w_pieces << " = " << total_pieces << " total pieces\n";
            }
            else if(NDimSpatial == 2)
            {
                std::cout << "[INVOKER] Dimensions: H=" << total_h << " W=" << total_w << "\n";
                std::cout << "[INVOKER] Pieces: H=" << num_h_pieces << " × W=" << num_w_pieces
                          << " = " << total_pieces << " total pieces\n";
            }
            else
            {
                std::cout << "[INVOKER] Dimensions: W=" << total_w << "\n";
                std::cout << "[INVOKER] Pieces: W=" << num_w_pieces << " = " << total_pieces
                          << " total pieces\n";
            }
        }

        // =====================================================================
        // Kernel launch lambda
        // =====================================================================
        const auto Run =
            [&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
                constexpr bool has_hot_loop_v   = has_hot_loop_.value;
                constexpr auto tail_number_v    = tail_number_.value;
                constexpr auto scheduler        = GemmConfig::Scheduler;
                constexpr auto memory_operation = memory_operation_.value;

                using UniversalGemmProblem =
                    ck_tile::UniversalGemmPipelineProblem<InDataType,
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
                                                          true,
                                                          VectorSizeA,
                                                          VectorSizeB>;

                using GemmPipeline = typename PipelineTypeTraits<
                    GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

                using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                    InDataType,
                    WeiDataType,
                    DsDataType,
                    AccDataType,
                    OutDataType,
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

                ave_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

                return ave_time;
            };

        // =====================================================================
        // Split-K lambda
        // =====================================================================
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

        // =====================================================================
        // Split-Image dispatch
        // =====================================================================
        if(!enable_split_image)
        {
            // ─────────────────────────────────────────────────────────────────
            // Path 1: NO Split-Image (when spatial dimensions fit in memory)
            // ─────────────────────────────────────────────────────────────────
            // May have: Split-N (grid.z > 1), Split-K (k_batch > 1)
            BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
        }
        else
        {
            // ─────────────────────────────────────────────────────────────────
            // Path 2: Split-Image (SINGLE kernel launch with all pieces)
            // ─────────────────────────────────────────────────────────────────

            if(s.log_level_ > 0)
            {
                std::cout << "[INVOKER] Split-Image: Creating " << total_pieces << " pieces\n";
            }

            // Base piece size (non-overlapping division)
            // Note: total_d, total_h, total_w already declared above
            const ck_tile::index_t base_piece_d = total_d / num_d_pieces;
            const ck_tile::index_t base_piece_h = total_h / num_h_pieces;
            const ck_tile::index_t base_piece_w = total_w / num_w_pieces;

            // Store piece descriptors temporarily (will populate in final kargs)
            std::array<ck_tile::SplitImagePieceInfo, 64> temp_pieces{};
            ck_tile::index_t total_blocks = 0;

            // Calculate piece info for all pieces using library utility function
            for(ck_tile::index_t piece = 0; piece < total_pieces; piece++)
            {
                temp_pieces[piece] =
                    ck_tile::calculate_spatial_piece<TilePartitioner>(piece,
                                                                      num_d_pieces,
                                                                      num_h_pieces,
                                                                      num_w_pieces,
                                                                      base_piece_d,
                                                                      base_piece_h,
                                                                      base_piece_w,
                                                                      total_d,
                                                                      total_h,
                                                                      total_w,
                                                                      args.N_,
                                                                      args.K_,
                                                                      total_blocks);
                total_blocks = temp_pieces[piece].block_end;
            }

            // ─────────────────────────────────────────────────────────────────
            // Split-Image kernel launch lambda (follows TailHandler pattern)
            // ─────────────────────────────────────────────────────────────────
            const auto RunSplitImage = [&](const auto has_hot_loop_, const auto tail_number_) {
                const auto LaunchSplitImageKernel = [&](const auto memory_operation_) {
                    constexpr bool has_hot_loop_v   = has_hot_loop_.value;
                    constexpr auto tail_number_v    = tail_number_.value;
                    constexpr auto scheduler        = GemmConfig::Scheduler;
                    constexpr auto memory_operation = memory_operation_.value;

                    using UniversalGemmProblem =
                        ck_tile::UniversalGemmPipelineProblem<InDataType,
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
                                                              true,
                                                              VectorSizeA,
                                                              VectorSizeB>;

                    using GemmPipeline = typename PipelineTypeTraits<
                        GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

                    using ConvEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
                        InDataType,
                        WeiDataType,
                        DsDataType,
                        AccDataType,
                        OutDataType,
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

                    using Kernel = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                                            TilePartitioner,
                                                                            GemmPipeline,
                                                                            ConvEpilogue>;

                    // Create kargs and populate split-image info
                    auto kargs = Kernel::MakeKernelArgs(args);

                    // Helper: Populate split-image metadata
                    auto populate_split_image_kargs = [&]() {
                        kargs.num_spatial_pieces       = total_pieces;
                        kargs.split_image.total_d      = total_d;
                        kargs.split_image.total_h      = total_h;
                        kargs.split_image.total_w      = total_w;
                        kargs.split_image.num_d_pieces = num_d_pieces;
                        kargs.split_image.num_h_pieces = num_h_pieces;
                        kargs.split_image.num_w_pieces = num_w_pieces;

                        for(ck_tile::index_t i = 0; i < total_pieces; i++)
                        {
                            kargs.split_image.pieces[i] = {temp_pieces[i].block_start,
                                                           temp_pieces[i].block_end,
                                                           temp_pieces[i].d_start,
                                                           temp_pieces[i].h_start,
                                                           temp_pieces[i].w_start,
                                                           temp_pieces[i].d_size,
                                                           temp_pieces[i].h_size,
                                                           temp_pieces[i].w_size};
                        }
                    };

                    populate_split_image_kargs();

                    // Calculate grid with total_blocks for ALL pieces
                    const dim3 grids(total_blocks, kargs.GemmBatch, kargs.n_splits);
                    const dim3 blocks = Kernel::BlockSize();

                    if(!Kernel::IsSupportedArgument(kargs))
                    {
                        throw std::runtime_error(
                            "Wrong! Arguments not supported! Skipping conv!\n");
                    }

                    ave_time = ck_tile::launch_kernel(
                        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));

                    return ave_time;
                };

                // Dispatch based on k_batch (same as RunSplitk)
                if(args.k_batch == 1)
                {
                    LaunchSplitImageKernel(MemoryOpSet{});
                }
                else
                {
                    LaunchSplitImageKernel(MemoryOpAtomicAdd{});
                }
            };

            // Use TailHandler to dispatch correct template instantiation
            BaseGemmPipeline::TailHandler(RunSplitImage, has_hot_loop, tail_num);
        }

        return ave_time;
    }
};
