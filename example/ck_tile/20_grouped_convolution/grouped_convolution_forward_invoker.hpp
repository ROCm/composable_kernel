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

            // using Kernel = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
            //                                                         TilePartitioner,
            //                                                         CodegenPipeline,
            //                                                         ConvEpilogue>;

            // float ave_time = 0.0f;

            // // Create kargs and check if split-image is needed
            // auto kargs = Kernel::MakeKernelArgs(args);

            // // Check if split-image is needed (uses unified threshold internally)
            // auto split_info = kargs.GetSplitImageInfo();

            // if(!split_info.should_split)
            // {
            //     // No split-image needed - use kargs directly (may have Split-N)
            //     if(s.log_level_ > 0)
            //     {
            //         std::cout << "[INVOKER] No split-image needed - launching with kargs"
            //                   << std::endl;
            //     }
            //     const dim3 grids  = Kernel::GridSize(kargs);
            //     const dim3 blocks = Kernel::BlockSize();

            //     if(!Kernel::IsSupportedArgument(kargs))
            //     {
            //         throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
            //     }

            //     ave_time = ck_tile::launch_kernel(
            //         s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
            //     return ave_time;
            // }

            // // RECURSIVE split-image path - delegate to transformer helper
            // ave_time = decltype(kargs.transformer_)::template LaunchWithRecursiveSplit<Kernel,
            //                                                                            kBlockPerCu>(
            //     args, s, kargs);

            // return ave_time;

        // =====================================================================
        // Split-K parameters
        // =====================================================================
        const ck_tile::index_t k_grain     = args.k_batch * GemmConfig::K_Tile;
        const ck_tile::index_t K_split     = (gemm_k + k_grain - 1) / k_grain * GemmConfig::K_Tile;
        const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
        const bool has_hot_loop            = BaseGemmPipeline::BlockHasHotloop(num_loop);
        const ck_tile::TailNumber tail_num = BaseGemmPipeline::GetBlockLoopTailNum(num_loop);
        float ave_time{0};

        // =====================================================================
        // Split-Image: Calculate number of pieces (temporary dynamic)
        // =====================================================================
        // TEMPORARY FIX: Choose split factor based on dimensions
        const ck_tile::index_t total_h = (NDimSpatial >= 2) ? args.output_spatial_lengths_[NDimSpatial - 2] : 1;
        const ck_tile::index_t total_w = args.output_spatial_lengths_[NDimSpatial - 1];

        // Temporary logic: choose split factor that aligns with MPerBlock=16
        ck_tile::index_t SPLIT_FACTOR = 4;  // Default
        if(NDimSpatial == 2) {
            // Check common cases for alignment
            if(total_h == 80 && total_w == 80) {
                SPLIT_FACTOR = 5;  // 80/5 = 16 (aligns with MPerBlock)
            } else if(total_h == 96 && total_w == 96) {
                SPLIT_FACTOR = 6;  // 96/6 = 16 (aligns with MPerBlock)
            } else if(total_h == 64 && total_w == 64) {
                SPLIT_FACTOR = 4;  // 64/4 = 16 (aligns with MPerBlock)
            } else if(total_h == 128 && total_w == 128) {
                SPLIT_FACTOR = 4;  // 128/4 = 32 (aligns with MPerBlock)
            }
        }

        const ck_tile::index_t num_w_pieces = SPLIT_FACTOR;
        const ck_tile::index_t num_h_pieces = (NDimSpatial >= 2) ? SPLIT_FACTOR : 1;
        const ck_tile::index_t num_d_pieces = (NDimSpatial == 3) ? SPLIT_FACTOR : 1;
        const ck_tile::index_t total_pieces = num_d_pieces * num_h_pieces * num_w_pieces;

        // Temporarily enable split-image to test piece creation logic
        constexpr bool enable_split_image = true;

        if(s.log_level_ > 0)
        {
            std::cout << "[INVOKER] Split-image: Using SPLIT_FACTOR=" << SPLIT_FACTOR
                      << " for " << total_h << "×" << total_w << "\n";
            std::cout << "[INVOKER] Split-image calculation: "
                      << "D=" << num_d_pieces << " × H=" << num_h_pieces
                      << " × W=" << num_w_pieces << " = " << total_pieces << " pieces\n";
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
        if constexpr (!enable_split_image)
        {
            // ─────────────────────────────────────────────────────────────────
            // Path 1: NO Split-Image (current path - always taken for now)
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

            // Calculate piece sizes for each dimension (reuse total_h, total_w from above)
            const ck_tile::index_t total_d = (NDimSpatial == 3) ? args.output_spatial_lengths_[0] : 1;

            const ck_tile::index_t piece_d = total_d / num_d_pieces;
            const ck_tile::index_t piece_h = total_h / num_h_pieces;
            const ck_tile::index_t piece_w = total_w / num_w_pieces;

            if(s.log_level_ > 0)
            {
                std::cout << "[SPLIT-IMAGE] Total: D=" << total_d << " H=" << total_h << " W=" << total_w << "\n"
                          << "[SPLIT-IMAGE] Piece: D=" << piece_d << " H=" << piece_h << " W=" << piece_w << "\n";
            }

            // Store piece descriptors temporarily (will populate in final kargs)
            struct TempPieceInfo {
                ck_tile::index_t block_start;
                ck_tile::index_t block_end;
            };
            std::array<TempPieceInfo, 64> temp_pieces{};
            ck_tile::index_t total_blocks = 0;

            // Calculate piece info for all pieces
            for(ck_tile::index_t piece = 0; piece < total_pieces; piece++)
            {
                // Calculate piece indices (d_idx, h_idx, w_idx)
                ck_tile::index_t w_idx = piece % num_w_pieces;
                ck_tile::index_t h_idx = (piece / num_w_pieces) % num_h_pieces;
                ck_tile::index_t d_idx = piece / (num_w_pieces * num_h_pieces);

                // Calculate spatial starting positions for this piece
                ck_tile::index_t w_start = w_idx * piece_w;
                ck_tile::index_t h_start = h_idx * piece_h;
                ck_tile::index_t d_start = d_idx * piece_d;

                // Calculate piece GEMM dimensions
                ck_tile::index_t piece_gemm_m = args.N_ * piece_d * piece_h * piece_w;
                ck_tile::index_t piece_gemm_n = args.K_;

                // Calculate grid size for this piece
                ck_tile::index_t piece_grid = ((piece_gemm_m + TilePartitioner::MPerBlock - 1) / TilePartitioner::MPerBlock) *
                                              ((piece_gemm_n + TilePartitioner::NPerBlock - 1) / TilePartitioner::NPerBlock);

                // Store piece info (only unique data)
                temp_pieces[piece].block_start = total_blocks;
                temp_pieces[piece].block_end = total_blocks + piece_grid;

                total_blocks += piece_grid;

                if(s.log_level_ > 0 && piece < 4)
                {
                    std::cout << "[SPLIT-IMAGE] Piece " << piece
                              << " (d=" << d_idx << ",h=" << h_idx << ",w=" << w_idx << ")"
                              << " starts at (d=" << d_start << ",h=" << h_start << ",w=" << w_start << ")"
                              << ": blocks [" << temp_pieces[piece].block_start
                              << "," << temp_pieces[piece].block_end << ")\n";
                }
            }

            if(s.log_level_ > 0)
            {
                std::cout << "[SPLIT-IMAGE] Total blocks: " << total_blocks << "\n";
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

                    // Create kargs
                    auto kargs = Kernel::MakeKernelArgs(args);

                    // Populate split-image info (common data stored once)
                    kargs.num_spatial_pieces = total_pieces;
                    kargs.split_image.piece_d = piece_d;
                    kargs.split_image.piece_h = piece_h;
                    kargs.split_image.piece_w = piece_w;
                    kargs.split_image.total_d = total_d;
                    kargs.split_image.total_h = total_h;
                    kargs.split_image.total_w = total_w;
                    kargs.split_image.num_d_pieces = num_d_pieces;
                    kargs.split_image.num_h_pieces = num_h_pieces;
                    kargs.split_image.num_w_pieces = num_w_pieces;

                    // Populate per-piece data (only unique values)
                    for(ck_tile::index_t i = 0; i < total_pieces; i++)
                    {
                        // Calculate piece indices (d_idx, h_idx, w_idx)
                        ck_tile::index_t w_idx = i % num_w_pieces;
                        ck_tile::index_t h_idx = (i / num_w_pieces) % num_h_pieces;
                        ck_tile::index_t d_idx = i / (num_w_pieces * num_h_pieces);

                        // Calculate spatial starting positions for this piece
                        ck_tile::index_t w_start = w_idx * piece_w;
                        ck_tile::index_t h_start = h_idx * piece_h;
                        ck_tile::index_t d_start = d_idx * piece_d;

                        // Store only unique per-piece data
                        kargs.split_image.pieces[i].block_start = temp_pieces[i].block_start;
                        kargs.split_image.pieces[i].block_end = temp_pieces[i].block_end;
                        kargs.split_image.pieces[i].d_start = d_start;
                        kargs.split_image.pieces[i].h_start = h_start;
                        kargs.split_image.pieces[i].w_start = w_start;
                    }

                    // Calculate grid with total_blocks for ALL pieces
                    const dim3 grids(total_blocks, kargs.GemmBatch, kargs.n_splits);
                    const dim3 blocks = Kernel::BlockSize();

                    if(!Kernel::IsSupportedArgument(kargs))
                    {
                        throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
                    }

                    if(s.log_level_ > 0)
                    {
                        std::cout << "[SPLIT-IMAGE] Launching kernel with args: " << Kernel::GetName() << '\n'
                                  << "  shape: " << GemmShape::GetName() << '\n'
                                  << "  problem: " << UniversalGemmProblem::GetName() << '\n'
                                  << "  pipeline: " << GemmPipeline::GetName() << '\n'
                                  << "  grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                                  << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}" << '\n'
                                  << "  num_spatial_pieces: " << kargs.num_spatial_pieces << std::endl;
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
