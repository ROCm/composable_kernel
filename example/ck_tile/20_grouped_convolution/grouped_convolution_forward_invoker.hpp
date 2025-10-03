// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "grouped_convolution_utils.hpp"
#include <queue>
#include <vector>

struct GroupedConvolutionForwardInvoker
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
    static float grouped_conv_fwd(const ck_tile::GroupedConvFwdHostArgs& args,
                                  const ck_tile::stream_config& s)
    {
        if(s.log_level_ > 0) {
            std::cout << "[INVOKER] grouped_conv_fwd called, NDimSpatial=" << NDimSpatial << "\n";
        }
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

        constexpr ck_tile::index_t VectorSizeA = 8;
        constexpr ck_tile::index_t VectorSizeB = 8;
        constexpr ck_tile::index_t VectorSizeC = 8;

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
            InDataType,
            WeiDataType,
            AccDataType,
            CodegenShape,
            typename GroupedConvTraitsType::GroupedConvImplicitGemmTraitsFwd,
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

            using Kernel = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                                    TilePartitioner,
                                                                    CodegenPipeline,
                                                                    ConvEpilogue>;

            float ave_time = 0.0f;

            // Create kargs and check if split-image is needed
            if(s.log_level_ > 0) {
                std::cout << "[INVOKER] Creating kargs with N=" << args.N_ << std::endl;
            }
            auto kargs = Kernel::MakeKernelArgs(args);

            if(s.log_level_ > 0) {
                std::cout << "[INVOKER] kargs: n_per_split=" << kargs.n_per_split
                          << ", n_splits=" << kargs.n_splits
                          << ", original_n=" << kargs.original_n << std::endl;
            }

            // Check if split-image is needed (uses unified threshold internally)
            auto split_info = kargs.GetSplitImageInfo();

            if(!split_info.should_split) {
                // No split-image needed - use kargs directly (may have Split-N)
                if(s.log_level_ > 0) {
                    std::cout << "[INVOKER] No split-image needed - launching with kargs" << std::endl;
                }
                const dim3 grids = Kernel::GridSize(kargs);
                const dim3 blocks = Kernel::BlockSize();

                if(!Kernel::IsSupportedArgument(kargs)) {
                    throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
                }

                ave_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
                return ave_time;
            }

            // RECURSIVE split-image path
            {
                if(s.log_level_ > 0) {
                    std::cout << "[RECURSIVE SPLIT] Starting recursive split-image" << std::endl;
                }

                const int split_dim = 0;  // Always split first spatial dimension (W/H/D)
                const int MAX_DEPTH = 10;  // Max recursion depth (2^10 = 1024 pieces max)

                // Define SplitPiece to track each piece with cumulative offsets and depth
                struct SplitPiece {
                    ck_tile::GroupedConvFwdHostArgs args;
                    ck_tile::long_index_t input_offset;   // Cumulative offset from original base
                    ck_tile::long_index_t output_offset;  // Cumulative offset from original base
                    int depth;                             // Recursion depth level

                    // Constructor to initialize from args
                    SplitPiece(const ck_tile::GroupedConvFwdHostArgs& a,
                               ck_tile::long_index_t in_off,
                               ck_tile::long_index_t out_off,
                               int d)
                        : args(a), input_offset(in_off), output_offset(out_off), depth(d) {}
                };

                std::queue<SplitPiece> split_queue;
                std::vector<SplitPiece> ready_list;

                // Start with original problem (offset = 0, depth = 0)
                auto initial_args = args;
                initial_args.N_ = kargs.n_per_split;  // Use split-N result
                split_queue.emplace(initial_args, 0, 0, 0);

                int level = 0;
                if(s.log_level_ > 0) {
                    std::cout << "[RECURSIVE SPLIT] Initial piece: N=" << initial_args.N_
                              << ", offset_in=0, offset_out=0, depth=0" << std::endl;
                }

                // BFS-style recursive splitting
                while(!split_queue.empty()) {
                    SplitPiece current = split_queue.front();
                    split_queue.pop();

                    // Create kargs for this piece and check if it needs splitting
                    auto piece_kargs = Kernel::MakeKernelArgs(current.args);
                    auto piece_split_info = piece_kargs.GetSplitImageInfo();

                    if(s.log_level_ > 0) {
                        std::cout << "[LEVEL " << level << "] Checking piece: ";
                        if constexpr (NDimSpatial == 1) {
                            std::cout << "Wo=" << current.args.output_spatial_lengths_[0];
                        } else if constexpr (NDimSpatial == 2) {
                            std::cout << "Ho=" << current.args.output_spatial_lengths_[0]
                                      << ", Wo=" << current.args.output_spatial_lengths_[1];
                        } else if constexpr (NDimSpatial == 3) {
                            std::cout << "Do=" << current.args.output_spatial_lengths_[0]
                                      << ", Ho=" << current.args.output_spatial_lengths_[1]
                                      << ", Wo=" << current.args.output_spatial_lengths_[2];
                        }
                        std::cout << ", offset_in=" << current.input_offset
                                  << ", offset_out=" << current.output_offset
                                  << ", depth=" << current.depth << std::endl;
                    }

                    // Check if we should stop splitting: either small enough OR max depth reached
                    if(!piece_split_info.should_split || current.depth >= MAX_DEPTH) {
                        // This piece is ready to launch
                        ready_list.push_back(current);
                        if(s.log_level_ > 0) {
                            if(!piece_split_info.should_split) {
                                std::cout << "  -> Ready to launch (below threshold)" << std::endl;
                            } else {
                                std::cout << "  -> Ready to launch (max depth " << MAX_DEPTH << " reached)" << std::endl;
                            }
                        }
                    } else {
                        // This piece needs to be split into LEFT and RIGHT
                        if(s.log_level_ > 0) {
                            std::cout << "  -> SPLIT! Left=" << piece_split_info.out_left
                                      << ", Right=" << piece_split_info.out_right << std::endl;
                        }

                        // Create LEFT piece (inherits parent's offset)
                        auto left_args = current.args;
                        left_args.input_spatial_lengths_[split_dim] = piece_split_info.in_left;
                        left_args.output_spatial_lengths_[split_dim] = piece_split_info.out_left;
                        left_args.input_left_pads_[split_dim] = piece_split_info.left_pad_left;
                        left_args.input_right_pads_[split_dim] = piece_split_info.right_pad_left;

                        // LEFT inherits parent's cumulative offset (no change)
                        auto left_input_offset = current.input_offset;
                        auto left_output_offset = current.output_offset;

                        if(s.log_level_ > 0) {
                            std::cout << "    LEFT: offset_in=" << left_input_offset
                                      << " (parent), offset_out=" << left_output_offset
                                      << " (parent)" << std::endl;
                        }

                        // Create RIGHT piece (parent offset + local offset)
                        auto right_args = current.args;
                        right_args.input_spatial_lengths_[split_dim] = piece_split_info.in_right;
                        right_args.output_spatial_lengths_[split_dim] = piece_split_info.out_right;
                        right_args.input_left_pads_[split_dim] = piece_split_info.left_pad_right;
                        right_args.input_right_pads_[split_dim] = piece_split_info.right_pad_right;

                        // CRITICAL: RIGHT accumulates offset (parent + local)
                        auto right_input_offset = current.input_offset + piece_split_info.input_offset;
                        auto right_output_offset = current.output_offset + piece_split_info.output_offset;

                        if(s.log_level_ > 0) {
                            std::cout << "    RIGHT: local_offset_in=" << piece_split_info.input_offset
                                      << ", local_offset_out=" << piece_split_info.output_offset << std::endl;
                            std::cout << "    RIGHT: cumulative_offset_in=" << right_input_offset
                                      << " (" << current.input_offset << "+" << piece_split_info.input_offset << ")"
                                      << ", cumulative_offset_out=" << right_output_offset
                                      << " (" << current.output_offset << "+" << piece_split_info.output_offset << ")"
                                      << std::endl;
                        }

                        // Push LEFT and RIGHT back to queue with incremented depth
                        split_queue.emplace(left_args, left_input_offset, left_output_offset, current.depth + 1);
                        split_queue.emplace(right_args, right_input_offset, right_output_offset, current.depth + 1);
                    }

                    level++;
                }

                if(s.log_level_ > 0) {
                    std::cout << "[RECURSIVE SPLIT] Split complete! Total pieces: " << ready_list.size() << std::endl;
                }

                // Launch all pieces from ready_list
                ave_time = 0.0f;
                for(size_t i = 0; i < ready_list.size(); i++) {
                    const auto& piece = ready_list[i];

                    if(s.log_level_ > 0) {
                        std::cout << "[LAUNCH " << (i+1) << "/" << ready_list.size() << "] ";
                        if constexpr (NDimSpatial == 1) {
                            std::cout << "Wo=" << piece.args.output_spatial_lengths_[0];
                        } else if constexpr (NDimSpatial == 2) {
                            std::cout << "Ho=" << piece.args.output_spatial_lengths_[0]
                                      << ", Wo=" << piece.args.output_spatial_lengths_[1];
                        } else if constexpr (NDimSpatial == 3) {
                            std::cout << "Do=" << piece.args.output_spatial_lengths_[0]
                                      << ", Ho=" << piece.args.output_spatial_lengths_[1]
                                      << ", Wo=" << piece.args.output_spatial_lengths_[2];
                        }
                        std::cout << ", offset_in=" << piece.input_offset
                                  << ", offset_out=" << piece.output_offset << std::endl;
                    }

                    // Create kargs for this piece
                    auto piece_kargs = Kernel::MakeKernelArgs(piece.args);

                    // Copy Split-N metadata from original kargs
                    piece_kargs.n_splits = kargs.n_splits;
                    piece_kargs.original_n = kargs.original_n;

                    // Use batch_stride from ORIGINAL kargs (not split piece's)
                    piece_kargs.input_batch_stride = kargs.input_batch_stride;
                    piece_kargs.output_batch_stride = kargs.output_batch_stride;

                    // Store cumulative spatial offset (applied per-batch in kernel)
                    piece_kargs.spatial_offset_in = piece.input_offset;
                    piece_kargs.spatial_offset_out = piece.output_offset;

                    const dim3 grids = Kernel::GridSize(piece_kargs);
                    const dim3 blocks = Kernel::BlockSize();

                    if(!Kernel::IsSupportedArgument(piece_kargs)) {
                        throw std::runtime_error("Wrong! Split piece arguments not supported!\n");
                    }

                    float piece_time = ck_tile::launch_kernel(
                        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, piece_kargs));

                    ave_time += piece_time;

                    if(s.log_level_ > 0) {
                        std::cout << "  Time: " << piece_time << "ms" << std::endl;
                    }
                }

                if(s.log_level_ > 0) {
                    std::cout << "[RECURSIVE SPLIT] Complete! Total time: " << ave_time << "ms" << std::endl;
                }
            }

            return ave_time;
        };

        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::set>{});
    }
};
