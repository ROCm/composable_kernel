// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "grouped_convolution_utils.hpp"

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

            // ═══════════════════════════════════════════════════════════
            // Create kargs and check if split-image is needed
            // ═══════════════════════════════════════════════════════════
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

            // ═══════════════════════════════════════════════════════════
            // Split-image path - create LEFT and RIGHT splits
            // ═══════════════════════════════════════════════════════════
            {
                if(s.log_level_ > 0) {
                    std::cout << "[INVOKER] Split-image needed! Creating left and right splits" << std::endl;
                    std::cout << "[INVOKER] Out_left=" << split_info.out_left
                              << ", Out_right=" << split_info.out_right << std::endl;
                    std::cout << "[INVOKER] LEFT: In=" << split_info.in_left
                              << ", Out=" << split_info.out_left << std::endl;
                    std::cout << "[INVOKER] RIGHT: In=" << split_info.in_right
                              << ", Out=" << split_info.out_right << std::endl;
                }

                const int split_dim = 0;  // Always split first spatial dimension (W/H/D)

                // Create LEFT descriptor
                auto left_args = args;
                left_args.input_spatial_lengths_[split_dim] = split_info.in_left;
                left_args.output_spatial_lengths_[split_dim] = split_info.out_left;
                left_args.input_left_pads_[split_dim] = split_info.left_pad_left;
                left_args.input_right_pads_[split_dim] = split_info.right_pad_left;

                // CRITICAL FIX: Use the ALREADY-SPLIT N from kargs!
                // Don't let LEFT/RIGHT do Split-N again - use n_per_split from first transformer
                left_args.N_ = kargs.n_per_split;

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG INVOKER] Creating LEFT kargs with N=" << left_args.N_
                              << " (using n_per_split from kargs)" << std::endl;
                }
                auto kargs_left = Kernel::MakeKernelArgs(left_args);

                // CRITICAL: Manually set n_splits to match kargs!
                // The LEFT transformer won't do Split-N (N=1), so it sets n_splits=1
                // But we need grid.z = original n_splits to process all batches
                kargs_left.n_splits = kargs.n_splits;
                kargs_left.original_n = kargs.original_n;

                // FIX: Use batch_stride from kargs (calculated with ORIGINAL dimensions)
                kargs_left.input_batch_stride = kargs.input_batch_stride;
                kargs_left.output_batch_stride = kargs.output_batch_stride;

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG INVOKER] LEFT kargs: n_per_split=" << kargs_left.n_per_split
                              << ", n_splits=" << kargs_left.n_splits << " (manually set)" << std::endl;
                }
                const dim3 grids_left = Kernel::GridSize(kargs_left);
                const dim3 blocks_left = Kernel::BlockSize();

                // ═══════════════════════════════════════════════════════════
                // COMMON: Create RIGHT descriptor WITHOUT pointer offset
                // ═══════════════════════════════════════════════════════════
                auto right_args = args;
                right_args.input_spatial_lengths_[split_dim] = split_info.in_right;
                right_args.output_spatial_lengths_[split_dim] = split_info.out_right;
                right_args.input_left_pads_[split_dim] = split_info.left_pad_right;
                right_args.input_right_pads_[split_dim] = split_info.right_pad_right;

                // FIX: Keep base pointer, don't apply offset here!
                right_args.in_ptr = args.in_ptr;   // Keep original base pointer
                right_args.out_ptr = args.out_ptr; // Keep original base pointer

                // CRITICAL FIX: Use the ALREADY-SPLIT N from kargs!
                // Don't let LEFT/RIGHT do Split-N again - use n_per_split from first transformer
                right_args.N_ = kargs.n_per_split;

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG INVOKER] Creating RIGHT kargs with N=" << right_args.N_
                              << " (using n_per_split from kargs)" << std::endl;
                    std::cout << "[DEBUG INVOKER] RIGHT spatial offset: input=" << split_info.input_offset
                              << ", output=" << split_info.output_offset << std::endl;
                }
                auto kargs_right = Kernel::MakeKernelArgs(right_args);

                // CRITICAL: Manually set n_splits to match kargs!
                // The RIGHT transformer won't do Split-N (N=1), so it sets n_splits=1
                // But we need grid.z = original n_splits to process all batches
                kargs_right.n_splits = kargs.n_splits;
                kargs_right.original_n = kargs.original_n;

                // FIX: Use batch_stride from kargs (calculated with ORIGINAL dimensions)
                // The kargs_right was created with MODIFIED dimensions, so batch_stride is wrong
                kargs_right.input_batch_stride = kargs.input_batch_stride;
                kargs_right.output_batch_stride = kargs.output_batch_stride;

                // FIX: Store spatial offset in kargs (applied per-batch in operator())
                kargs_right.spatial_offset_in = split_info.input_offset;
                kargs_right.spatial_offset_out = split_info.output_offset;

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG INVOKER] RIGHT kargs: n_per_split=" << kargs_right.n_per_split
                              << ", n_splits=" << kargs_right.n_splits << " (manually set)" << std::endl;
                }
                const dim3 grids_right = Kernel::GridSize(kargs_right);
                const dim3 blocks_right = Kernel::BlockSize();

                // ═══════════════════════════════════════════════════════════
                // COMMON: Launch LEFT and RIGHT kernels sequentially
                // ═══════════════════════════════════════════════════════════
                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT " << NDimSpatial << "D] Launching LEFT kernel..." << std::endl;
                }
                float left_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids_left, blocks_left, 0, kargs_left));

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT " << NDimSpatial << "D] Launching RIGHT kernel..." << std::endl;
                }
                float right_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids_right, blocks_right, 0, kargs_right));

                ave_time = left_time + right_time;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT " << NDimSpatial << "D] Complete! Left=" << left_time
                              << "ms, Right=" << right_time
                              << "ms, Total=" << ave_time << "ms\n";
                }
            }

            return ave_time;
        };

        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::set>{});
    }
};
