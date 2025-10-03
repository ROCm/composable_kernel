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

            // Check if output size exceeds threshold and needs splitting
            // Calculate output tensor size FIRST before creating kargs
            ck_tile::long_index_t output_size = static_cast<ck_tile::long_index_t>(args.N_) *
                                      static_cast<ck_tile::long_index_t>(args.K_) *
                                      static_cast<ck_tile::long_index_t>(args.G_);
            for(size_t i = 0; i < args.output_spatial_lengths_.size(); i++) {
                output_size *= static_cast<ck_tile::long_index_t>(args.output_spatial_lengths_[i]);
            }

            // Threshold: 2GB in production, 10MB for testing
            static constexpr ck_tile::long_index_t TwoGB = 10L * 1024L * 1024L;  // 10MB for testing
            const ck_tile::long_index_t threshold = TwoGB / sizeof(OutDataType);

            if(s.log_level_ > 0) {
                std::cout << "[INVOKER] Output size: " << output_size << " elements" << std::endl;
                std::cout << "[INVOKER] Threshold: " << threshold << " elements (OutDataType size: "
                          << sizeof(OutDataType) << " bytes)" << std::endl;
                std::cout << "[INVOKER] Comparison: " << output_size << " >= " << threshold << " is "
                          << (output_size >= threshold ? "TRUE" : "FALSE") << std::endl;
            }

            float ave_time = 0.0f;

            // ═══════════════════════════════════════════════════════════
            // UNIFIED SPLIT-IMAGE PATH (1D/2D/3D)
            // ═══════════════════════════════════════════════════════════
            if(output_size >= threshold && (NDimSpatial == 1 || NDimSpatial == 2 || NDimSpatial == 3)) {

                if(s.log_level_ > 0) {
                    std::cout << "[INVOKER] Entering split path (" << NDimSpatial << "D split-image)!" << std::endl;
                }

                // ═══════════════════════════════════════════════════════════
                // GET SPLIT INFO FROM TRANSFORMER (AFTER Split-N!)
                // ═══════════════════════════════════════════════════════════
                // Create temporary kargs to trigger Split-N in transformer
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG INVOKER] Creating temp_kargs with N=" << args.N_ << std::endl;
                }
                auto temp_kargs = Kernel::MakeKernelArgs(args);

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG INVOKER] temp_kargs: n_per_split=" << temp_kargs.n_per_split
                              << ", n_splits=" << temp_kargs.n_splits
                              << ", original_n=" << temp_kargs.original_n << std::endl;
                }

                // Get split info from transformer (uses N_ after Split-N!)
                auto split_info = temp_kargs.GetSplitImageInfo(threshold);

                if(!split_info.should_split) {
                    if(s.log_level_ > 0) {
                        std::cout << "[SPLIT " << NDimSpatial << "D] Cannot split safely or not needed! Using normal path." << std::endl;
                    }
                    // Fall back to normal launch
                    const dim3 grids = Kernel::GridSize(temp_kargs);
                    const dim3 blocks = Kernel::BlockSize();
                    ave_time = ck_tile::launch_kernel(
                        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, temp_kargs));
                    return ave_time;
                }

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT " << NDimSpatial << "D] Split info obtained from transformer!" << std::endl;
                    std::cout << "[SPLIT " << NDimSpatial << "D] Out_left=" << split_info.out_left
                              << ", Out_right=" << split_info.out_right << std::endl;
                    std::cout << "[SPLIT " << NDimSpatial << "D] Input offset: " << split_info.input_offset << " elements" << std::endl;
                    std::cout << "[SPLIT " << NDimSpatial << "D] Output offset: " << split_info.output_offset << " elements" << std::endl;
                }

                // ═══════════════════════════════════════════════════════════
                // Use split info from transformer for LEFT and RIGHT
                // ═══════════════════════════════════════════════════════════
                const int split_dim = 0;  // Always split first spatial dimension (W/H/D)

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT " << NDimSpatial << "D] LEFT: In=" << split_info.in_left
                              << ", Out=" << split_info.out_left
                              << ", left_pad=" << split_info.left_pad_left
                              << ", right_pad=" << split_info.right_pad_left << std::endl;
                    std::cout << "[SPLIT " << NDimSpatial << "D] RIGHT: In=" << split_info.in_right
                              << ", Out=" << split_info.out_right
                              << ", left_pad=" << split_info.left_pad_right
                              << ", right_pad=" << split_info.right_pad_right << std::endl;
                }

                // ═══════════════════════════════════════════════════════════
                // COMMON: Create LEFT descriptor
                // ═══════════════════════════════════════════════════════════
                auto left_args = args;
                left_args.input_spatial_lengths_[split_dim] = split_info.in_left;
                left_args.output_spatial_lengths_[split_dim] = split_info.out_left;
                left_args.input_left_pads_[split_dim] = split_info.left_pad_left;
                left_args.input_right_pads_[split_dim] = split_info.right_pad_left;

                // CRITICAL FIX: Use the ALREADY-SPLIT N from temp_kargs!
                // Don't let LEFT/RIGHT do Split-N again - use n_per_split from first transformer
                left_args.N_ = temp_kargs.n_per_split;

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG INVOKER] Creating LEFT kargs with N=" << left_args.N_
                              << " (using n_per_split from temp_kargs)" << std::endl;
                }
                auto kargs_left = Kernel::MakeKernelArgs(left_args);

                // CRITICAL: Manually set n_splits to match temp_kargs!
                // The LEFT transformer won't do Split-N (N=1), so it sets n_splits=1
                // But we need grid.z = original n_splits to process all batches
                kargs_left.n_splits = temp_kargs.n_splits;
                kargs_left.original_n = temp_kargs.original_n;

                // FIX: Use batch_stride from temp_kargs (calculated with ORIGINAL dimensions)
                kargs_left.input_batch_stride = temp_kargs.input_batch_stride;
                kargs_left.output_batch_stride = temp_kargs.output_batch_stride;

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

                // CRITICAL FIX: Use the ALREADY-SPLIT N from temp_kargs!
                // Don't let LEFT/RIGHT do Split-N again - use n_per_split from first transformer
                right_args.N_ = temp_kargs.n_per_split;

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG INVOKER] Creating RIGHT kargs with N=" << right_args.N_
                              << " (using n_per_split from temp_kargs)" << std::endl;
                    std::cout << "[DEBUG INVOKER] RIGHT spatial offset: input=" << split_info.input_offset
                              << ", output=" << split_info.output_offset << std::endl;
                }
                auto kargs_right = Kernel::MakeKernelArgs(right_args);

                // CRITICAL: Manually set n_splits to match temp_kargs!
                // The RIGHT transformer won't do Split-N (N=1), so it sets n_splits=1
                // But we need grid.z = original n_splits to process all batches
                kargs_right.n_splits = temp_kargs.n_splits;
                kargs_right.original_n = temp_kargs.original_n;

                // FIX: Use batch_stride from temp_kargs (calculated with ORIGINAL dimensions)
                // The kargs_right was created with MODIFIED dimensions, so batch_stride is wrong
                kargs_right.input_batch_stride = temp_kargs.input_batch_stride;
                kargs_right.output_batch_stride = temp_kargs.output_batch_stride;

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
            } else {
                // Normal launch - tensor fits in memory
                if(s.log_level_ > 0) {
                    std::cout << "[INVOKER] Output size " << output_size
                              << " elements fits in threshold - normal launch\n";
                }

                // Create kargs for normal launch
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

                ave_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
            }


            // // Use split-image helper which handles splitting if needed
            // using TransformerType = ck_tile::TransformConvFwdToGemm<
            //     GroupedConvTraitsType::NDimSpatial,
            //     GroupedConvTraitsType::ConvSpecialization,
            //     GroupedConvTraitsType::VectorSizeA,
            //     GroupedConvTraitsType::VectorSizeB,
            //     GroupedConvTraitsType::VectorSizeC,
            //     true>; // Split N enabled

            // // This helper will check if splitting is needed and handle everything
            // if(s.log_level_ > 0) {
            //     std::cout << "[INVOKER] About to call LaunchKernelWithSplitIfNeeded\n";
            // }
            // float ave_time = TransformerType::template LaunchKernelWithSplitIfNeeded<Kernel, kBlockPerCu>(args, s);
            // if(s.log_level_ > 0) {
            //     std::cout << "[INVOKER] LaunchKernelWithSplitIfNeeded returned, ave_time=" << ave_time << "\n";
            // }

            return ave_time;
        };

        return Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                              ck_tile::memory_operation_enum::set>{});
    }
};
