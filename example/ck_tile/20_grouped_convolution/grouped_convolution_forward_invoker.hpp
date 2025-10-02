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
                // ═══════════════════════════════════════════════════════════
                // COMMON: Get parameters for split dimension (always dim 0)
                // ═══════════════════════════════════════════════════════════
                const int split_dim = 0;  // Always split first spatial dimension (W/H/D)
                const ck_tile::long_index_t left_pad = args.input_left_pads_[split_dim];
                const ck_tile::long_index_t right_pad = args.input_right_pads_[split_dim];
                const ck_tile::long_index_t out_total = args.output_spatial_lengths_[split_dim];
                const ck_tile::long_index_t in_total = args.input_spatial_lengths_[split_dim];
                const ck_tile::long_index_t filter = args.filter_spatial_lengths_[split_dim];
                const ck_tile::long_index_t stride = args.conv_filter_strides_[split_dim];
                const ck_tile::long_index_t dilation = args.conv_filter_dilations_[split_dim];

                if(s.log_level_ > 0) {
                    std::cout << "[INVOKER] Entering split path (" << NDimSpatial << "D split-image)!" << std::endl;
                    if(NDimSpatial == 1) {
                        std::cout << "[SPLIT " << NDimSpatial << "D] Output W=" << out_total << ", splitting W" << std::endl;
                    } else if(NDimSpatial == 2) {
                        std::cout << "[SPLIT " << NDimSpatial << "D] Output H=" << out_total
                                  << ", W=" << args.output_spatial_lengths_[1]
                                  << ", splitting H (W not contiguous)" << std::endl;
                    } else if(NDimSpatial == 3) {
                        std::cout << "[SPLIT " << NDimSpatial << "D] Output D=" << out_total
                                  << ", H=" << args.output_spatial_lengths_[1]
                                  << ", W=" << args.output_spatial_lengths_[2]
                                  << ", splitting D (H,W not contiguous)" << std::endl;
                    }
                }

                // ═══════════════════════════════════════════════════════════
                // COMMON: Binary split calculation
                // ═══════════════════════════════════════════════════════════
                const ck_tile::long_index_t out_left = out_total / 2;
                const ck_tile::long_index_t out_right = out_total - out_left;
                const ck_tile::long_index_t x_eff = (filter - 1) * dilation + 1;

                // ═══════════════════════════════════════════════════════════
                // SAFETY CHECKS: Can we split this dimension safely?
                // ═══════════════════════════════════════════════════════════
                // Calculate split boundaries (same formula for all dimensions)
                const ck_tile::long_index_t right_start = out_left * stride;
                const ck_tile::long_index_t left_end = (out_left - 1) * stride + x_eff;

                // Check if split is safe:
                // 1. Output dimension must be > 1 (can't split a single element)
                // 2. RIGHT piece must start after left padding
                // 3. LEFT piece must end within input bounds
                const bool is_possible_to_split =
                    out_total != 1 &&
                    right_start > left_pad &&
                    left_end <= (left_pad + in_total);

                if(!is_possible_to_split) {
                    if(s.log_level_ > 0) {
                        std::cout << "[SPLIT " << NDimSpatial << "D] Cannot split safely! Falling back to normal path." << std::endl;
                        std::cout << "  Reason: out_total=" << out_total
                                  << ", right_start=" << right_start << " (must be > left_pad=" << left_pad << ")"
                                  << ", left_end=" << left_end << " (must be <= " << (left_pad + in_total) << ")" << std::endl;
                    }
                    // Fall back to normal launch
                    auto kargs = Kernel::MakeKernelArgs(args);
                    const dim3 grids = Kernel::GridSize(kargs);
                    const dim3 blocks = Kernel::BlockSize();
                    ave_time = ck_tile::launch_kernel(
                        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
                    return ave_time;
                }

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT " << NDimSpatial << "D] Safety check passed! Split is safe." << std::endl;
                }

                // ═══════════════════════════════════════════════════════════
                // DIMENSION-SPECIFIC: Calculate strides for offset computation
                // ═══════════════════════════════════════════════════════════
                ck_tile::long_index_t input_stride, output_stride;
                if constexpr (NDimSpatial == 1) {
                    // 1D NWGC: stride_W = G * C
                    input_stride = static_cast<ck_tile::long_index_t>(args.G_) *
                                   static_cast<ck_tile::long_index_t>(args.C_);
                    output_stride = static_cast<ck_tile::long_index_t>(args.G_) *
                                    static_cast<ck_tile::long_index_t>(args.K_);
                } else if constexpr (NDimSpatial == 2) {
                    // 2D NHWGC: stride_H = W_in * G * C (use INPUT width!)
                    const ck_tile::long_index_t w_in = args.input_spatial_lengths_[1];
                    const ck_tile::long_index_t w_out = args.output_spatial_lengths_[1];
                    input_stride = w_in * static_cast<ck_tile::long_index_t>(args.G_) *
                                   static_cast<ck_tile::long_index_t>(args.C_);
                    output_stride = w_out * static_cast<ck_tile::long_index_t>(args.G_) *
                                    static_cast<ck_tile::long_index_t>(args.K_);
                } else if constexpr (NDimSpatial == 3) {
                    // 3D NDHWGC: stride_D = H_in * W_in * G * C (use INPUT H and W!)
                    const ck_tile::long_index_t h_in = args.input_spatial_lengths_[1];
                    const ck_tile::long_index_t w_in = args.input_spatial_lengths_[2];
                    const ck_tile::long_index_t h_out = args.output_spatial_lengths_[1];
                    const ck_tile::long_index_t w_out = args.output_spatial_lengths_[2];
                    input_stride = h_in * w_in * static_cast<ck_tile::long_index_t>(args.G_) *
                                   static_cast<ck_tile::long_index_t>(args.C_);
                    output_stride = h_out * w_out * static_cast<ck_tile::long_index_t>(args.G_) *
                                    static_cast<ck_tile::long_index_t>(args.K_);
                }

                // ═══════════════════════════════════════════════════════════
                // COMMON: Calculate physical offset and memory offsets
                // ═══════════════════════════════════════════════════════════
                const ck_tile::long_index_t physical_offset = (out_left * stride) - left_pad;
                const ck_tile::long_index_t input_offset = physical_offset * input_stride;
                const ck_tile::long_index_t output_offset = out_left * output_stride;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT " << NDimSpatial << "D] Out_left=" << out_left
                              << ", Out_right=" << out_right << std::endl;
                    std::cout << "[SPLIT " << NDimSpatial << "D] Physical offset: " << physical_offset << std::endl;
                    std::cout << "[SPLIT " << NDimSpatial << "D] Input offset: " << input_offset << " elements" << std::endl;
                    std::cout << "[SPLIT " << NDimSpatial << "D] Output offset: " << output_offset << " elements" << std::endl;
                }

                // ═══════════════════════════════════════════════════════════
                // COMMON: Calculate input sizes for LEFT and RIGHT
                // ═══════════════════════════════════════════════════════════
                const ck_tile::long_index_t in_left_end = (out_left - 1) * stride + x_eff;
                const ck_tile::long_index_t in_left = in_left_end - left_pad;

                const ck_tile::long_index_t in_right_start = out_left * stride;
                const ck_tile::long_index_t in_right_available = in_total - (in_right_start - left_pad);
                const ck_tile::long_index_t in_right = ck_tile::min(in_right_available,
                                                                     (out_right - 1) * stride + x_eff);

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT " << NDimSpatial << "D] LEFT: In=" << in_left
                              << ", Out=" << out_left
                              << ", left_pad=" << left_pad
                              << ", right_pad=0" << std::endl;
                    std::cout << "[SPLIT " << NDimSpatial << "D] RIGHT: In=" << in_right
                              << ", Out=" << out_right
                              << ", left_pad=0"
                              << ", right_pad=" << right_pad << std::endl;
                }

                // ═══════════════════════════════════════════════════════════
                // COMMON: Create LEFT descriptor
                // ═══════════════════════════════════════════════════════════
                auto left_args = args;
                left_args.input_spatial_lengths_[split_dim] = in_left;
                left_args.output_spatial_lengths_[split_dim] = out_left;
                left_args.input_left_pads_[split_dim] = left_pad;
                left_args.input_right_pads_[split_dim] = 0;

                auto kargs_left = Kernel::MakeKernelArgs(left_args);
                const dim3 grids_left = Kernel::GridSize(kargs_left);
                const dim3 blocks_left = Kernel::BlockSize();

                // ═══════════════════════════════════════════════════════════
                // COMMON: Create RIGHT descriptor with offset pointers
                // ═══════════════════════════════════════════════════════════
                InDataType* orig_in_ptr = const_cast<InDataType*>(static_cast<const InDataType*>(args.in_ptr));
                OutDataType* orig_out_ptr = const_cast<OutDataType*>(static_cast<const OutDataType*>(args.out_ptr));

                auto right_args = args;
                right_args.input_spatial_lengths_[split_dim] = in_right;
                right_args.output_spatial_lengths_[split_dim] = out_right;
                right_args.input_left_pads_[split_dim] = 0;
                right_args.input_right_pads_[split_dim] = right_pad;
                right_args.in_ptr = orig_in_ptr + input_offset;
                right_args.out_ptr = orig_out_ptr + output_offset;

                auto kargs_right = Kernel::MakeKernelArgs(right_args);
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
