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

            if(output_size >= threshold && NDimSpatial == 1) {
                // Large tensor - use split-image helper (1D only for now)
                if(s.log_level_ > 0) {
                    std::cout << "[INVOKER] Entering split path (1D split-image)!" << std::endl;
                    std::cout << "[INVOKER] Output size " << output_size
                              << " elements exceeds threshold " << threshold
                              << " - calling split helper" << std::endl;
                }

                // === STEP 1: Calculate offset ===
                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 1D] STEP 1: Calculating offsets..." << std::endl;
                }
                // Original input width WITH padding
                const ck_tile::long_index_t left_pad = args.input_left_pads_[0];
                if(s.log_level_ > 0) { std::cout << "[DEBUG] Got left_pad" << std::endl; }
                const ck_tile::long_index_t right_pad = args.input_right_pads_[0];
                if(s.log_level_ > 0) { std::cout << "[DEBUG] Got right_pad" << std::endl; }
                const ck_tile::long_index_t w_in_total = args.input_spatial_lengths_[0] + left_pad + right_pad;
                if(s.log_level_ > 0) { std::cout << "[DEBUG] Got w_in_total" << std::endl; }
                const ck_tile::long_index_t w_out_total = args.output_spatial_lengths_[0];
                if(s.log_level_ > 0) { std::cout << "[DEBUG] Got w_out_total" << std::endl; }
                const ck_tile::long_index_t w_out_left = w_out_total / 2;
                const ck_tile::long_index_t w_out_right = w_out_total - w_out_left;

                const ck_tile::long_index_t filter_w = args.filter_spatial_lengths_[0];
                if(s.log_level_ > 0) { std::cout << "[DEBUG] Got filter_w" << std::endl; }
                const ck_tile::long_index_t stride_w = args.conv_filter_strides_[0];
                if(s.log_level_ > 0) { std::cout << "[DEBUG] Got stride_w" << std::endl; }
                const ck_tile::long_index_t dilation_w = args.conv_filter_dilations_[0];
                if(s.log_level_ > 0) { std::cout << "[DEBUG] Got dilation_w" << std::endl; }

                // IMPORTANT: Padding is VIRTUAL, not physical!
                // The input tensor in memory has NO padding.
                // So we offset based on the physical unpadded position.

                // Physical input offset: where to start reading in the unpadded input tensor
                // Reference formula: ((Wo_ / 2) * ConvStrideW_ - InLeftPadW_)
                const ck_tile::long_index_t physical_w_offset_in = (w_out_left * stride_w) - left_pad;  // 16383

                // Memory offsets (in elements) - use PHYSICAL offset!
                const ck_tile::long_index_t input_offset = physical_w_offset_in * args.C_ * args.G_;
                const ck_tile::long_index_t output_offset = w_out_left * args.K_ * args.G_;

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] About to print offsets..." << std::endl;
                    std::cout << "[SPLIT 1D] W_out: " << w_out_total << std::endl;
                    std::cout << "[DEBUG] After W_out" << std::endl;
                    std::cout << " left=" << w_out_left << ", right=" << w_out_right << std::endl;
                    std::cout << "[DEBUG] After left/right" << std::endl;
                    std::cout << "[SPLIT 1D] Physical W offset in: " << physical_w_offset_in << std::endl;
                    std::cout << "[DEBUG] After physical offset" << std::endl;
                    std::cout << " (elements: " << input_offset << ")" << std::endl;
                    std::cout << "[SPLIT 1D] W offset out: " << w_out_left << " (elements: " << output_offset << ")" << std::endl;
                }

                // === STEP 2: Create descriptor LEFT ===
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] STEP 2: Creating LEFT descriptor..." << std::endl;
                }
                auto left_args = args;
                // Calculate required input width for LEFT piece
                // Formula: w_in = (w_out - 1) * stride + dilated_filter - lpad - rpad
                const ck_tile::long_index_t dilated_filter_w = (filter_w - 1) * dilation_w + 1;
                const ck_tile::long_index_t w_in_left = (w_out_left - 1) * stride_w + dilated_filter_w - left_pad - 0;
                left_args.input_spatial_lengths_[0] = w_in_left;
                left_args.output_spatial_lengths_[0] = w_out_left;
                left_args.input_right_pads_[0] = 0;  // Remove right padding

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 1D] LEFT: Wi=" << w_in_left
                              << ", Wo=" << w_out_left
                              << ", left_pad=" << left_args.input_left_pads_[0]
                              << ", right_pad=" << left_args.input_right_pads_[0] << std::endl;
                }
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] About to call MakeKernelArgs for LEFT..." << std::endl;
                }
                auto kargs_left = Kernel::MakeKernelArgs(left_args);
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] MakeKernelArgs succeeded for LEFT" << std::endl;
                }
                const dim3 grids_left = Kernel::GridSize(kargs_left);
                const dim3 blocks_left = Kernel::BlockSize();
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] GridSize and BlockSize computed for LEFT" << std::endl;
                }

                // === STEP 3: Create descriptor RIGHT ===
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] STEP 3: Creating RIGHT descriptor..." << std::endl;
                }
                auto right_args = args;
                // RIGHT needs enough input for w_out_right outputs: (w_out_right - 1) * stride + filter
                // Use dilated_filter_w from LEFT calculation
                const ck_tile::long_index_t w_in_right_needed = (w_out_right - 1) * stride_w + dilated_filter_w;
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] Calculated w_in_right_needed=" << w_in_right_needed << std::endl;
                }
                // Available input: Use UNPADDED input width (like reference implementation)
                // Reference formula: Wi_ - (wi_right_transformer_start_idx - InLeftPadW_)
                //                  = Wi_ - ((Wo_/2) * stride - left_pad)
                const ck_tile::long_index_t w_in_unpadded = args.input_spatial_lengths_[0];
                const ck_tile::long_index_t w_in_right_available = w_in_unpadded - ((w_out_left * stride_w) - left_pad);
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] Calculated w_in_right_available=" << w_in_right_available << std::endl;
                }

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 1D DEBUG] w_in_right_needed = (" << w_out_right << " - 1) * " << stride_w
                              << " + " << dilated_filter_w << " = " << w_in_right_needed << std::endl;
                    std::cout << "[SPLIT 1D DEBUG] w_in_right_available = " << w_in_total
                              << " - (" << w_out_left << " * " << stride_w << ") = " << w_in_right_available << std::endl;
                }

                // Use minimum
                const ck_tile::long_index_t w_in_right = (w_in_right_available < w_in_right_needed)
                                                         ? w_in_right_available
                                                         : w_in_right_needed;
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] Calculated w_in_right=" << w_in_right << std::endl;
                }

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 1D DEBUG] w_in_right = min(" << w_in_right_available
                              << ", " << w_in_right_needed << ") = " << w_in_right << "\n";
                }

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] About to set right_args fields..." << std::endl;
                }
                right_args.input_spatial_lengths_[0] = w_in_right;
                right_args.output_spatial_lengths_[0] = w_out_right;
                right_args.input_left_pads_[0] = 0;  // Remove left padding
                // right_args.input_right_pads_[0] is already set from args (keep right padding)
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] Set spatial lengths and padding" << std::endl;
                }

                // Adjust pointers
                const InDataType* orig_in_ptr = static_cast<const InDataType*>(args.in_ptr);
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] Cast orig_in_ptr" << std::endl;
                }
                OutDataType* orig_out_ptr = static_cast<OutDataType*>(args.out_ptr);
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] Cast orig_out_ptr" << std::endl;
                }
                right_args.in_ptr = orig_in_ptr + input_offset;
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] Set right_args.in_ptr" << std::endl;
                }
                right_args.out_ptr = orig_out_ptr + output_offset;
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] Set right_args.out_ptr" << std::endl;
                }

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] About to print RIGHT info..." << std::endl;
                    std::cout << "[SPLIT 1D] RIGHT: Wi=" << w_in_right << std::endl;
                    std::cout << ", Wo=" << w_out_right << std::endl;
                    std::cout << ", left_pad=" << right_args.input_left_pads_[0] << std::endl;
                    std::cout << ", right_pad=" << right_args.input_right_pads_[0] << std::endl;
                    std::cout << ", in_ptr_offset=" << input_offset << std::endl;
                    std::cout << ", out_ptr_offset=" << output_offset << std::endl;
                }
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] About to call MakeKernelArgs for RIGHT..." << std::endl;
                }
                auto kargs_right = Kernel::MakeKernelArgs(right_args);
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] MakeKernelArgs succeeded for RIGHT" << std::endl;
                }
                const dim3 grids_right = Kernel::GridSize(kargs_right);
                const dim3 blocks_right = Kernel::BlockSize();
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] GridSize and BlockSize computed for RIGHT" << std::endl;
                }

                // === STEP 4: Run LEFT kernel ===
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] About to launch LEFT kernel..." << std::endl;
                    std::cout << "[SPLIT 1D] Launching LEFT kernel..." << std::endl;
                }
                float left_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids_left, blocks_left, 0, kargs_left));

                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] LEFT kernel returned! Time=" << left_time << "ms" << std::endl;
                }

                // === STEP 5: Run RIGHT kernel ===
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] About to launch RIGHT kernel..." << std::endl;
                    std::cout << "[SPLIT 1D] Launching RIGHT kernel..." << std::endl;
                }
                float right_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids_right, blocks_right, 0, kargs_right));
                if(s.log_level_ > 0) {
                    std::cout << "[DEBUG] RIGHT kernel returned! Time=" << right_time << "ms" << std::endl;
                }

                // === STEP 6: No explicit combine needed (write to different locations) ===
                // === STEP 7: Return total time ===
                ave_time = left_time + right_time;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 1D] Complete! Left=" << left_time
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
