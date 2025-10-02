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

            } else if(output_size >= threshold && NDimSpatial == 2) {
                // ═══════════════════════════════════════════════════════════
                // 2D SPLIT-IMAGE (N=1 only for now)
                // ═══════════════════════════════════════════════════════════
                if(s.log_level_ > 0) {
                    std::cout << "[INVOKER] Entering split path (2D split-image)!" << std::endl;
                }

                // === STEP 1: Always split H dimension ===
                // For NHWGC layout, only H is contiguous in memory
                // W is NOT contiguous (scattered across rows), so we can't split it with a single offset
                const ck_tile::long_index_t h_out_total = args.output_spatial_lengths_[0];
                const ck_tile::long_index_t w_out_total = args.output_spatial_lengths_[1];

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 2D] Output H=" << h_out_total
                              << ", W=" << w_out_total
                              << ", splitting H (W not contiguous)" << std::endl;
                }

                // === SPLIT H DIMENSION ===
                const ck_tile::long_index_t h_out_left = h_out_total / 2;
                const ck_tile::long_index_t h_out_right = h_out_total - h_out_left;

                const ck_tile::long_index_t left_pad_h = args.input_left_pads_[0];
                const ck_tile::long_index_t right_pad_h = args.input_right_pads_[0];
                const ck_tile::long_index_t filter_h = args.filter_spatial_lengths_[0];
                const ck_tile::long_index_t stride_h = args.conv_filter_strides_[0];
                const ck_tile::long_index_t dilation_h = args.conv_filter_dilations_[0];

                // Get input width (actual input, not output)
                const ck_tile::long_index_t w_in = args.input_spatial_lengths_[1];

                // Calculate physical offset
                const ck_tile::long_index_t physical_h_offset = (h_out_left * stride_h) - left_pad_h;

                // For NHWGC layout: stride_H = W_in*G*C (use INPUT width!)
                const ck_tile::long_index_t input_offset = physical_h_offset * w_in * args.G_ * args.C_;
                const ck_tile::long_index_t output_offset = h_out_left * w_out_total * args.G_ * args.K_;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 2D-H] H_out_left=" << h_out_left
                              << ", H_out_right=" << h_out_right
                              << ", offset=" << input_offset << std::endl;
                }

                // Calculate input sizes
                const ck_tile::long_index_t x_eff = (filter_h - 1) * dilation_h + 1;
                const ck_tile::long_index_t h_in_left_end = (h_out_left - 1) * stride_h + x_eff;
                const ck_tile::long_index_t h_in_left = h_in_left_end - left_pad_h;

                const ck_tile::long_index_t h_in_right_start = h_out_left * stride_h;
                const ck_tile::long_index_t h_in_right_available = args.input_spatial_lengths_[0] - (h_in_right_start - left_pad_h);
                const ck_tile::long_index_t h_in_right = ck_tile::min(h_in_right_available,
                                                                      (h_out_right - 1) * stride_h + x_eff);

                // Create LEFT args
                auto left_args = args;
                left_args.input_spatial_lengths_[0] = h_in_left;
                left_args.output_spatial_lengths_[0] = h_out_left;
                left_args.input_left_pads_[0] = left_pad_h;
                left_args.input_right_pads_[0] = 0;

                auto kargs_left = Kernel::MakeKernelArgs(left_args);
                const dim3 grids_left = Kernel::GridSize(kargs_left);
                const dim3 blocks_left = Kernel::BlockSize();

                // Create RIGHT args
                InDataType* orig_in_ptr = const_cast<InDataType*>(static_cast<const InDataType*>(args.in_ptr));
                OutDataType* orig_out_ptr = const_cast<OutDataType*>(static_cast<const OutDataType*>(args.out_ptr));

                auto right_args = args;
                right_args.input_spatial_lengths_[0] = h_in_right;
                right_args.output_spatial_lengths_[0] = h_out_right;
                right_args.input_left_pads_[0] = 0;
                right_args.input_right_pads_[0] = right_pad_h;
                right_args.in_ptr = orig_in_ptr + input_offset;
                right_args.out_ptr = orig_out_ptr + output_offset;

                auto kargs_right = Kernel::MakeKernelArgs(right_args);
                const dim3 grids_right = Kernel::GridSize(kargs_right);
                const dim3 blocks_right = Kernel::BlockSize();

                // Launch LEFT
                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 2D-H] Launching LEFT kernel..." << std::endl;
                }
                float left_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids_left, blocks_left, 0, kargs_left));

                // Launch RIGHT
                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 2D-H] Launching RIGHT kernel..." << std::endl;
                }
                float right_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids_right, blocks_right, 0, kargs_right));

                ave_time = left_time + right_time;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 2D-H] Complete! Left=" << left_time
                              << "ms, Right=" << right_time
                              << "ms, Total=" << ave_time << "ms\n";
                }

            } else if(output_size >= threshold && NDimSpatial == 3) {
                // ═══════════════════════════════════════════════════════════
                // 3D SPLIT-IMAGE (D-split only)
                // ═══════════════════════════════════════════════════════════
                if(s.log_level_ > 0) {
                    std::cout << "[INVOKER] Entering split path (3D split-image)!" << std::endl;
                }

                const ck_tile::long_index_t d_out_total = args.output_spatial_lengths_[0];
                const ck_tile::long_index_t h_out_total = args.output_spatial_lengths_[1];
                const ck_tile::long_index_t w_out_total = args.output_spatial_lengths_[2];

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 3D] Output D=" << d_out_total
                              << ", H=" << h_out_total
                              << ", W=" << w_out_total
                              << ", splitting D (H,W not contiguous)" << std::endl;
                }

                // === SPLIT D DIMENSION ===
                const ck_tile::long_index_t d_out_left = d_out_total / 2;
                const ck_tile::long_index_t d_out_right = d_out_total - d_out_left;

                const ck_tile::long_index_t left_pad_d = args.input_left_pads_[0];
                const ck_tile::long_index_t right_pad_d = args.input_right_pads_[0];
                const ck_tile::long_index_t filter_d = args.filter_spatial_lengths_[0];
                const ck_tile::long_index_t stride_d = args.conv_filter_strides_[0];
                const ck_tile::long_index_t dilation_d = args.conv_filter_dilations_[0];

                // Get input H and W (actual input dimensions)
                const ck_tile::long_index_t h_in = args.input_spatial_lengths_[1];
                const ck_tile::long_index_t w_in = args.input_spatial_lengths_[2];

                // Calculate physical offset
                const ck_tile::long_index_t physical_d_offset = (d_out_left * stride_d) - left_pad_d;

                // For NDHWGC layout: stride_D = H_in*W_in*G*C (use INPUT H and W!)
                const ck_tile::long_index_t input_offset = physical_d_offset * h_in * w_in * args.G_ * args.C_;
                const ck_tile::long_index_t output_offset = d_out_left * h_out_total * w_out_total * args.G_ * args.K_;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 3D-D] D_out_left=" << d_out_left
                              << ", D_out_right=" << d_out_right << std::endl;
                    std::cout << "[SPLIT 3D-D] Input dimensions: D_in=" << args.input_spatial_lengths_[0]
                              << ", H_in=" << h_in << ", W_in=" << w_in << std::endl;
                    std::cout << "[SPLIT 3D-D] Physical D offset: " << physical_d_offset << std::endl;
                    std::cout << "[SPLIT 3D-D] Stride_D = " << h_in << " * " << w_in << " * " << args.G_ << " * " << args.C_
                              << " = " << (h_in * w_in * args.G_ * args.C_) << " elements per D-slice" << std::endl;
                    std::cout << "[SPLIT 3D-D] Input offset: " << input_offset << " elements" << std::endl;
                    std::cout << "[SPLIT 3D-D] Output offset: " << output_offset << " elements" << std::endl;
                }

                // Calculate input sizes
                const ck_tile::long_index_t x_eff = (filter_d - 1) * dilation_d + 1;
                const ck_tile::long_index_t d_in_left_end = (d_out_left - 1) * stride_d + x_eff;
                const ck_tile::long_index_t d_in_left = d_in_left_end - left_pad_d;

                const ck_tile::long_index_t d_in_right_start = d_out_left * stride_d;
                const ck_tile::long_index_t d_in_right_available = args.input_spatial_lengths_[0] - (d_in_right_start - left_pad_d);
                const ck_tile::long_index_t d_in_right = ck_tile::min(d_in_right_available,
                                                                      (d_out_right - 1) * stride_d + x_eff);

                // Create LEFT args
                auto left_args = args;
                left_args.input_spatial_lengths_[0] = d_in_left;
                left_args.output_spatial_lengths_[0] = d_out_left;
                left_args.input_left_pads_[0] = left_pad_d;
                left_args.input_right_pads_[0] = 0;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 3D-D] LEFT: D_in=" << d_in_left
                              << ", D_out=" << d_out_left
                              << ", left_pad=" << left_pad_d
                              << ", right_pad=0" << std::endl;
                }

                auto kargs_left = Kernel::MakeKernelArgs(left_args);
                const dim3 grids_left = Kernel::GridSize(kargs_left);
                const dim3 blocks_left = Kernel::BlockSize();

                // Create RIGHT args
                InDataType* orig_in_ptr = const_cast<InDataType*>(static_cast<const InDataType*>(args.in_ptr));
                OutDataType* orig_out_ptr = const_cast<OutDataType*>(static_cast<const OutDataType*>(args.out_ptr));

                auto right_args = args;
                right_args.input_spatial_lengths_[0] = d_in_right;
                right_args.output_spatial_lengths_[0] = d_out_right;
                right_args.input_left_pads_[0] = 0;
                right_args.input_right_pads_[0] = right_pad_d;
                right_args.in_ptr = orig_in_ptr + input_offset;
                right_args.out_ptr = orig_out_ptr + output_offset;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 3D-D] RIGHT: D_in=" << d_in_right
                              << ", D_out=" << d_out_right
                              << ", left_pad=0"
                              << ", right_pad=" << right_pad_d
                              << ", in_ptr_offset=" << input_offset
                              << ", out_ptr_offset=" << output_offset << std::endl;
                }

                auto kargs_right = Kernel::MakeKernelArgs(right_args);
                const dim3 grids_right = Kernel::GridSize(kargs_right);
                const dim3 blocks_right = Kernel::BlockSize();

                // Launch LEFT
                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 3D-D] Launching LEFT kernel..." << std::endl;
                }
                float left_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids_left, blocks_left, 0, kargs_left));

                // Launch RIGHT
                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 3D-D] Launching RIGHT kernel..." << std::endl;
                }
                float right_time = ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids_right, blocks_right, 0, kargs_right));

                ave_time = left_time + right_time;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT 3D-D] Complete! Left=" << left_time
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
