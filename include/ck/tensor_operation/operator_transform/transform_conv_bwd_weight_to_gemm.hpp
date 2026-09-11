// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/library/utility/numeric.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"

namespace ck {
namespace tensor_operation {

template <index_t NDimSpatial,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t GemmK1Number,
          index_t K0PerBlock,
          device::ConvolutionBackwardWeightSpecialization ConvBackwardWeightSpecialization,
          typename IndexType = index_t>
struct TransformConvBwdWeightToGemm
{
    // Same contract as TransformConvBwdWeightToGemmV2 (non-zero K tile factors).
    static_assert(GemmK1Number > 0, "GemmK1Number must be positive");
    static_assert(K0PerBlock > 0, "K0PerBlock must be positive");

    template <index_t N>
    using NumberType =
        std::conditional_t<std::is_same_v<IndexType, index_t>, Number<N>, LongNumber<N>>;

    template <index_t NDim, typename enable_if<NDim == 2, bool>::type = false>
    constexpr static auto
    make_out_grid_desc(const IndexType N,
                       const IndexType Ho,
                       const IndexType Wo,
                       const IndexType K,
                       const std::array<IndexType, NDimSpatial + 3>& output_strides)
    {
        const IndexType WoStride = output_strides[4];
        const auto KStride       = NumberType<1>{};
        return make_naive_tensor_descriptor(make_tuple(N * Ho * Wo, K),
                                            make_tuple(WoStride, KStride));
    }

    template <index_t NDim, typename enable_if<NDim == 2, bool>::type = false>
    constexpr static auto
    make_in_grid_desc(const IndexType N,
                      const IndexType Hi,
                      const IndexType Wi,
                      const IndexType C,
                      const std::array<IndexType, NDimSpatial + 3>& input_strides)
    {
        const IndexType NStride  = input_strides[1];
        const IndexType HiStride = input_strides[3];
        const IndexType WiStride = input_strides[4];
        const auto CStride       = input_strides[2];
        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(N * Hi * Wi, C),
                                                make_tuple(WiStride, CStride));
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(N, Hi, Wi, C),
                                                make_tuple(NStride, HiStride, WiStride, CStride));
        }
    }

    template <index_t NDim, typename enable_if<NDim == 2, bool>::type = false>
    constexpr static auto
    make_wei_grid_desc(const IndexType K,
                       const IndexType Y,
                       const IndexType X,
                       const IndexType C,
                       const std::array<IndexType, NDimSpatial + 3>& weights_strides)
    {
        const auto CStride = NumberType<1>{};
        const auto KStride = weights_strides[1];
        return make_naive_tensor_descriptor(make_tuple(K, Y * X * C), make_tuple(KStride, CStride));
    }

    template <index_t NDim, typename enable_if<NDim == 3, bool>::type = false>
    constexpr static auto
    make_out_grid_desc(const IndexType N,
                       const IndexType Do,
                       const IndexType Ho,
                       const IndexType Wo,
                       const IndexType K,
                       const std::array<IndexType, NDimSpatial + 3>& output_strides)
    {
        const IndexType WoStride = output_strides[5];
        const auto KStride       = NumberType<1>{};
        return make_naive_tensor_descriptor(make_tuple(N * Do * Ho * Wo, K),
                                            make_tuple(WoStride, KStride));
    }

    template <index_t NDim, typename enable_if<NDim == 3, bool>::type = false>
    constexpr static auto
    make_in_grid_desc(const IndexType N,
                      const IndexType Di,
                      const IndexType Hi,
                      const IndexType Wi,
                      const IndexType C,
                      const std::array<IndexType, NDimSpatial + 3>& input_strides)
    {
        const IndexType NStride  = input_strides[1];
        const IndexType DiStride = input_strides[3];
        const IndexType HiStride = input_strides[4];
        const IndexType WiStride = input_strides[5];
        const auto CStride       = input_strides[2];
        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            return make_naive_tensor_descriptor(make_tuple(N * Di * Hi * Wi, C),
                                                make_tuple(WiStride, CStride));
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, C),
                make_tuple(NStride, DiStride, HiStride, WiStride, CStride));
        }
    }

    template <index_t NDim, typename enable_if<NDim == 3, bool>::type = false>
    constexpr static auto
    make_wei_grid_desc(const IndexType K,
                       const IndexType Z,
                       const IndexType Y,
                       const IndexType X,
                       const IndexType C,
                       const std::array<IndexType, NDimSpatial + 3>& weights_strides)
    {
        const auto CStride = NumberType<1>{};
        const auto KStride = weights_strides[1];
        return make_naive_tensor_descriptor(make_tuple(K, Z * Y * X * C),
                                            make_tuple(KStride, CStride));
    }

    template <index_t NDim, typename enable_if<NDim == 1, bool>::type = false>
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const IndexType N,
        const IndexType K,
        const IndexType C,
        const std::array<IndexType, NDimSpatial>& input_spatial_lengths,
        const std::array<IndexType, NDimSpatial>& filter_spatial_lengths,
        const std::array<IndexType, NDimSpatial>& output_spatial_lengths,
        const std::array<IndexType, NDimSpatial + 3>& /* input_strides */,
        const std::array<IndexType, NDimSpatial + 3>& /* weights_strides */,
        const std::array<IndexType, NDimSpatial + 3>& /* output_strides */,
        const std::array<IndexType, NDimSpatial>& conv_filter_strides,
        const std::array<IndexType, NDimSpatial>& conv_filter_dilations,
        const std::array<IndexType, NDimSpatial>& input_left_pads,
        const std::array<IndexType, NDimSpatial>& input_right_pads,
        const index_t batch_k,
        const bool split_k_offset_hack = false) // Deprecated parameter for backward compatibility
    {
        using namespace ck;

        const IndexType Wi            = input_spatial_lengths[0];
        const IndexType Wo            = output_spatial_lengths[0];
        const IndexType X             = filter_spatial_lengths[0];
        const IndexType ConvStrideW   = conv_filter_strides[0];
        const IndexType ConvDilationW = conv_filter_dilations[0];
        const IndexType InLeftPadW    = input_left_pads[0];
        const IndexType InRightPadW   = input_right_pads[0];

        const IndexType GemmKTotal = N * Wo;
        const IndexType GemmM      = K;
        const IndexType GemmN      = C * X;

        const auto PadGemmM = GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
        const auto PadGemmN = GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

        const IndexType GemmKBatch = batch_k;
        const IndexType GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;
        const IndexType KBatchDim = split_k_offset_hack ? 1 : GemmKBatch;
        const IndexType GemmKPad  = KBatchDim * GemmK0 * GemmK1Number;

        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmktotal_gemmm_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Wo, K));

            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_gemmktotal_gemmm_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_gemmktotal_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Wi, C));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, X * C));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_gemmm_gemmn_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
        else
        {
            const auto out_gemmktotal_gemmm_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N * Wo, K));
            const auto in_n_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Wi, C));

            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_gemmktotal_gemmm_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_n_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_n_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto in_gemmktotal_gemmn_grid_desc =
                transform_tensor_descriptor(in_n_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(X, C)),
                                                       make_merge_transform(make_tuple(N, Wo))),
                                            make_tuple(Sequence<1, 3>{}, Sequence<0, 2>{}),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // C: weight tensor
            const auto wei_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(K, X * C));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_gemmm_gemmn_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
    }

    template <index_t NDim, typename enable_if<NDim == 2, bool>::type = false>
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const IndexType N,
        const IndexType K,
        const IndexType C,
        const std::array<IndexType, NDimSpatial>& input_spatial_lengths,
        const std::array<IndexType, NDimSpatial>& filter_spatial_lengths,
        const std::array<IndexType, NDimSpatial>& output_spatial_lengths,
        const std::array<IndexType, NDimSpatial + 3>& input_strides,
        const std::array<IndexType, NDimSpatial + 3>& weights_strides,
        const std::array<IndexType, NDimSpatial + 3>& output_strides,
        const std::array<IndexType, NDimSpatial>& conv_filter_strides,
        const std::array<IndexType, NDimSpatial>& conv_filter_dilations,
        const std::array<IndexType, NDimSpatial>& input_left_pads,
        const std::array<IndexType, NDimSpatial>& input_right_pads,
        const index_t batch_k,
        const bool split_k_offset_hack = false)
    {
        using namespace ck;

        const IndexType Hi = input_spatial_lengths[0];
        const IndexType Wi = input_spatial_lengths[1];

        const IndexType Ho = output_spatial_lengths[0];
        const IndexType Wo = output_spatial_lengths[1];

        const IndexType Y = filter_spatial_lengths[0];
        const IndexType X = filter_spatial_lengths[1];

        const IndexType ConvStrideH = conv_filter_strides[0];
        const IndexType ConvStrideW = conv_filter_strides[1];

        const IndexType ConvDilationH = conv_filter_dilations[0];
        const IndexType ConvDilationW = conv_filter_dilations[1];

        const IndexType InLeftPadH = input_left_pads[0];
        const IndexType InLeftPadW = input_left_pads[1];

        const IndexType InRightPadH = input_right_pads[0];
        const IndexType InRightPadW = input_right_pads[1];

        const IndexType GemmKTotal = N * Ho * Wo;
        const IndexType GemmM      = K;
        const IndexType GemmN      = C * X * Y;

        const auto PadGemmM = GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
        const auto PadGemmN = GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

        const IndexType GemmKBatch = batch_k;
        const IndexType GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;
        const IndexType KBatchDim = split_k_offset_hack ? 1 : GemmKBatch;
        const IndexType GemmKPad  = KBatchDim * GemmK0 * GemmK1Number;

        const auto out_grid_desc = make_out_grid_desc<NDim>(N, Ho, Wo, K, output_strides);
        const auto in_grid_desc  = make_in_grid_desc<NDim>(N, Hi, Wi, C, input_strides);
        const auto wei_grid_desc = make_wei_grid_desc<NDim>(K, Y, X, C, weights_strides);

        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
        else
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_n_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemmktotal_gemmn_grid_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(Y, X, C)),
                                                       make_merge_transform(make_tuple(N, Ho, Wo))),
                                            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                                            make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
    }

    template <index_t NDim, typename enable_if<NDim == 3, bool>::type = false>
    static auto MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const IndexType N,
        const IndexType K,
        const IndexType C,
        const std::array<IndexType, NDimSpatial>& input_spatial_lengths,
        const std::array<IndexType, NDimSpatial>& filter_spatial_lengths,
        const std::array<IndexType, NDimSpatial>& output_spatial_lengths,
        const std::array<IndexType, NDimSpatial + 3>& input_strides,
        const std::array<IndexType, NDimSpatial + 3>& weights_strides,
        const std::array<IndexType, NDimSpatial + 3>& output_strides,
        const std::array<IndexType, NDimSpatial>& conv_filter_strides,
        const std::array<IndexType, NDimSpatial>& conv_filter_dilations,
        const std::array<IndexType, NDimSpatial>& input_left_pads,
        const std::array<IndexType, NDimSpatial>& input_right_pads,
        const index_t batch_k,
        const bool split_k_offset_hack = false)
    {
        using namespace ck;

        const IndexType Di = input_spatial_lengths[0];
        const IndexType Hi = input_spatial_lengths[1];
        const IndexType Wi = input_spatial_lengths[2];

        const IndexType Do = output_spatial_lengths[0];
        const IndexType Ho = output_spatial_lengths[1];
        const IndexType Wo = output_spatial_lengths[2];

        const IndexType Z = filter_spatial_lengths[0];
        const IndexType Y = filter_spatial_lengths[1];
        const IndexType X = filter_spatial_lengths[2];

        const IndexType ConvStrideD = conv_filter_strides[0];
        const IndexType ConvStrideH = conv_filter_strides[1];
        const IndexType ConvStrideW = conv_filter_strides[2];

        const IndexType ConvDilationD = conv_filter_dilations[0];
        const IndexType ConvDilationH = conv_filter_dilations[1];
        const IndexType ConvDilationW = conv_filter_dilations[2];

        const IndexType InLeftPadD = input_left_pads[0];
        const IndexType InLeftPadH = input_left_pads[1];
        const IndexType InLeftPadW = input_left_pads[2];

        const IndexType InRightPadD = input_right_pads[0];
        const IndexType InRightPadH = input_right_pads[1];
        const IndexType InRightPadW = input_right_pads[2];

        const IndexType GemmKTotal = N * Do * Ho * Wo;
        const IndexType GemmM      = K;
        const IndexType GemmN      = C * Z * X * Y;

        const auto PadGemmM = GemmM % MPerBlock == 0 ? 0 : MPerBlock - GemmM % MPerBlock;
        const auto PadGemmN = GemmN % NPerBlock == 0 ? 0 : NPerBlock - GemmN % NPerBlock;

        const IndexType GemmKBatch = batch_k;
        const IndexType GemmK0 =
            math::integer_divide_ceil(GemmKTotal, GemmK1Number * K0PerBlock * GemmKBatch) *
            K0PerBlock;
        const IndexType KBatchDim = split_k_offset_hack ? 1 : GemmKBatch;
        const IndexType GemmKPad  = KBatchDim * GemmK0 * GemmK1Number;

        const auto out_grid_desc = make_out_grid_desc<NDim>(N, Do, Ho, Wo, K, output_strides);
        const auto in_grid_desc  = make_in_grid_desc<NDim>(N, Di, Hi, Wi, C, input_strides);
        const auto wei_grid_desc = make_wei_grid_desc<NDim>(K, Z, Y, X, C, weights_strides);

        if constexpr(ConvBackwardWeightSpecialization ==
                     device::ConvolutionBackwardWeightSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
        else
        {
            // A: output tensor
            const auto out_gemmkpad_gemmm_grid_desc = transform_tensor_descriptor(
                out_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_gemmkpad_gemmm_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmM, PadGemmM)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // B: input tensor
            const auto in_n_dip_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Di, InLeftPadD, InRightPadD),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_n_z_do_y_ho_x_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_dip_hip_wip_c_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Z, Do), make_tuple(ConvDilationD, ConvStrideD)),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{},
                           Sequence<1, 2>{},
                           Sequence<3, 4>{},
                           Sequence<5, 6>{},
                           Sequence<7>{}));

            const auto in_gemmktotal_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(Z, Y, X, C)),
                           make_merge_transform(make_tuple(N, Do, Ho, Wo))),
                make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}),
                make_tuple(Sequence<1>{}, Sequence<0>{}));

            const auto in_gemmkpad_gemmn_grid_desc = transform_tensor_descriptor(
                in_gemmktotal_gemmn_grid_desc,
                make_tuple(make_right_pad_transform(GemmKTotal, GemmKPad - GemmKTotal),
                           make_pass_through_transform(GemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmkpad_gemmn_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(KBatchDim, GemmK0, GemmK1Number)),
                           make_right_pad_transform(GemmN, PadGemmN)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 1, 3>{}, Sequence<2>{}));

            // Padd
            const auto wei_gemmm_gemmn_pad_grid_desc =
                transform_tensor_descriptor(wei_grid_desc,
                                            make_tuple(make_right_pad_transform(GemmM, PadGemmM),
                                                       make_right_pad_transform(GemmN, PadGemmN)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return make_tuple(out_gemmkbatch_gemmk0_gemmm_gemmk1_grid_desc,
                              in_gemmkbatch_gemmk0_gemmn_gemmk1_grid_desc,
                              wei_gemmm_gemmn_pad_grid_desc);
        }
    } // function end
};

} // namespace tensor_operation
} // namespace ck
