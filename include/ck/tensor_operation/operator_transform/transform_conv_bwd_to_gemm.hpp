// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"

namespace ck {
namespace tensor_operation {

template <index_t NDimSpatial,
          device::ConvolutionBackwardDataSpecialization ConvBwdDataSpecialization,
          index_t GemmAK1,
          index_t GemmBK1,
          index_t GemmMPerBlock,
          index_t GemmNPerBlock,
          bool DoPadGemmM,
          bool DoPadGemmN>
struct TransformConvBwdDataToGemm_v1
{
    static constexpr auto I1 = Number<1>{};

    template <typename ALayout,
              typename std::enable_if<NDimSpatial == 2 &&
                                          is_same_v<ALayout, tensor_layout::convolution::GNHWC>,
                                      bool>::type = false>
    static auto MakeADescriptor_AK0_M_AK1(
        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
        const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const std::array<index_t, NDimSpatial>& tildes)
    {
        const index_t i_ytilde = tildes[0];
        const index_t i_xtilde = tildes[1];

        const index_t N = out_g_n_k_wos_lengths[1];
        const index_t K = out_g_n_k_wos_lengths[2];

        const index_t Y = wei_g_k_c_xs_lengths[3];
        const index_t X = wei_g_k_c_xs_lengths[4];

        const index_t Ho = out_g_n_k_wos_lengths[3];
        const index_t Wo = out_g_n_k_wos_lengths[4];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const auto GemmAK0 = K / GemmAK1;

        // use default conv_bwd_data specialization for now
        if constexpr(true)
        {
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto YDot = math::integer_divide_ceil(Y, YTilde);
            const auto XDot = math::integer_divide_ceil(X, XTilde);

            const auto HTilde =
                Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IHTildeSliceEnd = math::min(
                HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
            const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

            // A: output tensor
            // GemmK is different for each GEMM
            const auto out_desc_n_ho_wo_k =
                make_naive_tensor_descriptor_packed(make_tuple(N, Ho, Wo, K));

            const auto out_desc_n_hop_wop_k = transform_tensor_descriptor(
                out_desc_n_ho_wo_k,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Ho, I0, I0),
                           make_pad_transform(Wo, I0, I0),
                           make_pass_through_transform(K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto out_desc_n_ydot_htilde_xdot_wtilde_k = transform_tensor_descriptor(
                out_desc_n_hop_wop_k,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(YDot, HTilde),
                                         make_tuple(-ConvDilationH / GcdStrideDilationH, I1)),
                    make_embed_transform(make_tuple(XDot, WTilde),
                                         make_tuple(-ConvDilationW / GcdStrideDilationW, I1)),
                    make_pass_through_transform(K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto out_desc_n_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1 =
                transform_tensor_descriptor(
                    out_desc_n_ydot_htilde_xdot_wtilde_k,
                    make_tuple(make_pass_through_transform(N),
                               make_slice_transform(YDot, I0, YDotSlice),
                               make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                               make_slice_transform(XDot, I0, XDotSlice),
                               make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                               make_unmerge_transform(make_tuple(K0, K1))),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5, 6>{}));

            const auto out_desc_gemmak0_gemmmraw_gemmak1 = transform_tensor_descriptor(
                out_desc_n_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K0)),
                           make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}, Sequence<6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto out_desc_gemmak0_gemmm_gemmak1 =
                PadTensorDescriptor(out_desc_gemmak0_gemmmraw_gemmak1,
                                    make_tuple(ignore, GemmMPerBlock, ignore),
                                    Sequence<false, DoPadGemmM, false>{});

            return out_desc_gemmak0_gemmm_gemmak1;
        }
    }

    template <typename BLayout,
              typename std::enable_if<NDimSpatial == 2 &&
                                          is_same_v<BLayout, tensor_layout::convolution::GKYXC>,
                                      bool>::type = false>
    static auto MakeBDescriptor_BK0_N_BK1(
        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
        const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const std::array<index_t, NDimSpatial>& tildes)
    {
        const index_t i_ytilde = tildes[0];
        const index_t i_xtilde = tildes[1];

        const index_t N = out_g_n_k_wos_lengths[1];
        const index_t K = out_g_n_k_wos_lengths[2];

        const index_t Y = wei_g_k_c_xs_lengths[3];
        const index_t X = wei_g_k_c_xs_lengths[4];

        const index_t Ho = out_g_n_k_wos_lengths[3];
        const index_t Wo = out_g_n_k_wos_lengths[4];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const index_t GemmBK0 = K / GemmBK1

                                // use default conv_bwd_data specialization for now
                                if constexpr(true)
        {
            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            const auto YDot = math::integer_divide_ceil(Y, YTilde);
            const auto XDot = math::integer_divide_ceil(X, XTilde);

            const auto HTilde =
                Ho + math::integer_divide_ceil(ConvDilationH * (Y - I1), ConvStrideH);
            const auto WTilde =
                Wo + math::integer_divide_ceil(ConvDilationW * (X - I1), ConvStrideW);

            // only work on HTilde and WTilde that contribute to non-padding area of input tensor
            const auto IHTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadH - ConvDilationH * (YTilde - I1)), ConvStrideH);
            const auto IWTildeSliceBegin = math::integer_divide_floor(
                math::max(I0, InLeftPadW - ConvDilationW * (XTilde - I1)), ConvStrideW);

            const auto IHTildeSliceEnd = math::min(
                HTilde, math::integer_divide_ceil(InLeftPadH + Hi - I1, ConvStrideH) + I1);
            const auto IWTildeSliceEnd = math::min(
                WTilde, math::integer_divide_ceil(InLeftPadW + Wi - I1, ConvStrideW) + I1);

            const auto HTildeSlice = IHTildeSliceEnd - IHTildeSliceBegin;
            const auto WTildeSlice = IWTildeSliceEnd - IWTildeSliceBegin;

            const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
            const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

            // B: weight tensor
            // GemmK is different for each GEMM
            const auto wei_desc_k_y_x_c =
                make_naive_tensor_descriptor_packed(make_tuple(K, Y, X, C));

            const auto wei_desc_k_ydot_ytilde_xdot_xtilde_c = transform_tensor_descriptor(
                wei_desc_k_y_x_c,
                make_tuple(make_pass_through_transform(K),
                           make_embed_transform(make_tuple(YDot, YTilde),
                                                make_tuple(ConvStrideH / GcdStrideDilationH, I1)),
                           make_embed_transform(make_tuple(XDot, XTilde),
                                                make_tuple(ConvStrideW / GcdStrideDilationW, I1)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto wei_desc_k0_k1_ydotslice_xdotslice_c = transform_tensor_descriptor(
                wei_desc_k_ydot_ytilde_xdot_xtilde_c,
                make_tuple(make_unmerge_transform(make_tuple(GemmBK0, GemmBK1)),
                           make_slice_transform(YDot, I0, YDotSlice),
                           make_slice_transform(XDot, I0, XDotSlice),
                           make_freeze_transform(i_ytilde),
                           make_freeze_transform(i_xtilde),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<3>{},
                           Sequence<2>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<0, 1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<>{},
                           Sequence<>{},
                           Sequence<4>{}));

            const auto wei_gemmk0_gemmnraw_gemmk1_grid_desc = transform_tensor_descriptor(
                wei_k0_k1_ydotslice_xdotslice_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, GemmBK0)),
                           make_pass_through_transform(C),
                           make_pass_through_transform(GemmBK1)),
                make_tuple(Sequence<2, 3, 0>{}, Sequence<4>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto wei_desc_gemmbk0_gemmn_gemmbk1 =
                PadTensorDescriptor(wei_desc_gemmbk0_gemmnraw_gemmbk1,
                                    make_tuple(ignore, GemmNPerBlock, ignore),
                                    Sequence<false, DoPadGemmN, false>{});

            return wei_desc_gemmbk0_gemmn_gemmbk1;
        }
    }

    template <typename CLayout,
              typename std::enable_if<NDimSpatial == 2 &&
                                          is_same_v<CLayout, tensor_layout::convolution::GNHWK>,
                                      bool>::type = false>
    static auto
    MakeCDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
                        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
                        const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        const std::array<index_t, NDimSpatial>& tildes)
    {
        const index_t i_ytilde = tildes[0];
        const index_t i_xtilde = tildes[1];

        const index_t N = out_g_n_k_wos_lengths[1];
        const index_t K = out_g_n_k_wos_lengths[2];

        const index_t Y = wei_g_k_c_xs_lengths[3];
        const index_t X = wei_g_k_c_xs_lengths[4];

        const index_t Ho = out_g_n_k_wos_lengths[3];
        const index_t Wo = out_g_n_k_wos_lengths[4];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        if constexpr(true)
        {
            // C: input tensor
            const auto in_desc_n_hi_wi_c =
                make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

            const auto in_desc_n_hip_wip_c = transform_tensor_descriptor(
                in_desc_n_hi_wi_c,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_desc_n_ytilde_htilde_xtilde_wtilde_c = transform_tensor_descriptor(
                in_desc_n_hip_wip_c,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(YTilde, HTilde),
                                                make_tuple(ConvDilationH, ConvStrideH)),
                           make_embed_transform(make_tuple(XTilde, WTilde),
                                                make_tuple(ConvDilationW, ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_desc_n_htildeslice_wtildeslice_c = transform_tensor_descriptor(
                in_desc_n_ytilde_htilde_xtilde_wtilde_c,
                make_tuple(make_pass_through_transform(N),
                           make_freeze_transform(i_ytilde),
                           make_slice_transform(HTilde, IHTildeSliceBegin, HTildeSlice),
                           make_freeze_transform(i_xtilde),
                           make_slice_transform(WTilde, IWTildeSliceBegin, WTildeSlice),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{},
                           Sequence<1>{},
                           Sequence<2>{},
                           Sequence<3>{},
                           Sequence<4>{},
                           Sequence<5>{}),
                make_tuple(Sequence<0>{},
                           Sequence<>{},
                           Sequence<1>{},
                           Sequence<>{},
                           Sequence<2>{},
                           Sequence<3>{}));

            const auto in_desc_gemmmraw_gemmnraw = transform_tensor_descriptor(
                in_desc_n_htildeslice_wtildeslice_c,
                make_tuple(make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_desc_gemmm_gemmn =
                PadTensorDescriptor(in_desc_gemmmraw_gemmnraw,
                                    make_tuple(GemmMPerBlock, GemmNPerBlock),
                                    Sequence<DoPadGemmM, DoPadGemmN>{});

            return in_desc_gemmm_gemmn;
        }
    }
};

} // namespace tensor_operation
} // namespace ck
