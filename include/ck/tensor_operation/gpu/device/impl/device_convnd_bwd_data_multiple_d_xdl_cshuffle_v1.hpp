// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v3r2.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

// Conv backward data multiple D:
//   input : output image A: [G, N, K, Ho, Wo]
//   input : weight B: [G, K, C, Y, X],
//   input : D0, D1, ... : [G, N, K, Ho, Wo]
//   output : input image E: [G, N, C, Hi, Wi]
//   C = a_op(A) * b_op(B)
//   E = cde_op(C, D0, D1, ...)
template <
    ck::index_t NDimSpatial,
    typename ALayout,   // output image
    typename BLayout,   // weight
    typename DsLayout,  // bias
    typename ELayout,   // input image
    typename ADataType, // output image
    typename BDataType, // weight
    typename AccDataType,
    typename CShuffleDataType,
    typename DsDataType,       // bias
    typename EDataType,        // input image
    typename AElementwiseOp,   // output image
    typename BElementwiseOp,   // weight
    typename CDEElementwiseOp, // C, bias, and input image
    ConvolutionBackwardDataSpecialization ConvBackwardDataSpecialization,
    ck::index_t BlockSize,
    ck::index_t MPerBlock,
    ck::index_t NPerBlock,
    ck::index_t K0PerBlock,
    ck::index_t K1,
    ck::index_t MPerXdl,
    ck::index_t NPerXdl,
    ck::index_t MXdlPerWave,
    ck::index_t NXdlPerWave,
    typename ABlockTransferThreadClusterLengths_K0_M_K1,
    typename ABlockTransferThreadClusterArrangeOrder,
    typename ABlockTransferSrcAccessOrder,
    ck::index_t ABlockTransferSrcVectorDim,
    ck::index_t ABlockTransferSrcScalarPerVector,
    ck::index_t ABlockTransferDstScalarPerVector_K1,
    bool ABlockLdsAddExtraM,
    typename BBlockTransferThreadClusterLengths_K0_N_K1,
    typename BBlockTransferThreadClusterArrangeOrder,
    typename BBlockTransferSrcAccessOrder,
    ck::index_t BBlockTransferSrcVectorDim,
    ck::index_t BBlockTransferSrcScalarPerVector,
    ck::index_t BBlockTransferDstScalarPerVector_K1,
    bool BBlockLdsAddExtraN,
    index_t CShuffleMXdlPerWavePerShuffle,
    index_t CShuffleNXdlPerWavePerShuffle,
    typename CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
    index_t CBlockTransferScalarPerVector_NWaveNPerXdl>
struct DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1
    : public DeviceGroupedConvBwdDataMultipleD<NDimSpatial,
                                               ALayout,    // output image
                                               BLayout,    // weight
                                               DsLayout,   // bias
                                               ELayout,    // input image
                                               ADataType,  // output image
                                               BDataType,  // weight
                                               DsDataType, // bias
                                               EDataType,  // input image
                                               AElementwiseOp,
                                               BElementwiseOp,
                                               CDEElementwiseOp>
{
    using DeviceOp = DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1;

    static constexpr index_t NumDTensor = DsDataType::Size();

    // TODO make A/B datatype different
    using ABDataType = ADataType;

    using OutDataType = ADataType;
    using WeiDataType = BDataType;
    using InDataType  = EDataType;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    static_assert((K1 % ABlockTransferThreadClusterLengths_K0_M_K1{}[I2]) %
                      ABlockTransferSrcScalarPerVector ==
                  0);

    static_assert((NPerBlock / BBlockTransferThreadClusterLengths_K0_N_K1{}[I1]) %
                      BBlockTransferSrcScalarPerVector ==
                  0);

    static constexpr auto K1Number     = Number<K1>{};
    static constexpr auto GemmK1Number = K1Number;

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto MakeABCDsGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
        const std::array<index_t, NDimSpatial + 3>& out_g_n_k_wos_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* out_g_n_k_wos_strides */,
        const std::array<index_t, NDimSpatial + 3>& wei_g_k_c_xs_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* wei_g_k_c_xs_strides */,
        const std::array<index_t, NDimSpatial + 3>& in_g_n_c_wis_lengths,
        const std::array<index_t, NDimSpatial + 3>& /* in_g_n_c_wis_strides */,
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const std::array<index_t, NDimSpatial>& tildes)
    {
        using namespace ck;

        index_t i_ytilde = tildes[0];
        index_t i_xtilde = tildes[1];

        const index_t N = in_g_n_c_wis_lengths[1];
        const index_t K = wei_g_k_c_xs_lengths[1];
        const index_t C = wei_g_k_c_xs_lengths[2];

        const index_t Hi = in_g_n_c_wis_lengths[3];
        const index_t Wi = in_g_n_c_wis_lengths[4];

        const index_t Ho = out_g_n_k_wos_lengths[3];
        const index_t Wo = out_g_n_k_wos_lengths[4];

        const index_t Y = wei_g_k_c_xs_lengths[3];
        const index_t X = wei_g_k_c_xs_lengths[4];

        const index_t InLeftPadH = input_left_pads[0];
        const index_t InLeftPadW = input_left_pads[1];

        const index_t InRightPadH = input_right_pads[0];
        const index_t InRightPadW = input_right_pads[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        const index_t ConvDilationH = conv_filter_dilations[0];
        const index_t ConvDilationW = conv_filter_dilations[1];

        const auto K0 = K / K1;

        const auto out_n_ho_wo_k_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Ho, Wo, K));
        const auto wei_k_y_x_e_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(K, Y, X, C));
        const auto in_n_hi_wi_e_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // A: output tensor
            const auto out_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                make_naive_tensor_descriptor_packed(make_tuple(N * Ho * Wo, K)),
                make_tuple(make_pass_through_transform(N * Ho * Wo),
                           make_unmerge_transform(make_tuple(K0, K1))),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<1>{}, Sequence<0, 2>{}));

            // B: weight tensor
            const auto wei_gemmk0_gemmn_gemmk1_grid_desc =
                transform_tensor_descriptor(make_naive_tensor_descriptor_packed(make_tuple(K, C)),
                                            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // C: input tensor
            const auto in_n_y_ho_x_wo_e_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_e_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(I1, Ho), make_tuple(I1, ConvStrideH)),
                           make_embed_transform(make_tuple(I1, Wo), make_tuple(I1, ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_y_ho_x_wo_e_grid_desc,
                make_tuple(make_freeze_transform(I0),
                           make_freeze_transform(I0),
                           make_merge_transform(make_tuple(N, Ho, Wo)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<1>{}, Sequence<3>{}, Sequence<0, 2, 4>{}, Sequence<5>{}),
                make_tuple(Sequence<>{}, Sequence<>{}, Sequence<0>{}, Sequence<1>{}));

            const auto bias_gemmm_gemmn_grid_desc =
                make_naive_tensor_descriptor(make_tuple(N * Ho * Wo, C), make_tuple(I0, I1));

            return make_tuple(out_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              in_gemmm_gemmn_grid_desc,
                              bias_gemmm_gemmn_grid_desc);
        }
        else
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

            // GemmK is different for each GEMM
            const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
            const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

            // A: output tensor
            const auto out_n_hop_wop_k_grid_desc = transform_tensor_descriptor(
                out_n_ho_wo_k_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Ho, I0, I0),
                           make_pad_transform(Wo, I0, I0),
                           make_pass_through_transform(K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto out_n_ydot_htilde_xdot_wtilde_k_grid_desc = transform_tensor_descriptor(
                out_n_hop_wop_k_grid_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(YDot, HTilde),
                                         make_tuple(-ConvDilationH / GcdStrideDilationH, I1)),
                    make_embed_transform(make_tuple(XDot, WTilde),
                                         make_tuple(-ConvDilationW / GcdStrideDilationW, I1)),
                    make_pass_through_transform(K)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1_grid_desc =
                transform_tensor_descriptor(
                    out_n_ydot_htilde_xdot_wtilde_k_grid_desc,
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

            const auto out_gemmk0_gemmm_gemmk1_grid_desc = transform_tensor_descriptor(
                out_n_ydotslice_htildeslice_xdotslice_wtildeslice_k0_k1_grid_desc,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K0)),
                           make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}, Sequence<6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // B weight tensor
            const auto wei_k_ydot_ytilde_xdot_xtilde_e_grid_desc = transform_tensor_descriptor(
                wei_k_y_x_e_grid_desc,
                make_tuple(make_pass_through_transform(K),
                           make_embed_transform(make_tuple(YDot, YTilde),
                                                make_tuple(ConvStrideH / GcdStrideDilationH, I1)),
                           make_embed_transform(make_tuple(XDot, XTilde),
                                                make_tuple(ConvStrideW / GcdStrideDilationW, I1)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto wei_k0_k1_ydotslice_xdotslice_e_grid_desc =
                transform_tensor_descriptor(wei_k_ydot_ytilde_xdot_xtilde_e_grid_desc,
                                            make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
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

            const auto wei_gemmk0_gemmn_gemmk1_grid_desc = transform_tensor_descriptor(
                wei_k0_k1_ydotslice_xdotslice_e_grid_desc,
                make_tuple(make_merge_transform(make_tuple(YDotSlice, XDotSlice, K0)),
                           make_pass_through_transform(C),
                           make_pass_through_transform(K1)),
                make_tuple(Sequence<2, 3, 0>{}, Sequence<4>{}, Sequence<1>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            // C: input tensor
            const auto in_n_hip_wip_e_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_e_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_n_ytilde_htilde_xtilde_wtilde_e_grid_desc = transform_tensor_descriptor(
                in_n_hip_wip_e_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(YTilde, HTilde),
                                                make_tuple(ConvDilationH, ConvStrideH)),
                           make_embed_transform(make_tuple(XTilde, WTilde),
                                                make_tuple(ConvDilationW, ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_n_htildeslice_wtildeslice_e_grid_desc = transform_tensor_descriptor(
                in_n_ytilde_htilde_xtilde_wtilde_e_grid_desc,
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

            const auto in_gemmm_gemmn_grid_desc = transform_tensor_descriptor(
                in_n_htildeslice_wtildeslice_e_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, HTildeSlice, WTildeSlice)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            // bias tensor
            const auto bias_gemmm_gemmn_grid_desc = make_naive_tensor_descriptor(
                make_tuple(N * HTildeSlice * WTildeSlice, C), make_tuple(I0, I1));

            return make_tuple(out_gemmk0_gemmm_gemmk1_grid_desc,
                              wei_gemmk0_gemmn_gemmk1_grid_desc,
                              in_gemmm_gemmn_grid_desc,
                              bias_gemmm_gemmn_grid_desc);
        }

    } // function end

    template <ck::index_t NDim, typename ck::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCDsGridDesc()
    {
        return MakeABCDsGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<2>({1, 1, 1, 1, 1},
                                                                    {1, 1, 1, 1, 1},
                                                                    {1, 1, 1, 1, 1},
                                                                    {1, 1, 1, 1, 1},
                                                                    {1, 1, 1, 1, 1},
                                                                    {1, 1, 1, 1, 1},
                                                                    {1, 1},
                                                                    {1, 1},
                                                                    {1, 1},
                                                                    {1, 1},
                                                                    {0, 0});
    }

    using ABCDsGridDescs = decltype(GetABCDsGridDesc<NDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCDsGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCDsGridDescs{}[I1])>;
    using EGridDesc_M_N     = remove_cvref_t<decltype(ABCDsGridDescs{}[I2])>;
    using DsGridDesc_M_N    = remove_cvref_t<decltype(make_tuple(ABCDsGridDescs{}[I3]))>;

    // GridwiseGemm
#if 0
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v3r2<
        BlockSize,
        ABDataType, // TODO: distinguish A/B datatype
        AccDataType,
        EDataType,
        InMemoryDataOperationEnum::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        EGridDesc_M_N,
        tuple_element_t<0, DsGridDesc_M_N>,
        AElementwiseOp,
        BElementwiseOp,
        CDEElementwiseOp,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXdl,
        NPerXdl,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CBlockTransferClusterLengths_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl,
        CBlockTransferScalarPerVector_NWaveNPerXdl>;
#else
    using GridwiseGemm = GridwiseGemmMultipleD_xdl_cshuffle<
        ABDataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOp,
        BElementwiseOp,
        CDEElementwiseOp,
        InMemoryDataOperationEnum::Set,
        AGridDesc_M_K,
        BGridDesc_N_K,
        DsGridDesc_M_N,
        EGridDesc_M_N,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        AK1,
        BK1,
        MPerXDL,
        NPerXDL,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_AK0_M_AK1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_AK1,
        false,
        ABlockLdsExtraM,
        BBlockTransferThreadClusterLengths_BK0_N_BK1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_BK1,
        false,
        BBlockLdsExtraN,
        CShuffleMXdlPerWavePerShuffle,
        CShuffleNXdlPerWavePerShuffle,
        CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        CDEBlockTransferScalarPerVector_NPerBlock,
        LoopSched>;
#endif

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const void* p_a,                                 // output image
                 const void* p_b,                                 // weight
                 const std::array<const void*, NumDTensor>& p_ds, // bias
                 void* p_e,                                       // input image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths,
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_lengths,
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths,
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOp& a_element_op,
                 const BElementwiseOp& b_element_op,
                 const CDEElementwiseOp& cde_element_op)
            : p_a_grid_{p_a},
              p_b_grid_{p_b},
              p_d0_grid_{p_ds[I0]},
              p_e_grid_{p_e},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              cde_element_op_{cde_element_op},
              a_g_n_k_wos_lengths_{a_g_n_k_wos_lengths},
              a_g_n_k_wos_strides_{a_g_n_k_wos_strides},
              b_g_k_c_xs_lengths_{b_g_k_c_xs_lengths},
              b_g_k_c_xs_strides_{b_g_k_c_xs_strides},
              ds_g_n_c_wis_lengths_{ds_g_n_c_wis_lengths},
              ds_g_n_c_wis_strides_{ds_g_n_c_wis_strides},
              e_g_n_c_wis_lengths_{e_g_n_c_wis_lengths},
              e_g_n_c_wis_strides_{e_g_n_c_wis_strides},
              conv_filter_strides_{conv_filter_strides},
              conv_filter_dilations_{conv_filter_dilations},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            const index_t Y = b_g_k_c_xs_lengths[3];
            const index_t X = b_g_k_c_xs_lengths[4];

            const index_t ConvStrideH = conv_filter_strides_[0];
            const index_t ConvStrideW = conv_filter_strides_[1];

            const index_t ConvDilationH = conv_filter_dilations_[0];
            const index_t ConvDilationW = conv_filter_dilations_[1];

            const auto GcdStrideDilationH = math::gcd(ConvStrideH, ConvDilationH);
            const auto GcdStrideDilationW = math::gcd(ConvStrideW, ConvDilationW);

            const auto YTilde = ConvStrideH / GcdStrideDilationH;
            const auto XTilde = ConvStrideW / GcdStrideDilationW;

            for(index_t i_ytilde = 0; i_ytilde < YTilde; ++i_ytilde)
            {
                for(index_t i_xtilde = 0; i_xtilde < XTilde; ++i_xtilde)
                {
                    // check slice is valid
                    const auto YDotSlice = math::integer_divide_ceil(Y - i_ytilde, YTilde);
                    const auto XDotSlice = math::integer_divide_ceil(X - i_xtilde, XTilde);

                    if(YDotSlice * XDotSlice <= 0)
                    {
                        continue;
                    }

                    const auto descs =
                        DeviceOp::MakeABCDsGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<NDimSpatial>(
                            a_g_n_k_wos_lengths_,
                            a_g_n_k_wos_strides_,
                            b_g_k_c_xs_lengths_,
                            b_g_k_c_xs_strides_,
                            e_g_n_c_wis_lengths_,
                            e_g_n_c_wis_strides_,
                            conv_filter_strides_,
                            conv_filter_dilations_,
                            input_left_pads_,
                            input_right_pads_,
                            {i_ytilde, i_xtilde});

                    a_grid_desc_k0_m_k1_container_.push_back(descs[I0]);
                    b_grid_desc_k0_n_k1_container_.push_back(descs[I1]);
                    e_grid_desc_m_n_container_.push_back(descs[I2]);
                    ds_grid_desc_m_n_container_.push_back(ck::make_tuple(descs[I3]));

                    auto block_2_etile_map =
                        GridwiseGemm::MakeDefaultBlock2CTileMap(descs[I2], 1, 1);

                    if(GridwiseGemm::CheckValidity(
                           descs[I0], descs[I1], descs[I2], block_2_etile_map))
                    {
                        d0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_container_
                            .push_back(
                                GridwiseGemm::
                                    MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                                        descs[I3]));

                        e_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_container_
                            .push_back(
                                GridwiseGemm::
                                    MakeCGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl(
                                        descs[I2]));

                        block_2_etile_map_container_.push_back(block_2_etile_map);
                    }
                }
            }
        }

        void Print()
        {
            for(size_t i = 0; i < arg.a_grid_desc_k0_m_k1_container_.size(); i++)
            {
                std::cout << "arg.a_grid_desc_k0_m_k1_container_{"
                          << arg.a_grid_desc_k0_m_k1_container_[i].GetLength(I0) << ", "
                          << arg.a_grid_desc_k0_m_k1_container_[i].GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_container_[i].GetLength(I2) << "}"
                          << std::endl;

                std::cout << "arg.b_grid_desc_k0_n_k1_container_{"
                          << arg.b_grid_desc_k0_n_k1_container_[i].GetLength(I0) << ", "
                          << arg.b_grid_desc_k0_n_k1_container_[i].GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_container_[i].GetLength(I2) << "}"
                          << std::endl;

                std::cout << "arg.e_grid_desc_m_n_container_{ "
                          << arg.e_grid_desc_m_n_container_[i].GetLength(I0) << ", "
                          << arg.e_grid_desc_m_n_container_[i].GetLength(I1) << "}" << std::endl;

                std::cout << "arg.e_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_( "
                          << arg.e_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_[i].GetLength(I0)
                          << ", "
                          << arg.e_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_[i].GetLength(I1)
                                 a_g_n_k_wos_lengths_,
                    a_g_n_k_wos_strides_, b_g_k_c_xs_lengths_, b_g_k_c_xs_strides_,
                    e_g_n_c_wis_lengths_, e_g_n_c_wis_strides_,
                    << ", " << arg.e_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_[i].GetLength(I2)
                    << ", " << arg.e_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_[i].GetLength(I3)
                    << ", " << arg.e_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_[i].GetLength(I4)
                    << ", " << arg.e_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_[i].GetLength(I5)
                    << ", " << arg.e_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_[i].GetLength(I6)
                    << ", " << arg.e_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_container_[i].GetLength(I7)
                    << " ) " << std::endl;
            }
        }

        // pointers
        const void* p_a_grid_;
        const void* p_b_grid_;
        const void* p_d0_grid_;
        void* p_e_grid_;

        // tensor descriptor for problem definition
        std::vector<AGridDesc_K0_M_K1> a_grid_desc_k0_m_k1_container_;
        std::vector<BGridDesc_K0_N_K1> b_grid_desc_k0_n_k1_container_;
        std::vector<DsGridDesc_M_N> ds_grid_desc_m_n_container_;
        std::vector<EGridDesc_M_N> e_grid_desc_m_n_container_;

        // tensor descriptor for block-wise copy
        std::vector<
            typename GridwiseGemm::
                C0GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>
            d0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_container_;

        std::vector<
            typename GridwiseGemm::
                CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>
            e_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_container_;

        // block-to-e-tile map
        std::vector<typename GridwiseGemm::DefaultBlock2CTileMap> block_2_etile_map_container_;

        // element-wise op
        AElementwiseOp a_element_op_;
        BElementwiseOp b_element_op_;
        CDEElementwiseOp cde_element_op_;

        // for checking IsSupportedArgument()
        std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_lengths_;
        std::array<index_t, NDimSpatial + 3> a_g_n_k_wos_strides_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_lengths_;
        std::array<index_t, NDimSpatial + 3> b_g_k_c_xs_strides_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_lengths_;
        std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor> ds_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_lengths_;
        std::array<index_t, NDimSpatial + 3> e_g_n_c_wis_strides_;
        std::array<index_t, NDimSpatial> conv_filter_strides_;
        std::array<index_t, NDimSpatial> conv_filter_dilations_;
        std::array<index_t, NDimSpatial> input_left_pads_;
        std::array<index_t, NDimSpatial> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
#if 0
            arg.Print();
#endif

            float ave_time = 0;
            for(size_t i = 0; i < arg.a_grid_desc_k0_m_k1_container_.size(); i++)
            {
                if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_container_[i],
                                                arg.b_grid_desc_k0_n_k1_container_[i],
                                                arg.e_grid_desc_m_n_container_[i],
                                                arg.block_2_etile_map_container_[i]))
                {
                    throw std::runtime_error(
                        "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v3r1 has invalid setting");
                }

                const index_t grid_size = arg.block_2_etile_map_container_[i].CalculateGridSize(
                    arg.e_grid_desc_m_n_container_[i]);

                const auto K = arg.a_grid_desc_k0_m_k1_container_[i].GetLength(I0) *
                               arg.a_grid_desc_k0_m_k1_container_[i].GetLength(I2);

                auto launch_kernel = [&](auto has_main_k_block_loop) {
                    constexpr bool has_main_loop = has_main_k_block_loop.value;
                    const auto kernel            = kernel_gemm_xdlops_v3r2<
                        GridwiseGemm,
                        ADataType, // TODO: distiguish A/B datatype
                        EDataType,
                        remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                        remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                        remove_reference_t<
                            typename GridwiseGemm::
                                CGridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>,
                        remove_reference_t<
                            typename GridwiseGemm::
                                C0GridDescriptor_MBlock_MXdlPerWave_MWaveMPerXdl_NBlock_NXdlPerWave_NWaveNPerXdl>,
                        AElementwiseOp,
                        BElementwiseOp,
                        CDEElementwiseOp,
                        remove_reference_t<typename GridwiseGemm::DefaultBlock2CTileMap>,
                        has_main_loop>;

                    // fix me
                    using D0DataType = tuple_element_t<0, DsDataType>;

                    return launch_and_time_kernel(
                        stream_config,
                        kernel,
                        dim3(grid_size),
                        dim3(BlockSize),
                        0,
                        static_cast<const ADataType*>(arg.p_a_grid_),
                        static_cast<const BDataType*>(arg.p_b_grid_),
                        static_cast<EDataType*>(arg.p_e_grid_),
                        static_cast<const D0DataType*>(arg.p_d0_grid_),
                        arg.a_grid_desc_k0_m_k1_container_[i],
                        arg.b_grid_desc_k0_n_k1_container_[i],
                        arg.e_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_container_
                            [i],
                        arg.d0_grid_desc_mblock_mxdlperwave_mwavemperxdl_nblock_nxdlperwave_nwavenperxdl_container_
                            [i],
                        arg.a_element_op_,
                        arg.b_element_op_,
                        arg.cde_element_op_,
                        arg.block_2_etile_map_container_[i]);
                };

                if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
                {
                    ave_time += launch_kernel(integral_constant<bool, true>{});
                }
                else
                {
                    ave_time += launch_kernel(integral_constant<bool, false>{});
                }
            }
            return ave_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        const index_t ConvK = arg.b_g_k_c_xs_lengths_[1];
        const index_t ConvC = arg.b_g_k_c_xs_lengths_[2];

        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 pad = 0 conv
            for(int i = 0; i < NDimSpatial; i++)
            {
                if(!(arg.filter_spatial_lengths_[i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 1 &&
             ConvK % ABlockTransferSrcScalarPerVector == 0 &&
             ConvC % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(ConvC % CBlockTransferScalarPerVector_NWaveNPerXdl == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        for(std::size_t i = 0; i < arg.a_grid_desc_k0_m_k1_container_.size(); i++)
        {
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_container_[i],
                                            arg.b_grid_desc_k0_n_k1_container_[i],
                                            arg.e_grid_desc_m_n_container_[i],
                                            arg.block_2_etile_map_container_[i]))
            {
                return false;
            }
        }
        return true;
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto
    MakeArgument(const void* p_a,                                                 // output image
                 const void* p_b,                                                 // weight
                 const std::array<const void*, NumDTensor>& p_ds,                 // bias
                 void* p_e,                                                       // input image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
                 const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
                 const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_lengths, // bias
                 const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
                     ds_g_n_c_wis_strides,                                        // bias
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
                 const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
                 const std::array<index_t, NDimSpatial>& conv_filter_strides,
                 const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                 const std::array<index_t, NDimSpatial>& input_left_pads,
                 const std::array<index_t, NDimSpatial>& input_right_pads,
                 const AElementwiseOp& a_element_op,
                 const BElementwiseOp& b_element_op,
                 const CDEElementwiseOp& cde_element_op)
    {
        return Argument{p_a,
                        p_b,
                        p_ds,
                        p_e,
                        a_g_n_k_wos_lengths,
                        a_g_n_k_wos_strides,
                        b_g_k_c_xs_lengths,
                        b_g_k_c_xs_strides,
                        ds_g_n_c_wis_lengths,
                        ds_g_n_c_wis_strides,
                        e_g_n_c_wis_lengths,
                        e_g_n_c_wis_strides,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        a_element_op,
                        b_element_op,
                        cde_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument> MakeArgumentPointer(
        const void* p_a,                                                 // output image
        const void* p_b,                                                 // weight
        const std::array<const void*, NumDTensor>& p_ds,                 // bias
        void* p_e,                                                       // input image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_lengths, // output image
        const std::array<index_t, NDimSpatial + 3>& a_g_n_k_wos_strides, // output image
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,  // weight
        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides,  // weight
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_lengths, // bias
        const std::array<std::array<index_t, NDimSpatial + 3>, NumDTensor>&
            ds_g_n_c_wis_strides,                                        // bias
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_lengths, // input image
        const std::array<index_t, NDimSpatial + 3>& e_g_n_c_wis_strides, // input image
        const std::array<index_t, NDimSpatial>& conv_filter_strides,
        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
        const std::array<index_t, NDimSpatial>& input_left_pads,
        const std::array<index_t, NDimSpatial>& input_right_pads,
        const AElementwiseOp& a_element_op,
        const BElementwiseOp& b_element_op,
        const CDEElementwiseOp& cde_element_op) override
    {
        return std::make_unique<Argument>(p_a,
                                          p_b,
                                          p_ds,
                                          p_e,
                                          a_g_n_k_wos_lengths,
                                          a_g_n_k_wos_strides,
                                          b_g_k_c_xs_lengths,
                                          b_g_k_c_xs_strides,
                                          ds_g_n_c_wis_lengths,
                                          ds_g_n_c_wis_strides,
                                          e_g_n_c_wis_lengths,
                                          e_g_n_c_wis_strides,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          a_element_op,
                                          b_element_op,
                                          cde_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock
            << ">";
        if constexpr(ConvBackwardDataSpecialization ==
                     ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0){

            str<< " Filter1x1Stride1Pad0";
        }


        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
