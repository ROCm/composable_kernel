#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

#include "device.hpp"
#include "device_prop.hpp"
#include "device_base.hpp"
#include "device_conv_fwd.hpp"
#include "convolution_forward_specialization.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v2r3.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

//
// @brief      Device Convolution operation.
//
// Supports:
//  @li         Inputs with up to 3 spatial dimentions
//  @li         Input tensor in NHWC data format
//  @li         Weight tensor in KYXC data format
//  @li         Output tensor in NHWK data format
//
// 1D:
// out[N, Wo, K] = in[N, Wi, C] * wei[K, X, C]
// 2D:
// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
// 3D:
// out[N, Do, Ho, Wo, K] = in[N, Di, Hi, Wi, C] * wei[K, Z, Y, X, C]
//
template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename AccDataType,
          typename InElementwiseOperation,
          typename WeiElementwiseOperation,
          typename OutElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          ck::index_t NumDimSpatial,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
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
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector>
struct DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K
    : public DeviceConvFwd<InElementwiseOperation, WeiElementwiseOperation, OutElementwiseOperation>
{
    using DeviceOp = DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K;

    using ADataType = InDataType;
    using BDataType = WeiDataType;
    using CDataType = OutDataType;

    // TODO make A/B datatype different
    using ABDataType = InDataType;

    static constexpr index_t NDimSpatial = NumDimSpatial;

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto K1Number     = Number<K1>{};
    static constexpr auto GemmK1Number = K1Number;

    static auto GetWeightTensorDescriptor(ck::index_t gemm_n, ck::index_t gemm_k)
    {
        const ck::index_t gemm_k0 = gemm_k / GemmK1Number;
        const auto wei_k_yxc_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(gemm_n, gemm_k));

        // wei_gemmk0_gemmn_gemmk1_grid_desc
        return transform_tensor_descriptor(
            wei_k_yxc_grid_desc,
            make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                       make_pass_through_transform(gemm_n)),
            make_tuple(Sequence<1>{}, Sequence<0>{}),
            make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    static auto
    GetOutputTensorDescriptor(ck::index_t gemm_m, ck::index_t gemm_n, ck::index_t gemm_m_pad)
    {
        const auto out_gemmmraw_gemmn_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(gemm_m, gemm_n));

        // out_gemmm_gemmn_grid_desc
        return transform_tensor_descriptor(out_gemmmraw_gemmn_grid_desc,
                                           make_tuple(make_right_pad_transform(gemm_m, gemm_m_pad),
                                                      make_pass_through_transform(gemm_n)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}));
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 1, bool>::type = false>
    static auto GetInputTensorDescriptor(ck::index_t N,
                                         ck::index_t C,
                                         ck::index_t gemm_m,
                                         ck::index_t gemm_k,
                                         ck::index_t gemm_m_pad,
                                         const std::vector<ck::index_t>& input_spatial_lengths,
                                         const std::vector<ck::index_t>& filter_spatial_lengths,
                                         const std::vector<ck::index_t>& output_spatial_lengths,
                                         const std::vector<ck::index_t>& conv_filter_strides,
                                         const std::vector<ck::index_t>& conv_filter_dilations,
                                         const std::vector<ck::index_t>& input_left_pads,
                                         const std::vector<ck::index_t>& input_right_pads)
    {
        const ck::index_t gemm_k0 = gemm_k / GemmK1Number;
        const index_t Wi          = input_spatial_lengths[0];
        const index_t Wo          = output_spatial_lengths[0];
        const index_t ConvStrideW = conv_filter_strides[0];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const auto in_gemmmraw_gemmk_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(gemm_m, gemm_k));

            // in_gemmk0_gemmm_gemmk1_grid_desc
            return transform_tensor_descriptor(
                in_gemmmraw_gemmk_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                           make_right_pad_transform(gemm_m, gemm_m_pad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            const auto in_n_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Wi, C));

            const auto in_n_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_gemmk0_gemmmraw_gemmk1_grid_desc = transform_tensor_descriptor(
                in_n_wo_c_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                           make_merge_transform(make_tuple(N, Wo))),
                make_tuple(Sequence<2>{}, Sequence<0, 1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // in_gemmk0_gemmm_gemmk1_grid_desc
            return transform_tensor_descriptor(
                in_gemmk0_gemmmraw_gemmk1_grid_desc,
                make_tuple(make_pass_through_transform(gemm_k0),
                           make_right_pad_transform(gemm_m, gemm_m_pad),
                           make_pass_through_transform(GemmK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
        else
        {
            const index_t X             = filter_spatial_lengths[0];
            const index_t ConvDilationW = conv_filter_dilations[0];
            const index_t InLeftPadW    = input_left_pads[0];
            const index_t InRightPadW   = input_right_pads[0];

            const auto in_n_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Wi, C));

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

            const auto in_gemmk_gemmmraw_grid_desc =
                transform_tensor_descriptor(in_n_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(X, C)),
                                                       make_merge_transform(make_tuple(N, Wo))),
                                            make_tuple(Sequence<1, 3>{}, Sequence<0, 2>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmk0_gemmmraw_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmk_gemmmraw_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                           make_pass_through_transform(gemm_m)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // in_gemmk0_gemmm_gemmk1_grid_desc
            return transform_tensor_descriptor(
                in_gemmk0_gemmmraw_gemmk1_grid_desc,
                make_tuple(make_pass_through_transform(gemm_k0),
                           make_right_pad_transform(gemm_m, gemm_m_pad),
                           make_pass_through_transform(GemmK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 2, bool>::type = false>
    static auto GetInputTensorDescriptor(ck::index_t N,
                                         ck::index_t C,
                                         ck::index_t gemm_m,
                                         ck::index_t gemm_k,
                                         ck::index_t gemm_m_pad,
                                         const std::vector<ck::index_t>& input_spatial_lengths,
                                         const std::vector<ck::index_t>& filter_spatial_lengths,
                                         const std::vector<ck::index_t>& output_spatial_lengths,
                                         const std::vector<ck::index_t>& conv_filter_strides,
                                         const std::vector<ck::index_t>& conv_filter_dilations,
                                         const std::vector<ck::index_t>& input_left_pads,
                                         const std::vector<ck::index_t>& input_right_pads)
    {
        const ck::index_t gemm_k0 = gemm_k / GemmK1Number;
        const index_t Hi          = input_spatial_lengths[0];
        const index_t Wi          = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const auto in_gemmmraw_gemmk_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(gemm_m, gemm_k));

            // in_gemmk0_gemmm_gemmk1_grid_desc
            return transform_tensor_descriptor(
                in_gemmmraw_gemmk_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                           make_right_pad_transform(gemm_m, gemm_m_pad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            const auto in_n_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

            const auto in_n_ho_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_gemmk0_gemmmraw_gemmk1_grid_desc = transform_tensor_descriptor(
                in_n_ho_wo_c_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                           make_merge_transform(make_tuple(N, Ho, Wo))),
                make_tuple(Sequence<3>{}, Sequence<0, 1, 2>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // in_gemmk0_gemmm_gemmk1_grid_desc
            return transform_tensor_descriptor(
                in_gemmk0_gemmmraw_gemmk1_grid_desc,
                make_tuple(make_pass_through_transform(gemm_k0),
                           make_right_pad_transform(gemm_m, gemm_m_pad),
                           make_pass_through_transform(GemmK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
        else
        {
            const index_t Y = filter_spatial_lengths[0];
            const index_t X = filter_spatial_lengths[1];

            const index_t ConvDilationH = conv_filter_dilations[0];
            const index_t ConvDilationW = conv_filter_dilations[1];

            const index_t InLeftPadH = input_left_pads[0];
            const index_t InLeftPadW = input_left_pads[1];

            const index_t InRightPadH = input_right_pads[0];
            const index_t InRightPadW = input_right_pads[1];

            const auto in_n_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Hi, Wi, C));

            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_grid_desc,
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

            const auto in_gemmk_gemmmraw_grid_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(Y, X, C)),
                                                       make_merge_transform(make_tuple(N, Ho, Wo))),
                                            make_tuple(Sequence<1, 3, 5>{}, Sequence<0, 2, 4>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmk0_gemmmraw_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmk_gemmmraw_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                           make_pass_through_transform(gemm_m)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // in_gemmk0_gemmm_gemmk1_grid_desc
            return transform_tensor_descriptor(
                in_gemmk0_gemmmraw_gemmk1_grid_desc,
                make_tuple(make_pass_through_transform(gemm_k0),
                           make_right_pad_transform(gemm_m, gemm_m_pad),
                           make_pass_through_transform(GemmK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 3, bool>::type = false>
    static auto GetInputTensorDescriptor(ck::index_t N,
                                         ck::index_t C,
                                         ck::index_t gemm_m,
                                         ck::index_t gemm_k,
                                         ck::index_t gemm_m_pad,
                                         const std::vector<ck::index_t>& input_spatial_lengths,
                                         const std::vector<ck::index_t>& filter_spatial_lengths,
                                         const std::vector<ck::index_t>& output_spatial_lengths,
                                         const std::vector<ck::index_t>& conv_filter_strides,
                                         const std::vector<ck::index_t>& conv_filter_dilations,
                                         const std::vector<ck::index_t>& input_left_pads,
                                         const std::vector<ck::index_t>& input_right_pads)
    {
        const ck::index_t gemm_k0 = gemm_k / GemmK1Number;
        const index_t Di          = input_spatial_lengths[0];
        const index_t Hi          = input_spatial_lengths[1];
        const index_t Wi          = input_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        const index_t ConvStrideD = conv_filter_strides[0];
        const index_t ConvStrideH = conv_filter_strides[1];
        const index_t ConvStrideW = conv_filter_strides[2];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const auto in_gemmmraw_gemmk_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(gemm_m, gemm_k));

            // in_gemmk0_gemmm_gemmk1_grid_desc
            return transform_tensor_descriptor(
                in_gemmmraw_gemmk_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                           make_right_pad_transform(gemm_m, gemm_m_pad)),
                make_tuple(Sequence<1>{}, Sequence<0>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            const auto in_n_di_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Di, Hi, Wi, C));

            const auto in_n_do_ho_wo_c_grid_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_grid_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Do), make_tuple(ConvStrideD)),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_gemmk0_gemmmraw_gemmk1_grid_desc = transform_tensor_descriptor(
                in_n_do_ho_wo_c_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                           make_merge_transform(make_tuple(N, Do, Ho, Wo))),
                make_tuple(Sequence<4>{}, Sequence<0, 1, 2, 3>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // in_gemmk0_gemmm_gemmk1_grid_desc
            return transform_tensor_descriptor(
                in_gemmk0_gemmmraw_gemmk1_grid_desc,
                make_tuple(make_pass_through_transform(gemm_k0),
                           make_right_pad_transform(gemm_m, gemm_m_pad),
                           make_pass_through_transform(GemmK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
        else
        {
            const index_t Z = filter_spatial_lengths[0];
            const index_t Y = filter_spatial_lengths[1];
            const index_t X = filter_spatial_lengths[2];

            const index_t ConvDilationD = conv_filter_dilations[0];
            const index_t ConvDilationH = conv_filter_dilations[1];
            const index_t ConvDilationW = conv_filter_dilations[2];

            const index_t InLeftPadD = input_left_pads[0];
            const index_t InLeftPadH = input_left_pads[1];
            const index_t InLeftPadW = input_left_pads[2];

            const index_t InRightPadD = input_right_pads[0];
            const index_t InRightPadH = input_right_pads[1];
            const index_t InRightPadW = input_right_pads[2];

            const auto in_n_di_hi_wi_c_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(N, Di, Hi, Wi, C));

            const auto in_n_hip_wip_c_grid_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_grid_desc,
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
                in_n_hip_wip_c_grid_desc,
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

            const auto in_gemmk_gemmmraw_grid_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(Z, Y, X, C)),
                           make_merge_transform(make_tuple(N, Do, Ho, Wo))),
                make_tuple(Sequence<1, 3, 5, 7>{}, Sequence<0, 2, 4, 6>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmk0_gemmmraw_gemmk1_grid_desc = transform_tensor_descriptor(
                in_gemmk_gemmmraw_grid_desc,
                make_tuple(make_unmerge_transform(make_tuple(gemm_k0, GemmK1Number)),
                           make_pass_through_transform(gemm_m)),
                make_tuple(Sequence<0>{}, Sequence<1>{}),
                make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

            // in_gemmk0_gemmm_gemmk1_grid_desc
            return transform_tensor_descriptor(
                in_gemmk0_gemmmraw_gemmk1_grid_desc,
                make_tuple(make_pass_through_transform(gemm_k0),
                           make_right_pad_transform(gemm_m, gemm_m_pad),
                           make_pass_through_transform(GemmK1Number)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));
        }
    }

    static index_t GetGemmMRaw(ck::index_t N,
                               const std::vector<ck::index_t>& output_spatial_lengths)
    {
        return N * std::accumulate(std::begin(output_spatial_lengths),
                                   std::end(output_spatial_lengths),
                                   1,
                                   std::multiplies<ck::index_t>());
    }

    static index_t GetGemmK(ck::index_t C, const std::vector<ck::index_t>& filter_spatial_lengths)
    {
        return C * std::accumulate(std::begin(filter_spatial_lengths),
                                   std::end(filter_spatial_lengths),
                                   1,
                                   std::multiplies<ck::index_t>());
    }

    static auto
    MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(ck::index_t N,
                                                    ck::index_t K,
                                                    ck::index_t C,
                                                    std::vector<ck::index_t> input_spatial_lengths,
                                                    std::vector<ck::index_t> filter_spatial_lengths,
                                                    std::vector<ck::index_t> output_spatial_lengths,
                                                    std::vector<ck::index_t> conv_filter_strides,
                                                    std::vector<ck::index_t> conv_filter_dilations,
                                                    std::vector<ck::index_t> input_left_pads,
                                                    std::vector<ck::index_t> input_right_pads)
    {
        using namespace ck;

        const index_t GemmMRaw = GetGemmMRaw(N, output_spatial_lengths);
        const index_t GemmN    = K;
        const index_t GemmK    = GetGemmK(C, filter_spatial_lengths);

        const auto GemmMPad = math::integer_least_multiple(GemmMRaw, MPerBlock) - GemmMRaw;

        assert(GemmK % GemmK1Number == 0);

        // C = A^T*B
        // A:
        const auto in_gemmk0_gemmm_gemmk1_grid_desc =
            GetInputTensorDescriptor<NumDimSpatial>(N,
                                                    C,
                                                    GemmMRaw,
                                                    GemmK,
                                                    GemmMPad,
                                                    input_spatial_lengths,
                                                    filter_spatial_lengths,
                                                    output_spatial_lengths,
                                                    conv_filter_strides,
                                                    conv_filter_dilations,
                                                    input_left_pads,
                                                    input_right_pads);
        // B:
        const auto wei_gemmk0_gemmn_gemmk1_grid_desc = GetWeightTensorDescriptor(GemmN, GemmK);
        // C:
        const auto out_gemmm_gemmn_grid_desc = GetOutputTensorDescriptor(GemmMRaw, GemmN, GemmMPad);

        return make_tuple(in_gemmk0_gemmm_gemmk1_grid_desc,
                          wei_gemmk0_gemmn_gemmk1_grid_desc,
                          out_gemmm_gemmn_grid_desc);
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 1, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
            1, 1, 1, {1}, {1}, {1}, {1}, {1}, {1}, {1});
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 2, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
            1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    }

    template <ck::index_t NDim, typename std::enable_if<NDim == 3, bool>::type = false>
    static auto GetABCGridDesc()
    {
        return MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
            1, 1, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    }

    using ABCGridDescs = decltype(GetABCGridDesc<NumDimSpatial>());

    using AGridDesc_K0_M_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I0])>;
    using BGridDesc_K0_N_K1 = remove_cvref_t<decltype(ABCGridDescs{}[I1])>;
    using CGridDesc_M_N     = remove_cvref_t<decltype(ABCGridDescs{}[I2])>;

    using Block2CTileMap = BlockToCTileMap_M00_N0_M01<MPerBlock, NPerBlock, CGridDesc_M_N>;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<
        BlockSize,
        ABDataType, // TODO: distinguish A/B datatype
        AccDataType,
        CDataType,
        InMemoryDataOperationEnum::Set,
        AGridDesc_K0_M_K1,
        BGridDesc_K0_N_K1,
        CGridDesc_M_N,
        InElementwiseOperation,
        WeiElementwiseOperation,
        OutElementwiseOperation,
        MPerBlock,
        NPerBlock,
        K0PerBlock,
        MPerXDL,
        NPerXDL,
        K1,
        MXdlPerWave,
        NXdlPerWave,
        ABlockTransferThreadClusterLengths_K0_M_K1,
        Sequence<1, 0, 2>, // ABlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // ABlockTransferSrcAccessOrder,
        2,                 // ABlockTransferSrcVectorDim,
        ABlockTransferSrcScalarPerVector,
        ABlockTransferDstScalarPerVector_K1,
        false, // AThreadTransferSrcResetCoordinateAfterRun,
        ABlockLdsAddExtraM,
        BBlockTransferThreadClusterLengths_K0_N_K1,
        Sequence<1, 0, 2>, // BBlockTransferThreadClusterArrangeOrder,
        Sequence<1, 0, 2>, // BBlockTransferSrcAccessOrder,
        2,                 // BBlockTransferSrcVectorDim,
        BBlockTransferSrcScalarPerVector,
        BBlockTransferDstScalarPerVector_K1,
        false, // BThreadTransferSrcResetCoordinateAfterRun,
        BBlockLdsAddExtraN,
        Sequence<2, 3, 0, 1, 7, 5, 4, 6>, // CThreadTransferSrcDstAccessOrder,
        7,                                // CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector>;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const InDataType* p_in_grid,
                 const WeiDataType* p_wei_grid,
                 OutDataType* p_out_grid,
                 ck::index_t N,
                 ck::index_t K,
                 ck::index_t C,
                 std::vector<ck::index_t> input_spatial_lengths,
                 std::vector<ck::index_t> filter_spatial_lengths,
                 std::vector<ck::index_t> output_spatial_lengths,
                 std::vector<ck::index_t> conv_filter_strides,
                 std::vector<ck::index_t> conv_filter_dilations,
                 std::vector<ck::index_t> input_left_pads,
                 std::vector<ck::index_t> input_right_pads,
                 InElementwiseOperation in_element_op,
                 WeiElementwiseOperation wei_element_op,
                 OutElementwiseOperation out_element_op)
            : p_a_grid_{p_in_grid},
              p_b_grid_{p_wei_grid},
              p_c_grid_{p_out_grid},
              a_grid_desc_k0_m_k1_{},
              b_grid_desc_k0_n_k1_{},
              c_grid_desc_m_n_{},
              c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_{},
              block_2_ctile_map_{},
              in_element_op_{in_element_op},
              wei_element_op_{wei_element_op},
              out_element_op_{out_element_op},
              Conv_N_{N},
              Conv_K_{K},
              Conv_C_{C},
              filter_spatial_lengths_{filter_spatial_lengths},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            const auto descs =
                DeviceOp::MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(N,
                                                                          K,
                                                                          C,
                                                                          input_spatial_lengths,
                                                                          filter_spatial_lengths,
                                                                          output_spatial_lengths,
                                                                          conv_filter_strides,
                                                                          conv_filter_dilations,
                                                                          input_left_pads,
                                                                          input_right_pads);

            a_grid_desc_k0_m_k1_ = descs[I0];
            b_grid_desc_k0_n_k1_ = descs[I1];
            c_grid_desc_m_n_     = descs[I2];

            block_2_ctile_map_ = Block2CTileMap{c_grid_desc_m_n_};

            if(GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_,
                                           b_grid_desc_k0_n_k1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n_);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2
            c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        Block2CTileMap block_2_ctile_map_;
        InElementwiseOperation in_element_op_;
        WeiElementwiseOperation wei_element_op_;
        OutElementwiseOperation out_element_op_;
        // for checking IsSupportedArgument()
        index_t Conv_N_;
        index_t Conv_K_;
        index_t Conv_C_;
        std::vector<index_t> filter_spatial_lengths_;
        std::vector<index_t> conv_filter_strides_;
        std::vector<index_t> input_left_pads_;
        std::vector<index_t> input_right_pads_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceOp::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
#if 0
            {
                std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{ " << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }
#endif
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                            arg.b_grid_desc_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemm_km_kn_m0m1n0n1_xdlops_v2r3 has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_);

            const auto K =
                arg.a_grid_desc_k0_m_k1_.GetLength(I0) * arg.a_grid_desc_k0_m_k1_.GetLength(I2);

            float ave_time = 0;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                const auto kernel = kernel_gemm_xdlops_v2r3<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    Block2CTileMap,
                    true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_gemm_xdlops_v2r3<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceOp::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceOp::BGridDesc_K0_N_K1>,
                    remove_reference_t<typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    InElementwiseOperation,
                    WeiElementwiseOperation,
                    OutElementwiseOperation,
                    Block2CTileMap,
                    false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.in_element_op_,
                                                  arg.wei_element_op_,
                                                  arg.out_element_op_,
                                                  arg.block_2_ctile_map_);
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
        if(ck::get_device_name() == "gfx908")
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, float> ||
                           is_same_v<AccDataType, int32_t>))
            {
                return false;
            }
        }
        else if(ck::get_device_name() == "gfx90a")
        {
            if constexpr(!(is_same_v<AccDataType, float> || is_same_v<AccDataType, float> ||
                           is_same_v<AccDataType, int32_t> || is_same_v<AccDataType, double>))
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        // Input tensors can't be bigger than 2GB each.
        constexpr ck::long_index_t GB2 = (ck::long_index_t{1} << 31);

        if(arg.a_grid_desc_k0_m_k1_.GetElementSpaceSize() * sizeof(ADataType) > GB2 ||
           arg.b_grid_desc_k0_n_k1_.GetElementSpaceSize() * sizeof(BDataType) > GB2 ||
           arg.c_grid_desc_m_n_.GetElementSpaceSize() * sizeof(CDataType) > GB2)
        {
            return false;
        }

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(ck::index_t i = 0; i < NumDimSpatial; ++i)
            {
                if(!(arg.filter_spatial_lengths_[i] == 1 && arg.conv_filter_strides_[i] == 1 &&
                     arg.input_left_pads_[i] == 0 && arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            for(ck::index_t i = 0; i < NumDimSpatial; ++i)
            {
                if(!(arg.filter_spatial_lengths_[i] == 1 && arg.input_left_pads_[i] == 0 &&
                     arg.input_right_pads_[i] == 0))
                {
                    return false;
                }
            }
        }

        // vector load A/B matrix from global memory
        if(!(ABlockTransferSrcVectorDim == 2 && BBlockTransferSrcVectorDim == 2 &&
             arg.Conv_C_ % ABlockTransferSrcScalarPerVector == 0 &&
             arg.Conv_C_ % BBlockTransferSrcScalarPerVector == 0))
        {
            return false;
        }

        // vector store C matrix into global memory
        if(!(arg.Conv_K_ % CThreadTransferDstScalarPerVector == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                           arg.b_grid_desc_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const InDataType* p_in_grid,
                             const WeiDataType* p_wei_grid,
                             OutDataType* p_out_grid,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t C,
                             std::vector<ck::index_t> input_spatial_lengths,
                             std::vector<ck::index_t> filter_spatial_lengths,
                             std::vector<ck::index_t> output_spatial_lengths,
                             std::vector<ck::index_t> conv_filter_strides,
                             std::vector<ck::index_t> conv_filter_dilations,
                             std::vector<ck::index_t> input_left_pads,
                             std::vector<ck::index_t> input_right_pads,
                             InElementwiseOperation in_element_op,
                             WeiElementwiseOperation wei_element_op,
                             OutElementwiseOperation out_element_op)
    {
        return Argument{p_in_grid,
                        p_wei_grid,
                        p_out_grid,
                        N,
                        K,
                        C,
                        input_spatial_lengths,
                        filter_spatial_lengths,
                        output_spatial_lengths,
                        conv_filter_strides,
                        conv_filter_dilations,
                        input_left_pads,
                        input_right_pads,
                        in_element_op,
                        wei_element_op,
                        out_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_in_grid,
                        const void* p_wei_grid,
                        void* p_out_grid,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t C,
                        std::vector<ck::index_t> input_spatial_lengths,
                        std::vector<ck::index_t> filter_spatial_lengths,
                        std::vector<ck::index_t> output_spatial_lengths,
                        std::vector<ck::index_t> conv_filter_strides,
                        std::vector<ck::index_t> conv_filter_dilations,
                        std::vector<ck::index_t> input_left_pads,
                        std::vector<ck::index_t> input_right_pads,
                        InElementwiseOperation in_element_op,
                        WeiElementwiseOperation wei_element_op,
                        OutElementwiseOperation out_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const InDataType*>(p_in_grid),
                                          static_cast<const WeiDataType*>(p_wei_grid),
                                          static_cast<OutDataType*>(p_out_grid),
                                          N,
                                          K,
                                          C,
                                          input_spatial_lengths,
                                          filter_spatial_lengths,
                                          output_spatial_lengths,
                                          conv_filter_strides,
                                          conv_filter_dilations,
                                          input_left_pads,
                                          input_right_pads,
                                          in_element_op,
                                          wei_element_op,
                                          out_element_op);
    }

    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceConvNDFwdXdl_Input_N_Hi_Wi_C_Weight_K_Y_X_C_Output_N_Ho_Wo_K"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock << ", "
            << getConvFwdSpecializationStr(ConvForwardSpecialization)
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
