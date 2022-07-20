// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_conv_fwd_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/matrix_padder.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/device_utility/device_prop.hpp"
#include "ck/device_utility/kernel_launch.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

namespace {

template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatDsPointer,
          typename FloatE,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          typename AGridDesc_AK0_M_AK1,
          typename BGridDesc_BK0_N_BK1,
          typename DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
          typename Block2ETileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_multiple_d_xdl_cshuffle(const FloatAB* __restrict__ p_a_grid,
                                            const FloatAB* __restrict__ p_b_grid,
                                            FloatDsPointer p_ds_grid,
                                            FloatE* __restrict__ p_e_grid,
                                            const AElementwiseOperation a_element_op,
                                            const BElementwiseOperation b_element_op,
                                            const CDEElementwiseOperation cde_element_op,
                                            const AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1,
                                            const BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1,
                                            const DsGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                                ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                            const EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
                                                e_grid_desc_mblock_mperblock_nblock_nperblock,
                                            const Block2ETileMap block_2_etile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
                                                  p_b_grid,
                                                  p_ds_grid,
                                                  p_e_grid,
                                                  p_shared,
                                                  a_element_op,
                                                  b_element_op,
                                                  cde_element_op,
                                                  a_grid_desc_ak0_m_ak1,
                                                  b_grid_desc_bk0_n_bk1,
                                                  ds_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  e_grid_desc_mblock_mperblock_nblock_nperblock,
                                                  block_2_etile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_ds_grid;
    ignore = p_e_grid;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = cde_element_op;
    ignore = a_grid_desc_ak0_m_ak1;
    ignore = b_grid_desc_bk0_n_bk1;
    ignore = ds_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = e_grid_desc_mblock_mperblock_nblock_nperblock;
    ignore = block_2_etile_map;
#endif
}
} // namespace

//
// @brief      Device Convolution operation.
//
// Supports:
//  @li         Forward convolution with up to 3 spatial dimentions
//  @li         Input tensor in NWC data format
//  @li         Weight tensor in KXC data format
//  @li         Output tensor in NWK data format
//
// 1D:
// out[N, Wo, K] = in[N, Wi, C] * wei[K, X, C]
// 2D:
// out[N, Ho, Wo, K] = in[N, Hi, Wi, C] * wei[K, Y, X, C]
// 3D:
// out[N, Do, Ho, Wo, K] = in[N, Di, Hi, Wi, C] * wei[K, Z, Y, X, C]
//
template <index_t NDimSpatial,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CShuffleDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation,
          ConvolutionForwardSpecialization ConvForwardSpecialization,
          GemmSpecialization GemmSpec,
          index_t NumGemmKPrefetchStage,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t K1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          index_t BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          index_t CDEBlockTransferScalarPerVector_NPerBlock,
          LoopScheduler LoopSched = make_default_loop_scheduler()>
struct DeviceConvFwdMultipleD_Xdl_CShuffle : public DeviceConvFwdMultipleD<NDimSpatial,
                                                                           ALayout,
                                                                           BLayout,
                                                                           DsLayout,
                                                                           ELayout,
                                                                           ADataType,
                                                                           BDataType,
                                                                           DsDataType,
                                                                           EDataType,
                                                                           AElementwiseOperation,
                                                                           BElementwiseOperation,
                                                                           CDEElementwiseOperation>
{
    using DeviceOp = DeviceConvFwdMultipleD_Xdl_CShuffle;

    static constexpr index_t NumDTensor = DsDataType::Size();

    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};

    static constexpr auto K1Number     = Number<K1>{};
    static constexpr auto GemmK1Number = K1Number;

    static constexpr auto matrix_padder =
        MatrixPadder<GemmSpec, index_t, index_t, index_t>{MPerBlock, NPerBlock, KPerBlock};

    template <
        typename BLayout_,
        typename std::enable_if<is_same_v<BLayout_, ck::tensor_layout::convolution::KXC> ||
                                    is_same_v<BLayout_, ck::tensor_layout::convolution::KYXC> ||
                                    is_same_v<BLayout_, ck::tensor_layout::convolution::KZYXC>,
                                bool>::type = false>
    static auto MakeBGridDescriptor_N_K(index_t GemmNRaw, index_t GemmKRaw)
    {
        const auto wei_k_yxc_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(GemmNRaw, GemmKRaw));

        const auto wei_gemmn_gemmk_grid_desc =
            matrix_padder.PadBDescriptor_N_K(wei_k_yxc_grid_desc);

        return wei_gemmn_gemmk_grid_desc;
    }

    template <
        typename ELayout_,
        typename std::enable_if<is_same_v<ELayout_, ck::tensor_layout::convolution::NWK> ||
                                    is_same_v<ELayout_, ck::tensor_layout::convolution::NHWK> ||
                                    is_same_v<ELayout_, ck::tensor_layout::convolution::NDHWK>,
                                bool>::type = false>
    static auto MakeEGridDescriptor_M_N(index_t GemmMRaw, index_t GemmN)
    {
        const index_t GemmM = math::integer_least_multiple(GemmMRaw, MPerBlock);

        const auto out_gemmmraw_gemmn_grid_desc =
            make_naive_tensor_descriptor_packed(make_tuple(GemmM, GemmN));

        const auto out_gemmm_gemmn_grid_desc =
            matrix_padder.PadCDescriptor_M_N(out_gemmmraw_gemmn_grid_desc);

        return out_gemmm_gemmn_grid_desc;
    }

    template <typename ALayout_,
              typename std::enable_if<is_same_v<ALayout_, ck::tensor_layout::convolution::NWC>,
                                      bool>::type = false>
    static auto MakeAGridDescriptor_M_K(index_t N,
                                        index_t C,
                                        index_t GemmMRaw,
                                        index_t GemmKRaw,
                                        const std::vector<index_t>& input_spatial_lengths,
                                        const std::vector<index_t>& filter_spatial_lengths,
                                        const std::vector<index_t>& output_spatial_lengths,
                                        const std::vector<index_t>& conv_filter_strides,
                                        const std::vector<index_t>& conv_filter_dilations,
                                        const std::vector<index_t>& input_left_pads,
                                        const std::vector<index_t>& input_right_pads)
    {
        const index_t Wi          = input_spatial_lengths[0];
        const index_t Wo          = output_spatial_lengths[0];
        const index_t ConvStrideW = conv_filter_strides[0];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const auto in_gemmmraw_gemmk_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(GemmMRaw, GemmKRaw));

            const auto in_gemmm_gemmk_grid_desc =
                matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmk_grid_desc);

            return in_gemmm_gemmk_grid_desc;
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

            const auto in_gemmmraw_gemmkraw_grid_desc = transform_tensor_descriptor(
                in_n_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, Wo)), make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmm_gemmk_grid_desc =
                matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_grid_desc);

            return in_gemmm_gemmk_grid_desc;
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

            const auto in_gemmmraw_gemmk_grid_desc =
                transform_tensor_descriptor(in_n_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Wo)),
                                                       make_merge_transform(make_tuple(X, C))),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmm_gemmk_grid_desc =
                matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmk_grid_desc);

            return in_gemmm_gemmk_grid_desc;
        }
    }

    template <typename ALayout_,
              typename std::enable_if<is_same_v<ALayout_, ck::tensor_layout::convolution::NHWC>,
                                      bool>::type = false>
    static auto MakeAGridDescriptor_M_K(index_t N,
                                        index_t C,
                                        index_t GemmMRaw,
                                        index_t GemmKRaw,
                                        const std::vector<index_t>& input_spatial_lengths,
                                        const std::vector<index_t>& filter_spatial_lengths,
                                        const std::vector<index_t>& output_spatial_lengths,
                                        const std::vector<index_t>& conv_filter_strides,
                                        const std::vector<index_t>& conv_filter_dilations,
                                        const std::vector<index_t>& input_left_pads,
                                        const std::vector<index_t>& input_right_pads)
    {
        const index_t Hi = input_spatial_lengths[0];
        const index_t Wi = input_spatial_lengths[1];

        const index_t Ho = output_spatial_lengths[0];
        const index_t Wo = output_spatial_lengths[1];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const auto in_gemmmraw_gemmkraw_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(GemmMRaw, GemmKRaw));

            const auto in_gemmm_gemmk_grid_desc =
                matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_grid_desc);

            return in_gemmm_gemmk_grid_desc;
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

            const auto in_gemmmraw_gemmk_grid_desc =
                transform_tensor_descriptor(in_n_ho_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmm_gemmk_grid_desc =
                matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmk_grid_desc);

            return in_gemmm_gemmk_grid_desc;
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

            const auto in_gemmmraw_gemmk_grid_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_grid_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                                       make_merge_transform(make_tuple(Y, X, C))),
                                            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmm_gemmk_grid_desc =
                matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmk_grid_desc);

            return in_gemmm_gemmk_grid_desc;
        }
    }

    template <typename ALayout_,
              typename std::enable_if<is_same_v<ALayout_, ck::tensor_layout::convolution::NDHWC>,
                                      bool>::type = false>
    static auto MakeAGridDescriptor_M_K(index_t N,
                                        index_t C,
                                        index_t GemmMRaw,
                                        index_t GemmKRaw,
                                        const std::vector<index_t>& input_spatial_lengths,
                                        const std::vector<index_t>& filter_spatial_lengths,
                                        const std::vector<index_t>& output_spatial_lengths,
                                        const std::vector<index_t>& conv_filter_strides,
                                        const std::vector<index_t>& conv_filter_dilations,
                                        const std::vector<index_t>& input_left_pads,
                                        const std::vector<index_t>& input_right_pads)
    {
        const index_t Di = input_spatial_lengths[0];
        const index_t Hi = input_spatial_lengths[1];
        const index_t Wi = input_spatial_lengths[2];

        const index_t Do = output_spatial_lengths[0];
        const index_t Ho = output_spatial_lengths[1];
        const index_t Wo = output_spatial_lengths[2];

        const index_t ConvStrideD = conv_filter_strides[0];
        const index_t ConvStrideH = conv_filter_strides[1];
        const index_t ConvStrideW = conv_filter_strides[2];

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const auto in_gemmmraw_gemmkraw_grid_desc =
                make_naive_tensor_descriptor_packed(make_tuple(GemmMRaw, GemmKRaw));

            const auto in_gemmm_gemmk_grid_desc =
                matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_grid_desc);

            return in_gemmm_gemmk_grid_desc;
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

            const auto in_gemmmraw_gemmkraw_grid_desc = transform_tensor_descriptor(
                in_n_do_ho_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmm_gemmk_grid_desc =
                matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_grid_desc);

            return in_gemmm_gemmk_grid_desc;
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

            const auto in_gemmmraw_gemmkraw_grid_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_grid_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                           make_merge_transform(make_tuple(Z, Y, X, C))),
                make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            const auto in_gemmm_gemmk_grid_desc =
                matrix_padder.PadADescriptor_M_K(in_gemmmraw_gemmkraw_grid_desc);

            return in_gemmm_gemmk_grid_desc;
        }
    }

    static index_t GetGemmMRaw(index_t N, const std::vector<index_t>& output_spatial_lengths)
    {
        return N * std::accumulate(std::begin(output_spatial_lengths),
                                   std::end(output_spatial_lengths),
                                   1,
                                   std::multiplies<index_t>());
    }

    static index_t GetGemmKRaw(index_t C, const std::vector<index_t>& filter_spatial_lengths)
    {
        return C * std::accumulate(std::begin(filter_spatial_lengths),
                                   std::end(filter_spatial_lengths),
                                   1,
                                   std::multiplies<index_t>());
    }

    static auto
    MakeABEGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(index_t N,
                                                    index_t K,
                                                    index_t C,
                                                    std::vector<index_t> input_spatial_lengths,
                                                    std::vector<index_t> filter_spatial_lengths,
                                                    std::vector<index_t> output_spatial_lengths,
                                                    std::vector<index_t> conv_filter_strides,
                                                    std::vector<index_t> conv_filter_dilations,
                                                    std::vector<index_t> input_left_pads,
                                                    std::vector<index_t> input_right_pads)
    {
        using namespace ck;

        const index_t GemmMRaw = GetGemmMRaw(N, output_spatial_lengths);
        const index_t GemmNRaw = K;
        const index_t GemmKRaw = GetGemmKRaw(C, filter_spatial_lengths);

        // A:
        const auto in_gemmm_gemmk_grid_desc =
            MakeAGridDescriptor_M_K<ALayout>(N,
                                             C,
                                             GemmMRaw,
                                             GemmKRaw,
                                             input_spatial_lengths,
                                             filter_spatial_lengths,
                                             output_spatial_lengths,
                                             conv_filter_strides,
                                             conv_filter_dilations,
                                             input_left_pads,
                                             input_right_pads);

        // B:
        const auto wei_gemmn_gemmk_grid_desc = MakeBGridDescriptor_N_K<BLayout>(GemmNRaw, GemmKRaw);

        // E:
        const auto out_gemmm_gemmn_grid_desc = MakeEGridDescriptor_M_N<ELayout>(GemmMRaw, GemmNRaw);

        return make_tuple(
            in_gemmm_gemmk_grid_desc, wei_gemmn_gemmk_grid_desc, out_gemmm_gemmn_grid_desc);
    }

    template <index_t NDim, typename std::enable_if<NDim == 1, bool>::type = false>
    static auto GetABEGridDesc()
    {
        return MakeABEGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
            1, 1, 1, {1}, {1}, {1}, {1}, {1}, {1}, {1});
    }

    template <index_t NDim, typename std::enable_if<NDim == 2, bool>::type = false>
    static auto GetABEGridDesc()
    {
        return MakeABEGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
            1, 1, 1, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1});
    }

    template <index_t NDim, typename std::enable_if<NDim == 3, bool>::type = false>
    static auto GetABEGridDesc()
    {
        return MakeABEGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(
            1, 1, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1});
    }

    using ABEGridDescs = decltype(GetABEGridDesc<NDimSpatial>());

    using AGridDesc_M_K = remove_cvref_t<decltype(ABEGridDescs{}[I0])>;
    using BGridDesc_N_K = remove_cvref_t<decltype(ABEGridDescs{}[I1])>;
    using EGridDesc_M_N = remove_cvref_t<decltype(ABEGridDescs{}[I2])>;

    // GridwiseGemm
    using GridwiseGemm = GridwiseGemmMultipleD_k0mk1_k0nk1_mn_xdl_cshuffle<
        ADataType, // TODO: distinguish A/B datatype
        AccDataType,
        CShuffleDataType,
        DsDataType,
        EDataType,
        AElementwiseOperation,
        BElementwiseOperation,
        CDEElementwiseOperation,
        InMemoryDataOperationEnum::Set,
        AGridDesc_M_K,
        BGridDesc_N_K,
        EGridDesc_M_N,
        NumGemmKPrefetchStage,
        BlockSize,
        MPerBlock,
        NPerBlock,
        KPerBlock,
        K1,
        K1,
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

    using AGridDesc_AK0_M_AK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(AGridDesc_M_K{}))>;
    using BGridDesc_BK0_N_BK1 = remove_cvref_t<decltype(
        GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(BGridDesc_N_K{}))>;

    using Block2ETileMap = typename GridwiseGemm::DefaultBlock2ETileMap;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_in_grid,
                 const BDataType* p_wei_grid,
                 EDataType* p_out_grid,
                 index_t N,
                 index_t K,
                 index_t C,
                 std::vector<index_t> input_spatial_lengths,
                 std::vector<index_t> filter_spatial_lengths,
                 std::vector<index_t> output_spatial_lengths,
                 std::vector<index_t> conv_filter_strides,
                 std::vector<index_t> conv_filter_dilations,
                 std::vector<index_t> input_left_pads,
                 std::vector<index_t> input_right_pads,
                 AElementwiseOperation in_element_op,
                 BElementwiseOperation wei_element_op,
                 CDEElementwiseOperation out_element_op)
            : p_a_grid_{static_cast<const ADataType*>(p_in_grid)},
              p_b_grid_{static_cast<const BDataType*>(p_wei_grid)},
              p_ds_grid_{}, // FIXME
              p_e_grid_{static_cast<EDataType*>(p_out_grid)},
              a_grid_desc_ak0_m_ak1_{},
              b_grid_desc_bk0_n_bk1_{},
              e_grid_desc_m_n_{},
              e_grid_desc_mblock_mperblock_nblock_nperblock_{},
              block_2_etile_map_{},
              a_element_op_{in_element_op},
              b_element_op_{wei_element_op},
              cde_element_op_{out_element_op},
              Conv_N_{N},
              Conv_K_{K},
              Conv_C_{C},
              filter_spatial_lengths_{filter_spatial_lengths},
              conv_filter_strides_{conv_filter_strides},
              input_left_pads_{input_left_pads},
              input_right_pads_{input_right_pads}
        {
            const auto descs =
                DeviceOp::MakeABEGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N(N,
                                                                          K,
                                                                          C,
                                                                          input_spatial_lengths,
                                                                          filter_spatial_lengths,
                                                                          output_spatial_lengths,
                                                                          conv_filter_strides,
                                                                          conv_filter_dilations,
                                                                          input_left_pads,
                                                                          input_right_pads);

            const auto a_grid_desc_m_k = descs[I0];
            const auto b_grid_desc_n_k = descs[I1];
            e_grid_desc_m_n_           = descs[I2];

            a_grid_desc_ak0_m_ak1_ =
                GridwiseGemm::MakeDefaultAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k);
            b_grid_desc_bk0_n_bk1_ =
                GridwiseGemm::MakeDefaultBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k);

            block_2_etile_map_ = Block2ETileMap{e_grid_desc_m_n_};

            if(GridwiseGemm::CheckValidity(a_grid_desc_ak0_m_ak1_,
                                           b_grid_desc_bk0_n_bk1_,
                                           e_grid_desc_m_n_,
                                           block_2_etile_map_))
            {
                e_grid_desc_mblock_mperblock_nblock_nperblock_ =
                    GridwiseGemm::MakeEGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock(
                        e_grid_desc_m_n_);
            }
        }

        //  private:
        // pointers
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        typename GridwiseGemm::DsGridPointer p_ds_grid_;
        EDataType* p_e_grid_;

        // tensor descriptors
        AGridDesc_AK0_M_AK1 a_grid_desc_ak0_m_ak1_;
        BGridDesc_BK0_N_BK1 b_grid_desc_bk0_n_bk1_;
        StaticallyIndexedArray<
            typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
            NumDTensor>
            ds_grid_desc_mblock_mperblock_nblock_nperblock_; // FIXME: Ds desc may be of different
                                                             // type from E
        EGridDesc_M_N e_grid_desc_m_n_;
        typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock
            e_grid_desc_mblock_mperblock_nblock_nperblock_;

        // block-to-e-tile map
        Block2ETileMap block_2_etile_map_;

        // element-wise op
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CDEElementwiseOperation cde_element_op_;

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
#if 1
            {
                std::cout << "arg.a_grid_desc_ak0_m_ak1_{"
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_ak0_m_ak1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_bk0_n_bk1_{"
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I0) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_bk0_n_bk1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.e_grid_desc_m_n_{ " << arg.e_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.e_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }
#endif
            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                            arg.b_grid_desc_bk0_n_bk1_,
                                            arg.e_grid_desc_m_n_,
                                            arg.block_2_etile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseGemmMultipleD_k0mk1_k0nk1_mn_xdl_cshuffle has invalid setting");
            }

            const index_t grid_size =
                arg.block_2_etile_map_.CalculateGridSize(arg.e_grid_desc_m_n_);

            const auto K =
                arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) * arg.a_grid_desc_ak0_m_ak1_.GetLength(I2);

            auto launch_kernel = [&](auto has_main_k_block_loop) {
                constexpr bool has_main_loop = has_main_k_block_loop.value;

                const auto kernel = kernel_gemm_multiple_d_xdl_cshuffle<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    typename GridwiseGemm::DsGridPointer,
                    EDataType,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CDEElementwiseOperation,
                    DeviceOp::AGridDesc_AK0_M_AK1,
                    DeviceOp::BGridDesc_BK0_N_BK1,
                    ck::StaticallyIndexedArray<
                        typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                        NumDTensor>,
                    typename GridwiseGemm::EGridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
                    Block2ETileMap,
                    has_main_loop>;

                return launch_and_time_kernel(stream_config,
                                              kernel,
                                              dim3(grid_size),
                                              dim3(BlockSize),
                                              0,
                                              arg.p_a_grid_,
                                              arg.p_b_grid_,
                                              arg.p_ds_grid_,
                                              arg.p_e_grid_,
                                              arg.a_element_op_,
                                              arg.b_element_op_,
                                              arg.cde_element_op_,
                                              arg.a_grid_desc_ak0_m_ak1_,
                                              arg.b_grid_desc_bk0_n_bk1_,
                                              arg.ds_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.e_grid_desc_mblock_mperblock_nblock_nperblock_,
                                              arg.block_2_etile_map_);
            };

            float avg_time = 0;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                avg_time = launch_kernel(integral_constant<bool, true>{});
            }
            else
            {
                avg_time = launch_kernel(integral_constant<bool, false>{});
            }

            return avg_time;
        }

        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static bool IsSupportedArgument(const Argument& arg)
    {
#if 1
        {
            std::cout << "arg.a_grid_desc_ak0_m_ak1_{" << arg.a_grid_desc_ak0_m_ak1_.GetLength(I0)
                      << ", " << arg.a_grid_desc_ak0_m_ak1_.GetLength(I1) << ", "
                      << arg.a_grid_desc_ak0_m_ak1_.GetLength(I2) << "}" << std::endl;

            std::cout << "arg.b_grid_desc_bk0_n_bk1_{" << arg.b_grid_desc_bk0_n_bk1_.GetLength(I0)
                      << ", " << arg.b_grid_desc_bk0_n_bk1_.GetLength(I1) << ", "
                      << arg.b_grid_desc_bk0_n_bk1_.GetLength(I2) << "}" << std::endl;

            std::cout << "arg.e_grid_desc_m_n_{ " << arg.e_grid_desc_m_n_.GetLength(I0) << ", "
                      << arg.e_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
        }
#endif
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

        // tensors can't be bigger than 2GB each.
        constexpr ck::long_index_t GB2 = (ck::long_index_t{1} << 31);

        if(arg.a_grid_desc_ak0_m_ak1_.GetElementSpaceSize() * sizeof(ADataType) > GB2 ||
           arg.b_grid_desc_bk0_n_bk1_.GetElementSpaceSize() * sizeof(BDataType) > GB2 ||
           arg.e_grid_desc_m_n_.GetElementSpaceSize() * sizeof(EDataType) > GB2)
        {
            return false;
        }

        if constexpr(ConvForwardSpecialization ==
                     ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
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
            for(index_t i = 0; i < NDimSpatial; ++i)
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

        // vector store D/E matrix into global memory
        if(!(arg.Conv_K_ % CDEBlockTransferScalarPerVector_NPerBlock == 0))
        {
            return false;
        }

        // Gridwise GEMM size
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_ak0_m_ak1_,
                                           arg.b_grid_desc_bk0_n_bk1_,
                                           arg.e_grid_desc_m_n_,
                                           arg.block_2_etile_map_);
    }

    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_in_grid,
                             const BDataType* p_wei_grid,
                             EDataType* p_out_grid,
                             index_t N,
                             index_t K,
                             index_t C,
                             std::vector<index_t> input_spatial_lengths,
                             std::vector<index_t> filter_spatial_lengths,
                             std::vector<index_t> output_spatial_lengths,
                             std::vector<index_t> conv_filter_strides,
                             std::vector<index_t> conv_filter_dilations,
                             std::vector<index_t> input_left_pads,
                             std::vector<index_t> input_right_pads,
                             AElementwiseOperation in_element_op,
                             BElementwiseOperation wei_element_op,
                             CDEElementwiseOperation out_element_op)
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
    MakeArgumentPointer(const ADataType* p_in_grid,
                        const BDataType* p_wei_grid,
                        EDataType* p_out_grid,
                        index_t N,
                        index_t K,
                        index_t C,
                        std::vector<index_t> input_spatial_lengths,
                        std::vector<index_t> filter_spatial_lengths,
                        std::vector<index_t> output_spatial_lengths,
                        std::vector<index_t> conv_filter_strides,
                        std::vector<index_t> conv_filter_dilations,
                        std::vector<index_t> input_left_pads,
                        std::vector<index_t> input_right_pads,
                        AElementwiseOperation in_element_op,
                        BElementwiseOperation wei_element_op,
                        CDEElementwiseOperation out_element_op) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_in_grid),
                                          static_cast<const BDataType*>(p_wei_grid),
                                          static_cast<EDataType*>(p_out_grid),
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
        str << "DeviceConvFwdMultipleD_Xdl_CShuffle"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << KPerBlock << ", "
            << getConvForwardSpecializationString(ConvForwardSpecialization)
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
