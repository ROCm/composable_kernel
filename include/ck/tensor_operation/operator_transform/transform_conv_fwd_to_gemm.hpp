
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/library/utility/numeric.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"

namespace ck {
namespace tensor_operation {

// function to be used on device, emulates std::accumulate
template <typename T, typename ForwardIterator, typename Size>
__host__ __device__ auto mult_accumulate_n(ForwardIterator first, Size count, T init)
{
    for(ForwardIterator x = first; x != first + count; x++)
    {
        init *= *x;
    }
    return init;
}

template <index_t NDimSpatial, device::ConvolutionForwardSpecialization ConvForwardSpecialization>
struct TransformConvFwdToGemm
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    static long_index_t
    calculate_element_space_size_impl(const std::array<index_t, NDimSpatial + 3>& lengths,
                                      const std::array<index_t, NDimSpatial + 3>& strides,
                                      index_t i)
    {
        long_index_t acc = 1;
        for(; i < (NDimSpatial + 3); i++)
        {
            acc +=
                static_cast<long_index_t>(lengths[i] - I1) * static_cast<long_index_t>(strides[i]);
        }

        return acc;
    }

    template <typename ADataType, typename CDataType>
    static index_t GetSplitedNSize(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                                   const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                                   const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                                   const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_strides)
    {
        const long_index_t a_element_space_size =
            calculate_element_space_size_impl(a_g_n_c_wis_lengths, a_g_n_c_wis_strides, I1);
        const long_index_t c_element_space_size =
            calculate_element_space_size_impl(c_g_n_k_wos_lengths, c_g_n_k_wos_strides, I1);
        const long_index_t element_space_size = math::max(a_element_space_size * sizeof(ADataType),
                                                          c_element_space_size * sizeof(CDataType));
        constexpr long_index_t TwoGB          = (long_index_t{1} << 31);

        const index_t N = a_g_n_c_wis_lengths[I1];

        if(element_space_size > TwoGB)
        {
            // Minimum divisor of N to not exceed 2GB
            const auto divisor = math::integer_divide_ceil(element_space_size, TwoGB);

            if(divisor <= static_cast<double>(N))
            {
                // Find least divisor of N larger than element_space_size / TwoGB
                // Iterate up to sqrt(N). There are no divisors above this value.
                for(index_t least_divisor = divisor; least_divisor * least_divisor <= N;
                    least_divisor++)
                {
                    if(N % least_divisor == 0)
                    {
                        return N / least_divisor;
                    }
                }
                // Not found, process one Convolution N per block
                return 1;
            }
            else
            {
                // Not possible to support even after split N.
                // Too large tensor.
                return N;
            }
        }
        else
        {
            // Split N is not needed.
            return N;
        }
    }

    // TODO: implement ck::tensor_layout::convolution that describe packed/strided dimemsion as
    // properties
    template <typename ALayout,
              typename std::enable_if<NDimSpatial == 1 &&
                                          (is_same_v<ALayout, tensor_layout::convolution::G_NW_C> ||
                                           is_same_v<ALayout, tensor_layout::convolution::NWGC> ||
                                           is_same_v<ALayout, tensor_layout::convolution::GNWC>),
                                      bool>::type = false>
    static auto
    MakeADescriptor_M_K(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                        const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        const index_t N)
    {
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Wi = a_g_n_c_wis_lengths[3];

        const index_t Wo = c_g_n_k_wos_lengths[3];

        const index_t ConvStrideW = conv_filter_strides[0];

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const index_t NHoWo =
                N * ck::accumulate_n<index_t>(
                        c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

            // This is different
            const index_t WiStride = a_g_n_c_wis_strides[2 + NDimSpatial];
            const auto CStride     = I1;

            const auto in_gemmm_gemmk_desc =
                make_naive_tensor_descriptor(make_tuple(NHoWo, C), make_tuple(WiStride, CStride));

            return in_gemmm_gemmk_desc;
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t WiStride = a_g_n_c_wis_strides[3];
            const auto CStride     = I1;

            const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Wi, C), make_tuple(NStride, WiStride, CStride));

            const auto in_n_wo_c_desc = transform_tensor_descriptor(
                in_n_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_gemmm_gemmk_desc = transform_tensor_descriptor(
                in_n_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Wo)), make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
        else
        {
            const index_t X             = b_g_k_c_xs_lengths[3];
            const index_t ConvDilationW = conv_filter_dilations[0];
            const index_t InLeftPadW    = input_left_pads[0];
            const index_t InRightPadW   = input_right_pads[0];

            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t WiStride = a_g_n_c_wis_strides[3];
            const auto CStride     = I1;

            const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Wi, C), make_tuple(NStride, WiStride, CStride));

            const auto in_n_wip_c_desc = transform_tensor_descriptor(
                in_n_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                in_n_wip_c_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto in_gemmm_gemmk_desc =
                transform_tensor_descriptor(in_n_x_wo_c_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Wo)),
                                                       make_merge_transform(make_tuple(X, C))),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 2 && (is_same_v<ALayout, tensor_layout::convolution::G_NHW_C> ||
                                       is_same_v<ALayout, tensor_layout::convolution::NHWGC> ||
                                       is_same_v<ALayout, tensor_layout::convolution::GNHWC>),
                  bool>::type = false>
    static auto
    MakeADescriptor_M_K(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                        const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        const index_t N)

    {
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Hi = a_g_n_c_wis_lengths[3];
        const index_t Wi = a_g_n_c_wis_lengths[4];

        const index_t Ho = c_g_n_k_wos_lengths[3];
        const index_t Wo = c_g_n_k_wos_lengths[4];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const index_t NHoWo =
                N * ck::accumulate_n<index_t>(
                        c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

            // This is different
            const index_t WiStride = a_g_n_c_wis_strides[2 + NDimSpatial];
            const auto CStride     = I1;

            const auto in_gemmm_gemmk_desc =
                make_naive_tensor_descriptor(make_tuple(NHoWo, C), make_tuple(WiStride, CStride));

            return in_gemmm_gemmk_desc;
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t HiStride = a_g_n_c_wis_strides[3];
            const index_t WiStride = a_g_n_c_wis_strides[4];
            const auto CStride     = I1;

            const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Hi, Wi, C), make_tuple(NStride, HiStride, WiStride, CStride));

            const auto in_n_ho_wo_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_gemmm_gemmk_desc =
                transform_tensor_descriptor(in_n_ho_wo_c_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
        else
        {
            const index_t Y = b_g_k_c_xs_lengths[3];
            const index_t X = b_g_k_c_xs_lengths[4];

            const index_t ConvDilationH = conv_filter_dilations[0];
            const index_t ConvDilationW = conv_filter_dilations[1];

            const index_t InLeftPadH = input_left_pads[0];
            const index_t InLeftPadW = input_left_pads[1];

            const index_t InRightPadH = input_right_pads[0];
            const index_t InRightPadW = input_right_pads[1];

            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t HiStride = a_g_n_c_wis_strides[3];
            const index_t WiStride = a_g_n_c_wis_strides[4];
            const auto CStride     = I1;

            const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Hi, Wi, C), make_tuple(NStride, HiStride, WiStride, CStride));

            const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemmm_gemmk_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                                       make_merge_transform(make_tuple(Y, X, C))),
                                            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 3 && (is_same_v<ALayout, tensor_layout::convolution::G_NDHW_C> ||
                                       is_same_v<ALayout, tensor_layout::convolution::NDHWGC> ||
                                       is_same_v<ALayout, tensor_layout::convolution::GNDHWC>),
                  bool>::type = false>
    static auto
    MakeADescriptor_M_K(const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const std::array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                        const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides*/,
                        const std::array<index_t, NDimSpatial>& conv_filter_strides,
                        const std::array<index_t, NDimSpatial>& conv_filter_dilations,
                        const std::array<index_t, NDimSpatial>& input_left_pads,
                        const std::array<index_t, NDimSpatial>& input_right_pads,
                        const index_t N)

    {
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Di = a_g_n_c_wis_lengths[3];
        const index_t Hi = a_g_n_c_wis_lengths[4];
        const index_t Wi = a_g_n_c_wis_lengths[5];

        const index_t Do = c_g_n_k_wos_lengths[3];
        const index_t Ho = c_g_n_k_wos_lengths[4];
        const index_t Wo = c_g_n_k_wos_lengths[5];

        const index_t ConvStrideD = conv_filter_strides[0];
        const index_t ConvStrideH = conv_filter_strides[1];
        const index_t ConvStrideW = conv_filter_strides[2];

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const index_t NDoHoWo =
                N * ck::accumulate_n<index_t>(
                        c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

            // This is different
            const index_t WiStride = a_g_n_c_wis_strides[2 + NDimSpatial];
            const auto CStride     = I1;

            const auto in_gemmm_gemmk_desc =
                make_naive_tensor_descriptor(make_tuple(NDoHoWo, C), make_tuple(WiStride, CStride));

            return in_gemmm_gemmk_desc;
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t DiStride = a_g_n_c_wis_strides[3];
            const index_t HiStride = a_g_n_c_wis_strides[4];
            const index_t WiStride = a_g_n_c_wis_strides[5];
            const auto CStride     = I1;

            const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, C),
                make_tuple(NStride, DiStride, HiStride, WiStride, CStride));

            const auto in_n_do_ho_wo_c_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Do), make_tuple(ConvStrideD)),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_gemmm_gemmk_desc = transform_tensor_descriptor(
                in_n_do_ho_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
        else
        {
            const index_t Z = b_g_k_c_xs_lengths[3];
            const index_t Y = b_g_k_c_xs_lengths[4];
            const index_t X = b_g_k_c_xs_lengths[5];

            const index_t ConvDilationD = conv_filter_dilations[0];
            const index_t ConvDilationH = conv_filter_dilations[1];
            const index_t ConvDilationW = conv_filter_dilations[2];

            const index_t InLeftPadD = input_left_pads[0];
            const index_t InLeftPadH = input_left_pads[1];
            const index_t InLeftPadW = input_left_pads[2];

            const index_t InRightPadD = input_right_pads[0];
            const index_t InRightPadH = input_right_pads[1];
            const index_t InRightPadW = input_right_pads[2];

            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t DiStride = a_g_n_c_wis_strides[3];
            const index_t HiStride = a_g_n_c_wis_strides[4];
            const index_t WiStride = a_g_n_c_wis_strides[5];
            const auto CStride     = I1;

            const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, C),
                make_tuple(NStride, DiStride, HiStride, WiStride, CStride));

            const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Di, InLeftPadD, InRightPadD),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_desc,
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

            const auto in_gemmm_gemmk_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                           make_merge_transform(make_tuple(Z, Y, X, C))),
                make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
    }

    template <typename BLayout,
              typename std::enable_if<is_same_v<BLayout, tensor_layout::convolution::GKXC> ||
                                          is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                                          is_same_v<BLayout, tensor_layout::convolution::GKZYXC>,
                                      bool>::type = false>
    static auto
    MakeBDescriptor_N_K(const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */)
    {
        const index_t K = b_g_k_c_xs_lengths[1];
        const index_t C = b_g_k_c_xs_lengths[2];

        const index_t YX = ck::accumulate_n<index_t>(
            b_g_k_c_xs_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

        const auto wei_gemmn_gemmk_desc =
            make_naive_tensor_descriptor_packed(make_tuple(K, YX * C));

        return wei_gemmn_gemmk_desc;
    }

    template <
        typename BLayout,
        typename std::enable_if<is_same_v<BLayout, tensor_layout::convolution::G_K_X_C> ||
                                    is_same_v<BLayout, tensor_layout::convolution::G_K_YX_C> ||
                                    is_same_v<BLayout, tensor_layout::convolution::G_K_ZYX_C> ||
                                    is_same_v<BLayout, tensor_layout::convolution::KXGC> ||
                                    is_same_v<BLayout, tensor_layout::convolution::KYXGC> ||
                                    is_same_v<BLayout, tensor_layout::convolution::KZYXGC>,
                                bool>::type = false>
    static auto MakeBDescriptor_N_K(const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                                    const std::array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides)
    {
        const index_t K = b_g_k_c_xs_lengths[1];
        const index_t C = b_g_k_c_xs_lengths[2];

        const index_t YX = ck::accumulate_n<index_t>(
            b_g_k_c_xs_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

        const index_t KStride = b_g_k_c_xs_strides[1];
        const index_t XStride = b_g_k_c_xs_strides[2 + NDimSpatial];
        const auto CStride    = I1;

        const auto wei_k_yx_c_desc = make_naive_tensor_descriptor(
            make_tuple(K, YX, C), make_tuple(KStride, XStride, CStride));

        const auto wei_gemmn_gemmk_desc = transform_tensor_descriptor(
            wei_k_yx_c_desc,
            make_tuple(make_pass_through_transform(K), make_merge_transform(make_tuple(YX, C))),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return wei_gemmn_gemmk_desc;
    }

    template <typename CLayout,
              typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::GNWK> ||
                                          is_same_v<CLayout, tensor_layout::convolution::GNHWK> ||
                                          is_same_v<CLayout, tensor_layout::convolution::GNDHWK>,
                                      bool>::type = false>
    static auto
    MakeCDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const std::array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                        const index_t N)
    {
        const index_t K = c_g_n_k_wos_lengths[2];

        const index_t NHoWo =
            N * ck::accumulate_n<index_t>(
                    c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

        const auto out_gemmm_gemmn_desc = make_naive_tensor_descriptor_packed(make_tuple(NHoWo, K));

        return out_gemmm_gemmn_desc;
    }

    template <
        typename CLayout,
        typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::G_NW_K> ||
                                    is_same_v<CLayout, tensor_layout::convolution::G_NHW_K> ||
                                    is_same_v<CLayout, tensor_layout::convolution::G_NDHW_K> ||
                                    is_same_v<CLayout, tensor_layout::convolution::NWGK> ||
                                    is_same_v<CLayout, tensor_layout::convolution::NHWGK> ||
                                    is_same_v<CLayout, tensor_layout::convolution::NDHWGK>,
                                bool>::type = false>
    static auto MakeCDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                                    const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_strides,
                                    const index_t N)
    {
        const index_t K = c_g_n_k_wos_lengths[2];

        const auto KStride     = I1;
        const index_t WoStride = c_g_n_k_wos_strides[NDimSpatial + 2];

        const index_t NHoWo =
            N * ck::accumulate_n<index_t>(
                    c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

        const auto out_gemmm_gemmn_desc =
            make_naive_tensor_descriptor(make_tuple(NHoWo, K), make_tuple(WoStride, KStride));

        return out_gemmm_gemmn_desc;
    }

    // for output bias
    template <typename CLayout,
              typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::G_K>,
                                      bool>::type = false>
    static auto MakeCDescriptor_M_N(const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                                    const std::array<index_t, NDimSpatial + 3>& c_g_n_k_wos_strides,
                                    const index_t N)
    {
        const index_t K       = c_g_n_k_wos_lengths[2];
        const index_t KStride = c_g_n_k_wos_strides[2];

        const index_t NHoWo =
            N * ck::accumulate_n<index_t>(
                    c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

        const auto out_gemmm_gemmn_desc =
            make_naive_tensor_descriptor(make_tuple(NHoWo, K), make_tuple(I0, KStride));

        return out_gemmm_gemmn_desc;
    }

    // Overloaded functions for hipRTC purposes
    template <typename ALayout,
              typename std::enable_if<NDimSpatial == 1 &&
                                          (is_same_v<ALayout, tensor_layout::convolution::G_NW_C> ||
                                           is_same_v<ALayout, tensor_layout::convolution::NWGC> ||
                                           is_same_v<ALayout, tensor_layout::convolution::GNWC>),
                                      bool>::type = false>
    __host__ __device__ static auto
    MakeADescriptor_M_K(const ck::Array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const ck::Array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                        const ck::Array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                        const ck::Array<index_t, NDimSpatial>& conv_filter_strides,
                        const ck::Array<index_t, NDimSpatial>& conv_filter_dilations,
                        const ck::Array<index_t, NDimSpatial>& input_left_pads,
                        const ck::Array<index_t, NDimSpatial>& input_right_pads)
    {
        const index_t N = a_g_n_c_wis_lengths[1];
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Wi = a_g_n_c_wis_lengths[3];

        const index_t Wo = c_g_n_k_wos_lengths[3];

        const index_t ConvStrideW = conv_filter_strides[0];

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const index_t NHoWo =
                N * ck::accumulate_n<index_t>(
                        c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

            // This is different
            const index_t WiStride = a_g_n_c_wis_strides[2 + NDimSpatial];
            const auto CStride     = I1;

            const auto in_gemmm_gemmk_desc =
                make_naive_tensor_descriptor(make_tuple(NHoWo, C), make_tuple(WiStride, CStride));

            return in_gemmm_gemmk_desc;
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t WiStride = a_g_n_c_wis_strides[3];
            const auto CStride     = I1;

            const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Wi, C), make_tuple(NStride, WiStride, CStride));

            const auto in_n_wo_c_desc = transform_tensor_descriptor(
                in_n_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_gemmm_gemmk_desc = transform_tensor_descriptor(
                in_n_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Wo)), make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
        else
        {
            const index_t X             = b_g_k_c_xs_lengths[3];
            const index_t ConvDilationW = conv_filter_dilations[0];
            const index_t InLeftPadW    = input_left_pads[0];
            const index_t InRightPadW   = input_right_pads[0];

            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t WiStride = a_g_n_c_wis_strides[3];
            const auto CStride     = I1;

            const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Wi, C), make_tuple(NStride, WiStride, CStride));

            const auto in_n_wip_c_desc = transform_tensor_descriptor(
                in_n_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

            const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                in_n_wip_c_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

            const auto in_gemmm_gemmk_desc =
                transform_tensor_descriptor(in_n_x_wo_c_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Wo)),
                                                       make_merge_transform(make_tuple(X, C))),
                                            make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 2 && (is_same_v<ALayout, tensor_layout::convolution::G_NHW_C> ||
                                       is_same_v<ALayout, tensor_layout::convolution::NHWGC> ||
                                       is_same_v<ALayout, tensor_layout::convolution::GNHWC>),
                  bool>::type = false>
    __host__ __device__ static auto
    MakeADescriptor_M_K(const ck::Array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const ck::Array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                        const ck::Array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                        const ck::Array<index_t, NDimSpatial>& conv_filter_strides,
                        const ck::Array<index_t, NDimSpatial>& conv_filter_dilations,
                        const ck::Array<index_t, NDimSpatial>& input_left_pads,
                        const ck::Array<index_t, NDimSpatial>& input_right_pads)
    {
        const index_t N = a_g_n_c_wis_lengths[1];
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Hi = a_g_n_c_wis_lengths[3];
        const index_t Wi = a_g_n_c_wis_lengths[4];

        const index_t Ho = c_g_n_k_wos_lengths[3];
        const index_t Wo = c_g_n_k_wos_lengths[4];

        const index_t ConvStrideH = conv_filter_strides[0];
        const index_t ConvStrideW = conv_filter_strides[1];

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const index_t NHoWo =
                N * ck::accumulate_n<index_t>(
                        c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

            // This is different
            const index_t WiStride = a_g_n_c_wis_strides[2 + NDimSpatial];
            const auto CStride     = I1;

            const auto in_gemmm_gemmk_desc =
                make_naive_tensor_descriptor(make_tuple(NHoWo, C), make_tuple(WiStride, CStride));

            return in_gemmm_gemmk_desc;
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t HiStride = a_g_n_c_wis_strides[3];
            const index_t WiStride = a_g_n_c_wis_strides[4];
            const auto CStride     = I1;

            const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Hi, Wi, C), make_tuple(NStride, HiStride, WiStride, CStride));

            const auto in_n_ho_wo_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_gemmm_gemmk_desc =
                transform_tensor_descriptor(in_n_ho_wo_c_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                                       make_pass_through_transform(C)),
                                            make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
        else
        {
            const index_t Y = b_g_k_c_xs_lengths[3];
            const index_t X = b_g_k_c_xs_lengths[4];

            const index_t ConvDilationH = conv_filter_dilations[0];
            const index_t ConvDilationW = conv_filter_dilations[1];

            const index_t InLeftPadH = input_left_pads[0];
            const index_t InLeftPadW = input_left_pads[1];

            const index_t InRightPadH = input_right_pads[0];
            const index_t InRightPadW = input_right_pads[1];

            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t HiStride = a_g_n_c_wis_strides[3];
            const index_t WiStride = a_g_n_c_wis_strides[4];
            const auto CStride     = I1;

            const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Hi, Wi, C), make_tuple(NStride, HiStride, WiStride, CStride));

            const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                in_n_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

            const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_desc,
                make_tuple(
                    make_pass_through_transform(N),
                    make_embed_transform(make_tuple(Y, Ho), make_tuple(ConvDilationH, ConvStrideH)),
                    make_embed_transform(make_tuple(X, Wo), make_tuple(ConvDilationW, ConvStrideW)),
                    make_pass_through_transform(C)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

            const auto in_gemmm_gemmk_desc =
                transform_tensor_descriptor(in_n_y_ho_x_wo_c_desc,
                                            make_tuple(make_merge_transform(make_tuple(N, Ho, Wo)),
                                                       make_merge_transform(make_tuple(Y, X, C))),
                                            make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                                            make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 3 && (is_same_v<ALayout, tensor_layout::convolution::G_NDHW_C> ||
                                       is_same_v<ALayout, tensor_layout::convolution::NDHWGC> ||
                                       is_same_v<ALayout, tensor_layout::convolution::GNDHWC>),
                  bool>::type = false>
    static auto
    MakeADescriptor_M_K(const ck::Array<index_t, NDimSpatial + 3>& a_g_n_c_wis_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& a_g_n_c_wis_strides,
                        const ck::Array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */,
                        const ck::Array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */,
                        const ck::Array<index_t, NDimSpatial>& conv_filter_strides,
                        const ck::Array<index_t, NDimSpatial>& conv_filter_dilations,
                        const ck::Array<index_t, NDimSpatial>& input_left_pads,
                        const ck::Array<index_t, NDimSpatial>& input_right_pads)
    {
        const index_t N = a_g_n_c_wis_lengths[1];
        const index_t C = a_g_n_c_wis_lengths[2];

        const index_t Di = a_g_n_c_wis_lengths[3];
        const index_t Hi = a_g_n_c_wis_lengths[4];
        const index_t Wi = a_g_n_c_wis_lengths[5];

        const index_t Do = c_g_n_k_wos_lengths[3];
        const index_t Ho = c_g_n_k_wos_lengths[4];
        const index_t Wo = c_g_n_k_wos_lengths[5];

        const index_t ConvStrideD = conv_filter_strides[0];
        const index_t ConvStrideH = conv_filter_strides[1];
        const index_t ConvStrideW = conv_filter_strides[2];

        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            const index_t NDoHoWo =
                N * ck::accumulate_n<index_t>(
                        c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1, std::multiplies<>());

            // This is different
            const index_t WiStride = a_g_n_c_wis_strides[2 + NDimSpatial];
            const auto CStride     = I1;

            const auto in_gemmm_gemmk_desc =
                make_naive_tensor_descriptor(make_tuple(NDoHoWo, C), make_tuple(WiStride, CStride));

            return in_gemmm_gemmk_desc;
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t DiStride = a_g_n_c_wis_strides[3];
            const index_t HiStride = a_g_n_c_wis_strides[4];
            const index_t WiStride = a_g_n_c_wis_strides[5];
            const auto CStride     = I1;

            const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, C),
                make_tuple(NStride, DiStride, HiStride, WiStride, CStride));

            const auto in_n_do_ho_wo_c_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_embed_transform(make_tuple(Do), make_tuple(ConvStrideD)),
                           make_embed_transform(make_tuple(Ho), make_tuple(ConvStrideH)),
                           make_embed_transform(make_tuple(Wo), make_tuple(ConvStrideW)),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_gemmm_gemmk_desc = transform_tensor_descriptor(
                in_n_do_ho_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                           make_pass_through_transform(C)),
                make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
        else
        {
            const index_t Z = b_g_k_c_xs_lengths[3];
            const index_t Y = b_g_k_c_xs_lengths[4];
            const index_t X = b_g_k_c_xs_lengths[5];

            const index_t ConvDilationD = conv_filter_dilations[0];
            const index_t ConvDilationH = conv_filter_dilations[1];
            const index_t ConvDilationW = conv_filter_dilations[2];

            const index_t InLeftPadD = input_left_pads[0];
            const index_t InLeftPadH = input_left_pads[1];
            const index_t InLeftPadW = input_left_pads[2];

            const index_t InRightPadD = input_right_pads[0];
            const index_t InRightPadH = input_right_pads[1];
            const index_t InRightPadW = input_right_pads[2];

            // This is different
            const index_t NStride  = a_g_n_c_wis_strides[1];
            const index_t DiStride = a_g_n_c_wis_strides[3];
            const index_t HiStride = a_g_n_c_wis_strides[4];
            const index_t WiStride = a_g_n_c_wis_strides[5];
            const auto CStride     = I1;

            const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                make_tuple(N, Di, Hi, Wi, C),
                make_tuple(NStride, DiStride, HiStride, WiStride, CStride));

            const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                in_n_di_hi_wi_c_desc,
                make_tuple(make_pass_through_transform(N),
                           make_pad_transform(Di, InLeftPadD, InRightPadD),
                           make_pad_transform(Hi, InLeftPadH, InRightPadH),
                           make_pad_transform(Wi, InLeftPadW, InRightPadW),
                           make_pass_through_transform(C)),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                make_tuple(
                    Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

            const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                in_n_hip_wip_c_desc,
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

            const auto in_gemmm_gemmk_desc = transform_tensor_descriptor(
                in_n_z_do_y_ho_x_wo_c_desc,
                make_tuple(make_merge_transform(make_tuple(N, Do, Ho, Wo)),
                           make_merge_transform(make_tuple(Z, Y, X, C))),
                make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));

            return in_gemmm_gemmk_desc;
        }
    }

    template <typename BLayout,
              typename std::enable_if<is_same_v<BLayout, tensor_layout::convolution::GKXC> ||
                                          is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                                          is_same_v<BLayout, tensor_layout::convolution::GKZYXC>,
                                      bool>::type = false>
    __host__ __device__ static auto
    MakeBDescriptor_N_K(const ck::Array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& /* b_g_k_c_xs_strides */)
    {
        const index_t K = b_g_k_c_xs_lengths[1];
        const index_t C = b_g_k_c_xs_lengths[2];

        const index_t YX =
            mult_accumulate_n<index_t>(b_g_k_c_xs_lengths.begin() + 3, NDimSpatial, 1);

        const auto wei_gemmn_gemmk_desc =
            make_naive_tensor_descriptor_packed(make_tuple(K, YX * C));

        return wei_gemmn_gemmk_desc;
    }

    template <
        typename BLayout,
        typename std::enable_if<is_same_v<BLayout, tensor_layout::convolution::G_K_X_C> ||
                                    is_same_v<BLayout, tensor_layout::convolution::G_K_YX_C> ||
                                    is_same_v<BLayout, tensor_layout::convolution::G_K_ZYX_C> ||
                                    is_same_v<BLayout, tensor_layout::convolution::KXGC> ||
                                    is_same_v<BLayout, tensor_layout::convolution::KYXGC> ||
                                    is_same_v<BLayout, tensor_layout::convolution::KZYXGC>,
                                bool>::type = false>
    __host__ __device__ static auto
    MakeBDescriptor_N_K(const ck::Array<index_t, NDimSpatial + 3>& b_g_k_c_xs_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& b_g_k_c_xs_strides)
    {
        const index_t K = b_g_k_c_xs_lengths[1];
        const index_t C = b_g_k_c_xs_lengths[2];

        const index_t YX =
            mult_accumulate_n<index_t>(b_g_k_c_xs_lengths.begin() + 3, NDimSpatial, 1);

        const index_t KStride = b_g_k_c_xs_strides[1];
        const index_t XStride = b_g_k_c_xs_strides[2 + NDimSpatial];
        const auto CStride    = I1;

        const auto wei_k_yx_c_desc = make_naive_tensor_descriptor(
            make_tuple(K, YX, C), make_tuple(KStride, XStride, CStride));

        const auto wei_gemmn_gemmk_desc = transform_tensor_descriptor(
            wei_k_yx_c_desc,
            make_tuple(make_pass_through_transform(K), make_merge_transform(make_tuple(YX, C))),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return wei_gemmn_gemmk_desc;
    }

    template <typename CLayout,
              typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::GNWK> ||
                                          is_same_v<CLayout, tensor_layout::convolution::GNHWK> ||
                                          is_same_v<CLayout, tensor_layout::convolution::GNDHWK>,
                                      bool>::type = false>
    __host__ __device__ static auto
    MakeCDescriptor_M_N(const ck::Array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& /* c_g_n_k_wos_strides */)
    {
        const index_t N = c_g_n_k_wos_lengths[1];
        const index_t K = c_g_n_k_wos_lengths[2];

        const index_t NHoWo =
            N * mult_accumulate_n<index_t>(c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1);

        const auto out_gemmm_gemmn_desc = make_naive_tensor_descriptor_packed(make_tuple(NHoWo, K));

        return out_gemmm_gemmn_desc;
    }

    template <
        typename CLayout,
        typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::G_NW_K> ||
                                    is_same_v<CLayout, tensor_layout::convolution::G_NHW_K> ||
                                    is_same_v<CLayout, tensor_layout::convolution::G_NDHW_K> ||
                                    is_same_v<CLayout, tensor_layout::convolution::NWGK> ||
                                    is_same_v<CLayout, tensor_layout::convolution::NHWGK> ||
                                    is_same_v<CLayout, tensor_layout::convolution::NDHWGK>,
                                bool>::type = false>
    __host__ __device__ static auto
    MakeCDescriptor_M_N(const ck::Array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& c_g_n_k_wos_strides)
    {
        const index_t N = c_g_n_k_wos_lengths[1];
        const index_t K = c_g_n_k_wos_lengths[2];

        const auto KStride     = I1;
        const index_t WoStride = c_g_n_k_wos_strides[NDimSpatial + 2];

        const index_t NHoWo =
            N * mult_accumulate_n<index_t>(c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1);

        const auto out_gemmm_gemmn_desc =
            make_naive_tensor_descriptor(make_tuple(NHoWo, K), make_tuple(WoStride, KStride));

        return out_gemmm_gemmn_desc;
    }

    // for output bias
    template <typename CLayout,
              typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::G_K>,
                                      bool>::type = false>
    __host__ __device__ static auto
    MakeCDescriptor_M_N(const ck::Array<index_t, NDimSpatial + 3>& c_g_n_k_wos_lengths,
                        const ck::Array<index_t, NDimSpatial + 3>& c_g_n_k_wos_strides)
    {
        const index_t N       = c_g_n_k_wos_lengths[1];
        const index_t K       = c_g_n_k_wos_lengths[2];
        const index_t KStride = c_g_n_k_wos_strides[2];

        const index_t NHoWo =
            N * mult_accumulate_n<index_t>(c_g_n_k_wos_lengths.begin() + 3, NDimSpatial, 1);

        const auto out_gemmm_gemmn_desc =
            make_naive_tensor_descriptor(make_tuple(NHoWo, K), make_tuple(I0, KStride));

        return out_gemmm_gemmn_desc;
    }
};

// wrapper class to call member functions on TransformConvToGemm struct at runtime
// TODO: figure out aq way to properly pass in layout as an argument
struct TransformConv
{
    TransformConv() {}

    template <index_t NDimSpatial,
              device::ConvolutionForwardSpecialization ConvForwardSpecialization>
    auto
    transform_func(ck::Array<index_t, NDimSpatial + 3> out_lengths,
                   ck::Array<index_t, NDimSpatial + 3> out_strides,
                   TransformConvFwdToGemm<NDimSpatial, ConvForwardSpecialization> conv_fwd_to_gemm)
    {
        if(NDimSpatial == 2)
        {
            return conv_fwd_to_gemm
                .template MakeCDescriptor_M_N<ck::tensor_layout::convolution::NHWGK>(out_lengths,
                                                                                     out_strides);
        }
        else if(NDimSpatial == 3)
        {
            return conv_fwd_to_gemm
                .template MakeCDescriptor_M_N<tensor_layout::convolution::NDHWGK>(out_lengths,
                                                                                  out_strides);
        }
        else if(NDimSpatial == 1)
        {
            return conv_fwd_to_gemm.template MakeCDescriptor_M_N<tensor_layout::convolution::NWGK>(
                out_lengths, out_strides);
        }
    }
};

} // namespace tensor_operation
} // namespace ck
