
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

template <index_t NDimSpatial,
          device::ConvolutionForwardSpecialization ConvForwardSpecialization,
          bool SplitN              = false,
          typename ADataType       = float,
          typename CDataType       = float,
          index_t NumGroupsToMerge = 1>
struct TransformConvFwdToGemm
{
    private:
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};

    template <typename ConvDimsType>
    static long_index_t calculate_element_space_size_impl(const ConvDimsType& lengths,
                                                          const ConvDimsType& strides,
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

    template <typename ConvDimsType>
    static index_t GetSplitedNSize(const ConvDimsType& a_g_n_c_wis_lengths,
                                   const ConvDimsType& a_g_n_c_wis_strides,
                                   const ConvDimsType& c_g_n_k_wos_lengths,
                                   const ConvDimsType& c_g_n_k_wos_strides)
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

    public:
    __host__ __device__ constexpr TransformConvFwdToGemm() {}

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 1, bool>::type = false>
    __host__ __device__ TransformConvFwdToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                               const ConvDimsType& a_g_n_c_wis_strides,
                                               const ConvDimsType& b_g_k_c_xs_lengths,
                                               const ConvDimsType& b_g_k_c_xs_strides,
                                               const ConvDimsType& c_g_n_k_wos_lengths,
                                               const ConvDimsType& c_g_n_k_wos_strides,
                                               const ConvSpatialDimsType& conv_filter_strides,
                                               const ConvSpatialDimsType& conv_filter_dilations,
                                               const ConvSpatialDimsType& input_left_pads,
                                               const ConvSpatialDimsType& input_right_pads)
        : Di_{I1},
          Hi_{I1},
          Wi_{a_g_n_c_wis_lengths[I3]},
          Do_{I1},
          Ho_{I1},
          Wo_{c_g_n_k_wos_lengths[I3]},
          Z_{I1},
          Y_{I1},
          X_{b_g_k_c_xs_lengths[I3]},
          K_{c_g_n_k_wos_lengths[I2]},
          C_{b_g_k_c_xs_lengths[I2]},
          DiStride_{I1},
          HiStride_{I1},
          WiStride_{a_g_n_c_wis_strides[I3]},
          WoStride_{c_g_n_k_wos_strides[I3]},
          XStride_{b_g_k_c_xs_strides[I3]},
          CStrideTensorA_{a_g_n_c_wis_strides[I2]},
          CStrideTensorB_{b_g_k_c_xs_strides[I2]},
          KStrideTensorB_{b_g_k_c_xs_strides[I1]},
          KStrideTensorC_{c_g_n_k_wos_strides[I2]},
          NStrideTensorA_{a_g_n_c_wis_strides[I1]},
          GStrideTensorA_{a_g_n_c_wis_strides[I0]},
          GStrideTensorB_{b_g_k_c_xs_strides[I0]},
          GStrideTensorC_{c_g_n_k_wos_strides[I0]},
          ConvStrideD_{I1},
          ConvStrideH_{I1},
          ConvStrideW_{conv_filter_strides[I0]},
          ConvDilationD_{I1},
          ConvDilationH_{I1},
          ConvDilationW_{conv_filter_dilations[I0]},
          InLeftPadD_{I0},
          InLeftPadH_{I0},
          InLeftPadW_{input_left_pads[I0]},
          InRightPadD_{I0},
          InRightPadH_{I0},
          InRightPadW_{input_right_pads[I0]},
          ZYX_{X_}
    {
        static_assert(is_same_v<ConvSpatialDimsType, std::array<index_t, NDimSpatial>> ||
                      is_same_v<ConvSpatialDimsType, ck::Array<index_t, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, std::array<index_t, NDimSpatial + I3>> ||
                      is_same_v<ConvDimsType, ck::Array<index_t, NDimSpatial + I3>>);

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_c_wis_lengths, a_g_n_c_wis_strides, c_g_n_k_wos_lengths, c_g_n_k_wos_strides);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
        }
        NDoHoWo_ = N_ * Wo_;
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 2, bool>::type = false>
    __host__ __device__ TransformConvFwdToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                               const ConvDimsType& a_g_n_c_wis_strides,
                                               const ConvDimsType& b_g_k_c_xs_lengths,
                                               const ConvDimsType& b_g_k_c_xs_strides,
                                               const ConvDimsType& c_g_n_k_wos_lengths,
                                               const ConvDimsType& c_g_n_k_wos_strides,
                                               const ConvSpatialDimsType& conv_filter_strides,
                                               const ConvSpatialDimsType& conv_filter_dilations,
                                               const ConvSpatialDimsType& input_left_pads,
                                               const ConvSpatialDimsType& input_right_pads)
        : Di_{I1},
          Hi_{a_g_n_c_wis_lengths[I3]},
          Wi_{a_g_n_c_wis_lengths[I4]},
          Do_{I1},
          Ho_{c_g_n_k_wos_lengths[I3]},
          Wo_{c_g_n_k_wos_lengths[I4]},
          Z_{I1},
          Y_{b_g_k_c_xs_lengths[I3]},
          X_{b_g_k_c_xs_lengths[I4]},
          K_{c_g_n_k_wos_lengths[I2]},
          C_{b_g_k_c_xs_lengths[I2]},
          DiStride_{I1},
          HiStride_{a_g_n_c_wis_strides[I3]},
          WiStride_{a_g_n_c_wis_strides[I4]},
          WoStride_{c_g_n_k_wos_strides[I4]},
          XStride_{b_g_k_c_xs_strides[I4]},
          CStrideTensorA_{a_g_n_c_wis_strides[I2]},
          CStrideTensorB_{b_g_k_c_xs_strides[I2]},
          KStrideTensorB_{b_g_k_c_xs_strides[I1]},
          KStrideTensorC_{c_g_n_k_wos_strides[I2]},
          NStrideTensorA_{a_g_n_c_wis_strides[I1]},
          GStrideTensorA_{a_g_n_c_wis_strides[I0]},
          GStrideTensorB_{b_g_k_c_xs_strides[I0]},
          GStrideTensorC_{c_g_n_k_wos_strides[I0]},
          ConvStrideD_{I1},
          ConvStrideH_{conv_filter_strides[I0]},
          ConvStrideW_{conv_filter_strides[I1]},
          ConvDilationD_{I1},
          ConvDilationH_{conv_filter_dilations[I0]},
          ConvDilationW_{conv_filter_dilations[I1]},
          InLeftPadD_{I0},
          InLeftPadH_{input_left_pads[I0]},
          InLeftPadW_{input_left_pads[I1]},
          InRightPadD_{I0},
          InRightPadH_{input_right_pads[I0]},
          InRightPadW_{input_right_pads[I1]},
          ZYX_{Y_ * X_}
    {
        static_assert(is_same_v<ConvSpatialDimsType, std::array<index_t, NDimSpatial>> ||
                      is_same_v<ConvSpatialDimsType, ck::Array<index_t, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, std::array<index_t, NDimSpatial + I3>> ||
                      is_same_v<ConvDimsType, ck::Array<index_t, NDimSpatial + I3>>);

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_c_wis_lengths, a_g_n_c_wis_strides, c_g_n_k_wos_lengths, c_g_n_k_wos_strides);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
        }
        NDoHoWo_ = N_ * Ho_ * Wo_;
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 3, bool>::type = false>
    __host__ __device__ TransformConvFwdToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                               const ConvDimsType& a_g_n_c_wis_strides,
                                               const ConvDimsType& b_g_k_c_xs_lengths,
                                               const ConvDimsType& b_g_k_c_xs_strides,
                                               const ConvDimsType& c_g_n_k_wos_lengths,
                                               const ConvDimsType& c_g_n_k_wos_strides,
                                               const ConvSpatialDimsType& conv_filter_strides,
                                               const ConvSpatialDimsType& conv_filter_dilations,
                                               const ConvSpatialDimsType& input_left_pads,
                                               const ConvSpatialDimsType& input_right_pads)
        : Di_{a_g_n_c_wis_lengths[I3]},
          Hi_{a_g_n_c_wis_lengths[I4]},
          Wi_{a_g_n_c_wis_lengths[I5]},
          Do_{c_g_n_k_wos_lengths[I3]},
          Ho_{c_g_n_k_wos_lengths[I4]},
          Wo_{c_g_n_k_wos_lengths[I5]},
          Z_{b_g_k_c_xs_lengths[I3]},
          Y_{b_g_k_c_xs_lengths[I4]},
          X_{b_g_k_c_xs_lengths[I5]},
          K_{c_g_n_k_wos_lengths[I2]},
          C_{b_g_k_c_xs_lengths[I2]},
          DiStride_{a_g_n_c_wis_strides[I3]},
          HiStride_{a_g_n_c_wis_strides[I4]},
          WiStride_{a_g_n_c_wis_strides[I5]},
          WoStride_{c_g_n_k_wos_strides[I5]},
          XStride_{b_g_k_c_xs_strides[I5]},
          CStrideTensorA_{a_g_n_c_wis_strides[I2]},
          CStrideTensorB_{b_g_k_c_xs_strides[I2]},
          KStrideTensorB_{b_g_k_c_xs_strides[I1]},
          KStrideTensorC_{c_g_n_k_wos_strides[I2]},
          NStrideTensorA_{a_g_n_c_wis_strides[I1]},
          GStrideTensorA_{a_g_n_c_wis_strides[I0]},
          GStrideTensorB_{b_g_k_c_xs_strides[I0]},
          GStrideTensorC_{c_g_n_k_wos_strides[I0]},
          ConvStrideD_{conv_filter_strides[I0]},
          ConvStrideH_{conv_filter_strides[I1]},
          ConvStrideW_{conv_filter_strides[I2]},
          ConvDilationD_{conv_filter_dilations[I0]},
          ConvDilationH_{conv_filter_dilations[I1]},
          ConvDilationW_{conv_filter_dilations[I2]},
          InLeftPadD_{input_left_pads[I0]},
          InLeftPadH_{input_left_pads[I1]},
          InLeftPadW_{input_left_pads[I2]},
          InRightPadD_{input_right_pads[I0]},
          InRightPadH_{input_right_pads[I1]},
          InRightPadW_{input_right_pads[I2]},
          ZYX_{Z_ * Y_ * X_}
    {
        static_assert(is_same_v<ConvSpatialDimsType, std::array<index_t, NDimSpatial>> ||
                      is_same_v<ConvSpatialDimsType, ck::Array<index_t, NDimSpatial>>);
        static_assert(is_same_v<ConvDimsType, std::array<index_t, NDimSpatial + I3>> ||
                      is_same_v<ConvDimsType, ck::Array<index_t, NDimSpatial + I3>>);

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(
                a_g_n_c_wis_lengths, a_g_n_c_wis_strides, c_g_n_k_wos_lengths, c_g_n_k_wos_strides);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
        }
        NDoHoWo_ = N_ * Do_ * Ho_ * Wo_;
    }

    // TODO: implement ck::tensor_layout::convolution that describe packed/strided dimemsion as
    // properties
    template <typename ALayout,
              typename std::enable_if<NDimSpatial == 1 &&
                                          (is_same_v<ALayout, tensor_layout::convolution::G_NW_C> ||
                                           is_same_v<ALayout, tensor_layout::convolution::NWGC> ||
                                           is_same_v<ALayout, tensor_layout::convolution::GNWC>),
                                      bool>::type = false>
    __host__ __device__ auto MakeADescriptor_M_K() const
    {
        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                return make_naive_tensor_descriptor(make_tuple(NDoHoWo_, C_),
                                                    make_tuple(WiStride_, CStrideTensorA_));
            }
            else
            {
                const auto in_gemmm_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(NDoHoWo_, NumGroupsToMerge, C_),
                    make_tuple(WiStride_, GStrideTensorA_, CStrideTensorA_));

                return transform_tensor_descriptor(
                    in_gemmm_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(NDoHoWo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {

                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_), make_tuple(NStrideTensorA_, WiStride_));

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_pass_through_transform(Number<3>{})),
                    make_tuple(Sequence<0, 2>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_));

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(Number<3>{})),
                    make_tuple(Sequence<0, 2, 3>{}, Sequence<1>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, C_),
                    make_tuple(NStrideTensorA_, WiStride_, CStrideTensorA_));

                const auto in_n_wo_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                return transform_tensor_descriptor(
                    in_n_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                const auto in_n_wo_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, C_),
                    make_tuple(NStrideTensorA_, WiStride_, CStrideTensorA_));

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_merge_transform(make_tuple(X_, C_))),
                    make_tuple(Sequence<0, 2>{}, Sequence<1, 3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3>{}, Sequence<4>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_merge_transform(make_tuple(X_, C_))),
                    make_tuple(Sequence<0, 2, 3>{}, Sequence<1, 4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 2 && (is_same_v<ALayout, tensor_layout::convolution::G_NHW_C> ||
                                       is_same_v<ALayout, tensor_layout::convolution::NHWGC> ||
                                       is_same_v<ALayout, tensor_layout::convolution::GNHWC>),
                  bool>::type = false>
    __host__ __device__ auto MakeADescriptor_M_K() const

    {
        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                return make_naive_tensor_descriptor(make_tuple(NDoHoWo_, C_),
                                                    make_tuple(WiStride_, CStrideTensorA_));
            }
            else
            {
                const auto in_gemmm_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(NDoHoWo_, NumGroupsToMerge, C_),
                    make_tuple(WiStride_, GStrideTensorA_, CStrideTensorA_));

                return transform_tensor_descriptor(
                    in_gemmm_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(NDoHoWo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_), make_tuple(NStrideTensorA_, HiStride_, WiStride_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}));

                const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Number<3>{}, Number<3>{}))),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_));

                const auto in_n_hip_wip_groups_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_y_ho_x_wo_groups_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_groups_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_merge_transform(make_tuple(Number<3>{}, Number<3>{}))),
                    make_tuple(Sequence<0, 2, 4, 5>{}, Sequence<1, 3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, CStrideTensorA_));

                const auto in_n_ho_wo_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_ho_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(
                        NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                const auto in_n_ho_wo_groups_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                return transform_tensor_descriptor(
                    in_n_ho_wo_groups_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, CStrideTensorA_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Y_, X_, C_))),
                    make_tuple(Sequence<0, 2, 4>{}, Sequence<1, 3, 5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {

                const auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(
                        NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_, CStrideTensorA_));

                const auto in_n_hip_wip_groups_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto in_n_y_ho_x_wo_groups_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5>{},
                               Sequence<6>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_groups_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_merge_transform(make_tuple(Y_, X_, C_))),
                    make_tuple(Sequence<0, 2, 4, 5>{}, Sequence<1, 3, 6>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 3 && (is_same_v<ALayout, tensor_layout::convolution::G_NDHW_C> ||
                                       is_same_v<ALayout, tensor_layout::convolution::NDHWGC> ||
                                       is_same_v<ALayout, tensor_layout::convolution::GNDHWC>),
                  bool>::type = false>
    __host__ __device__ auto MakeADescriptor_M_K() const

    {
        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                return make_naive_tensor_descriptor(make_tuple(NDoHoWo_, C_),
                                                    make_tuple(WiStride_, CStrideTensorA_));
            }
            else
            {
                const auto in_gemmm_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(NDoHoWo_, NumGroupsToMerge, C_),
                    make_tuple(WiStride_, GStrideTensorA_, CStrideTensorA_));

                return transform_tensor_descriptor(
                    in_gemmm_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(NDoHoWo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_))),
                    make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1, 2>{}, Sequence<3, 4>{}, Sequence<5, 6>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                        make_merge_transform(make_tuple(Number<3>{}, Number<3>{}, Number<3>{}))),
                    make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, NumGroupsToMerge),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, GStrideTensorA_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Number<3>{}, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(Number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_merge_transform(make_tuple(Number<3>{}, Number<3>{}, Number<3>{}))),
                    make_tuple(Sequence<0, 2, 4, 6, 7>{}, Sequence<1, 3, 5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else if constexpr(ConvForwardSpecialization ==
                          device::ConvolutionForwardSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, CStrideTensorA_));

                const auto in_n_do_ho_wo_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Do_), make_tuple(ConvStrideD_)),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                return transform_tensor_descriptor(
                    in_n_do_ho_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_,
                               DiStride_,
                               HiStride_,
                               WiStride_,
                               GStrideTensorA_,
                               CStrideTensorA_));

                const auto in_n_do_ho_wo_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Do_), make_tuple(ConvStrideD_)),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
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
                               Sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_do_ho_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_pass_through_transform(C_)),
                    make_tuple(Sequence<0, 1, 2, 3, 4>{}, Sequence<5>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, CStrideTensorA_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Z_, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}, Sequence<4>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Z_, Y_, X_, C_))),
                    make_tuple(Sequence<0, 2, 4, 6>{}, Sequence<1, 3, 5, 7>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
            else
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_,
                               DiStride_,
                               HiStride_,
                               WiStride_,
                               GStrideTensorA_,
                               CStrideTensorA_));

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
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
                               Sequence<5>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Z_, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(Sequence<0>{},
                               Sequence<1>{},
                               Sequence<2>{},
                               Sequence<3>{},
                               Sequence<4>{},
                               Sequence<5>{}),
                    make_tuple(Sequence<0>{},
                               Sequence<1, 2>{},
                               Sequence<3, 4>{},
                               Sequence<5, 6>{},
                               Sequence<7>{},
                               Sequence<8>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_merge_transform(make_tuple(Z_, Y_, X_, C_))),
                    make_tuple(Sequence<0, 2, 4, 6, 7>{}, Sequence<1, 3, 5, 8>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
    }

    template <typename BLayout,
              typename std::enable_if<is_same_v<BLayout, tensor_layout::convolution::GKXC> ||
                                          is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                                          is_same_v<BLayout, tensor_layout::convolution::GKZYXC>,
                                      bool>::type = false>
    __host__ __device__ auto MakeBDescriptor_N_K() const
    {
        if constexpr(ConvForwardSpecialization ==
                     device::ConvolutionForwardSpecialization::Filter3x3)
        {
            using FilterSizeNumType =
                std::conditional_t<NDimSpatial == 1,
                                   Number<3>,
                                   std::conditional_t<NDimSpatial == 2, Number<9>, Number<27>>>;

            if constexpr(NumGroupsToMerge == 1)
            {
                return make_naive_tensor_descriptor_packed(make_tuple(K_, FilterSizeNumType{}));
            }
            else
            {

                const auto wei_gemmn_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(K_, NumGroupsToMerge, FilterSizeNumType{}),
                    make_tuple(KStrideTensorB_, GStrideTensorB_, CStrideTensorB_));
                return transform_tensor_descriptor(
                    wei_gemmn_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(K_, NumGroupsToMerge)),
                               make_pass_through_transform(FilterSizeNumType{})),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                return make_naive_tensor_descriptor_packed(make_tuple(K_, ZYX_ * C_));
            }
            else
            {
                const auto wei_gemmn_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(K_, NumGroupsToMerge, ZYX_ * C_),
                    make_tuple(KStrideTensorB_, GStrideTensorB_, CStrideTensorB_));
                return transform_tensor_descriptor(
                    wei_gemmn_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(K_, NumGroupsToMerge)),
                               make_pass_through_transform(ZYX_ * C_)),
                    make_tuple(Sequence<0, 1>{}, Sequence<2>{}),
                    make_tuple(Sequence<0>{}, Sequence<1>{}));
            }
        }
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
    __host__ __device__ auto MakeBDescriptor_N_K() const
    {
        const auto wei_k_yx_c_desc = make_naive_tensor_descriptor(
            make_tuple(K_, ZYX_, C_), make_tuple(KStrideTensorB_, XStride_, CStrideTensorB_));

        const auto wei_gemmn_gemmk_desc = transform_tensor_descriptor(
            wei_k_yx_c_desc,
            make_tuple(make_pass_through_transform(K_), make_merge_transform(make_tuple(ZYX_, C_))),
            make_tuple(Sequence<0>{}, Sequence<1, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return wei_gemmn_gemmk_desc;
    }

    template <typename CLayout,
              typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::GNWK> ||
                                          is_same_v<CLayout, tensor_layout::convolution::GNHWK> ||
                                          is_same_v<CLayout, tensor_layout::convolution::GNDHWK>,
                                      bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        return make_naive_tensor_descriptor_packed(make_tuple(NDoHoWo_, K_));
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
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        if constexpr(NumGroupsToMerge == 1)
        {
            return make_naive_tensor_descriptor(make_tuple(NDoHoWo_, K_),
                                                make_tuple(WoStride_, KStrideTensorC_));
        }
        else
        {
            const auto nhwo_groups_k_1_desc = make_naive_tensor_descriptor(
                make_tuple(NDoHoWo_, NumGroupsToMerge, K_, 1),
                make_tuple(WoStride_, GStrideTensorC_, KStrideTensorC_, GStrideTensorC_));
            // Padd 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                nhwo_groups_k_1_desc,
                make_tuple(make_pass_through_transform(NDoHoWo_),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1)),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}, Sequence<2>{}, Sequence<3>{}));
            // We need only matrices from diagonal. X_or returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_pass_through_transform(NDoHoWo_),
                           make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_)),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}),
                make_tuple(Sequence<0>{}, Sequence<1, 3>{}, Sequence<2>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NDoHoWo_, NumGroupsToMerge)),
                           make_merge_transform(make_tuple(K_, NumGroupsToMerge))),
                make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}),
                make_tuple(Sequence<0>{}, Sequence<1>{}));
        }
    }

    // for output bias
    template <typename CLayout,
              typename std::enable_if<is_same_v<CLayout, tensor_layout::convolution::G_K>,
                                      bool>::type = false>
    __host__ __device__ auto MakeCDescriptor_M_N() const
    {
        const auto out_gemmm_gemmn_desc =
            make_naive_tensor_descriptor(make_tuple(NDoHoWo_, K_), make_tuple(I0, KStrideTensorC_));

        return out_gemmm_gemmn_desc;
    }

    public:
    index_t N_;

    private:
    const index_t Di_, Hi_, Wi_;
    const index_t Do_, Ho_, Wo_;
    const index_t Z_, Y_, X_;
    const index_t K_, C_;
    const index_t DiStride_, HiStride_, WiStride_;
    const index_t WoStride_;
    const index_t XStride_;
    const index_t CStrideTensorA_, CStrideTensorB_, KStrideTensorB_, KStrideTensorC_;
    const index_t NStrideTensorA_;
    const index_t GStrideTensorA_, GStrideTensorB_, GStrideTensorC_;
    const index_t ConvStrideD_, ConvStrideH_, ConvStrideW_;
    const index_t ConvDilationD_, ConvDilationH_, ConvDilationW_;
    const index_t InLeftPadD_, InLeftPadH_, InLeftPadW_;
    const index_t InRightPadD_, InRightPadH_, InRightPadW_;
    const index_t ZYX_;
    index_t NDoHoWo_;
};

// wrapper class to call member functions on TransformConvToGemm struct at runtime
// TODO: figure out aq way to properly pass in layout as an argument
struct TransformConv
{
    TransformConv() {}

    template <index_t NDimSpatial,
              device::ConvolutionForwardSpecialization ConvForwardSpecialization>
    auto
    transform_func(TransformConvFwdToGemm<NDimSpatial, ConvForwardSpecialization> conv_fwd_to_gemm)
    {
        if(NDimSpatial == 2)
        {
            return conv_fwd_to_gemm
                .template MakeCDescriptor_M_N<ck::tensor_layout::convolution::NHWGK>();
        }
        else if(NDimSpatial == 3)
        {
            return conv_fwd_to_gemm
                .template MakeCDescriptor_M_N<tensor_layout::convolution::NDHWGK>();
        }
        else if(NDimSpatial == 1)
        {
            return conv_fwd_to_gemm
                .template MakeCDescriptor_M_N<tensor_layout::convolution::NWGK>();
        }
    }
};

} // namespace tensor_operation
} // namespace ck
