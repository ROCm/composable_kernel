
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"
#include <iostream>
#include <queue>
#include <vector>

namespace ck_tile {

// ═══════════════════════════════════════════════════════════════════════
// Split-Image Information Structure
// ═══════════════════════════════════════════════════════════════════════
// This structure holds all information needed to perform split-image
// for 1D/2D/3D convolutions. It is calculated AFTER Split-N to ensure
// correct offset calculations when both splitting strategies are active.
template <typename IndexType = index_t>
struct SplitImageInfo
{
    bool should_split; // Should we split?

    // Split dimensions (output)
    IndexType out_left;  // LEFT output size
    IndexType out_right; // RIGHT output size

    // Input sizes for LEFT and RIGHT pieces
    IndexType in_left;
    IndexType in_right;

    // Memory offsets (in elements)
    // These are calculated using N_ AFTER Split-N happens
    long_index_t input_offset;  // Offset for RIGHT piece input pointer
    long_index_t output_offset; // Offset for RIGHT piece output pointer

    // Padding adjustments for LEFT piece
    IndexType left_pad_left;
    IndexType right_pad_left;

    // Padding adjustments for RIGHT piece
    IndexType left_pad_right;
    IndexType right_pad_right;
};

template <index_t NDimSpatial,
          ConvolutionSpecialization ConvSpecialization,
          index_t VectorSizeA,
          index_t VectorSizeB,
          index_t VectorSizeC,
          bool SplitN              = false,
          typename ADataType       = float,
          typename CDataType       = float,
          index_t NumGroupsToMerge = 1,
          typename IndexType       = index_t>
struct TransformConvFwdToGemm
{
    private:
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};
    static constexpr auto I3 = number<3>{};
    static constexpr auto I4 = number<4>{};
    static constexpr auto I5 = number<5>{};

    // Unified 2GB limit constant for both Split-N and Split-Image
    static constexpr long_index_t TwoGB = (long_index_t{1} << 31); // 2GB

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
    static IndexType GetSplitedNSize(const ConvDimsType& a_g_n_c_wis_lengths,
                                     const ConvDimsType& c_g_n_k_wos_lengths)
    {
        // Removed verbose debug prints for cleaner output

        // Calculate strides internally assuming contiguous memory layout
        ConvDimsType a_g_n_c_wis_strides, c_g_n_k_wos_strides;
        const index_t num_dims = a_g_n_c_wis_lengths.size();

        // Calculate strides for input tensor (innermost to outermost)
        a_g_n_c_wis_strides[num_dims - 1] = 1;
        for(index_t i = num_dims - 2; i >= 0; i--)
        {
            a_g_n_c_wis_strides[i] = a_g_n_c_wis_strides[i + 1] * a_g_n_c_wis_lengths[i + 1];
        }

        // Calculate strides for output tensor
        c_g_n_k_wos_strides[num_dims - 1] = 1;
        for(index_t i = num_dims - 2; i >= 0; i--)
        {
            c_g_n_k_wos_strides[i] = c_g_n_k_wos_strides[i + 1] * c_g_n_k_wos_lengths[i + 1];
        }

        const long_index_t a_element_space_size =
            calculate_element_space_size_impl(a_g_n_c_wis_lengths, a_g_n_c_wis_strides, I1);
        const long_index_t c_element_space_size =
            calculate_element_space_size_impl(c_g_n_k_wos_lengths, c_g_n_k_wos_strides, I1);
        const long_index_t element_space_size = ck_tile::max(
            a_element_space_size * sizeof(ADataType), c_element_space_size * sizeof(CDataType));

        const IndexType N = a_g_n_c_wis_lengths[I1];

        if(element_space_size > TwoGB)
        {
            // Minimum divisor of N to not exceed 2GB
            const auto divisor = ck_tile::integer_divide_ceil(element_space_size, TwoGB);

            if(divisor <= static_cast<double>(N))
            {
                // Find least divisor of N larger than element_space_size / TwoGB
                // Iterate up to sqrt(N). There are no divisors above this value.
                for(IndexType least_divisor = divisor; least_divisor * least_divisor <= N;
                    least_divisor++)
                {
                    if(N % least_divisor == 0)
                    {
                        IndexType result = N / least_divisor;
                        return result;
                    }
                }
                // Not found, process one Convolution N per block
                return 1;
            }
            else
            {
                // Split Convolution's N dimension into N workgroups. However
                // this still might not result in sufficiently small tensor,
                // but at least later on we could divide the image as well.
                return 1;
            }
        }
        else
        {
            // Split N is not needed.
            return N;
        }
    }

    public:
    // Public getter methods for Split-N support
    CK_TILE_HOST constexpr IndexType GetN() const { return N_; }
    CK_TILE_HOST constexpr IndexType GetOriginalN() const { return original_N_; }

    CK_TILE_HOST constexpr TransformConvFwdToGemm() {}

    template <typename TransformConvFwdToGemmBase>
    CK_TILE_HOST
    TransformConvFwdToGemm(const TransformConvFwdToGemmBase& transform_conv_fwd_to_gemm_base)
        : G_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.G_)},
          N_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.N_)},
          original_N_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.original_N_)},
          Di_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Di_)},
          Hi_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Hi_)},
          Wi_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Wi_)},
          Do_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Do_)},
          Ho_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Ho_)},
          Wo_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Wo_)},
          Z_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Z_)},
          Y_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.Y_)},
          X_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.X_)},
          K_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.K_)},
          C_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.C_)},
          ConvStrideD_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvStrideD_)},
          ConvStrideH_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvStrideH_)},
          ConvStrideW_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvStrideW_)},
          ConvDilationD_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvDilationD_)},
          ConvDilationH_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvDilationH_)},
          ConvDilationW_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ConvDilationW_)},
          InLeftPadD_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InLeftPadD_)},
          InLeftPadH_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InLeftPadH_)},
          InLeftPadW_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InLeftPadW_)},
          InRightPadD_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InRightPadD_)},
          InRightPadH_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InRightPadH_)},
          InRightPadW_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.InRightPadW_)},
          ZYX_{static_cast<IndexType>(transform_conv_fwd_to_gemm_base.ZYX_)}
    {
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 1, bool>::type = false>
    CK_TILE_HOST TransformConvFwdToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                        const ConvDimsType& b_g_k_c_xs_lengths,
                                        const ConvDimsType& c_g_n_k_wos_lengths,
                                        const ConvSpatialDimsType& conv_filter_strides,
                                        const ConvSpatialDimsType& conv_filter_dilations,
                                        const ConvSpatialDimsType& input_left_pads,
                                        const ConvSpatialDimsType& input_right_pads)
        : G_{a_g_n_c_wis_lengths[I0]},
          Di_{I1},
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
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);

        // Store original N and initialize N_
        original_N_ = N_ = c_g_n_k_wos_lengths[I1];

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(a_g_n_c_wis_lengths, c_g_n_k_wos_lengths);
        }
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 2, bool>::type = false>
    CK_TILE_HOST TransformConvFwdToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                        const ConvDimsType& b_g_k_c_xs_lengths,
                                        const ConvDimsType& c_g_n_k_wos_lengths,
                                        const ConvSpatialDimsType& conv_filter_strides,
                                        const ConvSpatialDimsType& conv_filter_dilations,
                                        const ConvSpatialDimsType& input_left_pads,
                                        const ConvSpatialDimsType& input_right_pads)
        : G_{a_g_n_c_wis_lengths[I0]},
          Di_{I1},
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
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);

        // Store original N and initialize N_
        original_N_ = N_ = c_g_n_k_wos_lengths[I1];

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(a_g_n_c_wis_lengths, c_g_n_k_wos_lengths);
        }
    }

    template <typename ConvDimsType,
              typename ConvSpatialDimsType,
              index_t NDim                                   = NDimSpatial,
              typename std::enable_if<NDim == 3, bool>::type = false>
    CK_TILE_HOST TransformConvFwdToGemm(const ConvDimsType& a_g_n_c_wis_lengths,
                                        const ConvDimsType& b_g_k_c_xs_lengths,
                                        const ConvDimsType& c_g_n_k_wos_lengths,
                                        const ConvSpatialDimsType& conv_filter_strides,
                                        const ConvSpatialDimsType& conv_filter_dilations,
                                        const ConvSpatialDimsType& input_left_pads,
                                        const ConvSpatialDimsType& input_right_pads)
        : G_{a_g_n_c_wis_lengths[I0]},
          Di_{a_g_n_c_wis_lengths[I3]},
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
        static_assert(std::is_same_v<ConvSpatialDimsType, std::array<IndexType, NDimSpatial>> ||
                      std::is_same_v<ConvSpatialDimsType, ck_tile::array<IndexType, NDimSpatial>>);
        static_assert(std::is_same_v<ConvDimsType, std::array<IndexType, NDimSpatial + I3>> ||
                      std::is_same_v<ConvDimsType, ck_tile::array<IndexType, NDimSpatial + I3>>);

        // Store original N and initialize N_
        original_N_ = N_ = c_g_n_k_wos_lengths[I1];

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(a_g_n_c_wis_lengths, c_g_n_k_wos_lengths);
        }
    }

    // Check if descriptors fit within memory threshold
    // NOTE: Not used by forward convolution (uses CalculateSplitImage() instead)
    // May be used by backward convolution implementations
    CK_TILE_HOST bool AreDescriptorsSmallerThan2GB() const
    {
        const long_index_t input_size  = static_cast<long_index_t>(N_) * Di_ * Hi_ * Wi_ * C_;
        const long_index_t output_size = static_cast<long_index_t>(N_) * Do_ * Ho_ * Wo_ * K_;

        const long_index_t threshold = TwoGB / sizeof(ADataType);
        return (input_size < threshold) && (output_size < threshold);
    }

    // TODO: implement ck_tile::tensor_layout::convolution that describe packed/strided dimemsion as
    // properties
    template <typename ALayout,
              typename std::enable_if<NDimSpatial == 1 &&
                                          std::is_same_v<ALayout, tensor_layout::convolution::NWGC>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const
    {
        IndexType WiStride_       = G_ * C_;
        IndexType CStrideTensorA_ = 1;
        IndexType NStrideTensorA_ = Di_ * Hi_ * Wi_ * G_ * C_;
        IndexType GStrideTensorA_ = C_;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_gemmm_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wo_, C_),
                    make_tuple(NStrideTensorA_, WiStride_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);
                return transform_tensor_descriptor(
                    in_gemmm_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_gemmm_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wo_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                return transform_tensor_descriptor(
                    in_gemmm_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {

                const auto in_n_wi_c_desc =
                    make_naive_tensor_descriptor(make_tuple(N_, Wi_),
                                                 make_tuple(NStrideTensorA_, WiStride_),
                                                 number<VectorSizeA>{},
                                                 I1);

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                    make_tuple(sequence<0>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_))),
                    make_tuple(sequence<0>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_pass_through_transform(number<3>{})),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(number<3>{})),
                    make_tuple(sequence<0, 2, 3>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, C_),
                    make_tuple(NStrideTensorA_, WiStride_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_wo_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                return transform_tensor_descriptor(
                    in_n_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_wo_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, C_),
                    make_tuple(NStrideTensorA_, WiStride_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                               make_merge_transform(make_tuple(X_, C_))),
                    make_tuple(sequence<0, 2>{}, sequence<1, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_n_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_, WiStride_, GStrideTensorA_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_wip_c_desc = transform_tensor_descriptor(
                    in_n_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

                const auto in_n_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3>{}, sequence<4>{}));

                return transform_tensor_descriptor(
                    in_n_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Wo_, NumGroupsToMerge)),
                               make_merge_transform(make_tuple(X_, C_))),
                    make_tuple(sequence<0, 2, 3>{}, sequence<1, 4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 2 && std::is_same_v<ALayout, tensor_layout::convolution::NHWGC>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const

    {
        IndexType HiStride_       = Wi_ * G_ * C_;
        IndexType WiStride_       = G_ * C_;
        IndexType CStrideTensorA_ = 1;
        IndexType NStrideTensorA_ = Di_ * Hi_ * Wi_ * G_ * C_;
        IndexType GStrideTensorA_ = C_;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_gemmm_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Ho_, Wo_, C_),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                return transform_tensor_descriptor(
                    in_gemmm_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_gemmm_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Ho_, Wo_, NumGroupsToMerge, C_),
                    make_tuple(
                        NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                return transform_tensor_descriptor(
                    in_gemmm_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2, 3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c_desc =
                    make_naive_tensor_descriptor(make_tuple(N_, Hi_, Wi_),
                                                 make_tuple(NStrideTensorA_, HiStride_, WiStride_),
                                                 number<VectorSizeA>{},
                                                 I1);

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}));

                const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_))),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(number<3>{}, number<3>{}))),
                    make_tuple(sequence<0, 2, 4>{}, sequence<1, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_hip_wip_groups_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

                const auto in_n_y_ho_x_wo_groups_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_groups_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_merge_transform(make_tuple(number<3>{}, number<3>{}))),
                    make_tuple(sequence<0, 2, 4, 5>{}, sequence<1, 3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_ho_wo_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

                return transform_tensor_descriptor(
                    in_n_ho_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(
                        NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_ho_wo_groups_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));

                return transform_tensor_descriptor(
                    in_n_ho_wo_groups_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2, 3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, HiStride_, WiStride_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

                const auto in_n_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Y_, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(X_, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Y_, X_, C_))),
                    make_tuple(sequence<0, 2, 4>{}, sequence<1, 3, 5>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {

                const auto in_n_hi_wi_groups_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Hi_, Wi_, NumGroupsToMerge, C_),
                    make_tuple(
                        NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_hip_wip_groups_c_desc = transform_tensor_descriptor(
                    in_n_hi_wi_groups_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));

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
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{},
                               sequence<1, 2>{},
                               sequence<3, 4>{},
                               sequence<5>{},
                               sequence<6>{}));

                return transform_tensor_descriptor(
                    in_n_y_ho_x_wo_groups_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge)),
                               make_merge_transform(make_tuple(Y_, X_, C_))),
                    make_tuple(sequence<0, 2, 4, 5>{}, sequence<1, 3, 6>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
    }

    template <typename ALayout,
              typename std::enable_if<
                  NDimSpatial == 3 && std::is_same_v<ALayout, tensor_layout::convolution::NDHWGC>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeADescriptor_M_K() const

    {
        IndexType DiStride_       = Hi_ * Wi_ * G_ * C_;
        IndexType HiStride_       = Wi_ * G_ * C_;
        IndexType WiStride_       = G_ * C_;
        IndexType CStrideTensorA_ = 1;
        IndexType NStrideTensorA_ = Di_ * Hi_ * Wi_ * G_ * C_;
        IndexType GStrideTensorA_ = C_;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_gemmm_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Do_, Ho_, Wo_, C_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                return transform_tensor_descriptor(
                    in_gemmm_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2, 3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_gemmm_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge, C_),
                    make_tuple(NStrideTensorA_,
                               DiStride_,
                               HiStride_,
                               WiStride_,
                               GStrideTensorA_,
                               CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                return transform_tensor_descriptor(
                    in_gemmm_groups_gemmk_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2, 3, 4>{}, sequence<5>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_)),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_))),
                    make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}),
                    make_tuple(
                        sequence<0>{}, sequence<1, 2>{}, sequence<3, 4>{}, sequence<5, 6>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                        make_merge_transform(make_tuple(number<3>{}, number<3>{}, number<3>{}))),
                    make_tuple(sequence<0, 2, 4, 6>{}, sequence<1, 3, 5>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, NumGroupsToMerge),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, GStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));

                const auto in_n_z_do_y_ho_x_wo_c_desc = transform_tensor_descriptor(
                    in_n_hip_wip_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(number<3>{}, Do_),
                                                    make_tuple(ConvDilationD_, ConvStrideD_)),
                               make_embed_transform(make_tuple(number<3>{}, Ho_),
                                                    make_tuple(ConvDilationH_, ConvStrideH_)),
                               make_embed_transform(make_tuple(number<3>{}, Wo_),
                                                    make_tuple(ConvDilationW_, ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{},
                               sequence<1, 2>{},
                               sequence<3, 4>{},
                               sequence<5, 6>{},
                               sequence<7>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_merge_transform(make_tuple(number<3>{}, number<3>{}, number<3>{}))),
                    make_tuple(sequence<0, 2, 4, 6, 7>{}, sequence<1, 3, 5>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_do_ho_wo_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Do_), make_tuple(ConvStrideD_)),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));

                return transform_tensor_descriptor(
                    in_n_do_ho_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2, 3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
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
                               CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_do_ho_wo_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_embed_transform(make_tuple(Do_), make_tuple(ConvStrideD_)),
                               make_embed_transform(make_tuple(Ho_), make_tuple(ConvStrideH_)),
                               make_embed_transform(make_tuple(Wo_), make_tuple(ConvStrideW_)),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}));

                return transform_tensor_descriptor(
                    in_n_do_ho_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_pass_through_transform(C_)),
                    make_tuple(sequence<0, 1, 2, 3, 4>{}, sequence<5>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                const auto in_n_di_hi_wi_c_desc = make_naive_tensor_descriptor(
                    make_tuple(N_, Di_, Hi_, Wi_, C_),
                    make_tuple(NStrideTensorA_, DiStride_, HiStride_, WiStride_, CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(C_)),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}));

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
                        sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                    make_tuple(sequence<0>{},
                               sequence<1, 2>{},
                               sequence<3, 4>{},
                               sequence<5, 6>{},
                               sequence<7>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                               make_merge_transform(make_tuple(Z_, Y_, X_, C_))),
                    make_tuple(sequence<0, 2, 4, 6>{}, sequence<1, 3, 5, 7>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
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
                               CStrideTensorA_),
                    number<VectorSizeA>{},
                    I1);

                const auto in_n_hip_wip_c_desc = transform_tensor_descriptor(
                    in_n_di_hi_wi_c_desc,
                    make_tuple(make_pass_through_transform(N_),
                               make_pad_transform(Di_, InLeftPadD_, InRightPadD_),
                               make_pad_transform(Hi_, InLeftPadH_, InRightPadH_),
                               make_pad_transform(Wi_, InLeftPadW_, InRightPadW_),
                               make_pass_through_transform(NumGroupsToMerge),
                               make_pass_through_transform(C_)),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}));

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
                    make_tuple(sequence<0>{},
                               sequence<1>{},
                               sequence<2>{},
                               sequence<3>{},
                               sequence<4>{},
                               sequence<5>{}),
                    make_tuple(sequence<0>{},
                               sequence<1, 2>{},
                               sequence<3, 4>{},
                               sequence<5, 6>{},
                               sequence<7>{},
                               sequence<8>{}));

                return transform_tensor_descriptor(
                    in_n_z_do_y_ho_x_wo_c_desc,
                    make_tuple(
                        make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge)),
                        make_merge_transform(make_tuple(Z_, Y_, X_, C_))),
                    make_tuple(sequence<0, 2, 4, 6, 7>{}, sequence<1, 3, 5, 8>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
    }

    template <
        typename BLayout,
        typename std::enable_if<std::is_same_v<BLayout, tensor_layout::convolution::GKXC> ||
                                    std::is_same_v<BLayout, tensor_layout::convolution::GKYXC> ||
                                    std::is_same_v<BLayout, tensor_layout::convolution::GKZYXC>,
                                bool>::type = false>
    CK_TILE_HOST auto MakeBDescriptor_N_K() const
    {
        IndexType CStrideTensorB_ = 1;
        IndexType KStrideTensorB_ = Z_ * Y_ * X_ * C_;
        IndexType GStrideTensorB_ = K_ * Z_ * Y_ * X_ * C_;

        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            using FilterSizeNumType =
                std::conditional_t<NDimSpatial == 1,
                                   number<3>,
                                   std::conditional_t<NDimSpatial == 2, number<9>, number<27>>>;

            if constexpr(NumGroupsToMerge == 1)
            {
                return make_naive_tensor_descriptor(make_tuple(K_, FilterSizeNumType{}),
                                                    make_tuple(FilterSizeNumType{}, I1),
                                                    number<VectorSizeB>{},
                                                    I1);
            }
            else
            {

                const auto wei_gemmn_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(K_, NumGroupsToMerge, FilterSizeNumType{}),
                    make_tuple(KStrideTensorB_, GStrideTensorB_, CStrideTensorB_),
                    number<VectorSizeB>{},
                    I1);
                return transform_tensor_descriptor(
                    wei_gemmn_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(K_, NumGroupsToMerge)),
                               make_pass_through_transform(FilterSizeNumType{})),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
        else
        {
            if constexpr(NumGroupsToMerge == 1)
            {
                return make_naive_tensor_descriptor(make_tuple(K_, ZYX_ * C_),
                                                    make_tuple(ZYX_ * C_, I1),
                                                    number<VectorSizeB>{},
                                                    I1);
            }
            else
            {
                const auto wei_gemmn_groups_gemmk_desc = make_naive_tensor_descriptor(
                    make_tuple(K_, NumGroupsToMerge, ZYX_ * C_),
                    make_tuple(KStrideTensorB_, GStrideTensorB_, CStrideTensorB_),
                    number<VectorSizeB>{},
                    I1);
                return transform_tensor_descriptor(
                    wei_gemmn_groups_gemmk_desc,
                    make_tuple(make_merge_transform(make_tuple(K_, NumGroupsToMerge)),
                               make_pass_through_transform(ZYX_ * C_)),
                    make_tuple(sequence<0, 1>{}, sequence<2>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
        }
    }

    template <typename CLayout,
              index_t NDimSp                      = NDimSpatial,
              typename std::enable_if<NDimSp == 1 &&
                                          std::is_same_v<CLayout, tensor_layout::convolution::NWGK>,
                                      bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        IndexType WoStride_       = G_ * K_;
        IndexType KStrideTensorC_ = 1;
        IndexType NStrideTensorC_ = Do_ * Ho_ * Wo_ * G_ * K_;
        IndexType GStrideTensorC_ = K_;

        const IndexType NDoHoWo = N_ * Wo_;
        if constexpr(NumGroupsToMerge == 1)
        {
            return make_naive_tensor_descriptor(make_tuple(NDoHoWo, K_),
                                                make_tuple(WoStride_, KStrideTensorC_),
                                                number<VectorSizeC>{},
                                                I1);
        }
        else
        {
            const auto nhwo_groups_k_1_desc = make_naive_tensor_descriptor(
                make_tuple(N_, Wo_, NumGroupsToMerge, K_, 1),
                make_tuple(
                    NStrideTensorC_, WoStride_, GStrideTensorC_, KStrideTensorC_, GStrideTensorC_),
                number<VectorSizeC>{},
                I1);
            // Padd 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                nhwo_groups_k_1_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Wo_)),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}, sequence<3>{}, sequence<4>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));
            // We need only matrices from diagonal. X_or returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_pass_through_transform(NDoHoWo),
                           make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_)),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NDoHoWo, NumGroupsToMerge)),
                           make_merge_transform(make_tuple(K_, NumGroupsToMerge))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    template <typename CLayout,
              index_t NDimSp = NDimSpatial,

              typename std::enable_if<
                  NDimSp == 2 && std::is_same_v<CLayout, tensor_layout::convolution::NHWGK>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        IndexType HoStride_       = Wo_ * G_ * K_;
        IndexType WoStride_       = G_ * K_;
        IndexType KStrideTensorC_ = 1;
        IndexType NStrideTensorC_ = Do_ * Ho_ * Wo_ * G_ * K_;
        IndexType GStrideTensorC_ = K_;

        const IndexType NDoHoWo = N_ * Ho_ * Wo_;
        if constexpr(NumGroupsToMerge == 1)
        {
            return make_naive_tensor_descriptor(make_tuple(NDoHoWo, K_),
                                                make_tuple(WoStride_, KStrideTensorC_),
                                                number<VectorSizeC>{},
                                                I1);
        }
        else
        {
            const auto nhwo_groups_k_1_desc =
                make_naive_tensor_descriptor(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge, K_, 1),
                                             make_tuple(NStrideTensorC_,
                                                        HoStride_,
                                                        WoStride_,
                                                        GStrideTensorC_,
                                                        KStrideTensorC_,
                                                        GStrideTensorC_),
                                             number<VectorSizeC>{},
                                             I1);
            // Padd 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                nhwo_groups_k_1_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1)),
                make_tuple(sequence<0, 1, 2>{}, sequence<3>{}, sequence<4>{}, sequence<5>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));
            // We need only matrices from diagonal. X_or returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_pass_through_transform(NDoHoWo),
                           make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_)),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NDoHoWo, NumGroupsToMerge)),
                           make_merge_transform(make_tuple(K_, NumGroupsToMerge))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    template <typename CLayout,
              index_t NDimSp = NDimSpatial,
              typename std::enable_if<
                  NDimSp == 3 && std::is_same_v<CLayout, tensor_layout::convolution::NDHWGK>,
                  bool>::type = false>
    CK_TILE_HOST auto MakeCDescriptor_M_N() const
    {
        IndexType DoStride_       = Ho_ * Wo_ * G_ * K_;
        IndexType HoStride_       = Wo_ * G_ * K_;
        IndexType WoStride_       = G_ * K_;
        IndexType KStrideTensorC_ = 1;
        IndexType NStrideTensorC_ = Do_ * Ho_ * Wo_ * G_ * K_;
        IndexType GStrideTensorC_ = K_;

        const IndexType NDoHoWo = N_ * Do_ * Ho_ * Wo_;
        if constexpr(NumGroupsToMerge == 1)
        {
            return make_naive_tensor_descriptor(make_tuple(NDoHoWo, K_),
                                                make_tuple(WoStride_, KStrideTensorC_),
                                                number<VectorSizeC>{},
                                                I1);
        }
        else
        {
            const auto nhwo_groups_k_1_desc =
                make_naive_tensor_descriptor(make_tuple(N_, Do_, Ho_, Wo_, NumGroupsToMerge, K_, 1),
                                             make_tuple(NStrideTensorC_,
                                                        DoStride_,
                                                        HoStride_,
                                                        WoStride_,
                                                        GStrideTensorC_,
                                                        KStrideTensorC_,
                                                        GStrideTensorC_),
                                             number<VectorSizeC>{},
                                             I1);
            // Padd 1 to NumGroupsToMerge
            const auto padded_desc = transform_tensor_descriptor(
                nhwo_groups_k_1_desc,
                make_tuple(make_merge_transform(make_tuple(N_, Do_, Ho_, Wo_)),
                           make_pass_through_transform(NumGroupsToMerge),
                           make_pass_through_transform(K_),
                           make_pad_transform(1, 0, NumGroupsToMerge - 1)),
                make_tuple(sequence<0, 1, 2, 3>{}, sequence<4>{}, sequence<5>{}, sequence<6>{}),
                make_tuple(sequence<0>{}, sequence<1>{}, sequence<2>{}, sequence<3>{}));
            // We need only matrices from diagonal. X_or returns 0 for the same
            // values. So if matrices is not on diagonal then it will be stored in padding.
            // To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
            static_assert(NumGroupsToMerge == 1 || NumGroupsToMerge == 2 || NumGroupsToMerge == 4 ||
                          NumGroupsToMerge == 8 || NumGroupsToMerge == 16 ||
                          NumGroupsToMerge == 32 || NumGroupsToMerge == 64);
            const auto unmerged_padded_desc = transform_tensor_descriptor(
                padded_desc,
                make_tuple(make_pass_through_transform(NDoHoWo),
                           make_xor_transform(make_tuple(NumGroupsToMerge, NumGroupsToMerge)),
                           make_pass_through_transform(K_)),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}),
                make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}));
            // Merge To M, N
            return transform_tensor_descriptor(
                unmerged_padded_desc,
                make_tuple(make_merge_transform(make_tuple(NDoHoWo, NumGroupsToMerge)),
                           make_merge_transform(make_tuple(K_, NumGroupsToMerge))),
                make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
                make_tuple(sequence<0>{}, sequence<1>{}));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Split-Image Calculation (AFTER Split-N)
    // ═══════════════════════════════════════════════════════════════════════
    // This method calculates split-image information using N_ (after Split-N).
    // This ensures correct offset calculations when both Split-N and Split-Image
    // are active simultaneously.

    public:
    CK_TILE_HOST SplitImageInfo<IndexType> CalculateSplitImage() const
    {
        SplitImageInfo<IndexType> info;
        info.should_split = false;

        // Use unified threshold from TwoGB constant
        const long_index_t threshold_elements = TwoGB / sizeof(CDataType);

        // Determine which dimension to split based on NDimSpatial
        IndexType out_total, in_total, left_pad, right_pad, stride, dilation, filter;

        if constexpr(NDimSpatial == 1)
        {
            out_total = Wo_;
            in_total  = Wi_;
            left_pad  = InLeftPadW_;
            right_pad = InRightPadW_;
            stride    = ConvStrideW_;
            dilation  = ConvDilationW_;
            filter    = X_;
        }
        else if constexpr(NDimSpatial == 2)
        {
            out_total = Ho_;
            in_total  = Hi_;
            left_pad  = InLeftPadH_;
            right_pad = InRightPadH_;
            stride    = ConvStrideH_;
            dilation  = ConvDilationH_;
            filter    = Y_;
        }
        else if constexpr(NDimSpatial == 3)
        {
            out_total = Do_;
            in_total  = Di_;
            left_pad  = InLeftPadD_;
            right_pad = InRightPadD_;
            stride    = ConvStrideD_;
            dilation  = ConvDilationD_;
            filter    = Z_;
        }
        else
        {
            return info; // Unsupported dimension
        }

        // Check if split is needed - IMPORTANT: Use N_ (after Split-N!)
        long_index_t output_size;
        if constexpr(NDimSpatial == 1)
        {
            output_size = static_cast<long_index_t>(N_) * static_cast<long_index_t>(Wo_) *
                          static_cast<long_index_t>(K_) * static_cast<long_index_t>(G_);
        }
        else if constexpr(NDimSpatial == 2)
        {
            output_size = static_cast<long_index_t>(N_) * static_cast<long_index_t>(Ho_) *
                          static_cast<long_index_t>(Wo_) * static_cast<long_index_t>(K_) *
                          static_cast<long_index_t>(G_);
        }
        else if constexpr(NDimSpatial == 3)
        {
            output_size = static_cast<long_index_t>(N_) * static_cast<long_index_t>(Do_) *
                          static_cast<long_index_t>(Ho_) * static_cast<long_index_t>(Wo_) *
                          static_cast<long_index_t>(K_) * static_cast<long_index_t>(G_);
        }

        if(output_size < threshold_elements)
        {
            return info;
        }

        // Binary split
        info.out_left  = out_total / 2;
        info.out_right = out_total - info.out_left;

        // Effective filter size
        const IndexType x_eff = (filter - 1) * dilation + 1;

        // Safety checks
        const IndexType right_start = info.out_left * stride;
        const IndexType left_end    = (info.out_left - 1) * stride + x_eff;

        const bool is_possible_to_split =
            out_total != 1 && right_start > left_pad && left_end <= (left_pad + in_total);

        if(!is_possible_to_split)
        {
            return info; // Cannot split safely
        }

        info.should_split = true;

        // Calculate input sizes
        const IndexType in_left_end = (info.out_left - 1) * stride + x_eff;
        info.in_left                = in_left_end - left_pad;

        const IndexType in_right_start     = info.out_left * stride;
        const IndexType in_right_available = in_total - (in_right_start - left_pad);
        info.in_right = ck_tile::min(in_right_available, (info.out_right - 1) * stride + x_eff);

        // Calculate physical offset
        const IndexType physical_offset = (info.out_left * stride) - left_pad;

        // Calculate strides - for WITHIN a single batch
        // The stride to jump from one W-position to the next WITHIN the same batch
        long_index_t input_stride, output_stride;
        if constexpr(NDimSpatial == 1)
        {
            // 1D NWGC: stride_W = G * C (within ONE batch)
            input_stride  = static_cast<long_index_t>(G_) * static_cast<long_index_t>(C_);
            output_stride = static_cast<long_index_t>(G_) * static_cast<long_index_t>(K_);
        }
        else if constexpr(NDimSpatial == 2)
        {
            // 2D NHWGC: stride_H = W_in * G * C (within ONE batch)
            input_stride = static_cast<long_index_t>(Wi_) * static_cast<long_index_t>(G_) *
                           static_cast<long_index_t>(C_);
            output_stride = static_cast<long_index_t>(Wo_) * static_cast<long_index_t>(G_) *
                            static_cast<long_index_t>(K_);
        }
        else if constexpr(NDimSpatial == 3)
        {
            // 3D NDHWGC: stride_D = H_in * W_in * G * C (within ONE batch)
            input_stride = static_cast<long_index_t>(Hi_) * static_cast<long_index_t>(Wi_) *
                           static_cast<long_index_t>(G_) * static_cast<long_index_t>(C_);
            output_stride = static_cast<long_index_t>(Ho_) * static_cast<long_index_t>(Wo_) *
                            static_cast<long_index_t>(G_) * static_cast<long_index_t>(K_);
        }

        // Calculate offsets in elements
        info.input_offset  = static_cast<long_index_t>(physical_offset) * input_stride;
        info.output_offset = static_cast<long_index_t>(info.out_left) * output_stride;

        // Padding adjustments
        info.left_pad_left   = left_pad;
        info.right_pad_left  = 0;
        info.left_pad_right  = 0;
        info.right_pad_right = right_pad;

        return info;
    }

    // Helper function for recursive split-image kernel launching
    // Handles BFS-style recursive splitting with depth limit
    template <typename Kernel, index_t kBlockPerCu, typename StreamConfig>
    CK_TILE_HOST static float LaunchWithRecursiveSplit(
        const ck_tile::GroupedConvFwdHostArgs& args,
        const StreamConfig& s,
        const typename Kernel::GroupedConvFwdKernelArgsSpecialized& original_kargs)
    {
        constexpr int split_dim = 0;  // Always split first spatial dimension (W/H/D)
        constexpr int MAX_DEPTH = 10; // Max recursion depth (2^10 = 1024 pieces max)
                                      // With 2GB threshold: handles up to 2TB initial tensor

        // Structure to track each piece with cumulative offsets and depth
        struct SplitPiece
        {
            ck_tile::GroupedConvFwdHostArgs args;
            long_index_t input_offset;  // Cumulative offset from original base
            long_index_t output_offset; // Cumulative offset from original base
            int depth;                  // Recursion depth level

            SplitPiece(const ck_tile::GroupedConvFwdHostArgs& a,
                       long_index_t in_off,
                       long_index_t out_off,
                       int d)
                : args(a), input_offset(in_off), output_offset(out_off), depth(d)
            {
            }
        };

        std::queue<SplitPiece> split_queue;
        std::vector<SplitPiece> ready_list;

        // Start with original problem (offset = 0, depth = 0)
        auto initial_args = args;
        initial_args.N_   = original_kargs.n_per_split; // Use split-N result
        split_queue.emplace(initial_args, 0, 0, 0);

        // BFS-style recursive splitting
        while(!split_queue.empty())
        {
            SplitPiece current = split_queue.front();
            split_queue.pop();

            // Create kargs for this piece and check if it needs splitting
            auto piece_kargs      = Kernel::MakeKernelArgs(current.args);
            auto piece_split_info = piece_kargs.GetSplitImageInfo();

            // Check if we should stop splitting: either small enough OR max depth reached
            if(!piece_split_info.should_split || current.depth >= MAX_DEPTH)
            {
                // This piece is ready to launch
                ready_list.push_back(current);
            }
            else
            {
                // This piece needs to be split into LEFT and RIGHT

                // Create LEFT piece (inherits parent's offset)
                auto left_args                               = current.args;
                left_args.input_spatial_lengths_[split_dim]  = piece_split_info.in_left;
                left_args.output_spatial_lengths_[split_dim] = piece_split_info.out_left;
                left_args.input_left_pads_[split_dim]        = piece_split_info.left_pad_left;
                left_args.input_right_pads_[split_dim]       = piece_split_info.right_pad_left;

                // LEFT inherits parent's cumulative offset (no change)
                auto left_input_offset  = current.input_offset;
                auto left_output_offset = current.output_offset;

                // Create RIGHT piece (parent offset + local offset)
                auto right_args                               = current.args;
                right_args.input_spatial_lengths_[split_dim]  = piece_split_info.in_right;
                right_args.output_spatial_lengths_[split_dim] = piece_split_info.out_right;
                right_args.input_left_pads_[split_dim]        = piece_split_info.left_pad_right;
                right_args.input_right_pads_[split_dim]       = piece_split_info.right_pad_right;

                // CRITICAL: RIGHT accumulates offset (parent + local)
                auto right_input_offset  = current.input_offset + piece_split_info.input_offset;
                auto right_output_offset = current.output_offset + piece_split_info.output_offset;

                // Push LEFT and RIGHT back to queue with incremented depth
                split_queue.emplace(
                    left_args, left_input_offset, left_output_offset, current.depth + 1);
                split_queue.emplace(
                    right_args, right_input_offset, right_output_offset, current.depth + 1);
            }
        }

        // Launch all pieces from ready_list
        float ave_time = 0.0f;
        for(size_t i = 0; i < ready_list.size(); i++)
        {
            const auto& piece = ready_list[i];

            // Create kargs for this piece
            auto piece_kargs = Kernel::MakeKernelArgs(piece.args);

            // Copy Split-N metadata from original kargs
            piece_kargs.n_splits   = original_kargs.n_splits;
            piece_kargs.original_n = original_kargs.original_n;

            // Use batch_stride from ORIGINAL kargs (not split piece's)
            piece_kargs.input_batch_stride  = original_kargs.input_batch_stride;
            piece_kargs.output_batch_stride = original_kargs.output_batch_stride;

            // Store cumulative spatial offset (applied per-batch in kernel)
            piece_kargs.spatial_offset_in  = piece.input_offset;
            piece_kargs.spatial_offset_out = piece.output_offset;

            const dim3 grids  = Kernel::GridSize(piece_kargs);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(piece_kargs))
            {
                throw std::runtime_error("Wrong! Split piece arguments not supported!\n");
            }

            float piece_time = ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, piece_kargs));

            ave_time += piece_time;
        }

        return ave_time;
    }

    private:
    IndexType G_, N_, original_N_;
    IndexType Di_, Hi_, Wi_;
    IndexType Do_, Ho_, Wo_;
    IndexType Z_, Y_, X_;
    IndexType K_, C_;
    IndexType ConvStrideD_, ConvStrideH_, ConvStrideW_;
    IndexType ConvDilationD_, ConvDilationH_, ConvDilationW_;
    IndexType InLeftPadD_, InLeftPadH_, InLeftPadW_;
    IndexType InRightPadD_, InRightPadH_, InRightPadW_;
    IndexType ZYX_;
};

} // namespace ck_tile
