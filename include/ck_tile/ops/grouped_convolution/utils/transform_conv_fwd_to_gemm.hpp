
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"
#include <vector>
#include <queue>
#include <iostream>

namespace ck_tile {

// ═══════════════════════════════════════════════════════════════════════
// Split-Image Information Structure
// ═══════════════════════════════════════════════════════════════════════
// This structure holds all information needed to perform split-image
// for 1D/2D/3D convolutions. It is calculated AFTER Split-N to ensure
// correct offset calculations when both splitting strategies are active.
template <typename IndexType = index_t>
struct SplitImageInfo {
    bool should_split;

    // Split dimensions (output)
    IndexType out_left;
    IndexType out_right;

    // Input sizes for LEFT and RIGHT pieces
    IndexType in_left;
    IndexType in_right;

    // Memory offsets (in elements)
    // These are calculated using N_ AFTER Split-N happens
    long_index_t input_offset;   // Offset for RIGHT piece input pointer
    long_index_t output_offset;  // Offset for RIGHT piece output pointer

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
    // static constexpr long_index_t TwoGB = (long_index_t{1} << 31);
    // static constexpr long_index_t TwoGB = 100L * 1024L * 1024L;  // 100MB for testing
    // static constexpr long_index_t TwoGB = 10L * 1024L * 1024L;  // 10MB for testing split-image
    static constexpr long_index_t TwoGB = 100L * 1024L;  // 100KB for easy testing with small sizes - pieces won't trigger nested split

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

        // Debug: Show actual sizes being compared
        if(N > 1) {
            printf("[DEBUG Split-N Check] N=%ld, element_space_size=%ldMB, threshold=%ldMB\n",
                   static_cast<long>(N),
                   static_cast<long>(element_space_size / (1024 * 1024)),
                   static_cast<long>(TwoGB / (1024 * 1024)));
        }

        if(element_space_size > TwoGB)
        {
            // Minimum divisor of N to not exceed 2GB
            const auto divisor = ck_tile::integer_divide_ceil(element_space_size, TwoGB);

            // Only show debug for actual Split-N cases (N > 1)
            if(N > 1) {
                printf("[DEBUG Split-N] N=%ld needs splitting (tensor exceeds %.0fMB limit)\n",
                       static_cast<long>(N),
                       static_cast<double>(TwoGB) / (1024.0 * 1024.0));
                printf("[DEBUG Split-N] Searching for divisor >= %ld\n",
                       static_cast<long>(divisor));
            }

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
                        if(N > 1) {
                            printf("[DEBUG Split-N] Found divisor: %ld, n_per_split: %ld\n",
                                   static_cast<long>(least_divisor), static_cast<long>(result));
                        }
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

    // Simple check if descriptors fit within memory threshold
    CK_TILE_HOST bool AreDescriptorsSmallerThan2GB() const {
        const long_index_t input_size =
            static_cast<long_index_t>(N_) * Di_ * Hi_ * Wi_ * C_;
        const long_index_t output_size =
            static_cast<long_index_t>(N_) * Do_ * Ho_ * Wo_ * K_;

        const long_index_t threshold = TwoGB / sizeof(ADataType);
        return (input_size < threshold) && (output_size < threshold);
    }

#if 0  // ═══════════════════════════════════════════════════════════════════
       // OUTDATED SPLIT-IMAGE CODE (disabled - replaced by CalculateSplitImage)
       // ═══════════════════════════════════════════════════════════════════
       // This old implementation used recursive queue-based splitting.
       // The new implementation (CalculateSplitImage, line ~2163) is simpler
       // and correctly handles Split-N + Split-Image interaction.
       // ═══════════════════════════════════════════════════════════════════
    // Binary split for recursive queue approach (old CK way)
    // Returns tuple of (left_transformer, right_transformer, input_offset, output_offset)
    CK_TILE_HOST auto SplitConvProblem() const {
        auto left_transformer = *this;
        auto right_transformer = *this;
        long_index_t input_offset = 0;
        long_index_t output_offset = 0;

        // Determine split dimension and do binary split
        // Prefer larger dimensions for split
        if constexpr (NDimSpatial == 3) {
            if (Do_ > 1) {  // D-split
                IndexType left_do = Do_ / 2;
                IndexType right_do = Do_ - left_do;  // Handles odd numbers

                left_transformer.Do_ = left_do;
                right_transformer.Do_ = right_do;

                // Calculate input split point
                IndexType input_d_split = (left_do * ConvStrideD_) - InLeftPadD_ +
                                         (Z_ - 1) * ConvDilationD_;
                right_transformer.Di_ = Di_ - input_d_split;
                right_transformer.InLeftPadD_ = 0;

                // Calculate offsets
                input_offset = input_d_split * Hi_ * Wi_ * G_ * C_;
                output_offset = left_do * Ho_ * Wo_ * G_ * K_;
            } else if (Ho_ > 1) {  // H-split fallback
                IndexType left_ho = Ho_ / 2;
                IndexType right_ho = Ho_ - left_ho;

                left_transformer.Ho_ = left_ho;
                right_transformer.Ho_ = right_ho;

                IndexType input_h_split = (left_ho * ConvStrideH_) - InLeftPadH_ +
                                         (Y_ - 1) * ConvDilationH_;
                right_transformer.Hi_ = Hi_ - input_h_split;
                right_transformer.InLeftPadH_ = 0;

                input_offset = input_h_split * Wi_ * G_ * C_;
                output_offset = left_ho * Wo_ * G_ * K_;
            } else if (Wo_ > 1) {  // W-split fallback
                IndexType left_wo = Wo_ / 2;
                IndexType right_wo = Wo_ - left_wo;

                left_transformer.Wo_ = left_wo;
                right_transformer.Wo_ = right_wo;

                IndexType input_w_split = (left_wo * ConvStrideW_) - InLeftPadW_ +
                                         (X_ - 1) * ConvDilationW_;
                right_transformer.Wi_ = Wi_ - input_w_split;
                right_transformer.InLeftPadW_ = 0;

                input_offset = input_w_split * G_ * C_;
                output_offset = left_wo * G_ * K_;
            }
        } else if constexpr (NDimSpatial == 2) {
            if (Ho_ > 1) {  // H-split
                IndexType left_ho = Ho_ / 2;
                IndexType right_ho = Ho_ - left_ho;  // Handles odd numbers

                left_transformer.Ho_ = left_ho;
                right_transformer.Ho_ = right_ho;

                // Calculate input split point
                IndexType input_h_split = (left_ho * ConvStrideH_) - InLeftPadH_ +
                                         (Y_ - 1) * ConvDilationH_;
                right_transformer.Hi_ = Hi_ - input_h_split;
                right_transformer.InLeftPadH_ = 0;

                // Calculate offsets
                input_offset = input_h_split * Wi_ * G_ * C_;
                output_offset = left_ho * Wo_ * G_ * K_;
            } else if (Wo_ > 1) {  // W-split fallback
                IndexType left_wo = Wo_ / 2;
                IndexType right_wo = Wo_ - left_wo;

                left_transformer.Wo_ = left_wo;
                right_transformer.Wo_ = right_wo;

                IndexType input_w_split = (left_wo * ConvStrideW_) - InLeftPadW_ +
                                         (X_ - 1) * ConvDilationW_;
                right_transformer.Wi_ = Wi_ - input_w_split;
                right_transformer.InLeftPadW_ = 0;

                input_offset = input_w_split * G_ * C_;
                output_offset = left_wo * G_ * K_;
            }
        } else {  // NDimSpatial == 1
            if (Wo_ > 1) {  // W-split
                // === STEP 1: Split output dimension in half ===
                // Example: Wo=32768 → left_wo=16384, right_wo=16384
                IndexType left_wo = Wo_ / 2;
                IndexType right_wo = Wo_ - left_wo;  // Handles odd numbers (e.g., 61→30+31)

                left_transformer.Wo_ = left_wo;
                right_transformer.Wo_ = right_wo;

                // === STEP 2: Calculate where right piece starts in INPUT ===
                // Formula: input_w_split = (left_wo * stride) - left_pad + (filter - 1) * dilation
                //
                // Why this formula?
                // - (left_wo * stride): Base position in "unpadded" input coordinate system
                // - (- left_pad): Adjust for padding (padding shifts coordinates left)
                // - (+ (X-1) * dilation): Need extra input elements for filter receptive field
                //
                // Example: Wo=32768, left_wo=16384, stride=1, left_pad=1, X=3, dilation=1
                //   input_w_split = (16384 * 1) - 1 + (3 - 1) * 1
                //                 = 16384 - 1 + 2
                //                 = 16385
                //
                // This means:
                //   - Left piece processes:  input[0..16385+2]     → output[0..16384]
                //   - Right piece processes: input[16385..32770]   → output[16384..32768]
                //   - There's a 3-element overlap for the 3x3 filter at the boundary
                IndexType input_w_split = (left_wo * ConvStrideW_) - InLeftPadW_ +
                                         (X_ - 1) * ConvDilationW_;

                // === STEP 3: Calculate right piece input dimension ===
                // Right piece gets remaining input elements
                // Example: Wi=32770, input_w_split=16385
                //   right_transformer.Wi_ = 32770 - 16385 = 16385
                right_transformer.Wi_ = Wi_ - input_w_split;

                // Right piece has no left padding (starts in middle of original tensor)
                right_transformer.InLeftPadW_ = 0;

                // === STEP 4: Calculate memory offsets (in elements) ===
                // Input offset: Where right piece starts reading
                // Example: input_w_split=16385, G=1, C=32
                //   input_offset = 16385 * 1 * 32 = 524320 elements
                input_offset = input_w_split * G_ * C_;

                // Output offset: Where right piece starts writing
                // Example: left_wo=16384, G=1, K=16
                //   output_offset = 16384 * 1 * 16 = 262144 elements
                output_offset = left_wo * G_ * K_;
            }
        }

        return make_tuple(left_transformer, right_transformer, input_offset, output_offset);
    }

    // Static helper to launch kernel with split-image if needed (recursive queue approach)
    // This implements old CK's binary splitting: 10GB → 5GB+5GB → 2.5GB+2.5GB → ...
    template<typename Kernel, index_t kBlockPerCu = 1>
    CK_TILE_HOST static float LaunchKernelWithSplitIfNeeded(
        const GroupedConvFwdHostArgs& args,
        const stream_config& s)
    {

        // Structure to hold a split problem in the queue
        struct SplitProblem {
            TransformConvFwdToGemm transformer;
            long_index_t input_offset;
            long_index_t output_offset;
            int depth;  // For debugging
        };

        // Extract the transformer from kernel args creation
        // We need to recreate it from args
        std::array<IndexType, NDimSpatial + 3> in_lengths;
        std::array<IndexType, NDimSpatial + 3> wei_lengths;
        std::array<IndexType, NDimSpatial + 3> out_lengths;

        in_lengths[0] = args.G_;
        in_lengths[1] = args.N_;
        in_lengths[2] = args.C_;

        wei_lengths[0] = args.G_;
        wei_lengths[1] = args.K_;
        wei_lengths[2] = args.C_;

        out_lengths[0] = args.G_;
        out_lengths[1] = args.N_;
        out_lengths[2] = args.K_;

        // Add spatial dimensions
        for(index_t i = 0; i < NDimSpatial; i++) {
            in_lengths[3 + i] = static_cast<IndexType>(args.input_spatial_lengths_[i]);
            wei_lengths[3 + i] = static_cast<IndexType>(args.filter_spatial_lengths_[i]);
            out_lengths[3 + i] = static_cast<IndexType>(args.output_spatial_lengths_[i]);
        }

        // Create arrays for spatial parameters
        std::array<IndexType, NDimSpatial> conv_strides;
        std::array<IndexType, NDimSpatial> conv_dilations;
        std::array<IndexType, NDimSpatial> input_left_pads;
        std::array<IndexType, NDimSpatial> input_right_pads;

        for(index_t i = 0; i < NDimSpatial; i++) {
            conv_strides[i] = static_cast<IndexType>(args.conv_filter_strides_[i]);
            conv_dilations[i] = static_cast<IndexType>(args.conv_filter_dilations_[i]);
            input_left_pads[i] = static_cast<IndexType>(args.input_left_pads_[i]);
            input_right_pads[i] = static_cast<IndexType>(args.input_right_pads_[i]);
        }

        // Quick check before creating transformer - avoid crash for huge tensors
        if(s.log_level_ > 0) {
            std::cout << "[SPLIT-IMAGE] Entering LaunchKernelWithSplitIfNeeded\n";
            std::cout << "[SPLIT-IMAGE] Args: N=" << args.N_ << " C=" << args.C_ << " K=" << args.K_
                      << " G=" << args.G_ << " input_spatial[0]=" << args.input_spatial_lengths_[0] << "\n";
        }

        const long_index_t input_size_estimate =
            static_cast<long_index_t>(args.N_) *
            static_cast<long_index_t>(args.C_) *
            static_cast<long_index_t>(args.input_spatial_lengths_[0]) *
            (args.input_spatial_lengths_.size() > 1 ? args.input_spatial_lengths_[1] : 1) *
            (args.input_spatial_lengths_.size() > 2 ? args.input_spatial_lengths_[2] : 1);

        const long_index_t output_size_estimate =
            static_cast<long_index_t>(args.N_) *
            static_cast<long_index_t>(args.K_) *
            static_cast<long_index_t>(args.output_spatial_lengths_[0]) *
            (args.output_spatial_lengths_.size() > 1 ? args.output_spatial_lengths_[1] : 1) *
            (args.output_spatial_lengths_.size() > 2 ? args.output_spatial_lengths_[2] : 1);

        const long_index_t threshold = TwoGB / sizeof(ADataType);
        const bool needs_split = (input_size_estimate >= threshold) || (output_size_estimate >= threshold);

        if(s.log_level_ > 0) {
            std::cout << "[SPLIT-IMAGE] Size check: input=" << input_size_estimate
                      << " output=" << output_size_estimate
                      << " threshold=" << threshold
                      << " needs_split=" << needs_split << "\n";
        }

        // Check if split is needed BEFORE creating transformer
        if(!needs_split) {
            if(s.log_level_ > 0) {
                std::cout << "[SPLIT-IMAGE] No split needed - tensors fit in memory threshold\n";
            }

            // Create kernel args only if no split is needed
            auto kargs = Kernel::MakeKernelArgs(args);

            // No split needed - launch original kernel with validation and logging
            const dim3 grids = Kernel::GridSize(kargs);
            const dim3 blocks = Kernel::BlockSize();

            if(!Kernel::IsSupportedArgument(kargs)) {
                throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
            }

            if(s.log_level_ > 0) {
                std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                          << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                          << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z
                          << "}" << std::endl;
            }

            return ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
        }

        // 1D SPLIT-IMAGE IMPLEMENTATION
        // Binary split approach: split the last spatial dimension (W for 1D, W for 2D, etc.)
        // For NDimSpatial dimensions, the last spatial dim is at index [3 + NDimSpatial - 1]
        constexpr index_t last_spatial_idx = 3 + NDimSpatial - 1;

        if(s.log_level_ > 0) {
            std::cout << "[SPLIT-IMAGE 1D] Split needed - last_spatial_dim[" << last_spatial_idx
                      << "]=" << in_lengths[last_spatial_idx] << "\n";
        }

        // Split last spatial dimension (W) in half
        const long_index_t w_out = out_lengths[last_spatial_idx];

        // Split output width in half
        const long_index_t w_out_piece1 = w_out / 2;
        const long_index_t w_out_piece2 = w_out - w_out_piece1;

        // Calculate input width needed for each output piece
        // input_w = (output_w - 1) * stride + filter_w
        const long_index_t filter_w = wei_lengths[last_spatial_idx];
        const long_index_t stride_w = conv_strides[NDimSpatial - 1];  // Last spatial stride
        const long_index_t dilation_w = conv_dilations[NDimSpatial - 1];  // Last spatial dilation
        const long_index_t dilated_filter_w = (filter_w - 1) * dilation_w + 1;

        // Piece 1: Calculate needed input for output
        const long_index_t w_in_piece1 = (w_out_piece1 - 1) * stride_w + dilated_filter_w;

        // Piece 2: Calculate needed input, but clamp to available input
        // From old CK: Wi_right = min(Wi - wi_start_idx, theoretical_needed)
        const long_index_t w_in_available = in_lengths[last_spatial_idx] - (w_out_piece1 * stride_w);
        const long_index_t w_in_theoretical = (w_out_piece2 - 1) * stride_w + dilated_filter_w;
        const long_index_t w_in_piece2 = (w_in_available < w_in_theoretical) ? w_in_available : w_in_theoretical;

        const long_index_t orig_left_pad = input_left_pads[NDimSpatial - 1];
        const long_index_t orig_right_pad = input_right_pads[NDimSpatial - 1];

        if(s.log_level_ > 0) {
            std::cout << "[SPLIT-IMAGE 1D] Original: W_in=" << in_lengths[last_spatial_idx]
                      << " W_out=" << w_out
                      << " left_pad=" << orig_left_pad
                      << " right_pad=" << orig_right_pad
                      << " stride=" << stride_w
                      << " filter=" << filter_w << "\n";
            std::cout << "[SPLIT-IMAGE 1D] Piece 1: w_in=" << w_in_piece1 << " w_out=" << w_out_piece1 << "\n";
            std::cout << "[SPLIT-IMAGE 1D] Piece 2: w_in=" << w_in_piece2
                      << " (available=" << w_in_available
                      << " theoretical=" << w_in_theoretical
                      << ") w_out=" << w_out_piece2 << "\n";
        }

        float total_time = 0.0f;

        const ADataType* orig_in_ptr = static_cast<const ADataType*>(args.in_ptr);
        CDataType* orig_out_ptr = static_cast<CDataType*>(args.out_ptr);

        // Process piece 1 - Also use batch-by-batch to test consistency
        if(s.log_level_ > 0) {
            std::cout << "[SPLIT-IMAGE 1D] About to process piece 1\n";
        }
        {
            auto piece_args = args;
            const long_index_t orig_n = args.N_;

            piece_args.N_ = 1;  // Process one batch at a time
            piece_args.input_spatial_lengths_[NDimSpatial - 1] = w_in_piece1;
            piece_args.output_spatial_lengths_[NDimSpatial - 1] = w_out_piece1;
            // Piece 1 keeps left padding, removes right padding
            piece_args.input_right_pads_[NDimSpatial - 1] = 0;

            // Full batch strides in ORIGINAL tensor (with original spatial dimensions)
            const long_index_t input_batch_stride = in_lengths[last_spatial_idx] * args.C_ * args.G_;
            const long_index_t output_batch_stride = out_lengths[last_spatial_idx] * args.K_ * args.G_;

            if(s.log_level_ > 0) {
                std::cout << "[SPLIT-IMAGE 1D] Processing piece 1: " << orig_n << " batches (batch-by-batch)\n";
                std::cout << "[SPLIT-IMAGE 1D] Piece 1: input_batch_stride=" << input_batch_stride
                          << " output_batch_stride=" << output_batch_stride << "\n";
            }

            // Process each batch separately
            for(long_index_t n = 0; n < orig_n; n++) {
                // Piece 1 has no W offset, only batch offset
                const long_index_t batch_input_offset = n * input_batch_stride;
                const long_index_t batch_output_offset = n * output_batch_stride;

                piece_args.in_ptr = orig_in_ptr + batch_input_offset;
                piece_args.out_ptr = orig_out_ptr + batch_output_offset;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT-IMAGE 1D] Piece 1 batch " << n << ": in_offset=" << batch_input_offset
                              << " out_offset=" << batch_output_offset << "\n";
                }

                auto kargs = Kernel::MakeKernelArgs(piece_args);
                const dim3 grids = Kernel::GridSize(kargs);
                const dim3 blocks = Kernel::BlockSize();

                total_time += ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
            }
        }
        if(s.log_level_ > 0) {
            std::cout << "[SPLIT-IMAGE 1D] Piece 1 completed\n";
        }

        // Process piece 2 - Use N=1 batch-by-batch to avoid stride mismatch
        {
            auto piece_args = args;
            const long_index_t orig_n = args.N_;

            // Set piece 2 dimensions
            piece_args.N_ = 1;  // Process one batch at a time to avoid stride calculation issues
            piece_args.input_spatial_lengths_[NDimSpatial - 1] = w_in_piece2;
            piece_args.output_spatial_lengths_[NDimSpatial - 1] = w_out_piece2;
            piece_args.input_left_pads_[NDimSpatial - 1] = 0;

            // Calculate spatial offsets for piece 2 (right piece)
            // Output offset is simply where piece 1 ends
            const long_index_t w_offset_out = w_out_piece1;

            // Input offset: Use same formula as SplitConvProblem
            // Formula: (w_offset_out * stride) - left_pad + (filter - 1) * dilation
            // Example: w_offset_out=16384, stride=1, left_pad=1, filter=3, dilation=1
            //   w_offset_in = (16384 * 1) - 1 + (3 - 1) * 1 = 16385
            const long_index_t w_offset_in = (w_offset_out * stride_w) - orig_left_pad +
                                            (filter_w - 1) * dilation_w;

            // Convert W position to element offset (accounting for C and G dimensions)
            const long_index_t input_w_stride = args.C_ * args.G_;
            const long_index_t output_w_stride = args.K_ * args.G_;
            const long_index_t input_offset_per_batch = w_offset_in * input_w_stride;
            const long_index_t output_offset_per_batch = w_offset_out * output_w_stride;

            // Batch strides in ORIGINAL tensor
            const long_index_t input_batch_stride = in_lengths[last_spatial_idx] * args.C_ * args.G_;
            const long_index_t output_batch_stride = out_lengths[last_spatial_idx] * args.K_ * args.G_;

            if(s.log_level_ > 0) {
                std::cout << "[SPLIT-IMAGE 1D] Processing piece 2: " << orig_n << " batches (batch-by-batch to fix stride)\n";
                std::cout << "[SPLIT-IMAGE 1D] Piece 2: w_offset_in=" << w_offset_in
                          << " w_offset_out=" << w_offset_out << "\n";
            }

            // Process each batch separately
            for(long_index_t n = 0; n < orig_n; n++) {
                const long_index_t batch_input_offset = n * input_batch_stride + input_offset_per_batch;
                const long_index_t batch_output_offset = n * output_batch_stride + output_offset_per_batch;

                piece_args.in_ptr = orig_in_ptr + batch_input_offset;
                piece_args.out_ptr = orig_out_ptr + batch_output_offset;

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT-IMAGE 1D] Piece 2 batch " << n
                              << ": in_ptr=" << piece_args.in_ptr
                              << " out_ptr=" << piece_args.out_ptr
                              << " in_offset=" << batch_input_offset
                              << " out_offset=" << batch_output_offset << "\n";
                }

                auto kargs = Kernel::MakeKernelArgs(piece_args);
                const dim3 grids = Kernel::GridSize(kargs);
                const dim3 blocks = Kernel::BlockSize();

                if(s.log_level_ > 0) {
                    std::cout << "[SPLIT-IMAGE 1D] Piece 2 batch " << n
                              << " grid=(" << grids.x << "," << grids.y << "," << grids.z << ")"
                              << " blocks=(" << blocks.x << "," << blocks.y << "," << blocks.z << ")\n";
                }

                total_time += ck_tile::launch_kernel(
                    s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
            }
        }

        if(s.log_level_ > 0) {
            std::cout << "[SPLIT-IMAGE 1D] Both pieces completed, total_time=" << total_time << "ms\n";
        }

        
        return total_time;

        // OLD MULTI-DIMENSIONAL SPLIT CODE (NOT USED):
        /*
        // Don't create the full transformer - it causes memory issues
        // Instead, directly split and create smaller transformers
        std::vector<SplitProblem> ready_list;

        // For 1D: simple binary split approach
        if constexpr(NDimSpatial == 1) {
            // Calculate number of splits needed
            int num_splits = 2;  // Start with 2, increase if needed

            while(num_splits <= 64) {
                // Check if splitting by num_splits makes pieces small enough
                long_index_t piece_w_out = out_lengths[3] / num_splits;
                long_index_t piece_w_in = in_lengths[3] / num_splits + wei_lengths[3];  // Conservative estimate

                long_index_t piece_input_size = static_cast<long_index_t>(args.N_) * args.C_ * piece_w_in;
                long_index_t piece_output_size = static_cast<long_index_t>(args.N_) * args.K_ * piece_w_out;

                if(piece_input_size < threshold && piece_output_size < threshold) {
                    break;
                }
                num_splits *= 2;
            }

            if(s.log_level_ > 0) {
                std::cout << "[SPLIT-IMAGE] Creating " << num_splits << " pieces for 1D convolution\n";
            }

            // Create the split pieces
            for(int i = 0; i < num_splits; i++) {
                // For simplicity, equal splits (can be improved later)
                auto piece_in_lengths = in_lengths;
                auto piece_out_lengths = out_lengths;

                piece_out_lengths[3] = out_lengths[3] / num_splits;
                if(i == num_splits - 1) {
                    // Last piece gets remainder
                    piece_out_lengths[3] = out_lengths[3] - (out_lengths[3] / num_splits) * (num_splits - 1);
                }

                // Calculate input dimension for this output piece
                // This is simplified - proper calculation would consider stride/dilation
                piece_in_lengths[3] = piece_out_lengths[3] * conv_strides[0] + (wei_lengths[3] - 1) * conv_dilations[0];

                // Calculate offsets (simplified)
                long_index_t input_offset = i * (in_lengths[3] / num_splits) * args.C_ * args.G_;
                long_index_t output_offset = i * (out_lengths[3] / num_splits) * args.K_ * args.G_;

                // Adjust padding for non-edge pieces
                auto piece_left_pads = input_left_pads;
                auto piece_right_pads = input_right_pads;
                if(i > 0) piece_left_pads[0] = 0;
                if(i < num_splits - 1) piece_right_pads[0] = 0;

                TransformConvFwdToGemm piece_transformer(
                    piece_in_lengths,
                    wei_lengths,
                    piece_out_lengths,
                    conv_strides,
                    conv_dilations,
                    piece_left_pads,
                    piece_right_pads);

                ready_list.push_back({piece_transformer, input_offset, output_offset, 0});
            }
        } else if constexpr(NDimSpatial == 2) {
            // For 2D: split H dimension first (simpler than splitting both)
            if(s.log_level_ > 0) {
                std::cout << "[SPLIT-IMAGE] 2D conv: H=" << in_lengths[3] << " W=" << in_lengths[4] << "\n";
                std::cout << "[SPLIT-IMAGE] Output: H=" << out_lengths[3] << " W=" << out_lengths[4] << "\n";
            }

            int num_splits = 2;

            while(num_splits <= 64) {
                // Check if splitting H by num_splits makes pieces small enough
                long_index_t piece_h_out = out_lengths[3] / num_splits;  // H is at index 3 for 2D
                long_index_t piece_h_in = in_lengths[3] / num_splits + wei_lengths[3];

                long_index_t piece_input_size = static_cast<long_index_t>(args.N_) * args.C_ *
                                                piece_h_in * in_lengths[4];  // W is at index 4
                long_index_t piece_output_size = static_cast<long_index_t>(args.N_) * args.K_ *
                                                 piece_h_out * out_lengths[4];

                if(piece_input_size < threshold && piece_output_size < threshold) {
                    break;
                }
                num_splits *= 2;
            }

            if(s.log_level_ > 0) {
                std::cout << "[SPLIT-IMAGE] Creating " << num_splits << " pieces for 2D convolution\n";
            }

            // Create the split pieces
            for(int i = 0; i < num_splits; i++) {
                auto piece_in_lengths = in_lengths;
                auto piece_out_lengths = out_lengths;

                // Split H dimension
                piece_out_lengths[3] = out_lengths[3] / num_splits;
                if(i == num_splits - 1) {
                    piece_out_lengths[3] = out_lengths[3] - (out_lengths[3] / num_splits) * (num_splits - 1);
                }

                // Calculate input H for this output piece
                piece_in_lengths[3] = piece_out_lengths[3] * conv_strides[0] +
                                     (wei_lengths[3] - 1) * conv_dilations[0];

                // Calculate offsets
                long_index_t h_offset_out = i * (out_lengths[3] / num_splits);
                long_index_t h_offset_in = h_offset_out * conv_strides[0] - input_left_pads[0];
                if(h_offset_in < 0) h_offset_in = 0;

                long_index_t input_offset = h_offset_in * in_lengths[4] * args.C_ * args.G_;
                long_index_t output_offset = h_offset_out * out_lengths[4] * args.K_ * args.G_;

                // Adjust padding
                auto piece_left_pads = input_left_pads;
                auto piece_right_pads = input_right_pads;
                if(i > 0) piece_left_pads[0] = 0;  // No top padding for non-first pieces
                if(i < num_splits - 1) piece_right_pads[0] = 0;  // No bottom padding for non-last pieces

                TransformConvFwdToGemm piece_transformer(
                    piece_in_lengths,
                    wei_lengths,
                    piece_out_lengths,
                    conv_strides,
                    conv_dilations,
                    piece_left_pads,
                    piece_right_pads);

                ready_list.push_back({piece_transformer, input_offset, output_offset, 0});
            }
        } else {
            // For 3D, would need similar approach
            throw std::runtime_error("Split-Image: 3D not yet implemented in this approach");
        }

        // Queue processing removed - we create pieces directly

        if(s.log_level_ > 0) {
            std::cout << "[SPLIT-IMAGE] Total pieces after splitting: " << ready_list.size() << "\n";
        }

        // Launch kernel for each split piece
        float total_time = 0.0f;

        for(size_t i = 0; i < ready_list.size(); i++) {
            const auto& piece = ready_list[i];

            // Create modified args for this piece
            auto piece_args = args;  // Copy original args

            // Update the input and output spatial dimensions for this piece
            // For 1D: index 0, for 2D: indices 0,1, for 3D: indices 0,1,2
            if constexpr(NDimSpatial >= 1) {
                // For 1D convolution, spatial dimension is at index 0
                piece_args.input_spatial_lengths_[0] = piece.transformer.Wi_;
                piece_args.output_spatial_lengths_[0] = piece.transformer.Wo_;
                if(i == 0) {  // Only print for first piece
                    std::cout << "[DEBUG] Piece 1: Setting input_spatial[0]=" << piece.transformer.Wi_
                              << " output_spatial[0]=" << piece.transformer.Wo_ << "\n";
                }
            }
            if constexpr(NDimSpatial >= 2) {
                // For 2D: H at index 0, W at index 1
                piece_args.input_spatial_lengths_[0] = piece.transformer.Hi_;
                piece_args.output_spatial_lengths_[0] = piece.transformer.Ho_;
                piece_args.input_spatial_lengths_[1] = piece.transformer.Wi_;
                piece_args.output_spatial_lengths_[1] = piece.transformer.Wo_;
            }
            if constexpr(NDimSpatial >= 3) {
                // For 3D: D at index 0, H at index 1, W at index 2
                piece_args.input_spatial_lengths_[0] = piece.transformer.Di_;
                piece_args.output_spatial_lengths_[0] = piece.transformer.Do_;
                piece_args.input_spatial_lengths_[1] = piece.transformer.Hi_;
                piece_args.output_spatial_lengths_[1] = piece.transformer.Ho_;
                piece_args.input_spatial_lengths_[2] = piece.transformer.Wi_;
                piece_args.output_spatial_lengths_[2] = piece.transformer.Wo_;
            }

            // Debug: Print what we're passing to kernel
            if(s.log_level_ > 1) {
                std::cout << "[SPLIT-IMAGE] Piece " << (i+1) << "/" << ready_list.size() << ":\n";
                std::cout << "  Transformer dims: Wi=" << piece.transformer.Wi_
                          << " Wo=" << piece.transformer.Wo_;
                if constexpr(NDimSpatial >= 2) {
                    std::cout << " Hi=" << piece.transformer.Hi_
                              << " Ho=" << piece.transformer.Ho_;
                }
                if constexpr(NDimSpatial >= 3) {
                    std::cout << " Di=" << piece.transformer.Di_
                              << " Do=" << piece.transformer.Do_;
                }
                std::cout << "\n  Args spatial: ";
                for(size_t j = 0; j < piece_args.input_spatial_lengths_.size(); j++) {
                    if(j > 0) std::cout << "x";
                    std::cout << piece_args.input_spatial_lengths_[j];
                }
                std::cout << " -> ";
                for(size_t j = 0; j < piece_args.output_spatial_lengths_.size(); j++) {
                    if(j > 0) std::cout << "x";
                    std::cout << piece_args.output_spatial_lengths_[j];
                }
                std::cout << "\n  Offsets: in=" << piece.input_offset
                          << ", out=" << piece.output_offset << "\n";
            }

            // Update padding for this piece
            if constexpr(NDimSpatial >= 1) {
                piece_args.input_left_pads_[NDimSpatial - 1] = piece.transformer.InLeftPadW_;
                piece_args.input_right_pads_[NDimSpatial - 1] = piece.transformer.InRightPadW_;
            }
            if constexpr(NDimSpatial >= 2) {
                piece_args.input_left_pads_[NDimSpatial - 2] = piece.transformer.InLeftPadH_;
                piece_args.input_right_pads_[NDimSpatial - 2] = piece.transformer.InRightPadH_;
            }
            if constexpr(NDimSpatial >= 3) {
                piece_args.input_left_pads_[NDimSpatial - 3] = piece.transformer.InLeftPadD_;
                piece_args.input_right_pads_[NDimSpatial - 3] = piece.transformer.InRightPadD_;
            }

            // Adjust pointers to the correct offset
            piece_args.in_ptr = static_cast<const void*>(
                static_cast<const ADataType*>(args.in_ptr) + piece.input_offset);
            piece_args.out_ptr = static_cast<void*>(
                static_cast<CDataType*>(args.out_ptr) + piece.output_offset);

            // Create kernel args for this piece
            auto piece_kargs = Kernel::MakeKernelArgs(piece_args);

            // Get grid and block sizes for this piece
            const dim3 grids = Kernel::GridSize(piece_kargs);
            const dim3 blocks = Kernel::BlockSize();

            if(s.log_level_ > 0) {
                std::cout << "[SPLIT-IMAGE] Launching piece " << (i+1) << "/" << ready_list.size()
                          << " with offsets: in=" << piece.input_offset
                          << ", out=" << piece.output_offset << "\n";
            }

            // Launch kernel for this piece
            float piece_time = ck_tile::launch_kernel(
                s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, piece_kargs));

            total_time += piece_time;
        }

        // Return average time per kernel launch
        return total_time / ready_list.size();
        */
    }
#endif  // End of outdated split-image code


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

        if constexpr (NDimSpatial == 1) {
            out_total = Wo_;
            in_total = Wi_;
            left_pad = InLeftPadW_;
            right_pad = InRightPadW_;
            stride = ConvStrideW_;
            dilation = ConvDilationW_;
            filter = X_;
        } else if constexpr (NDimSpatial == 2) {
            out_total = Ho_;
            in_total = Hi_;
            left_pad = InLeftPadH_;
            right_pad = InRightPadH_;
            stride = ConvStrideH_;
            dilation = ConvDilationH_;
            filter = Y_;
        } else if constexpr (NDimSpatial == 3) {
            out_total = Do_;
            in_total = Di_;
            left_pad = InLeftPadD_;
            right_pad = InRightPadD_;
            stride = ConvStrideD_;
            dilation = ConvDilationD_;
            filter = Z_;
        } else {
            return info;  // Unsupported dimension
        }

        // DEBUG: Print N values used in calculation
        std::cout << "[DEBUG CalculateSplitImage] N_=" << N_
                  << ", original_N_=" << original_N_ << std::endl;

        // Check if split is needed - IMPORTANT: Use N_ (after Split-N!)
        long_index_t output_size;
        if constexpr (NDimSpatial == 1) {
            output_size = static_cast<long_index_t>(N_) *
                         static_cast<long_index_t>(Wo_) *
                         static_cast<long_index_t>(K_) *
                         static_cast<long_index_t>(G_);
        } else if constexpr (NDimSpatial == 2) {
            output_size = static_cast<long_index_t>(N_) *
                         static_cast<long_index_t>(Ho_) *
                         static_cast<long_index_t>(Wo_) *
                         static_cast<long_index_t>(K_) *
                         static_cast<long_index_t>(G_);
        } else if constexpr (NDimSpatial == 3) {
            output_size = static_cast<long_index_t>(N_) *
                         static_cast<long_index_t>(Do_) *
                         static_cast<long_index_t>(Ho_) *
                         static_cast<long_index_t>(Wo_) *
                         static_cast<long_index_t>(K_) *
                         static_cast<long_index_t>(G_);
        }

        if (output_size < threshold_elements) {
            return info;
        }

        // Binary split
        info.out_left = out_total / 2;
        info.out_right = out_total - info.out_left;

        // Effective filter size
        const IndexType x_eff = (filter - 1) * dilation + 1;

        // Safety checks
        const IndexType right_start = info.out_left * stride;
        const IndexType left_end = (info.out_left - 1) * stride + x_eff;

        const bool is_possible_to_split =
            out_total != 1 &&
            right_start > left_pad &&
            left_end <= (left_pad + in_total);

        if (!is_possible_to_split) {
            return info;  // Cannot split safely
        }

        info.should_split = true;

        // Calculate input sizes
        const IndexType in_left_end = (info.out_left - 1) * stride + x_eff;
        info.in_left = in_left_end - left_pad;

        const IndexType in_right_start = info.out_left * stride;
        const IndexType in_right_available = in_total - (in_right_start - left_pad);
        info.in_right = ck_tile::min(in_right_available,
                                     (info.out_right - 1) * stride + x_eff);

        // Calculate physical offset
        const IndexType physical_offset = (info.out_left * stride) - left_pad;

        // Calculate strides - for WITHIN a single batch
        // The stride to jump from one W-position to the next WITHIN the same batch
        long_index_t input_stride, output_stride;
        if constexpr (NDimSpatial == 1) {
            // 1D NWGC: stride_W = G * C (within ONE batch)
            input_stride = static_cast<long_index_t>(G_) *
                          static_cast<long_index_t>(C_);
            output_stride = static_cast<long_index_t>(G_) *
                           static_cast<long_index_t>(K_);
        } else if constexpr (NDimSpatial == 2) {
            // 2D NHWGC: stride_H = W_in * G * C (within ONE batch)
            input_stride = static_cast<long_index_t>(Wi_) *
                          static_cast<long_index_t>(G_) *
                          static_cast<long_index_t>(C_);
            output_stride = static_cast<long_index_t>(Wo_) *
                           static_cast<long_index_t>(G_) *
                           static_cast<long_index_t>(K_);
        } else if constexpr (NDimSpatial == 3) {
            // 3D NDHWGC: stride_D = H_in * W_in * G * C (within ONE batch)
            input_stride = static_cast<long_index_t>(Hi_) *
                          static_cast<long_index_t>(Wi_) *
                          static_cast<long_index_t>(G_) *
                          static_cast<long_index_t>(C_);
            output_stride = static_cast<long_index_t>(Ho_) *
                           static_cast<long_index_t>(Wo_) *
                           static_cast<long_index_t>(G_) *
                           static_cast<long_index_t>(K_);
        }

        // Calculate offsets in elements
        info.input_offset = static_cast<long_index_t>(physical_offset) * input_stride;
        info.output_offset = static_cast<long_index_t>(info.out_left) * output_stride;

        // DEBUG: Print stride and offset calculation
        std::cout << "[DEBUG CalculateSplitImage] input_stride=" << input_stride
                  << ", output_stride=" << output_stride << std::endl;
        std::cout << "[DEBUG CalculateSplitImage] physical_offset=" << physical_offset
                  << ", out_left=" << info.out_left << std::endl;
        std::cout << "[DEBUG CalculateSplitImage] input_offset=" << info.input_offset
                  << ", output_offset=" << info.output_offset << std::endl;

        // Padding adjustments
        info.left_pad_left = left_pad;
        info.right_pad_left = 0;
        info.left_pad_right = 0;
        info.right_pad_right = right_pad;

        return info;
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
