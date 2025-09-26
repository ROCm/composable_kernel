
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "ck_tile/core.hpp"
#include "ck_tile/ops/grouped_convolution/utils/convolution_specialization.hpp"

namespace ck_tile {

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
    static constexpr long_index_t TwoGB = 10L * 1024L * 1024L;  // 10MB for testing

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

        // Store original N (this was missing in 1D constructor - bug fix)
        original_N_ = c_g_n_k_wos_lengths[I1];

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(a_g_n_c_wis_lengths, c_g_n_k_wos_lengths);
        }
        else
        {
            N_ = c_g_n_k_wos_lengths[I1];
            original_N_ = N_;
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

        // Store original N
        original_N_ = c_g_n_k_wos_lengths[I1];

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(a_g_n_c_wis_lengths, c_g_n_k_wos_lengths);
        }
        else
        {
            N_          = c_g_n_k_wos_lengths[I1];
            original_N_ = N_;
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

        // Store original N before potential splitting
        original_N_ = c_g_n_k_wos_lengths[I1];

        if constexpr(SplitN)
        {
            N_ = GetSplitedNSize(a_g_n_c_wis_lengths, c_g_n_k_wos_lengths);
        }
        else
        {
            N_ = original_N_;
        }
    }

    // NEW: Unified split analysis structure
    struct SplitDecision {
        // Decision flags
        bool needs_split = false;          // Do tensors exceed threshold?
        bool can_split = false;            // Is split physically possible?
        IndexType split_dimension = -1;    // 0=D, 1=H, 2=W, -1=none

        // Pre-calculated dimension information
        struct DimensionInfo {
            IndexType right_start_idx = 0;        // Where right split starts in input
            bool is_splittable = false;           // Can this dimension be split?
        } dim_info[3];  // [0]=D, [1]=H, [2]=W

        // Split parameters (valid when can_split=true)
        IndexType left_output_size = 0;   // e.g., Ho/2 for H-split
        IndexType right_output_size = 0;  // e.g., Ho - Ho/2
    };

    // NEW: Analyze split requirements (combines AreDescriptorsSmallerThan2GB + CanSplitImage)
    __host__ SplitDecision AnalyzeSplitRequirements() const
    {
        SplitDecision decision;

        // STEP 1: Check if split needed (replaces AreDescriptorsSmallerThan2GB logic)
        const long_index_t input_tensor_size =
            static_cast<long_index_t>(N_) *
            static_cast<long_index_t>(Di_) *
            static_cast<long_index_t>(Hi_) *
            static_cast<long_index_t>(Wi_) *
            static_cast<long_index_t>(C_);

        const long_index_t output_tensor_size =
            static_cast<long_index_t>(N_) *
            static_cast<long_index_t>(Do_) *
            static_cast<long_index_t>(Ho_) *
            static_cast<long_index_t>(Wo_) *
            static_cast<long_index_t>(K_);

        const long_index_t input_bytes = input_tensor_size * sizeof(ADataType);
        const long_index_t output_bytes = output_tensor_size * sizeof(CDataType);

        decision.needs_split = (input_bytes > TwoGB) || (output_bytes > TwoGB);

        // Debug output
        std::cerr << "[DEBUG Split-Image] Input=" << (input_bytes / 1024 / 1024) << "MB\n";
        std::cerr << "[DEBUG Split-Image] Output=" << (output_bytes / 1024 / 1024) << "MB\n";

        if (!decision.needs_split) {
            return decision;  // Early exit - no split needed
        }

        std::cerr << "[DEBUG Split-Image] Exceeds " << (TwoGB / 1024 / 1024) << "MB threshold:\n";

        // STEP 2: Pre-calculate all dimension info ONCE (eliminates duplication)

        // D dimension (for 3D)
        if constexpr(NDimSpatial == 3) {
            const IndexType z_eff = (Z_ - 1) * ConvDilationD_ + 1;
            decision.dim_info[0].right_start_idx = (Do_ / 2) * ConvStrideD_;
            const IndexType left_end_idx = (Do_ / 2 - 1) * ConvStrideD_ + z_eff;
            decision.dim_info[0].is_splittable =
                (Do_ > 1) &&
                (decision.dim_info[0].right_start_idx > InLeftPadD_) &&
                (left_end_idx <= (InLeftPadD_ + Di_));
        }

        // H dimension (for 2D and 3D)
        if constexpr(NDimSpatial >= 2) {
            const IndexType y_eff = (Y_ - 1) * ConvDilationH_ + 1;
            decision.dim_info[1].right_start_idx = (Ho_ / 2) * ConvStrideH_;
            const IndexType left_end_idx = (Ho_ / 2 - 1) * ConvStrideH_ + y_eff;
            decision.dim_info[1].is_splittable =
                (Ho_ > 1) &&
                (decision.dim_info[1].right_start_idx > InLeftPadH_) &&
                (left_end_idx <= (InLeftPadH_ + Hi_));
        }

        // W dimension (for 1D, 2D, and 3D)
        const IndexType x_eff = (X_ - 1) * ConvDilationW_ + 1;
        decision.dim_info[2].right_start_idx = (Wo_ / 2) * ConvStrideW_;
        const IndexType w_left_end_idx = (Wo_ / 2 - 1) * ConvStrideW_ + x_eff;
        decision.dim_info[2].is_splittable =
            (Wo_ > 1) &&
            (decision.dim_info[2].right_start_idx > InLeftPadW_) &&
            (w_left_end_idx <= (InLeftPadW_ + Wi_));

        // STEP 3: Decide which dimension to split based on NDimSpatial
        if constexpr(NDimSpatial == 3) {
            // 3D: ONLY check D dimension (no H/W to maintain memory contiguity)
            if (decision.dim_info[0].is_splittable) {
                decision.can_split = true;
                decision.split_dimension = 0;
                decision.left_output_size = Do_ / 2;
                decision.right_output_size = Do_ - Do_ / 2;
                std::cerr << "[DEBUG Split-Image] 3D Conv can split on D dimension\n";
            } else {
                std::cerr << "[DEBUG Split-Image] 3D Conv CANNOT split (D not splittable)\n";
            }
        }
        else if constexpr(NDimSpatial == 2) {
            // 2D: Try H first (preferred), then W as fallback
            if (decision.dim_info[1].is_splittable) {
                decision.can_split = true;
                decision.split_dimension = 1;
                decision.left_output_size = Ho_ / 2;
                decision.right_output_size = Ho_ - Ho_ / 2;
                std::cerr << "[DEBUG Split-Image] 2D Conv can split on H dimension\n";
            } else if (decision.dim_info[2].is_splittable) {
                decision.can_split = true;
                decision.split_dimension = 2;
                decision.left_output_size = Wo_ / 2;
                decision.right_output_size = Wo_ - Wo_ / 2;
                std::cerr << "[DEBUG Split-Image] 2D Conv can split on W dimension (fallback)\n";
            } else {
                std::cerr << "[DEBUG Split-Image] 2D Conv CANNOT split (neither H nor W splittable)\n";
            }
        }
        else { // NDimSpatial == 1
            // 1D: Only W dimension exists
            if (decision.dim_info[2].is_splittable) {
                decision.can_split = true;
                decision.split_dimension = 2;
                decision.left_output_size = Wo_ / 2;
                decision.right_output_size = Wo_ - Wo_ / 2;
                std::cerr << "[DEBUG Split-Image] 1D Conv can split on W dimension\n";
            } else {
                std::cerr << "[DEBUG Split-Image] 1D Conv CANNOT split (W not splittable)\n";
            }
        }

        return decision;
    }

    // REFACTORED: Now accepts SplitDecision to avoid recalculation
    __host__ auto SplitConvProblem(const SplitDecision& decision) const
    {
        // Guard clause - return zeros if split not possible
        if (!decision.can_split) {
            return ck_tile::make_tuple(IndexType{0}, IndexType{0},
                                      long_index_t{0}, long_index_t{0});
        }

        // Use pre-calculated values from decision
        const auto& split_info = decision.dim_info[decision.split_dimension];
        const IndexType right_start_idx = split_info.right_start_idx;

        // Calculate offsets and GemmM dimensions based on split dimension
        IndexType left_gemm_m, right_gemm_m;
        long_index_t a_right_offset, c_right_offset;

        if (decision.split_dimension == 0) {
            // D-split (3D only)
            a_right_offset = (right_start_idx - InLeftPadD_) * Hi_ * Wi_;
            c_right_offset = decision.left_output_size * Ho_ * Wo_;

            left_gemm_m  = N_ * decision.left_output_size * Ho_ * Wo_;
            right_gemm_m = N_ * decision.right_output_size * Ho_ * Wo_;
        }
        else if (decision.split_dimension == 1) {
            // H-split (2D only - 3D uses D-split)
            a_right_offset = (right_start_idx - InLeftPadH_) * Wi_;
            c_right_offset = decision.left_output_size * Wo_;

            left_gemm_m  = N_ * decision.left_output_size * Wo_;
            right_gemm_m = N_ * decision.right_output_size * Wo_;
        }
        else { // decision.split_dimension == 2
            // W-split (1D, 2D, 3D)
            a_right_offset = right_start_idx - InLeftPadW_;
            c_right_offset = decision.left_output_size;

            if constexpr (NDimSpatial == 1) { // 1D
                left_gemm_m  = N_ * decision.left_output_size;
                right_gemm_m = N_ * decision.right_output_size;
            } else if constexpr (NDimSpatial == 2) { // 2D
                left_gemm_m  = N_ * Ho_ * decision.left_output_size;
                right_gemm_m = N_ * Ho_ * decision.right_output_size;
            } else { // 3D
                left_gemm_m  = N_ * Do_ * Ho_ * decision.left_output_size;
                right_gemm_m = N_ * Do_ * Ho_ * decision.right_output_size;
            }
        }

        return ck_tile::make_tuple(IndexType(left_gemm_m),
                                  IndexType(right_gemm_m),
                                  long_index_t(a_right_offset),
                                  long_index_t(c_right_offset));
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
