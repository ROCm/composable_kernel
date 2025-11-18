// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/core/tensor/tile_elementwise.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_forward.hpp"
#endif

namespace ck_tile {

/// @brief The Grouped Convolution kernel device arguments.
template <typename GroupedConvTraitsType_, typename CDElementwise_>
struct GroupedConvFwdKernelArgs
{

    using ConvToGemmFwdTransformer =
        TransformConvFwdToGemm<GroupedConvTraitsType_::NDimSpatial,
                               GroupedConvTraitsType_::ConvSpecialization,
                               GroupedConvTraitsType_::VectorSizeA,
                               GroupedConvTraitsType_::VectorSizeB,
                               GroupedConvTraitsType_::VectorSizeC,
                               GroupedConvTraitsType_::NumGroupsToMerge,
                               true>; // Split N enabled
    using CDElementwise                 = CDElementwise_;
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvFwdKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& args)
        : elfunc(args.elfunc)
    {
        in_g_n_c_wis_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0])};
        wei_g_k_c_xs_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.output_spatial_lengths_[0])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0])};

        k_batch = args.k_batch;

        // GemmM will be set after Split-N calculation
        GemmN     = args.K_;
        GemmK     = args.C_ * args.filter_spatial_lengths_[0];
        GemmBatch = args.G_;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        // Create and STORE transformer (for split-image support)
        transformer_ = ConvToGemmFwdTransformer{in_g_n_c_wis_lengths,
                                                wei_g_k_c_xs_lengths,
                                                out_g_n_k_wos_lengths,
                                                conv_filter_strides,
                                                conv_filter_dilations,
                                                input_left_pads,
                                                input_right_pads};

        a_grid_desc_m_k =
            transformer_.template MakeADescriptor_M_K<typename GroupedConvTraitsType_::InLayout>();
        b_grid_desc_n_k =
            transformer_.template MakeBDescriptor_N_K<typename GroupedConvTraitsType_::WeiLayout>();
        c_grid_desc_m_n =
            transformer_.template MakeCDescriptor_M_N<typename GroupedConvTraitsType_::OutLayout>();

        group_stride_a = args.C_;
        group_stride_b = args.K_ * args.C_ *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());
        group_stride_c = args.K_;

        // Initialize Split-N support fields for 1D convolution (NWGC layout)
        // Get the actual split N from transformer
        n_per_split = transformer_.GetN();
        original_n  = transformer_.GetOriginalN();
        n_splits    = ck_tile::integer_divide_ceil(original_n, n_per_split);

        // Calculate batch strides using the original argument dimensions.
        // These are the original dimensions passed to the constructor, not modified by the invoker
        // yet. (The invoker modifies args after calling MakeKernelArgs.) VERIFIED: G_ MUST be
        // included - NWGC layout has all groups within each batch
        input_batch_stride  = args.G_ * args.C_ * args.input_spatial_lengths_[0];
        output_batch_stride = args.G_ * args.K_ * args.output_spatial_lengths_[0];

        // Update GemmM to use split N (not original N)
        GemmM = n_per_split * args.output_spatial_lengths_[0];
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvFwdKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& args)
        : elfunc(args.elfunc)
    {
        in_g_n_c_wis_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0]),
                                 static_cast<index_t>(args.input_spatial_lengths_[1])};
        wei_g_k_c_xs_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[1])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.output_spatial_lengths_[0]),
                                 static_cast<index_t>(args.output_spatial_lengths_[1])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0]),
                                 static_cast<index_t>(args.conv_filter_strides_[1])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0]),
                                 static_cast<index_t>(args.conv_filter_dilations_[1])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0]),
                                 static_cast<index_t>(args.input_left_pads_[1])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0]),
                                 static_cast<index_t>(args.input_right_pads_[1])};

        k_batch = args.k_batch;

        // Note: GemmM will be set after Split-N calculation
        GemmN     = args.K_;
        GemmK     = args.C_ * args.filter_spatial_lengths_[0] * args.filter_spatial_lengths_[1];
        GemmBatch = args.G_;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        // Create and STORE transformer (for split-image support)
        transformer_ = ConvToGemmFwdTransformer{in_g_n_c_wis_lengths,
                                                wei_g_k_c_xs_lengths,
                                                out_g_n_k_wos_lengths,
                                                conv_filter_strides,
                                                conv_filter_dilations,
                                                input_left_pads,
                                                input_right_pads};

        a_grid_desc_m_k =
            transformer_.template MakeADescriptor_M_K<typename GroupedConvTraitsType_::InLayout>();
        b_grid_desc_n_k =
            transformer_.template MakeBDescriptor_N_K<typename GroupedConvTraitsType_::WeiLayout>();
        c_grid_desc_m_n =
            transformer_.template MakeCDescriptor_M_N<typename GroupedConvTraitsType_::OutLayout>();

        group_stride_a = args.C_;
        group_stride_b = args.K_ * args.C_ *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());
        group_stride_c = args.K_;

        // Initialize Split-N support fields for 2D convolution (NHWGC layout)
        // Get the actual split N from transformer
        n_per_split = transformer_.GetN();
        original_n  = transformer_.GetOriginalN();
        n_splits    = ck_tile::integer_divide_ceil(original_n, n_per_split);

        // Calculate batch strides for NHWGC layout
        // VERIFIED: G_ MUST be included - NHWGC layout has all groups within each batch
        input_batch_stride =
            args.G_ * args.C_ * args.input_spatial_lengths_[0] * args.input_spatial_lengths_[1];
        output_batch_stride =
            args.G_ * args.K_ * args.output_spatial_lengths_[0] * args.output_spatial_lengths_[1];

        // Update GemmM to use split N (not original N)
        GemmM = n_per_split * args.output_spatial_lengths_[0] * args.output_spatial_lengths_[1];
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NDHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKZYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NDHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvFwdKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& args)
        : elfunc(args.elfunc)
    {
        in_g_n_c_wis_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.input_spatial_lengths_[0]),
                                 static_cast<index_t>(args.input_spatial_lengths_[1]),
                                 static_cast<index_t>(args.input_spatial_lengths_[2])};
        wei_g_k_c_xs_lengths  = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.C_),
                                 static_cast<index_t>(args.filter_spatial_lengths_[0]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[1]),
                                 static_cast<index_t>(args.filter_spatial_lengths_[2])};
        out_g_n_k_wos_lengths = {static_cast<index_t>(args.G_),
                                 static_cast<index_t>(args.N_),
                                 static_cast<index_t>(args.K_),
                                 static_cast<index_t>(args.output_spatial_lengths_[0]),
                                 static_cast<index_t>(args.output_spatial_lengths_[1]),
                                 static_cast<index_t>(args.output_spatial_lengths_[2])};

        conv_filter_strides   = {static_cast<index_t>(args.conv_filter_strides_[0]),
                                 static_cast<index_t>(args.conv_filter_strides_[1]),
                                 static_cast<index_t>(args.conv_filter_strides_[2])};
        conv_filter_dilations = {static_cast<index_t>(args.conv_filter_dilations_[0]),
                                 static_cast<index_t>(args.conv_filter_dilations_[1]),
                                 static_cast<index_t>(args.conv_filter_dilations_[2])};
        input_left_pads       = {static_cast<index_t>(args.input_left_pads_[0]),
                                 static_cast<index_t>(args.input_left_pads_[1]),
                                 static_cast<index_t>(args.input_left_pads_[2])};
        input_right_pads      = {static_cast<index_t>(args.input_right_pads_[0]),
                                 static_cast<index_t>(args.input_right_pads_[1]),
                                 static_cast<index_t>(args.input_right_pads_[2])};

        k_batch = args.k_batch;

        // Note: GemmM will be set after Split-N calculation
        GemmN = args.K_;
        GemmK = args.C_ * args.filter_spatial_lengths_[0] * args.filter_spatial_lengths_[1] *
                args.filter_spatial_lengths_[2];
        GemmBatch = args.G_;

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        // Create and STORE transformer (for split-image support)
        transformer_ = ConvToGemmFwdTransformer{in_g_n_c_wis_lengths,
                                                wei_g_k_c_xs_lengths,
                                                out_g_n_k_wos_lengths,
                                                conv_filter_strides,
                                                conv_filter_dilations,
                                                input_left_pads,
                                                input_right_pads};

        a_grid_desc_m_k =
            transformer_.template MakeADescriptor_M_K<typename GroupedConvTraitsType_::InLayout>();
        b_grid_desc_n_k =
            transformer_.template MakeBDescriptor_N_K<typename GroupedConvTraitsType_::WeiLayout>();
        c_grid_desc_m_n =
            transformer_.template MakeCDescriptor_M_N<typename GroupedConvTraitsType_::OutLayout>();

        group_stride_a = args.C_;
        group_stride_b = args.K_ * args.C_ *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());
        group_stride_c = args.K_;

        // Initialize Split-N support fields for 3D convolution (NDHWGC layout)
        // Get the actual split N from transformer
        n_per_split = transformer_.GetN();
        original_n  = transformer_.GetOriginalN();
        n_splits    = ck_tile::integer_divide_ceil(original_n, n_per_split);

        // Calculate batch strides for NDHWGC layout
        // VERIFIED: G_ MUST be included - NDHWGC layout has all groups within each batch
        input_batch_stride = args.G_ * args.C_ * args.input_spatial_lengths_[0] *
                             args.input_spatial_lengths_[1] * args.input_spatial_lengths_[2];
        output_batch_stride = args.G_ * args.K_ * args.output_spatial_lengths_[0] *
                              args.output_spatial_lengths_[1] * args.output_spatial_lengths_[2];

        // Update GemmM to use split N (not original N)
        GemmM = n_per_split * args.output_spatial_lengths_[0] * args.output_spatial_lengths_[1] *
                args.output_spatial_lengths_[2];
    }

    using AGridDescMK = remove_cvref_t<
        decltype(ConvToGemmFwdTransformer{}
                     .template MakeADescriptor_M_K<typename GroupedConvTraitsType_::InLayout>())>;
    using BGridDescNK = remove_cvref_t<
        decltype(ConvToGemmFwdTransformer{}
                     .template MakeBDescriptor_N_K<typename GroupedConvTraitsType_::WeiLayout>())>;
    using CGridDescMN = remove_cvref_t<
        decltype(ConvToGemmFwdTransformer{}
                     .template MakeCDescriptor_M_N<typename GroupedConvTraitsType_::OutLayout>())>;

    static constexpr index_t NonSpatialDims = 3;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> in_g_n_c_wis_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> wei_g_k_c_xs_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType_::NDimSpatial> out_g_n_k_wos_lengths;

    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_strides;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> conv_filter_dilations;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_left_pads;
    array<index_t, GroupedConvTraitsType_::NDimSpatial> input_right_pads;

    index_t k_batch;
    index_t GemmM;
    index_t GemmN;
    index_t GemmK;
    index_t GemmBatch;

    const void* in_ptr;
    const void* wei_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    const CDElementwise elfunc;
    void* out_ptr;

    AGridDescMK a_grid_desc_m_k;
    BGridDescNK b_grid_desc_n_k;
    CGridDescMN c_grid_desc_m_n;

    long_index_t group_stride_a;
    long_index_t group_stride_b;
    long_index_t group_stride_c;

    // Split-N support fields - initialize to safe defaults
    index_t n_splits            = 1; // Number of batch splits (e.g., 2 for 128→64×2)
    index_t n_per_split         = 1; // Batches per split (N_ from transformer)
    index_t original_n          = 1; // Original batch size before splitting
    index_t input_batch_stride  = 0; // Stride to next batch in input tensor
    index_t output_batch_stride = 0; // Stride to next batch in output tensor

    // Split-image support - spatial offsets (applied per-batch in operator())
    long_index_t spatial_offset_in  = 0; // Spatial offset for input (e.g., W/2 for 1D split)
    long_index_t spatial_offset_out = 0; // Spatial offset for output (e.g., W/2 for 1D split)

    // Split-image support - transformer instance
    ConvToGemmFwdTransformer transformer_;

    // Forward declare descriptor types (will be defined after using declarations)
    using ConvToGemmFwdTransformer_t = ConvToGemmFwdTransformer;
    using AGridDescMK_t              = AGridDescMK;
    using CGridDescMN_t              = CGridDescMN;

    // Split-image support: Common data for all pieces
    struct SplitImageInfo
    {
        // Common dimensions (same for all pieces)
        index_t total_d = 1, total_h = 1, total_w = 1; // Total tensor dimensions
        index_t total_spatial = 1; // Pre-calculated: total_d * total_h * total_w
        index_t num_d_pieces = 1, num_h_pieces = 1, num_w_pieces = 1; // Split factors

        // Minimal per-piece data (only unique values)
        struct PieceInfo
        {
            index_t block_start;               // Starting block index for this piece
            index_t block_end;                 // Ending block index (exclusive)
            index_t d_start, h_start, w_start; // Piece starting position in OUTPUT space
            index_t d_size, h_size, w_size;    // Piece size in OUTPUT space
        };

        static constexpr index_t MaxPieces = 64; // Max pieces: 4 (1D), 16 (2D), 64 (3D)
        std::array<PieceInfo, MaxPieces> pieces; // Array of minimal piece descriptors
    };

    index_t num_spatial_pieces = 1; // Number of spatial pieces (1 = no split)
    SplitImageInfo split_image;     // Nested structure with common + per-piece data
};

/// @brief The Grouped Convolution Forward kernel template.
///
/// @paragraph Overview Overview
///            This class provides the grouped convolution forward kernel template. By semantic
///            division of Implicit GEMM algorithm into following parts we achieve flexible,
///            versatile and robust kernel implementation.
///
///            @li @b Prolog - The start of GEMM kernel implementation in @ref operator()
///                function call operator" which determines the work scope of each workgroup.
///            @li @b GemmPipeline - The core part @a "heart" of matrix multiplication algorithm.
///                This is the place where each workgroup is loading data from global memory and
///                carrying out dot products.
///            @li @b Epilogue - The @a "final" part of matrix multiplication implementation
///                 responsible for storing results to global memory. This is also the place where
///                 any additional operator fusion may take place.
///
///            Additionally both @ref GemmPipeline_ "GemmPipeline" and @ref EpiloguePipeline_
///            "EpiloguePipeline" are parameterized with so called @a Policy which determines all
///            internal details of those functional parts. You can think of it like both gemm and
///            epilogue pipelines provides the control-flow logic controlled by policies. Moreover
///            the policy is responsible for definition of all necessary data layouts and thread's
///            work distribution.
///
/// @tparam GroupedConvTraitsType_       The type of class providing traits for grouped convolution.
/// @tparam TilePartitioner_            The type of class providing mapping of workgroup index into
/// the
///                                     output data tile to be calculated. It determines the
///                                     workgroup to data relationship (or in other words - which
///                                     data would be processed and calculated by which workgroup).
/// @tparam GemmPipeline_               The type of class which provides the core part of matrix
///                                     multiplication. This class should provide implementation of
///                                     data loading from global memory and performing block-wise
///                                     matrix multiplication. You can think of it as a work done by
///                                     single workgroup point of view.
/// @tparam EpiloguePipeline_           The type of class providing the final part of matrix
///                                     multiplication implementation. It is responsible for storing
///                                     results calculated by @ref GemmPipeline_ "GemmPipeline" to
///                                     the output C tensor in global memory.
template <typename GroupedConvTraitsType_,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvolutionForwardKernel
{
    static constexpr bool EnableSplitImage = GroupedConvTraitsType_::EnableSplitImage;
    static constexpr index_t NDimSpatial   = GroupedConvTraitsType_::NDimSpatial;
    static constexpr ConvolutionSpecialization ConvSpecialization =
        GroupedConvTraitsType_::ConvSpecialization;
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using GemmALayout      = remove_cvref_t<typename GemmPipeline::ALayout>;
    using GemmBLayout      = remove_cvref_t<typename GemmPipeline::BLayout>;
    using GemmCLayout      = remove_cvref_t<typename GemmPipeline::CLayout>;

    using InLayout  = remove_cvref_t<typename GroupedConvTraitsType_::InLayout>;
    using WeiLayout = remove_cvref_t<typename GroupedConvTraitsType_::WeiLayout>;
    using OutLayout = remove_cvref_t<typename GroupedConvTraitsType_::OutLayout>;
    using DsLayout  = remove_cvref_t<typename GroupedConvTraitsType_::DsLayout>;

    using GemmDsLayout                  = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    static constexpr index_t kBlockSize = GemmPipeline::BlockSize;

    using InDataType  = remove_cvref_t<typename GemmPipeline::ADataType>;
    using WeiDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using DsDataType  = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using OutDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using CDElementwise = typename EpiloguePipeline::CDElementwise;

    using GroupedConvFwdKernelArgsSpecialized =
        GroupedConvFwdKernelArgs<GroupedConvTraitsType_, CDElementwise>;

    static constexpr bool IsSplitKSupported = false;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();

    static_assert(GemmPipeline::kPadM && GemmPipeline::kPadN && GemmPipeline::kPadK,
                  "Not supported!");
    static_assert(std::is_same_v<GemmALayout, tensor_layout::gemm::RowMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmBLayout, tensor_layout::gemm::ColumnMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>, "Not supported!");

    // Helper struct for spatial coordinates
    struct SpatialCoords
    {
        index_t d, h, w;
    };

    // Helper: Convert flat spatial index to (d,h,w) coordinates
    CK_TILE_DEVICE static SpatialCoords
    UnflattenSpatial(index_t flat, index_t h_size, index_t w_size)
    {
        if constexpr(NDimSpatial == 1)
        {
            return SpatialCoords{0, 0, flat};
        }
        else if constexpr(NDimSpatial == 2)
        {
            return SpatialCoords{0, flat / w_size, flat % w_size};
        }
        else // NDimSpatial == 3
        {
            const index_t hw        = h_size * w_size;
            const index_t d         = flat / hw;
            const index_t remainder = flat % hw;
            return SpatialCoords{d, remainder / w_size, remainder % w_size};
        }
    }

    // Helper: Convert (d,h,w) to flat spatial index
    CK_TILE_DEVICE static index_t
    FlattenSpatial(index_t d, index_t h, index_t w, index_t total_h, index_t total_w)
    {
        if constexpr(NDimSpatial == 1)
        {
            return w;
        }
        else if constexpr(NDimSpatial == 2)
        {
            return h * total_w + w;
        }
        else // NDimSpatial == 3
        {
            return (d * total_h + h) * total_w + w;
        }
    }

    // Helper: Find which piece owns a block using binary search
    template <typename SplitImageInfo>
    CK_TILE_DEVICE static index_t
    FindPieceId(index_t block_id, const SplitImageInfo& split_info, index_t num_pieces)
    {
        index_t left     = 0;
        index_t right    = num_pieces - 1;
        index_t piece_id = (left + right) / 2;

        while(!(block_id >= split_info.pieces[piece_id].block_start &&
                block_id < split_info.pieces[piece_id].block_end) &&
              left <= right)
        {
            if(block_id < split_info.pieces[piece_id].block_start)
            {
                right = piece_id - 1;
            }
            else
            {
                left = piece_id + 1;
            }
            piece_id = (left + right) / 2;
        }
        return piece_id;
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "grouped_convolution_forward", 
            gemm_prec_str<InDataType, WeiDataType>(), 
            "gemm",
            GemmPipeline::GetName(),
            "epilogue",
            EpiloguePipeline::GetName());
        // clang-format on
    }

#ifdef CK_EXPERIMENTAL_BUILDER
    CK_TILE_HOST std::string GetInstanceString() const
    {
        static_assert(ck_tile::reflect::HasInstanceTraits<GroupedConvolutionForwardKernel>,
                      "Specialization of instance_traits not found. Please check that a "
                      "specialization exists in file "
                      "ck_tile/builder/reflect/"
                      "instance_traits_tile_grouped_convolution_forward.hpp "
                      "for the given template parameters.");
        return ck_tile::reflect::instance_string<GroupedConvolutionForwardKernel>();
    }
#endif

    CK_TILE_HOST static auto GridSize(const GroupedConvFwdKernelArgsSpecialized& kargs)
    {
        return dim3(
            TilePartitioner::GridSize(kargs.GemmM, kargs.GemmN), kargs.GemmBatch, kargs.n_splits);
    }

    CK_TILE_HOST static auto BlockSize()
    {
        return is_wave32() ? dim3(kBlockSize / 2) : dim3(kBlockSize);
    }

    CK_TILE_HOST static constexpr GroupedConvFwdKernelArgsSpecialized
    MakeKernelArgs(const GroupedConvFwdHostArgs<CDElementwise>& hostArgs)
    {
        auto kargs = GroupedConvFwdKernelArgsSpecialized(hostArgs);
        return kargs;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST static bool IsSupportedArgument(const GroupedConvFwdKernelArgsSpecialized& kargs)
    {
        if constexpr((GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                      is_any_of<OutDataType, fp16_t, bf16_t>::value) ||
                     !IsSplitKSupported)
        {
            if(kargs.k_batch != 1)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
                }
                return false;
            }
        }

        const index_t ConvK = kargs.wei_g_k_c_xs_lengths[number<1>{}];
        const index_t ConvC = kargs.wei_g_k_c_xs_lengths[number<2>{}];

        // check ConvolutionSpecialization
        if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            // check if it's 1x1, stride=1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t SpatialDim = kargs.wei_g_k_c_xs_lengths[i + 3];
                const index_t ConvStride = kargs.conv_filter_strides[i];
                const index_t LeftPad    = kargs.input_left_pads[i];
                const index_t RightPad   = kargs.input_right_pads[i];

                if(!(SpatialDim == 1 && ConvStride == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter1x1Pad0)
        {
            // check if it's 1x1 conv
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t SpatialDim = kargs.wei_g_k_c_xs_lengths[i + 3];
                const index_t LeftPad    = kargs.input_left_pads[i];
                const index_t RightPad   = kargs.input_right_pads[i];

                if(!(SpatialDim == 1 && LeftPad == 0 && RightPad == 0))
                {
                    return false;
                }
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            if(ConvC != 1)
            {
                return false;
            }
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t filter_spatial_dim = kargs.wei_g_k_c_xs_lengths[i + I3];

                if(filter_spatial_dim != I3)
                {
                    return false;
                }
            }
        }

        namespace ctc = tensor_layout::convolution;

        if constexpr(std::is_same_v<InLayout, ctc::NWGC> || std::is_same_v<InLayout, ctc::NHWGC> ||
                     std::is_same_v<InLayout, ctc::NDHWGC>)
        {
            // Check access per C
            if(ConvC % GroupedConvTraitsType_::VectorSizeA != 0)
            {
                CK_TILE_ERROR("Conv C is not a multiple of vector load size for input image!");
                return false;
            }
        }
        else
        {
            CK_TILE_ERROR("Not supported input layout!");
            return false;
        }

        // check vector access of B
        // FIXME: layout
        if constexpr(std::is_same_v<WeiLayout, ctc::GKXC> ||
                     std::is_same_v<WeiLayout, ctc::GKYXC> ||
                     std::is_same_v<WeiLayout, ctc::GKZYXC>)
        {
            if(ConvC % GroupedConvTraitsType_::VectorSizeB != 0)
            {
                CK_TILE_ERROR("Conv C is not a multiple of vector load size for weight!");
                return false;
            }
        }
        else
        {
            CK_TILE_ERROR("Not supported weight layout!");
            return false;
        }

        // check vector access of E
        if constexpr(std::is_same_v<OutLayout, ctc::NWGK> ||
                     std::is_same_v<OutLayout, ctc::NHWGK> ||
                     std::is_same_v<OutLayout, ctc::NDHWGK>)
        {
            if(ConvK % GroupedConvTraitsType_::VectorSizeC != 0)
            {
                CK_TILE_ERROR("Conv K is not a multiple of vector store size for output image!");
                return false;
            }
        }
        else
        {
            CK_TILE_ERROR("Not supported output layout!");
            return false;
        }

        return true;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set,
              typename ADescType,
              typename BDescType,
              typename CDescType>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const InDataType* a_ptr,
                        const WeiDataType* b_ptr,
                        const std::array<const void*, NumDTensor>& ds_ptr,
                        OutDataType* c_ptr,
                        const ADescType& a_desc,
                        const BDescType& b_desc,
                        const CDescType& c_desc)
    {
        static_assert(!TilePartitioner::BlockGemmShape::PermuteA, "Not implemented!");
        static_assert(!TilePartitioner::BlockGemmShape::PermuteB, "Not implemented!");
        const auto& a_tensor_view = [&]() {
            return make_tensor_view<address_space_enum::global>(a_ptr, a_desc);
        }();

        const auto& b_tensor_view = [&]() {
            return make_tensor_view<address_space_enum::global>(b_ptr, b_desc);
        }();

        // TODO: enable vector write for C in ColMajor
        const auto& c_tensor_view = [&]() {
            return make_tensor_view<address_space_enum::global>(c_ptr, c_desc);
        }();

        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                static_assert(std::is_same_v<std::tuple_element_t<i, DsLayout>, OutLayout>,
                              "Not supported!");
                static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>,
                              "Not supported!");
                static_assert(std::is_same_v<std::tuple_element_t<i, DsDataType>, OutDataType>,
                              "Not supported!");

                return make_tensor_view<address_space_enum::global>(
                    static_cast<const OutDataType*>(ds_ptr[i]), c_desc);
            },
            number<NumDTensor>{});

        return make_tuple(a_tensor_view, b_tensor_view, ds_tensor_view, c_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            return pad_tensor_view(a_tensor_view,
                                   make_tuple(number<TilePartitioner::MPerBlock>{},
                                              number<TilePartitioner::KPerBlock>{}),
                                   sequence<true, true>{});
        }();

        const auto& b_pad_view = [&]() {
            const auto& b_tensor_view = views.at(I1);
            return pad_tensor_view(b_tensor_view,
                                   make_tuple(number<TilePartitioner::NPerBlock>{},
                                              number<TilePartitioner::KPerBlock>{}),
                                   sequence<true, true>{});
        }();

        const auto& ds_tensor_view = views.at(I2);
        const auto& ds_pad_view    = generate_tuple(
            [&](auto i) {
                return pad_tensor_view(ds_tensor_view[i],
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<true, true>{});
            },
            number<NumDTensor>{});

        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view = views.at(I3);
            return pad_tensor_view(c_tensor_view,
                                   make_tuple(number<TilePartitioner::MPerBlock>{},
                                              number<TilePartitioner::NPerBlock>{}),
                                   sequence<true, true>{});
        }();

        return make_tuple(a_pad_view, b_pad_view, ds_pad_view, c_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, const index_t i_m, const index_t i_n)
    {
        const auto& a_pad_view  = views.at(I0);
        const auto& b_pad_view  = views.at(I1);
        const auto& ds_pad_view = views.at(I2);
        const auto& c_pad_view  = views.at(I3);

        const auto& a_block_window = [&]() {
            return make_tile_window(a_pad_view,
                                    make_tuple(number<TilePartitioner::MPerBlock>{},
                                               number<TilePartitioner::KPerBlock>{}),
                                    {i_m, 0});
        }();

        const auto& b_block_window = [&]() {
            return make_tile_window(b_pad_view,
                                    make_tuple(number<TilePartitioner::NPerBlock>{},
                                               number<TilePartitioner::KPerBlock>{}),
                                    {i_n, 0});
        }();

        const auto ds_block_window = generate_tuple(
            [&](auto i) {
                return make_tile_window(ds_pad_view[i],
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {i_m, i_n});
            },
            number<NumDTensor>{});

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return make_tuple(a_block_window, b_block_window, ds_block_window, c_block_window);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param ds_ptr input D tensors pointer array
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param a_desc Input tensor A descriptor
     * @param b_desc Weight tensor B descriptor
     * @param c_desc Output tensor C descriptor
     * @param gemm_k The GEMM K dimension
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    template <typename ADescType, typename BDescType, typename CDescType>
    CK_TILE_DEVICE static void RunGemm(const InDataType* a_ptr,
                                       const WeiDataType* b_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       OutDataType* c_ptr,
                                       void* smem_ptr_0,
                                       const ADescType& a_desc,
                                       const BDescType& b_desc,
                                       const CDescType& c_desc,
                                       const index_t gemm_k,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_ptr, ds_ptr, c_ptr, a_desc, b_desc, c_desc);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = amd_wave_read_first_lane(TilePartitioner::GetLoopNum(gemm_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(I0);
        const auto& b_block_window = gemm_tile_windows.at(I1);
        const auto& d_block_window = gemm_tile_windows.at(I2);

        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I3);

        EpiloguePipeline{}.template operator()<decltype(c_block_window), decltype(c_block_tile)>(
            c_block_window, c_block_tile, d_block_window, smem_ptr_0);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @note RunGEMM2LDS in with two shared memory buffers using the ping pong buffer mechanism.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param ds_ptr input D tensors pointer array
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The starting pointer of 1st shared memory block.
     * @param smem_ptr_1 The starting pointer of 2nd shared memory block.
     * @param a_desc Input tensor A descriptor
     * @param b_desc Weight tensor B descriptor
     * @param c_desc Output tensor C descriptor
     * @param gemm_k The GEMM K dimension
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    template <typename ADescType, typename BDescType, typename CDescType>
    CK_TILE_DEVICE static void RunGemm2LDS(const InDataType* a_ptr,
                                           const WeiDataType* b_ptr,
                                           const std::array<const void*, NumDTensor>& ds_ptr,
                                           OutDataType* c_ptr,
                                           void* __restrict__ smem_ptr_0,
                                           void* __restrict__ smem_ptr_1,
                                           const ADescType& a_desc,
                                           const BDescType& b_desc,
                                           const CDescType& c_desc,
                                           const index_t gemm_k,
                                           const index_t block_idx_m,
                                           const index_t block_idx_n)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_ptr, ds_ptr, c_ptr, a_desc, b_desc, c_desc);
        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);
        auto gemm_tile_windows     = MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n);

        const index_t num_loop = amd_wave_read_first_lane(TilePartitioner::GetLoopNum(gemm_k));

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(I0);
        const auto& b_block_window = gemm_tile_windows.at(I1);
        const auto& d_block_window = gemm_tile_windows.at(I2);

        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0, smem_ptr_1);

        // Run Epilogue Pipeline
        auto& c_block_window = gemm_tile_windows.at(I3);

        EpiloguePipeline{}.template operator()<decltype(c_block_window), decltype(c_block_tile)>(
            c_block_window, c_block_tile, d_block_window, smem_ptr_0);
    }

    CK_TILE_DEVICE void operator()(GroupedConvFwdKernelArgsSpecialized kargs) const
    {
        const auto blockIdX = amd_wave_read_first_lane(blockIdx.x);
        const auto blockIdY = amd_wave_read_first_lane(blockIdx.y);

        const auto group_offset_a = amd_wave_read_first_lane(kargs.group_stride_a * blockIdY);
        const auto group_offset_b = amd_wave_read_first_lane(kargs.group_stride_b * blockIdY);
        const auto group_offset_c = amd_wave_read_first_lane(kargs.group_stride_c * blockIdY);

        // Split-N handling: Get which split this workgroup handles
        const auto blockIdZ = amd_wave_read_first_lane(blockIdx.z);

        // Calculate batch offset for this split
        const index_t batch_offset = amd_wave_read_first_lane(blockIdZ * kargs.n_per_split);

        // Calculate memory offsets for this split
        const long_index_t input_batch_offset = static_cast<long_index_t>(batch_offset) *
                                                static_cast<long_index_t>(kargs.input_batch_stride);
        const long_index_t output_batch_offset =
            static_cast<long_index_t>(batch_offset) *
            static_cast<long_index_t>(kargs.output_batch_stride);

        // Calculate base pointers with group and batch offsets
        const InDataType* base_a_ptr =
            static_cast<const InDataType*>(kargs.in_ptr) + group_offset_a + input_batch_offset;
        const WeiDataType* b_ptr = static_cast<const WeiDataType*>(kargs.wei_ptr) +
                                   group_offset_b; // No batch offset for weights!
        OutDataType* base_c_ptr =
            static_cast<OutDataType*>(kargs.out_ptr) + group_offset_c + output_batch_offset;

        // =====================================================================
        // Split-image: Map local block to global tile index (if enabled)
        // =====================================================================
        const InDataType* a_ptr;
        OutDataType* c_ptr;
        index_t i_m = 0;
        index_t i_n = 0;

        // Pre-calculate block_id (used in both split-image and non-split paths)
        const index_t block_id = static_cast<index_t>(blockIdX);

        if constexpr(EnableSplitImage)
        {
            // Add spatial offsets for split-image (constexpr optimization)
            a_ptr = base_a_ptr + kargs.spatial_offset_in;
            c_ptr = base_c_ptr + kargs.spatial_offset_out;

            // Find which piece owns this block using binary search
            // Reference: device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp
            const index_t piece_id =
                FindPieceId(block_id, kargs.split_image, kargs.num_spatial_pieces);
            const auto& piece      = kargs.split_image.pieces[piece_id];
            const auto& split_info = kargs.split_image;

            // Calculate local block ID and tile indices
            const index_t local_block_id = block_id - piece.block_start;
            const index_t local_gemm_m =
                kargs.n_per_split * piece.d_size * piece.h_size * piece.w_size;
            const auto [local_tile_m, local_tile_n] =
                TilePartitioner{local_gemm_m, kargs.GemmN}.GetOutputTileIndex(local_block_id);

            // Extract batch and spatial coordinates from local tile
            const index_t local_m_start      = local_tile_m * TilePartitioner::MPerBlock;
            const index_t spatial_per_batch  = piece.d_size * piece.h_size * piece.w_size;
            const index_t local_n            = local_m_start / spatial_per_batch;
            const index_t local_spatial_flat = local_m_start % spatial_per_batch;

            // Convert to local spatial coordinates
            const auto local_coords =
                UnflattenSpatial(local_spatial_flat, piece.h_size, piece.w_size);

            // Convert to global spatial coordinates
            const index_t global_n = local_n;
            const index_t global_d = piece.d_start + local_coords.d;
            const index_t global_h = piece.h_start + local_coords.h;
            const index_t global_w = piece.w_start + local_coords.w;

            // Convert to global M index
            const index_t global_spatial_per_batch = split_info.total_spatial; // Pre-calculated
            const index_t global_spatial_flat      = FlattenSpatial(
                global_d, global_h, global_w, split_info.total_h, split_info.total_w);
            const index_t global_m = global_n * global_spatial_per_batch + global_spatial_flat;

            // Set tile indices for GEMM operation
            i_m = amd_wave_read_first_lane(global_m);
            i_n = amd_wave_read_first_lane(local_tile_n * TilePartitioner::NPerBlock);
        }
        else
        {
            // No spatial offsets needed for regular path
            a_ptr = base_a_ptr;
            c_ptr = base_c_ptr;

            // No split-image: use standard tile partitioning
            const auto [iM, iN] =
                TilePartitioner{kargs.GemmM, kargs.GemmN}.GetOutputTileIndex(block_id);
            i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);
        }

        // Use global descriptors for all cases
        const auto& a_desc = kargs.a_grid_desc_m_k;
        const auto& b_desc = kargs.b_grid_desc_n_k;
        const auto& c_desc = kargs.c_grid_desc_m_n;

        // allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];

        if constexpr(GemmPipeline::DoubleSmemBuffer == true)
        {
            __shared__ char smem_ptr_1[GetSmemSize()];
            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                           is_any_of<OutDataType, fp16_t, bf16_t>::value))
            {
                RunGemm2LDS(a_ptr,
                            b_ptr,
                            kargs.ds_ptr,
                            c_ptr,
                            smem_ptr_0,
                            smem_ptr_1,
                            a_desc,
                            b_desc,
                            c_desc,
                            kargs.GemmK,
                            i_m,
                            i_n);
            }
        }
        else
        {
            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                           is_any_of<OutDataType, fp16_t, bf16_t>::value))
            {
                RunGemm(a_ptr,
                        b_ptr,
                        kargs.ds_ptr,
                        c_ptr,
                        smem_ptr_0,
                        a_desc,
                        b_desc,
                        c_desc,
                        kargs.GemmK,
                        i_m,
                        i_n);
            }
        }
    }
};

} // namespace ck_tile
