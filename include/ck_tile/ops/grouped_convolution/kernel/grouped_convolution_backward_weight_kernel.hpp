// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/host/convolution_parameter.hpp"
#include "ck_tile/ops/grouped_convolution/utils/transform_conv_bwd_weight_to_gemm.hpp"
#include "ck_tile/ops/grouped_convolution/utils/grouped_convolution_utils.hpp"

namespace ck_tile {

/// @brief The Grouped Convolution kernel device arguments.
template <typename GroupedConvTraitsType>
struct GroupedConvBwdWeightKernelArgs
{

    using ConvToGemmTransformer =
        TransformConvBwdWeightToGemm<GroupedConvTraitsType::NDimSpatial,
                                     GroupedConvTraitsType::ConvSpecialization>;
    static constexpr index_t NumDTensor = GroupedConvTraitsType::NumDTensor;

    template <
        typename InLay                      = typename GroupedConvTraitsType::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvBwdWeightKernelArgs(const GroupedConvBwdWeightHostArgs& args)
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

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        ConvToGemmTransformer conv_to_gemm_transformer{in_g_n_c_wis_lengths,
                                                       wei_g_k_c_xs_lengths,
                                                       out_g_n_k_wos_lengths,
                                                       conv_filter_strides,
                                                       conv_filter_dilations,
                                                       input_left_pads,
                                                       input_right_pads};

        // tuple
        auto grid_descs =
            conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                GroupedConvTraitsType::NDimSpatial>();

        a_grid_desc_m_k = grid_descs.at(number<0>{});
        b_grid_desc_n_k = grid_descs.at(number<1>{});
        c_grid_desc_m_n = grid_descs.at(number<2>{});

        group_stride_a = args.K_;            // A: Out NWGK
        group_stride_b = args.C_;            // B: In  NWGC
        group_stride_c = args.K_ * args.C_ * // C: Wei GKXC
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());

        GemmM     = a_grid_desc_m_k.get_length(number<0>{});
        GemmN     = b_grid_desc_n_k.get_length(number<0>{});
        GemmK     = a_grid_desc_m_k.get_length(number<1>{});
        GemmBatch = args.G_;
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvBwdWeightKernelArgs(const GroupedConvBwdWeightHostArgs& args)
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

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        ConvToGemmTransformer conv_to_gemm_transformer{in_g_n_c_wis_lengths,
                                                       wei_g_k_c_xs_lengths,
                                                       out_g_n_k_wos_lengths,
                                                       conv_filter_strides,
                                                       conv_filter_dilations,
                                                       input_left_pads,
                                                       input_right_pads};

        // tuple
        auto grid_descs =
            conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                GroupedConvTraitsType::NDimSpatial>();

        a_grid_desc_m_k = grid_descs.at(number<0>{});
        b_grid_desc_n_k = grid_descs.at(number<1>{});
        c_grid_desc_m_n = grid_descs.at(number<2>{});

        group_stride_a = args.K_;            // A: Out NHWGK
        group_stride_b = args.C_;            // B: In  NHWGC
        group_stride_c = args.K_ * args.C_ * // C: Wei GKYXC
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());

        GemmM     = a_grid_desc_m_k.get_length(number<0>{});
        GemmN     = b_grid_desc_n_k.get_length(number<0>{});
        GemmK     = a_grid_desc_m_k.get_length(number<1>{});
        GemmBatch = args.G_;
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType::OutLayout,
        typename std::enable_if<std::is_same_v<InLay, tensor_layout::convolution::NDHWGC> &&
                                    std::is_same_v<WeiLay, tensor_layout::convolution::GKZYXC> &&
                                    std::is_same_v<OutLay, tensor_layout::convolution::NDHWGK>,
                                bool>::type = false>
    CK_TILE_HOST GroupedConvBwdWeightKernelArgs(const GroupedConvBwdWeightHostArgs& args)
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

        in_ptr  = args.in_ptr;
        wei_ptr = args.wei_ptr;
        for(index_t d = 0; d < NumDTensor; d++)
        {
            ds_ptr[d] = args.ds_ptr[d];
        }
        out_ptr = args.out_ptr;

        ConvToGemmTransformer conv_to_gemm_transformer{in_g_n_c_wis_lengths,
                                                       wei_g_k_c_xs_lengths,
                                                       out_g_n_k_wos_lengths,
                                                       conv_filter_strides,
                                                       conv_filter_dilations,
                                                       input_left_pads,
                                                       input_right_pads};

        // tuple
        auto grid_descs =
            conv_to_gemm_transformer.template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N<
                GroupedConvTraitsType::NDimSpatial>();

        a_grid_desc_m_k = grid_descs.at(number<0>{});
        b_grid_desc_n_k = grid_descs.at(number<1>{});
        c_grid_desc_m_n = grid_descs.at(number<2>{});

        group_stride_a = args.K_;            // A: Out NDHWGK
        group_stride_b = args.C_;            // B: In  NDHWGC
        group_stride_c = args.K_ * args.C_ * // C: wEI GKZYXC
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());

        GemmM     = a_grid_desc_m_k.get_length(number<0>{});
        GemmN     = b_grid_desc_n_k.get_length(number<0>{});
        GemmK     = a_grid_desc_m_k.get_length(number<1>{});
        GemmBatch = args.G_;
    }

    using ABCGridDescs =
        remove_cvref_t<decltype(ConvToGemmTransformer{}
                                    .template MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N())>;

    using AGridDescMK = remove_cvref_t<decltype(ABCGridDescs{}[number<0>{}])>;
    using BGridDescNK = remove_cvref_t<decltype(ABCGridDescs{}[number<1>{}])>;
    using CGridDescMN = remove_cvref_t<decltype(ABCGridDescs{}[number<2>{}])>;

    static constexpr index_t NonSpatialDims = 3;
    array<index_t, NonSpatialDims + GroupedConvTraitsType::NDimSpatial> in_g_n_c_wis_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType::NDimSpatial> wei_g_k_c_xs_lengths;
    array<index_t, NonSpatialDims + GroupedConvTraitsType::NDimSpatial> out_g_n_k_wos_lengths;

    array<index_t, GroupedConvTraitsType::NDimSpatial> conv_filter_strides;
    array<index_t, GroupedConvTraitsType::NDimSpatial> conv_filter_dilations;
    array<index_t, GroupedConvTraitsType::NDimSpatial> input_left_pads;
    array<index_t, GroupedConvTraitsType::NDimSpatial> input_right_pads;

    index_t k_batch;
    index_t GemmM;
    index_t GemmN;
    index_t GemmK;
    index_t GemmBatch;

    const void* out_ptr;
    const void* in_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* wei_ptr;

    AGridDescMK a_grid_desc_m_k;
    BGridDescNK b_grid_desc_n_k;
    CGridDescMN c_grid_desc_m_n;

    long_index_t group_stride_a;
    long_index_t group_stride_b;
    long_index_t group_stride_c;
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
/// tparam ConvSpecialization  Tensor descriptors specialization.
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
template <typename GroupedConvTraitsType,
          typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_>
struct GroupedConvolutionBackwardWeightKernel
{
    static constexpr index_t NDimSpatial = GroupedConvTraitsType::NDimSpatial_;
    static constexpr ConvolutionSpecialization ConvSpecialization =
        GroupedConvTraitsType::ConvSpecialization;
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using GemmALayout      = remove_cvref_t<typename GemmPipeline::ALayout>;
    using GemmBLayout      = remove_cvref_t<typename GemmPipeline::BLayout>;
    using GemmCLayout      = remove_cvref_t<typename GemmPipeline::CLayout>;

    using InLayout  = remove_cvref_t<typename GroupedConvTraitsType::InLayout>;
    using WeiLayout = remove_cvref_t<typename GroupedConvTraitsType::WeiLayout>;
    using OutLayout = remove_cvref_t<typename GroupedConvTraitsType::OutLayout>;
    using DsLayout  = remove_cvref_t<typename GroupedConvTraitsType::DsLayout>;

    using GemmDsLayout                  = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    static constexpr index_t NumDTensor = GroupedConvTraitsType::NumDTensor;

    static constexpr index_t KernelBlockSize = GemmPipeline::BlockSize;

    using InDataType  = remove_cvref_t<typename GemmPipeline::ADataType>;
    using WeiDataType = remove_cvref_t<typename GemmPipeline::BDataType>;
    using DsDataType  = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using OutDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using GroupedConvBwdWeightKernelArgsSpecialized =
        GroupedConvBwdWeightKernelArgs<GroupedConvTraitsType>;

    // TODO: Enable this
    static constexpr bool IsSplitKSupported = true;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();

    static_assert(GemmPipeline::kPadM && GemmPipeline::kPadN && GemmPipeline::kPadK,
                  "Not supported!");
    static_assert(std::is_same_v<GemmALayout, tensor_layout::gemm::RowMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmBLayout, tensor_layout::gemm::ColumnMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>, "Not supported!");

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "grouped_convolution_backward_weight", gemm_prec_str<InDataType, WeiDataType>, GemmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto
    GridSize(const GroupedConvBwdWeightKernelArgsSpecialized& kargs)
    {
        return dim3(
            TilePartitioner::GridSize(kargs.GemmM, kargs.GemmN), kargs.GemmBatch, kargs.k_batch);
    }

    CK_TILE_HOST static constexpr auto BlockSize() { return dim3(KernelBlockSize); }

    CK_TILE_HOST static constexpr GroupedConvBwdWeightKernelArgsSpecialized
    MakeKernelArgs(const GroupedConvBwdWeightHostArgs& hostArgs)
    {
        return GroupedConvBwdWeightKernelArgsSpecialized(hostArgs);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        __device__ SplitKBatchOffset(const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                                     const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1 = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t = __builtin_amdgcn_readfirstlane(kargs.k_batch * K1);
            const index_t KRead =
                __builtin_amdgcn_readfirstlane((kargs.GemmK + K_t - 1) / K_t * K1);

            a_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);
            b_k_split_offset = __builtin_amdgcn_readfirstlane(k_id * KRead);

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = __builtin_amdgcn_readfirstlane(KRead);
            }
            else
            {
                splitted_k =
                    __builtin_amdgcn_readfirstlane(kargs.GemmK - KRead * (kargs.k_batch - 1));
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    CK_TILE_HOST static auto Preprocess(const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                                        const stream_config& s)
    {
        return [&]() {
            if(kargs.k_batch > 1)
                hipGetErrorString(hipMemsetAsync(kargs.wei_ptr,
                                                 0,
                                                 kargs.GemmBatch * kargs.GemmM * kargs.GemmN *
                                                     sizeof(WeiDataType),
                                                 s.stream_id_));
        };
    }

    CK_TILE_HOST static bool
    IsSupportedArgument(const GroupedConvBwdWeightKernelArgsSpecialized& kargs)
    {
        if constexpr((EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
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

        // check ConvSpecialization
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
            if(ConvC % GemmPipeline::GetVectorSizeB() != 0)
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
            if(ConvC % EpiloguePipeline::GetVectorSizeC() != 0)
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
            if(ConvK % GemmPipeline::GetVectorSizeA() != 0)
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

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto
    MakeGemmTensorViews(const OutDataType* a_ptr,
                        const InDataType* b_ptr,
                        const std::array<const void*, NumDTensor>& ds_ptr,
                        WeiDataType* c_ptr,
                        const GroupedConvBwdWeightKernelArgsSpecialized& kargs)
    {
        static_assert(!TilePartitioner::BlockGemmShape::PermuteA, "Not implemented!");
        static_assert(!TilePartitioner::BlockGemmShape::PermuteB, "Not implemented!");
        const auto& a_tensor_view = [&]() {
            return make_tensor_view<address_space_enum::global>(a_ptr,
                                                                kargs.a_grid_desc_m_k); // A: out
        }();

        const auto& b_tensor_view = [&]() {
            return make_tensor_view<address_space_enum::global>(b_ptr,
                                                                kargs.b_grid_desc_n_k); // B: in
        }();

        const auto& c_tensor_view = [&]() {
            return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                c_ptr,
                make_tuple(kargs.GemmM, kargs.GemmN),
                make_tuple(kargs.GemmN, 1),
                number<EpiloguePipeline::GetVectorSizeC()>{},
                number<1>{});
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
                    static_cast<OutDataType*>(ds_ptr[i]), kargs.c_grid_desc_m_n);
            },
            number<NumDTensor>{});

        return make_tuple(a_tensor_view, b_tensor_view, ds_tensor_view, c_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views, const index_t k_batch)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            return pad_tensor_view(a_tensor_view,
                                   make_tuple(number<TilePartitioner::MPerBlock>{},
                                              number<TilePartitioner::KPerBlock>{} * k_batch),
                                   sequence<true, true>{});
        }();

        const auto& b_pad_view = [&]() {
            const auto& b_tensor_view = views.at(I1);
            return pad_tensor_view(b_tensor_view,
                                   make_tuple(number<TilePartitioner::NPerBlock>{},
                                              number<TilePartitioner::KPerBlock>{} * k_batch),
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
    CK_TILE_DEVICE static auto MakeGemmTileWindows(const PadView& views,
                                                   const index_t i_m,
                                                   const index_t i_n,
                                                   const index_t i_k)
    {
        const auto& a_pad_view  = views.at(I0);
        const auto& b_pad_view  = views.at(I1);
        const auto& ds_pad_view = views.at(I2);
        const auto& c_pad_view  = views.at(I3);

        const auto& a_block_window = [&]() {
            return make_tile_window(a_pad_view,
                                    make_tuple(number<TilePartitioner::MPerBlock>{},
                                               number<TilePartitioner::KPerBlock>{}),
                                    {i_m, i_k});
        }();

        const auto& b_block_window = [&]() {
            return make_tile_window(b_pad_view,
                                    make_tuple(number<TilePartitioner::NPerBlock>{},
                                               number<TilePartitioner::KPerBlock>{}),
                                    {i_n, i_k});
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
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs Grouped Convolution Forward kernel arguments
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm(const OutDataType* a_ptr,
                                       const InDataType* b_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       WeiDataType* c_ptr,
                                       void* smem_ptr_0,
                                       const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                                       const index_t num_loop,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n,
                                       const index_t block_idx_k)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_ptr, ds_ptr, c_ptr, kargs);

        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple, kargs.k_batch);
        auto gemm_tile_windows =
            MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n, block_idx_k);

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
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The starting pointer of 1st shared memory block.
     * @param smem_ptr_1 The starting pointer of 2nd shared memory block.
     * @param kargs Grouped Convolution Forward kernel arguments
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm2LDS(const OutDataType* a_ptr,
                                           const InDataType* b_ptr,
                                           const std::array<const void*, NumDTensor>& ds_ptr,
                                           WeiDataType* c_ptr,
                                           void* __restrict__ smem_ptr_0,
                                           void* __restrict__ smem_ptr_1,
                                           const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                                           const index_t num_loop,
                                           const index_t block_idx_m,
                                           const index_t block_idx_n,
                                           const index_t block_idx_k)
    {
        // Create Gemm tensor views, pad views and tile windows
        const auto& gemm_tensor_views_tuple =
            MakeGemmTensorViews<EpiloguePipeline::MemoryOperation>(
                a_ptr, b_ptr, ds_ptr, c_ptr, kargs);
        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple, kargs.k_batch);
        auto gemm_tile_windows =
            MakeGemmTileWindows(gemm_pad_views, block_idx_m, block_idx_n, block_idx_k);

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

    CK_TILE_DEVICE void operator()(GroupedConvBwdWeightKernelArgsSpecialized kargs) const
    {
        const auto blockIdX = __builtin_amdgcn_readfirstlane(blockIdx.x);
        const auto [iM, iN] =
            TilePartitioner{kargs.GemmM, kargs.GemmN}.GetOutputTileIndex(blockIdX);
        const index_t i_m = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        const auto blockIdZ    = __builtin_amdgcn_readfirstlane(blockIdx.z);
        const index_t num_loop = __builtin_amdgcn_readfirstlane(
            ck_tile::integer_divide_ceil(kargs.GemmK, kargs.k_batch * TilePartitioner::KPerBlock));
        const index_t i_k =
            __builtin_amdgcn_readfirstlane(blockIdZ * num_loop * TilePartitioner::KPerBlock);

        const auto blockIdY       = __builtin_amdgcn_readfirstlane(blockIdx.y);
        const auto group_offset_a = __builtin_amdgcn_readfirstlane(kargs.group_stride_a * blockIdY);
        const auto group_offset_b = __builtin_amdgcn_readfirstlane(kargs.group_stride_b * blockIdY);
        const auto group_offset_c = __builtin_amdgcn_readfirstlane(kargs.group_stride_c * blockIdY);

        // options
        // conv_bwd_weight = Out * In = Weight
        const OutDataType* a_ptr = static_cast<const OutDataType*>(kargs.out_ptr) + group_offset_a;
        const InDataType* b_ptr  = static_cast<const InDataType*>(kargs.in_ptr) + group_offset_b;
        WeiDataType* c_ptr       = static_cast<WeiDataType*>(kargs.wei_ptr) + group_offset_c;

        // allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];

        if constexpr(GemmPipeline::DoubleSmemBuffer == true)
        {
            __shared__ char smem_ptr_1[GetSmemSize()];
            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<OutDataType, fp16_t, bf16_t>::value))
            {
                RunGemm2LDS(a_ptr,
                            b_ptr,
                            kargs.ds_ptr,
                            c_ptr,
                            smem_ptr_0,
                            smem_ptr_1,
                            kargs,
                            num_loop,
                            i_m,
                            i_n,
                            i_k);
            }
        }
        else
        {
            if constexpr(!(EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add &&
                           EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                           is_any_of<OutDataType, fp16_t, bf16_t>::value))
            {
                RunGemm(
                    a_ptr, b_ptr, kargs.ds_ptr, c_ptr, smem_ptr_0, kargs, num_loop, i_m, i_n, i_k);
            }
        }
    }
};

} // namespace ck_tile
