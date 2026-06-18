// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

#include "ck_tile/ops/grouped_convolution/utils/split_k_utils.hpp"

#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/streamk_gemm/streamk_gemm_coherency.hpp"
#include "ck_tile/ops/common/streamk_common.hpp"

#ifdef CK_EXPERIMENTAL_BUILDER
#include "ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_backward_weight.hpp"
#endif

namespace ck_tile {

template <typename T>
struct is_streamk_partitioner : std::false_type
{
};
template <typename Shape, StreamKReductionStrategy S, bool P>
struct is_streamk_partitioner<StreamKTilePartitioner<Shape, S, P>> : std::true_type
{
};

template <typename T>
struct is_compute_v6_pipeline : std::false_type
{
};
template <typename Problem, typename Policy>
struct is_compute_v6_pipeline<GemmPipelineAgBgCrCompV6<Problem, Policy>> : std::true_type
{
};

template <typename... Args>
CK_TILE_HOST void LogInfo(Args&&... args) noexcept
{
    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
    {
        CK_TILE_INFO(std::forward<Args>(args)...);
    }
}

/// @brief The Grouped Convolution kernel device arguments.
template <typename GroupedConvTraitsType_, typename TilePartitioner_ = void>
struct GroupedConvBwdWeightKernelArgs
{

    using ConvToGemmTransformer =
        TransformConvBwdWeightToGemm<GroupedConvTraitsType_::NDimSpatial,
                                     GroupedConvTraitsType_::ConvSpecialization,
                                     GroupedConvTraitsType_::VectorSizeA,
                                     GroupedConvTraitsType_::VectorSizeB,
                                     GroupedConvTraitsType_::VectorSizeC,
                                     GroupedConvTraitsType_::NumGroupsToMerge>;
    static constexpr index_t NumDTensor = GroupedConvTraitsType_::NumDTensor;

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
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
                GroupedConvTraitsType_::NDimSpatial>();

        a_grid_desc_k_m = grid_descs.at(number<0>{});
        b_grid_desc_k_n = grid_descs.at(number<1>{});
        c_grid_desc_m_n = grid_descs.at(number<2>{});

        NumGroupsPerBatch = GroupedConvTraitsType_::NumGroupsToMerge;
        group_stride_a    = args.K_ * NumGroupsPerBatch; // A: Out NWGK
        group_stride_b    = args.C_ * NumGroupsPerBatch; // B: In  NWGC
        group_stride_c    = args.K_ * args.C_            // C: Wei GKXC
                         * NumGroupsPerBatch *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());

        GemmM     = a_grid_desc_k_m.get_length(number<1>{});
        GemmN     = b_grid_desc_k_n.get_length(number<1>{});
        GemmK     = a_grid_desc_k_m.get_length(number<0>{});
        GemmBatch = integer_divide_ceil(args.G_, NumGroupsPerBatch);

        k_batch = args.k_batch;

        LogInfo("GemmM: ",
                GemmM,
                ", GemmN: ",
                GemmN,
                ", GemmK: ",
                GemmK,
                ", GemmBatch: ",
                GemmBatch,
                ", NumGroupsPerBatch: ",
                NumGroupsPerBatch,
                ", k_batch: ",
                k_batch);
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
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
                GroupedConvTraitsType_::NDimSpatial>();

        a_grid_desc_k_m = grid_descs.at(number<0>{});
        b_grid_desc_k_n = grid_descs.at(number<1>{});
        c_grid_desc_m_n = grid_descs.at(number<2>{});

        NumGroupsPerBatch = GroupedConvTraitsType_::NumGroupsToMerge;
        group_stride_a    = args.K_ * NumGroupsPerBatch; // A: Out NHWGK
        group_stride_b    = args.C_ * NumGroupsPerBatch; // B: In  NHWGC
        group_stride_c    = args.K_ * args.C_            // C: Wei GKYXC
                         * NumGroupsPerBatch *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());

        GemmM     = a_grid_desc_k_m.get_length(number<1>{});
        GemmN     = b_grid_desc_k_n.get_length(number<1>{});
        GemmK     = a_grid_desc_k_m.get_length(number<0>{});
        GemmBatch = integer_divide_ceil(args.G_, NumGroupsPerBatch);

        k_batch = args.k_batch;

        LogInfo("GemmM: ",
                GemmM,
                ", GemmN: ",
                GemmN,
                ", GemmK: ",
                GemmK,
                ", GemmBatch: ",
                GemmBatch,
                ", NumGroupsPerBatch: ",
                NumGroupsPerBatch,
                ", k_batch: ",
                k_batch);
    }

    template <
        typename InLay                      = typename GroupedConvTraitsType_::InLayout,
        typename WeiLay                     = typename GroupedConvTraitsType_::WeiLayout,
        typename OutLay                     = typename GroupedConvTraitsType_::OutLayout,
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
                GroupedConvTraitsType_::NDimSpatial>();

        a_grid_desc_k_m = grid_descs.at(number<0>{});
        b_grid_desc_k_n = grid_descs.at(number<1>{});
        c_grid_desc_m_n = grid_descs.at(number<2>{});

        NumGroupsPerBatch = GroupedConvTraitsType_::NumGroupsToMerge;
        group_stride_a    = args.K_ * NumGroupsPerBatch; // A: Out NDHWGK
        group_stride_b    = args.C_ * NumGroupsPerBatch; // B: In  NDHWGC
        group_stride_c    = args.K_ * args.C_            // C: Wei GKZYXC
                         * NumGroupsPerBatch *
                         std::accumulate(args.filter_spatial_lengths_.begin(),
                                         args.filter_spatial_lengths_.end(),
                                         1,
                                         std::multiplies<index_t>());

        GemmM     = a_grid_desc_k_m.get_length(number<1>{});
        GemmN     = b_grid_desc_k_n.get_length(number<1>{});
        GemmK     = a_grid_desc_k_m.get_length(number<0>{});
        GemmBatch = integer_divide_ceil(args.G_, NumGroupsPerBatch);

        k_batch = args.k_batch;

        LogInfo("GemmM: ",
                GemmM,
                ", GemmN: ",
                GemmN,
                ", GemmK: ",
                GemmK,
                ", GemmBatch: ",
                GemmBatch,
                ", NumGroupsPerBatch: ",
                NumGroupsPerBatch,
                ", k_batch: ",
                k_batch);
    }

    using ABCGridDescs = remove_cvref_t<
        decltype(ConvToGemmTransformer{}.MakeABCGridDescriptor_A_K0_M_K1_B_K0_N_K1_C_M_N())>;

    using AGridDescKM = remove_cvref_t<decltype(ABCGridDescs{}[number<0>{}])>;
    using BGridDescKN = remove_cvref_t<decltype(ABCGridDescs{}[number<1>{}])>;
    using CGridDescMN = remove_cvref_t<decltype(ABCGridDescs{}[number<2>{}])>;

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
    index_t NumGroupsPerBatch;

    const void* out_ptr;
    const void* in_ptr;
    std::array<const void*, NumDTensor> ds_ptr;
    void* wei_ptr;

    AGridDescKM a_grid_desc_k_m;
    BGridDescKN b_grid_desc_k_n;
    CGridDescMN c_grid_desc_m_n;

    long_index_t group_stride_a;
    long_index_t group_stride_b;
    long_index_t group_stride_c;

    void* workspace_ptr = nullptr;

    // StreamK tile partitioner - stored directly when TilePartitioner_ is a real type,
    // empty struct when void (Split-K path). Constructed with dummy values here;
    // properly initialized in MakeKernelArgs before device-side use.
    struct EmptyPartitioner
    {
    };
    using PartitionerType =
        std::conditional_t<std::is_void_v<TilePartitioner_>, EmptyPartitioner, TilePartitioner_>;
    PartitionerType tile_partitioner = []() {
        if constexpr(std::is_void_v<TilePartitioner_>)
            return EmptyPartitioner{};
        else
            return TilePartitioner_(1, 1, 1, 1);
    }();
};

/// @brief The Grouped Convolution Backward Weight kernel template.
///
/// @paragraph Overview Overview
///            This class provides the grouped convolution backward weight kernel template. By
///            semantic division of Implicit GEMM algorithm into following parts we achieve
///            flexible, versatile and robust kernel implementation.
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
/// @tparam GroupedConvTraitsType_      The type of class providing traits for grouped convolution.
/// @tparam TilePartitioner_            The type of class providing mapping of workgroup index into
///                                     the output data tile to be calculated. It determines the
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
struct GroupedConvolutionBackwardWeightKernel
{
    static constexpr index_t NDimSpatial = GroupedConvTraitsType_::NDimSpatial;
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

    // Wavelet pipelines launch extra load waves (LaunchBlockSize > BlockSize); others use
    // BlockSize. See GroupedConvLaunchBlockSize in grouped_convolution_utils.hpp.
    static constexpr index_t kBlockSize = GroupedConvLaunchBlockSize<GemmPipeline>;

    using OutDataType = remove_cvref_t<typename GemmPipeline::ADataType>;
    using InDataType  = remove_cvref_t<typename GemmPipeline::BDataType>;
    using DsDataType  = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    using WeiDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    static constexpr bool LargeTensors        = GemmPipeline::LargeTensors;
    static constexpr bool IsSplitKSupported   = true;
    static constexpr bool IsStreamK           = is_streamk_partitioner<TilePartitioner>::value;
    static constexpr bool IsComputeV6Pipeline = is_compute_v6_pipeline<GemmPipeline>::value;

    using GroupedConvBwdWeightKernelArgsSpecialized =
        std::conditional_t<IsStreamK,
                           GroupedConvBwdWeightKernelArgs<GroupedConvTraitsType_, TilePartitioner>,
                           GroupedConvBwdWeightKernelArgs<GroupedConvTraitsType_>>;

    using AccDataType = remove_cvref_t<typename EpiloguePipeline::AccDataType>;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();

    static_assert(GemmPipeline::kPadM && GemmPipeline::kPadN && GemmPipeline::kPadK,
                  "Not supported!");
    static_assert(std::is_same_v<GemmALayout, tensor_layout::gemm::ColumnMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmBLayout, tensor_layout::gemm::RowMajor>, "Not supported!");
    static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>, "Not supported!");
    static_assert(GroupedConvTraitsType_::ExplicitGemm == false ||
                      GroupedConvTraitsType_::NumGroupsToMerge == 1,
                  "Not supported!");
    static_assert(!IsStreamK || NumDTensor == 0,
                  "D tensor per-group offsets not implemented for StreamK path");

    // StreamK reduction helpers (partial store/load, flag signaling, tile accumulation).
    // Shared with the StreamK GEMM kernel via StreamKReductionOps in streamk_common.hpp.
    using StreamKOps = StreamKReductionOps<TilePartitioner,
                                           GemmPipeline,
                                           GroupedConvBwdWeightKernelArgsSpecialized>;

    CK_TILE_HOST static index_t
    GetWorkSpaceSize(const GroupedConvBwdWeightKernelArgsSpecialized& kargs)
    {
        if constexpr(IsStreamK)
            return kargs.tile_partitioner.get_workspace_size(sizeof(AccDataType)) * kargs.GemmBatch;
        else
            return 0;
    }

    // Post-construction setter: workspace is allocated by the caller after
    // GetWorkSpaceSize() and must outlive the kernel launch. Can't be moved into
    // the constructor because kargs is a POD value type copied to GPU constant memory.
    CK_TILE_HOST static void SetWorkSpacePointer(GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                                                 void* workspace_ptr)
    {
        kargs.workspace_ptr = workspace_ptr;
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        static constexpr bool EnableSplitImage = GroupedConvTraitsType_::EnableSplitImage;
        constexpr auto NumGroupsToMerge        = GroupedConvTraitsType_::NumGroupsToMerge;
        // clang-format off
        return concat('_', "grouped_convolution_backward_weight", 
            gemm_prec_str<InDataType, WeiDataType>(), 
            InLayout::name,
            WeiLayout::name,
            OutLayout::name,
            "gemm",
            GemmPipeline::GetName(),
            "epilogue",
            EpiloguePipeline::GetName(),
            getConvSpecializationString(ConvSpecialization),
            "MergedGroups",
            NumGroupsToMerge,
            "SplitImage",
            EnableSplitImage,
            "ExplicitGemm",
            GroupedConvTraitsType_::ExplicitGemm,
            IsStreamK ? "StreamK" : "SplitK"
        );
        // clang-format on
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetTypeString() { return GetName(); }

#ifdef CK_EXPERIMENTAL_BUILDER
    CK_TILE_HOST std::string GetInstanceString() const
    {
        static_assert(ck_tile::reflect::HasInstanceTraits<GroupedConvolutionBackwardWeightKernel>,
                      "Specialization of instance_traits not found. Please check that a "
                      "specialization exists in file "
                      "ck_tile/builder/reflect/"
                      "instance_traits_tile_grouped_convolution_backward_weight.hpp "
                      "for the given template parameters.");
        return ck_tile::reflect::instance_string<GroupedConvolutionBackwardWeightKernel>();
    }
#endif

    CK_TILE_HOST static auto GridSize(const GroupedConvBwdWeightKernelArgsSpecialized& kargs)
    {
        if constexpr(IsStreamK)
        {
            auto sk_grid = kargs.tile_partitioner.grid_size();
            return dim3(sk_grid.x, 1, 1);
        }
        else
            return dim3(TilePartitioner::GridSize(kargs.GemmM, kargs.GemmN),
                        kargs.GemmBatch,
                        kargs.k_batch);
    }

    CK_TILE_HOST static constexpr auto BlockSize()
    {
        return is_wave32() ? dim3(kBlockSize / 2) : dim3(kBlockSize);
    }

    CK_TILE_HOST static GroupedConvBwdWeightKernelArgsSpecialized
    MakeKernelArgs(const GroupedConvBwdWeightHostArgs& hostArgs,
                   [[maybe_unused]] int num_cu    = 0,
                   [[maybe_unused]] int occupancy = 0)
    {
        LogInfo("MPerBlock: ",
                number<TilePartitioner::MPerBlock>{},
                ", NPerBlock: ",
                number<TilePartitioner::NPerBlock>{},
                ", KPerBlock: ",
                number<TilePartitioner::KPerBlock>{});

        auto kernel_args = GroupedConvBwdWeightKernelArgsSpecialized(hostArgs);

        if constexpr(IsStreamK)
        {
            // StreamK: construct tile partitioner and embed it in the args.
            // Use provided num_cu/occupancy, or query HW.
            if(num_cu == 0)
            {
                hipDeviceProp_t dev_prop;
                hipDevice_t dev;
                ck_tile::hip_check_error(hipGetDevice(&dev));
                ck_tile::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
                num_cu = dev_prop.multiProcessorCount;
            }
            if(occupancy == 0)
            {
                constexpr index_t minimum_occupancy =
                    GemmPipeline::Scheduler == ck_tile::GemmPipelineScheduler::Intrawave ? 1 : 2;
                constexpr int dynamic_smem_size = 0;
                int max_occupancy               = 0;
                hip_check_error(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                    &max_occupancy,
                    kentry<minimum_occupancy,
                           GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType_,
                                                                  TilePartitioner_,
                                                                  GemmPipeline_,
                                                                  EpiloguePipeline_>,
                           GroupedConvBwdWeightKernelArgsSpecialized>,
                    BlockSize().x,
                    dynamic_smem_size));
                occupancy = ck_tile::max(minimum_occupancy, max_occupancy);
            }

            const index_t grid = num_cu * occupancy;
            kernel_args.tile_partitioner =
                TilePartitioner(kernel_args.GemmM * kernel_args.GemmBatch,
                                kernel_args.GemmN,
                                kernel_args.GemmK,
                                grid);
        }
        else
        {
            using KernelImpl = GroupedConvolutionBackwardWeightKernel<GroupedConvTraitsType_,
                                                                      TilePartitioner_,
                                                                      GemmPipeline_,
                                                                      EpiloguePipeline_>;

            // Negative k_batch value: split-K autodeduction.
            if(kernel_args.k_batch < 0)
            {
                const auto optimal_split_k =
                    calculate_optimal_k_batch<GemmPipeline_::BlockSize,
                                              KernelImpl,
                                              TilePartitioner_>(kernel_args);
                kernel_args.k_batch = optimal_split_k;
            }
        }

        return kernel_args;
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    CK_TILE_HOST static bool
    IsSupportedArgument(const GroupedConvBwdWeightKernelArgsSpecialized& kargs)
    {
        if constexpr(GemmPipeline_::Async)
        {
            if(get_device_name() != "gfx950")
            {
                return false;
            }
        }
        // Runtime arch check - complements the static_assert in operator().
        // Both are needed: this check runs on the host (where get_compiler_target()
        // isn't available since HIP's host pass doesn't define __gfx*__ macros),
        // while the static_assert in operator() catches misuse at device compile time.
        if constexpr(IsStreamK)
        {
            const auto name = get_device_name();
            if(name != "gfx90a" && name != "gfx942" && name != "gfx950")
            {
                LogInfo("StreamK requires cross-CU buffer coherence. "
                        "Supported: gfx90a, gfx942, gfx950. Got: ",
                        name);
                return false;
            }
            if(kargs.k_batch != 1)
            {
                LogInfo("StreamK handles work distribution internally; k_batch must be 1.");
                return false;
            }
        }
        if(kargs.k_batch < 1)
        {
            LogInfo("k_batch must be at least one. Ensure argument is created via MakeKernelArgs.");
            return false;
        }

        // V6 pipeline requires num_loop >= PrefetchStages + 1 = 4
        // Otherwise it produces incorrect results (num_loop=1) or is just inefficient (num_loop=2
        // or 3).
        if constexpr(IsComputeV6Pipeline)
        {
            const index_t num_loop =
                integer_divide_ceil(kargs.GemmK, kargs.k_batch * TilePartitioner::KPerBlock);
            constexpr int num_loop_threashold = GemmPipeline_::PrefetchStages + 1;
            if(num_loop < num_loop_threashold)
            {
                LogInfo("For V6 pipeline, GemmK / (k_batch * KPerBlock) must be >= ",
                        num_loop_threashold,
                        ". Now GemmK is ",
                        kargs.GemmK,
                        ", k_batch is ",
                        kargs.k_batch,
                        ", KPerBlock is ",
                        number<TilePartitioner::KPerBlock>{},
                        ", num_loop is ",
                        num_loop);
                return false;
            }
        }

        if constexpr(!std::is_same_v<typename EpiloguePipeline::ODataType, float> &&
                     !std::is_same_v<typename EpiloguePipeline::ODataType, double>)
        {
            // The epilogue performs atomic add related to split-K using the ODataType.
            // If the type is less accurate than float, large split-K values may lead to
            // accuracy issues. Hence, we limit the maximum split-K value to 128 in such cases.
            if(kargs.k_batch > 128)
            {
                LogInfo("For epilogue output data type that is not float/double, we must have "
                        "k_batch <= 128.");
                return false;
            }
        }

        if constexpr((GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                      is_any_of<WeiDataType, fp16_t, bf16_t>::value))
        {
            if(kargs.k_batch != 1)
            {
                LogInfo("Conditions not met for K_batch > 1: VectorSizeC must be a multiple of 2 "
                        "for fp16/bf16 when K_batch > 1.",
                        "Now k_batch is ",
                        kargs.k_batch,
                        ", VectorSizeC is ",
                        GroupedConvTraitsType_::VectorSizeC);
                return false;
            }
        }

        if constexpr(!IsStreamK)
        {
            if(integer_divide_ceil(kargs.GemmK,
                                   TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{})) <
               kargs.k_batch)
            {
                LogInfo("KBatch is too large, part of GPU wouldn't be utilized! GemmK: ",
                        kargs.GemmK,
                        ", BlockGemmShape K: ",
                        TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{}),
                        ", k_batch: ",
                        kargs.k_batch);
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
                    LogInfo("For Filter1x1Stride1Pad0 specialization, all spatial dimensions must "
                            "be 1, stride must be 1, and padding must be 0. Now for dimension ",
                            i,
                            ": SpatialDim is ",
                            SpatialDim,
                            ", ConvStride is ",
                            ConvStride,
                            ", LeftPad is ",
                            LeftPad,
                            ", RightPad is ",
                            RightPad);
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
                    LogInfo("For Filter1x1Pad0 specialization, all spatial dimensions must be 1 "
                            "and padding must be 0. Now for dimension ",
                            i,
                            ": SpatialDim is ",
                            SpatialDim,
                            ", LeftPad is ",
                            LeftPad,
                            ", RightPad is ",
                            RightPad);
                    return false;
                }
            }
        }
        else if constexpr(ConvSpecialization == ConvolutionSpecialization::Filter3x3)
        {
            if(ConvC != 1)
            {
                LogInfo("For Filter3x3 specialization, ConvC must be 1. Now ConvC is ", ConvC);
                return false;
            }
            for(index_t i = 0; i < NDimSpatial; ++i)
            {
                const index_t filter_spatial_dim = kargs.wei_g_k_c_xs_lengths[i + I3];

                if(filter_spatial_dim != I3)
                {
                    LogInfo("For Filter3x3 specialization, all spatial dimensions of the filter "
                            "must be 3. Now for dimension ",
                            i,
                            ", filter_spatial_dim is ",
                            filter_spatial_dim);
                    return false;
                }
            }
        }

        if constexpr(GroupedConvTraitsType_::ExplicitGemm &&
                     ConvSpecialization != ConvolutionSpecialization::Filter1x1Stride1Pad0)
        {
            LogInfo("ExplicitGemm is only supported for Filter1x1Stride1Pad0 specialization.");
            return false;
        }

        namespace ctc = tensor_layout::convolution;

        if constexpr(std::is_same_v<InLayout, ctc::NWGC> || std::is_same_v<InLayout, ctc::NHWGC> ||
                     std::is_same_v<InLayout, ctc::NDHWGC>)
        {
            // Check access per C
            if(ConvC % GroupedConvTraitsType_::VectorSizeB != 0)
            {
                LogInfo("Conv C is not a multiple of vector load size for input! ConvC: ",
                        ConvC,
                        ", VectorSizeB: ",
                        GroupedConvTraitsType_::VectorSizeB);
                return false;
            }
        }
        else
        {
            LogInfo("Not supported input layout! Now InLayout is ", InLayout::name);
            return false;
        }

        if constexpr(std::is_same_v<WeiLayout, ctc::GKXC> ||
                     std::is_same_v<WeiLayout, ctc::GKYXC> ||
                     std::is_same_v<WeiLayout, ctc::GKZYXC>)
        {
            if(ConvC % GroupedConvTraitsType_::VectorSizeC != 0)
            {
                LogInfo("Conv C is not a multiple of vector load size for weight! ConvC: ",
                        ConvC,
                        ", VectorSizeC: ",
                        GroupedConvTraitsType_::VectorSizeC);
                return false;
            }
        }
        else
        {
            LogInfo("Not supported weight layout! Now WeiLayout is ", WeiLayout::name);
            return false;
        }

        if constexpr(std::is_same_v<OutLayout, ctc::NWGK> ||
                     std::is_same_v<OutLayout, ctc::NHWGK> ||
                     std::is_same_v<OutLayout, ctc::NDHWGK>)
        {
            if(ConvK % GroupedConvTraitsType_::VectorSizeA != 0)
            {
                LogInfo("Conv K is not a multiple of vector load size for output! ConvK: ",
                        ConvK,
                        ", VectorSizeA: ",
                        GroupedConvTraitsType_::VectorSizeA);
                return false;
            }
        }
        else
        {
            LogInfo("Not supported output layout! Now OutLayout is ", OutLayout::name);
            return false;
        }

        if constexpr(GroupedConvTraitsType_::NumGroupsToMerge > 1)
        {
            const index_t ConvG = kargs.wei_g_k_c_xs_lengths[number<0>{}];
            if(ConvG % GroupedConvTraitsType_::NumGroupsToMerge != 0)
            {
                LogInfo("Number of groups must be divisible by NumGroupsToMerge! ConvG: ",
                        ConvG,
                        ", NumGroupsToMerge: ",
                        GroupedConvTraitsType_::NumGroupsToMerge);
                return false;
            }
        }

        return true;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto
    MakeCBlockWindow(WeiDataType* c_ptr,
                     const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                     const index_t block_idx_m,
                     const index_t block_idx_n)
    {
        const auto& c_tensor_view = make_tensor_view<address_space_enum::global,
                                                     DstInMemOp,
                                                     amd_buffer_coherence_enum::coherence_default,
                                                     LargeTensors>(c_ptr, kargs.c_grid_desc_m_n);

        // For bf16_t and atomic_add global_atomic_add is used instead of buffer_atomic_add.
        // Add padding for not contiguous dim due to the lack of OOB check.
        // On gfx950, the bf16 atomic_add-specific padding is not needed, but LargeTensors
        // still require padding.
#if defined(__gfx950__)
        constexpr bool pad_not_contiguous_dim = LargeTensors;
#else
        constexpr bool pad_not_contiguous_dim =
            LargeTensors || (std::is_same_v<WeiDataType, bf16_t> &&
                             DstInMemOp == memory_operation_enum::atomic_add);
#endif
        const auto& c_pad_view = pad_tensor_view(
            c_tensor_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            sequence<pad_not_contiguous_dim, true>{});

        return make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {block_idx_m, block_idx_n});
    }

    CK_TILE_DEVICE static auto
    MakeDBlockWindows(const std::array<const void*, NumDTensor>& ds_ptr,
                      const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                      const index_t block_idx_m,
                      const index_t block_idx_n)
    {
        constexpr bool pad_not_contiguous_dim = LargeTensors;

        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                static_assert(std::is_same_v<std::tuple_element_t<i, DsLayout>, OutLayout>,
                              "Not supported!");
                static_assert(std::is_same_v<GemmCLayout, tensor_layout::gemm::RowMajor>,
                              "Not supported!");
                static_assert(std::is_same_v<std::tuple_element_t<i, DsDataType>, WeiDataType>,
                              "Not supported!");

                return make_tensor_view<address_space_enum::global,
                                        memory_operation_enum::set,
                                        amd_buffer_coherence_enum::coherence_default,
                                        LargeTensors>(static_cast<WeiDataType*>(ds_ptr[i]),
                                                      kargs.c_grid_desc_m_n);
            },
            number<NumDTensor>{});

        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                return pad_tensor_view(ds_tensor_view[i],
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<pad_not_contiguous_dim, true>{});
            },
            number<NumDTensor>{});

        return generate_tuple(
            [&](auto i) {
                return make_tile_window(ds_pad_view[i],
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::NPerBlock>{}),
                                        {block_idx_m, block_idx_n});
            },
            number<NumDTensor>{});
    }

    CK_TILE_DEVICE static auto
    MakeBBlockWindow(const InDataType* b_ptr,
                     const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                     const index_t block_idx_n,
                     const index_t block_idx_k)
    {
        constexpr bool pad_not_contiguous_dim = LargeTensors;

        static_assert(!GemmPipeline::BlockGemmShape::PermuteB, "Not implemented!");
        const auto& b_tensor_view = make_tensor_view<address_space_enum::global,
                                                     memory_operation_enum::set,
                                                     amd_buffer_coherence_enum::coherence_default,
                                                     LargeTensors>(b_ptr, kargs.b_grid_desc_k_n);

        const auto& b_pad_view =
            pad_tensor_view(b_tensor_view,
                            make_tuple(number<TilePartitioner::KPerBlock>{} * kargs.k_batch,
                                       number<TilePartitioner::NPerBlock>{}),
                            sequence<pad_not_contiguous_dim, true>{});

        return make_tile_window(
            b_pad_view,
            make_tuple(number<TilePartitioner::KPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {block_idx_k, block_idx_n});
    }

    CK_TILE_DEVICE static auto
    MakeABlockWindow(const OutDataType* a_ptr,
                     const GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                     const index_t block_idx_m,
                     const index_t block_idx_k)
    {
        constexpr bool pad_not_contiguous_dim = LargeTensors;

        static_assert(!GemmPipeline::BlockGemmShape::PermuteA, "Not implemented!");
        const auto& a_tensor_view = make_tensor_view<address_space_enum::global,
                                                     memory_operation_enum::set,
                                                     amd_buffer_coherence_enum::coherence_default,
                                                     LargeTensors>(a_ptr, kargs.a_grid_desc_k_m);

        const auto& a_pad_view =
            pad_tensor_view(a_tensor_view,
                            make_tuple(number<TilePartitioner::KPerBlock>{} * kargs.k_batch,
                                       number<TilePartitioner::MPerBlock>{}),
                            sequence<pad_not_contiguous_dim, true>{});

        return make_tile_window(
            a_pad_view,
            make_tuple(number<TilePartitioner::KPerBlock>{}, number<TilePartitioner::MPerBlock>{}),
            {block_idx_k, block_idx_m});
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param a_ptr input A pointer
     * @param b_ptr input B pointer
     * @param c_ptr output C pointer
     * @param smem_ptr_0 The start memory pointer of the shared memory block.
     * @param kargs Grouped Convolution Backward Weight kernel arguments
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
        // Create block windows using helper methods
        const auto& a_block_window = MakeABlockWindow(a_ptr, kargs, block_idx_m, block_idx_k);
        const auto& b_block_window = MakeBBlockWindow(b_ptr, kargs, block_idx_n, block_idx_k);
        const auto& d_block_window = MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);

        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile = GemmPipeline{}.template operator()(
            a_block_window, b_block_window, num_loop, smem_ptr_0);

        // Run the epilogue with split-K dispatch, wrapped for wavelet load/math waves.
        RunWaveletAwareEpilogue<GemmPipeline, EpiloguePipeline>([&]() {
            if(kargs.k_batch == 1)
            {
                auto c_block_window = MakeCBlockWindow<memory_operation_enum::set>(
                    c_ptr, kargs, block_idx_m, block_idx_n);

                EpiloguePipeline{}(c_block_window, c_block_tile, d_block_window, smem_ptr_0);
            }
            else
            {
                if constexpr(!(GroupedConvTraitsType_::VectorSizeC % 2 != 0 &&
                               is_any_of<WeiDataType, fp16_t, bf16_t>::value))
                {
                    auto c_block_window = MakeCBlockWindow<memory_operation_enum::atomic_add>(
                        c_ptr, kargs, block_idx_m, block_idx_n);

                    EpiloguePipeline{}(c_block_window, c_block_tile, d_block_window, smem_ptr_0);
                }
            }
        });
    }

    CK_TILE_DEVICE void CallExplicitGemm(GroupedConvBwdWeightKernelArgsSpecialized& kargs) const
    {
        static_assert(NumDTensor == 0, "Not supported!");
        using ExplicitBatchedGemmKernel =
            BatchedGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
        const auto batched_gemm_kargs = typename ExplicitBatchedGemmKernel::BatchedGemmKernelArgs{
            {{kargs.out_ptr},
             {kargs.in_ptr},
             {},
             kargs.wei_ptr,
             kargs.GemmM,
             kargs.GemmN,
             kargs.GemmK,
             {kargs.GemmM * kargs.GemmBatch},
             {kargs.GemmN * kargs.GemmBatch},
             {},
             kargs.GemmN,
             kargs.k_batch},
            kargs.GemmM,
            kargs.GemmN,
            kargs.GemmM * kargs.GemmN,
            kargs.GemmBatch};
        ExplicitBatchedGemmKernel{}(batched_gemm_kargs);
    }

    CK_TILE_DEVICE void RunStreamK(GroupedConvBwdWeightKernelArgsSpecialized& kargs) const
    {
        // Device-side compile-time arch check - complements the runtime check in
        // IsSupportedArgument(). Both are needed: the runtime check runs on the host
        // (where get_compiler_target() isn't available since HIP's host pass doesn't
        // define __gfx*__ macros), while this catches misuse at device compile time.
        static_assert(
            StreamKCoherency<decltype(core::arch::get_compiler_target())>::BUFFER_COHERENCE !=
                amd_buffer_coherence_enum::coherence_default,
            "StreamK requires cross-CU buffer coherence (StreamKCoherency specialization). "
            "Currently supported: gfx90a, gfx942, gfx950.");

        __shared__ char smem_ptr[GetSmemSize()];

        // Group offset (blockIdx.y = group batch index)
        const auto blockIdX       = amd_wave_read_first_lane(blockIdx.x);
        const index_t dp_num_loop = kargs.tile_partitioner.get_iters_per_tile();

        StreamKDispatch(
            kargs.tile_partitioner,
            [&](index_t tile_idx) {
                // Data-parallel workgroup: process one full tile
                const auto tile_mn = kargs.tile_partitioner.get_output_tile_index(tile_idx);
                const index_t i_m  = amd_wave_read_first_lane((tile_mn[I0] / kargs.GemmBatch) *
                                                             TilePartitioner::MPerBlock);
                const index_t i_n =
                    amd_wave_read_first_lane(tile_mn[I1] * TilePartitioner::NPerBlock);
                const index_t i_g = amd_wave_read_first_lane(tile_mn[I0] % kargs.GemmBatch);

                // Group offset derived from tile index (gridDim.z = 1 for StreamK)
                const auto group_offset_a = amd_wave_read_first_lane(kargs.group_stride_a * i_g);
                const auto group_offset_b = amd_wave_read_first_lane(kargs.group_stride_b * i_g);
                const auto group_offset_c = amd_wave_read_first_lane(kargs.group_stride_c * i_g);

                const OutDataType* a_ptr =
                    static_cast<const OutDataType*>(kargs.out_ptr) + group_offset_a;
                const InDataType* b_ptr =
                    static_cast<const InDataType*>(kargs.in_ptr) + group_offset_b;
                WeiDataType* c_ptr = static_cast<WeiDataType*>(kargs.wei_ptr) + group_offset_c;

                RunGemm(a_ptr,
                        b_ptr,
                        kargs.ds_ptr,
                        c_ptr,
                        smem_ptr,
                        kargs,
                        dp_num_loop,
                        i_m,
                        i_n,
                        /*block_idx_k=*/0);
            },
            [&](index_t sk_cta_idx) {
                RunStreamKLoop(kargs,
                               sk_cta_idx,
                               static_cast<const OutDataType*>(kargs.out_ptr),
                               static_cast<const InDataType*>(kargs.in_ptr),
                               static_cast<WeiDataType*>(kargs.wei_ptr),
                               smem_ptr);
            },
            blockIdX);
    }

    /// @brief Stream-K loop: iterate over assigned K-iterations, run GEMM pipeline,
    ///        and perform Linear or Tree reduction to accumulate partial results.
    CK_TILE_DEVICE void RunStreamKLoop(GroupedConvBwdWeightKernelArgsSpecialized& kargs,
                                       index_t sk_cta_idx,
                                       const OutDataType* a_ptr_base,
                                       const InDataType* b_ptr_base,
                                       WeiDataType* c_ptr_base,
                                       char* smem_ptr) const
    {
        const StreamKOps sk_ops{};

        index_t iter_start, iter_end;
        kargs.tile_partitioner.get_iter_boundaries(iter_start, iter_end, sk_cta_idx);

        while(iter_start < iter_end)
        {
            index_t tile_idx =
                amd_wave_read_first_lane(kargs.tile_partitioner.get_tile_index(iter_start));

            index_t tile_iter_start, tile_iter_end;
            kargs.tile_partitioner.get_tile_boundaries(tile_iter_start, tile_iter_end, tile_idx);

            index_t local_iter_start = amd_wave_read_first_lane(
                kargs.tile_partitioner.get_local_iter(iter_start, tile_iter_start));
            index_t local_iter_end =
                amd_wave_read_first_lane(kargs.tile_partitioner.get_local_iter_end(
                    tile_iter_start, iter_end, tile_iter_end));

            index_t num_loop_sk = local_iter_end - local_iter_start;

            // Compute M/N tile indices from 1D tile index
            const auto c_macro_tile_idx = kargs.tile_partitioner.get_output_tile_index(tile_idx);
            const index_t i_m = amd_wave_read_first_lane((c_macro_tile_idx[I0] / kargs.GemmBatch) *
                                                         TilePartitioner::MPerBlock);
            const index_t i_n =
                amd_wave_read_first_lane(c_macro_tile_idx[I1] * TilePartitioner::NPerBlock);
            const index_t i_g = amd_wave_read_first_lane(c_macro_tile_idx[I0] % kargs.GemmBatch);

            // K offset = local_iter_start * KPerBlock
            const index_t i_k =
                amd_wave_read_first_lane(local_iter_start * TilePartitioner::KPerBlock);

            // Group offset (blockIdx.y = group batch index)
            const auto group_offset_a = amd_wave_read_first_lane(kargs.group_stride_a * i_g);
            const auto group_offset_b = amd_wave_read_first_lane(kargs.group_stride_b * i_g);
            const auto group_offset_c = amd_wave_read_first_lane(kargs.group_stride_c * i_g);

            const OutDataType* a_ptr = static_cast<const OutDataType*>(a_ptr_base) + group_offset_a;
            const InDataType* b_ptr  = static_cast<const InDataType*>(b_ptr_base) + group_offset_b;
            WeiDataType* c_ptr       = static_cast<WeiDataType*>(c_ptr_base) + group_offset_c;

            // Create block windows and run pipeline
            const auto& a_block_window = MakeABlockWindow(a_ptr, kargs, i_m, i_k);
            const auto& b_block_window = MakeBBlockWindow(b_ptr, kargs, i_n, i_k);
            const auto& d_block_window = MakeDBlockWindows(kargs.ds_ptr, kargs, i_m, i_n);

            const auto& c_block_tile = GemmPipeline{}.template operator()(
                a_block_window, b_block_window, num_loop_sk, smem_ptr);

            auto tile_started = iter_start == tile_iter_start;
            auto tile_ended   = iter_end >= tile_iter_end;

            if constexpr(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Linear)
            {
                // Linear Reduction: tile-starter sequentially accumulates all
                // partials from subsequent CTAs in order.
                if(!tile_started)
                {
                    sk_ops.StorePartial(kargs, sk_cta_idx, c_block_tile);
                    sk_ops.SignalStorePartialDone(kargs, sk_cta_idx);
                }
                else
                {
                    auto accum_block_tile = c_block_tile;
                    if(!tile_ended)
                    {
                        const index_t iter_per_tile = kargs.tile_partitioner.get_iters_per_tile();
                        const index_t iter_per_cta  = kargs.tile_partitioner.get_iters_per_sk_cta();
                        const index_t extra_iters   = kargs.tile_partitioner.get_extra_iters();
                        int accum_iters             = local_iter_end - local_iter_start;
                        int next_cta                = sk_cta_idx + 1;

                        while(accum_iters < iter_per_tile)
                        {
                            sk_ops.WaitStorePartialDone(kargs, next_cta);

                            using BlockType = remove_cvref_t<decltype(c_block_tile)>;
                            sk_ops.AddBlockTile(
                                accum_block_tile,
                                sk_ops.template LoadPartial<typename BlockType::DataType>(
                                    kargs, next_cta, c_block_tile.get_tile_distribution()));

                            accum_iters += iter_per_cta + (next_cta < extra_iters);
                            ++next_cta;
                        }
                    }

                    auto c_block_window_out =
                        MakeCBlockWindow<memory_operation_enum::set>(c_ptr, kargs, i_m, i_n);
                    EpiloguePipeline{}(
                        c_block_window_out, accum_block_tile, d_block_window, smem_ptr);
                }
            }
            else if constexpr(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Tree)
            {
                // Tree Reduction: pairwise reduction with stride doubling.
                // At each round, half the CTAs store their accumulated partial
                // and exit; the other half load and accumulate from their partner.
                // The tile-starter writes the final result.
                auto accum_block_tile      = c_block_tile;
                index_t tile_local_cta_idx = amd_wave_read_first_lane(
                    kargs.tile_partitioner.get_tile_local_cta_index(tile_iter_start, sk_cta_idx));

                index_t stride = amd_wave_read_first_lane(1);

                for(;; stride <<= 1)
                {
                    // Partner index is a *global* SK CTA index. This works because
                    // CTAs contributing to the same tile always have contiguous global
                    // SK CTA indices (guaranteed by the partitioner's iteration assignment).
                    const index_t partner_cta_idx = amd_wave_read_first_lane(sk_cta_idx + stride);
                    const index_t partner_start_iter = amd_wave_read_first_lane(
                        kargs.tile_partitioner.get_start_iter(partner_cta_idx));
                    bool partner_in_tile =
                        amd_wave_read_first_lane(partner_start_iter < tile_iter_end);

                    // If the partner of the tile-starter is not in this tile,
                    // then all partials are accumulated - write final result.
                    if(tile_started && !partner_in_tile)
                    {
                        auto c_block_window_out =
                            MakeCBlockWindow<memory_operation_enum::set>(c_ptr, kargs, i_m, i_n);
                        EpiloguePipeline{}(
                            c_block_window_out, accum_block_tile, d_block_window, smem_ptr);
                        break;
                    }

                    // This CTA's turn to read from its partner and accumulate.
                    if(tile_local_cta_idx % (stride << 1) == 0)
                    {
                        if(partner_in_tile)
                        {
                            sk_ops.WaitStorePartialDone(kargs, partner_cta_idx);
                            using BlockType = remove_cvref_t<decltype(c_block_tile)>;
                            sk_ops.AddBlockTile(
                                accum_block_tile,
                                sk_ops.template LoadPartial<typename BlockType::DataType>(
                                    kargs, partner_cta_idx, c_block_tile.get_tile_distribution()));
                        }
                    }
                    // This CTA's turn to write its partial and exit.
                    else
                    {
                        sk_ops.StorePartial(kargs, sk_cta_idx, accum_block_tile);
                        sk_ops.SignalStorePartialDone(kargs, sk_cta_idx);
                        break;
                    }
                }
            }
            else
            {
                static_assert(
                    TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Linear ||
                        TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Tree,
                    "Unsupported StreamK reduction strategy for conv bwd weight.");
            }

            // Advance to next tile
            iter_start = tile_iter_end;
            block_sync_lds();
        }
    }

    CK_TILE_DEVICE void operator()(GroupedConvBwdWeightKernelArgsSpecialized& kargs) const
    {
        if constexpr(IsStreamK)
        {
#if !defined(__gfx1250__)
            if constexpr(GemmPipeline_::Async)
            {
#if defined(__gfx950__)
                RunStreamK(kargs);
#endif
            }
            else
            {
                RunStreamK(kargs);
            }
#endif
        }
        else if constexpr(GroupedConvTraitsType_::ExplicitGemm)
        {
            if constexpr(GemmPipeline_::Async)
            {
#if defined(__gfx950__)
                CallExplicitGemm(kargs);
#endif
            }
            else
            {
                CallExplicitGemm(kargs);
            }
        }
        else
        {
            const auto blockIdX = amd_wave_read_first_lane(blockIdx.x);
            const auto [iM, iN] =
                TilePartitioner{kargs.GemmM, kargs.GemmN}.GetOutputTileIndex(blockIdX);
            const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
            const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            const auto blockIdZ    = amd_wave_read_first_lane(blockIdx.z);
            const index_t num_loop = amd_wave_read_first_lane(ck_tile::integer_divide_ceil(
                kargs.GemmK, kargs.k_batch * TilePartitioner::KPerBlock));
            const index_t i_k =
                amd_wave_read_first_lane(blockIdZ * num_loop * TilePartitioner::KPerBlock);

            const auto blockIdY       = amd_wave_read_first_lane(blockIdx.y);
            const auto group_offset_a = amd_wave_read_first_lane(kargs.group_stride_a * blockIdY);
            const auto group_offset_b = amd_wave_read_first_lane(kargs.group_stride_b * blockIdY);
            const auto group_offset_c = amd_wave_read_first_lane(kargs.group_stride_c * blockIdY);

            // options
            // conv_bwd_weight = Out * In = Weight
            const OutDataType* a_ptr =
                static_cast<const OutDataType*>(kargs.out_ptr) + group_offset_a;
            const InDataType* b_ptr = static_cast<const InDataType*>(kargs.in_ptr) + group_offset_b;
            WeiDataType* c_ptr      = static_cast<WeiDataType*>(kargs.wei_ptr) + group_offset_c;

            __shared__ char smem_ptr[GetSmemSize()];

            // Disable Async for other archs than gfx950
            if constexpr(GemmPipeline_::Async)
            {
#if defined(__gfx950__)
                RunGemm(
                    a_ptr, b_ptr, kargs.ds_ptr, c_ptr, smem_ptr, kargs, num_loop, i_m, i_n, i_k);
#endif
            }
            else
            {
                RunGemm(
                    a_ptr, b_ptr, kargs.ds_ptr, c_ptr, smem_ptr, kargs, num_loop, i_m, i_n, i_k);
            }
        }
    }
};

} // namespace ck_tile
