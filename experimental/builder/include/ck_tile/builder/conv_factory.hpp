// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A factory for instantiating CK convolution kernels.
//
// This file translates a semantic description of a convolution operation
// (`ConvSignatureDescriptor` and `ConvAlgorithmDescriptor`) into specific,
// low-level template arguments required by the underlying CK device-level
// kernel implementations. This abstraction enables more complex build
// time logic and simplifies the kernel specification.
//
// Key Components:
//
// Template Metaprogram:
//  - ConvFactory: The main factory, with specializations for different
//                 convolution directions (currently only forward).
//
// Template Metaprogram Helpers:
//  - ConvTensorLayouts: Maps layout enums to CK layout types for different
//                       spatial dimensions (2D/3D) and directions.
//  - ConvTensorTypes:   Maps data type enums (FP16, BF16, FP32) to C++ types used by CK.
//  - ConvPassThroughOps: Hard-coded pass-through element-wise operations.
//  - ConvSpec:          Encapsulates convolution and GEMM specialization enums.
//
// `constexpr` Helper Functions:
//  - SetThreadBlockInfo:           Determines thread block dimensions and tile sizes.
//  - SetConvTuningInfo:            Sets XDL and AK1/BK1 tuning parameters.
//  - SetFwdConvABlockTransfer:     Configures A tensor block transfer parameters.
//  - SetFwdConvBBlockTransfer:     Configures B tensor block transfer parameters.
//  - SetCBlockTransfer:            Configures C tensor block transfer parameters.
//  - SetBlockGemmPipelineVersion:  Maps pipeline version enum to CK types.
//
// The primary entry point is the `ConvFactory` struct, which is currently
// specialized for forward convolutions and produces instances of
// DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3.

#pragma once

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
// WORKAROUND: Macro namespace collision in upstream CK device operation headers.
// device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp (line 41) and
// device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp (line 51) both define
// GridwiseGemmTemplateParameters macro without #undef, causing redefinition errors.
// Use pragma push/pop to isolate the Large_Tensor header's macro scope.
#pragma push_macro("GridwiseGemmTemplateParameters")
#ifdef GridwiseGemmTemplateParameters
#undef GridwiseGemmTemplateParameters
#endif
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_large_tensor_cshuffle.hpp"
#pragma pop_macro("GridwiseGemmTemplateParameters")
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_concepts.hpp"
#include "ck_tile/builder/conv_algorithm_limits.hpp"
#include "ck_tile/builder/builder_utils.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/builder/versions.hpp"

namespace ck_tile::builder::factory_internal {

// Type mappings from the builder FwdGroupConvLayout enum classes to the CK tensor data types.
template <auto LayoutValue, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM> && ValidConvLayoutForSpatialDim<LayoutValue, SPATIAL_DIM>)
struct ConvTensorLayouts
{
    // This will trigger if a specialization for the given layout is not found.
    // We should always catch this in an earlier validation check.
    using Layout = decltype(LayoutValue);
    static_assert(sizeof(Layout) == 0,
                  "Internal error. Unsupported layout for convolution factory.");
};

// 1D Forward Convolution Layout Specializations
template <>
struct ConvTensorLayouts<GroupConvLayout1D::NWGC_GKXC_NWGK, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NWGC;
    using BLayout  = ck::tensor_layout::convolution::GKXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NWGK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout1D::NGCW_GKXC_NGKW, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCW;
    using BLayout  = ck::tensor_layout::convolution::GKXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout1D::GNWC_GKXC_GNWK, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::GNWC;
    using BLayout  = ck::tensor_layout::convolution::GKXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::GNWK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout1D::NGCW_GKCX_NGKW, 1, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCW;
    using BLayout  = ck::tensor_layout::convolution::GKCX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout2D::NGCHW_GKYXC_NGKHW, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCHW;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout2D::NHWGC_GKYXC_NHWGK, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NHWGK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout2D::GNHWC_GKYXC_GNHWK, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::GNHWC;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::GNHWK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout2D::NGCHW_GKCYX_NGKHW, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCHW;
    using BLayout  = ck::tensor_layout::convolution::GKCYX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout3D::NGCDHW_GKCZYX_NGKDHW, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NGCDHW;
    using BLayout  = ck::tensor_layout::convolution::GKCZYX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKDHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NDHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKZYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NDHWGK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout3D::GNDHWC_GKZYXC_GNDHWK, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::GNDHWC;
    using BLayout  = ck::tensor_layout::convolution::GKZYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::GNDHWK;
};

template <GroupConvLayout Layout, size_t SPATIAL_DIM, ConvDirection DIR>
consteval auto GetTensorLayout()
{

    if constexpr(SPATIAL_DIM == 1)
    {
        return factory_internal::ConvTensorLayouts<Layout._1d, 1, DIR>{};
    }
    else if constexpr(SPATIAL_DIM == 2)
    {
        return factory_internal::ConvTensorLayouts<Layout._2d, 2, DIR>{};
    }
    else if constexpr(SPATIAL_DIM == 3)
    {
        return factory_internal::ConvTensorLayouts<Layout._3d, 3, DIR>{};
    }
    else
    {
        static_assert(false, "Unsupported spatial dimension for convolution layout.");
    }
}

// Type mappings from builder convolution data type to CK tensor types.
template <DataType T>
struct ConvTensorTypes
{
    // This will trigger if a specialization for the given DataType is not found.
    // We should always catch this in an earlier validation check.
    static_assert(sizeof(UnsupportedEnumValue<T>) == 0,
                  "Internal error. Unsupported data type for convolution factory.");
};

template <>
struct ConvTensorTypes<DataType::FP16>
{
    using ADataType        = ck::half_t;
    using AComputeType     = ck::half_t;
    using BDataType        = ck::half_t;
    using BComputeType     = ck::half_t;
    using CShuffleDataType = ck::half_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = ck::half_t;
};

template <>
struct ConvTensorTypes<DataType::BF16>
{
    using ADataType        = ck::bhalf_t;
    using AComputeType     = ck::bhalf_t;
    using BDataType        = ck::bhalf_t;
    using BComputeType     = ck::bhalf_t;
    using CShuffleDataType = ck::bhalf_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = ck::bhalf_t;
};

template <>
struct ConvTensorTypes<DataType::FP32>
{
    using ADataType        = float;
    using AComputeType     = float;
    using BDataType        = float;
    using BComputeType     = float;
    using CShuffleDataType = float;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = float;
};

template <>
struct ConvTensorTypes<DataType::I8>
{
    using ADataType        = int8_t;
    using AComputeType     = int8_t;
    using BDataType        = int8_t;
    using BComputeType     = int8_t;
    using CShuffleDataType = int8_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = int32_t;
    using EDataType        = int8_t;
};

template <ElementwiseOperation T>
struct ElementwiseOps
{
    // This will trigger if a specialization for the given DataType is not found.
    // We should always catch this in an earlier validation check.
    static_assert(sizeof(UnsupportedEnumValue<T>) == 0,
                  "Internal error. Unsupported elementwise operation for convolution factory.");
};

template <>
struct ElementwiseOps<ElementwiseOperation::PASS_THROUGH>
{
    using AElementwiseOp   = ck::tensor_operation::element_wise::PassThrough;
    using BElementwiseOp   = ck::tensor_operation::element_wise::PassThrough;
    using CDEElementwiseOp = ck::tensor_operation::element_wise::PassThrough;
};

template <>
struct ElementwiseOps<ElementwiseOperation::SCALE>
{
    using AElementwiseOp   = ck::tensor_operation::element_wise::PassThrough;
    using BElementwiseOp   = ck::tensor_operation::element_wise::PassThrough;
    using CDEElementwiseOp = ck::tensor_operation::element_wise::Scale;
};

// The algorithm specializations for the convolution and GEMM.
template <typename CONV_ENUM>
    requires(
        std::is_same_v<CONV_ENUM, ck::tensor_operation::device::ConvolutionForwardSpecialization>)
struct ConvSpec
{
    CONV_ENUM conv_spec;
    ck::tensor_operation::device::GemmSpecialization gemm_spec;
};

// Deduction guide for ConvSpec to simplify brace initialization.
template <typename CONV_ENUM, typename GEMM_ENUM>
ConvSpec(CONV_ENUM, GEMM_ENUM) -> ConvSpec<CONV_ENUM>;

struct BlockGemmSpec
{
    ck::BlockGemmPipelineVersion pipeline_version;
    ck::BlockGemmPipelineScheduler scheduler;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr BlockGemmSpec SetBlockGemm()
{
    constexpr auto& BG = ALGORITHM.block_gemm;

    ck::BlockGemmPipelineScheduler scheduler;
    ck::BlockGemmPipelineVersion version;

    if constexpr(BG.scheduler == PipelineScheduler::INTRAWAVE)
    {
        scheduler = ck::BlockGemmPipelineScheduler::Intrawave;
    }
    else if constexpr(BG.scheduler == PipelineScheduler::INTERWAVE)
    {
        scheduler = ck::BlockGemmPipelineScheduler::Interwave;
    }
    else
    {
        static_assert(false, "Unknown PipelineScheduler");
    }

    if constexpr(BG.pipeline_version == PipelineVersion::V1)
    {
        version = ck::BlockGemmPipelineVersion::v1;
    }
    else if constexpr(BG.pipeline_version == PipelineVersion::V2)
    {
        version = ck::BlockGemmPipelineVersion::v2;
    }
    else if constexpr(BG.pipeline_version == PipelineVersion::V3)
    {
        version = ck::BlockGemmPipelineVersion::v3;
    }
    else if constexpr(BG.pipeline_version == PipelineVersion::V4)
    {
        version = ck::BlockGemmPipelineVersion::v4;
    }
    else if constexpr(BG.pipeline_version == PipelineVersion::V5)
    {
        version = ck::BlockGemmPipelineVersion::v5;
    }
    else
    {
        static_assert(false, "Unknown PipelineVersion");
    }

    return BlockGemmSpec{.pipeline_version = version, .scheduler = scheduler};
}

// Block info for a convolution.
struct MNK
{
    size_t m{};
    size_t n{};
    size_t k{};
};
struct ConvBlock
{
    size_t block_size = 0;
    MNK per_block     = {};
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr ConvBlock SetThreadBlockInfo()
{
    constexpr auto& TB = ALGORITHM.thread_block;
    return ConvBlock{.block_size = TB.block_size,
                     .per_block  = {.m = TB.tile_size.m, .n = TB.tile_size.n, .k = TB.tile_size.k}};
}

// Block transfer parameters for A or B tensor.
struct BlockTransfer
{
    ck::Array<size_t, 3> thread_cluster_dims  = {0, 0, 0}; // k0, m, k1
    ck::Array<size_t, 3> thread_cluster_order = {0, 0, 0};
    ck::Array<size_t, 3> src_access_order     = {0, 0, 0};
    size_t src_vector_dim                     = 0;
    size_t src_scalar_per_vector              = 0;
    size_t lds_dst_scalar_per_vector          = 0;
    bool is_direct_load                       = false;
    bool lds_padding                          = false;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr BlockTransfer SetFwdConvABlockTransfer()
{
    constexpr auto& TCL = ALGORITHM.block_transfer.block_transfer_a;
    constexpr auto& TCO = ALGORITHM.block_transfer.block_transfer_access_order_a;
    constexpr auto& SAO = ALGORITHM.block_transfer.src_access_order_a;
    constexpr auto& LDS = ALGORITHM.block_transfer.lds_transfer_a;

    BlockTransfer block_transfer{.thread_cluster_dims  = {TCL.k0, TCL.m_n, TCL.k1},
                                 .thread_cluster_order = {TCO.order[0], TCO.order[1], TCO.order[2]},
                                 .src_access_order     = {SAO.order[0], SAO.order[1], SAO.order[2]},
                                 .src_vector_dim       = LDS.src_vector_dim,
                                 .src_scalar_per_vector     = LDS.src_scalar_per_vector,
                                 .lds_dst_scalar_per_vector = LDS.lds_dst_scalar_per_vector,
                                 .is_direct_load            = LDS.is_direct_load,
                                 .lds_padding               = LDS.lds_padding};
    return block_transfer;
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr BlockTransfer SetFwdConvBBlockTransfer()
{
    constexpr auto& TCL = ALGORITHM.block_transfer.block_transfer_b;
    constexpr auto& TCO = ALGORITHM.block_transfer.block_transfer_access_order_b;
    constexpr auto& SAO = ALGORITHM.block_transfer.src_access_order_b;
    constexpr auto& LDS = ALGORITHM.block_transfer.lds_transfer_b;

    BlockTransfer block_transfer{.thread_cluster_dims  = {TCL.k0, TCL.m_n, TCL.k1},
                                 .thread_cluster_order = {TCO.order[0], TCO.order[1], TCO.order[2]},
                                 .src_access_order     = {SAO.order[0], SAO.order[1], SAO.order[2]},
                                 .src_vector_dim       = LDS.src_vector_dim,
                                 .src_scalar_per_vector     = LDS.src_scalar_per_vector,
                                 .lds_dst_scalar_per_vector = LDS.lds_dst_scalar_per_vector,
                                 .is_direct_load            = LDS.is_direct_load,
                                 .lds_padding               = LDS.lds_padding};
    return block_transfer;
}

// Block transfer parameters for C tensor.
struct CBlockTransfer
{
    size_t m_per_wave_per_shuffle            = 0;
    size_t n_per_wave_per_shuffle            = 0;
    ck::Array<size_t, 4> thread_cluster_dims = {0, 0, 0, 0};
    size_t scalar_per_vector                 = 0;
};

template <ConvSignatureDescriptor auto SIGNATURE, ConvAlgorithmDescriptor auto ALGORITHM>
constexpr CBlockTransfer SetCBlockTransfer()
{
    constexpr auto& TCL = ALGORITHM.block_transfer.thread_cluster_dims_c;
    constexpr auto& EPC = ALGORITHM.block_transfer.epilogue_c;
    CBlockTransfer block_transfer{.m_per_wave_per_shuffle = EPC.m_per_wave_per_shuffle,
                                  .n_per_wave_per_shuffle = EPC.n_per_wave_per_shuffle,
                                  .thread_cluster_dims =
                                      {
                                          TCL.m_block,
                                          TCL.m_wave_per_xdl,
                                          TCL.n_block,
                                          TCL.n_wave_per_xdl,
                                      },
                                  .scalar_per_vector = EPC.scalar_per_vector};
    return block_transfer;
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::LoopScheduler SetLoopScheduler()
{
    constexpr auto loop_scheduler = ALGORITHM.loop_scheduler;

    if constexpr(loop_scheduler == PipelineScheduler::DEFAULT)
    {
        return ck::LoopScheduler::Default;
    }
    else if constexpr(loop_scheduler == PipelineScheduler::INTERWAVE)
    {
        return ck::LoopScheduler::Interwave;
    }
    else
    {
        static_assert(false, "Unknown PipelineScheduler");
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::PipelineVersion SetGridwiseGemmPipelineVersion()
{
    constexpr auto pipeline_version = ALGORITHM.gridwise_gemm.pipeline_version;
    if constexpr(pipeline_version == PipelineVersion::V1)
    {
        return ck::PipelineVersion::v1;
    }
    else if constexpr(pipeline_version == PipelineVersion::V2)
    {
        return ck::PipelineVersion::v2;
    }
    else if constexpr(pipeline_version == PipelineVersion::V3)
    {
        static_assert(false, "V3 is used only for stream-K.");
    }
    else if constexpr(pipeline_version == PipelineVersion::V4)
    {
        return ck::PipelineVersion::v4;
    }
    else if constexpr(pipeline_version == PipelineVersion::WEIGHT_ONLY)
    {
        return ck::PipelineVersion::weight_only;
    }
    else
    {
        static_assert(false, "Unknown PipelineVersion");
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::tensor_operation::device::GemmSpecialization SetGemmSpecialization()
{
    constexpr auto gemm_spec = ALGORITHM.gemm_specialization;

    if constexpr(gemm_spec == GemmSpecialization::Default)
    {
        return ck::tensor_operation::device::GemmSpecialization::Default;
    }
    else if constexpr(gemm_spec == GemmSpecialization::MPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::MPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::NPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::NPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::KPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::KPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::MNPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::MNPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::MKPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::MKPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::NKPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::NKPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::MNKPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::MNKPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::OPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::OPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::MOPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::MOPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::NOPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::NOPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::KOPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::KOPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::MNOPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::MNOPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::MKOPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::MKOPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::NKOPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::NKOPadding;
    }
    else if constexpr(gemm_spec == GemmSpecialization::MNKOPadding)
    {
        return ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
    }
    else
    {
        static_assert(false, "Unknown GemmSpecialization");
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::BlockGemmPipelineVersion SetBlockGemmPipelineVersion()
{
    constexpr auto version = ALGORITHM.pipeline_version;

    if constexpr(version == PipelineVersion::V1)
    {
        return ck::BlockGemmPipelineVersion::v1;
    }
    else if constexpr(version == PipelineVersion::V2)
    {
        return ck::BlockGemmPipelineVersion::v2;
    }
    else if constexpr(version == PipelineVersion::V3)
    {
        return ck::BlockGemmPipelineVersion::v3;
    }
    else if constexpr(version == PipelineVersion::V4)
    {
        return ck::BlockGemmPipelineVersion::v4;
    }
    else if constexpr(version == PipelineVersion::V5)
    {
        return ck::BlockGemmPipelineVersion::v5;
    }
    else
    {
        static_assert(false, "Unknown PipelineVersion");
    }
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
consteval ck::tensor_operation::device::ConvolutionForwardSpecialization SetFwdConvSpecialization()
{
    constexpr auto specialization = ALGORITHM.fwd_specialization;

    if constexpr(specialization == ConvFwdSpecialization::DEFAULT)
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
    }
    else if constexpr(specialization == ConvFwdSpecialization::FILTER_1X1_PAD0)
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Pad0;
    }
    else if constexpr(specialization == ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0)
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter1x1Stride1Pad0;
    }
    else if constexpr(specialization == ConvFwdSpecialization::FILTER_3x3)
    {
        return ck::tensor_operation::device::ConvolutionForwardSpecialization::Filter3x3;
    }
    else
    {
        static_assert(false, "Unknown ConvFwdSpecialization");
    }
}

} // namespace ck_tile::builder::factory_internal

namespace ck_tile::builder {

// Primary template for the convolution factory.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          auto VERSION>
struct ConvFactory;

// Factory specialization for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> &&
             ConvDeviceOpIs_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<SIGNATURE>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE.layout,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using Types         = factory_internal::ConvTensorTypes<SIGNATURE.data_type>;
    using Ops           = factory_internal::ElementwiseOps<SIGNATURE.elementwise_operation>;
    using AlgorithmType = decltype(ALGORITHM);

    static_assert(SpecifiesThreadBlock<AlgorithmType>,
                  "The convolution algorithm descriptor must specify thread block info.");
    static_assert(SpecifiesGridwiseXdlGemm<AlgorithmType>,
                  "The convolution algorithm descriptor must specify gridwise GEMM info.");
    static_assert(SpecifiesBlockTransfer<AlgorithmType>,
                  "The convolution algorithm descriptor must specify block transfer info.");
    static_assert(SpecifiesLdsTransfer<AlgorithmType>,
                  "The convolution algorithm descriptor must specify LDS transfer info.");
    static_assert(
        SpecifiesThreadClusterAccessOrder<AlgorithmType>,
        "The convolution algorithm descriptor must specify thread cluster access order info.");
    static_assert(SpecifiesSourceAccessOrder<AlgorithmType>,
                  "The convolution algorithm descriptor must specify source access order info.");
    static_assert(SpecifiesBlockGemm<AlgorithmType>,
                  "The convolution algorithm descriptor must specify block gemm pipeline.");
    static_assert(SpecifiesFwdConcSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify forward convolution "
                  "specialization.");
    static_assert(SpecifiesGemmSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify gemm specialization.");
    static_assert(ALGORITHM.block_transfer.lds_transfer_a.is_direct_load ==
                      ALGORITHM.block_transfer.lds_transfer_b.is_direct_load,
                  "A and B block transfers must both be direct load or not.");

    static constexpr bool IS_DIRECT_LOAD = ALGORITHM.block_transfer.lds_transfer_a.is_direct_load;
    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<ALGORITHM>();
    static constexpr factory_internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                               .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto BLOCK         = factory_internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM = ALGORITHM.gridwise_gemm;
    static constexpr auto A_BLOCK_TRANSFER =
        factory_internal::SetFwdConvABlockTransfer<ALGORITHM>();
    static constexpr auto B_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBBlockTransfer<ALGORITHM>();
    static constexpr auto C_BLOCK_TRANSFER =
        factory_internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();
    static constexpr auto BLOCK_GEMM = factory_internal::SetBlockGemm<ALGORITHM>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

    // The forward convolution kernel class instance.
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<
        SPATIAL_DIM,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename Layouts::DsLayout,
        typename Layouts::ELayout,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::AccDataType,
        typename Types::CShuffleDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        SPECIALIZATION.conv_spec,
        SPECIALIZATION.gemm_spec,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        BLOCK.per_block.k,
        GRIDWISE_GEMM.ak1,
        GRIDWISE_GEMM.bk1,
        GRIDWISE_GEMM.m_per_xdl,
        GRIDWISE_GEMM.n_per_xdl,
        GRIDWISE_GEMM.m_xdl_per_wave,
        GRIDWISE_GEMM.n_xdl_per_wave,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
        A_BLOCK_TRANSFER.src_vector_dim,
        A_BLOCK_TRANSFER.src_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_padding,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
        B_BLOCK_TRANSFER.src_vector_dim,
        B_BLOCK_TRANSFER.src_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_padding,
        C_BLOCK_TRANSFER.m_per_wave_per_shuffle,
        C_BLOCK_TRANSFER.n_per_wave_per_shuffle,
        to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
        C_BLOCK_TRANSFER.scalar_per_vector,
        BLOCK_GEMM.scheduler,
        BLOCK_GEMM.pipeline_version,
        typename Types::AComputeType,
        typename Types::BComputeType,
        IS_DIRECT_LOAD>;
};

// Factory specialization for DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> &&
             ConvDeviceOpIs_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<SIGNATURE>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE.layout,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using Types         = factory_internal::ConvTensorTypes<SIGNATURE.data_type>;
    using Ops           = factory_internal::ElementwiseOps<SIGNATURE.elementwise_operation>;
    using AlgorithmType = decltype(ALGORITHM);

    static_assert(SpecifiesThreadBlock<AlgorithmType>,
                  "The convolution algorithm descriptor must specify thread block info.");
    static_assert(SpecifiesGridwiseXdlGemm<AlgorithmType>,
                  "The convolution algorithm descriptor must specify gridwise GEMM info.");
    static_assert(SpecifiesBlockTransfer<AlgorithmType>,
                  "The convolution algorithm descriptor must specify block transfer info.");
    static_assert(SpecifiesLdsTransfer<AlgorithmType>,
                  "The convolution algorithm descriptor must specify LDS transfer info.");
    static_assert(
        SpecifiesThreadClusterAccessOrder<AlgorithmType>,
        "The convolution algorithm descriptor must specify thread cluster access order info.");
    static_assert(SpecifiesSourceAccessOrder<AlgorithmType>,
                  "The convolution algorithm descriptor must specify source access order info.");
    static_assert(SpecifiesFwdConcSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify forward convolution "
                  "specialization.");
    static_assert(SpecifiesGemmSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify gemm specialization.");
    static_assert(SpecifiesNumPrefetchStages<AlgorithmType>,
                  "The convolution algorithm descriptor must specify number of prefetch stages.");
    static_assert(SpecifiesLoopScheduler<AlgorithmType>,
                  "The convolution algorithm descriptor must specify loop scheduler.");
    static_assert(SpecifiesNumGroupsToMerge<AlgorithmType>,
                  "The convolution algorithm descriptor must specify number of groups to merge.");

    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<ALGORITHM>();
    static constexpr factory_internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                               .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto LOOP_SCHEDULER = factory_internal::SetLoopScheduler<ALGORITHM>();
    static constexpr auto BLOCK          = factory_internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM  = ALGORITHM.gridwise_gemm;
    static constexpr auto A_BLOCK_TRANSFER =
        factory_internal::SetFwdConvABlockTransfer<ALGORITHM>();
    static constexpr auto B_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBBlockTransfer<ALGORITHM>();
    static constexpr auto C_BLOCK_TRANSFER =
        factory_internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

    // The forward convolution kernel class instance.
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
        SPATIAL_DIM,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename Layouts::DsLayout,
        typename Layouts::ELayout,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::AccDataType,
        typename Types::CShuffleDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        SPECIALIZATION.conv_spec,
        SPECIALIZATION.gemm_spec,
        ALGORITHM.num_gemm_k_prefetch_stages,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        BLOCK.per_block.k,
        GRIDWISE_GEMM.ak1,
        GRIDWISE_GEMM.bk1,
        GRIDWISE_GEMM.m_per_xdl,
        GRIDWISE_GEMM.n_per_xdl,
        GRIDWISE_GEMM.m_xdl_per_wave,
        GRIDWISE_GEMM.n_xdl_per_wave,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
        A_BLOCK_TRANSFER.src_vector_dim,
        A_BLOCK_TRANSFER.src_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_padding,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
        B_BLOCK_TRANSFER.src_vector_dim,
        B_BLOCK_TRANSFER.src_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_padding,
        C_BLOCK_TRANSFER.m_per_wave_per_shuffle,
        C_BLOCK_TRANSFER.n_per_wave_per_shuffle,
        to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
        C_BLOCK_TRANSFER.scalar_per_vector,
        typename Types::AComputeType,
        typename Types::BComputeType,
        LOOP_SCHEDULER,
        ALGORITHM.num_groups_to_merge>;
};

// Factory specialization for DeviceGroupedConvFwdMultipleD_Wmma_CShuffle instance
// of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> &&
             ConvDeviceOpIs_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<SIGNATURE>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE.layout,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using Types         = factory_internal::ConvTensorTypes<SIGNATURE.data_type>;
    using Ops           = factory_internal::ElementwiseOps<SIGNATURE.elementwise_operation>;
    using AlgorithmType = decltype(ALGORITHM);

    static_assert(SpecifiesThreadBlock<AlgorithmType>,
                  "The convolution algorithm descriptor must specify thread block info.");
    static_assert(SpecifiesGridwiseWmmaGemm<AlgorithmType>,
                  "The convolution algorithm descriptor must specify gridwise GEMM info.");
    static_assert(SpecifiesBlockTransfer<AlgorithmType>,
                  "The convolution algorithm descriptor must specify block transfer info.");
    static_assert(SpecifiesLdsTransfer<AlgorithmType>,
                  "The convolution algorithm descriptor must specify LDS transfer info.");
    static_assert(
        SpecifiesThreadClusterAccessOrder<AlgorithmType>,
        "The convolution algorithm descriptor must specify thread cluster access order info.");
    static_assert(SpecifiesSourceAccessOrder<AlgorithmType>,
                  "The convolution algorithm descriptor must specify source access order info.");
    static_assert(SpecifiesFwdConcSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify forward convolution "
                  "specialization.");
    static_assert(SpecifiesNumPrefetchStages<AlgorithmType>,
                  "The convolution algorithm descriptor must specify number of prefetch stages.");
    static_assert(SpecifiesLoopScheduler<AlgorithmType>,
                  "The convolution algorithm descriptor must specify loop scheduler.");

    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<ALGORITHM>();
    static constexpr factory_internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                               .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto LOOP_SCHEDULER = factory_internal::SetLoopScheduler<ALGORITHM>();
    static constexpr auto BLOCK          = factory_internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM  = ALGORITHM.gridwise_gemm;
    static constexpr auto GRIDWISE_GEMM_PIPELINE_VERSION =
        factory_internal::SetGridwiseGemmPipelineVersion<ALGORITHM>();
    static constexpr auto A_BLOCK_TRANSFER =
        factory_internal::SetFwdConvABlockTransfer<ALGORITHM>();
    static constexpr auto B_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBBlockTransfer<ALGORITHM>();
    static constexpr auto C_BLOCK_TRANSFER =
        factory_internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

    // The forward convolution kernel class instance.
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<
        SPATIAL_DIM,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename Layouts::DsLayout,
        typename Layouts::ELayout,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::AccDataType,
        typename Types::CShuffleDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        SPECIALIZATION.conv_spec,
        SPECIALIZATION.gemm_spec,
        ALGORITHM.num_gemm_k_prefetch_stages,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        BLOCK.per_block.k,
        GRIDWISE_GEMM.k1,
        GRIDWISE_GEMM.m_per_wmma,
        GRIDWISE_GEMM.n_per_wmma,
        GRIDWISE_GEMM.m_wmma_per_wave,
        GRIDWISE_GEMM.n_wmma_per_wave,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
        A_BLOCK_TRANSFER.src_vector_dim,
        A_BLOCK_TRANSFER.src_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        A_BLOCK_TRANSFER.lds_padding,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
        to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
        to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
        B_BLOCK_TRANSFER.src_vector_dim,
        B_BLOCK_TRANSFER.src_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
        B_BLOCK_TRANSFER.lds_padding,
        C_BLOCK_TRANSFER.m_per_wave_per_shuffle,
        C_BLOCK_TRANSFER.n_per_wave_per_shuffle,
        to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
        C_BLOCK_TRANSFER.scalar_per_vector,
        LOOP_SCHEDULER,
        GRIDWISE_GEMM_PIPELINE_VERSION>;
};

// Factory specialization for DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK instance
// of a grouped forward convolution kernel using Direct Load (DL) approach.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> &&
             ConvDeviceOpIs_DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<SIGNATURE>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE.layout,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using Types         = factory_internal::ConvTensorTypes<SIGNATURE.data_type>;
    using Ops           = factory_internal::ElementwiseOps<SIGNATURE.elementwise_operation>;
    using AlgorithmType = decltype(ALGORITHM);

    static_assert(SpecifiesThreadBlock<AlgorithmType>,
                  "The convolution algorithm descriptor must specify thread block info.");
    static_assert(SpecifiesFwdConcSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify forward convolution "
                  "specialization.");
    static_assert(SpecifiesGemmSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify gemm specialization.");
    static_assert(SpecifiesDlThreadConfig<AlgorithmType>,
                  "DL algorithm must specify thread config.");
    static_assert(SpecifiesDlThreadCluster<AlgorithmType>,
                  "DL algorithm must specify thread cluster.");
    static_assert(SpecifiesDlBlockTransferA<AlgorithmType>,
                  "DL algorithm must specify A block transfer.");
    static_assert(SpecifiesDlBlockTransferB<AlgorithmType>,
                  "DL algorithm must specify B block transfer.");
    static_assert(SpecifiesDlCThreadTransfer<AlgorithmType>,
                  "DL algorithm must specify C thread transfer.");

    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<ALGORITHM>();

    static constexpr auto BLOCK = factory_internal::SetThreadBlockInfo<ALGORITHM>();

    // DL-specific parameters from algorithm descriptor
    static constexpr auto DL_THREAD_CFG      = ALGORITHM.dl_thread_config;
    static constexpr ck::index_t K0PerBlock  = DL_THREAD_CFG.k0_per_block;
    static constexpr ck::index_t K1          = DL_THREAD_CFG.k1;
    static constexpr ck::index_t M1PerThread = DL_THREAD_CFG.m1_per_thread;
    static constexpr ck::index_t N1PerThread = DL_THREAD_CFG.n1_per_thread;
    static constexpr ck::index_t KPerThread  = DL_THREAD_CFG.k_per_thread;

    // Thread cluster from descriptor
    static constexpr auto DL_CLUSTER = ALGORITHM.dl_thread_cluster;
    using M1N1ThreadClusterM1Xs      = to_sequence_v<DL_CLUSTER.m1_xs>;
    using M1N1ThreadClusterN1Xs      = to_sequence_v<DL_CLUSTER.n1_xs>;

    // A Block Transfer from descriptor - K0_M0_M1_K1 tensor format
    static constexpr auto DL_A_TRANSFER = ALGORITHM.dl_block_transfer_a;
    using ABlockTransferThreadSliceLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.thread_slice_lengths>;
    using ABlockTransferThreadClusterLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.thread_cluster_lengths>;
    using ABlockTransferThreadClusterArrangeOrder =
        to_sequence_v<DL_A_TRANSFER.thread_cluster_arrange_order>;
    using ABlockTransferSrcAccessOrder = to_sequence_v<DL_A_TRANSFER.src_access_order>;
    using ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.src_vector_tensor_lengths>;
    using ABlockTransferSrcVectorTensorContiguousDimOrder =
        to_sequence_v<DL_A_TRANSFER.src_vector_tensor_contiguous_dim_order>;
    using ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1 =
        to_sequence_v<DL_A_TRANSFER.dst_vector_tensor_lengths>;

    // B Block Transfer from descriptor - K0_N0_N1_K1 tensor format
    static constexpr auto DL_B_TRANSFER = ALGORITHM.dl_block_transfer_b;
    using BBlockTransferThreadSliceLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.thread_slice_lengths>;
    using BBlockTransferThreadClusterLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.thread_cluster_lengths>;
    using BBlockTransferThreadClusterArrangeOrder =
        to_sequence_v<DL_B_TRANSFER.thread_cluster_arrange_order>;
    using BBlockTransferSrcAccessOrder = to_sequence_v<DL_B_TRANSFER.src_access_order>;
    using BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.src_vector_tensor_lengths>;
    using BBlockTransferSrcVectorTensorContiguousDimOrder =
        to_sequence_v<DL_B_TRANSFER.src_vector_tensor_contiguous_dim_order>;
    using BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1 =
        to_sequence_v<DL_B_TRANSFER.dst_vector_tensor_lengths>;

    // C Thread Transfer from descriptor
    static constexpr auto DL_C_TRANSFER    = ALGORITHM.dl_c_thread_transfer;
    using CThreadTransferSrcDstAccessOrder = to_sequence_v<DL_C_TRANSFER.src_dst_access_order>;
    static constexpr ck::index_t CThreadTransferSrcDstVectorDim = DL_C_TRANSFER.src_dst_vector_dim;
    static constexpr ck::index_t CThreadTransferDstScalarPerVector =
        DL_C_TRANSFER.dst_scalar_per_vector;

    // The DL forward convolution kernel class instance
    using Instance = ck::tensor_operation::device::DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<
        SPATIAL_DIM,
        typename Types::ADataType,
        typename Types::BDataType,
        typename Types::DsDataTypes,
        typename Types::EDataType,
        typename Types::AccDataType,
        typename Layouts::ALayout,
        typename Layouts::BLayout,
        typename Layouts::DsLayout,
        typename Layouts::ELayout,
        typename Ops::AElementwiseOp,
        typename Ops::BElementwiseOp,
        typename Ops::CDEElementwiseOp,
        FWD_CONV_SPECIALIZATION,
        GEMM_SPECIALIZATION,
        BLOCK.block_size,
        BLOCK.per_block.m,
        BLOCK.per_block.n,
        K0PerBlock,
        K1,
        M1PerThread,
        N1PerThread,
        KPerThread,
        M1N1ThreadClusterM1Xs,
        M1N1ThreadClusterN1Xs,
        ABlockTransferThreadSliceLengths_K0_M0_M1_K1,
        ABlockTransferThreadClusterLengths_K0_M0_M1_K1,
        ABlockTransferThreadClusterArrangeOrder,
        ABlockTransferSrcAccessOrder,
        ABlockTransferSrcVectorTensorLengths_K0_M0_M1_K1,
        ABlockTransferSrcVectorTensorContiguousDimOrder,
        ABlockTransferDstVectorTensorLengths_K0_M0_M1_K1,
        BBlockTransferThreadSliceLengths_K0_N0_N1_K1,
        BBlockTransferThreadClusterLengths_K0_N0_N1_K1,
        BBlockTransferThreadClusterArrangeOrder,
        BBlockTransferSrcAccessOrder,
        BBlockTransferSrcVectorTensorLengths_K0_N0_N1_K1,
        BBlockTransferSrcVectorTensorContiguousDimOrder,
        BBlockTransferDstVectorTensorLengths_K0_N0_N1_K1,
        CThreadTransferSrcDstAccessOrder,
        CThreadTransferSrcDstVectorDim,
        CThreadTransferDstScalarPerVector>;
};

// Factory specialization for DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor instance
// of a grouped forward convolution kernel with large tensor support (N-splitting).
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE> &&
             ConvDeviceOpIs_DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<SIGNATURE>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts       = decltype(factory_internal::GetTensorLayout<SIGNATURE.layout,
                                                                     SPATIAL_DIM,
                                                                     ConvDirection::FORWARD>());
    using Types         = factory_internal::ConvTensorTypes<SIGNATURE.data_type>;
    using Ops           = factory_internal::ElementwiseOps<SIGNATURE.elementwise_operation>;
    using AlgorithmType = decltype(ALGORITHM);

    static_assert(SpecifiesThreadBlock<AlgorithmType>,
                  "The convolution algorithm descriptor must specify thread block info.");
    static_assert(SpecifiesGridwiseXdlGemm<AlgorithmType>,
                  "The convolution algorithm descriptor must specify gridwise GEMM info.");
    static_assert(SpecifiesBlockTransfer<AlgorithmType>,
                  "The convolution algorithm descriptor must specify block transfer info.");
    static_assert(SpecifiesLdsTransfer<AlgorithmType>,
                  "The convolution algorithm descriptor must specify LDS transfer info.");
    static_assert(
        SpecifiesThreadClusterAccessOrder<AlgorithmType>,
        "The convolution algorithm descriptor must specify thread cluster access order info.");
    static_assert(SpecifiesSourceAccessOrder<AlgorithmType>,
                  "The convolution algorithm descriptor must specify source access order info.");
    static_assert(SpecifiesFwdConcSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify forward convolution "
                  "specialization.");
    static_assert(SpecifiesGemmSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify gemm specialization.");
    static_assert(SpecifiesNumPrefetchStages<AlgorithmType>,
                  "The convolution algorithm descriptor must specify number of prefetch stages.");
    static_assert(SpecifiesLoopScheduler<AlgorithmType>,
                  "The convolution algorithm descriptor must specify loop scheduler.");

    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr auto GEMM_SPECIALIZATION =
        factory_internal::SetGemmSpecialization<ALGORITHM>();
    static constexpr factory_internal::ConvSpec SPECIALIZATION{.conv_spec = FWD_CONV_SPECIALIZATION,
                                                               .gemm_spec = GEMM_SPECIALIZATION};

    static constexpr auto LOOP_SCHEDULER = factory_internal::SetLoopScheduler<ALGORITHM>();
    static constexpr auto BLOCK          = factory_internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM  = ALGORITHM.gridwise_gemm;
    static constexpr auto A_BLOCK_TRANSFER =
        factory_internal::SetFwdConvABlockTransfer<ALGORITHM>();
    static constexpr auto B_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBBlockTransfer<ALGORITHM>();
    static constexpr auto C_BLOCK_TRANSFER =
        factory_internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();

    // Check limits for the algorithm parameters.
    static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

    // The forward convolution kernel class instance with large tensor support.
    using Instance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleD_Xdl_CShuffle_Large_Tensor<
            SPATIAL_DIM,
            typename Layouts::ALayout,
            typename Layouts::BLayout,
            typename Layouts::DsLayout,
            typename Layouts::ELayout,
            typename Types::ADataType,
            typename Types::BDataType,
            typename Types::AccDataType,
            typename Types::CShuffleDataType,
            typename Types::DsDataTypes,
            typename Types::EDataType,
            typename Ops::AElementwiseOp,
            typename Ops::BElementwiseOp,
            typename Ops::CDEElementwiseOp,
            SPECIALIZATION.conv_spec,
            SPECIALIZATION.gemm_spec,
            ALGORITHM.num_gemm_k_prefetch_stages,
            BLOCK.block_size,
            BLOCK.per_block.m,
            BLOCK.per_block.n,
            BLOCK.per_block.k,
            GRIDWISE_GEMM.ak1,
            GRIDWISE_GEMM.bk1,
            GRIDWISE_GEMM.m_per_xdl,
            GRIDWISE_GEMM.n_per_xdl,
            GRIDWISE_GEMM.m_xdl_per_wave,
            GRIDWISE_GEMM.n_xdl_per_wave,
            to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
            to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
            to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
            A_BLOCK_TRANSFER.src_vector_dim,
            A_BLOCK_TRANSFER.src_scalar_per_vector,
            A_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
            A_BLOCK_TRANSFER.lds_padding,
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
            to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
            B_BLOCK_TRANSFER.src_vector_dim,
            B_BLOCK_TRANSFER.src_scalar_per_vector,
            B_BLOCK_TRANSFER.lds_dst_scalar_per_vector,
            B_BLOCK_TRANSFER.lds_padding,
            C_BLOCK_TRANSFER.m_per_wave_per_shuffle,
            C_BLOCK_TRANSFER.n_per_wave_per_shuffle,
            to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
            C_BLOCK_TRANSFER.scalar_per_vector,
            typename Types::AComputeType,
            typename Types::BComputeType,
            LOOP_SCHEDULER>;
};

} // namespace ck_tile::builder
