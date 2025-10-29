// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp"
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
    using BDataType        = ck::half_t;
    using CShuffleDataType = ck::half_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = ck::half_t;
};

template <>
struct ConvTensorTypes<DataType::BF16>
{
    using ADataType        = ck::bhalf_t;
    using BDataType        = ck::bhalf_t;
    using CShuffleDataType = ck::bhalf_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = ck::bhalf_t;
};

template <>
struct ConvTensorTypes<DataType::FP32>
{
    using ADataType        = float;
    using BDataType        = float;
    using CShuffleDataType = float;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataType        = float;
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

// Convolution tuning parameters.
struct GridwiseGemm
{
    size_t ak1            = 0;
    size_t bk1            = 0;
    size_t m_per_xdl      = 0;
    size_t n_per_xdl      = 0;
    size_t m_xdl_per_wave = 0;
    size_t n_xdl_per_wave = 0;
};

template <ConvSignatureDescriptor auto SIGNATURE, ConvAlgorithmDescriptor auto ALGORITHM>
constexpr GridwiseGemm SetGridwiseGemmInfo()
{
    constexpr auto& TP = ALGORITHM.gridwise_gemm;
    return GridwiseGemm{
        .ak1            = TP.ak1,
        .bk1            = TP.bk1,
        .m_per_xdl      = TP.m_per_xdl,
        .n_per_xdl      = TP.n_per_xdl,
        .m_xdl_per_wave = TP.m_xdl_per_wave,
        .n_xdl_per_wave = TP.n_xdl_per_wave,
    };
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
    size_t m_xdl_per_wave_per_shuffle        = 0;
    size_t n_xdl_per_wave_per_shuffle        = 0;
    ck::Array<size_t, 4> thread_cluster_dims = {0, 0, 0, 0};
    size_t scalar_per_vector                 = 0;
};

template <ConvSignatureDescriptor auto SIGNATURE, ConvAlgorithmDescriptor auto ALGORITHM>
constexpr CBlockTransfer SetCBlockTransfer()
{
    constexpr auto& TCL = ALGORITHM.block_transfer.thread_cluster_dims_c;
    constexpr auto& EPC = ALGORITHM.block_transfer.epilogue_c;
    CBlockTransfer block_transfer{.m_xdl_per_wave_per_shuffle = EPC.m_xdl_per_wave_per_shuffle,
                                  .n_xdl_per_wave_per_shuffle = EPC.n_xdl_per_wave_per_shuffle,
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
consteval ck::BlockGemmPipelineVersion SetBlockGemmPipelineVersion()
{
    constexpr auto version = ALGORITHM.pipeline_version;

    if constexpr(version == BlockGemmPipelineVersion::V1)
    {
        return ck::BlockGemmPipelineVersion::v1;
    }
    else if constexpr(version == BlockGemmPipelineVersion::V2)
    {
        return ck::BlockGemmPipelineVersion::v2;
    }
    else if constexpr(version == BlockGemmPipelineVersion::V3)
    {
        return ck::BlockGemmPipelineVersion::v3;
    }
    else if constexpr(version == BlockGemmPipelineVersion::V4)
    {
        return ck::BlockGemmPipelineVersion::v4;
    }
    else if constexpr(version == BlockGemmPipelineVersion::V5)
    {
        return ck::BlockGemmPipelineVersion::v5;
    }
    else
    {
        static_assert(false, "Unknown BlockGemmPipelineVersion");
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

// Factory specialization for an instance of a grouped forward convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          StringLiteral VERSION>
    requires ConvDirectionIsForward<SIGNATURE>
struct ConvFactory<SIGNATURE, ALGORITHM, VERSION>
{
    static constexpr size_t SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts =
        factory_internal::ConvTensorLayouts<SIGNATURE.layout, SPATIAL_DIM, ConvDirection::FORWARD>;
    using Types         = factory_internal::ConvTensorTypes<SIGNATURE.data_type>;
    using Ops           = factory_internal::ElementwiseOps<SIGNATURE.elementwise_operation>;
    using AlgorithmType = decltype(ALGORITHM);

    static_assert(SpecifiesThreadBlock<AlgorithmType>,
                  "The convolution algorithm descriptor must specify thread block info.");
    static_assert(SpecifiesGridwiseGemm<AlgorithmType>,
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
    static_assert(SpecifiesGemmPipelineVersion<AlgorithmType>,
                  "The convolution algorithm descriptor must specify block gemm pipeline version.");
    static_assert(SpecifiesFwdConcSpecialization<AlgorithmType>,
                  "The convolution algorithm descriptor must specify forward convolution "
                  "specialization.");

    static constexpr auto FWD_CONV_SPECIALIZATION =
        factory_internal::SetFwdConvSpecialization<ALGORITHM>();
    static constexpr factory_internal::ConvSpec SPECIALIZATION{
        .conv_spec = FWD_CONV_SPECIALIZATION,
        .gemm_spec = ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    };
    static constexpr auto BLOCK = factory_internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto GRIDWISE_GEMM =
        factory_internal::SetGridwiseGemmInfo<SIGNATURE, ALGORITHM>();
    static constexpr auto A_BLOCK_TRANSFER =
        factory_internal::SetFwdConvABlockTransfer<ALGORITHM>();
    static constexpr auto B_BLOCK_TRANSFER =
        factory_internal::SetFwdConvBBlockTransfer<ALGORITHM>();
    static constexpr auto C_BLOCK_TRANSFER =
        factory_internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();
    static constexpr auto PIPELINE_SCHEDULER = ck::BlockGemmPipelineScheduler::Intrawave;
    static constexpr auto PIPELINE_VERSION =
        factory_internal::SetBlockGemmPipelineVersion<ALGORITHM>();

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
    using Instance =
        ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3< //
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
            C_BLOCK_TRANSFER.m_xdl_per_wave_per_shuffle,
            C_BLOCK_TRANSFER.n_xdl_per_wave_per_shuffle,
            to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
            C_BLOCK_TRANSFER.scalar_per_vector,
            PIPELINE_SCHEDULER,
            PIPELINE_VERSION>;
};

} // namespace ck_tile::builder
