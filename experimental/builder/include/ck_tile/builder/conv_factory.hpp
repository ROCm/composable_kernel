// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

// A factory for instantiating CK convolution kernels.
//
// This file translates a semantic description of a convolution operation
// (`ConvSignatureDescriptor` and `ConvAlgorithmDescriptor`) into specific, 
// low-level template arguments required by the underlying CK device-level 
// kernel implementations. This abstraction also enables more complex build
// time logic and simplifies the kernel specification.
//
// Key Components:
//
// Template Metaprogram:
//  - ConvFactory: The main factory, with specializations for different
//                 convolution directions.
//
// Template Metaprogram Helpers:
//  - ConvTensorLayouts: Maps layout enums to CK layout types.
//  - ConvTensorTypes:   Maps data type enums to C++ types used by CK.
//  - ConvPassThroughOps: Hard-coded pass-through element-wise operations.
//
// `constexpr` Helper Functions:
//  - SetThreadBlockInfo:      Determines thread block dimensions from the algorithm
//                             descriptor or provides defaults.
//  - SetConvTuningInfo:       Sets low-level tuning parameters.
//  - Set*BlockTransfer:       Configures tensor data movement parameters for
//                             tensors A, B, and C.
//  - SetBlockGemmPipelineVersion: Selects the GEMM pipeline version.
//
// The primary entry point is the `ConvFactory` struct, which is specialized
// for forward and backward-data convolutions.

#pragma once

#include <ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>
#include <ck_tile/builder/conv_signature_concepts.hpp>
#include <ck_tile/builder/conv_algorithm_concepts.hpp>
#include <ck_tile/builder/conv_algorithm_limits.hpp>
#include <ck_tile/builder/builder_utils.hpp>
#include <ck_tile/builder/types.hpp>
#include <ck_tile/builder/versions.hpp>

namespace ck_tile::builder::factory_internal {

// Type mappings from the builder GroupConvLayout enum class to the CK tensor data types.
template <GroupConvLayout Layout, size_t SPATIAL_DIM, ConvDirection DIR>
    requires(ConvSpatialDim<SPATIAL_DIM>)
struct ConvTensorLayouts
{
    // This will trigger if a specialization for the given layout is not found.
    // We should always catch this in an earlier validation check.
    static_assert(sizeof(Layout) == 0,
                  "Internal error. Unsupported layout for convolution factory.");
};

template <>
struct ConvTensorLayouts<GroupConvLayout::CHANNELS_FIRST, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKCYX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout::CHANNELS_LAST, 2, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NHWGK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout::CHANNELS_FIRST, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NDHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKCZYX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKDHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout::CHANNELS_LAST, 3, ConvDirection::FORWARD>
{
    using ALayout  = ck::tensor_layout::convolution::NDHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKZYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NDHWGK;
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

// Hard-coded pass-through ops.
// TODO: Generalize this for more fused operations.
struct ConvPassThroughOps
{
    using AElementwiseOp   = ck::tensor_operation::element_wise::PassThrough;
    using BElementwiseOp   = ck::tensor_operation::element_wise::PassThrough;
    using CDEElementwiseOp = ck::tensor_operation::element_wise::PassThrough;
};

// The algorithm specializations for the convolution and GEMM.
template <typename CONV_ENUM>
    requires(
        std::is_same_v<CONV_ENUM, ck::tensor_operation::device::ConvolutionForwardSpecialization> 
        // ||
        // std::is_same_v<CONV_ENUM,
        //                ck::tensor_operation::device::ConvolutionBackwardDataSpecialization>
                    )
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
    MNK per_block  = {};
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr ConvBlock SetThreadBlockInfo()
{
    constexpr auto& TB = ALGORITHM.thread_block;
    return ConvBlock{
        .block_size = TB.block_size,
        .per_block  = {.m = TB.tile_size.m, .n = TB.tile_size.n, .k = TB.tile_size.k}
    };
}

// Convolution tuning parameters.
struct ConvTuning
{
    size_t ak1            = 0;
    size_t bk1            = 0;
    size_t m_per_xdl      = 0;
    size_t n_per_xdl      = 0;
    size_t m_xdl_per_wave = 0;
    size_t n_xdl_per_wave = 0;
};

template <ConvSignatureDescriptor auto SIGNATURE, ConvAlgorithmDescriptor auto ALGORITHM>
constexpr ConvTuning SetConvTuningInfo()
{
    constexpr auto& TP = ALGORITHM.tuning_params;
    return ConvTuning{
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
    size_t dest_scalar_per_vector_k1          = 0;
    size_t add_extra                          = 0;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr BlockTransfer SetFwdConvABlockTransfer()
{
    constexpr auto& TCL              = ALGORITHM.block_transfer.thread_cluster_dims_a;
    constexpr auto& TCO              = ALGORITHM.block_transfer.thread_cluster_access_order_a;
    constexpr auto& SAO              = ALGORITHM.block_transfer.src_access_order_a;
    constexpr auto& VTD              = ALGORITHM.block_transfer.vector_transfer_a;

    BlockTransfer block_transfer{
        .thread_cluster_dims       = {TCL.k0, TCL.m_n, TCL.k1},
        .thread_cluster_order      = {TCO.order[0], TCO.order[1], TCO.order[2]},
        .src_access_order          = {SAO.order[0], SAO.order[1], SAO.order[2]},
        .src_vector_dim            = VTD.src_vector_dim,
        .src_scalar_per_vector     = VTD.src_scalar_per_vector,
        .dest_scalar_per_vector_k1 = VTD.dest_scalar_per_vector_k1, 
        .add_extra                 = VTD.add_extra 
    };
    return block_transfer;
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr BlockTransfer SetFwdConvBBlockTransfer()
{
    constexpr auto& TCL              = ALGORITHM.block_transfer.thread_cluster_dims_b;
    constexpr auto& TCO              = ALGORITHM.block_transfer.thread_cluster_access_order_b;
    constexpr auto& SAO              = ALGORITHM.block_transfer.src_access_order_b;
    constexpr auto& VTD              = ALGORITHM.block_transfer.vector_transfer_b;

    BlockTransfer block_transfer{
        .thread_cluster_dims       = {TCL.k0, TCL.m_n, TCL.k1},
        .thread_cluster_order      = {TCO.order[0], TCO.order[1], TCO.order[2]},
        .src_access_order          = {SAO.order[0], SAO.order[1], SAO.order[2]},
        .src_vector_dim            = VTD.src_vector_dim,
        .src_scalar_per_vector     = VTD.src_scalar_per_vector,
        .dest_scalar_per_vector_k1 = VTD.dest_scalar_per_vector_k1, 
        .add_extra                 = VTD.add_extra 
    };
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
    constexpr auto& VTC = ALGORITHM.block_transfer.vector_transfer_c;
    CBlockTransfer block_transfer 
    {
        .m_xdl_per_wave_per_shuffle = VTC.m_xdl_per_wave_per_shuffle,
        .n_xdl_per_wave_per_shuffle = VTC.n_xdl_per_wave_per_shuffle,
        .thread_cluster_dims        = {
                    TCL.m_block,
                    TCL.m_wave_per_xdl,
                    TCL.n_block,
                    TCL.n_wave_per_xdl,
                },
        .scalar_per_vector          = VTC.scalar_per_vector
    };
    return block_transfer;
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr ck::BlockGemmPipelineVersion SetBlockGemmPipelineVersion()
{
    switch(ALGORITHM.pipeline_version)
    {
    case BlockGemmPipelineVersion::V1: return ck::BlockGemmPipelineVersion::v1;
    case BlockGemmPipelineVersion::V3: return ck::BlockGemmPipelineVersion::v3;
    case BlockGemmPipelineVersion::V4: return ck::BlockGemmPipelineVersion::v4;
    case BlockGemmPipelineVersion::V5: return ck::BlockGemmPipelineVersion::v5;
    default:                           return ck::BlockGemmPipelineVersion::v4;
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
    using Layouts = factory_internal::ConvTensorLayouts<SIGNATURE.layout, SPATIAL_DIM, ConvDirection::FORWARD>;
    using Types   = factory_internal::ConvTensorTypes<SIGNATURE.data_type>;
    using Ops     = factory_internal::ConvPassThroughOps;
    using AlgorithmType = decltype(ALGORITHM);

    // Check preconditions for the algorithm description.
    static_assert(SPATIAL_DIM == 2 || SPATIAL_DIM == 3,
                  "Only 2D and 3D convolutions are supported in this factory.");
    static_assert(SpecifiesThreadBlock<AlgorithmType>, 
                  "The convolution algorithm descriptor must specify thread block info.");
    static_assert(SpecifiesGridwiseGemm<AlgorithmType>,
                  "The convolution algorithm descriptor must specify gridwise GEMM info.");
    static_assert(SpecifiesBlockTransfer<AlgorithmType>,
                 "The convolution algorithm descriptor must specify block transfer info.");
    static_assert(SpecifiesBlockVectorTransfer<AlgorithmType>,
                 "The convolution algorithm descriptor must specify block vector transfer info.");
    static_assert(SpecifiesThreadClusterAccessOrder<AlgorithmType>,
                 "The convolution algorithm descriptor must specify thread cluster access order info.");
    static_assert(SpecifiesSourceAccessOrder<AlgorithmType>,
                 "The convolution algorithm descriptor must specify source access order info.");
    static_assert(SpecifiesGemmPipelineVersion<AlgorithmType>,
                 "The convolution algorithm descriptor must specify block gemm pipeline version.");


    static constexpr factory_internal::ConvSpec SPECIALIZATION{
        .conv_spec = ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        .gemm_spec = ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    };
    static constexpr auto BLOCK              = factory_internal::SetThreadBlockInfo<ALGORITHM>();
    static constexpr auto TUNING             = factory_internal::SetConvTuningInfo<SIGNATURE, ALGORITHM>();
    static constexpr auto A_BLOCK_TRANSFER   = factory_internal::SetFwdConvABlockTransfer<ALGORITHM>();
    static constexpr auto B_BLOCK_TRANSFER   = factory_internal::SetFwdConvBBlockTransfer<ALGORITHM>();
    static constexpr auto C_BLOCK_TRANSFER   = factory_internal::SetCBlockTransfer<SIGNATURE, ALGORITHM>();
    static constexpr auto PIPELINE_SCHEDULER = ck::BlockGemmPipelineScheduler::Intrawave;
    static constexpr auto PIPELINE_VERSION   = factory_internal::SetBlockGemmPipelineVersion<ALGORITHM>();

    // Check limits for the algorithm parameters.
    // TODO: Add more limits checks as needed.
    // static_assert(InputVectorTransferLimits<A_BLOCK_TRANSFER>);
    // static_assert(InputVectorTransferLimits<B_BLOCK_TRANSFER>);
    // static_assert(OutputVectorTransferLimits<C_BLOCK_TRANSFER>);
    // static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.thread_cluster_order>);
    // static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.thread_cluster_order>);
    // static_assert(AccessOrderLimits<A_BLOCK_TRANSFER.src_access_order>);
    // static_assert(AccessOrderLimits<B_BLOCK_TRANSFER.src_access_order>);

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
            TUNING.ak1,
            TUNING.bk1,
            TUNING.m_per_xdl,
            TUNING.n_per_xdl,
            TUNING.m_xdl_per_wave,
            TUNING.n_xdl_per_wave,
            to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
            to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
            to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
            A_BLOCK_TRANSFER.src_vector_dim,
            A_BLOCK_TRANSFER.src_scalar_per_vector,
            A_BLOCK_TRANSFER.dest_scalar_per_vector_k1,
            A_BLOCK_TRANSFER.add_extra,
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
            to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
            B_BLOCK_TRANSFER.src_vector_dim,
            B_BLOCK_TRANSFER.src_scalar_per_vector,
            B_BLOCK_TRANSFER.dest_scalar_per_vector_k1,
            B_BLOCK_TRANSFER.add_extra,
            C_BLOCK_TRANSFER.m_xdl_per_wave_per_shuffle,
            C_BLOCK_TRANSFER.n_xdl_per_wave_per_shuffle,
            to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
            C_BLOCK_TRANSFER.scalar_per_vector,
            PIPELINE_SCHEDULER,
            PIPELINE_VERSION>;
};

} // namespace ck_tile::builder
