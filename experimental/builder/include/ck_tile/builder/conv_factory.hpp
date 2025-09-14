#pragma once

// #include
// "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp"
#include <ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>
#include <ck_tile/builder/conv_signature.hpp>
#include <ck_tile/builder/conv_algorithm.hpp>
#include <ck_tile/builder/builder_utils.hpp>
#include <ck_tile/builder/types.hpp>
#include <ck_tile/builder/versions.h>

namespace ck_tile::builder {

// Type mappings from the builder GroupConvLayout enum class to the CK tensor data types.
template <GroupConvLayout Layout, int SPATIAL_DIM>
    requires(ConvSpatialDim<SPATIAL_DIM>)
struct ConvTensorLayouts
{
    // This will trigger if a specialization for the given layout is not found.
    // We should always catch this in an earlier validation check.
    static_assert(sizeof(Layout) == 0,
                  "Internal error. Unsupported layout for convolution factory.");
};

template <>
struct ConvTensorLayouts<GroupConvLayout::CHANNELS_FIRST, 2>
{
    // Channels first convolution layout.
    using ALayout  = ck::tensor_layout::convolution::NHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKCYX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout::CHANNELS_LAST, 2>
{
    // Channels last convolution layout.
    using ALayout  = ck::tensor_layout::convolution::NHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NHWGK;
};

template <>
struct ConvTensorLayouts<GroupConvLayout::CHANNELS_LAST, 3>
{
    // Channels last convolution layout.
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
    static_assert(sizeof(unsupported_data_type<T>) == 0,
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
struct ConvPassThroughOps
{
    using AElementwiseOp   = ck::tensor_operation::element_wise::PassThrough;
    using BElementwiseOp   = ck::tensor_operation::element_wise::PassThrough;
    using CDEElementwiseOp = ck::tensor_operation::element_wise::PassThrough;
};

// The specializations for the convolution and GEMM.
struct ConvSpec
{
    ck::tensor_operation::device::ConvolutionForwardSpecialization conv_spec =
        ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
    ck::tensor_operation::device::GemmSpecialization gemm_spec =
        ck::tensor_operation::device::GemmSpecialization::MNKPadding;
};

// Block info for a convlution.
struct ConvBlock
{
    int block_size = 0;
    MNK<int> per_block;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr ConvBlock SetThreadBlockInfo()
{
    using AlgorithmType = decltype(ALGORITHM);
    if constexpr(SpecifiesThreadBlock<AlgorithmType>)
    {
        constexpr auto& TB = ALGORITHM.thread_block;
        return ConvBlock{
            .block_size = TB.block_size,
            .per_block  = {.m = TB.submatrix.m, .n = TB.submatrix.n, .k = TB.submatrix.k}};
    }
    // Default values if  thread block info isn't specified.
    return ConvBlock{
        .block_size = 256,
        .per_block  = {.m = 256, .n = 256, .k = 32},
    };
}

// Convolution tuning parameters.
struct ConvTuning
{
    int ak1            = 0;
    int bk1            = 0;
    int m_per_xdl      = 0;
    int n_per_dxl      = 0;
    int m_xdl_per_wave = 0;
    int n_xdl_per_wave = 0;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr ConvTuning SetConvTuningInfo()
{
    using AlgorithmType = decltype(ALGORITHM);
    if constexpr(SpecifiesConvTuning<AlgorithmType>)
    {
        constexpr auto& TP = ALGORITHM.tuning_params;
        return ConvTuning{
            .ak1            = TP.ak1,
            .bk1            = TP.bk1,
            .m_per_xdl      = 32,
            .n_per_dxl      = 32,
            .m_xdl_per_wave = TP.m_xdl_per_wave,
            .n_xdl_per_wave = TP.n_xdl_per_wave,
        };
    }
    // Default values.
    return ConvTuning{
        .ak1            = 8,
        .bk1            = 8,
        .m_per_xdl      = 32,
        .n_per_dxl      = 32,
        .m_xdl_per_wave = 4,
        .n_xdl_per_wave = 4,
    };
}

// Block transfer paramters for A or B tensor.
struct BlockTransfer
{
    ck::Array<int, 3> thread_cluster_dims  = {0, 0, 0}; // k0, m, k1
    ck::Array<int, 3> thread_cluster_order = {0, 0, 0};
    ck::Array<int, 3> src_access_order     = {0, 0, 0};
    int src_vector_dim                     = 0;
    int src_scaler_per_vector              = 0;
    int dest_scaler_per_vector_k1          = 0;
    int add_extra                          = 0;
};

// Block transfer parameters for C tensor.
struct CBlockTransfer
{
    int m_xdl_per_wave_per_shuffle        = 0;
    int n_xdl_per_wave_per_shuffle        = 0;
    ck::Array<int, 4> thread_cluster_dims = {0, 0, 0, 0};
    int scaler_per_vector                 = 8;
};

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr BlockTransfer SetABlockTransfer()
{
    BlockTransfer block_transfer{
        .thread_cluster_dims       = {4, 64, 1},
        .thread_cluster_order      = {1, 0, 2},
        .src_access_order          = {1, 0, 2},
        .src_vector_dim            = 2,
        .src_scaler_per_vector     = 8,
        .dest_scaler_per_vector_k1 = 8,
        .add_extra                 = 0,
    };
    using AlgorithmType = decltype(ALGORITHM);
    if constexpr(SpecifiesBlockATransfer<AlgorithmType>)
    {
        constexpr auto& TCL                = ALGORITHM.block_transfer.thread_cluster_dims_a;
        block_transfer.thread_cluster_dims = {TCL.k0, TCL.m, TCL.k1};
    }
    // Default.
    return block_transfer;
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr BlockTransfer SetBBlockTransfer()
{
    BlockTransfer block_transfer{
        .thread_cluster_dims       = {4, 64, 1},
        .thread_cluster_order      = {1, 0, 2},
        .src_access_order          = {1, 0, 2},
        .src_vector_dim            = 2,
        .src_scaler_per_vector     = 8,
        .dest_scaler_per_vector_k1 = 8,
        .add_extra                 = 0,
    };
    using AlgorithmType = decltype(ALGORITHM);
    if constexpr(SpecifiesBlockBTransfer<AlgorithmType>)
    {
        constexpr auto& TCL                = ALGORITHM.block_transfer.thread_cluster_dims_b;
        block_transfer.thread_cluster_dims = {TCL.k0, TCL.n, TCL.k1};
    }
    // Default.
    return block_transfer;
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr CBlockTransfer SetCBlockTransfer()
{
    CBlockTransfer block_transfer{
        .m_xdl_per_wave_per_shuffle = 1,
        .n_xdl_per_wave_per_shuffle = 1,
        .thread_cluster_dims        = {1, 32, 1, 8},
        .scaler_per_vector          = 8,
    };
    using AlgorithmType = decltype(ALGORITHM);
    if constexpr(SpecifiesBlockCTransfer<AlgorithmType>)
    {
        constexpr auto& TCL                = ALGORITHM.block_transfer.thread_cluster_dims_c;
        block_transfer.thread_cluster_dims = {
            TCL.m_block,
            TCL.m_wave_per_xdl,
            TCL.n_block,
            TCL.n_wave_per_xdl,
        };
    }
    return block_transfer;
}

template <ConvAlgorithmDescriptor auto ALGORITHM>
constexpr ck::BlockGemmPipelineVersion SetBlockGemmPipelineVersion()
{
    using AlgorithmType = decltype(ALGORITHM);
    if constexpr(SpecifiesGemmPipelineVersion<AlgorithmType>)
    {
        switch(ALGORITHM.pipeline_version)
        {
        case BlockGemmPipelineVersion::V1: return ck::BlockGemmPipelineVersion::v1;
        case BlockGemmPipelineVersion::V3: return ck::BlockGemmPipelineVersion::v3;
        case BlockGemmPipelineVersion::V4: return ck::BlockGemmPipelineVersion::v4;
        case BlockGemmPipelineVersion::V5: return ck::BlockGemmPipelineVersion::v5;
        }
    }
    // Default value.
    return ck::BlockGemmPipelineVersion::v4;
}

// Factory builds an instance of a grouped convolution kernel.
template <ConvSignatureDescriptor auto SIGNATURE,
          ConvAlgorithmDescriptor auto ALGORITHM,
          auto Version>
    requires SupportedVersion<Version>
struct ConvFactory
{
    static constexpr int SPATIAL_DIM = SIGNATURE.spatial_dim;
    using Layouts                    = ConvTensorLayouts<SIGNATURE.layout, SPATIAL_DIM>;
    using Types                      = ConvTensorTypes<SIGNATURE.data_type>;
    using Ops                        = ConvPassThroughOps;
    static constexpr ConvSpec SPECIALIZATION{
        .conv_spec = ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        .gemm_spec = ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    };
    static constexpr ConvBlock BLOCK                 = SetThreadBlockInfo<ALGORITHM>();
    static constexpr ConvTuning TUNING               = SetConvTuningInfo<ALGORITHM>();
    static constexpr BlockTransfer A_BLOCK_TRANSFER  = SetABlockTransfer<ALGORITHM>();
    static constexpr BlockTransfer B_BLOCK_TRANSFER  = SetBBlockTransfer<ALGORITHM>();
    static constexpr CBlockTransfer C_BLOCK_TRANSFER = SetCBlockTransfer<ALGORITHM>();
    static constexpr auto PIPELINE_SCHEDULER         = ck::BlockGemmPipelineScheduler::Intrawave;
    static constexpr auto PIPELINE_VERSION           = SetBlockGemmPipelineVersion<ALGORITHM>();
    // The convlution kernel class instance.
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
            TUNING.n_per_dxl,
            TUNING.m_xdl_per_wave,
            TUNING.n_xdl_per_wave,
            to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_dims>,
            to_sequence_v<A_BLOCK_TRANSFER.thread_cluster_order>,
            to_sequence_v<A_BLOCK_TRANSFER.src_access_order>,
            A_BLOCK_TRANSFER.src_vector_dim,
            A_BLOCK_TRANSFER.src_scaler_per_vector,
            A_BLOCK_TRANSFER.dest_scaler_per_vector_k1,
            A_BLOCK_TRANSFER.add_extra,
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_dims>,
            to_sequence_v<B_BLOCK_TRANSFER.thread_cluster_order>,
            to_sequence_v<B_BLOCK_TRANSFER.src_access_order>,
            B_BLOCK_TRANSFER.src_vector_dim,
            B_BLOCK_TRANSFER.src_scaler_per_vector,
            B_BLOCK_TRANSFER.dest_scaler_per_vector_k1,
            B_BLOCK_TRANSFER.add_extra,
            C_BLOCK_TRANSFER.m_xdl_per_wave_per_shuffle,
            C_BLOCK_TRANSFER.n_xdl_per_wave_per_shuffle,
            to_sequence_v<C_BLOCK_TRANSFER.thread_cluster_dims>,
            C_BLOCK_TRANSFER.scaler_per_vector,
            PIPELINE_SCHEDULER,
            PIPELINE_VERSION>;
};

} // namespace ck_tile::builder
