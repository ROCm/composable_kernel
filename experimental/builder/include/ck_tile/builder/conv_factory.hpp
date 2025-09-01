#pragma once

// #include
// "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp"
#include <ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp>
#include <ck_tile/builder/conv_signature.hpp>
#include <ck_tile/builder/conv_algorithm.hpp>
#include <ck_tile/builder/sequence_util.hpp>
#include <ck_tile/builder/versions.h>

namespace ck_tile::builder {

// Type mappings from the builder GroupConvLayout enum class to the CK tensor data types.
template <GroupConvLayout Layout>
struct ConvTensorLayouts;

template <>
struct ConvTensorLayouts<GroupConvLayout::NGCHW_GKCYX_NGKHW>
{
    // Channels first convolution layout.
    using ALayout  = ck::tensor_layout::convolution::NHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKCYX;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NGKHW;
};

template <>
struct ConvTensorLayouts<GroupConvLayout::NHWGC_GKYXC_NHWGK>
{
    // Channels last convolution layout.
    using ALayout  = ck::tensor_layout::convolution::NHWGC;
    using BLayout  = ck::tensor_layout::convolution::GKYXC;
    using DsLayout = ck::Tuple<>;
    using ELayout  = ck::tensor_layout::convolution::NHWGK;
};

// Type mappings from builder convolution data type to CK tensor types.
template <DataType T>
struct ConvTensorTypes;

template <>
struct ConvTensorTypes<DataType::FP16>
{
    using ADataType        = ck::bhalf_t;
    using BDataType        = ck::bhalf_t;
    using CShuffleDataType = ck::bhalf_t;
    using DsDataTypes      = ck::Tuple<>;
    using AccDataType      = float;
    using EDataTYpe        = ck::bhalf_t;
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

template <ConvAlgorithm Algo>
constexpr ConvBlock SetThreadBlockInfo()
{
    if constexpr(HasThreadBlockInfo<Algo>)
    {
        constexpr auto& TB = Algo::THREAD_BLOCK;
        return ConvBlock{
            .block_size = TB.block_size,
            .per_block  = {.m = TB.sub_matrix.m, .n = TB.sub_matrix.n, .k = TB.sub_matrix.k}};
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
    int ak2            = 0;
    int m_per_xdl      = 0;
    int n_per_dxl      = 0;
    int m_xdl_per_wave = 0;
    int n_xdl_per_wave = 0;
};

// Block tranfser paramters for A or B tensor.
struct BlockTransfer
{
    ck::Array<int, 3> thread_cluster_lengths = {0, 0, 0}; // k0, m, k1
    ck::Array<int, 3> thread_cluster_order   = {0, 0, 0};
    ck::Array<int, 3> src_access_order       = {0, 0, 0};
    int src_vector_dim                       = 0;
    int src_scaler_per_vector                = 0;
    int dest_scaler_per_vector_k1            = 0;
    int add_extra                            = 0;
};

// Block transfer parameters for C tensor.
struct CBlockTransfer
{
    int m_xdl_per_wave_per_shuffle    = 0;
    int n_xdl_per_wave_per_shuffle    = 0;
    ck::Array<int, 4> cluster_lengths = {0, 0, 0, 0};
    int scaler_per_vector             = 8;
};

// Factory builds an instance of a grouped convolution kernel.
template <ConvSignature Signature, ConvAlgorithm Algorithm, auto Version>
    requires SupportedVersion<Version>
struct GroupedConvForwardXldCShuffleFactoryV3
{
    static constexpr int SPATIAL_DIM = Signature::SPATIAL_DIM;
    using Layouts                    = ConvTensorLayouts<Signature::LAYOUT>;
    using Types                      = ConvTensorTypes<Signature::DATA_TYPE>;
    using Ops                        = ConvPassThroughOps;
    static constexpr ConvSpec SPECIALIZATION{
        .conv_spec = ck::tensor_operation::device::ConvolutionForwardSpecialization::Default,
        .gemm_spec = ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    };
    static constexpr ConvBlock BLOCK = SetThreadBlockInfo<Algorithm>();
    static constexpr ConvTuning TUNING{
        .ak1            = 8,
        .ak2            = 8,
        .m_per_xdl      = 32,
        .n_per_dxl      = 32,
        .m_xdl_per_wave = 4,
        .n_xdl_per_wave = 4,
    };
    static constexpr BlockTransfer A_BLOCK_TRANSFER{
        .thread_cluster_lengths    = {4, 64, 1},
        .thread_cluster_order      = {1, 0, 2},
        .src_access_order          = {1, 0, 2},
        .src_vector_dim            = 2,
        .src_scaler_per_vector     = 8,
        .dest_scaler_per_vector_k1 = 8,
        .add_extra                 = 0,
    };
    static constexpr BlockTransfer B_BLOCK_TRANSFER{
        .thread_cluster_lengths    = {4, 64, 1},
        .thread_cluster_order      = {1, 0, 2},
        .src_access_order          = {1, 0, 2},
        .src_vector_dim            = 2,
        .src_scaler_per_vector     = 8,
        .dest_scaler_per_vector_k1 = 8,
        .add_extra                 = 0,
    };
    static constexpr CBlockTransfer C_BLOCK_TRANSFER{
        .m_xdl_per_wave_per_shuffle = 1,
        .n_xdl_per_wave_per_shuffle = 1,
        .cluster_lengths            = {1, 32, 1, 8},
        .scaler_per_vector          = 8,
    };
    static constexpr auto PIPELINE_SCHEDULER = ck::BlockGemmPipelineScheduler::Intrawave;
    static constexpr auto PIPELINE_VERSION   = ck::BlockGemmPipelineVersion::v4;
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
            typename Types::EDataTYpe,
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
            TUNING.ak2,
            TUNING.m_per_xdl,
            TUNING.n_per_dxl,
            TUNING.m_xdl_per_wave,
            TUNING.n_xdl_per_wave,
            ToSequence<A_BLOCK_TRANSFER.thread_cluster_lengths>,
            ToSequence<A_BLOCK_TRANSFER.thread_cluster_order>,
            ToSequence<A_BLOCK_TRANSFER.src_access_order>,
            A_BLOCK_TRANSFER.src_vector_dim,
            A_BLOCK_TRANSFER.src_scaler_per_vector,
            A_BLOCK_TRANSFER.dest_scaler_per_vector_k1,
            A_BLOCK_TRANSFER.add_extra,
            ToSequence<B_BLOCK_TRANSFER.thread_cluster_lengths>,
            ToSequence<B_BLOCK_TRANSFER.thread_cluster_order>,
            ToSequence<B_BLOCK_TRANSFER.src_access_order>,
            B_BLOCK_TRANSFER.src_vector_dim,
            B_BLOCK_TRANSFER.src_scaler_per_vector,
            B_BLOCK_TRANSFER.dest_scaler_per_vector_k1,
            B_BLOCK_TRANSFER.add_extra,
            C_BLOCK_TRANSFER.m_xdl_per_wave_per_shuffle,
            C_BLOCK_TRANSFER.n_xdl_per_wave_per_shuffle,
            ToSequence<C_BLOCK_TRANSFER.cluster_lengths>,
            C_BLOCK_TRANSFER.scaler_per_vector,
            PIPELINE_SCHEDULER,
            PIPELINE_VERSION>;
};

} // namespace ck_tile::builder
