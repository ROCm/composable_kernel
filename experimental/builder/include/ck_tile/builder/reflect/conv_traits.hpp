// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/conv_factory.hpp>
#include <ck_tile/builder/conv_signature_concepts.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck_tile/builder/types.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>

namespace ck_tile::reflect::conv {

/// @brief Helper structures for organizing trait data with domain-specific naming

/// @brief Data tile dimensions processed by workgroup
struct DataTileInfo
{
    int m; // Processed tile m dimension
    int n; // Processed tile n dimension
    int k; // Processed tile k dimension
};

struct InputTileTransferDimensions
{
    int k0;
    int m_or_n; // m for A transfer, n for B transfer
    int k1;
};

struct InputTileTransferParams
{
    int k1;
    std::array<int, 3> thread_cluster_dims;
    std::array<int, 3> thread_cluster_order;
    std::array<int, 3> src_access_order;
    int src_vector_dim;
    int src_scalar_per_vector;
    int dst_scalar_per_vector_k1;
    bool lds_padding;
};

struct InputTileTransferInfo
{
    InputTileTransferDimensions tile_dimensions;
    InputTileTransferParams transfer_params;
};

struct WarpGemmParams
{
    int gemm_m;
    int gemm_n;
    int num_m_gemms;
    int num_n_gemms;
};

struct WarpShuffleParams
{
    int m_gemms_per_shuffle;
    int n_gemms_per_shuffle;
};

struct OutputTileTransferInfo
{
    WarpShuffleParams shuffle_params;
    // m_block, m_wave_per_xdl, n_block, n_wave_per_xdl
    std::array<int, 4> thread_cluster_dims;
    int scalar_per_vector;
};

// Helper metafunctions to derive signature information from Instance types

// Derive ConvDirection from device kernel type
template <typename Instance>
constexpr builder::ConvDirection conv_direction()
{
    using InstTraits = InstanceTraits<Instance>;

    // Check if conv_forward_specialization exists
    if constexpr(requires { &InstTraits::kConvForwardSpecialization; })
    {
        return builder::ConvDirection::FORWARD;
    }
    // Check if kConvBwdDataSpecialization exists
    else if constexpr(requires { &InstTraits::kConvBwdDataSpecialization; })
    {
        return builder::ConvDirection::BACKWARD_DATA;
    }
    else if constexpr(requires { &InstTraits::kConvBwdWeightSpecialization; })
    {
        return builder::ConvDirection::BACKWARD_WEIGHT;
    }
    else
    {
        return builder::ConvDirection::FORWARD; // Default fallback
    }
}

// Derive GroupConvLayout from layout types/devel/composable_kernel
template <typename Instance>
constexpr auto conv_layout()
{
    using InstTraits = InstanceTraits<Instance>;
    using ALayout    = typename InstTraits::ALayout;
    using BLayout    = typename InstTraits::BLayout;
    using ELayout    = typename InstTraits::ELayout;

    namespace ctc = ck::tensor_layout::convolution;

    if constexpr(InstTraits::kSpatialDim == 1)
    {
        if constexpr(std::is_same_v<ALayout, ctc::GNWC> && std::is_same_v<BLayout, ctc::GKXC> &&
                     std::is_same_v<ELayout, ctc::GNWK>)
        {
            return builder::GroupConvLayout1D::GNWC_GKXC_GNWK;
        }
        else if constexpr(std::is_same_v<ALayout, ctc::NWGC> &&
                          std::is_same_v<BLayout, ctc::GKXC> && std::is_same_v<ELayout, ctc::NWGK>)
        {
            return builder::GroupConvLayout1D::NWGC_GKXC_NWGK;
        }
        else if constexpr(std::is_same_v<ALayout, ctc::NGCW> &&
                          std::is_same_v<BLayout, ctc::GKXC> && std::is_same_v<ELayout, ctc::NGKW>)
        {
            return builder::GroupConvLayout1D::NGCW_GKXC_NGKW;
        }
        else if constexpr(std::is_same_v<ALayout, ctc::NGCW> &&
                          std::is_same_v<BLayout, ctc::GKCX> && std::is_same_v<ELayout, ctc::NGKW>)
        {
            return builder::GroupConvLayout1D::NGCW_GKCX_NGKW;
        }
    }
    else if constexpr(InstTraits::kSpatialDim == 2)
    {
        if constexpr(std::is_same_v<ALayout, ctc::GNHWC> && std::is_same_v<BLayout, ctc::GKYXC> &&
                     std::is_same_v<ELayout, ctc::GNHWK>)
        {
            return builder::GroupConvLayout2D::GNHWC_GKYXC_GNHWK;
        }
        else if constexpr(std::is_same_v<ALayout, ctc::NHWGC> &&
                          std::is_same_v<BLayout, ctc::GKYXC> &&
                          std::is_same_v<ELayout, ctc::NHWGK>)
        {
            return builder::GroupConvLayout2D::NHWGC_GKYXC_NHWGK;
        }
        else if constexpr(std::is_same_v<ALayout, ctc::NGCHW> &&
                          std::is_same_v<BLayout, ctc::GKYXC> &&
                          std::is_same_v<ELayout, ctc::NGKHW>)
        {
            return builder::GroupConvLayout2D::NGCHW_GKYXC_NGKHW;
        }
        else if constexpr(std::is_same_v<ALayout, ctc::NGCHW> &&
                          std::is_same_v<BLayout, ctc::GKCYX> &&
                          std::is_same_v<ELayout, ctc::NGKHW>)
        {
            return builder::GroupConvLayout2D::NGCHW_GKCYX_NGKHW;
        }
    }
    else if constexpr(InstTraits::kSpatialDim == 3)
    {
        if constexpr(std::is_same_v<ALayout, ctc::GNDHWC> && std::is_same_v<BLayout, ctc::GKZYXC> &&
                     std::is_same_v<ELayout, ctc::GNDHWK>)
        {
            return builder::GroupConvLayout3D::GNDHWC_GKZYXC_GNDHWK;
        }
        else if constexpr(std::is_same_v<ALayout, ctc::NDHWGC> &&
                          std::is_same_v<BLayout, ctc::GKZYXC> &&
                          std::is_same_v<ELayout, ctc::NDHWGK>)
        {
            return builder::GroupConvLayout3D::NDHWGC_GKZYXC_NDHWGK;
        }
        else if constexpr(std::is_same_v<ALayout, ctc::NGCDHW> &&
                          std::is_same_v<BLayout, ctc::GKZYXC> &&
                          std::is_same_v<ELayout, ctc::NGKDHW>)
        {
            return builder::GroupConvLayout3D::NGCDHW_GKZYXC_NGKDHW;
        }
        else if constexpr(std::is_same_v<ALayout, ctc::NGCDHW> &&
                          std::is_same_v<BLayout, ctc::GKCZYX> &&
                          std::is_same_v<ELayout, ctc::NGKDHW>)
        {
            return builder::GroupConvLayout3D::NGCDHW_GKCZYX_NGKDHW;
        }
    }
}

// Derive DataType from data type
template <typename Instance>
constexpr builder::DataType conv_data_type()
{
    using InstTraits = InstanceTraits<Instance>;
    using ADataType  = typename InstTraits::ADataType;

    if constexpr(std::is_same_v<ADataType, ck::half_t>)
    {
        return builder::DataType::FP16;
    }
    else if constexpr(std::is_same_v<ADataType, ck::bhalf_t>)
    {
        return builder::DataType::BF16;
    }
    else if constexpr(std::is_same_v<ADataType, float>)
    {
        return builder::DataType::FP32;
    }
    else if constexpr(std::is_same_v<ADataType, ck::f8_t>)
    {
        return builder::DataType::FP8;
    }
    else if constexpr(std::is_same_v<ADataType, int8_t>)
    {
        return builder::DataType::I8;
    }
    else if constexpr(std::is_same_v<ADataType, uint8_t>)
    {
        return builder::DataType::I8;
    }
    else
    {
        // Default fallback
        return builder::DataType::FP32;
    }
}

// Helper to extract values from Sequence types at compile time
template <typename Seq, ck::index_t Idx>
struct SequenceAt;

template <ck::index_t... Is, ck::index_t Idx>
struct SequenceAt<ck::Sequence<Is...>, Idx>
{
    static constexpr int value = ck::Sequence<Is...>::At(Idx);
};

// Primary template for ConvTraits
template <typename T>
struct ConvTraits;

// Specialization 1: Direct from Instance (Primary use case)
template <typename Instance>
    requires requires { typename InstanceTraits<Instance>; }
struct ConvTraits<Instance>
{
    using InstTraits = InstanceTraits<Instance>;

    // Signature information (derived from Instance template parameters)
    static constexpr int spatial_dim                  = InstTraits::kSpatialDim;
    static constexpr builder::ConvDirection direction = conv_direction<Instance>();
    static constexpr auto layout                      = conv_layout<Instance>();
    static constexpr builder::DataType data_type      = conv_data_type<Instance>();

    static constexpr auto gemm_specialization = InstTraits::kGemmSpecialization;
    static constexpr auto conv_specialization = InstTraits::kConvForwardSpecialization;

    // Algorithm information (extracted from Instance template parameters)
    static constexpr int thread_block_size  = InstTraits::kBlockSize;
    static constexpr DataTileInfo tile_dims = {
        .m = InstTraits::kMPerBlock, .n = InstTraits::kNPerBlock, .k = InstTraits::kKPerBlock};

    static constexpr InputTileTransferInfo a_tile_transfer = {
        .tile_dimensions = {.k0     = InstTraits::kKPerBlock / InstTraits::kAK1,
                            .m_or_n = InstTraits::kMPerBlock,
                            .k1     = InstTraits::kAK1},
        .transfer_params = {.k1                    = InstTraits::kAK1,
                            .thread_cluster_dims   = InstTraits::kAThreadClusterLengths,
                            .thread_cluster_order  = InstTraits::kAThreadClusterArrangeOrder,
                            .src_access_order      = InstTraits::kABlockTransferSrcAccessOrder,
                            .src_vector_dim        = InstTraits::kABlockTransferSrcVectorDim,
                            .src_scalar_per_vector = InstTraits::kABlockTransferSrcScalarPerVector,
                            .dst_scalar_per_vector_k1 =
                                InstTraits::kABlockTransferDstScalarPerVectorK1,
                            .lds_padding = static_cast<bool>(InstTraits::kABlockLdsExtraM)}};

    static constexpr InputTileTransferInfo b_tile_transfer = {
        .tile_dimensions = {.k0     = InstTraits::kKPerBlock / InstTraits::kBK1,
                            .m_or_n = InstTraits::kNPerBlock,
                            .k1     = InstTraits::kBK1},
        .transfer_params = {.k1                    = InstTraits::kBK1,
                            .thread_cluster_dims   = InstTraits::kBThreadClusterLengths,
                            .thread_cluster_order  = InstTraits::kBThreadClusterArrangeOrder,
                            .src_access_order      = InstTraits::kBBlockTransferSrcAccessOrder,
                            .src_vector_dim        = InstTraits::kBBlockTransferSrcVectorDim,
                            .src_scalar_per_vector = InstTraits::kBBlockTransferSrcScalarPerVector,
                            .dst_scalar_per_vector_k1 =
                                InstTraits::kBBlockTransferDstScalarPerVectorK1,
                            .lds_padding = static_cast<bool>(InstTraits::kBBlockLdsExtraN)}};

    static constexpr WarpGemmParams warp_gemm = {.gemm_m      = InstTraits::kMPerXDL,
                                                 .gemm_n      = InstTraits::kNPerXDL,
                                                 .num_m_gemms = InstTraits::kMXdlPerWave,
                                                 .num_n_gemms = InstTraits::kNXdlPerWave};

    static constexpr OutputTileTransferInfo c_tile_transfer = {
        .shuffle_params      = {.m_gemms_per_shuffle = InstTraits::kCShuffleMXdlPerWavePerShuffle,
                                .n_gemms_per_shuffle = InstTraits::kCShuffleNXdlPerWavePerShuffle},
        .thread_cluster_dims = {InstTraits::kCThreadClusterLengths[0],
                                InstTraits::kCThreadClusterLengths[1],
                                InstTraits::kCThreadClusterLengths[2],
                                InstTraits::kCThreadClusterLengths[3]},
        .scalar_per_vector   = InstTraits::kCBlockTransferScalarPerVector};

    // Pipeline version (only available for forward convolutions)
    // For backward data, this member doesn't exist in InstanceTraits
    template <typename T = InstTraits>
    static constexpr auto get_pipeline_version()
    {
        if constexpr(requires { T::kPipelineVersion; })
        {
            return T::kPipelineVersion;
        }
        else
        {
            // Return a default or indicate not available
            return ck::BlockGemmPipelineVersion::v1;
        }
    }

    static constexpr auto pipeline_version = get_pipeline_version();

    // Pipeline version (only available for forward convolutions)
    // For backward data, this member doesn't exist in InstanceTraits
    template <typename T = InstTraits>
    static constexpr auto get_pipeline_scheduler()
    {
        if constexpr(requires { T::kPipelineScheduler; })
        {
            return T::kPipelineScheduler;
        }
        else
        {
            // Return a default or indicate not available
            return ck::BlockGemmPipelineScheduler::Intrawave;
        }
    }

    static constexpr auto pipeline_scheduler = get_pipeline_scheduler();
};

// Specialization 2: From Builder (Backward compatibility)
template <builder::ConvSignatureDescriptor auto SIGNATURE,
          builder::ConvAlgorithmDescriptor auto ALGORITHM,
          builder::StringLiteral VERSION>
struct ConvTraits<builder::ConvBuilder<SIGNATURE, ALGORITHM, VERSION>>
{
    using Factory  = builder::ConvFactory<SIGNATURE, ALGORITHM, VERSION>;
    using Instance = typename Factory::Instance;

    // Delegate to Instance-based ConvTraits
    using InstanceConvTraits = ConvTraits<Instance>;

    // Forward all members from Instance-based traits
    static constexpr int spatial_dim                  = InstanceConvTraits::spatial_dim;
    static constexpr builder::ConvDirection direction = InstanceConvTraits::direction;
    static constexpr auto layout                      = InstanceConvTraits::layout;
    static constexpr builder::DataType data_type      = InstanceConvTraits::data_type;

    static constexpr int thread_block_size                  = InstanceConvTraits::thread_block_size;
    static constexpr DataTileInfo tile_dims                 = InstanceConvTraits::tile_dims;
    static constexpr InputTileTransferInfo a_tile_transfer  = InstanceConvTraits::a_tile_transfer;
    static constexpr InputTileTransferInfo b_tile_transfer  = InstanceConvTraits::b_tile_transfer;
    static constexpr WarpGemmParams warp_gemm               = InstanceConvTraits::warp_gemm;
    static constexpr OutputTileTransferInfo c_tile_transfer = InstanceConvTraits::c_tile_transfer;
    static constexpr auto pipeline_version                  = InstanceConvTraits::pipeline_version;
    static constexpr auto pipeline_scheduler = InstanceConvTraits::pipeline_scheduler;
};

} // namespace ck_tile::reflect::conv
