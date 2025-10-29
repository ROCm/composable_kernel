// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/conv_factory.hpp>
#include <ck_tile/builder/conv_signature_concepts.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck_tile/builder/types.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>

namespace ck_tile::reflect::conv {

/// @brief Helper structures for organizing trait data with domain-specific naming

/// @brief Data tile dimensions processed by a workgroup.
/// @details This struct defines the M, N, and K dimensions of the data tile
/// that a single workgroup (thread block) is responsible for processing in the
/// underlying GEMM computation.
struct DataTileInfo
{
    int m; ///< M dimension of the tile processed by the workgroup (MPerBlock).
    int n; ///< N dimension of the tile processed by the workgroup (NPerBlock).
    int k; ///< K dimension of the tile processed by the workgroup (KPerBlock).
};

/// @brief Dimensions for an input data tile transfer.
/// @details Defines the shape of the input tile (A or B matrix) as it is
/// transferred from global memory to LDS. The tile is conceptually divided
/// into k0 and k1 dimensions.
struct InputTileTransferDimensions
{
    int k0;     ///< The outer dimension of K, where K = k0 * k1.
    int m_or_n; ///< The M dimension for the A matrix transfer, or the N dimension for the B matrix.
    int k1; ///< The inner dimension of K, often corresponding to the vector load size from global
            ///< memory.
};

/// @brief Parameters governing the transfer of an input tile.
/// @details This struct holds configuration details for how an input tile is
/// loaded from global memory into LDS, including thread clustering, memory
/// access patterns, and vectorization settings.
struct InputTileTransferParams
{
    int k1; ///< The inner K dimension size, often matching the vectorization width.
    std::array<int, 3>
        thread_cluster_dims; ///< Spatial thread distribution over the input data tile; defines how
                             ///< many threads are arranged on each axis.
    std::array<int, 3> thread_cluster_order; ///< The order of thread spatial distribution over the
                                             ///< input tensor dimensions.
    std::array<int, 3> src_access_order; ///< The order of accessing input tensor axes (e.g., which
                                         ///< dimension to read first).
    int src_vector_dim; ///< The index of the axis on which vectorized memory access is performed
                        ///< (the contiguous dimension).
    int src_scalar_per_vector;    ///< The size of the vector access instruction; the number of
                                  ///< elements accessed per thread per instruction.
    int dst_scalar_per_vector_k1; ///< The size of the vectorized store into LDS memory along the K1
                                  ///< dimension.
    bool lds_padding; ///< Flag indicating if padding is used for the LDS tensor to prevent bank
                      ///< conflicts.
};

/// @brief Complete information for an input tile transfer.
/// @details Combines the dimensional information and transfer parameters for
/// a full description of an input tile's journey from global memory to LDS.
struct InputTileTransferInfo
{
    InputTileTransferDimensions tile_dimensions; ///< The shape and layout of the tile.
    InputTileTransferParams transfer_params; ///< The parameters for the memory transfer operation.
};

/// @brief Parameters for the warp-level GEMM computation.
/// @details Defines the configuration of the GEMM operation performed by each
/// warp using hardware MFMA (Matrix Fused Multiply-Add) instructions.
struct WarpGemmParams
{
    int gemm_m;      ///< The M dimension of a single MFMA instruction (MPerXdl).
    int gemm_n;      ///< The N dimension of a single MFMA instruction (NPerXdl).
    int num_m_gemms; ///< The number of MFMA iterations along the M dimension of the output tile per
                     ///< wavefront (MXdlPerWave).
    int num_n_gemms; ///< The number of MFMA iterations along the N dimension of the output tile per
                     ///< wavefront (NXdlPerWave).
};

/// @brief Parameters for shuffling data between warps (CShuffle optimization).
/// @details Configures how many MFMA instruction results are processed per
/// wave in each iteration of the CShuffle routine.
struct WarpShuffleParams
{
    int m_gemms_per_shuffle; ///< Number of MFMA results along the M dimension to process per wave
                             ///< per shuffle iteration.
    int n_gemms_per_shuffle; ///< Number of MFMA results along the N dimension to process per wave
                             ///< per shuffle iteration.
};

/// @brief Information for the output tile transfer (CShuffle).
/// @details Describes how the final computed tile (C matrix) is written out from
/// LDS to global memory, including shuffling, thread clustering, and vectorization.
struct OutputTileTransferInfo
{
    WarpShuffleParams shuffle_params; ///< Configuration for cross-warp data shuffling.
    // m_block, m_wave_per_xdl, n_block, n_wave_per_xdl
    std::array<int, 4> thread_cluster_dims; ///< The spatial thread distribution used for storing
                                            ///< data into the output tensor.
    int scalar_per_vector; ///< The size of the vectorized memory access when storing data to the
                           ///< output tensor.
};

// Helper metafunctions to derive signature information from Instance types

/// @brief Derives the convolution direction from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return A `builder::ConvDirection` enum value (FORWARD, BACKWARD_DATA, or BACKWARD_WEIGHT).
template <typename Instance>
constexpr builder::ConvDirection conv_direction()
{
    using InstTraits = InstanceTraits<Instance>;

    if constexpr(requires { &InstTraits::kConvForwardSpecialization; })
    {
        return builder::ConvDirection::FORWARD;
    }
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

/// @brief Derives the grouped convolution layout from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return A `builder::GroupConvLayout{1D|2D|3D}` enum value corresponding to the tensor layouts.
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

/// @brief Derives the data type from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return A `builder::DataType` enum value (e.g., FP16, BF16, FP32).
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
        return builder::DataType::U8;
    }
    else
    {
        // Default fallback
        return builder::DataType::FP32;
    }
}

/// @brief Helper to extract a value from a `ck::Sequence` type at a specific index.
/// @tparam Seq The `ck::Sequence` type.
/// @tparam Idx The index of the value to extract.
template <typename Seq, ck::index_t Idx>
struct SequenceAt;

/// @brief Specialization of `SequenceAt` for `ck::Sequence`.
template <ck::index_t... Is, ck::index_t Idx>
struct SequenceAt<ck::Sequence<Is...>, Idx>
{
    /// The integer value at the specified index within the sequence.
    static constexpr int value = ck::Sequence<Is...>::At(Idx);
};

/// @brief Primary template for extracting convolution traits.
/// @details This struct is the main entry point for reflecting on a convolution
/// kernel's properties. It is specialized to handle different kinds of input types.
template <typename T>
struct ConvTraits;

/// @brief Specialization of `ConvTraits` for a direct device kernel `Instance`.
/// @details This is the primary specialization used to extract a comprehensive
/// set of traits directly from a fully-formed device kernel `Instance` type.
/// It uses `InstanceTraits` to access the kernel's template parameters.
template <typename Instance>
    requires requires { typename InstanceTraits<Instance>; }
struct ConvTraits<Instance>
{
    using InstTraits = InstanceTraits<Instance>;

    // --- Signature Information ---
    /// @brief The number of spatial dimensions in the convolution (1, 2, or 3).
    static constexpr int spatial_dim = InstTraits::kSpatialDim;
    /// @brief The direction of the convolution (Forward, Backward Data, or Backward Weight).
    static constexpr builder::ConvDirection direction = conv_direction<Instance>();
    /// @brief The memory layout of the convolution tensors (e.g., GNHWC_GKYXC_GNHWK).
    static constexpr auto layout = conv_layout<Instance>();
    /// @brief The primary data type used in the computation (e.g., FP16, FP32).
    static constexpr builder::DataType data_type = conv_data_type<Instance>();

    /// @brief The GEMM specialization used by the kernel (e.g., Tiling, Partition).
    static constexpr auto gemm_specialization = InstTraits::kGemmSpecialization;
    /// @brief The convolution-specific specialization (e.g., Default, 1x1).
    static constexpr auto conv_specialization = InstTraits::kConvForwardSpecialization;

    // --- Algorithm Information ---
    /// @brief The total number of threads in a thread block (workgroup).
    static constexpr int thread_block_size = InstTraits::kBlockSize;
    /// @brief The dimensions of the data tile processed by the thread block.
    static constexpr DataTileInfo tile_dims = {
        .m = InstTraits::kMPerBlock, .n = InstTraits::kNPerBlock, .k = InstTraits::kKPerBlock};

    /// @brief Configuration for the A-matrix (input) tile transfer.
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

    /// @brief Configuration for the B-matrix (weights) tile transfer.
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

    /// @brief Parameters for the warp-level GEMM computation.
    static constexpr WarpGemmParams warp_gemm = {.gemm_m      = InstTraits::kMPerXDL,
                                                 .gemm_n      = InstTraits::kNPerXDL,
                                                 .num_m_gemms = InstTraits::kMXdlPerWave,
                                                 .num_n_gemms = InstTraits::kNXdlPerWave};

    /// @brief Configuration for the C-matrix (output) tile transfer.
    static constexpr OutputTileTransferInfo c_tile_transfer = {
        .shuffle_params      = {.m_gemms_per_shuffle = InstTraits::kCShuffleMXdlPerWavePerShuffle,
                                .n_gemms_per_shuffle = InstTraits::kCShuffleNXdlPerWavePerShuffle},
        .thread_cluster_dims = {InstTraits::kCThreadClusterLengths[0],
                                InstTraits::kCThreadClusterLengths[1],
                                InstTraits::kCThreadClusterLengths[2],
                                InstTraits::kCThreadClusterLengths[3]},
        .scalar_per_vector   = InstTraits::kCBlockTransferScalarPerVector};

    /// @brief Helper to safely get the pipeline version.
    /// @details This is only available for some convolutions (e.g., forward).
    /// If not present in `InstanceTraits`, it returns a default value.
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

    /// @brief The block GEMM pipeline version used by the kernel.
    static constexpr auto pipeline_version = get_pipeline_version();

    /// @brief Helper to safely get the pipeline scheduler.
    /// @details This is only available for some convolutions. If not present
    /// in `InstanceTraits`, it returns a default value.
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

    /// @brief The pipeline scheduler used by the kernel.
    static constexpr auto pipeline_scheduler = get_pipeline_scheduler();
};

/// @brief Specialization of `ConvTraits` for a `ConvBuilder` type.
/// @details This specialization provides backward compatibility for reflecting
/// on kernels defined via the `ConvBuilder` interface. It works by first
/// creating the `Instance` via the builder's factory, and then delegating
/// all trait extraction to the `ConvTraits<Instance>` specialization.
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
