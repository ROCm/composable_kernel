// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <ck_tile/builder/conv_builder.hpp>
#include <ck_tile/builder/conv_signature_concepts.hpp>
#include <ck_tile/builder/reflect/instance_traits.hpp>
#include <ck_tile/builder/reflect/instance_traits_util.hpp>
#include <ck_tile/builder/types.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp>
#include <ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp>
#include <ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp>
#include <ck/utility/loop_scheduler.hpp>
#include <ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp>
#include <ck_tile/ops/gemm.hpp>
#include "ck_tile/ops/epilogue.hpp"
#include <ck_tile/ops/grouped_convolution.hpp>

namespace ck_tile::reflect::conv {

// Forward convolution layout concept - checks for A/B/E layout types
template <typename T>
concept HasFwdConvLayouts = requires {
    typename T::ALayout;
    typename T::BLayout;
    typename T::ELayout;
};

// GEMM specialization concept - checks for kGemmSpecialization member
template <typename T>
concept HasGemmSpec = requires {
    {
        T::kGemmSpecialization
    } -> std::convertible_to<ck::tensor_operation::device::GemmSpecialization>;
};

// Data types concept - checks for ADataType member
template <typename T>
concept HasDataTypes = requires { typename T::ADataType; };

// Elementwise operations concept - checks for A/B/CDE elementwise operation types
template <typename T>
concept HasElementwiseOps = requires {
    typename T::AElementwiseOperation;
    typename T::BElementwiseOperation;
    typename T::CDEElementwiseOperation;
};

// Tile parameters concept - checks for tile dimension and transfer members
template <typename T>
concept HasTileParams = requires {
    { T::kKPerBlock } -> std::convertible_to<int>;
    { T::kMPerBlock } -> std::convertible_to<int>;
    { T::kNPerBlock } -> std::convertible_to<int>;
    { T::kAK1 } -> std::convertible_to<int>;
    { T::kBK1 } -> std::convertible_to<int>;
    T::kCThreadClusterLengths;
};

// Comprehensive concept that checks if an instance has all XDL forward convolution traits
// This concept is used to constrain ConvTraits specialization that expect XDL forward convolutions
template <typename T>
concept IsXdlFwdConv = HasFwdConvLayouts<T> && HasGemmSpec<T> && HasDataTypes<T> &&
                       HasElementwiseOps<T> && HasTileParams<T>;

// Primary concept for checking if a type can be described
// Currently only forward convolutions are supported, but this can be extended
// in the future to include backward data and backward weight convolutions
template <typename T>
concept HasConvTraits = IsXdlFwdConv<InstanceTraits<T>>;

// Helper metafunctions to convert from ck enums to builder enums

/// @brief Converts a CK BlockGemmPipelineVersion enum to a builder PipelineVersion enum.
/// @tparam ck_ver The CK BlockGemmPipelineVersion enum value to convert.
/// @return The corresponding builder::PipelineVersion enum value (V1, V2, V3, V4, or V5).
/// @details This function maps CK's block GEMM pipeline version identifiers to the
/// builder framework's standardized pipeline version enum. The pipeline version
/// determines the strategy used for data movement and computation overlap in the
/// GEMM kernel's main loop.
template <ck::BlockGemmPipelineVersion ck_ver>
constexpr auto convert_pipeline_version()
{
    using enum ck::BlockGemmPipelineVersion;
    using enum builder::PipelineVersion;

    switch(ck_ver)
    {
    case v1: return V1;
    case v2: return V2;
    case v3: return V3;
    case v4: return V4;
    case v5: return V5;
    }
}

/// @brief Converts a CK PipelineVersion enum to a builder PipelineVersion enum.
/// @tparam ck_ver The CK PipelineVersion enum value to convert.
/// @return The corresponding builder::PipelineVersion enum value (V1, V2, V4, or WEIGHT_ONLY).
/// @details This function maps CK's general pipeline version identifiers to the
/// builder framework's standardized pipeline version enum. Note that this overload
/// handles a different set of pipeline versions compared to the BlockGemmPipelineVersion
/// variant, including support for specialized weight-only pipelines.
template <ck::PipelineVersion ck_ver>
constexpr auto convert_pipeline_version()
{
    using enum ck::PipelineVersion;
    using enum builder::PipelineVersion;

    switch(ck_ver)
    {
    case v1: return V1;
    case v2: return V2;
    case v4: return V4;
    case weight_only: return WEIGHT_ONLY;
    }
}

/// @brief Converts a CK BlockGemmPipelineScheduler enum to a builder PipelineScheduler enum.
/// @tparam ck_sched The CK BlockGemmPipelineScheduler enum value to convert.
/// @return The corresponding builder::PipelineScheduler enum value (INTRAWAVE or INTERWAVE).
/// @details This function maps CK's block GEMM pipeline scheduler identifiers to the
/// builder framework's standardized scheduler enum. The scheduler determines how work
/// is distributed and synchronized within and across wavefronts during pipeline execution.
/// INTRAWAVE scheduling operates within a single wavefront, while INTERWAVE coordinates
/// across multiple wavefronts.
template <ck::BlockGemmPipelineScheduler ck_sched>
constexpr auto convert_pipeline_scheduler()
{
    using enum ck::BlockGemmPipelineScheduler;
    using enum builder::PipelineScheduler;

    switch(ck_sched)
    {
    case Intrawave: return INTRAWAVE;
    case Interwave: return INTERWAVE;
    }
}

/// @brief Converts a CK LoopScheduler enum to a builder PipelineScheduler enum.
/// @tparam ck_sched The CK LoopScheduler enum value to convert.
/// @return The corresponding builder::PipelineScheduler enum value (DEFAULT or INTERWAVE).
/// @details This function maps CK's loop scheduler identifiers to the builder framework's
/// standardized pipeline scheduler enum. The loop scheduler controls how iterations of
/// the main computational loop are scheduled across threads. DEFAULT uses the standard
/// scheduling strategy, while INTERWAVE enables cross-wavefront coordination for improved
/// performance in certain scenarios.
template <ck::LoopScheduler ck_sched>
constexpr auto convert_pipeline_scheduler()
{
    using enum ck::LoopScheduler;
    using enum builder::PipelineScheduler;

    switch(ck_sched)
    {
    case Default: return DEFAULT;
    case Interwave: return INTERWAVE;
    }
}

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
    int gemm_m; ///< The M dimension of a single MFMA instruction (MPerXdl).
    int gemm_n; ///< The N dimension of a single MFMA instruction (NPerXdl).
    int m_iter; ///< The number of MFMA iterations along the M dimension of the output tile per
                ///< wavefront (MXdlPerWave).
    int n_iter; ///< The number of MFMA iterations along the N dimension of the output tile per
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
        return builder::ConvDirection::FORWARD;
    else if constexpr(requires { &InstTraits::kConvBwdDataSpecialization; })
        return builder::ConvDirection::BACKWARD_DATA;
    else if constexpr(requires { &InstTraits::kConvBwdWeightSpecialization; })
        return builder::ConvDirection::BACKWARD_WEIGHT;
    else
        return builder::ConvDirection::FORWARD; // Default fallback
}

/// @brief Derives the convolution-specific specialization from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return A `builder::ConvFwdSpecialization`, `builder::ConvBwdDataSpecialization`, or
/// `builder::ConvBwdWeightSpecialization` enum value.
template <typename Instance>
constexpr auto conv_spec()
{
    using InstTraits = InstanceTraits<Instance>;

    if constexpr(requires { InstTraits::kConvForwardSpecialization; })
    {
        using enum ck::tensor_operation::device::ConvolutionForwardSpecialization;
        using enum builder::ConvFwdSpecialization;

        switch(InstTraits::kConvForwardSpecialization)
        {
        case Default: return DEFAULT;
        case Filter1x1Pad0: return FILTER_1X1_PAD0;
        case Filter1x1Stride1Pad0: return FILTER_1X1_STRIDE1_PAD0;
        case Filter3x3: return FILTER_3x3;
        }
    }
    else if constexpr(requires { InstTraits::kConvBwdDataSpecialization; })
    {
        using enum ck::tensor_operation::device::ConvolutionBackwardDataSpecialization;
        using enum builder::ConvBwdDataSpecialization;

        switch(InstTraits::kConvBwdDataSpecialization)
        {
        case Default: return DEFAULT;
        case Filter1x1Stride1Pad0: return FILTER_1X1_STRIDE1_PAD0;
        }
    }
    else if constexpr(requires { InstTraits::kConvBwdWeightSpecialization; })
    {
        using enum ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization;
        using enum builder::ConvBwdWeightSpecialization;

        switch(InstTraits::kConvBwdWeightSpecialization)
        {
        case Default: return DEFAULT;
        case Filter1x1Stride1Pad0: return FILTER_1X1_STRIDE1_PAD0;
        case Filter1x1Pad0: return FILTER_1X1_PAD0;
        case OddC: return ODD_C;
        }
    }
}

// Helper variable template to check if CK layout enums match
template <typename A,
          typename B,
          typename E,
          typename ExpectedA,
          typename ExpectedB,
          typename ExpectedE>
inline constexpr bool layouts_are =
    std::is_same_v<A, ExpectedA> && std::is_same_v<B, ExpectedB> && std::is_same_v<E, ExpectedE>;

/// @brief Derives the grouped convolution layout from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return An std::array corresponding to the tensor layouts:
///             index 0 -> Input layout
///             index 1 -> Weight layout
///             index 2 -> Output layout
template <typename Instance>
constexpr auto conv_layout()
    requires HasFwdConvLayouts<InstanceTraits<Instance>>
{
    // Helper lambda to construct layout array
    auto layouts = [](auto... Ls) { return std::array<builder::TensorLayout, 3>{Ls...}; };

    using A       = typename InstanceTraits<Instance>::ALayout;
    using B       = typename InstanceTraits<Instance>::BLayout;
    using E       = typename InstanceTraits<Instance>::ELayout;
    namespace ctl = ck::tensor_layout::convolution;
    using enum builder::TensorLayout;

    switch(InstanceTraits<Instance>::kSpatialDim)
    {
    case 1:
        if constexpr(layouts_are<A, B, E, ctl::GNWC, ctl::GKXC, ctl::GNWK>)
            return layouts(GNWC, GKXC, GNWK);
        if constexpr(layouts_are<A, B, E, ctl::NWGC, ctl::GKXC, ctl::NWGK>)
            return layouts(NWGC, GKXC, NWGK);
        if constexpr(layouts_are<A, B, E, ctl::NGCW, ctl::GKXC, ctl::NGKW>)
            return layouts(NGCW, GKXC, NGKW);
        if constexpr(layouts_are<A, B, E, ctl::NGCW, ctl::GKCX, ctl::NGKW>)
            return layouts(NGCW, GKCX, NGKW);
        break;
    case 2:
        if constexpr(layouts_are<A, B, E, ctl::GNHWC, ctl::GKYXC, ctl::GNHWK>)
            return layouts(GNHWC, GKYXC, GNHWK);
        if constexpr(layouts_are<A, B, E, ctl::NHWGC, ctl::GKYXC, ctl::NHWGK>)
            return layouts(NHWGC, GKYXC, NHWGK);
        if constexpr(layouts_are<A, B, E, ctl::NGCHW, ctl::GKYXC, ctl::NGKHW>)
            return layouts(NGCHW, GKYXC, NGKHW);
        if constexpr(layouts_are<A, B, E, ctl::NGCHW, ctl::GKCYX, ctl::NGKHW>)
            return layouts(NGCHW, GKCYX, NGKHW);
        break;
    case 3:
        if constexpr(layouts_are<A, B, E, ctl::GNDHWC, ctl::GKZYXC, ctl::GNDHWK>)
            return layouts(GNDHWC, GKZYXC, GNDHWK);
        if constexpr(layouts_are<A, B, E, ctl::NDHWGC, ctl::GKZYXC, ctl::NDHWGK>)
            return layouts(NDHWGC, GKZYXC, NDHWGK);
        if constexpr(layouts_are<A, B, E, ctl::NGCDHW, ctl::GKZYXC, ctl::NGKDHW>)
            return layouts(NGCDHW, GKZYXC, NGKDHW);
        if constexpr(layouts_are<A, B, E, ctl::NGCDHW, ctl::GKCZYX, ctl::NGKDHW>)
            return layouts(NGCDHW, GKCZYX, NGKDHW);
        break;
    }
}

/// @brief Derives the data type from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return A `builder::DataType` enum value (e.g., FP16, BF16, FP32).
template <typename Instance>
constexpr builder::DataType conv_data_type()
    requires HasDataTypes<InstanceTraits<Instance>>
{
    using InstTraits = InstanceTraits<Instance>;
    using ADataType  = typename InstTraits::ADataType;
    using enum builder::DataType;

    if constexpr(std::is_same_v<ADataType, ck::half_t>)
        return FP16;
    else if constexpr(std::is_same_v<ADataType, ck::bhalf_t>)
        return BF16;
    else if constexpr(std::is_same_v<ADataType, float>)
        return FP32;
    else if constexpr(std::is_same_v<ADataType, ck::f8_t>)
        return FP8;
    else if constexpr(std::is_same_v<ADataType, int8_t>)
        return I8;
    else if constexpr(std::is_same_v<ADataType, uint8_t>)
        return U8;
    else
        return FP32; // Default fallback
}

/// @brief Derives the elementwise operation from op type.
/// @tparam ElementwiseOp Elementwise operation functor type.
/// @return A `builder::ElementwiseOperation` enum value corresponding to elementwise operation.
template <typename ElementwiseOp>
constexpr builder::ElementwiseOperation elementwise_op()
{
    using enum builder::ElementwiseOperation;
    constexpr std::string_view name = detail::elementwise_op_name<ElementwiseOp>();

    if constexpr(detail::case_insensitive_equal(name, "BiasBnormClamp"))
        return BIAS_BNORM_CLAMP;
    if constexpr(detail::case_insensitive_equal(name, "Clamp"))
        return CLAMP;
    if constexpr(detail::case_insensitive_equal(name, "Scale"))
        return SCALE;
    if constexpr(detail::case_insensitive_equal(name, "PassThrough"))
        return PASS_THROUGH;
    if constexpr(detail::case_insensitive_equal(name, "ScaleAddScaleAddRelu"))
        return SCALEADD_SCALEADD_RELU;
}

/// @brief Derives a gemm padding from a kernel instance type.
/// @tparam Instance - A Device Kernel object type.
/// @return A `builder::GemmPadding` enum value corresponding to kernel padding.
template <typename Instance>
constexpr builder::GemmPadding gemm_spec()
    requires HasGemmSpec<InstanceTraits<Instance>>
{
    using InstTraits = InstanceTraits<Instance>;
    using enum builder::GemmPadding;
    using enum ck::tensor_operation::device::GemmSpecialization;

    constexpr auto gemm_spec = InstTraits::kGemmSpecialization;

    switch(gemm_spec)
    {
    case Default: return DEFAULT;
    case MPadding: return M_PADDING;
    case NPadding: return N_PADDING;
    case KPadding: return K_PADDING;
    case MNPadding: return MN_PADDING;
    case MKPadding: return MK_PADDING;
    case NKPadding: return NK_PADDING;
    case MNKPadding: return MNK_PADDING;
    case OPadding: return O_PADDING;
    case MOPadding: return MO_PADDING;
    case NOPadding: return NO_PADDING;
    case KOPadding: return KO_PADDING;
    case MNOPadding: return MNO_PADDING;
    case MKOPadding: return MKO_PADDING;
    case NKOPadding: return NKO_PADDING;
    case MNKOPadding: return MNKO_PADDING;
    }
}

/// @brief Primary template for extracting convolution traits.
/// @details This struct is the main entry point for reflecting on a convolution
/// kernel's properties. It is specialized to handle different kinds of input types.
template <typename T>
struct ConvTraits;

/// @brief Specialization of `ConvTraits` for a direct device kernel `Instance`.
/// @details This is the primary specialization used to extract a comprehensive
/// set of traits directly from a fully-formed device kernel `Instance` type.
/// It uses `InstanceTraits` to access the kernel's template parameters.
template <HasInstanceTraits Instance>
    requires IsXdlFwdConv<InstanceTraits<Instance>>
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

    static constexpr builder::ElementwiseOperation input_element_op =
        elementwise_op<typename InstTraits::AElementwiseOperation>();
    static constexpr builder::ElementwiseOperation weight_element_op =
        elementwise_op<typename InstTraits::BElementwiseOperation>();
    static constexpr builder::ElementwiseOperation output_element_op =
        elementwise_op<typename InstTraits::CDEElementwiseOperation>();

    /// @brief The GEMM specialization used by the kernel - padding
    static constexpr auto gemm_padding = gemm_spec<Instance>();
    /// @brief The convolution-specific specialization (e.g., Default, 1x1).
    static constexpr auto conv_specialization = conv_spec<Instance>();

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
    static constexpr WarpGemmParams warp_gemm = {.gemm_m = InstTraits::kMPerXDL,
                                                 .gemm_n = InstTraits::kNPerXDL,
                                                 .m_iter = InstTraits::kMXdlPerWave,
                                                 .n_iter = InstTraits::kNXdlPerWave};

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
            return convert_pipeline_version<T::kPipelineVersion>();
        }
        else
        {
            // Return a default or indicate not available
            return builder::PipelineVersion::V1;
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
            return convert_pipeline_scheduler<T::kPipelineScheduler>();
        }
        else if constexpr(requires { T::kLoopScheduler; })
        {
            return convert_pipeline_scheduler<T::kLoopScheduler>();
        }
        else
        {
            // Return a default or indicate not available
            return builder::PipelineScheduler::DEFAULT;
        }
    }

    /// @brief The pipeline scheduler used by the kernel.
    static constexpr auto pipeline_scheduler = get_pipeline_scheduler();
};

/// @brief Specialization of `ConvTraits` for a `ConvBuilder` type.
/// @details This specialization provides backward compatibility for reflecting
/// on kernels defined via the `ConvBuilder` interface. It works by first
/// creating the `Instance` via the builder, and then delegating
/// all trait extraction to the `ConvTraits<Instance>` specialization.
template <builder::ConvSignatureDescriptor auto SIGNATURE,
          builder::ConvAlgorithmDescriptor auto ALGORITHM,
          builder::StringLiteral VERSION>
struct ConvTraits<builder::ConvBuilder<SIGNATURE, ALGORITHM, VERSION>>
{
    using Instance = typename builder::ConvBuilder<SIGNATURE, ALGORITHM, VERSION>::Instance;

    // Delegate to Instance-based ConvTraits
    using InstanceConvTraits = ConvTraits<Instance>;

    // Forward all members from Instance-based traits
    static constexpr int spatial_dim                  = InstanceConvTraits::spatial_dim;
    static constexpr builder::ConvDirection direction = InstanceConvTraits::direction;
    static constexpr auto layout                      = InstanceConvTraits::layout;
    static constexpr builder::DataType data_type      = InstanceConvTraits::data_type;

    static constexpr builder::ElementwiseOperation input_element_op =
        InstanceConvTraits::input_element_op;
    static constexpr builder::ElementwiseOperation weight_element_op =
        InstanceConvTraits::weight_element_op;
    static constexpr builder::ElementwiseOperation output_element_op =
        InstanceConvTraits::output_element_op;

    static constexpr auto gemm_padding        = InstanceConvTraits::gemm_padding;
    static constexpr auto conv_specialization = InstanceConvTraits::conv_specialization;

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
