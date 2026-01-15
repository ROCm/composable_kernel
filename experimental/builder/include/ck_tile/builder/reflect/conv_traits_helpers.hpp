// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <concepts>
#include <string_view>
#include <type_traits>

#include "ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp"
#include "ck/tensor_operation/gpu/device/convolution_backward_weight_specialization.hpp"
#include "ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/pipeline_enum.hpp"
#include "ck/utility/scheduler_enum.hpp"
#include "ck_tile/builder/conv_signature_concepts.hpp"
#include "ck_tile/builder/reflect/conv_types.hpp"
#include "ck_tile/builder/reflect/instance_traits.hpp"
#include "ck_tile/builder/reflect/instance_traits_util.hpp"
#include "ck_tile/builder/types.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"

/// @file conv_traits_helpers.hpp
/// @brief Helper utilities for extracting convolution traits from kernel instances
///
/// This file provides compile-time reflection utilities to extract configuration
/// information from CK convolution kernel instances and convert them to the builder
/// framework's standardized representation.
///
/// ## Organization
///
/// The file is organized into the following sections:
///
/// 1. **Enum Conversions**: Functions to convert CK enums to builder enums
///    - Pipeline version conversions (BlockGemmPipelineVersion, PipelineVersion)
///    - Pipeline scheduler conversions (BlockGemmPipelineScheduler, LoopScheduler)
///
/// 2. **Signature Derivation**: Functions to extract signature information from instances
///    - Convolution direction (conv_direction)
///    - Convolution specialization (conv_spec)
///    - Tensor layouts (conv_layout)
///    - Data types (conv_data_type)
///    - Elementwise operations (elementwise_op)
///    - GEMM padding (gemm_spec)
///
/// 3. **Pipeline Configuration Helpers**: Safe extraction of pipeline parameters
///    - Pipeline version extraction (get_pipeline_version)
///    - Pipeline scheduler extraction (get_pipeline_scheduler)
///
/// ## Error Handling Strategy
///
/// This file uses a specific error handling pattern for compile-time errors:
/// - **consteval functions with throw**: Used for error reporting to ensure SFINAE doesn't
///   silently ignore errors. The thrown string becomes part of the compiler error message,
///   providing clear context to developers.
/// - **DO NOT replace with static_assert**: static_assert is silently ignored during SFINAE,
///   which would hide errors instead of reporting them clearly.
///
/// @example
/// ```cpp
/// using Instance =
/// ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<...>;
///
/// // Extract convolution direction
/// constexpr auto dir = conv_direction<Instance>();
///
/// // Extract data type
/// constexpr auto dtype = conv_data_type<Instance>();
///
/// // Extract layout configuration
/// constexpr auto layouts = conv_layout<Instance>();
/// ```

namespace ck_tile::reflect::conv {

// ============================================================================
// SECTION 1: ENUM CONVERSIONS
// ============================================================================

/// @brief Converts a CK BlockGemmPipelineVersion enum to a builder PipelineVersion enum.
/// @tparam ck_ver The CK BlockGemmPipelineVersion enum value to convert.
/// @return The corresponding builder::PipelineVersion enum value.
/// @details This function maps CK's block GEMM pipeline version identifiers to the
/// builder framework's standardized pipeline version enum. The pipeline version
/// determines the strategy used for data movement and computation overlap in the
/// GEMM kernel's main loop.
///
/// Supported mappings:
/// - v1 -> V1
/// - v2 -> V2
/// - v3 -> V3
/// - v4 -> V4
/// - v5 -> V5
template <ck::BlockGemmPipelineVersion ck_ver>
constexpr builder::PipelineVersion convert_pipeline_version()
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
/// @return The corresponding builder::PipelineVersion enum value.
/// @details This function maps CK's general pipeline version identifiers to the
/// builder framework's standardized pipeline version enum. Note that this overload
/// handles a different set of pipeline versions compared to the BlockGemmPipelineVersion
/// variant, including support for specialized weight-only pipelines.
///
/// Supported mappings:
/// - v1 -> V1
/// - v2 -> V2
/// - v4 -> V4
/// - weight_only -> WEIGHT_ONLY
template <ck::PipelineVersion ck_ver>
constexpr builder::PipelineVersion convert_pipeline_version()
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
/// @return The corresponding builder::PipelineScheduler enum value.
/// @details This function maps CK's block GEMM pipeline scheduler identifiers to the
/// builder framework's standardized scheduler enum. The scheduler determines how work
/// is distributed and synchronized within and across wavefronts during pipeline execution.
///
/// Supported mappings:
/// - Intrawave -> INTRAWAVE: Scheduling within a single wavefront
/// - Interwave -> INTERWAVE: Coordination across multiple wavefronts
template <ck::BlockGemmPipelineScheduler ck_sched>
constexpr builder::PipelineScheduler convert_pipeline_scheduler()
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
/// @return The corresponding builder::PipelineScheduler enum value.
/// @details This function maps CK's loop scheduler identifiers to the builder framework's
/// standardized pipeline scheduler enum. The loop scheduler controls how iterations of
/// the main computational loop are scheduled across threads.
///
/// Supported mappings:
/// - Default -> DEFAULT: Standard scheduling strategy
/// - Interwave -> INTERWAVE: Cross-wavefront coordination for improved performance
template <ck::LoopScheduler ck_sched>
constexpr builder::PipelineScheduler convert_pipeline_scheduler()
{
    using enum ck::LoopScheduler;
    using enum builder::PipelineScheduler;

    switch(ck_sched)
    {
    case Default: return DEFAULT;
    case Interwave: return INTERWAVE;
    }
}

// ============================================================================
// SECTION 2: SIGNATURE DERIVATION FUNCTIONS
// ============================================================================

// ----------------------------------------------------------------------------
// Convolution Direction
// ----------------------------------------------------------------------------

/// @brief Helper function to report unsupported convolution direction with a clear error message.
/// @details This consteval function uses throw (not static_assert) to ensure the error is not
/// silently ignored during SFINAE. The thrown string becomes part of the compiler error message.
template <typename Instance>
[[noreturn]] consteval void report_unsupported_conv_direction_error()
{
    throw "Unsupported convolution direction detected!\n"
          "The kernel instance does not have a recognized convolution specialization.\n"
          "Expected one of: kConvForwardSpecialization, kConvBwdDataSpecialization, or "
          "kConvBwdWeightSpecialization.\n"
          "Please verify that your kernel instance is properly configured.";
}

/// @brief Derives the convolution direction from a device kernel Instance type.
/// @tparam Instance The device kernel instance type.
/// @return A builder::ConvDirection enum value (FORWARD, BACKWARD_DATA, or BACKWARD_WEIGHT).
/// @details This function inspects the Instance's InstanceTraits to determine which
/// convolution specialization field is present, and returns the corresponding direction.
///
/// The function checks for the presence of:
/// - kConvForwardSpecialization -> FORWARD
/// - kConvBwdDataSpecialization -> BACKWARD_DATA
/// - kConvBwdWeightSpecialization -> BACKWARD_WEIGHT
///
/// @note Compilation will fail with a clear error message if the instance does not
/// have a recognized convolution specialization field.
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
    {
        report_unsupported_conv_direction_error<Instance>();
        return builder::ConvDirection::FORWARD; // Unreachable
    }
}

// ----------------------------------------------------------------------------
// Convolution Specialization
// ----------------------------------------------------------------------------

/// @brief Helper function to report unsupported convolution specialization with a clear error
/// message.
/// @details This consteval function uses throw (not static_assert) to ensure the error is not
/// silently ignored during SFINAE. The thrown string becomes part of the compiler error message.
template <typename Instance>
[[noreturn]] consteval void report_unsupported_conv_spec_error()
{
    throw "Unsupported convolution specialization detected!\n"
          "The kernel instance does not have a recognized convolution specialization field.\n"
          "Expected one of: kConvForwardSpecialization, kConvBwdDataSpecialization, or "
          "kConvBwdWeightSpecialization.\n"
          "Please verify that your kernel instance is properly configured.";
}

/// @brief Derives the convolution-specific specialization from a device kernel Instance type.
/// @tparam Instance The device kernel instance type.
/// @return A builder::ConvSpecialization enum value.
/// @details This function extracts the specialization enum from the Instance's InstanceTraits
/// and converts it to the corresponding builder framework enum.
///
/// For forward convolutions, supported specializations include:
/// - Default, Filter1x1Pad0, Filter1x1Stride1Pad0, Filter3x3, OddC
///
/// For backward data convolutions:
/// - Default, Filter1x1Stride1Pad0
///
/// For backward weight convolutions:
/// - Default, Filter1x1Stride1Pad0, Filter1x1Pad0, OddC
template <typename Instance>
constexpr builder::ConvSpecialization conv_spec()
{
    using InstTraits = InstanceTraits<Instance>;

    if constexpr(requires { InstTraits::kConvForwardSpecialization; })
    {
        using enum ck::tensor_operation::device::ConvolutionForwardSpecialization;
        using enum builder::ConvSpecialization;

        switch(InstTraits::kConvForwardSpecialization)
        {
        case Default: return DEFAULT;
        case Filter1x1Pad0: return FILTER_1X1_PAD0;
        case Filter1x1Stride1Pad0: return FILTER_1X1_STRIDE1_PAD0;
        case Filter3x3: return FILTER_3x3;
        case OddC: return ODD_C;
        }
    }
    else if constexpr(requires { InstTraits::kConvBwdDataSpecialization; })
    {
        using enum ck::tensor_operation::device::ConvolutionBackwardDataSpecialization;
        using enum builder::ConvSpecialization;

        switch(InstTraits::kConvBwdDataSpecialization)
        {
        case Default: return DEFAULT;
        case Filter1x1Stride1Pad0: return FILTER_1X1_STRIDE1_PAD0;
        }
    }
    else if constexpr(requires { InstTraits::kConvBwdWeightSpecialization; })
    {
        using enum ck::tensor_operation::device::ConvolutionBackwardWeightSpecialization;
        using enum builder::ConvSpecialization;

        switch(InstTraits::kConvBwdWeightSpecialization)
        {
        case Default: return DEFAULT;
        case Filter1x1Stride1Pad0: return FILTER_1X1_STRIDE1_PAD0;
        case Filter1x1Pad0: return FILTER_1X1_PAD0;
        case OddC: return ODD_C;
        }
    }
    else
    {
        report_unsupported_conv_spec_error<Instance>();
        return builder::ConvSpecialization::DEFAULT; // Unreachable
    }
}

// ----------------------------------------------------------------------------
// Tensor Layouts
// ----------------------------------------------------------------------------

/// @brief Helper function to report unsupported layout combinations with a clear error message.
/// @details This consteval function uses throw (not static_assert) to ensure the error is not
/// silently ignored during SFINAE. The thrown string becomes part of the compiler error message.
template <typename A, typename B, typename E, int SpatialDim>
[[noreturn]] consteval void report_unsupported_layout_error()
{
    throw "Unsupported convolution layout combination detected!\n"
          "The combination of ALayout, BLayout, and ELayout template parameters\n"
          "is not recognized for the given spatial dimension.\n"
          "Please verify that your convolution instance uses a supported layout configuration.\n"
          "Check the conv_layout() function for the list of supported layout combinations.";
}

/// @brief Derives the grouped convolution layout from a device kernel Instance type.
/// @tparam Instance The device kernel instance type.
/// @return An std::array<builder::TensorLayout, 3> containing the layouts for:
///         - [0] Input tensor layout
///         - [1] Weight tensor layout
///         - [2] Output tensor layout
/// @details This function examines the Instance's ALayout, BLayout, and ELayout types
/// along with the spatial dimension to determine the appropriate layout configuration.
///
/// Supported layout combinations vary by spatial dimension (1D, 2D, 3D convolutions).
/// Common patterns include GNHWC (grouped, batch, spatial, channels) and variants.
///
/// @note Compilation will fail with a clear error message if the layout combination
/// is not supported for the given spatial dimension.
///
/// TODO: If we don't check for supported layouts, this function can be simplified.
template <typename Instance>
constexpr std::array<builder::TensorLayout, 3> conv_layout()
{
    using InstTraits = InstanceTraits<Instance>;
    using A          = typename InstTraits::ALayout;
    using B          = typename InstTraits::BLayout;
    using E          = typename InstTraits::ELayout;
    namespace ctl    = ck::tensor_layout::convolution;
    using enum builder::TensorLayout;

    // Helper to check if layouts match expected types
    constexpr auto layouts_match = []<typename ExpA, typename ExpB, typename ExpE>() {
        return std::is_same_v<A, ExpA> && std::is_same_v<B, ExpB> && std::is_same_v<E, ExpE>;
    };

    // Helper to construct layout array
    constexpr auto make_layouts = [](auto in, auto weight, auto out) {
        return std::array<builder::TensorLayout, 3>{in, weight, out};
    };

    constexpr int spatial_dim = InstTraits::kSpatialDim;

    if constexpr(spatial_dim == 1)
    {
        if constexpr(layouts_match.template operator()<ctl::GNWC, ctl::GKXC, ctl::GNWK>())
            return make_layouts(GNWC, GKXC, GNWK);
        else if constexpr(layouts_match
                              .template operator()<ctl::G_NW_C, ctl::G_K_X_C, ctl::G_NW_K>())
            return make_layouts(GNWC, GKXC, GNWK);
        else if constexpr(layouts_match.template operator()<ctl::NWGC, ctl::GKXC, ctl::NWGK>())
            return make_layouts(NWGC, GKXC, NWGK);
        else if constexpr(layouts_match.template operator()<ctl::NGCW, ctl::GKXC, ctl::NGKW>())
            return make_layouts(NGCW, GKXC, NGKW);
        else if constexpr(layouts_match.template operator()<ctl::NGCW, ctl::GKCX, ctl::NGKW>())
            return make_layouts(NGCW, GKCX, NGKW);
        else
        {
            report_unsupported_layout_error<A, B, E, spatial_dim>();
            return make_layouts(GNWC, GKXC, GNWK); // Unreachable
        }
    }
    else if constexpr(spatial_dim == 2)
    {
        if constexpr(layouts_match.template operator()<ctl::GNHWC, ctl::GKYXC, ctl::GNHWK>())
            return make_layouts(GNHWC, GKYXC, GNHWK);
        else if constexpr(layouts_match
                              .template operator()<ctl::G_NHW_C, ctl::G_K_YX_C, ctl::G_NHW_K>())
            return make_layouts(GNHWC, GKYXC, GNHWK);
        else if constexpr(layouts_match.template operator()<ctl::NHWGC, ctl::GKYXC, ctl::NHWGK>())
            return make_layouts(NHWGC, GKYXC, NHWGK);
        else if constexpr(layouts_match.template operator()<ctl::NHWGC, ctl::KYXGC, ctl::NHWGK>())
            return make_layouts(NHWGC, GKYXC, NHWGK);
        else if constexpr(layouts_match.template operator()<ctl::NGCHW, ctl::GKYXC, ctl::NGKHW>())
            return make_layouts(NGCHW, GKYXC, NGKHW);
        else if constexpr(layouts_match.template operator()<ctl::NGCHW, ctl::GKCYX, ctl::NGKHW>())
            return make_layouts(NGCHW, GKCYX, NGKHW);
        else
        {
            report_unsupported_layout_error<A, B, E, spatial_dim>();
            return make_layouts(GNHWC, GKYXC, GNHWK); // Unreachable
        }
    }
    else if constexpr(spatial_dim == 3)
    {
        if constexpr(layouts_match.template operator()<ctl::GNDHWC, ctl::GKZYXC, ctl::GNDHWK>())
            return make_layouts(GNDHWC, GKZYXC, GNDHWK);
        else if constexpr(layouts_match
                              .template operator()<ctl::G_NDHW_C, ctl::G_K_ZYX_C, ctl::G_NDHW_K>())
            return make_layouts(GNDHWC, GKZYXC, GNDHWK);
        else if constexpr(layouts_match
                              .template operator()<ctl::NDHWGC, ctl::GKZYXC, ctl::NDHWGK>())
            return make_layouts(NDHWGC, GKZYXC, NDHWGK);
        else if constexpr(layouts_match
                              .template operator()<ctl::NGCDHW, ctl::GKZYXC, ctl::NGKDHW>())
            return make_layouts(NGCDHW, GKZYXC, NGKDHW);
        else if constexpr(layouts_match
                              .template operator()<ctl::NGCDHW, ctl::GKCZYX, ctl::NGKDHW>())
            return make_layouts(NGCDHW, GKCZYX, NGKDHW);
        else
        {
            report_unsupported_layout_error<A, B, E, spatial_dim>();
            return make_layouts(GNDHWC, GKZYXC, GNDHWK); // Unreachable
        }
    }
    else
    {
        report_unsupported_layout_error<A, B, E, spatial_dim>();
        return make_layouts(GNHWC, GKYXC, GNHWK); // Unreachable
    }
}

// ----------------------------------------------------------------------------
// Data Types
// ----------------------------------------------------------------------------

/// @brief Helper function to report unsupported data type with a clear error message.
/// @details This consteval function uses throw (not static_assert) to ensure the error is not
/// silently ignored during SFINAE. The thrown string becomes part of the compiler error message.
template <typename ADataType>
[[noreturn]] consteval void report_unsupported_data_type_error()
{
    throw "Unsupported data type detected!\n"
          "The ADataType is not recognized.\n"
          "Supported types are: ck::half_t (FP16), ck::Tuple<ck::half_t, ck::half_t> (FP16_FP16), "
          "ck::bhalf_t (BF16), ck::Tuple<ck::bhalf_t, ck::bhalf_t> (BF16_BF16), float (FP32), "
          "ck::Tuple<float, float> (FP32_FP32), double (FP64), ck::f8_t (FP8), ck::bf8_fnuz_t "
          "(BF8), "
          "int8_t (I8), ck::Tuple<int8_t, int8_t> (I8_I8), uint8_t (U8).\n"
          "Please verify that your kernel instance uses a supported data type.";
}

/// @brief Derives the data type from a device kernel Instance type.
/// @tparam Instance The device kernel instance type.
/// @return A builder::DataType enum value representing the input data type.
/// @details This function examines the Instance's ADataType to determine the data type
/// used for the input tensor. The function supports various floating-point and integer
/// types, including tuple types for mixed-precision operations.
///
/// Supported data types include:
/// - FP16 (ck::half_t)
/// - FP16_FP16 (ck::Tuple<ck::half_t, ck::half_t>)
/// - BF16 (ck::bhalf_t)
/// - BF16_BF16 (ck::Tuple<ck::bhalf_t, ck::bhalf_t>)
/// - FP32 (float)
/// - FP32_FP32 (ck::Tuple<float, float>)
/// - FP64 (double)
/// - FP8 (ck::f8_t)
/// - BF8 (ck::bf8_fnuz_t, ck::bf8_ocp_t)
/// - I8 (int8_t)
/// - I8_I8 (ck::Tuple<int8_t, int8_t>)
/// - U8 (uint8_t)
template <typename Instance>
constexpr builder::DataType conv_data_type()
{
    using InstTraits = InstanceTraits<Instance>;
    using ADataType  = typename InstTraits::ADataType;
    using enum builder::DataType;

    if constexpr(std::is_same_v<ADataType, ck::half_t>)
        return FP16;
    else if constexpr(std::is_same_v<ADataType, ck::Tuple<ck::half_t, ck::half_t>>)
        return FP16_FP16;
    else if constexpr(std::is_same_v<ADataType, ck::bhalf_t>)
        return BF16;
    else if constexpr(std::is_same_v<ADataType, ck::Tuple<ck::bhalf_t, ck::bhalf_t>>)
        return BF16_BF16;
    else if constexpr(std::is_same_v<ADataType, float>)
        return FP32;
    else if constexpr(std::is_same_v<ADataType, ck::Tuple<float, float>>)
        return FP32_FP32;
    else if constexpr(std::is_same_v<ADataType, double>)
        return FP64;
    else if constexpr(std::is_same_v<ADataType, ck::f8_t>)
        return FP8;
    else if constexpr(std::is_same_v<ADataType, ck::bf8_fnuz_t>)
        return BF8;
    else if constexpr(std::is_same_v<ADataType, ck::bf8_ocp_t>)
        return BF8;
    else if constexpr(std::is_same_v<ADataType, int8_t>)
        return I8;
    else if constexpr(std::is_same_v<ADataType, ck::Tuple<int8_t, int8_t>>)
        return I8_I8;
    else if constexpr(std::is_same_v<ADataType, uint8_t>)
        return U8;
    else
    {
        report_unsupported_data_type_error<ADataType>();
        return FP32; // Unreachable
    }
}

// ----------------------------------------------------------------------------
// Elementwise Operations
// ----------------------------------------------------------------------------

/// @brief Helper function to report unsupported elementwise operation with a clear error message.
/// @details This consteval function uses throw (not static_assert) to ensure the error is not
/// silently ignored during SFINAE. The thrown string becomes part of the compiler error message.
template <typename ElementwiseOp>
[[noreturn]] consteval void report_unsupported_elementwise_op_error()
{
    throw "Unsupported elementwise operation detected!\n"
          "The elementwise operation type is not recognized.\n"
          "Supported operations are: AddClamp, AddReluAdd, BiasBnormClamp, Bilinear, "
          "BiasNormalizeInInferClamp, Clamp, ConvInvscale, ConvScale, ConvScaleAdd, "
          "ConvScaleRelu, Scale, ScaleAdd, PassThrough, ScaleAddScaleAddRelu, DynamicUnaryOp, "
          "UnaryCombinedOp, Activation_Mul2_Clamp, Activation_Mul_Clamp, Add_Activation_Mul_Clamp, "
          "Add_Activation_Mul2_Clamp, Add_Mul_Activation_Mul_Clamp, Add_Mul2_Activation_Mul_Clamp, "
          "UnaryConvert.\n"
          "Please verify that your kernel instance uses a supported elementwise operation.";
}

/// @brief Derives the elementwise operation from an operation functor type.
/// @tparam ElementwiseOp Elementwise operation functor type.
/// @return A builder::ElementwiseOperation enum value corresponding to the operation.
/// @details This function uses the operation's type name to determine which elementwise
/// operation is being used. The comparison is case-insensitive.
///
/// Supported operations include:
/// - Activation operations: Relu, Sigmoid, Tanh, Gelu, Silu, Elu, Swish, etc.
/// - Scaling operations: Scale, ScaleAdd, ConvScale, ConvScaleAdd, etc.
/// - Clamping operations: Clamp, AddClamp, etc.
/// - Combined operations: Add_Activation_Mul_Clamp, etc.
/// - Utility operations: PassThrough, UnaryConvert, etc.
///
/// TODO: Consider changing this to direct checks on the types, not strings.
template <typename ElementwiseOp>
constexpr builder::ElementwiseOperation elementwise_op()
{
    using enum builder::ElementwiseOperation;
    constexpr std::string_view name = detail::elementwise_op_name<ElementwiseOp>();

    if constexpr(detail::case_insensitive_equal(name, "AddClamp"))
        return ADD_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "AddReluAdd"))
        return ADD_RELU_ADD;
    else if constexpr(detail::case_insensitive_equal(name, "BiasBnormClamp"))
        return BIAS_BNORM_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Bilinear"))
        return BILINEAR;
    else if constexpr(detail::case_insensitive_equal(name, "BiasNormalizeInInferClamp"))
        return BIAS_BNORM_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Clamp"))
        return CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "ConvInvscale"))
        return CONV_INVSCALE;
    else if constexpr(detail::case_insensitive_equal(name, "ConvScale"))
        return CONV_SCALE;
    else if constexpr(detail::case_insensitive_equal(name, "ConvScaleAdd"))
        return CONV_SCALE_ADD;
    else if constexpr(detail::case_insensitive_equal(name, "ConvScaleRelu"))
        return CONV_SCALE_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "Scale"))
        return SCALE;
    else if constexpr(detail::case_insensitive_equal(name, "ScaleAdd"))
        return SCALE_ADD;
    else if constexpr(detail::case_insensitive_equal(name, "PassThrough"))
        return PASS_THROUGH;
    else if constexpr(detail::case_insensitive_equal(name, "ScaleAddScaleAddRelu"))
        return SCALEADD_SCALEADD_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "DynamicUnaryOp"))
        return DYNAMIC_UNARY_OP;
    else if constexpr(detail::case_insensitive_equal(name, "UnaryCombinedOp"))
        return UNARY_COMBINED_OP;
    else if constexpr(detail::case_insensitive_equal(name, "Activation_Mul2_Clamp"))
        return ACTIVATION_MUL2_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Activation_Mul_Clamp"))
        return ACTIVATION_MUL_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Add_Activation_Mul_Clamp"))
        return ADD_ACTIVATION_MUL_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Add_Activation_Mul2_Clamp"))
        return ADD_ACTIVATION_MUL2_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Add_Mul_Activation_Mul_Clamp"))
        return ADD_MUL_ACTIVATION_MUL_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "Add_Mul2_Activation_Mul_Clamp"))
        return ADD_MUL2_ACTIVATION_MUL_CLAMP;
    else if constexpr(detail::case_insensitive_equal(name, "UnaryConvert"))
        return UNARY_CONVERT;
    else if constexpr(detail::case_insensitive_equal(name, "Logistic"))
        return LOGISTIC;
    else if constexpr(detail::case_insensitive_equal(name, "ClippedRelu"))
        return CLIPPED_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "Swish"))
        return SWISH;
    else if constexpr(detail::case_insensitive_equal(name, "Elu"))
        return ELU;
    else if constexpr(detail::case_insensitive_equal(name, "Power"))
        return POWER;
    else if constexpr(detail::case_insensitive_equal(name, "LeakyRelu"))
        return LEAKY_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "UnaryAbs"))
        return UNARY_ABS;
    else if constexpr(detail::case_insensitive_equal(name, "Relu"))
        return RELU;
    else if constexpr(detail::case_insensitive_equal(name, "SoftRelu"))
        return SOFT_RELU;
    else if constexpr(detail::case_insensitive_equal(name, "Sigmoid"))
        return SIGMOID;
    else if constexpr(detail::case_insensitive_equal(name, "TanH"))
        return TANH;
    else if constexpr(detail::case_insensitive_equal(name, "Gelu"))
        return GELU;
    else if constexpr(detail::case_insensitive_equal(name, "Silu"))
        return SILU;
    else
    {
        report_unsupported_elementwise_op_error<ElementwiseOp>();
        return PASS_THROUGH; // Unreachable
    }
}

// ----------------------------------------------------------------------------
// GEMM Padding
// ----------------------------------------------------------------------------

/// @brief Derives the GEMM padding specification from a kernel instance type.
/// @tparam Instance A device kernel instance type.
/// @return A builder::GemmPadding enum value corresponding to the kernel's padding configuration.
/// @details This function extracts the GEMM specialization from the Instance's InstanceTraits
/// and converts it to the builder framework's GemmPadding enum. The padding specification
/// indicates which dimensions (M, N, K, O) are padded to handle non-aligned tensor sizes.
///
/// Supported padding configurations include:
/// - DEFAULT: No padding
/// - M_PADDING, N_PADDING, K_PADDING, O_PADDING: Single dimension padding
/// - MN_PADDING, MK_PADDING, NK_PADDING, etc.: Two dimension padding
/// - MNK_PADDING, MNO_PADDING, etc.: Three dimension padding
/// - MNKO_PADDING: All dimensions padded
template <typename Instance>
constexpr builder::GemmPadding gemm_spec()
{
    using InstTraits = InstanceTraits<Instance>;
    using enum builder::GemmPadding;
    using enum ck::tensor_operation::device::GemmSpecialization;

    constexpr auto spec = InstTraits::kGemmSpecialization;

    switch(spec)
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

// ============================================================================
// SECTION 3: PIPELINE CONFIGURATION HELPERS
// ============================================================================

/// @brief Safely extracts the pipeline version from InstanceTraits.
/// @tparam InstTraits The InstanceTraits type to extract pipeline version from.
/// @return The pipeline version as a builder::PipelineVersion enum value.
/// @details This helper function checks if the InstanceTraits has a kPipelineVersion
/// field and extracts it if present. If not present, it returns a default value (V1).
/// This is necessary because not all convolution types expose pipeline version information.
template <typename InstTraits>
constexpr builder::PipelineVersion get_pipeline_version()
{
    if constexpr(requires { InstTraits::kPipelineVersion; })
    {
        return convert_pipeline_version<InstTraits::kPipelineVersion>();
    }
    else
    {
        return builder::PipelineVersion::V1;
    }
}

/// @brief Safely extracts the pipeline scheduler from InstanceTraits.
/// @tparam InstTraits The InstanceTraits type to extract pipeline scheduler from.
/// @return The pipeline scheduler as a builder::PipelineScheduler enum value.
/// @details This helper function checks if the InstanceTraits has a kPipelineScheduler
/// or kLoopScheduler field and extracts it if present. If neither is present, it returns
/// a default value (DEFAULT). This is necessary because different convolution types may
/// expose scheduler information through different field names.
template <typename InstTraits>
constexpr builder::PipelineScheduler get_pipeline_scheduler()
{
    if constexpr(requires { InstTraits::kPipelineScheduler; })
    {
        return convert_pipeline_scheduler<InstTraits::kPipelineScheduler>();
    }
    else if constexpr(requires { InstTraits::kLoopScheduler; })
    {
        return convert_pipeline_scheduler<InstTraits::kLoopScheduler>();
    }
    else
    {
        return builder::PipelineScheduler::DEFAULT;
    }
}

} // namespace ck_tile::reflect::conv
