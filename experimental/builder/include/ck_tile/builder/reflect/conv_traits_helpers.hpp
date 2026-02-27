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

// Forward convolution layout concept - checks for A/B/E layout types
template <typename T>
concept HasFwdConvLayouts = requires {
    typename T::ALayout;
    typename T::BLayout;
    typename T::ELayout;
};

// Backwards weight layout concept - checks for In, wei and out layouts
template <typename T>
concept HasBwdWeiLayouts = requires {
    typename T::InLayout;
    typename T::WeiLayout;
    typename T::OutLayout;
};

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

// Helper variable template to check if CK layout enums match
template <typename A,
          typename B,
          typename E,
          typename ExpectedA,
          typename ExpectedB,
          typename ExpectedE>
inline constexpr bool layouts_are =
    std::is_same_v<A, ExpectedA> && std::is_same_v<B, ExpectedB> && std::is_same_v<E, ExpectedE>;

/// @brief Helper function to report unsupported layout combinations with a clear error message.
/// @details This consteval function uses throw (not static_assert) to ensure the error is not
/// silently ignored during SFINAE. The thrown string becomes part of the compiler error message.
/// @details This consteval function is designed to fail at compile time with a descriptive
/// error message when an unsupported layout combination is encountered.
template <typename A, typename B, typename E, int SpatialDim>
[[noreturn]] consteval void report_unsupported_layout_error()
{
    // This will produce a compile-time error with the exception message
    throw "Unsupported convolution layout combination detected!\n"
          "The combination of ALayout, BLayout, and ELayout template parameters\n"
          "is not recognized for the given spatial dimension.\n"
          "Please verify that your convolution instance uses a supported layout configuration.\n"
          "Check the conv_layout() function for the list of supported layout combinations.";
}

template <typename A, typename B, typename E, int kSpatialDim>
constexpr auto conv_layout()
{

    // Helper lambda to construct layout array
    auto layouts = [](auto... Ls) { return std::array<builder::TensorLayout, 3>{Ls...}; };

    namespace ctl = ck::tensor_layout::convolution;
    using enum builder::TensorLayout;

    switch(kSpatialDim)
    {
    case 1:
        if constexpr(layouts_are<A, B, E, ctl::GNWC, ctl::GKXC, ctl::GNWK>)
            return layouts(GNWC, GKXC, GNWK);
        if constexpr(layouts_are<A, B, E, ctl::G_NW_C, ctl::G_K_X_C, ctl::G_NW_K>)
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
        if constexpr(layouts_are<A, B, E, ctl::G_NHW_C, ctl::G_K_YX_C, ctl::G_NHW_K>)
            return layouts(GNHWC, GKYXC, GNHWK);
        if constexpr(layouts_are<A, B, E, ctl::NHWGC, ctl::GKYXC, ctl::NHWGK>)
            return layouts(NHWGC, GKYXC, NHWGK);
        if constexpr(layouts_are<A, B, E, ctl::NHWGC, ctl::KYXGC, ctl::NHWGK>)
            return layouts(NHWGC, GKYXC, NHWGK);
        if constexpr(layouts_are<A, B, E, ctl::NGCHW, ctl::GKYXC, ctl::NGKHW>)
            return layouts(NGCHW, GKYXC, NGKHW);
        if constexpr(layouts_are<A, B, E, ctl::NGCHW, ctl::GKCYX, ctl::NGKHW>)
            return layouts(NGCHW, GKCYX, NGKHW);
        break;
    case 3:
        if constexpr(layouts_are<A, B, E, ctl::GNDHWC, ctl::GKZYXC, ctl::GNDHWK>)
            return layouts(GNDHWC, GKZYXC, GNDHWK);
        if constexpr(layouts_are<A, B, E, ctl::G_NDHW_C, ctl::G_K_ZYX_C, ctl::G_NDHW_K>)
            return layouts(GNDHWC, GKZYXC, GNDHWK);
        if constexpr(layouts_are<A, B, E, ctl::NDHWGC, ctl::GKZYXC, ctl::NDHWGK>)
            return layouts(NDHWGC, GKZYXC, NDHWGK);
        if constexpr(layouts_are<A, B, E, ctl::NGCDHW, ctl::GKZYXC, ctl::NGKDHW>)
            return layouts(NGCDHW, GKZYXC, NGKDHW);
        if constexpr(layouts_are<A, B, E, ctl::NGCDHW, ctl::GKCZYX, ctl::NGKDHW>)
            return layouts(NGCDHW, GKCZYX, NGKDHW);
        break;
    }

    // If we reach here, the layout combination is not supported
    // Call consteval function to trigger a compile-time error with a clear message
    report_unsupported_layout_error<A, B, E, kSpatialDim>();

    // This return is unreachable but needed to satisfy the compiler
    return layouts(GNHWC, GKYXC, GNHWK);
}

/// @brief Derives the grouped convolution layout from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return An std::array corresponding to the tensor layouts:
///             index 0 -> Input layout
///             index 1 -> Weight layout
///             index 2 -> Output layout

template <typename Instance>
constexpr auto fwd_conv_layout()
    requires HasFwdConvLayouts<InstanceTraits<Instance>>
{

    using A = typename InstanceTraits<Instance>::ALayout;
    using B = typename InstanceTraits<Instance>::BLayout;
    using E = typename InstanceTraits<Instance>::ELayout;
    return conv_layout<A, B, E, InstanceTraits<Instance>::kSpatialDim>();
}

/// @brief Derives the grouped convolution layout from a device kernel `Instance` type.
/// @tparam Instance The device kernel instance type.
/// @return An std::array corresponding to the tensor layouts:
///             index 0 -> Input layout
///             index 1 -> Weight layout
///             index 2 -> Output layout
template <typename Instance>
constexpr auto bwd_wei_conv_layout()
    requires HasBwdWeiLayouts<InstanceTraits<Instance>>
{

    using A = typename InstanceTraits<Instance>::InLayout;
    using B = typename InstanceTraits<Instance>::WeiLayout;
    using E = typename InstanceTraits<Instance>::OutLayout;
    return conv_layout<A, B, E, InstanceTraits<Instance>::kSpatialDim>();
}

// ----------------------------------------------------------------------------
// Data Types
// ----------------------------------------------------------------------------

/// @brief Helper function to report unsupported data type with a clear error message.
template <typename DataTypeFromInstance>
[[noreturn]] consteval void report_unsupported_data_type_error()
{
    throw "Unsupported data type detected!\n"
          "The DataTypeFromInstance is not recognized.\n"
          "Supported types are: ck::half_t (FP16), ck::Tuple<ck::half_t, ck::half_t> (FP16_FP16), "
          "ck::bhalf_t (BF16), ck::Tuple<ck::bhalf_t, ck::bhalf_t> (BF16_BF16), float (FP32), "
          "ck::Tuple<float, float> (FP32_FP32), double (FP64), ck::f8_t (FP8), ck::bf8_fnuz_t "
          "(BF8), "
          "int8_t (I8), ck::Tuple<int8_t, int8_t> (I8_I8), uint8_t (U8).\n"
          "Please verify that your kernel instance uses a supported data type.";
}

/// @brief Derives the data type from a device kernel `Instance` type.
/// Returns a `builder::DataType` enum value (e.g., FP16, BF16, FP32, BF8).
// Note: maybe move to types.hpp?
template <typename DataTypeFromInstance>
constexpr builder::DataType conv_data_type()

{
    using enum builder::DataType;

    if constexpr(std::is_same_v<DataTypeFromInstance, ck::half_t>)
        return FP16;
    else if constexpr(std::is_same_v<DataTypeFromInstance, ck::Tuple<ck::half_t, ck::half_t>>)
        return FP16_FP16;
    else if constexpr(std::is_same_v<DataTypeFromInstance, ck::bhalf_t>)
        return BF16;
    else if constexpr(std::is_same_v<DataTypeFromInstance, ck::Tuple<ck::bhalf_t, ck::bhalf_t>>)
        return BF16_BF16;
    else if constexpr(std::is_same_v<DataTypeFromInstance, float>)
        return FP32;
    else if constexpr(std::is_same_v<DataTypeFromInstance, ck::Tuple<float, float>>)
        return FP32_FP32;
    else if constexpr(std::is_same_v<DataTypeFromInstance, double>)
        return FP64;
    else if constexpr(std::is_same_v<DataTypeFromInstance, ck::f8_t>)
        return FP8;
    else if constexpr(std::is_same_v<DataTypeFromInstance, ck::bf8_fnuz_t>)
        return BF8;
    else if constexpr(std::is_same_v<DataTypeFromInstance, ck::bf8_ocp_t>)
        return BF8;
    else if constexpr(std::is_same_v<DataTypeFromInstance, int8_t>)
        return I8;
    else if constexpr(std::is_same_v<DataTypeFromInstance, ck::Tuple<int8_t, int8_t>>)
        return I8_I8;
    else if constexpr(std::is_same_v<DataTypeFromInstance, uint8_t>)
        return U8;
    else
    {
        report_unsupported_data_type_error<DataTypeFromInstance>();
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

// ============================================================================
// SECTION 4: Helper functions for common structures often used in reflection
// ============================================================================

template <typename InstTraits>
constexpr DataTileInfo conv_traits_data_tile(int k_or_k0 = InstTraits::kKPerBlock)
{
    return DataTileInfo{.m = InstTraits::kMPerBlock, .n = InstTraits::kNPerBlock, .k = k_or_k0};
}

template <typename InstTraits>
constexpr InputTileTransferInfo
conv_traits_a_transfer_params(int _k1, int kPerBlock = InstTraits::kKPerBlock)
{
    return InputTileTransferInfo{
        .tile_dimensions = {.k0 = kPerBlock / _k1, .m_or_n = InstTraits::kMPerBlock, .k1 = _k1},
        .transfer_params = {.k1                    = _k1,
                            .thread_cluster_dims   = InstTraits::kAThreadClusterLengths,
                            .thread_cluster_order  = InstTraits::kAThreadClusterArrangeOrder,
                            .src_access_order      = InstTraits::kABlockTransferSrcAccessOrder,
                            .src_vector_dim        = InstTraits::kABlockTransferSrcVectorDim,
                            .src_scalar_per_vector = InstTraits::kABlockTransferSrcScalarPerVector,
                            .dst_scalar_per_vector_k1 =
                                InstTraits::kABlockTransferDstScalarPerVectorK1,
                            .lds_padding = static_cast<bool>(InstTraits::kABlockLdsExtraM)}};
}

template <typename InstTraits>
constexpr InputTileTransferInfo
conv_traits_b_transfer_params(int _k1, int kPerBlock = InstTraits::kKPerBlock)
{
    return InputTileTransferInfo{
        .tile_dimensions = {.k0 = kPerBlock / _k1, .m_or_n = InstTraits::kNPerBlock, .k1 = _k1},
        .transfer_params = {.k1                    = _k1,
                            .thread_cluster_dims   = InstTraits::kBThreadClusterLengths,
                            .thread_cluster_order  = InstTraits::kBThreadClusterArrangeOrder,
                            .src_access_order      = InstTraits::kBBlockTransferSrcAccessOrder,
                            .src_vector_dim        = InstTraits::kBBlockTransferSrcVectorDim,
                            .src_scalar_per_vector = InstTraits::kBBlockTransferSrcScalarPerVector,
                            .dst_scalar_per_vector_k1 =
                                InstTraits::kBBlockTransferDstScalarPerVectorK1,
                            .lds_padding = static_cast<bool>(InstTraits::kBBlockLdsExtraN)}};
}

template <typename InstTraits>
constexpr WarpGemmParams conv_traits_wmma_warp_gemm_params()
{
    return WarpGemmParams{.gemm_m = InstTraits::kMPerWmma,
                          .gemm_n = InstTraits::kNPerWmma,
                          .m_iter = InstTraits::kMRepeat,
                          .n_iter = InstTraits::kNRepeat};
}

template <typename InstTraits>
constexpr WarpGemmParams conv_traits_xdl_warp_gemm_params()
{
    return WarpGemmParams{.gemm_m = InstTraits::kMPerXDL,
                          .gemm_n = InstTraits::kNPerXDL,
                          .m_iter = InstTraits::kMXdlPerWave,
                          .n_iter = InstTraits::kNXdlPerWave};
}

template <typename InstTraits>
constexpr OutputTileTransferInfo conv_traits_wmma_c_tile_transfer(
    ck::index_t CDEBlockTansferScalarPerVector = InstTraits::kCDEBlockTransferScalarPerVector)
{
    return OutputTileTransferInfo{
        .shuffle_params      = {.m_gemms_per_shuffle = InstTraits::kCShuffleMRepeatPerShuffle,
                                .n_gemms_per_shuffle = InstTraits::kCShuffleNRepeatPerShuffle},
        .thread_cluster_dims = {InstTraits::kCDEThreadClusterLengths[0],
                                InstTraits::kCDEThreadClusterLengths[1],
                                InstTraits::kCDEThreadClusterLengths[2],
                                InstTraits::kCDEThreadClusterLengths[3]},
        .scalar_per_vector   = CDEBlockTansferScalarPerVector};
}

template <typename InstTraits>
constexpr OutputTileTransferInfo conv_traits_xdl_c_tile_transfer()
{
    return OutputTileTransferInfo{
        .shuffle_params      = {.m_gemms_per_shuffle = InstTraits::kCShuffleMXdlPerWavePerShuffle,
                                .n_gemms_per_shuffle = InstTraits::kCShuffleNXdlPerWavePerShuffle},
        .thread_cluster_dims = {InstTraits::kCThreadClusterLengths[0],
                                InstTraits::kCThreadClusterLengths[1],
                                InstTraits::kCThreadClusterLengths[2],
                                InstTraits::kCThreadClusterLengths[3]},
        .scalar_per_vector   = InstTraits::kCBlockTransferScalarPerVector};
}

} // namespace ck_tile::reflect::conv
