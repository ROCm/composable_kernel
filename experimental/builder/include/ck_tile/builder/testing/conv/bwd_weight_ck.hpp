// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/testing.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_tensor_type.hpp"
#include <type_traits>
#include <array>

/// This file contains the implementation details for invoking/testing
/// bwd grouped convolution operations in old CK. The main item is the
/// `run()` function, which is the main implementation used to invoke
/// CK grouped forward convolution kernels.

namespace ck_tile::builder::test {

namespace detail {

/// @brief Concept for checking whether a bwd weight convolution is invoked like old CK.
///
/// This is the same as `::ck_tile::builder::test::CkConvBwdWeightInstance`, except
/// with some utility aliases. For that reason, its moved to this detail
/// namespace.
template <typename Conv,
          auto SIGNATURE,
          size_t SPATIAL_DIM = SIGNATURE.spatial_dim,
          // TODO: We shouldn't need to call into an internal namespace here.
          typename Types = factory::internal::ConvTensorDataTypes<SIGNATURE>,
          typename Ops   = factory::internal::ConvElementwiseOps<SIGNATURE>>
concept CkConvBwdWeightInstance = requires(Conv& conv,
                                           const Types::InDataType* p_a,
                                           Types::WeiDataType* p_b,
                                           const Types::OutDataType* p_e,
                                           std::array<index_t, SPATIAL_DIM + 3> lengths,
                                           std::array<index_t, SPATIAL_DIM + 3> strides,
                                           std::array<index_t, SPATIAL_DIM> filter,
                                           Ops::InElementwiseOp elementwise_a,
                                           Ops::WeiElementwiseOp elementwise_b,
                                           Ops::OutElementwiseOp elementwise_cde,
                                           ck::index_t split_k) {
    requires ValidConvSignature<SIGNATURE>;
    requires ConvDirectionIsBackwardWeight<SIGNATURE>;

    {
        conv.MakeArgument(p_a,
                          p_b,
                          p_e,
                          // A lengths/strides
                          lengths,
                          strides,
                          // B lengths/strides
                          lengths,
                          strides,
                          // E lengths/strides
                          lengths,
                          strides,
                          // strides/dilations/pads
                          filter,
                          filter,
                          filter,
                          filter,
                          // element-wise operations.
                          elementwise_a,
                          elementwise_b,
                          elementwise_cde,
                          split_k)
    };
};

/// @brief Concept for checking whether a bwd weight convolution is multiple-d and
/// invoked like old CK.
///
/// This is the same as `::ck_tile::builder::test::CkConvBwdWeightMultipleDInstance`, except
/// with some utility aliases. For that reason, its moved to this detail
/// namespace.
template <typename Conv,
          auto SIGNATURE,
          size_t SPATIAL_DIM = SIGNATURE.spatial_dim,
          // TODO: We shouldn't need to call into an internal namespace here.
          typename Types = factory::internal::ConvTensorDataTypes<SIGNATURE>,
          typename Ops   = factory::internal::ConvElementwiseOps<SIGNATURE>>
concept CkConvBwdWeightMultipleDInstance = requires(Conv& conv,
                                                    const Types::InDataType* p_a,
                                                    Types::WeiDataType* p_b,
                                                    const Types::OutDataType* p_e,
                                                    std::array<index_t, SPATIAL_DIM + 3> lengths,
                                                    std::array<index_t, SPATIAL_DIM + 3> strides,
                                                    std::array<index_t, SPATIAL_DIM> filter,
                                                    Ops::InElementwiseOp elementwise_a,
                                                    Ops::WeiElementwiseOp elementwise_b,
                                                    Ops::OutElementwiseOp elementwise_cde,
                                                    ck::index_t split_k) {
    requires ValidConvSignature<SIGNATURE>;
    requires ConvDirectionIsBackwardWeight<SIGNATURE>;

    {
        conv.MakeArgument(p_a,
                          p_b,
                          p_e,
                          // TODO: Actually support multiple d
                          {},
                          // A lengths/strides
                          lengths,
                          strides,
                          // B lengths/strides
                          lengths,
                          strides,
                          // E lengths/strides
                          lengths,
                          strides,
                          // TODO: Multiple D lengths/strides
                          {},
                          {},
                          // strides/dilations/pads
                          filter,
                          filter,
                          filter,
                          filter,
                          // element-wise operations.
                          elementwise_a,
                          elementwise_b,
                          elementwise_cde,
                          split_k)
    };
};

} // namespace detail

/// @brief Concept for checking whether a bwd weight convolution is invoked like old CK.
///
/// - SIGNATURE is the operation signature.
/// - Conv is a convolution instance created by the CK Builder API.
template <typename Conv, auto SIGNATURE>
concept CkConvBwdWeightInstance = detail::CkConvBwdWeightInstance<Conv, SIGNATURE>;

/// @brief Concept for checking whether a bwd weight convolution is multiple-d and
/// invoked like old CK.
///
/// - SIGNATURE is the operation signature.
/// - Conv is a convolution instance created by the CK Builder API.
template <typename Conv, auto SIGNATURE>
concept CkConvBwdWeightMultipleDInstance =
    detail::CkConvBwdWeightMultipleDInstance<Conv, SIGNATURE>;

/// @brief `run()` specialization for backward weight convolution and old CK.
///
/// @tparam SIGNATURE Forward convolution signature.
/// @returns RunResult about how the operation completed (or not).
///
/// @see run()
template <auto SIGNATURE>
[[nodiscard]] RunResult run(CkConvBwdWeightInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs)
{
    using Types = factory::internal::ConvTensorDataTypes<SIGNATURE>;

    constexpr auto spatial_dim = SIGNATURE.spatial_dim;

    const auto copy = [](const auto& src, auto& dst) {
        std::copy(src.begin(), src.end(), dst.begin());
    };

    const auto to_ck_lengths = [&](const auto& src) {
        std::array<ck::index_t, spatial_dim + 3> result;
        copy(src, result);
        return result;
    };

    const auto to_ck_extent = [&](const auto& extent) {
        std::array<ck::index_t, spatial_dim> result;
        copy(extent, result);
        return result;
    };

    const auto param = args.to_ck_conv_param();

    const auto input_desc  = args.make_input_descriptor();
    const auto weight_desc = args.make_weight_descriptor();
    const auto output_desc = args.make_output_descriptor();

    auto ck_args = conv.MakeArgument(static_cast<const Types::InDataType*>(inputs.input),
                                     static_cast<Types::WeiDataType*>(outputs.weight),
                                     static_cast<const Types::OutDataType*>(inputs.output),
                                     to_ck_lengths(input_desc.get_lengths()),
                                     to_ck_lengths(input_desc.get_strides()),
                                     to_ck_lengths(weight_desc.get_lengths()),
                                     to_ck_lengths(weight_desc.get_strides()),
                                     to_ck_lengths(output_desc.get_lengths()),
                                     to_ck_lengths(output_desc.get_strides()),
                                     to_ck_extent(param.conv_filter_strides_),
                                     to_ck_extent(param.conv_filter_dilations_),
                                     to_ck_extent(param.input_left_pads_),
                                     to_ck_extent(param.input_right_pads_),
                                     args.a_elementwise_op,
                                     args.b_elementwise_op,
                                     args.cde_elementwise_op,
                                     args.k_batch);

    if(!conv.IsSupportedArgument(ck_args))
        return RunResult::not_supported("invalid ck arguments");

    return RunResult::from_runtime(conv.MakeInvoker().Run(ck_args, {}));
}

/// @brief `run()` specialization for backward weight convolution and old CK.
///
/// This overload is specialized for Multiple-D.
///
/// @tparam SIGNATURE Forward convolution signature.
/// @returns RunResult about how the operation completed (or not).
///
/// @see run()
template <auto SIGNATURE>
[[nodiscard]] RunResult run(CkConvBwdWeightMultipleDInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs)
{
    using Types = factory::internal::ConvTensorDataTypes<SIGNATURE>;

    constexpr auto spatial_dim = SIGNATURE.spatial_dim;

    const auto copy = [](const auto& src, auto& dst) {
        std::copy(src.begin(), src.end(), dst.begin());
    };

    const auto to_ck_lengths = [&](const auto& src) {
        std::array<ck::index_t, spatial_dim + 3> result;
        copy(src, result);
        return result;
    };

    const auto to_ck_extent = [&](const auto& extent) {
        std::array<ck::index_t, spatial_dim> result;
        copy(extent, result);
        return result;
    };

    const auto param = args.to_ck_conv_param();

    const auto input_desc  = args.make_input_descriptor();
    const auto weight_desc = args.make_weight_descriptor();
    const auto output_desc = args.make_output_descriptor();

    auto ck_args = conv.MakeArgument(static_cast<const Types::InDataType*>(inputs.input),
                                     static_cast<Types::WeiDataType*>(outputs.weight),
                                     static_cast<const Types::OutDataType*>(inputs.output),
                                     {}, // TODO
                                     to_ck_lengths(input_desc.get_lengths()),
                                     to_ck_lengths(input_desc.get_strides()),
                                     to_ck_lengths(weight_desc.get_lengths()),
                                     to_ck_lengths(weight_desc.get_strides()),
                                     to_ck_lengths(output_desc.get_lengths()),
                                     to_ck_lengths(output_desc.get_strides()),
                                     {}, // TODO
                                     {}, // TODO
                                     to_ck_extent(param.conv_filter_strides_),
                                     to_ck_extent(param.conv_filter_dilations_),
                                     to_ck_extent(param.input_left_pads_),
                                     to_ck_extent(param.input_right_pads_),
                                     args.a_elementwise_op,
                                     args.b_elementwise_op,
                                     args.cde_elementwise_op,
                                     args.k_batch);

    if(!conv.IsSupportedArgument(ck_args))
        return RunResult::not_supported("invalid ck arguments");

    return RunResult::from_runtime(conv.MakeInvoker().Run(ck_args, {}));
}

} // namespace ck_tile::builder::test
