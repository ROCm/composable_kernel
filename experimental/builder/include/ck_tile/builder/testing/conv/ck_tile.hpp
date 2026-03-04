// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/testing.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/bwd_weight.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include <type_traits>
#include <array>

/// This file contains the implementation details for invoking/testing
/// grouped convolution operations in CK Tile. The main item is the
/// `run()` function, which is the main implementation used to invoke
/// CK Tile grouped forward convolution kernels.

namespace ck_tile::builder::test {

namespace detail {

/// @brief Concept for checking whether this is the CK Tile convolution
/// implementation.
///
/// This is the same as `::ck_tile::builder::test::CkConvInstance`, except
/// with some utility aliases. For that reason, its moved to this detail
/// namespace.
template <typename Conv, auto SIGNATURE>
concept CkTileConvInstance = requires(Conv&) {
    requires ValidConvSignature<SIGNATURE>;
    { Conv::BlockSize() };
};

template <auto SIGNATURE, typename InDataType, typename WeiDataType, typename OutDataType>
[[nodiscard]] RunResult run(CkTileConvInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            InDataType* input,
                            WeiDataType* weight,
                            OutDataType* output,
                            const ck_tile::stream_config s_conf)
{
    using Conv       = std::remove_reference_t<decltype(conv)>;
    const auto param = args.to_ck_tile_conv_param();

    ck_tile::GroupedConvHostArgs<InDataType*, WeiDataType*, OutDataType*, ck_tile::PassThrough>
        host_args(param, input, weight, {}, output, args.k_batch);

    auto kargs = Conv::MakeKernelArgs(host_args);

    const dim3 grids  = Conv::GridSize(kargs);
    const dim3 blocks = Conv::BlockSize();

    if(!Conv::IsSupportedArgument(kargs))
        return RunResult::not_supported("unsupported ck_tile arguments");

    const std::size_t zeroing_size = std::accumulate(std::begin(kargs.wei_g_k_c_xs_lengths.data),
                                                     std::end(kargs.wei_g_k_c_xs_lengths.data),
                                                     1,
                                                     std::multiplies<std::size_t>());
    auto preprocess                = [&]() {
        if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
        {
            if(args.k_batch > 1)
            {
                ck_tile::hip_check_error(
                    hipMemsetAsync(kargs.wei_ptr, 0, zeroing_size, s_conf.stream_id_));
            }
        }
    };

    constexpr index_t minimum_occupancy =
        Conv::GemmPipeline::Scheduler == ck_tile::GemmPipelineScheduler::Intrawave ? 1 : 2;

    return RunResult::from_runtime(ck_tile::launch_kernel_time_mask(
        s_conf,
        preprocess,
        ck_tile::make_kernel<minimum_occupancy>(conv, grids, blocks, 0, kargs)));
}

} // namespace detail

/// @brief Concept for checking whether a convolution is invoked like CK Tile.
///
/// This concept is used to tell whether a convolution implementation is
/// likely to be an "CK Tile" implementation - that is, whether we should
/// invoke it as an CK Tile kernel. This is mainly used with `run()` to
/// differentiate which implementation that should be invoked.
///
/// - SIGNATURE is the operation signature.
/// - Conv is a convolution instance created by the CK Builder API.
template <typename Conv, auto SIGNATURE>
concept CkTileConvInstance = detail::CkTileConvInstance<Conv, SIGNATURE>;

/// @brief `run()` specialization for forward convolution and CK Tile.
///
/// @tparam SIGNATURE Forward convolution signature.
/// @returns RunResult about how the operation completed (or not).
///
/// @see run()
template <auto SIGNATURE>
    requires ConvDirectionIsForward<SIGNATURE>
[[nodiscard]] RunResult run(CkTileConvInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs,
                            const ck_tile::stream_config s_conf = {})
{
    return detail::run(conv,
                       args,
                       static_cast<const void*>(inputs.input),
                       static_cast<const void*>(inputs.weight),
                       static_cast<void*>(outputs.output),
                       s_conf);
}

/// @brief `run()` specialization for backwards weight convolution and CK Tile.
///
/// @tparam SIGNATURE Backwards weight convolution signature.
/// @returns RunResult about how the operation completed (or not).
///
/// @see run()
template <auto SIGNATURE>
    requires ConvDirectionIsBackwardWeight<SIGNATURE>
[[nodiscard]] RunResult run(CkTileConvInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs,
                            const ck_tile::stream_config s_conf = {})
{
    return detail::run(conv,
                       args,
                       static_cast<const void*>(inputs.input),
                       static_cast<void*>(outputs.weight),
                       static_cast<const void*>(inputs.output),
                       s_conf);
}

} // namespace ck_tile::builder::test
