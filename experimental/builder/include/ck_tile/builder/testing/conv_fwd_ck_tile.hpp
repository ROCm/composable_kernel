// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/conv_fwd.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/builder/factory/helpers/ck/conv_elementwise_op.hpp"
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
    { Conv::BlockSize() };
};

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
/// @throws std::runtime_error if the arguments weren't actually valid for the
/// operation. This should be caught and reported by the testing framework.
/// @return std::tuple<bool, float> - whether the problem is supported and
///         kernel execution time (0.0f if s_conf time_kernel is false).
///
/// @see run()
template <auto SIGNATURE>
    requires ValidConvSignature<SIGNATURE> && ConvDirectionIsForward<SIGNATURE>
std::tuple<bool, float> run(CkTileConvInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs,
                            const ck_tile::stream_config s_conf = {})
{
    using Conv       = std::remove_reference_t<decltype(conv)>;
    const auto param = args.to_ck_tile_conv_param();

    ck_tile::GroupedConvFwdHostArgs<> host_args(
        param, inputs.input, inputs.weight, {}, outputs.output, args.k_batch);

    auto kargs = Conv::MakeKernelArgs(host_args);

    const dim3 grids  = Conv::GridSize(kargs);
    const dim3 blocks = Conv::BlockSize();

    if(!Conv::IsSupportedArgument(kargs))
    {
        std::cout << "Not supported!";
        return std::make_tuple(false, 0.f);
    }

    constexpr index_t minimum_occupancy =
        Conv::GemmPipeline::Scheduler == ck_tile::GemmPipelineScheduler::Intrawave ? 1 : 2;

    return std::make_tuple(
        true,
        ck_tile::launch_kernel(
            s_conf, ck_tile::make_kernel<minimum_occupancy>(conv, grids, blocks, 0, kargs)));
}

} // namespace ck_tile::builder::test
