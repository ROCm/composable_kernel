// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/builder/testing/testing.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/bwd_weight.hpp"
#include "ck_tile/builder/testing/conv/bwd_data.hpp"
#include "ck_tile/builder/factory/helpers/ck_tile/conv_tile_tensor_type.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/grouped_convolution.hpp"
#include "ck_tile/host.hpp"
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

template <typename Conv>
concept HasGemmPipelineScheduler = requires {
    { Conv::GemmPipeline::Scheduler } -> std::convertible_to<ck_tile::GemmPipelineScheduler>;
};

template <typename Conv>
consteval ck_tile::index_t get_minimum_occupancy()
{
    if constexpr(HasGemmPipelineScheduler<Conv>)
        return Conv::GemmPipeline::Scheduler == ck_tile::GemmPipelineScheduler::Intrawave ? 1 : 2;
    else
        return 1;
}

template <auto SIGNATURE>
std::size_t gemm_split_k_output_size(auto kargs)
{
    std::size_t zeroing_size = 0;
    if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
    {
        zeroing_size = std::accumulate(std::begin(kargs.wei_g_k_c_xs_lengths.data),
                                       std::end(kargs.wei_g_k_c_xs_lengths.data),
                                       1,
                                       std::multiplies<std::size_t>());
    }

    if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
    {
        zeroing_size = std::accumulate(std::begin(kargs.in_g_n_c_wis_lengths.data),
                                       std::end(kargs.in_g_n_c_wis_lengths.data),
                                       1,
                                       std::multiplies<std::size_t>());
    }

    return zeroing_size;
}

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

    if(!Conv::IsSupportedArgument(kargs))
        return RunResult::not_supported("unsupported ck_tile arguments");

    // Workspace allocation (bwd weight only): may be non-zero for StreamK.
    [[maybe_unused]] std::size_t ws_size = 0;
    if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
        ws_size = Conv::GetWorkSpaceSize(kargs);
    ck_tile::DeviceMem workspace_dev(ws_size);
    if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
        Conv::SetWorkSpacePointer(kargs, workspace_dev.GetDeviceBuffer());

    const dim3 grids  = Conv::GridSize(kargs);
    const dim3 blocks = Conv::BlockSize();

    using Types = ck_tile::builder::factory::internal::TileConvTensorTypes<SIGNATURE.data_type>;

    const std::size_t zeroing_size = gemm_split_k_output_size<SIGNATURE>(kargs);

    auto preprocess = [&]() {
        if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
        {
            if constexpr(Conv::IsStreamK)
            {
                // StreamK: zero workspace flags before each kernel launch
                if(ws_size > 0)
                {
                    ck_tile::hip_check_error(hipMemsetAsync(
                        workspace_dev.GetDeviceBuffer(), 0, ws_size, s_conf.stream_id_));
                }
            }
            else if(kargs.k_batch > 1)
            {
                // Split-K: zero weight buffer for atomic accumulation
                ck_tile::hip_check_error(
                    hipMemsetAsync(kargs.wei_ptr,
                                   0,
                                   zeroing_size * sizeof(typename Types::EDataType),
                                   s_conf.stream_id_));
            }
        }

        if constexpr(ConvDirectionIsBackwardData<SIGNATURE>)
        {
            ck_tile::hip_check_error(
                hipMemsetAsync(kargs.in_ptr,
                               0,
                               zeroing_size * sizeof(typename Types::EDataType),
                               s_conf.stream_id_));
        }
    };

    constexpr index_t minimum_occupancy = get_minimum_occupancy<Conv>();

    if(s_conf.flush_cache_)
    {
        return RunResult::from_runtime(ck_tile::launch_kernel_time_mask_flush_cache(
            s_conf,
            preprocess,
            ck_tile::make_kernel<minimum_occupancy>(conv, grids, blocks, 0, kargs)));
    }
    else
    {
        return RunResult::from_runtime(ck_tile::launch_kernel_time_mask(
            s_conf,
            preprocess,
            ck_tile::make_kernel<minimum_occupancy>(conv, grids, blocks, 0, kargs)));
    }
}

template <auto SIGNATURE, typename InDataType, typename WeiDataType, typename OutDataType>
[[nodiscard]] RunResult run(CkTileConvInstance<SIGNATURE> auto& conv,
                            auto& elementwise_op,
                            const Args<SIGNATURE>& args,
                            InDataType* input,
                            WeiDataType* weight,
                            OutDataType* output,
                            const ck_tile::stream_config s_conf)
{
    using Conv              = std::remove_reference_t<decltype(conv)>;
    using ElementwiseOp     = std::remove_reference_t<decltype(elementwise_op)>;
    using WorkspaceDataType = typename ElementwiseOp::ComputeDataType;
    using CDataType         = typename ElementwiseOp::YDataType;
    using BlockShape        = typename ElementwiseOp::Problem::BlockShape;

    const auto param = args.to_ck_tile_conv_param();

    ck_tile::GroupedConvHostArgs<InDataType*, WeiDataType*, OutDataType*, ck_tile::PassThrough>
        host_args(param, input, weight, {}, output, args.k_batch);

    // Set-up for elementwise op kernel.
    const ck_tile::index_t spatial_lengths_accum =
        std::accumulate(host_args.filter_spatial_lengths_.begin(),
                        host_args.filter_spatial_lengths_.end(),
                        1,
                        std::multiplies<ck_tile::index_t>());
    ck_tile::DeviceMem ws_m_n_dev_buf(host_args.G_ * host_args.K_ * host_args.C_ *
                                      spatial_lengths_accum * sizeof(WorkspaceDataType));
    ck_tile::GroupedConvBwdWeightHostArgs ws_args =
        ck_tile::GroupedConvBwdWeightHostArgs(host_args);
    auto c_ptr      = ws_args.wei_ptr;
    ws_args.wei_ptr = ws_m_n_dev_buf.GetDeviceBuffer();

    auto kargs        = Conv::MakeKernelArgs(ws_args);
    const dim3 grids  = Conv::GridSize(kargs);
    const dim3 blocks = Conv::BlockSize();

    if(!Conv::IsSupportedArgument(kargs))
        return RunResult::not_supported("unsupported ck_tile arguments");

    ck_tile::index_t total_elements     = 1;
    std::vector<ck_tile::index_t> shape = {
        static_cast<ck_tile::index_t>(host_args.G_ * host_args.K_),
        static_cast<ck_tile::index_t>(host_args.C_ * spatial_lengths_accum)};

    for(auto d : shape)
        total_elements *= d;

    const ck_tile::index_t kBlockSize = ElementwiseOp::BlockSize();

    constexpr ck_tile::index_t elements_per_block = BlockShape::kBlockM;
    ck_tile::index_t kGridSize = (total_elements + elements_per_block - 1) / elements_per_block;

    auto input_tensors = ck_tile::make_tuple(static_cast<WorkspaceDataType*>(ws_args.wei_ptr));
    auto input_size    = ck_tile::make_tuple(shape[0], shape[1]);

    // Check if the kernel configuration is supported
    if(!ElementwiseOp::IsSupportedArgument(input_size))
    {
        return RunResult::not_supported("unsupported ck_tile arguments for elementwise op");
    }

    auto preprocess = [&]() {
        if constexpr(ConvDirectionIsBackwardWeight<SIGNATURE>)
        {
            if(kargs.k_batch > 1)
            {
                ck_tile::hip_check_error(
                    hipMemsetAsync(ws_args.wei_ptr,
                                   0,
                                   shape[0] * shape[1] * sizeof(WorkspaceDataType),
                                   s_conf.stream_id_));
            }
        }
    };

    constexpr index_t minimum_occupancy = get_minimum_occupancy<Conv>();

    if(s_conf.flush_cache_)
    {
        return RunResult::from_runtime(ck_tile::launch_kernel_time_mask_flush_cache(
            s_conf,
            preprocess,
            ck_tile::make_kernel<minimum_occupancy>(conv, grids, blocks, 0, kargs),
            ck_tile::make_kernel<minimum_occupancy>(
                elementwise_op,
                kGridSize,
                kBlockSize,
                0,
                input_size,
                ck_tile::make_tuple(shape[1], 1), // Input Stride
                ck_tile::make_tuple(shape[1], 1), // Output Stride
                input_tensors,
                static_cast<CDataType*>(c_ptr))));
    }
    else
    {
        return RunResult::from_runtime(ck_tile::launch_kernel_time_mask(
            s_conf,
            preprocess,
            ck_tile::make_kernel<minimum_occupancy>(conv, grids, blocks, 0, kargs),
            ck_tile::make_kernel<minimum_occupancy>(
                elementwise_op,
                kGridSize,
                kBlockSize,
                0,
                input_size,
                ck_tile::make_tuple(shape[1], 1), // Input Stride
                ck_tile::make_tuple(shape[1], 1), // Output Stride
                input_tensors,
                static_cast<CDataType*>(c_ptr))));
    }
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

/// @brief `run()` specialization for two-stage backwards weight convolution and CK Tile.
///
/// @tparam SIGNATURE Backwards weight convolution signature.
/// @returns RunResult about how the operation completed (or not).
///
/// @see run()
template <auto SIGNATURE>
    requires ConvDirectionIsBackwardWeight<SIGNATURE>
[[nodiscard]] RunResult run(CkTileConvInstance<SIGNATURE> auto& conv,
                            auto& elementwise_op,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs,
                            const ck_tile::stream_config s_conf = {})
{
    return detail::run(conv,
                       elementwise_op,
                       args,
                       static_cast<const void*>(inputs.input),
                       static_cast<void*>(outputs.weight),
                       static_cast<const void*>(inputs.output),
                       s_conf);
}

/// @brief `run()` specialization for backwards data convolution and CK Tile.
///
/// @tparam SIGNATURE Backward data convolution signature.
/// @returns RunResult about how the operation completed (or not).
///
/// @see run()
template <auto SIGNATURE>
    requires ConvDirectionIsBackwardData<SIGNATURE>
[[nodiscard]] RunResult run(CkTileConvInstance<SIGNATURE> auto& conv,
                            const Args<SIGNATURE>& args,
                            const Inputs<SIGNATURE>& inputs,
                            const Outputs<SIGNATURE>& outputs,
                            const ck_tile::stream_config s_conf = {})
{
    return detail::run(conv,
                       args,
                       static_cast<void*>(outputs.input),
                       static_cast<const void*>(inputs.weight),
                       static_cast<const void*>(inputs.output),
                       s_conf);
}

} // namespace ck_tile::builder::test
