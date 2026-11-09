// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_xdl_instance.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

namespace detail {

using F32BaseOp = DeviceGroupedConvBwdDataMultipleD<2,
                                                    NHWGK,
                                                    GKYXC,
                                                    Empty_Tuple,
                                                    NHWGC,
                                                    F32,
                                                    F32,
                                                    Empty_Tuple,
                                                    F32,
                                                    PassThrough,
                                                    PassThrough,
                                                    PassThrough>;

template <ConvolutionBackwardDataSpecialization ConvSpec>
using GenericF32 =
    device_grouped_conv_bwd_data_xdl_f32_instances<2, NHWGK, GKYXC, Empty_Tuple, NHWGC, ConvSpec>;

template <ConvolutionBackwardDataSpecialization ConvSpec>
using NoshuffleF32 = device_grouped_conv_bwd_data_xdl_f32_noshuffle_instances<2,
                                                                              NHWGK,
                                                                              GKYXC,
                                                                              Empty_Tuple,
                                                                              NHWGC,
                                                                              ConvSpec>;

inline constexpr std::size_t GenericF32Count   = 17;
inline constexpr std::size_t NoshuffleF32Count = 15;
using F32KernelSlots =
    std::array<std::unique_ptr<F32BaseOp>, 2 * (GenericF32Count + NoshuffleF32Count)>;

// Keep shared single-group GEMM instantiations in one translation unit. Each
// (BlockSize, MPerBlock, NPerBlock) tile bucket owns both epilogues and both
// convolution specializations. These indices refer to the existing full tuples.
using GenericGroup0   = std::index_sequence<1, 2, 3, 5, 7, 11, 12, 13, 14, 16>;
using GenericGroup1   = std::index_sequence<0, 4, 6, 8, 9, 10, 15>;
using NoshuffleGroup0 = std::index_sequence<0, 1, 3, 5, 9, 10, 11, 12, 14>;
using NoshuffleGroup1 = std::index_sequence<2, 4, 6, 7, 8, 13>;

template <std::size_t Size, std::size_t... Left, std::size_t... Right>
constexpr bool covers_tuple_once(std::index_sequence<Left...>, std::index_sequence<Right...>)
{
    std::array<unsigned, Size> counts{};
    ((++counts[Left]), ...);
    ((++counts[Right]), ...);
    for(auto count : counts)
    {
        if(count != 1)
            return false;
    }
    return true;
}

static_assert(std::tuple_size_v<GenericF32<ConvBwdDataDefault>> == GenericF32Count);
static_assert(std::tuple_size_v<GenericF32<ConvBwdDataFilter1x1Stride1Pad0>> == GenericF32Count);
static_assert(std::tuple_size_v<NoshuffleF32<ConvBwdDataDefault>> == NoshuffleF32Count);
static_assert(std::tuple_size_v<NoshuffleF32<ConvBwdDataFilter1x1Stride1Pad0>> ==
              NoshuffleF32Count);
static_assert(covers_tuple_once<GenericF32Count>(GenericGroup0{}, GenericGroup1{}));
static_assert(covers_tuple_once<NoshuffleF32Count>(NoshuffleGroup0{}, NoshuffleGroup1{}));

template <typename Instances, std::size_t Offset, std::size_t... Indices>
void fill_f32_kernel_slots(F32KernelSlots& slots, std::index_sequence<Indices...>)
{
    static_assert((std::is_base_of_v<F32BaseOp, std::tuple_element_t<Indices, Instances>> && ...));
    static_assert(
        (std::is_default_constructible_v<std::tuple_element_t<Indices, Instances>> && ...));
    // Name tuple elements without constructing the full tuple: unselected
    // operator constructors and vtables must remain in their owning source.
    ((slots[Offset + Indices] = std::make_unique<std::tuple_element_t<Indices, Instances>>()), ...);
}

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_kernel_group1_instances(
    F32KernelSlots& slots);

void add_device_grouped_conv2d_bwd_data_xdl_nhwgk_gkyxc_nhwgc_f32_nongrouped_match_instances(
    std::vector<std::unique_ptr<F32BaseOp>>& instances);

} // namespace detail

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
