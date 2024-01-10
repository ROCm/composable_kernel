// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include <numeric>
#include <cstdlib>
#include <iostream>
#include <initializer_list>
#include <vector>
#include <gtest/gtest.h>

#include "ck/library/reference_tensor_operation/cpu/reference_image_to_column.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/library/utility/host_tensor.hpp"

#include "ck/host_utility/kernel_launch.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/utility/common_header.hpp"
#include "ck/wrapper/layout.hpp"
#include "ck/wrapper/tensor.hpp"
#include "ck/wrapper/operations/copy.hpp"

static constexpr ck::index_t NumDimSpatial = 3;
using DataType                             = float;
using InputLayout                          = ck::tensor_layout::convolution::NDHWGC;

void CheckResult(std::vector<DataType>& input_data,
                 std::vector<DataType>& output_data,
                 const ck::index_t G,
                 const ck::index_t N,
                 const ck::index_t Di,
                 const ck::index_t Hi,
                 const ck::index_t Wi,
                 const ck::index_t Do,
                 const ck::index_t Ho,
                 const ck::index_t Wo,
                 const ck::index_t C,
                 const ck::index_t Z,
                 const ck::index_t Y,
                 const ck::index_t X,
                 std::array<ck::index_t, NumDimSpatial> filter_strides,
                 std::array<ck::index_t, NumDimSpatial> filter_dilations)
{
    const ck::index_t GC      = G * C;
    const ck::index_t NDoHoWo = N * Do * Ho * Wo;
    const ck::index_t ZYXC    = Z * Y * X * C;

    const std::vector<ck::index_t> left_pads  = {0, 0, 0};
    const std::vector<ck::index_t> right_pads = {0, 0, 0};

    const auto image_desc = HostTensorDescriptor(
        {G, N, C, Di, Hi, Wi}, {C, Di * Hi * Wi * GC, 1, Hi * Wi * GC, Wi * GC, GC});
    const auto gemm_desc = HostTensorDescriptor({G, NDoHoWo, ZYXC}, {ZYXC, ZYXC * G, 1});

    Tensor<DataType> host_input(image_desc);
    Tensor<DataType> host_output(gemm_desc);

    host_input.mData = input_data;

    auto ref_conv_tensor_rearrange = ck::tensor_operation::host::
        ReferenceImageToColumn<NumDimSpatial, InputLayout, DataType, DataType>{};
    auto ref_invoker  = ref_conv_tensor_rearrange.MakeInvoker();
    auto ref_argument = ref_conv_tensor_rearrange.MakeArgument(
        host_input,
        host_output,
        {Z, Y, X},
        {filter_strides[0], filter_strides[1], filter_strides[2]},
        {filter_dilations[0], filter_dilations[1], filter_dilations[2]},
        left_pads,
        right_pads);

    // init host output to zero
    host_output.SetZero();

    ref_invoker.Run(ref_argument);
    EXPECT_TRUE(ck::utils::check_err(output_data, host_output.mData));
}

// Test copy from Global to Global through LDS and VGPR
template <typename InputTensor,
          typename OutputTensor,
          typename BlockShape,
          typename ThreadLayoutShape>
__global__ void DeviceImageToColumnPad0(InputTensor input_tensor,
                                        OutputTensor output_tensor,
                                        const BlockShape tile_shape,
                                        const ThreadLayoutShape thread_layout)
{
    const ck::index_t block_idx = static_cast<ck::index_t>(blockIdx.x);

    // Get local tiles for global memory
    auto input_local_tile  = ck::wrapper::make_local_tile(input_tensor, tile_shape, block_idx);
    auto output_local_tile = ck::wrapper::make_local_tile(output_tensor, tile_shape, block_idx);

    // Get partition per thread
    const auto input_local_partition =
        ck::wrapper::make_local_partition(input_local_tile, thread_layout, threadIdx.x);
    auto output_local_partition =
        ck::wrapper::make_local_partition(output_local_tile, thread_layout, threadIdx.x);

    // Perform copy
    using DimAccessOrder                    = ck::Tuple<ck::Number<0>, ck::Number<1>>;
    constexpr ck::index_t vector_dim        = 1;
    constexpr ck::index_t scalar_per_vector = 4;
    ck::wrapper::copy<DimAccessOrder, vector_dim, scalar_per_vector>(input_local_partition,
                                                                     output_local_partition);
}

void PerformImageToColumnPad0(const ck::index_t G,
                              const ck::index_t N,
                              const ck::index_t Di,
                              const ck::index_t Hi,
                              const ck::index_t Wi,
                              const ck::index_t Do,
                              const ck::index_t Ho,
                              const ck::index_t Wo,
                              const ck::index_t C,
                              const ck::index_t Z,
                              const ck::index_t Y,
                              const ck::index_t X,
                              std::array<ck::index_t, NumDimSpatial> filter_strides,
                              std::array<ck::index_t, NumDimSpatial> filter_dilations)
{
    const ck::index_t ZYXC = Z * Y * X * C;
    const ck::index_t GC   = G * C;

    // shape: (G, (Wo, Ho, Do, N)), (C, X, Y, Z))
    const auto shape = ck::make_tuple(ck::make_tuple(G, ck::make_tuple(Wo, Ho, Do, N)),
                                      ck::make_tuple(C, X, Y, Z));
    const auto in_strides =
        ck::make_tuple(ck::make_tuple(C,
                                      ck::make_tuple(filter_strides[2] * GC,
                                                     filter_strides[1] * Wi * GC,
                                                     filter_strides[0] * Hi * Wi * GC,
                                                     Di * Hi * Wi * GC)),
                       ck::make_tuple(1,
                                      filter_dilations[2] * GC,
                                      filter_dilations[1] * Wi * GC,
                                      filter_dilations[0] * Hi * Wi * GC));
    const auto in_layout = ck::wrapper::make_layout(shape, in_strides);

    const auto out_strides = ck::make_tuple(
        ck::make_tuple(
            ZYXC,
            ck::make_tuple(ZYXC * G, Wo * ZYXC * G, Ho * Wo * ZYXC * G, Do * Ho * Wo * ZYXC * G)),
        ck::make_tuple(1, C, X * C, Y * X * C));
    const auto out_layout = ck::wrapper::make_layout(shape, out_strides);

    // 0,1,2...size(shape) - 1
    std::vector<DataType> input_data(N * Di * Hi * Wi * GC);
    std::iota(input_data.begin(), input_data.end(), 0);

    // Global memory buffers
    DeviceMem in_buf(input_data.size() * sizeof(DataType));
    DeviceMem out_buf(ck::wrapper::size(out_layout) * sizeof(DataType));

    in_buf.ToDevice(input_data.data());
    out_buf.SetZero();

    const auto thread_layout = ck::make_tuple(ck::Number<8>{}, ck::Number<16>{});
    const auto tile_shape    = ck::make_tuple(ck::Number<32>{}, ck::Number<64>{});

    // Create tensors for global memory
    auto input_tensor_global = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<const DataType*>(in_buf.GetDeviceBuffer()), in_layout);
    auto output_tensor_global = ck::wrapper::make_tensor<ck::wrapper::MemoryTypeEnum::Global>(
        static_cast<DataType*>(out_buf.GetDeviceBuffer()), out_layout);

    const ck::index_t grid_size = ck::math::integer_divide_ceil(ck::wrapper::size<0>(in_layout),
                                                                ck::wrapper::size<0>(tile_shape)) *
                                  ck::math::integer_divide_ceil(ck::wrapper::size<1>(in_layout),
                                                                ck::wrapper::size<1>(tile_shape));

    const auto kernel    = DeviceImageToColumnPad0<decltype(input_tensor_global),
                                                decltype(output_tensor_global),
                                                decltype(tile_shape),
                                                decltype(thread_layout)>;
    const float avg_time = launch_and_time_kernel(StreamConfig{nullptr, true},
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(ck::wrapper::size(thread_layout)),
                                                  0,
                                                  input_tensor_global,
                                                  output_tensor_global,
                                                  tile_shape,
                                                  thread_layout);

    std::size_t num_btype = G * N * Do * Ho * Wo * ZYXC * 2 * sizeof(DataType);
    float gb_per_sec      = num_btype / 1.E6 / avg_time;
    std::cout << "Perf: " << std::setw(10) << avg_time << " ms, " << gb_per_sec << " GB/s, "
              << std::endl;

    std::vector<DataType> output_data(ck::wrapper::size(shape));
    out_buf.FromDevice(output_data.data());
    CheckResult(input_data,
                output_data,
                G,
                N,
                Di,
                Hi,
                Wi,
                Do,
                Ho,
                Wo,
                C,
                Z,
                Y,
                X,
                filter_strides,
                filter_dilations);
}

TEST(TestImg2Col, PerformImageToColumn)
{
    constexpr ck::index_t G  = 4;
    constexpr ck::index_t N  = 32; // batch
    constexpr ck::index_t C  = 64; // input channel (per group)
    constexpr ck::index_t Z  = 3;  // filter D
    constexpr ck::index_t Y  = 3;  // filter H
    constexpr ck::index_t X  = 3;  // filter W
    constexpr ck::index_t Di = 9;  // input D
    constexpr ck::index_t Hi = 9;  // input H
    constexpr ck::index_t Wi = 7;  // input W
    constexpr ck::index_t Do = 7;  // output D
    constexpr ck::index_t Ho = 7;  // output H
    constexpr ck::index_t Wo = 5;  // output W
    PerformImageToColumnPad0(G,
                             N,
                             Di,
                             Hi,
                             Wi,
                             Do,
                             Ho,
                             Wo,
                             C,
                             Z,
                             Y,
                             X,
                             {1, 1, 1} /*filter_strides*/,
                             {1, 1, 1} /*filter_dilations*/);
}
