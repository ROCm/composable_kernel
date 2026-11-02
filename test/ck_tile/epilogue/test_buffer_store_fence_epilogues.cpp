// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <vector>

#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/ops/common/generic_2d_block_shape.hpp"
#include "ck_tile/ops/epilogue/default_2d_epilogue.hpp"
#include "ck_tile/ops/epilogue/dynamic_quant_epilogue.hpp"
#include "ck_tile/ops/smoothquant/pipeline/smoothquant_pipeline_default_policy.hpp"

namespace {
using namespace ck_tile;
constexpr index_t Rows  = 3;
constexpr index_t Cols  = 120;
constexpr index_t Guard = 16;
using Shape             = Generic2dBlockShape<sequence<4, 128>, sequence<4, 32>, sequence<1, 4>>;
struct DistributionProblem
{
    using BlockShape = Shape;
};

template <typename T>
CK_TILE_DEVICE auto OutputWindow(T* output)
{
    auto view = make_naive_tensor_view<address_space_enum::global>(
        output, make_tuple(Rows, Cols), make_tuple(Cols, 1), number<4>{}, number<1>{});
    auto padded =
        pad_tensor_view(view, make_tuple(number<4>{}, number<128>{}), sequence<true, true>{});
    return make_tile_window(padded, make_tuple(number<4>{}, number<128>{}), {0, 0});
}

__global__ void DefaultEpilogue(float* output)
{
    auto tile = make_static_distributed_tensor<float>(
        SmoothquantPipelineDefaultPolicy::MakeXBlockTileDistribution<DistributionProblem>());
    set_tile(tile, 42.25f);
    auto window   = OutputWindow(output);
    using Problem = Default2DEpilogueProblem<float, float, true, true, true>;
    Default2DEpilogue<Problem>{}(window, tile, nullptr);
}

__global__ void QuantEpilogue(int8_t* output, float* scales)
{
    auto tile = make_static_distributed_tensor<float>(
        SmoothquantPipelineDefaultPolicy::MakeXBlockTileDistribution<DistributionProblem>());
    set_tile(tile, 254.0f);
    auto window     = OutputWindow(output);
    auto scale_view = make_naive_tensor_view<address_space_enum::global>(
        scales, make_tuple(Rows), make_tuple(1), number<1>{});
    auto padded_scales = pad_tensor_view(scale_view, make_tuple(number<4>{}), sequence<true>{});
    auto scale_window  = make_tile_window(padded_scales, make_tuple(number<4>{}), {0});
    using Traits       = DynamicQuantEpilogueTraits<true, true, false, true>;
    using Problem      = DynamicQuantEpilogueProblem<float, float, float, int8_t, Shape, Traits>;
    __shared__ char smem[1024];
    DynamicQuantEpilogue<Problem>{}(window, scale_window, tile, smem);
}

TEST(TestBufferStoreFenceEpilogues, DefaultPaddedRawStores)
{
    std::vector<float> output(Rows * Cols + 2 * Guard, -123.0f);
    DeviceMem device(output.size() * sizeof(float));
    device.ToDevice(output.data());
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    hipLaunchKernelGGL(DefaultEpilogue,
                       dim3(1),
                       dim3(128),
                       0,
                       nullptr,
                       static_cast<float*>(device.GetDeviceBuffer()) + Guard);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    device.FromDevice(output.data());
    for(index_t i = 0; i < static_cast<index_t>(output.size()); ++i)
        EXPECT_EQ(output[i], i >= Guard && i < Guard + Rows * Cols ? 42.25f : -123.0f)
            << "element " << i;
}

TEST(TestBufferStoreFenceEpilogues, DynamicQuantPaddedRawStores)
{
    std::vector<int8_t> output(Rows * Cols + 2 * Guard, -123);
    std::vector<float> scales(Rows + 2 * Guard, -123.0f);
    DeviceMem device(output.size());
    DeviceMem device_scales(scales.size() * sizeof(float));
    device.ToDevice(output.data());
    device_scales.ToDevice(scales.data());
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    hipLaunchKernelGGL(QuantEpilogue,
                       dim3(1),
                       dim3(128),
                       0,
                       nullptr,
                       static_cast<int8_t*>(device.GetDeviceBuffer()) + Guard,
                       static_cast<float*>(device_scales.GetDeviceBuffer()) + Guard);
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    device.FromDevice(output.data());
    device_scales.FromDevice(scales.data());
    for(index_t i = 0; i < static_cast<index_t>(output.size()); ++i)
        EXPECT_EQ(output[i], i >= Guard && i < Guard + Rows * Cols ? 127 : -123) << "element " << i;
    for(index_t i = 0; i < static_cast<index_t>(scales.size()); ++i)
        EXPECT_EQ(scales[i], i >= Guard && i < Guard + Rows ? 2.0f : -123.0f) << "scale " << i;
}
} // namespace
