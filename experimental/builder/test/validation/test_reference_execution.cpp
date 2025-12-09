// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/conv_builder.hpp"
#include "ck_tile/builder/types.hpp"
#include "utils/ckb_conv_test_configs.hpp"
#include "ck/library/utility/device_memory.hpp"
#include <gtest/gtest.h>
#include <vector>

namespace {

using namespace ck_tile::builder;
using namespace ck_tile::builder::test_utils;

TEST(ReferenceExecution, Forward_2D_FP16)
{
    // Test: Can we call Run() method on reference kernel?

    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::FORWARD,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    // Simple dimensions
    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    // Allocate minimal device memory (just to test API)
    const size_t in_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_dev(out_size);

    in_dev.SetZero();
    wei_dev.SetZero();
    out_dev.SetZero();

    // Prepare parameters for Run()
    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    // Test: Can we call Run()?
    RefKernel ref_kernel;

    // This should compile and execute:
    ref_kernel.Run(reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
                   reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
                   reinterpret_cast<ck::half_t*>(out_dev.GetDeviceBuffer()),
                   G,
                   N,
                   K,
                   C,
                   input_spatial,
                   filter_spatial,
                   output_spatial,
                   strides,
                   dilations,
                   left_pads);

    // If we get here, Run() worked!
    std::cout << "✓ Reference Forward kernel executed!" << std::endl;
    EXPECT_TRUE(true);
}

TEST(ReferenceExecution, BackwardData_2D_FP16)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_DATA,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    const size_t in_grad_size  = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_size      = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_grad_dev(in_grad_size);
    ck::DeviceMem wei_dev(wei_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    in_grad_dev.SetZero();
    wei_dev.SetZero();
    out_grad_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel ref_kernel;
    ref_kernel.Run(reinterpret_cast<ck::half_t*>(in_grad_dev.GetDeviceBuffer()),
                   reinterpret_cast<const ck::half_t*>(wei_dev.GetDeviceBuffer()),
                   reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
                   G,
                   N,
                   K,
                   C,
                   input_spatial,
                   filter_spatial,
                   output_spatial,
                   strides,
                   dilations,
                   left_pads);

    std::cout << "✓ Reference Backward Data kernel executed!" << std::endl;
    EXPECT_TRUE(true);
}

TEST(ReferenceExecution, BackwardWeight_2D_FP16)
{
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::BACKWARD_WEIGHT,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::GNHWK}}};

    constexpr auto ref_alg = ConvAlgorithm_Reference{};
    using RefKernel        = ConvBuilder<sig, ref_alg>::Instance;

    const int G = 1, N = 2, C = 4, K = 4, H = 3, W = 3;

    const size_t in_size       = G * N * C * H * W * sizeof(ck::half_t);
    const size_t wei_grad_size = G * K * C * 3 * 3 * sizeof(ck::half_t);
    const size_t out_grad_size = G * N * K * H * W * sizeof(ck::half_t);

    ck::DeviceMem in_dev(in_size);
    ck::DeviceMem wei_grad_dev(wei_grad_size);
    ck::DeviceMem out_grad_dev(out_grad_size);

    in_dev.SetZero();
    wei_grad_dev.SetZero();
    out_grad_dev.SetZero();

    std::vector<ck_tile::long_index_t> input_spatial{H, W};
    std::vector<ck_tile::long_index_t> filter_spatial{3, 3};
    std::vector<ck_tile::long_index_t> output_spatial{H, W};
    std::vector<ck_tile::long_index_t> strides{1, 1};
    std::vector<ck_tile::long_index_t> dilations{1, 1};
    std::vector<ck_tile::long_index_t> left_pads{1, 1};

    RefKernel ref_kernel;
    ref_kernel.Run(reinterpret_cast<const ck::half_t*>(in_dev.GetDeviceBuffer()),
                   reinterpret_cast<ck::half_t*>(wei_grad_dev.GetDeviceBuffer()),
                   reinterpret_cast<const ck::half_t*>(out_grad_dev.GetDeviceBuffer()),
                   G,
                   N,
                   K,
                   C,
                   input_spatial,
                   filter_spatial,
                   output_spatial,
                   strides,
                   dilations,
                   left_pads);

    std::cout << "✓ Reference Backward Weight kernel executed!" << std::endl;
    EXPECT_TRUE(true);
}

} // namespace
