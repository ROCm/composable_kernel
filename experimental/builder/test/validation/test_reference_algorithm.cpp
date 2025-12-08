// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck_tile/builder/conv_builder.hpp"
#include "ck_tile/builder/types.hpp"
#include "utils/ckb_conv_test_configs.hpp"
#include <gtest/gtest.h>

namespace {

using namespace ck_tile::builder;
using namespace ck_tile::builder::test_utils;

TEST(ReferenceAlgorithm, Can_Instantiate_Reference_Forward_2D_FP16)
{
    // Define signature
    constexpr ConvSignature sig{.spatial_dim            = 2,
                                .direction              = ConvDirection::FORWARD,
                                .data_type              = DataType::FP16,
                                .accumulation_data_type = DataType::FP32,
                                .input  = {.config = {.layout = TensorLayout::GNHWC}},
                                .weight = {.config = {.layout = TensorLayout::GKYXC}},
                                .output = {.config = {.layout = TensorLayout::GNHWK}}};

    // Define reference algorithm
    constexpr auto ref_alg = ConvAlgorithm_Reference{};

    // Try to instantiate
    using RefBuilder  = ConvBuilder<sig, ref_alg>;
    using RefInstance = RefBuilder::Instance;

    // Create instance
    RefInstance ref_kernel;

    // Check GetTypeString works
    auto type_string = ref_kernel.GetTypeString();
    std::cout << "Reference kernel type: " << type_string << std::endl;

    EXPECT_GT(type_string.size(), 0);
    EXPECT_TRUE(type_string.find("Reference") != std::string::npos);
    EXPECT_TRUE(type_string.find("Forward") != std::string::npos);
}

} // namespace
