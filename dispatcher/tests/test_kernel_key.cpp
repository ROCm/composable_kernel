// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for KernelKey using Google Test

#include "ck_tile/dispatcher/kernel_key.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

TEST(KernelKeyTest, Construction)
{
    KernelKey key;
    key.signature.dtype_a        = DataType::FP16;
    key.signature.dtype_b        = DataType::FP16;
    key.signature.dtype_c        = DataType::FP16;
    key.signature.dtype_acc      = DataType::FP32;
    key.signature.elementwise_op = "PassThrough";
    key.signature.num_d_tensors  = 0;

    key.algorithm.tile_shape.m = 256;
    key.algorithm.tile_shape.n = 256;
    key.algorithm.tile_shape.k = 32;

    key.gfx_arch = "gfx942";

    EXPECT_EQ(key.signature.dtype_a, DataType::FP16);
    EXPECT_EQ(key.algorithm.tile_shape.m, 256);
    EXPECT_EQ(key.gfx_arch, "gfx942");
}

TEST(KernelKeyTest, Equality)
{
    // Use helper function to ensure all fields are initialized
    KernelKey key1 = make_test_key(256, 256, 32, "gfx942");
    KernelKey key2 = make_test_key(256, 256, 32, "gfx942");

    EXPECT_EQ(key1, key2);
    EXPECT_FALSE(key1 != key2);

    // Change one value
    KernelKey key3 = make_test_key(128, 256, 32, "gfx942");
    EXPECT_NE(key1, key3);
    EXPECT_FALSE(key1 == key3);
}

TEST(KernelKeyTest, EncodeIdentifier)
{
    KernelKey key;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.algorithm.tile_shape.m        = 256;
    key.algorithm.tile_shape.n        = 256;
    key.algorithm.tile_shape.k        = 32;
    key.algorithm.wave_shape.m        = 2;
    key.algorithm.wave_shape.n        = 2;
    key.algorithm.wave_shape.k        = 1;
    key.algorithm.warp_tile_shape.m   = 32;
    key.algorithm.warp_tile_shape.n   = 32;
    key.algorithm.warp_tile_shape.k   = 16;
    key.algorithm.persistent          = true;
    key.algorithm.preshuffle          = false;
    key.signature.structured_sparsity = false;

    std::string id = key.encode_identifier();

    // Check that identifier contains expected components
    EXPECT_NE(id.find("256x256x32"), std::string::npos); // tile shape
    EXPECT_NE(id.find("2x2x1"), std::string::npos);      // wave shape
    EXPECT_NE(id.find("32x32x16"), std::string::npos);   // warp tile shape

    // Verify persistent flag is encoded by toggling it and asserting the
    // identifier changes. Robust to encoding spelling changes.
    KernelKey non_persistent_key            = key;
    non_persistent_key.algorithm.persistent = false;
    EXPECT_NE(id, non_persistent_key.encode_identifier());
}

TEST(KernelKeyTest, EncodeIdentifierWithFusion)
{
    KernelKey key;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "Relu";
    key.signature.num_d_tensors       = 2;
    key.algorithm.tile_shape.m        = 128;
    key.algorithm.tile_shape.n        = 128;
    key.algorithm.tile_shape.k        = 64;
    key.algorithm.wave_shape.m        = 2;
    key.algorithm.wave_shape.n        = 2;
    key.algorithm.wave_shape.k        = 1;
    key.algorithm.warp_tile_shape.m   = 16;
    key.algorithm.warp_tile_shape.n   = 16;
    key.algorithm.warp_tile_shape.k   = 32;
    key.algorithm.persistent          = false;
    key.signature.structured_sparsity = false;

    std::string id = key.encode_identifier();

    // Check fusion-specific components
    EXPECT_NE(id.find("Relu"), std::string::npos);
    EXPECT_NE(id.find("_d2"), std::string::npos);

    // Verify persistent flag is encoded by toggling it and asserting the
    // identifier changes. Robust to encoding spelling changes.
    KernelKey persistent_key            = key;
    persistent_key.algorithm.persistent = true;
    EXPECT_NE(id, persistent_key.encode_identifier());
}

TEST(KernelKeyTest, EncodeIdentifierWithSplitK)
{
    KernelKey key;
    key.signature.split_k             = 4;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.algorithm.tile_shape.m        = 256;
    key.algorithm.tile_shape.n        = 256;
    key.algorithm.tile_shape.k        = 32;
    key.algorithm.wave_shape.m        = 2;
    key.algorithm.wave_shape.n        = 2;
    key.algorithm.wave_shape.k        = 1;
    key.algorithm.warp_tile_shape.m   = 32;
    key.algorithm.warp_tile_shape.n   = 32;
    key.algorithm.warp_tile_shape.k   = 16;
    key.algorithm.persistent          = false;
    key.signature.structured_sparsity = false;

    std::string id = key.encode_identifier();

    EXPECT_NE(id.find("_splitk4"), std::string::npos);
}

TEST(KernelKeyTest, EncodeIdentifierWithSparsity)
{
    KernelKey key;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = true;
    key.algorithm.tile_shape.m        = 256;
    key.algorithm.tile_shape.n        = 256;
    key.algorithm.tile_shape.k        = 32;
    key.algorithm.wave_shape.m        = 2;
    key.algorithm.wave_shape.n        = 2;
    key.algorithm.wave_shape.k        = 1;
    key.algorithm.warp_tile_shape.m   = 32;
    key.algorithm.warp_tile_shape.n   = 32;
    key.algorithm.warp_tile_shape.k   = 16;
    key.algorithm.persistent          = false;

    std::string id = key.encode_identifier();

    EXPECT_NE(id.find("_sparse"), std::string::npos);
}
