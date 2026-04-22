// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for CK Tile backend using Google Test
/// Note: This test validates the dispatcher wrapper infrastructure, not actual kernel execution

#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/problem.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

namespace {

// Note: Actual CK Tile backend tests require real generated kernels and GPU hardware.
// These tests verify the dispatcher's tile backend interface and wrapper functionality
// using mock kernels instead of real tile kernels.
} // anonymous namespace

// These tests verify the tile backend can be used with mock kernels
// Real tile kernel integration would require generated CK Tile kernels

TEST(TileBackendTest, KernelKeyCreation)
{
    // Test creating a kernel key for tile backend
    KernelKey key = make_test_key(256, 256, 32, "gfx942");

    EXPECT_EQ(key.algorithm.tile_shape.m, 256);
    EXPECT_EQ(key.algorithm.tile_shape.n, 256);
    EXPECT_EQ(key.algorithm.tile_shape.k, 32);
    EXPECT_EQ(key.gfx_arch, "gfx942");
    EXPECT_EQ(key.signature.dtype_a, DataType::FP16);
}

TEST(TileBackendTest, MockKernelRegistration)
{
    // Clear registry for clean test
    Registry::instance().clear();

    KernelKey key = make_test_key(256, 256, 32, "gfx942");
    auto kernel =
        std::make_shared<MockKernelInstance>(key, "mock_tile_kernel", false); // strict divisibility

    // Register kernel
    bool registered = Registry::instance().register_kernel(kernel);
    EXPECT_TRUE(registered);

    // Lookup kernel
    std::string kernel_id = key.encode_identifier();
    auto found_kernel     = Registry::instance().lookup(kernel_id);
    EXPECT_NE(found_kernel, nullptr);
    EXPECT_EQ(found_kernel->get_name(), "mock_tile_kernel");

    Registry::instance().clear();
}

TEST(TileBackendTest, DispatcherWithMockTileKernel)
{
    // Clear registry
    Registry::instance().clear();

    // Create and register mock tile kernel
    KernelKey key = make_test_key(256, 256, 32, "gfx942");
    auto kernel =
        std::make_shared<MockKernelInstance>(key, "mock_tile_kernel", false); // strict divisibility
    Registry::instance().register_kernel(kernel);

    // Create dispatcher
    Dispatcher dispatcher;

    // Test kernel selection - divisible dimensions
    Problem problem1(512, 512, 512); // Divisible by 256, 256, 32
    auto selected1 = dispatcher.select_kernel(problem1);
    EXPECT_NE(selected1, nullptr);
    EXPECT_EQ(selected1->get_name(), "mock_tile_kernel");

    // Test with non-divisible problem
    Problem problem2(100, 200, 300); // Not divisible
    auto not_selected = dispatcher.select_kernel(problem2);
    EXPECT_EQ(not_selected, nullptr);

    Registry::instance().clear();
}

TEST(TileBackendTest, TileKernelIdentifierEncoding)
{
    KernelKey key = make_test_key(256, 256, 32, "gfx942");

    std::string id = key.encode_identifier();

    // Should contain tile dimensions
    EXPECT_NE(id.find("256x256x32"), std::string::npos);
    EXPECT_NE(id.find("2x2x1"), std::string::npos);
    EXPECT_NE(id.find("32x32x16"), std::string::npos);

    // Verify persistent flag affects identifier
    KernelKey persistent_key            = key;
    persistent_key.algorithm.persistent = true;
    EXPECT_NE(id, persistent_key.encode_identifier());
}

TEST(TileBackendTest, MultipleKernelRegistration)
{
    // Clear registry
    Registry::instance().clear();

    // Register multiple kernels with different tile sizes
    KernelKey key1 = make_test_key(256, 256, 32, "gfx942");
    auto kernel1   = std::make_shared<MockKernelInstance>(key1, "kernel_256x256x32", false);

    KernelKey key2 = make_test_key(128, 128, 64, "gfx942");
    auto kernel2   = std::make_shared<MockKernelInstance>(key2, "kernel_128x128x64", false);

    Registry::instance().register_kernel(kernel1);
    Registry::instance().register_kernel(kernel2);

    EXPECT_EQ(Registry::instance().size(), 2);

    // Verify both are accessible
    auto found1 = Registry::instance().lookup(key1.encode_identifier());
    auto found2 = Registry::instance().lookup(key2.encode_identifier());

    EXPECT_NE(found1, nullptr);
    EXPECT_NE(found2, nullptr);
    EXPECT_EQ(found1->get_name(), "kernel_256x256x32");
    EXPECT_EQ(found2->get_name(), "kernel_128x128x64");

    Registry::instance().clear();
}

TEST(TileBackendTest, TileSizeSupport)
{
    Registry::instance().clear();

    // Create kernel with 256x256x32 tiles (no padding)
    KernelKey key = make_test_key(256, 256, 32, "gfx942");
    auto kernel =
        std::make_shared<MockKernelInstance>(key, "test_kernel", false); // strict divisibility

    // Should support 512x512x512 (divisible)
    EXPECT_TRUE(kernel->supports(Problem(512, 512, 512)));

    // Should support 256x256x32 (exact match)
    EXPECT_TRUE(kernel->supports(Problem(256, 256, 32)));

    // Should NOT support 100x200x300 (not divisible)
    EXPECT_FALSE(kernel->supports(Problem(100, 200, 300)));

    // Should support 1024x1024x1024 (divisible)
    EXPECT_TRUE(kernel->supports(Problem(1024, 1024, 1024)));

    Registry::instance().clear();
}
