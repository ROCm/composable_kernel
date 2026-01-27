// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for Registry using Google Test

#include "ck_tile/dispatcher/registry.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

TEST(RegistryTest, Registration)
{
    Registry& registry = Registry::instance();
    registry.clear();

    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");

    bool registered = registry.register_kernel(kernel);
    EXPECT_TRUE(registered);
    EXPECT_EQ(registry.size(), 1);
}

TEST(RegistryTest, Lookup)
{
    Registry& registry = Registry::instance();
    registry.clear();

    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");
    registry.register_kernel(kernel);

    // Lookup by key
    auto found = registry.lookup(key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_name(), "test_kernel");

    // Lookup by identifier
    std::string id = key.encode_identifier();
    auto found2    = registry.lookup(id);
    ASSERT_NE(found2, nullptr);
    EXPECT_EQ(found2->get_name(), "test_kernel");

    // Lookup non-existent
    auto key2      = make_test_key(128);
    auto not_found = registry.lookup(key2);
    EXPECT_EQ(not_found, nullptr);
}

TEST(RegistryTest, Priority)
{
    Registry& registry = Registry::instance();
    registry.clear();

    auto key     = make_test_key(256);
    auto kernel1 = std::make_shared<MockKernelInstance>(key, "kernel_low");
    auto kernel2 = std::make_shared<MockKernelInstance>(key, "kernel_high");

    // Register with low priority
    registry.register_kernel(kernel1, Registry::Priority::Low);

    // Try to register with normal priority (should replace)
    bool replaced = registry.register_kernel(kernel2, Registry::Priority::Normal);
    EXPECT_TRUE(replaced);

    auto found = registry.lookup(key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_name(), "kernel_high");

    // Try to register with low priority again (should fail)
    auto kernel3      = std::make_shared<MockKernelInstance>(key, "kernel_low2");
    bool not_replaced = registry.register_kernel(kernel3, Registry::Priority::Low);
    EXPECT_FALSE(not_replaced);

    found = registry.lookup(key);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_name(), "kernel_high");
}

TEST(RegistryTest, GetAll)
{
    Registry& registry = Registry::instance();
    registry.clear();

    auto key1    = make_test_key(256);
    auto key2    = make_test_key(128);
    auto kernel1 = std::make_shared<MockKernelInstance>(key1, "kernel1");
    auto kernel2 = std::make_shared<MockKernelInstance>(key2, "kernel2");

    registry.register_kernel(kernel1);
    registry.register_kernel(kernel2);

    auto all = registry.get_all();
    EXPECT_EQ(all.size(), 2);
}

TEST(RegistryTest, Filter)
{
    Registry& registry = Registry::instance();
    registry.clear();

    // Create kernels with different tile sizes
    for(int tile_m : {128, 256, 512})
    {
        auto key    = make_test_key(tile_m);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(tile_m));
        registry.register_kernel(kernel);
    }

    // Filter for large tiles (>= 256)
    auto large_tiles = registry.filter(
        [](const KernelInstance& k) { return k.get_key().algorithm.tile_shape.m >= 256; });

    EXPECT_EQ(large_tiles.size(), 2);
}

TEST(RegistryTest, Clear)
{
    Registry& registry = Registry::instance();
    registry.clear();

    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");
    registry.register_kernel(kernel);

    EXPECT_EQ(registry.size(), 1);

    registry.clear();
    EXPECT_EQ(registry.size(), 0);
}

TEST(RegistryTest, MultipleKernels)
{
    Registry& registry = Registry::instance();
    registry.clear();

    // Register multiple kernels
    for(int i = 0; i < 10; ++i)
    {
        auto key    = make_test_key(256 + i);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(i));
        registry.register_kernel(kernel);
    }

    EXPECT_EQ(registry.size(), 10);

    // Verify all can be looked up
    for(int i = 0; i < 10; ++i)
    {
        auto key   = make_test_key(256 + i);
        auto found = registry.lookup(key);
        ASSERT_NE(found, nullptr);
        EXPECT_EQ(found->get_name(), "kernel_" + std::to_string(i));
    }
}

TEST(RegistryTest, Singleton)
{
    Registry& reg1 = Registry::instance();
    Registry& reg2 = Registry::instance();

    // Should be the same instance
    EXPECT_EQ(&reg1, &reg2);
}
