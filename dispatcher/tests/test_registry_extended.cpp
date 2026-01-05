// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Extended unit tests for Registry - covers multiple registries, merging, filtering

#include "ck_tile/dispatcher/registry.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>
#include <thread>
#include <atomic>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

// =============================================================================
// Basic Registration Tests
// =============================================================================

class RegistryBasicTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegistryBasicTest, RegisterSingleKernel)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");

    EXPECT_TRUE(Registry::instance().register_kernel(kernel));
    EXPECT_EQ(Registry::instance().size(), 1);
}

TEST_F(RegistryBasicTest, RegisterNullKernel)
{
    EXPECT_FALSE(Registry::instance().register_kernel(nullptr));
    EXPECT_EQ(Registry::instance().size(), 0);
}

TEST_F(RegistryBasicTest, RegisterMultipleKernels)
{
    for(int i = 0; i < 100; i++)
    {
        auto key    = make_test_key(100 + i);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(i));
        EXPECT_TRUE(Registry::instance().register_kernel(kernel));
    }
    EXPECT_EQ(Registry::instance().size(), 100);
}

TEST_F(RegistryBasicTest, RegisterDuplicateKey)
{
    auto key     = make_test_key(256);
    auto kernel1 = std::make_shared<MockKernelInstance>(key, "kernel1");
    auto kernel2 = std::make_shared<MockKernelInstance>(key, "kernel2");

    EXPECT_TRUE(Registry::instance().register_kernel(kernel1, Registry::Priority::Normal));

    // Same priority should not replace
    EXPECT_FALSE(Registry::instance().register_kernel(kernel2, Registry::Priority::Normal));

    auto found = Registry::instance().lookup(key);
    EXPECT_EQ(found->get_name(), "kernel1");
}

// =============================================================================
// Priority Tests
// =============================================================================

class RegistryPriorityTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegistryPriorityTest, HigherPriorityReplaces)
{
    auto key = make_test_key(256);

    auto low    = std::make_shared<MockKernelInstance>(key, "low");
    auto normal = std::make_shared<MockKernelInstance>(key, "normal");
    auto high   = std::make_shared<MockKernelInstance>(key, "high");

    EXPECT_TRUE(Registry::instance().register_kernel(low, Registry::Priority::Low));
    EXPECT_EQ(Registry::instance().lookup(key)->get_name(), "low");

    EXPECT_TRUE(Registry::instance().register_kernel(normal, Registry::Priority::Normal));
    EXPECT_EQ(Registry::instance().lookup(key)->get_name(), "normal");

    EXPECT_TRUE(Registry::instance().register_kernel(high, Registry::Priority::High));
    EXPECT_EQ(Registry::instance().lookup(key)->get_name(), "high");
}

TEST_F(RegistryPriorityTest, LowerPriorityDoesNotReplace)
{
    auto key = make_test_key(256);

    auto high = std::make_shared<MockKernelInstance>(key, "high");
    auto low  = std::make_shared<MockKernelInstance>(key, "low");

    EXPECT_TRUE(Registry::instance().register_kernel(high, Registry::Priority::High));
    EXPECT_FALSE(Registry::instance().register_kernel(low, Registry::Priority::Low));

    EXPECT_EQ(Registry::instance().lookup(key)->get_name(), "high");
}

TEST_F(RegistryPriorityTest, SamePriorityDoesNotReplace)
{
    auto key = make_test_key(256);

    auto first  = std::make_shared<MockKernelInstance>(key, "first");
    auto second = std::make_shared<MockKernelInstance>(key, "second");

    EXPECT_TRUE(Registry::instance().register_kernel(first, Registry::Priority::Normal));
    EXPECT_FALSE(Registry::instance().register_kernel(second, Registry::Priority::Normal));

    EXPECT_EQ(Registry::instance().lookup(key)->get_name(), "first");
}

// =============================================================================
// Lookup Tests
// =============================================================================

class RegistryLookupTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        // Register several kernels
        for(int tile : {128, 256, 512})
        {
            auto key = make_test_key(tile);
            auto kernel =
                std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(tile));
            Registry::instance().register_kernel(kernel);
        }
    }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegistryLookupTest, LookupByKey)
{
    auto key   = make_test_key(256);
    auto found = Registry::instance().lookup(key);

    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_name(), "kernel_256");
}

TEST_F(RegistryLookupTest, LookupByIdentifier)
{
    auto key       = make_test_key(256);
    std::string id = key.encode_identifier();

    auto found = Registry::instance().lookup(id);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_name(), "kernel_256");
}

TEST_F(RegistryLookupTest, LookupNonExistent)
{
    auto key = make_test_key(1024); // Not registered
    EXPECT_EQ(Registry::instance().lookup(key), nullptr);
    EXPECT_EQ(Registry::instance().lookup("nonexistent_id"), nullptr);
}

TEST_F(RegistryLookupTest, LookupEmptyIdentifier)
{
    EXPECT_EQ(Registry::instance().lookup(""), nullptr);
}

// =============================================================================
// Filter Tests
// =============================================================================

class RegistryFilterTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        // Register kernels with various tile sizes
        for(int tile : {64, 128, 256, 512, 1024})
        {
            auto key              = make_test_key(tile);
            key.signature.dtype_a = (tile < 256) ? DataType::FP16 : DataType::BF16;
            auto kernel =
                std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(tile));
            Registry::instance().register_kernel(kernel);
        }
    }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegistryFilterTest, FilterByTileSize)
{
    auto large = Registry::instance().filter(
        [](const KernelInstance& k) { return k.get_key().algorithm.tile_shape.m >= 256; });

    EXPECT_EQ(large.size(), 3); // 256, 512, 1024
}

TEST_F(RegistryFilterTest, FilterByDataType)
{
    auto fp16 = Registry::instance().filter(
        [](const KernelInstance& k) { return k.get_key().signature.dtype_a == DataType::FP16; });

    EXPECT_EQ(fp16.size(), 2); // 64, 128
}

TEST_F(RegistryFilterTest, FilterMatchesNone)
{
    auto none = Registry::instance().filter(
        [](const KernelInstance& k) { return k.get_key().algorithm.tile_shape.m > 2048; });

    EXPECT_EQ(none.size(), 0);
}

TEST_F(RegistryFilterTest, FilterMatchesAll)
{
    auto all = Registry::instance().filter([](const KernelInstance& k) { return true; });

    EXPECT_EQ(all.size(), 5);
}

// =============================================================================
// Multiple Registries Tests
// =============================================================================

class MultipleRegistriesTest : public ::testing::Test
{
    protected:
    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(MultipleRegistriesTest, CreateIndependentRegistries)
{
    Registry reg1;
    Registry reg2;

    reg1.set_name("registry1");
    reg2.set_name("registry2");

    auto key1 = make_test_key(256);
    auto key2 = make_test_key(512);

    reg1.register_kernel(std::make_shared<MockKernelInstance>(key1, "kernel1"));
    reg2.register_kernel(std::make_shared<MockKernelInstance>(key2, "kernel2"));

    EXPECT_EQ(reg1.size(), 1);
    EXPECT_EQ(reg2.size(), 1);

    EXPECT_NE(reg1.lookup(key1), nullptr);
    EXPECT_EQ(reg1.lookup(key2), nullptr);

    EXPECT_EQ(reg2.lookup(key1), nullptr);
    EXPECT_NE(reg2.lookup(key2), nullptr);
}

TEST_F(MultipleRegistriesTest, RegistryNaming)
{
    Registry reg;
    reg.set_name("my_custom_registry");

    EXPECT_EQ(reg.get_name(), "my_custom_registry");
}

TEST_F(MultipleRegistriesTest, MergeRegistries)
{
    Registry reg1;
    Registry reg2;

    auto key1 = make_test_key(128);
    auto key2 = make_test_key(256);
    auto key3 = make_test_key(512);

    reg1.register_kernel(std::make_shared<MockKernelInstance>(key1, "k1"));
    reg1.register_kernel(std::make_shared<MockKernelInstance>(key2, "k2"));

    reg2.register_kernel(std::make_shared<MockKernelInstance>(key3, "k3"));

    Registry combined;
    combined.merge_from(reg1, Registry::Priority::Normal);
    combined.merge_from(reg2, Registry::Priority::Normal);

    EXPECT_EQ(combined.size(), 3);
    EXPECT_NE(combined.lookup(key1), nullptr);
    EXPECT_NE(combined.lookup(key2), nullptr);
    EXPECT_NE(combined.lookup(key3), nullptr);
}

TEST_F(MultipleRegistriesTest, MergeWithPriorityConflict)
{
    Registry reg1;
    Registry reg2;

    auto key = make_test_key(256);

    reg1.register_kernel(std::make_shared<MockKernelInstance>(key, "from_reg1"));
    reg2.register_kernel(std::make_shared<MockKernelInstance>(key, "from_reg2"));

    Registry combined;
    combined.merge_from(reg1, Registry::Priority::Low);
    combined.merge_from(reg2, Registry::Priority::High);

    EXPECT_EQ(combined.size(), 1);
    EXPECT_EQ(combined.lookup(key)->get_name(), "from_reg2");
}

TEST_F(MultipleRegistriesTest, SingletonIndependence)
{
    Registry local_reg;
    local_reg.set_name("local");

    auto key1 = make_test_key(256);
    auto key2 = make_test_key(512);

    local_reg.register_kernel(std::make_shared<MockKernelInstance>(key1, "local_kernel"));
    Registry::instance().register_kernel(
        std::make_shared<MockKernelInstance>(key2, "global_kernel"));

    EXPECT_EQ(local_reg.size(), 1);
    EXPECT_EQ(Registry::instance().size(), 1);

    EXPECT_NE(local_reg.lookup(key1), nullptr);
    EXPECT_EQ(local_reg.lookup(key2), nullptr);

    EXPECT_EQ(Registry::instance().lookup(key1), nullptr);
    EXPECT_NE(Registry::instance().lookup(key2), nullptr);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

class RegistryThreadSafetyTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegistryThreadSafetyTest, ConcurrentRegistrations)
{
    const int num_threads        = 10;
    const int kernels_per_thread = 100;

    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for(int t = 0; t < num_threads; t++)
    {
        threads.emplace_back([t, kernels_per_thread, &success_count]() {
            for(int k = 0; k < kernels_per_thread; k++)
            {
                int tile = t * 1000 + k; // Unique tile size
                auto key = make_test_key(tile);
                auto kernel =
                    std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(tile));

                if(Registry::instance().register_kernel(kernel))
                {
                    success_count++;
                }
            }
        });
    }

    for(auto& t : threads)
    {
        t.join();
    }

    EXPECT_EQ(success_count.load(), num_threads * kernels_per_thread);
    EXPECT_EQ(Registry::instance().size(), num_threads * kernels_per_thread);
}

TEST_F(RegistryThreadSafetyTest, ConcurrentLookups)
{
    // Pre-register kernels
    for(int i = 0; i < 100; i++)
    {
        auto key    = make_test_key(i);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(i));
        Registry::instance().register_kernel(kernel);
    }

    const int num_threads        = 10;
    const int lookups_per_thread = 1000;
    std::atomic<int> found_count{0};

    std::vector<std::thread> threads;
    for(int t = 0; t < num_threads; t++)
    {
        threads.emplace_back([lookups_per_thread, &found_count]() {
            for(int k = 0; k < lookups_per_thread; k++)
            {
                auto key = make_test_key(k % 100);
                if(Registry::instance().lookup(key) != nullptr)
                {
                    found_count++;
                }
            }
        });
    }

    for(auto& t : threads)
    {
        t.join();
    }

    EXPECT_EQ(found_count.load(), num_threads * lookups_per_thread);
}

// =============================================================================
// Clear and Size Tests
// =============================================================================

class RegistryClearTest : public ::testing::Test
{
    protected:
    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegistryClearTest, ClearEmptyRegistry)
{
    Registry::instance().clear();
    EXPECT_EQ(Registry::instance().size(), 0);

    Registry::instance().clear(); // Should not crash
    EXPECT_EQ(Registry::instance().size(), 0);
}

TEST_F(RegistryClearTest, ClearNonEmptyRegistry)
{
    for(int i = 0; i < 10; i++)
    {
        auto key    = make_test_key(i);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
        Registry::instance().register_kernel(kernel);
    }

    EXPECT_EQ(Registry::instance().size(), 10);

    Registry::instance().clear();
    EXPECT_EQ(Registry::instance().size(), 0);
}

TEST_F(RegistryClearTest, RegisterAfterClear)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");

    Registry::instance().register_kernel(kernel);
    EXPECT_EQ(Registry::instance().size(), 1);

    Registry::instance().clear();
    EXPECT_EQ(Registry::instance().size(), 0);

    Registry::instance().register_kernel(kernel);
    EXPECT_EQ(Registry::instance().size(), 1);
}

// =============================================================================
// GetAll Tests
// =============================================================================

class RegistryGetAllTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegistryGetAllTest, GetAllEmpty)
{
    auto all = Registry::instance().get_all();
    EXPECT_EQ(all.size(), 0);
}

TEST_F(RegistryGetAllTest, GetAllMultiple)
{
    for(int i = 0; i < 5; i++)
    {
        auto key    = make_test_key(100 + i);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(i));
        Registry::instance().register_kernel(kernel);
    }

    auto all = Registry::instance().get_all();
    EXPECT_EQ(all.size(), 5);
}
