// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Regression tests for known issues and edge cases.
 * Add a new test here whenever a bug is fixed to prevent regression.
 */

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/kernel_key.hpp"
#include "ck_tile/dispatcher/problem.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>
#include <sstream>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;
using SelectionStrategy = Dispatcher::SelectionStrategy;

// =============================================================================
// Issue: Uninitialized 'grouped' field in KernelKey caused JSON corruption
// Fix: Ensure all fields in make_test_key() are initialized
// =============================================================================

class RegressionGroupedFieldTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegressionGroupedFieldTest, GroupedFieldInitialized)
{
    KernelKey key = make_test_key(256);

    // grouped should be explicitly initialized
    EXPECT_FALSE(key.signature.grouped);

    // Encoding should not crash or produce garbage
    std::string id = key.encode_identifier();
    EXPECT_FALSE(id.empty());

    // ID should not contain garbage characters
    for(char c : id)
    {
        EXPECT_TRUE(std::isprint(c) || c == '_' || c == '-')
            << "Invalid character in identifier: " << static_cast<int>(c);
    }
}

TEST_F(RegressionGroupedFieldTest, GroupedFieldInJSON)
{
    KernelKey key         = make_test_key(256);
    key.signature.grouped = false;

    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");
    Registry::instance().register_kernel(kernel);

    // Export to JSON
    std::string json = Registry::instance().export_json(true);

    // JSON should be valid (not contain null bytes or garbage)
    EXPECT_FALSE(json.empty());

    // Should contain the grouped field with proper value
    EXPECT_NE(json.find("\"grouped\""), std::string::npos);
    EXPECT_NE(json.find("false"), std::string::npos);
}

// =============================================================================
// Issue: Priority comparison was incorrect
// Fix: Higher priority should replace lower, same priority should not replace
// =============================================================================

class RegressionPriorityTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegressionPriorityTest, LowThenHighReplaces)
{
    auto key  = make_test_key(256);
    auto low  = std::make_shared<MockKernelInstance>(key, "low");
    auto high = std::make_shared<MockKernelInstance>(key, "high");

    EXPECT_TRUE(Registry::instance().register_kernel(low, Registry::Priority::Low));
    EXPECT_TRUE(Registry::instance().register_kernel(high, Registry::Priority::High));

    auto found = Registry::instance().lookup(key);
    EXPECT_EQ(found->get_name(), "high");
}

TEST_F(RegressionPriorityTest, HighThenLowDoesNotReplace)
{
    auto key  = make_test_key(256);
    auto high = std::make_shared<MockKernelInstance>(key, "high");
    auto low  = std::make_shared<MockKernelInstance>(key, "low");

    EXPECT_TRUE(Registry::instance().register_kernel(high, Registry::Priority::High));
    EXPECT_FALSE(Registry::instance().register_kernel(low, Registry::Priority::Low));

    auto found = Registry::instance().lookup(key);
    EXPECT_EQ(found->get_name(), "high");
}

TEST_F(RegressionPriorityTest, SamePriorityDoesNotReplace)
{
    auto key    = make_test_key(256);
    auto first  = std::make_shared<MockKernelInstance>(key, "first");
    auto second = std::make_shared<MockKernelInstance>(key, "second");

    EXPECT_TRUE(Registry::instance().register_kernel(first, Registry::Priority::Normal));
    EXPECT_FALSE(Registry::instance().register_kernel(second, Registry::Priority::Normal));

    auto found = Registry::instance().lookup(key);
    EXPECT_EQ(found->get_name(), "first");
}

// =============================================================================
// Issue: Empty heuristic caused crash
// Fix: Fall back to FirstFit when heuristic returns empty or invalid results
// =============================================================================

class RegressionHeuristicTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        auto key    = make_test_key(256);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
        Registry::instance().register_kernel(kernel);
    }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegressionHeuristicTest, EmptyHeuristicFallback)
{
    Dispatcher dispatcher;

    dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
        return {}; // Empty
    });
    dispatcher.set_strategy(SelectionStrategy::Heuristic);

    Problem problem(1024, 1024, 1024);

    // Should not crash, should fall back to FirstFit
    auto selected = dispatcher.select_kernel(problem);
    EXPECT_NE(selected, nullptr);
}

TEST_F(RegressionHeuristicTest, AllInvalidHeuristicFallback)
{
    Dispatcher dispatcher;

    dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
        return {"invalid1", "invalid2", "invalid3"};
    });
    dispatcher.set_strategy(SelectionStrategy::Heuristic);

    Problem problem(1024, 1024, 1024);

    // Should not crash, should fall back to FirstFit
    auto selected = dispatcher.select_kernel(problem);
    EXPECT_NE(selected, nullptr);
}

TEST_F(RegressionHeuristicTest, NullHeuristicSafe)
{
    Dispatcher dispatcher;

    // Don't set any heuristic
    dispatcher.set_strategy(SelectionStrategy::Heuristic);

    Problem problem(1024, 1024, 1024);

    // Should not crash
    auto selected = dispatcher.select_kernel(problem);
    // Behavior depends on implementation - may return nullptr or fall back
}

// =============================================================================
// Issue: Lookup by empty string caused crash or undefined behavior
// =============================================================================

class RegressionLookupTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegressionLookupTest, EmptyStringLookup)
{
    EXPECT_EQ(Registry::instance().lookup(""), nullptr);
}

TEST_F(RegressionLookupTest, VeryLongStringLookup)
{
    std::string very_long(10000, 'x');
    EXPECT_EQ(Registry::instance().lookup(very_long), nullptr);
}

TEST_F(RegressionLookupTest, SpecialCharactersLookup)
{
    EXPECT_EQ(Registry::instance().lookup("kernel\0name"), nullptr);
    EXPECT_EQ(Registry::instance().lookup("kernel\nname"), nullptr);
    EXPECT_EQ(Registry::instance().lookup("kernel\tname"), nullptr);
}

// =============================================================================
// Issue: Problem with zero dimensions passed to dispatcher
// =============================================================================

class RegressionProblemTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        auto key    = make_test_key(256);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
        Registry::instance().register_kernel(kernel);
    }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegressionProblemTest, ZeroMDimension)
{
    Problem problem;
    problem.M = 0;
    problem.N = 1024;
    problem.K = 1024;

    EXPECT_FALSE(problem.is_valid());
}

TEST_F(RegressionProblemTest, ZeroNDimension)
{
    Problem problem;
    problem.M = 1024;
    problem.N = 0;
    problem.K = 1024;

    EXPECT_FALSE(problem.is_valid());
}

TEST_F(RegressionProblemTest, ZeroKDimension)
{
    Problem problem;
    problem.M = 1024;
    problem.N = 1024;
    problem.K = 0;

    EXPECT_FALSE(problem.is_valid());
}

// =============================================================================
// Issue: Dispatcher run with null pointers
// =============================================================================

class RegressionNullPointerTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        auto key    = make_test_key(256);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
        Registry::instance().register_kernel(kernel);
    }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegressionNullPointerTest, RunWithNullPointers)
{
    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    // Mock kernel doesn't use pointers, so this should work
    float time = dispatcher.run(nullptr, nullptr, nullptr, problem);

    // Mock returns 1.0f
    EXPECT_FLOAT_EQ(time, 1.0f);
}

// =============================================================================
// Issue: Thread safety - concurrent access to singleton
// =============================================================================

class RegressionThreadSafetyTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegressionThreadSafetyTest, SingletonAddressStable)
{
    Registry* addr1 = &Registry::instance();
    Registry* addr2 = &Registry::instance();
    Registry* addr3 = &Registry::instance();

    EXPECT_EQ(addr1, addr2);
    EXPECT_EQ(addr2, addr3);
}

// =============================================================================
// Issue: encode_identifier could produce duplicate IDs for different configs
// =============================================================================

class RegressionIdentifierTest : public ::testing::Test
{
};

TEST_F(RegressionIdentifierTest, DifferentConfigsDifferentIDs)
{
    // Create two keys that differ only in one field
    KernelKey key1            = make_test_key(256);
    KernelKey key2            = make_test_key(256);
    key2.algorithm.persistent = true; // Only difference

    std::string id1 = key1.encode_identifier();
    std::string id2 = key2.encode_identifier();

    EXPECT_NE(id1, id2) << "Different persistent flag should produce different IDs";
}

TEST_F(RegressionIdentifierTest, DifferentTileShapesDifferentIDs)
{
    KernelKey key1 = make_test_key(128, 128, 32);
    KernelKey key2 = make_test_key(256, 256, 32);

    EXPECT_NE(key1.encode_identifier(), key2.encode_identifier());
}

TEST_F(RegressionIdentifierTest, DifferentWarpConfigsDifferentIDs)
{
    KernelKey key1            = make_test_key(256);
    key1.algorithm.wave_shape = {2, 2, 1};

    KernelKey key2            = make_test_key(256);
    key2.algorithm.wave_shape = {4, 1, 1};

    EXPECT_NE(key1.encode_identifier(), key2.encode_identifier());
}

// =============================================================================
// Issue: Negative k_batch could cause issues
// =============================================================================

class RegressionKBatchTest : public ::testing::Test
{
};

TEST_F(RegressionKBatchTest, ZeroKBatchInvalid)
{
    Problem problem(1024, 1024, 1024);
    problem.k_batch = 0;

    EXPECT_FALSE(problem.is_valid());
}

TEST_F(RegressionKBatchTest, NegativeKBatchInvalid)
{
    Problem problem(1024, 1024, 1024);
    problem.k_batch = -1;

    EXPECT_FALSE(problem.is_valid());
}

TEST_F(RegressionKBatchTest, LargeKBatchValid)
{
    Problem problem(1024, 1024, 1024);
    problem.k_batch = 1000;

    EXPECT_TRUE(problem.is_valid());
}

// =============================================================================
// Issue: Filter returning shared_ptr leaks
// =============================================================================

class RegressionFilterTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        for(int i = 0; i < 10; i++)
        {
            auto key    = make_test_key(100 + i);
            auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(i));
            Registry::instance().register_kernel(kernel);
        }
    }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegressionFilterTest, FilterResultsAreValid)
{
    auto results = Registry::instance().filter(
        [](const KernelInstance& k) { return k.get_key().algorithm.tile_shape.m >= 105; });

    EXPECT_EQ(results.size(), 5);

    for(const auto& kernel : results)
    {
        EXPECT_NE(kernel, nullptr);
        EXPECT_GE(kernel->get_key().algorithm.tile_shape.m, 105);
    }
}

// =============================================================================
// Issue: Double clear() could cause issues
// =============================================================================

class RegressionDoubleClearTest : public ::testing::Test
{
};

TEST_F(RegressionDoubleClearTest, DoubleClearSafe)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");

    Registry::instance().register_kernel(kernel);
    EXPECT_EQ(Registry::instance().size(), 1);

    Registry::instance().clear();
    EXPECT_EQ(Registry::instance().size(), 0);

    Registry::instance().clear(); // Second clear
    EXPECT_EQ(Registry::instance().size(), 0);

    // Should still work after double clear
    Registry::instance().register_kernel(kernel);
    EXPECT_EQ(Registry::instance().size(), 1);
}

// =============================================================================
// Issue: Multiple dispatchers with same registry
// =============================================================================

class RegressionMultiDispatcherTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        auto key    = make_test_key(256);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
        Registry::instance().register_kernel(kernel);
    }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(RegressionMultiDispatcherTest, MultipleDispatchersShareRegistry)
{
    Dispatcher d1;
    Dispatcher d2;
    Dispatcher d3;

    Problem problem(1024, 1024, 1024);

    auto k1 = d1.select_kernel(problem);
    auto k2 = d2.select_kernel(problem);
    auto k3 = d3.select_kernel(problem);

    // All should select the same kernel
    EXPECT_NE(k1, nullptr);
    EXPECT_EQ(k1, k2);
    EXPECT_EQ(k2, k3);
}
