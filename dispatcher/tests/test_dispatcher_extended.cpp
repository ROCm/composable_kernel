// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Extended unit tests for Dispatcher - covers selection strategies, heuristics, edge cases

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>
#include <algorithm>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;
using SelectionStrategy = Dispatcher::SelectionStrategy;

// =============================================================================
// Basic Dispatcher Tests
// =============================================================================

class DispatcherBasicTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(DispatcherBasicTest, DefaultConstruction)
{
    Dispatcher dispatcher;
    // Should not crash
    SUCCEED();
}

TEST_F(DispatcherBasicTest, SelectKernelEmpty)
{
    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    auto kernel = dispatcher.select_kernel(problem);
    EXPECT_EQ(kernel, nullptr);
}

TEST_F(DispatcherBasicTest, SelectKernelSingle)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "test_kernel");
    Registry::instance().register_kernel(kernel);

    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    auto selected = dispatcher.select_kernel(problem);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "test_kernel");
}

TEST_F(DispatcherBasicTest, SelectKernelMultiple)
{
    // Register multiple kernels
    for(int tile : {128, 256, 512})
    {
        auto key    = make_test_key(tile);
        auto kernel = std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(tile));
        Registry::instance().register_kernel(kernel);
    }

    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    auto selected = dispatcher.select_kernel(problem);
    ASSERT_NE(selected, nullptr);
    // Should select one of the registered kernels
    EXPECT_TRUE(selected->get_name() == "kernel_128" || selected->get_name() == "kernel_256" ||
                selected->get_name() == "kernel_512");
}

// =============================================================================
// Selection Strategy Tests
// =============================================================================

class SelectionStrategyTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        // Register kernels with different tile sizes
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

TEST_F(SelectionStrategyTest, FirstFitStrategy)
{
    Dispatcher dispatcher;
    dispatcher.set_strategy(SelectionStrategy::FirstFit);

    Problem problem(1024, 1024, 1024);
    auto selected = dispatcher.select_kernel(problem);

    ASSERT_NE(selected, nullptr);
    // FirstFit returns first matching kernel
}

TEST_F(SelectionStrategyTest, HeuristicStrategy)
{
    Dispatcher dispatcher;

    // Set heuristic that prefers larger tiles for large problems
    dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
        if(p.M >= 1024 && p.N >= 1024)
        {
            // For large problems, prefer 512 tile
            auto key = make_test_key(512);
            return {key.encode_identifier()};
        }
        // For small problems, prefer 128 tile
        auto key = make_test_key(128);
        return {key.encode_identifier()};
    });

    dispatcher.set_strategy(SelectionStrategy::Heuristic);

    // Large problem should get 512 tile
    Problem large_problem(2048, 2048, 2048);
    auto selected = dispatcher.select_kernel(large_problem);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "kernel_512");

    // Small problem should get 128 tile
    Problem small_problem(256, 256, 256);
    selected = dispatcher.select_kernel(small_problem);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "kernel_128");
}

TEST_F(SelectionStrategyTest, HeuristicWithFallback)
{
    Dispatcher dispatcher;

    // Heuristic returns non-existent kernel first, then valid one
    dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
        auto key = make_test_key(256);
        return {"nonexistent_kernel", key.encode_identifier()};
    });

    dispatcher.set_strategy(SelectionStrategy::Heuristic);

    Problem problem(1024, 1024, 1024);
    auto selected = dispatcher.select_kernel(problem);

    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "kernel_256");
}

TEST_F(SelectionStrategyTest, SwitchBetweenStrategies)
{
    Dispatcher dispatcher;

    // Start with FirstFit
    dispatcher.set_strategy(SelectionStrategy::FirstFit);

    Problem problem(1024, 1024, 1024);
    auto selected1 = dispatcher.select_kernel(problem);
    ASSERT_NE(selected1, nullptr);

    // Switch to Heuristic
    dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
        auto key = make_test_key(256);
        return {key.encode_identifier()};
    });
    dispatcher.set_strategy(SelectionStrategy::Heuristic);

    auto selected2 = dispatcher.select_kernel(problem);
    ASSERT_NE(selected2, nullptr);
}

// =============================================================================
// Heuristic Function Tests
// =============================================================================

class HeuristicTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        for(int tile : {64, 128, 256, 512})
        {
            auto key = make_test_key(tile);
            auto kernel =
                std::make_shared<MockKernelInstance>(key, "kernel_" + std::to_string(tile));
            Registry::instance().register_kernel(kernel);
        }
    }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(HeuristicTest, SizeBasedHeuristic)
{
    Dispatcher dispatcher;

    dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
        std::vector<std::string> candidates;

        // Problem-size based selection
        int size = p.M * p.N * p.K;

        if(size >= 1024 * 1024 * 1024)
        {
            candidates.push_back(make_test_key(512).encode_identifier());
            candidates.push_back(make_test_key(256).encode_identifier());
        }
        else if(size >= 256 * 256 * 256)
        {
            candidates.push_back(make_test_key(256).encode_identifier());
            candidates.push_back(make_test_key(128).encode_identifier());
        }
        else
        {
            candidates.push_back(make_test_key(64).encode_identifier());
            candidates.push_back(make_test_key(128).encode_identifier());
        }

        return candidates;
    });

    dispatcher.set_strategy(SelectionStrategy::Heuristic);

    // Large problem
    auto selected = dispatcher.select_kernel(Problem(1024, 1024, 1024));
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "kernel_512");

    // Medium problem
    selected = dispatcher.select_kernel(Problem(256, 256, 256));
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "kernel_256");

    // Small problem
    selected = dispatcher.select_kernel(Problem(64, 64, 64));
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "kernel_64");
}

TEST_F(HeuristicTest, EmptyHeuristicFallsBackToFirstFit)
{
    Dispatcher dispatcher;

    dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
        return {}; // Empty list
    });

    dispatcher.set_strategy(SelectionStrategy::Heuristic);

    Problem problem(1024, 1024, 1024);
    auto selected = dispatcher.select_kernel(problem);

    // Should fall back to FirstFit
    ASSERT_NE(selected, nullptr);
}

TEST_F(HeuristicTest, InvalidHeuristicFallsBackToFirstFit)
{
    Dispatcher dispatcher;

    dispatcher.set_heuristic([](const Problem& p) -> std::vector<std::string> {
        return {"invalid_kernel_1", "invalid_kernel_2"}; // All invalid
    });

    dispatcher.set_strategy(SelectionStrategy::Heuristic);

    Problem problem(1024, 1024, 1024);
    auto selected = dispatcher.select_kernel(problem);

    // Should fall back to FirstFit
    ASSERT_NE(selected, nullptr);
}

// =============================================================================
// Dispatcher with Custom Registry Tests
// =============================================================================

class DispatcherCustomRegistryTest : public ::testing::Test
{
    protected:
    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(DispatcherCustomRegistryTest, UseCustomRegistry)
{
    Registry custom_registry;
    custom_registry.set_name("custom");

    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "custom_kernel");
    custom_registry.register_kernel(kernel);

    Dispatcher dispatcher(&custom_registry);
    Problem problem(1024, 1024, 1024);

    auto selected = dispatcher.select_kernel(problem);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "custom_kernel");
}

TEST_F(DispatcherCustomRegistryTest, CustomRegistryIsolation)
{
    Registry custom_registry;

    auto key_custom = make_test_key(256);
    auto key_global = make_test_key(512);

    custom_registry.register_kernel(
        std::make_shared<MockKernelInstance>(key_custom, "custom_kernel"));
    Registry::instance().register_kernel(
        std::make_shared<MockKernelInstance>(key_global, "global_kernel"));

    Dispatcher custom_dispatcher(&custom_registry);
    Dispatcher global_dispatcher;

    Problem problem(1024, 1024, 1024);

    auto custom_selected = custom_dispatcher.select_kernel(problem);
    auto global_selected = global_dispatcher.select_kernel(problem);

    ASSERT_NE(custom_selected, nullptr);
    ASSERT_NE(global_selected, nullptr);

    EXPECT_EQ(custom_selected->get_name(), "custom_kernel");
    EXPECT_EQ(global_selected->get_name(), "global_kernel");
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

class DispatcherEdgeCasesTest : public ::testing::Test
{
    protected:
    void SetUp() override { Registry::instance().clear(); }

    void TearDown() override { Registry::instance().clear(); }
};

TEST_F(DispatcherEdgeCasesTest, InvalidProblem)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    Dispatcher dispatcher;

    // Zero dimensions
    Problem invalid(0, 1024, 1024);
    EXPECT_FALSE(invalid.is_valid());

    // The dispatcher should still attempt selection
    // (validation is up to the kernel's supports() method)
}

TEST_F(DispatcherEdgeCasesTest, KernelDoesNotSupportProblem)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "selective_kernel", false);
    Registry::instance().register_kernel(kernel);

    Dispatcher dispatcher;

    // Problem not divisible by tile size - kernel doesn't support it
    Problem problem(1000, 1000, 1000); // Not divisible by 256

    auto selected = dispatcher.select_kernel(problem);
    // Should return nullptr since kernel doesn't support this problem
    EXPECT_EQ(selected, nullptr);
}

TEST_F(DispatcherEdgeCasesTest, MultipleSelectionsConsistent)
{
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel");
    Registry::instance().register_kernel(kernel);

    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    // Multiple selections should return the same kernel
    auto selected1 = dispatcher.select_kernel(problem);
    auto selected2 = dispatcher.select_kernel(problem);
    auto selected3 = dispatcher.select_kernel(problem);

    ASSERT_NE(selected1, nullptr);
    EXPECT_EQ(selected1, selected2);
    EXPECT_EQ(selected2, selected3);
}

// =============================================================================
// Validate Method Tests
// =============================================================================

class DispatcherValidateTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        auto key = make_test_key(256);
        kernel_  = std::make_shared<MockKernelInstance>(key, "kernel");
        Registry::instance().register_kernel(kernel_);
    }

    void TearDown() override { Registry::instance().clear(); }

    std::shared_ptr<MockKernelInstance> kernel_;
};

TEST_F(DispatcherValidateTest, ValidateWithMockKernel)
{
    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    // MockKernelInstance always validates successfully
    bool valid = dispatcher.validate(nullptr, nullptr, nullptr, nullptr, problem);

    // This depends on implementation - mock returns true
    // Real validation would need actual data
}

// =============================================================================
// Run Method Tests (with mock)
// =============================================================================

class DispatcherRunTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        Registry::instance().clear();

        auto key = make_test_key(256);
        kernel_  = std::make_shared<MockKernelInstance>(key, "kernel");
        Registry::instance().register_kernel(kernel_);
    }

    void TearDown() override { Registry::instance().clear(); }

    std::shared_ptr<MockKernelInstance> kernel_;
};

TEST_F(DispatcherRunTest, RunWithMockKernel)
{
    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    // Mock run (with null pointers - mock doesn't use them)
    float time = dispatcher.run(nullptr, nullptr, nullptr, problem);

    // Mock kernel returns 1.0f
    EXPECT_FLOAT_EQ(time, 1.0f);

    // Verify execution count
    EXPECT_EQ(kernel_->get_execution_count(), 1);
}

TEST_F(DispatcherRunTest, MultipleRuns)
{
    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    for(int i = 0; i < 10; i++)
    {
        (void)dispatcher.run(nullptr, nullptr, nullptr, problem);
    }

    EXPECT_EQ(kernel_->get_execution_count(), 10);
}

TEST_F(DispatcherRunTest, RunWithNoKernelThrows)
{
    Registry::instance().clear();

    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    // Should throw when no kernel found
    EXPECT_THROW((void)dispatcher.run(nullptr, nullptr, nullptr, problem), std::runtime_error);
}
