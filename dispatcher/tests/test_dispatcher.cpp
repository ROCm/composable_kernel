// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for Dispatcher using Google Test

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

class DispatcherTest : public ::testing::Test
{
    protected:
    void SetUp() override
    {
        // Clear registry before each test
        Registry::instance().clear();
    }

    void TearDown() override
    {
        // Clean up after each test
        Registry::instance().clear();
    }
};

TEST_F(DispatcherTest, SelectKernelFirstFit)
{
    Dispatcher dispatcher;

    // Register kernels
    auto key1    = make_test_key(256);
    auto key2    = make_test_key(128);
    auto kernel1 = std::make_shared<MockKernelInstance>(key1, "kernel1");
    auto kernel2 = std::make_shared<MockKernelInstance>(key2, "kernel2");

    Registry::instance().register_kernel(kernel1);
    Registry::instance().register_kernel(kernel2);

    // Select kernel for valid problem
    Problem problem(1024, 1024, 1024);
    auto selected = dispatcher.select_kernel(problem);

    ASSERT_NE(selected, nullptr);
    // Should select a kernel that supports the problem
    // (order is not guaranteed, so just verify one is selected)
    EXPECT_TRUE(selected->get_name() == "kernel1" || selected->get_name() == "kernel2");
    EXPECT_TRUE(selected->supports(problem));
}

TEST_F(DispatcherTest, SelectKernelInvalidProblem)
{
    Dispatcher dispatcher;

    // Register kernel
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel1");
    Registry::instance().register_kernel(kernel);

    // Invalid problem
    Problem invalid_problem(0, 0, 0);
    auto selected = dispatcher.select_kernel(invalid_problem);

    EXPECT_EQ(selected, nullptr);
}

TEST_F(DispatcherTest, SelectKernelNoMatch)
{
    Dispatcher dispatcher;

    // Register kernel that doesn't support the problem
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel1", false);
    Registry::instance().register_kernel(kernel);

    // Problem with dimensions not divisible by tile size
    Problem problem(100, 100, 100); // Not divisible by 256
    auto selected = dispatcher.select_kernel(problem);

    EXPECT_EQ(selected, nullptr);
}

TEST_F(DispatcherTest, SelectKernelHeuristic)
{
    Dispatcher dispatcher;

    // Register kernels
    auto key1    = make_test_key(256);
    auto key2    = make_test_key(128);
    auto kernel1 = std::make_shared<MockKernelInstance>(key1, "kernel1");
    auto kernel2 = std::make_shared<MockKernelInstance>(key2, "kernel2");

    Registry::instance().register_kernel(kernel1);
    Registry::instance().register_kernel(kernel2);

    // Set heuristic that prefers kernel2
    dispatcher.set_heuristic([](const Problem&) {
        std::vector<std::string> candidates;
        auto key2 = make_test_key(128);
        candidates.push_back(key2.encode_identifier());
        auto key1 = make_test_key(256);
        candidates.push_back(key1.encode_identifier());
        return candidates;
    });

    Problem problem(1024, 1024, 1024);
    auto selected = dispatcher.select_kernel(problem);

    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "kernel2");
}

TEST_F(DispatcherTest, SelectKernelHeuristicFallback)
{
    Dispatcher dispatcher;

    // Register kernel
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel1");
    Registry::instance().register_kernel(kernel);

    // Set heuristic that returns non-existent kernel
    dispatcher.set_heuristic(
        [](const Problem&) { return std::vector<std::string>{"nonexistent_kernel"}; });

    Problem problem(1024, 1024, 1024);
    auto selected = dispatcher.select_kernel(problem);

    // Should fall back to first-fit
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "kernel1");
}

TEST_F(DispatcherTest, RunBasic)
{
    Dispatcher dispatcher;

    // Register kernel
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel1");
    Registry::instance().register_kernel(kernel);

    Problem problem(1024, 1024, 1024);

    // Mock pointers (not actually used)
    float a[1], b[1], c[1];

    float time_ms = dispatcher.run(a, b, c, problem);

    EXPECT_GT(time_ms, 0.0f);
    EXPECT_EQ(kernel->get_execution_count(), 1);
}

TEST_F(DispatcherTest, RunNoKernel)
{
    Dispatcher dispatcher;

    // No kernels registered
    Problem problem(1024, 1024, 1024);

    float a[1], b[1], c[1];

    EXPECT_THROW((void)dispatcher.run(a, b, c, problem), std::runtime_error);
}

TEST_F(DispatcherTest, RunExplicit)
{
    Dispatcher dispatcher;

    // Register kernel
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel1");
    Registry::instance().register_kernel(kernel);

    Problem problem(1024, 1024, 1024);
    std::string kernel_id = key.encode_identifier();

    float a[1], b[1], c[1];

    float time_ms = dispatcher.run_explicit(kernel_id, a, b, c, nullptr, problem);

    EXPECT_GT(time_ms, 0.0f);
    EXPECT_EQ(kernel->get_execution_count(), 1);
}

TEST_F(DispatcherTest, RunExplicitNotFound)
{
    Dispatcher dispatcher;

    Problem problem(1024, 1024, 1024);

    float a[1], b[1], c[1];

    EXPECT_THROW((void)dispatcher.run_explicit("nonexistent", a, b, c, nullptr, problem),
                 std::runtime_error);
}

TEST_F(DispatcherTest, RunExplicitNotSupported)
{
    Dispatcher dispatcher;

    // Register kernel that doesn't support the problem
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel1", false);
    Registry::instance().register_kernel(kernel);

    Problem problem(100, 100, 100); // Not divisible by 256
    std::string kernel_id = key.encode_identifier();

    float a[1], b[1], c[1];

    EXPECT_THROW((void)dispatcher.run_explicit(kernel_id, a, b, c, nullptr, problem),
                 std::runtime_error);
}

TEST_F(DispatcherTest, Validate)
{
    Dispatcher dispatcher;

    // Register kernel
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel1");
    Registry::instance().register_kernel(kernel);

    Problem problem(1024, 1024, 1024);

    float a[1], b[1], c[1];

    bool valid = dispatcher.validate(a, b, c, nullptr, problem);

    EXPECT_TRUE(valid);
}

TEST_F(DispatcherTest, ValidateNoKernel)
{
    Dispatcher dispatcher;

    // No kernels registered
    Problem problem(1024, 1024, 1024);

    float a[1], b[1], c[1];

    bool valid = dispatcher.validate(a, b, c, nullptr, problem);

    EXPECT_FALSE(valid);
}

TEST_F(DispatcherTest, StrategySelection)
{
    Dispatcher dispatcher;

    // Register kernel
    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel1");
    Registry::instance().register_kernel(kernel);

    Problem problem(1024, 1024, 1024);

    // Test FirstFit strategy
    dispatcher.set_strategy(Dispatcher::SelectionStrategy::FirstFit);
    auto selected1 = dispatcher.select_kernel(problem);
    ASSERT_NE(selected1, nullptr);

    // Test Heuristic strategy (without heuristic function - should fallback)
    dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
    auto selected2 = dispatcher.select_kernel(problem);
    ASSERT_NE(selected2, nullptr);
}

TEST_F(DispatcherTest, CustomRegistry)
{
    // Create custom registry instance (not singleton)
    // Note: This requires Registry to allow non-singleton instances
    // For now, we'll test with a separate registry instance
    // In practice, custom registry would be created differently

    // Since Registry is singleton-only, we'll test that dispatcher
    // can work with the singleton registry
    Registry& registry = Registry::instance();
    registry.clear();

    auto key    = make_test_key(256);
    auto kernel = std::make_shared<MockKernelInstance>(key, "kernel1");
    registry.register_kernel(kernel);

    // Dispatcher defaults to singleton registry
    Dispatcher dispatcher;

    Problem problem(1024, 1024, 1024);
    auto selected = dispatcher.select_kernel(problem);

    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "kernel1");
}
