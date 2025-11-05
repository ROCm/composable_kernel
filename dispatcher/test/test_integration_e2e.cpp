// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/// End-to-end integration tests for CK Tile Dispatcher
/// Tests complete workflows from kernel registration through dispatch and validation

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "test_mock_kernel.hpp"
#include <gtest/gtest.h>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

class IntegrationE2ETest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clear registry before each test
        Registry::instance().clear();
    }
    
    void TearDown() override {
        // Clean up after each test
        Registry::instance().clear();
    }
};

/// Test 1: Complete workflow - single kernel registration and dispatch
TEST_F(IntegrationE2ETest, SingleKernelWorkflow) {
    // Step 1: Create a kernel
    KernelKey key = make_test_key(256, 256, 32, 942);
    auto kernel = std::make_shared<MockKernelInstance>(
        key, "test_kernel_256x256x32", true);
    
    // Step 2: Register kernel
    bool registered = Registry::instance().register_kernel(kernel);
    ASSERT_TRUE(registered);
    
    // Step 3: Create dispatcher
    Dispatcher dispatcher;
    
    // Step 4: Define problem
    Problem problem(512, 512, 512);  // Divisible by tile sizes
    
    // Step 5: Select kernel
    auto selected = dispatcher.select_kernel(problem);
    ASSERT_NE(selected, nullptr);
    EXPECT_EQ(selected->get_name(), "test_kernel_256x256x32");
    
    // Step 6: Execute (mock execution)
    const void* a_ptr = nullptr;  // Mock pointers
    const void* b_ptr = nullptr;
    void* c_ptr = nullptr;
    
    float time = selected->run(a_ptr, b_ptr, c_ptr, nullptr, problem, nullptr);
    EXPECT_GT(time, 0.0f);
}

/// Test 2: Multiple kernels - dispatcher selects appropriate one
TEST_F(IntegrationE2ETest, MultipleKernelSelection) {
    // Register multiple kernels with different tile sizes
    auto kernel1 = std::make_shared<MockKernelInstance>(
        make_test_key(256, 256, 32, 942), "kernel_256", false);  // strict divisibility
    
    auto kernel2 = std::make_shared<MockKernelInstance>(
        make_test_key(128, 128, 64, 942), "kernel_128", false);  // strict divisibility
    
    Registry::instance().register_kernel(kernel1);
    Registry::instance().register_kernel(kernel2);
    
    Dispatcher dispatcher;
    
    // Problem 1: Divisible by 256 (should select kernel1)
    Problem problem1(512, 512, 512);
    auto selected1 = dispatcher.select_kernel(problem1);
    ASSERT_NE(selected1, nullptr);
    // First-fit will return the first registered kernel that supports the problem
    
    // Problem 2: Divisible by 128 but not 256 (should select kernel2)
    Problem problem2(384, 384, 384);  // 384 = 3 * 128, not divisible by 256
    auto selected2 = dispatcher.select_kernel(problem2);
    ASSERT_NE(selected2, nullptr);
    
    // Problem 3: Not divisible by either (should fail)
    Problem problem3(100, 100, 100);
    auto selected3 = dispatcher.select_kernel(problem3);
    EXPECT_EQ(selected3, nullptr);
}

/// Test 3: Heuristic-based selection
TEST_F(IntegrationE2ETest, HeuristicBasedSelection) {
    // Register two kernels
    auto kernel1 = std::make_shared<MockKernelInstance>(
        make_test_key(256, 256, 32, 942), "kernel_256", true);
    auto kernel2 = std::make_shared<MockKernelInstance>(
        make_test_key(128, 128, 64, 942), "kernel_128", true);
    
    Registry::instance().register_kernel(kernel1);
    Registry::instance().register_kernel(kernel2);
    
    // Define heuristic: prefer kernel_128 for small problems
    auto heuristic = [](const Problem& p) -> std::vector<std::string> {
        if (p.M < 512 || p.N < 512 || p.K < 512) {
            // Small problem - prefer smaller tile
            return {"128x128x64_2x2x1_32x32x16_nopers"};
        } else {
            // Large problem - prefer larger tile
            return {"256x256x32_2x2x1_32x32x16_nopers"};
        }
    };
    
    Dispatcher dispatcher;
    dispatcher.set_heuristic(heuristic);
    
    // Small problem
    Problem small_problem(256, 256, 256);
    auto selected_small = dispatcher.select_kernel(small_problem);
    ASSERT_NE(selected_small, nullptr);
    
    // Large problem
    Problem large_problem(1024, 1024, 1024);
    auto selected_large = dispatcher.select_kernel(large_problem);
    ASSERT_NE(selected_large, nullptr);
}

/// Test 4: Priority-based conflict resolution
TEST_F(IntegrationE2ETest, PriorityConflictResolution) {
    KernelKey key = make_test_key(256, 256, 32, 942);
    
    // Register kernel with Normal priority
    auto kernel1 = std::make_shared<MockKernelInstance>(
        key, "kernel_v1", true);
    bool reg1 = Registry::instance().register_kernel(kernel1, Registry::Priority::Normal);
    ASSERT_TRUE(reg1);
    
    // Try to register another kernel with same key but Low priority
    auto kernel2 = std::make_shared<MockKernelInstance>(
        key, "kernel_v2", true);
    bool reg2 = Registry::instance().register_kernel(kernel2, Registry::Priority::Low);
    EXPECT_FALSE(reg2);  // Should fail - existing kernel has higher priority
    
    // Verify original kernel is still registered
    std::string id = key.encode_identifier();
    auto found = Registry::instance().lookup(id);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->get_name(), "kernel_v1");
    
    // Register with High priority - should replace
    auto kernel3 = std::make_shared<MockKernelInstance>(
        key, "kernel_v3", true);
    bool reg3 = Registry::instance().register_kernel(kernel3, Registry::Priority::High);
    EXPECT_TRUE(reg3);  // Should succeed - higher priority
    
    // Verify new kernel replaced old one
    auto found2 = Registry::instance().lookup(id);
    ASSERT_NE(found2, nullptr);
    EXPECT_EQ(found2->get_name(), "kernel_v3");
}

/// Test 5: Explicit kernel selection via run_explicit
TEST_F(IntegrationE2ETest, ExplicitKernelSelection) {
    // Register multiple kernels
    auto kernel1 = std::make_shared<MockKernelInstance>(
        make_test_key(256, 256, 32, 942), "kernel_256", true);
    auto kernel2 = std::make_shared<MockKernelInstance>(
        make_test_key(128, 128, 64, 942), "kernel_128", true);
    
    Registry::instance().register_kernel(kernel1);
    Registry::instance().register_kernel(kernel2);
    
    Dispatcher dispatcher;
    Problem problem(512, 512, 512);
    
    // Explicitly select kernel_128
    std::string kernel2_id = kernel2->get_key().encode_identifier();
    const void* a_ptr = nullptr;
    const void* b_ptr = nullptr;
    void* c_ptr = nullptr;
    
    float time = dispatcher.run_explicit(
        kernel2_id, a_ptr, b_ptr, c_ptr, nullptr, problem, nullptr);
    
    EXPECT_GT(time, 0.0f);
}

/// Test 6: Error handling - no suitable kernel
TEST_F(IntegrationE2ETest, NoSuitableKernel) {
    // Register kernel with strict divisibility requirements
    auto kernel = std::make_shared<MockKernelInstance>(
        make_test_key(256, 256, 32, 942), "kernel_256", false);
    Registry::instance().register_kernel(kernel);
    
    Dispatcher dispatcher;
    
    // Problem not divisible by tile sizes
    Problem problem(100, 100, 100);
    
    // select_kernel should return nullptr
    auto selected = dispatcher.select_kernel(problem);
    EXPECT_EQ(selected, nullptr);
    
    // run() should throw
    const void* a_ptr = nullptr;
    const void* b_ptr = nullptr;
    void* c_ptr = nullptr;
    
    EXPECT_THROW(
        dispatcher.run(a_ptr, b_ptr, c_ptr, problem, nullptr),
        std::runtime_error
    );
}

/// Test 7: Error handling - invalid kernel ID
TEST_F(IntegrationE2ETest, InvalidKernelID) {
    Dispatcher dispatcher;
    Problem problem(512, 512, 512);
    
    const void* a_ptr = nullptr;
    const void* b_ptr = nullptr;
    void* c_ptr = nullptr;
    
    // Non-existent kernel ID
    EXPECT_THROW(
        dispatcher.run_explicit(
            "non_existent_kernel", a_ptr, b_ptr, c_ptr, nullptr, problem, nullptr),
        std::runtime_error
    );
}

/// Test 8: Registry enumeration and filtering
TEST_F(IntegrationE2ETest, RegistryEnumerationAndFiltering) {
    // Register multiple kernels
    auto kernel1 = std::make_shared<MockKernelInstance>(
        make_test_key(256, 256, 32, 942), "kernel_256", true);
    auto kernel2 = std::make_shared<MockKernelInstance>(
        make_test_key(128, 128, 64, 942), "kernel_128", true);
    auto kernel3 = std::make_shared<MockKernelInstance>(
        make_test_key(64, 64, 128, 942), "kernel_64", true);
    
    Registry::instance().register_kernel(kernel1);
    Registry::instance().register_kernel(kernel2);
    Registry::instance().register_kernel(kernel3);
    
    // Test: get all kernels
    auto all_kernels = Registry::instance().get_all();
    EXPECT_EQ(all_kernels.size(), 3);
    
    // Test: filter kernels by problem support
    Problem problem(512, 512, 512);
    auto compatible = Registry::instance().filter(
        [&problem](const KernelInstance& k) {
            return k.supports(problem);
        }
    );
    
    // All should support since we used supports_all=true
    EXPECT_EQ(compatible.size(), 3);
    
    // Test: filter by name pattern
    auto kernel_256_filtered = Registry::instance().filter(
        [](const KernelInstance& k) {
            return k.get_name().find("256") != std::string::npos;
        }
    );
    
    EXPECT_EQ(kernel_256_filtered.size(), 1);
    EXPECT_EQ(kernel_256_filtered[0]->get_name(), "kernel_256");
}

/// Test 9: Problem validation
TEST_F(IntegrationE2ETest, ProblemValidation) {
    auto kernel = std::make_shared<MockKernelInstance>(
        make_test_key(256, 256, 32, 942), "test_kernel", true);
    Registry::instance().register_kernel(kernel);
    
    Dispatcher dispatcher;
    
    // Valid problem
    Problem valid_problem(512, 512, 512);
    EXPECT_TRUE(valid_problem.is_valid());
    auto selected = dispatcher.select_kernel(valid_problem);
    EXPECT_NE(selected, nullptr);
    
    // Invalid problem - zero dimension
    Problem invalid_problem1(0, 512, 512);
    EXPECT_FALSE(invalid_problem1.is_valid());
    auto not_selected1 = dispatcher.select_kernel(invalid_problem1);
    EXPECT_EQ(not_selected1, nullptr);
    
    // Invalid problem - negative dimension
    Problem invalid_problem2(-100, 512, 512);
    EXPECT_FALSE(invalid_problem2.is_valid());
    auto not_selected2 = dispatcher.select_kernel(invalid_problem2);
    EXPECT_EQ(not_selected2, nullptr);
}

/// Test 10: Complete workflow with validation
TEST_F(IntegrationE2ETest, WorkflowWithValidation) {
    auto kernel = std::make_shared<MockKernelInstance>(
        make_test_key(256, 256, 32, 942), "test_kernel", true);
    Registry::instance().register_kernel(kernel);
    
    Dispatcher dispatcher;
    Problem problem(512, 512, 512);
    problem.enable_validation = true;
    
    // Select and execute
    auto selected = dispatcher.select_kernel(problem);
    ASSERT_NE(selected, nullptr);
    
    const void* a_ptr = nullptr;
    const void* b_ptr = nullptr;
    void* c_ptr = nullptr;
    
    // Execute
    float time = selected->run(a_ptr, b_ptr, c_ptr, nullptr, problem, nullptr);
    EXPECT_GT(time, 0.0f);
    
    // Validate (mock validation always passes)
    bool valid = selected->validate(a_ptr, b_ptr, c_ptr, nullptr, problem, 1e-3f);
    EXPECT_TRUE(valid);
    
    // Can also validate through dispatcher
    bool valid2 = dispatcher.validate(a_ptr, b_ptr, c_ptr, nullptr, problem, 1e-3f);
    EXPECT_TRUE(valid2);
}

/// Test 11: Strategy switching
TEST_F(IntegrationE2ETest, StrategySwitching) {
    auto kernel = std::make_shared<MockKernelInstance>(
        make_test_key(256, 256, 32, 942), "test_kernel", true);
    Registry::instance().register_kernel(kernel);
    
    Dispatcher dispatcher;
    Problem problem(512, 512, 512);
    
    // Default strategy (FirstFit)
    auto selected1 = dispatcher.select_kernel(problem);
    EXPECT_NE(selected1, nullptr);
    
    // Switch to Heuristic without setting heuristic (should fall back to FirstFit)
    dispatcher.set_strategy(Dispatcher::SelectionStrategy::Heuristic);
    auto selected2 = dispatcher.select_kernel(problem);
    EXPECT_NE(selected2, nullptr);
    
    // Set heuristic
    auto heuristic = [](const Problem&) -> std::vector<std::string> {
        return {"256x256x32_2x2x1_32x32x16_nopers"};
    };
    dispatcher.set_heuristic(heuristic);
    
    auto selected3 = dispatcher.select_kernel(problem);
    EXPECT_NE(selected3, nullptr);
    
    // Switch back to FirstFit
    dispatcher.set_strategy(Dispatcher::SelectionStrategy::FirstFit);
    auto selected4 = dispatcher.select_kernel(problem);
    EXPECT_NE(selected4, nullptr);
}

