// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Minimal test: Verify dispatcher can select and run a kernel
#include <iostream>
#include <memory>
#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "test_mock_kernel.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::test;

int main()
{
    std::cout << "Minimal Dispatcher Test\n";
    std::cout << "=======================\n\n";

    // Create a mock kernel for testing
    KernelKey key = make_test_key(128, 128, 64, "gfx942");
    auto kernel   = std::make_shared<MockKernelInstance>(key, "test_kernel_128x128x64", true);

    // Register kernel
    Registry::instance().clear();
    Registry::instance().register_kernel(kernel);

    std::cout << "OK Registered kernel: " << kernel->get_name() << "\n";

    // Create dispatcher and problem
    Dispatcher dispatcher;
    Problem problem(1024, 1024, 1024);

    std::cout << "OK Created problem: M=" << problem.M << " N=" << problem.N << " K=" << problem.K
              << "\n";

    // Select kernel
    auto selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cerr << "[FAIL] Failed to select kernel\n";
        return 1;
    }

    std::cout << "OK Selected kernel: " << selected->get_name() << "\n";

    // Mock execution (no actual GPU computation in mock kernel)
    void* a_ptr = nullptr;
    void* b_ptr = nullptr;
    void* c_ptr = nullptr;

    float time = dispatcher.run(a_ptr, b_ptr, c_ptr, problem);

    std::cout << "OK Executed kernel: " << time << " ms\n";
    std::cout << "\n[OK] Minimal test passed!\n";

    return 0;
}
