// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_conv_registry.cpp
 * @brief Unit tests for ConvRegistry and ConvDispatcher
 */

#include <iostream>
#include <cassert>
#include <string>

#include "ck_tile/dispatcher/conv_utils.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_decl;
using namespace ck_tile::dispatcher::conv_utils;

void test_conv_registry_basic()
{
    std::cout << "  test_conv_registry_basic... ";

    ConvRegistry registry;
    registry.set_name("test_registry");

    assert(registry.name() == "test_registry");
    assert(registry.size() == 0);
    assert(registry.empty());

    std::cout << "PASSED\n";
}

void test_conv_registry_register_kernel_set()
{
    std::cout << "  test_conv_registry_register_kernel_set... ";

    ConvRegistry registry;

    // Create a kernel set
    ConvKernelSet set;
    set.add(ConvSignature().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 128, 128).wave(2, 2, 1),
            "gfx942");
    set.add(ConvSignature().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 64, 64).wave(1, 4, 1),
            "gfx942");

    registry.register_set(set, ConvRegistry::Priority::High);

    assert(registry.size() == 2);
    assert(!registry.empty());

    std::cout << "PASSED\n";
}

void test_conv_registry_all_kernels()
{
    std::cout << "  test_conv_registry_all_kernels... ";

    ConvRegistry registry;

    ConvKernelSet set;
    set.add(ConvSignature().dtype("fp16").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 128, 128),
            "gfx942");
    set.add(ConvSignature().dtype("bf16").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 64, 64),
            "gfx942");

    registry.register_set(set, ConvRegistry::Priority::Normal);

    auto kernels = registry.all_kernels();
    assert(kernels.size() == 2);

    // Check kernel names
    bool found_fp16 = false;
    bool found_bf16 = false;
    for(const auto* k : kernels)
    {
        if(k->name().find("fp16") != std::string::npos)
            found_fp16 = true;
        if(k->name().find("bf16") != std::string::npos)
            found_bf16 = true;
    }
    assert(found_fp16);
    assert(found_bf16);

    std::cout << "PASSED\n";
}

void test_conv_registry_clear()
{
    std::cout << "  test_conv_registry_clear... ";

    ConvRegistry registry;

    ConvKernelSet set;
    set.add(ConvSignature().dtype("fp16").dims(2), ConvAlgorithm().tile(1, 128, 128), "gfx942");

    registry.register_set(set, ConvRegistry::Priority::High);
    assert(registry.size() == 1);

    registry.clear();
    assert(registry.size() == 0);
    assert(registry.empty());

    std::cout << "PASSED\n";
}

void test_conv_dispatcher_basic()
{
    std::cout << "  test_conv_dispatcher_basic... ";

    ConvRegistry registry;

    ConvKernelSet set;
    set.add(ConvSignature().dtype("fp16").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 128, 128),
            "gfx942");

    registry.register_set(set, ConvRegistry::Priority::High);

    ConvDispatcher dispatcher(&registry);

    // Check registry size via registry reference
    assert(registry.size() == 1);

    std::cout << "PASSED\n";
}

void test_conv_dispatcher_select()
{
    std::cout << "  test_conv_dispatcher_select... ";

    ConvRegistry registry;

    // Add multiple kernels with different tile sizes
    ConvKernelSet set;
    set.add(ConvSignature().dtype("fp16").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 64, 64),
            "gfx942");
    set.add(ConvSignature().dtype("fp16").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 128, 128),
            "gfx942");
    set.add(ConvSignature().dtype("fp16").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 256, 256),
            "gfx942");

    registry.register_set(set, ConvRegistry::Priority::Normal);

    ConvDispatcher dispatcher(&registry);

    // Create a problem
    auto problem = create_conv2d_problem(1, 64, 128, 28, 28, 3, 3, 1, 1, ConvOp::Forward);

    const auto* selected = dispatcher.select(problem);
    assert(selected != nullptr);

    // The dispatcher should select a kernel
    std::cout << "  [Selected: " << selected->name() << "] ";

    std::cout << "PASSED\n";
}

void test_multiple_registries()
{
    std::cout << "  test_multiple_registries... ";

    // Create throughput registry with large tiles
    ConvRegistry throughput_reg;
    throughput_reg.set_name("throughput");

    ConvKernelSet throughput_set;
    throughput_set.add(ConvSignature().dtype("fp16").conv_type("forward").dims(2),
                       ConvAlgorithm().tile(1, 256, 256),
                       "gfx942");
    throughput_reg.register_set(throughput_set, ConvRegistry::Priority::High);

    // Create latency registry with small tiles
    ConvRegistry latency_reg;
    latency_reg.set_name("latency");

    ConvKernelSet latency_set;
    latency_set.add(ConvSignature().dtype("fp16").conv_type("forward").dims(2),
                    ConvAlgorithm().tile(1, 64, 64),
                    "gfx942");
    latency_reg.register_set(latency_set, ConvRegistry::Priority::High);

    // Create dispatchers
    ConvDispatcher throughput_disp(&throughput_reg);
    ConvDispatcher latency_disp(&latency_reg);

    auto problem = create_conv2d_problem(1, 64, 128, 28, 28, 3, 3, 1, 1, ConvOp::Forward);

    const auto* throughput_kernel = throughput_disp.select(problem);
    const auto* latency_kernel    = latency_disp.select(problem);

    assert(throughput_kernel != nullptr);
    assert(latency_kernel != nullptr);

    // They should select different kernels
    assert(throughput_kernel->name() != latency_kernel->name());

    std::cout << "PASSED\n";
}

void test_conv_problem_matching()
{
    std::cout << "  test_conv_problem_matching... ";

    ConvRegistry registry;

    // Add 2D forward kernel only
    ConvKernelSet set;
    set.add(ConvSignature().dtype("fp16").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 128, 128),
            "gfx942");
    registry.register_set(set, ConvRegistry::Priority::High);

    ConvDispatcher dispatcher(&registry);

    // Test forward problem - should match
    auto fwd_problem       = create_conv2d_problem(1, 64, 128, 28, 28, 3, 3, 1, 1, ConvOp::Forward);
    const auto* fwd_kernel = dispatcher.select(fwd_problem);
    assert(fwd_kernel != nullptr);

    std::cout << "PASSED\n";
}

void test_conv_utilities_integration()
{
    std::cout << "  test_conv_utilities_integration... ";

    // Test problem creation helpers
    auto prob2d = create_conv2d_problem(1, 64, 128, 28, 28, 3, 3, 1, 1, ConvOp::Forward);
    assert(prob2d.N == 1);
    assert(prob2d.C == 64);
    assert(prob2d.K == 128);

    auto prob3d = create_conv3d_problem(1, 32, 64, 8, 16, 16, 3, 3, 3, 1, 1, ConvOp::Forward);
    assert(prob3d.N == 1);
    assert(prob3d.C == 32);

    // Test kernel set builders
    auto fwd_set = build_conv2d_fwd_set("fp16", "gfx942");
    assert(fwd_set.size() >= 3); // Should have multiple tile sizes

    auto full_set = build_conv2d_full_set("fp16", "gfx942");
    assert(full_set.size() >= 3); // Should have fwd, bwd_data, bwd_weight

    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Conv Registry Tests ===\n\n";

    test_conv_registry_basic();
    test_conv_registry_register_kernel_set();
    test_conv_registry_all_kernels();
    test_conv_registry_clear();
    test_conv_dispatcher_basic();
    test_conv_dispatcher_select();
    test_multiple_registries();
    test_conv_problem_matching();
    test_conv_utilities_integration();

    std::cout << "\n=== All Conv Registry Tests Passed! ===\n\n";
    return 0;
}
