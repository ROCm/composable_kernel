// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for GroupedConvRegistry and GroupedConvDispatcher using assert() and std::cout

#include "ck_tile/dispatcher/grouped_conv_utils.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <atomic>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::grouped_conv_decl;

void test_grouped_conv_registry_basic()
{
    std::cout << "  test_grouped_conv_registry_basic... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    reg.set_name("test_registry");
    assert(reg.get_name() == "test_registry");

    assert(reg.size() == 0);
    assert(reg.empty());

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_register_set()
{
    std::cout << "  test_grouped_conv_registry_register_set... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    set.add("fp16", "nhwc", "forward", 256, 256);

    bool ok = reg.register_set(set);
    assert(ok);
    assert(reg.size() == 2);
    assert(!reg.empty());

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_all_kernels()
{
    std::cout << "  test_grouped_conv_registry_all_kernels... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    reg.register_set(set);

    auto all = reg.all_kernels();
    assert(all.size() == 1);
    assert(all[0]->name().find("grouped_conv_") != std::string::npos);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_clear()
{
    std::cout << "  test_grouped_conv_registry_clear... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    reg.register_set(set);
    assert(reg.size() == 1);

    reg.clear();
    assert(reg.size() == 0);
    assert(reg.empty());

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_thread_safe()
{
    std::cout << "  test_grouped_conv_registry_thread_safe... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    const int num_threads     = 4;
    const int sets_per_thread = 10;
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};

    for(int t = 0; t < num_threads; t++)
    {
        threads.emplace_back([t, &reg, &success_count]() {
            for(int k = 0; k < sets_per_thread; k++)
            {
                GroupedConvKernelSet set;
                set.add("fp16", "nhwc", "forward", 128 + t * 32 + k, 128);
                if(reg.register_set(set))
                {
                    success_count++;
                }
            }
        });
    }

    for(auto& th : threads)
        th.join();

    assert(reg.size() == num_threads * sets_per_thread);
    assert(success_count.load() == num_threads * sets_per_thread);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_export_json()
{
    std::cout << "  test_grouped_conv_registry_export_json... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    reg.register_set(set);

    std::string json = reg.export_json(false);
    assert(!json.empty());
    assert(json.find("\"kernels\"") != std::string::npos);
    assert(json.find("\"metadata\"") != std::string::npos);
    assert(json.find("grouped_conv_") != std::string::npos);

    std::string json_stats = reg.export_json(true);
    assert(json_stats.find("\"statistics\"") != std::string::npos);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_registry_filter()
{
    std::cout << "  test_grouped_conv_registry_filter... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    set.add("fp16", "nhwc", "forward", 256, 256);
    set.add("bf16", "nhwc", "forward", 128, 128);
    reg.register_set(set);

    auto fp16_only =
        reg.filter([](const GroupedConvKernelInstance& k) { return k.key().dtype_in == "fp16"; });
    assert(fp16_only.size() == 2);

    auto large_tile = reg.filter([](const GroupedConvKernelInstance& k) {
        return k.key().tile_m >= 256 || k.key().tile_n >= 256;
    });
    assert(large_tile.size() >= 1);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_dispatcher_basic()
{
    std::cout << "  test_grouped_conv_dispatcher_basic... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    reg.register_set(set);

    GroupedConvDispatcher dispatcher(&reg);
    GroupedConvProblem problem = grouped_conv_utils::create_grouped_conv2d_problem(
        4, 64, 128, 28, 28, 3, 3, 1, 1, GroupedConvOp::Forward);

    float time = dispatcher.run(problem, nullptr);
    assert(time >= 0.0f);

    reg.clear();
    std::cout << "PASSED\n";
}

void test_grouped_conv_dispatcher_select()
{
    std::cout << "  test_grouped_conv_dispatcher_select... ";
    GroupedConvRegistry& reg = GroupedConvRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    set.add("fp16", "nhwc", "forward", 256, 256);
    reg.register_set(set);

    GroupedConvDispatcher dispatcher(&reg);
    GroupedConvProblem problem = grouped_conv_utils::create_grouped_conv2d_problem(
        4, 64, 128, 28, 28, 3, 3, 1, 1, GroupedConvOp::Forward);

    const auto* selected = dispatcher.select(problem);
    assert(selected != nullptr);
    assert(selected->name().find("grouped_conv_") != std::string::npos);
    assert(selected->matches(problem));

    reg.clear();
    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Test Grouped Conv Registry ===\n\n";
    test_grouped_conv_registry_basic();
    test_grouped_conv_registry_register_set();
    test_grouped_conv_registry_all_kernels();
    test_grouped_conv_registry_clear();
    test_grouped_conv_registry_thread_safe();
    test_grouped_conv_registry_export_json();
    test_grouped_conv_registry_filter();
    test_grouped_conv_dispatcher_basic();
    test_grouped_conv_dispatcher_select();
    std::cout << "\n=== All Tests Passed! ===\n\n";
    return 0;
}
