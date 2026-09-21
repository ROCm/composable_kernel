// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for GroupedConvKernelDecl using assert() and std::cout

#include "ck_tile/dispatcher/grouped_conv_kernel_decl.hpp"
#include <cassert>
#include <iostream>

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::grouped_conv_decl;

void test_grouped_conv_signature_builder()
{
    std::cout << "  test_grouped_conv_signature_builder... ";
    GroupedConvSignature sig;
    sig.dtype("fp16").layout("nhwc").conv_type("forward").dims(2).groups(4);
    assert(sig.dtype_in_ == "fp16");
    assert(sig.dtype_wei_ == "fp16");
    assert(sig.dtype_out_ == "fp16");
    assert(sig.layout_ == "nhwc");
    assert(sig.conv_op_ == "forward");
    assert(sig.num_dims_ == 2);
    assert(sig.groups_ == 4);
    assert(sig.op_str() == "fwd");
    sig.conv_type("bwd_data");
    assert(sig.op_str() == "bwd_data");
    sig.conv_type("bwd_weight");
    assert(sig.op_str() == "bwd_weight");
    std::cout << "PASSED\n";
}

void test_grouped_conv_algorithm_builder()
{
    std::cout << "  test_grouped_conv_algorithm_builder... ";
    GroupedConvAlgorithm algo;
    algo.tile(128, 128, 64)
        .wave(2, 2, 1)
        .warp(32, 32, 16)
        .pipeline("compv4")
        .scheduler("intrawave");
    assert(algo.tile_m_ == 128);
    assert(algo.tile_n_ == 128);
    assert(algo.tile_k_ == 64);
    assert(algo.wave_m_ == 2);
    assert(algo.wave_n_ == 2);
    assert(algo.warp_m_ == 32);
    assert(algo.warp_n_ == 32);
    assert(algo.pipeline_ == "compv4");
    assert(algo.scheduler_ == "intrawave");
    assert(!algo.needs_expansion());
    algo.wave_m_ = ANY_INT;
    assert(algo.needs_wave_expansion());
    std::cout << "PASSED\n";
}

void test_grouped_conv_kernel_decl()
{
    std::cout << "  test_grouped_conv_kernel_decl... ";
    GroupedConvSignature sig;
    sig.dtype("fp16").layout("nhwc").conv_type("forward").dims(2);
    GroupedConvAlgorithm algo;
    algo.tile(128, 128, 64).wave(2, 2, 1).warp(32, 32, 16);
    GroupedConvKernelDecl decl(sig, algo, "gfx942");
    std::string name = decl.name();
    assert(!name.empty());
    assert(name.find("grouped_conv_") != std::string::npos);
    assert(name.find("fwd") != std::string::npos);
    assert(name.find("fp16") != std::string::npos);
    assert(name.find("128x128x64") != std::string::npos);
    assert(!decl.has_wildcards());
    std::cout << "PASSED\n";
}

void test_grouped_conv_kernel_set()
{
    std::cout << "  test_grouped_conv_kernel_set... ";
    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    assert(set.size() == 1);
    set.add("fp16", "nhwc", "forward", 256, 256);
    assert(set.size() == 2);
    const auto& decls = set.declarations();
    assert(decls[0].algorithm.tile_n_ == 128);
    assert(decls[0].algorithm.tile_k_ == 128);
    assert(decls[1].algorithm.tile_n_ == 256);
    assert(decls[1].algorithm.tile_k_ == 256);
    set.tag("test_set");
    assert(set.tag() == "test_set");
    std::cout << "PASSED\n";
}

void test_grouped_conv_kernel_set_merge()
{
    std::cout << "  test_grouped_conv_kernel_set_merge... ";
    GroupedConvKernelSet set1;
    set1.add("fp16", "nhwc", "forward", 128, 128);
    GroupedConvKernelSet set2;
    set2.add("fp16", "nhwc", "forward", 256, 256);
    set1.merge(set2);
    assert(set1.size() == 2);
    assert(set1.declarations()[0].algorithm.tile_n_ == 128);
    assert(set1.declarations()[1].algorithm.tile_n_ == 256);
    std::cout << "PASSED\n";
}

void test_grouped_conv_kernel_set_registry()
{
    std::cout << "  test_grouped_conv_kernel_set_registry... ";
    auto& reg = GroupedConvKernelSetRegistry::instance();
    reg.clear();

    GroupedConvKernelSet set;
    set.add("fp16", "nhwc", "forward", 128, 128);
    reg.register_set("gconv_test", set);
    assert(reg.has("gconv_test"));
    assert(reg.size() >= 1);

    const auto& retrieved = reg.get("gconv_test");
    assert(retrieved.size() == 1);

    const auto& empty = reg.get("nonexistent");
    assert(empty.size() == 0);

    reg.clear();
    assert(!reg.has("gconv_test"));
    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Test Grouped Conv Kernel Decl ===\n\n";
    test_grouped_conv_signature_builder();
    test_grouped_conv_algorithm_builder();
    test_grouped_conv_kernel_decl();
    test_grouped_conv_kernel_set();
    test_grouped_conv_kernel_set_merge();
    test_grouped_conv_kernel_set_registry();
    std::cout << "\n=== All Tests Passed! ===\n\n";
    return 0;
}
