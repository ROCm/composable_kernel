// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

/**
 * @file test_conv_kernel_decl.cpp
 * @brief Unit tests for ConvKernelDecl, ConvKernelSet and declarative macros
 */

#include <iostream>
#include <cassert>
#include <string>

#include "ck_tile/dispatcher/conv_kernel_decl.hpp"

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::conv_decl;

void test_conv_signature_builder()
{
    std::cout << "  test_conv_signature_builder... ";

    ConvSignature sig;
    sig.dtype("fp16").layout("nhwc").conv_type("forward").dims(2).groups(1);

    assert(sig.dtype_in_ == "fp16");
    assert(sig.dtype_wei_ == "fp16");
    assert(sig.dtype_out_ == "fp16");
    assert(sig.dtype_acc_ == "fp32");
    assert(sig.layout_ == "nhwc");
    assert(sig.conv_op_ == "forward");
    assert(sig.num_dims_ == 2);
    assert(sig.groups_ == 1);

    std::cout << "PASSED\n";
}

void test_conv_algorithm_builder()
{
    std::cout << "  test_conv_algorithm_builder... ";

    ConvAlgorithm algo;
    algo.tile(1, 128, 64)
        .wave(2, 2, 1)
        .warp(32, 32, 16)
        .pipeline("compv4")
        .scheduler("intrawave")
        .epilogue("cshuffle");

    assert(algo.tile_n_ == 1);
    assert(algo.tile_k_ == 128);
    assert(algo.tile_c_ == 64);
    assert(algo.wave_m_ == 2);
    assert(algo.wave_n_ == 2);
    assert(algo.wave_k_ == 1);
    assert(algo.warp_m_ == 32);
    assert(algo.warp_n_ == 32);
    assert(algo.warp_k_ == 16);
    assert(algo.pipeline_ == "compv4");
    assert(algo.scheduler_ == "intrawave");
    assert(algo.epilogue_ == "cshuffle");

    std::cout << "PASSED\n";
}

void test_conv_kernel_decl()
{
    std::cout << "  test_conv_kernel_decl... ";

    ConvKernelDecl decl(ConvSignature().dtype("bf16").layout("nhwgc").conv_type("forward").dims(2),
                        ConvAlgorithm().tile(1, 256, 128).wave(4, 1, 1).pipeline("compv3"),
                        "gfx942");

    assert(decl.signature.dtype_in_ == "bf16");
    assert(decl.algorithm.tile_k_ == 256);
    assert(decl.algorithm.tile_c_ == 128);
    assert(decl.arch == "gfx942");

    // Test name generation
    std::string name = decl.name();
    assert(name.find("bf16") != std::string::npos);
    assert(name.find("256x128") != std::string::npos);

    std::cout << "PASSED\n";
}

void test_conv_kernel_set()
{
    std::cout << "  test_conv_kernel_set... ";

    ConvKernelSet set;

    // Add kernels
    set.add(ConvSignature().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 128, 128).wave(2, 2, 1),
            "gfx942");

    set.add(ConvSignature().dtype("fp16").layout("nhwc").conv_type("forward").dims(2),
            ConvAlgorithm().tile(1, 64, 64).wave(1, 4, 1),
            "gfx942");

    set.add(ConvSignature().dtype("fp16").layout("nhwc").conv_type("bwd_data").dims(2),
            ConvAlgorithm().tile(1, 128, 64).wave(2, 2, 1),
            "gfx942");

    assert(set.size() == 3);

    auto decls = set.declarations();
    assert(decls.size() == 3);

    // Check first declaration
    assert(decls[0].signature.conv_op_ == "forward");
    assert(decls[0].algorithm.tile_k_ == 128);

    // Check last declaration
    assert(decls[2].signature.conv_op_ == "bwd_data");
    assert(decls[2].algorithm.tile_c_ == 64);

    std::cout << "PASSED\n";
}

void test_conv_kernel_set_merge()
{
    std::cout << "  test_conv_kernel_set_merge... ";

    ConvKernelSet set1;
    set1.add(ConvSignature().dtype("fp16").conv_type("forward").dims(2),
             ConvAlgorithm().tile(1, 128, 128),
             "gfx942");

    ConvKernelSet set2;
    set2.add(ConvSignature().dtype("fp16").conv_type("bwd_data").dims(2),
             ConvAlgorithm().tile(1, 64, 64),
             "gfx942");
    set2.add(ConvSignature().dtype("fp16").conv_type("bwd_weight").dims(2),
             ConvAlgorithm().tile(1, 32, 32),
             "gfx942");

    assert(set1.size() == 1);
    assert(set2.size() == 2);

    set1.merge(set2);

    assert(set1.size() == 3);

    std::cout << "PASSED\n";
}

void test_conv_kernel_set_registry()
{
    std::cout << "  test_conv_kernel_set_registry... ";

    // Clear existing registry
    ConvKernelSetRegistry::instance().clear();

    // Register a set
    ConvKernelSetRegistry::instance().register_set(
        "test_set",
        ConvKernelSet()
            .add(ConvSignature().dtype("fp16").dims(2), ConvAlgorithm().tile(1, 128, 128), "gfx942")
            .add(ConvSignature().dtype("bf16").dims(2), ConvAlgorithm().tile(1, 64, 64), "gfx942"));

    // Retrieve and check
    const auto& retrieved = ConvKernelSetRegistry::instance().get("test_set");
    assert(retrieved.size() == 2);

    // Check that non-existent returns empty set
    const auto& empty_set = ConvKernelSetRegistry::instance().get("nonexistent");
    assert(empty_set.size() == 0);

    std::cout << "PASSED\n";
}

void test_conv_signature_variations()
{
    std::cout << "  test_conv_signature_variations... ";

    // 1D conv
    ConvSignature sig1d;
    sig1d.dtype("fp32").dims(1).conv_type("forward");
    assert(sig1d.num_dims_ == 1);

    // 3D conv
    ConvSignature sig3d;
    sig3d.dtype("fp16").dims(3).conv_type("forward").layout("ndhwgc");
    assert(sig3d.num_dims_ == 3);
    assert(sig3d.layout_ == "ndhwgc");

    // Backward data
    ConvSignature bwd_data;
    bwd_data.dtype("bf16").conv_type("bwd_data");
    assert(bwd_data.conv_op_ == "bwd_data");

    // Backward weight
    ConvSignature bwd_weight;
    bwd_weight.dtype("fp16").conv_type("bwd_weight");
    assert(bwd_weight.conv_op_ == "bwd_weight");

    // Grouped conv
    ConvSignature grouped;
    grouped.dtype("fp16").groups(4);
    assert(grouped.groups_ == 4);

    std::cout << "PASSED\n";
}

void test_conv_algorithm_variations()
{
    std::cout << "  test_conv_algorithm_variations... ";

    // Different pipelines
    ConvAlgorithm v3;
    v3.pipeline("compv3");
    assert(v3.pipeline_ == "compv3");

    ConvAlgorithm v4;
    v4.pipeline("compv4");
    assert(v4.pipeline_ == "compv4");

    ConvAlgorithm v5;
    v5.pipeline("compv5");
    assert(v5.pipeline_ == "compv5");

    ConvAlgorithm mem;
    mem.pipeline("mem");
    assert(mem.pipeline_ == "mem");

    // Different schedulers
    ConvAlgorithm intra;
    intra.scheduler("intrawave");
    assert(intra.scheduler_ == "intrawave");

    ConvAlgorithm inter;
    inter.scheduler("interwave");
    assert(inter.scheduler_ == "interwave");

    // Different tile sizes
    ConvAlgorithm small;
    small.tile(1, 32, 32).wave(1, 4, 1).warp(16, 16, 32);
    assert(small.tile_k_ == 32);

    ConvAlgorithm large;
    large.tile(1, 256, 256).wave(4, 1, 1).warp(32, 32, 16);
    assert(large.tile_k_ == 256);

    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Conv Kernel Decl Tests ===\n\n";

    test_conv_signature_builder();
    test_conv_algorithm_builder();
    test_conv_kernel_decl();
    test_conv_kernel_set();
    test_conv_kernel_set_merge();
    test_conv_kernel_set_registry();
    test_conv_signature_variations();
    test_conv_algorithm_variations();

    std::cout << "\n=== All Conv Kernel Decl Tests Passed! ===\n\n";
    return 0;
}
