// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Unit tests for GroupedConvConfig using assert() and std::cout

#include "ck_tile/dispatcher/grouped_conv_config.hpp"
#include <cassert>
#include <iostream>

using namespace ck_tile::dispatcher;

void test_grouped_conv_direction_enum()
{
    std::cout << "  test_grouped_conv_direction_enum... ";
    assert(GroupedConvSignatureInfo::direction_str(GroupedConvDirection::FORWARD) ==
           std::string("fwd"));
    assert(GroupedConvSignatureInfo::direction_str(GroupedConvDirection::BACKWARD_DATA) ==
           std::string("bwd_data"));
    assert(GroupedConvSignatureInfo::direction_str(GroupedConvDirection::BACKWARD_WEIGHT) ==
           std::string("bwd_weight"));
    std::cout << "PASSED\n";
}

void test_grouped_conv_signature_info()
{
    std::cout << "  test_grouped_conv_signature_info... ";
    GroupedConvSignatureInfo sig;
    assert(sig.spatial_dim == 2);
    assert(sig.direction == GroupedConvDirection::FORWARD);
    assert(sig.in_type == "fp16");
    assert(sig.wei_type == "fp16");
    assert(sig.out_type == "fp16");
    assert(sig.acc_type == "fp32");
    assert(sig.num_groups == 1);
    sig.in_type    = "bf16";
    sig.num_groups = 4;
    assert(sig.in_type == "bf16");
    assert(sig.num_groups == 4);
    std::cout << "PASSED\n";
}

void test_grouped_conv_algorithm_info()
{
    std::cout << "  test_grouped_conv_algorithm_info... ";
    GroupedConvAlgorithmInfo algo;
    assert(algo.tile.m == 128);
    assert(algo.tile.n == 128);
    assert(algo.tile.k == 64);
    assert(algo.pipeline == PipelineVersion::V4);
    assert(algo.scheduler == PipelineScheduler::INTRAWAVE);
    assert(GroupedConvAlgorithmInfo::pipeline_str(PipelineVersion::V4) == std::string("compv4"));
    assert(GroupedConvAlgorithmInfo::scheduler_str(PipelineScheduler::INTRAWAVE) ==
           std::string("intrawave"));
    std::cout << "PASSED\n";
}

void test_grouped_conv_config()
{
    std::cout << "  test_grouped_conv_config... ";
    GroupedConvConfig cfg;
    std::string name = cfg.name();
    assert(!name.empty());
    assert(name.find("grouped_conv_") != std::string::npos);
    assert(name.find("fwd") != std::string::npos);
    assert(name.find("fp16") != std::string::npos);
    assert(name.find("2d") != std::string::npos);

    std::string brief = cfg.brief();
    assert(!brief.empty());
    assert(brief.find("2D") != std::string::npos || brief.find("Grouped") != std::string::npos);

    std::string detailed = cfg.detailed();
    assert(!detailed.empty());
    assert(detailed.find("Signature:") != std::string::npos);
    assert(detailed.find("Algorithm:") != std::string::npos);
    assert(detailed.find("Arch:") != std::string::npos);
    std::cout << "PASSED\n";
}

void test_predefined_grouped_conv_configs()
{
    std::cout << "  test_predefined_grouped_conv_configs... ";
    configs::Memory<float> mem_cfg;
    assert(mem_cfg.algorithm.pipeline == PipelineVersion::MEMORY);
    assert(mem_cfg.algorithm.tile.m == 128);
    assert(mem_cfg.algorithm.tile.n == 32);

    configs::CompV3_Small<float> compv3_small;
    assert(compv3_small.algorithm.pipeline == PipelineVersion::V3);
    assert(compv3_small.algorithm.tile.m == 16);
    assert(compv3_small.algorithm.tile.n == 64);

    configs::CompV4<float> compv4;
    assert(compv4.algorithm.pipeline == PipelineVersion::V4);
    assert(compv4.algorithm.double_smem_buffer == true);

    configs::WMMA<float> wmma_cfg;
    assert(wmma_cfg.arch.name == "gfx1100");
    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Test Grouped Conv Config ===\n\n";
    test_grouped_conv_direction_enum();
    test_grouped_conv_signature_info();
    test_grouped_conv_algorithm_info();
    test_grouped_conv_config();
    test_predefined_grouped_conv_configs();
    std::cout << "\n=== All Tests Passed! ===\n\n";
    return 0;
}
