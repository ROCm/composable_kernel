// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file test_conv_config.cpp
 * @brief Unit tests for convolution configuration classes
 */

#include <iostream>
#include <cassert>
#include <string>

#include "ck_tile/dispatcher/conv_config.hpp"

using namespace ck_tile::dispatcher;

void test_conv_direction_enum()
{
    std::cout << "  test_conv_direction_enum... ";

    assert(ConvSignatureInfo::direction_str(ConvDirection::FORWARD) == std::string("fwd"));
    assert(ConvSignatureInfo::direction_str(ConvDirection::BACKWARD_DATA) == std::string("bwdd"));
    assert(ConvSignatureInfo::direction_str(ConvDirection::BACKWARD_WEIGHT) == std::string("bwdw"));

    std::cout << "PASSED\n";
}

void test_pipeline_version_enum()
{
    std::cout << "  test_pipeline_version_enum... ";

    assert(ConvAlgorithmInfo::pipeline_str(PipelineVersion::V3) == std::string("compv3"));
    assert(ConvAlgorithmInfo::pipeline_str(PipelineVersion::V4) == std::string("compv4"));
    assert(ConvAlgorithmInfo::pipeline_str(PipelineVersion::V5) == std::string("compv5"));
    assert(ConvAlgorithmInfo::pipeline_str(PipelineVersion::MEMORY) == std::string("mem"));

    std::cout << "PASSED\n";
}

void test_scheduler_enum()
{
    std::cout << "  test_scheduler_enum... ";

    assert(ConvAlgorithmInfo::scheduler_str(PipelineScheduler::DEFAULT) == std::string("default"));
    assert(ConvAlgorithmInfo::scheduler_str(PipelineScheduler::INTRAWAVE) ==
           std::string("intrawave"));
    assert(ConvAlgorithmInfo::scheduler_str(PipelineScheduler::INTERWAVE) ==
           std::string("interwave"));

    std::cout << "PASSED\n";
}

void test_conv_signature_info()
{
    std::cout << "  test_conv_signature_info... ";

    ConvSignatureInfo sig;

    // Test defaults
    assert(sig.spatial_dim == 2);
    assert(sig.direction == ConvDirection::FORWARD);
    assert(sig.in_type == "fp16");
    assert(sig.num_groups == 1);

    // Test modifications
    sig.spatial_dim = 3;
    sig.direction   = ConvDirection::BACKWARD_DATA;
    sig.in_type     = "bf16";

    assert(sig.spatial_dim == 3);
    assert(sig.direction == ConvDirection::BACKWARD_DATA);
    assert(sig.in_type == "bf16");

    std::cout << "PASSED\n";
}

void test_conv_algorithm_info()
{
    std::cout << "  test_conv_algorithm_info... ";

    ConvAlgorithmInfo algo;

    // Test defaults
    assert(algo.tile.m == 128);
    assert(algo.tile.n == 128);
    assert(algo.tile.k == 64);
    assert(algo.warp.m_warp == 2);
    assert(algo.warp.n_warp == 2);
    assert(algo.pipeline == PipelineVersion::V4);

    // Test modifications
    algo.tile.m      = 256;
    algo.tile.n      = 256;
    algo.warp.m_warp = 4;
    algo.pipeline    = PipelineVersion::V3;

    assert(algo.tile.m == 256);
    assert(algo.tile.n == 256);
    assert(algo.warp.m_warp == 4);
    assert(algo.pipeline == PipelineVersion::V3);

    std::cout << "PASSED\n";
}

void test_arch_info()
{
    std::cout << "  test_arch_info... ";

    ArchInfo arch;

    // Test defaults
    assert(arch.name == "gfx942");
    assert(arch.supports_mfma_fp16() == true);
    assert(arch.supports_wmma() == false);

    // Test gfx11xx
    ArchInfo arch2;
    arch2.name = "gfx1100";
    assert(arch2.supports_mfma_fp16() == false);
    assert(arch2.supports_wmma() == true);

    std::cout << "PASSED\n";
}

void test_conv_config()
{
    std::cout << "  test_conv_config... ";

    ConvConfig cfg;
    cfg.signature.in_type     = "fp16";
    cfg.signature.direction   = ConvDirection::FORWARD;
    cfg.signature.spatial_dim = 2;
    cfg.algorithm.tile.m      = 128;
    cfg.algorithm.tile.n      = 128;
    cfg.algorithm.tile.k      = 64;
    cfg.algorithm.pipeline    = PipelineVersion::V4;
    cfg.arch.name             = "gfx942";

    // Test name generation
    std::string name = cfg.name();
    assert(name.find("conv_fwd") != std::string::npos);
    assert(name.find("fp16") != std::string::npos);
    assert(name.find("2d") != std::string::npos);
    assert(name.find("compv4") != std::string::npos);

    // Test brief
    std::string brief = cfg.brief();
    assert(brief.find("2D") != std::string::npos);
    assert(brief.find("convolution") != std::string::npos);

    // Test detailed
    std::string detailed = cfg.detailed();
    assert(detailed.find("Signature") != std::string::npos);
    assert(detailed.find("Algorithm") != std::string::npos);
    assert(detailed.find("Arch") != std::string::npos);

    std::cout << "PASSED\n";
}

void test_predefined_configs()
{
    std::cout << "  test_predefined_configs... ";

    // Test Memory config
    configs::Memory<float> mem_cfg;
    assert(mem_cfg.algorithm.pipeline == PipelineVersion::MEMORY);

    // Test CompV3 configs
    configs::CompV3_Small<float> v3_small;
    assert(v3_small.algorithm.pipeline == PipelineVersion::V3);

    configs::CompV3_Medium<float> v3_med;
    assert(v3_med.algorithm.pipeline == PipelineVersion::V3);

    configs::CompV3_Large<float> v3_large;
    assert(v3_large.algorithm.pipeline == PipelineVersion::V3);

    // Test CompV4 config
    configs::CompV4<float> v4_cfg;
    assert(v4_cfg.algorithm.pipeline == PipelineVersion::V4);
    assert(v4_cfg.algorithm.double_smem_buffer == true);

    // Test CompV5 config
    configs::CompV5<float> v5_cfg;
    assert(v5_cfg.algorithm.pipeline == PipelineVersion::V5);

    // Test WMMA config
    configs::WMMA<float> wmma_cfg;
    assert(wmma_cfg.arch.name == "gfx1100");

    std::cout << "PASSED\n";
}

int main()
{
    std::cout << "\n=== Conv Config Tests ===\n\n";

    test_conv_direction_enum();
    test_pipeline_version_enum();
    test_scheduler_enum();
    test_conv_signature_info();
    test_conv_algorithm_info();
    test_arch_info();
    test_conv_config();
    test_predefined_configs();

    std::cout << "\n=== All Conv Config Tests Passed! ===\n\n";
    return 0;
}
