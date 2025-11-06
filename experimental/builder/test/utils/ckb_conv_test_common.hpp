// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include "impl/conv_algorithm_types.hpp"
#include "impl/conv_signature_types.hpp"
#include "ck_tile/builder/conv_builder.hpp"

namespace ck_tile::builder::test_utils {

using namespace ck_tile::builder;
using namespace test;

// Common test implementation
template <ConvSignature FwdConvSignature,
          ThreadBlock FwdThreadBlock,
          PipelineVersion FwdPipelineVersion,
          ConvFwdSpecialization FwdConvSpecialization>
constexpr void run_test_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3()
{
    constexpr GridwiseXdlGemm FwdGemmParams{.ak1            = 8,
                                            .bk1            = 8,
                                            .m_per_xdl      = 32,
                                            .n_per_xdl      = 32,
                                            .m_xdl_per_wave = 4,
                                            .n_xdl_per_wave = 4};

    constexpr BlockTransferABC FwdBlockTransfer{.block_transfer_a = {.k0 = 4, .m_n = 64, .k1 = 1},
                                                .block_transfer_b = {.k0 = 4, .m_n = 64, .k1 = 1},
                                                .thread_cluster_dims_c = {.m_block        = 1,
                                                                          .m_wave_per_xdl = 32,
                                                                          .n_block        = 1,
                                                                          .n_wave_per_xdl = 8},
                                                .lds_transfer_a        = {.src_vector_dim            = 2,
                                                                          .src_scalar_per_vector     = 2,
                                                                          .lds_dst_scalar_per_vector = 8,
                                                                          .is_direct_load = false,
                                                                          .lds_padding    = false},
                                                .lds_transfer_b        = {.src_vector_dim            = 2,
                                                                          .src_scalar_per_vector     = 8,
                                                                          .lds_dst_scalar_per_vector = 8,
                                                                          .is_direct_load = false,
                                                                          .lds_padding    = false},
                                                .epilogue_c = {.m_per_wave_per_shuffle = 1,
                                                               .n_per_wave_per_shuffle = 1,
                                                               .scalar_per_vector      = 8},
                                                .block_transfer_access_order_a = {1, 0, 2},
                                                .block_transfer_access_order_b = {1, 0, 2},
                                                .src_access_order_a            = {1, 0, 2},
                                                .src_access_order_b            = {1, 0, 2}};

    constexpr BlockGemm BlockGemmDesc = {.pipeline_version = FwdPipelineVersion,
                                         .scheduler        = PipelineScheduler::INTRAWAVE};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3 FwdConvAlgorithm{
        .thread_block        = FwdThreadBlock,
        .gridwise_gemm       = FwdGemmParams,
        .block_transfer      = FwdBlockTransfer,
        .fwd_specialization  = FwdConvSpecialization,
        .gemm_specialization = GemmSpecialization::MNKPadding,
        .block_gemm          = BlockGemmDesc};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    auto instance = typename Builder::Instance{};

    const auto kernel_string = instance.GetTypeString();
    std::cout << "Generated kernel: " << kernel_string << std::endl;
    EXPECT_GT(kernel_string.size(), 0);

    EXPECT_TRUE(kernel_string.starts_with("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3"));

    // Verify pipeline version is correct
    if(FwdPipelineVersion == PipelineVersion::V1)
        EXPECT_TRUE(kernel_string.find("BlkGemmPipelineVersion: v1") != std::string::npos);
    else if(FwdPipelineVersion == PipelineVersion::V3)
        EXPECT_TRUE(kernel_string.find("BlkGemmPipelineVersion: v3") != std::string::npos);
    else if(FwdPipelineVersion == PipelineVersion::V4)
        EXPECT_TRUE(kernel_string.find("BlkGemmPipelineVersion: v4") != std::string::npos);
    else if(FwdPipelineVersion == PipelineVersion::V5)
        EXPECT_TRUE(kernel_string.find("BlkGemmPipelineVersion: v5") != std::string::npos);

    // Verify specialization is correct
    if(FwdConvSpecialization == ConvFwdSpecialization::DEFAULT)
        EXPECT_TRUE(kernel_string.find("Default") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_1X1_PAD0)
        EXPECT_TRUE(kernel_string.find("Filter1x1Pad0") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0)
        EXPECT_TRUE(kernel_string.find("Filter1x1Stride1Pad0") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_3x3)
        EXPECT_TRUE(kernel_string.find("Filter3x3") != std::string::npos);

    const auto invoker_ptr = instance.MakeInvokerPointer();
    EXPECT_NE(invoker_ptr, nullptr);
}

template <ConvSignature FwdConvSignature,
          ThreadBlock FwdThreadBlock,
          ConvFwdSpecialization FwdConvSpecialization>
constexpr void run_test_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle()
{
    constexpr GridwiseXdlGemm FwdGemmParams{.ak1            = 8,
                                            .bk1            = 8,
                                            .m_per_xdl      = 32,
                                            .n_per_xdl      = 32,
                                            .m_xdl_per_wave = 2,
                                            .n_xdl_per_wave = 1};

    constexpr BlockTransferABC FwdBlockTransfer{.block_transfer_a = {.k0 = 4, .m_n = 16, .k1 = 1},
                                                .block_transfer_b = {.k0 = 4, .m_n = 16, .k1 = 1},
                                                .thread_cluster_dims_c = {.m_block        = 1,
                                                                          .m_wave_per_xdl = 16,
                                                                          .n_block        = 1,
                                                                          .n_wave_per_xdl = 4},
                                                .lds_transfer_a        = {.src_vector_dim            = 2,
                                                                          .src_scalar_per_vector     = 8,
                                                                          .lds_dst_scalar_per_vector = 8,
                                                                          .is_direct_load = false,
                                                                          .lds_padding    = true},
                                                .lds_transfer_b        = {.src_vector_dim            = 2,
                                                                          .src_scalar_per_vector     = 8,
                                                                          .lds_dst_scalar_per_vector = 8,
                                                                          .is_direct_load = false,
                                                                          .lds_padding    = true},
                                                .epilogue_c = {.m_per_wave_per_shuffle = 1,
                                                               .n_per_wave_per_shuffle = 1,
                                                               .scalar_per_vector      = 8},
                                                .block_transfer_access_order_a = {1, 0, 2},
                                                .block_transfer_access_order_b = {1, 0, 2},
                                                .src_access_order_a            = {1, 0, 2},
                                                .src_access_order_b            = {1, 0, 2}};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle FwdConvAlgorithm{
        .thread_block               = FwdThreadBlock,
        .gridwise_gemm              = FwdGemmParams,
        .block_transfer             = FwdBlockTransfer,
        .fwd_specialization         = FwdConvSpecialization,
        .gemm_specialization        = GemmSpecialization::MNKPadding,
        .num_gemm_k_prefetch_stages = 1,
        .num_groups_to_merge        = 2,
        .loop_scheduler             = PipelineScheduler::DEFAULT};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    auto instance = typename Builder::Instance{};

    const auto kernel_string = instance.GetTypeString();
    std::cout << "Generated kernel: " << kernel_string << std::endl;
    EXPECT_GT(kernel_string.size(), 0);

    EXPECT_TRUE(kernel_string.starts_with("DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle"));

    // Verify specialization is correct
    if(FwdConvSpecialization == ConvFwdSpecialization::DEFAULT)
        EXPECT_TRUE(kernel_string.find("Default") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_1X1_PAD0)
        EXPECT_TRUE(kernel_string.find("Filter1x1Pad0") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0)
        EXPECT_TRUE(kernel_string.find("Filter1x1Stride1Pad0") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_3x3)
        EXPECT_TRUE(kernel_string.find("Filter3x3") != std::string::npos);

    const auto invoker_ptr = instance.MakeInvokerPointer();
    EXPECT_NE(invoker_ptr, nullptr);
}

template <ConvSignature FwdConvSignature,
          ThreadBlock FwdThreadBlock,
          ConvFwdSpecialization FwdConvSpecialization>
constexpr void run_test_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle()
{
    constexpr GridwiseWmmaGemm FwdGemmParams{.k1               = 8,
                                             .m_per_wmma       = 32,
                                             .n_per_wmma       = 32,
                                             .m_wmma_per_wave  = 2,
                                             .n_wmma_per_wave  = 1,
                                             .pipeline_version = PipelineVersion::V1};

    constexpr BlockTransferABC FwdBlockTransfer{.block_transfer_a = {.k0 = 4, .m_n = 32, .k1 = 1},
                                                .block_transfer_b = {.k0 = 4, .m_n = 32, .k1 = 1},
                                                .thread_cluster_dims_c = {.m_block        = 1,
                                                                          .m_wave_per_xdl = 32,
                                                                          .n_block        = 1,
                                                                          .n_wave_per_xdl = 4},
                                                .lds_transfer_a        = {.src_vector_dim            = 2,
                                                                          .src_scalar_per_vector     = 16,
                                                                          .lds_dst_scalar_per_vector = 16,
                                                                          .is_direct_load = false,
                                                                          .lds_padding    = true},
                                                .lds_transfer_b        = {.src_vector_dim            = 2,
                                                                          .src_scalar_per_vector     = 16,
                                                                          .lds_dst_scalar_per_vector = 16,
                                                                          .is_direct_load = false,
                                                                          .lds_padding    = true},
                                                .epilogue_c = {.m_per_wave_per_shuffle = 1,
                                                               .n_per_wave_per_shuffle = 1,
                                                               .scalar_per_vector      = 8},
                                                .block_transfer_access_order_a = {1, 0, 2},
                                                .block_transfer_access_order_b = {1, 0, 2},
                                                .src_access_order_a            = {1, 0, 2},
                                                .src_access_order_b            = {1, 0, 2}};

    constexpr ConvAlgorithm_DeviceGroupedConvFwdMultipleD_Wmma_CShuffle FwdConvAlgorithm{
        .thread_block               = FwdThreadBlock,
        .gridwise_gemm              = FwdGemmParams,
        .block_transfer             = FwdBlockTransfer,
        .fwd_specialization         = FwdConvSpecialization,
        .gemm_specialization        = GemmSpecialization::MNKPadding,
        .num_gemm_k_prefetch_stages = 1,
        .loop_scheduler             = PipelineScheduler::DEFAULT};

    using Builder = ConvBuilder<FwdConvSignature, FwdConvAlgorithm>;

    auto instance = typename Builder::Instance{};

    const auto kernel_string = instance.GetTypeString();
    std::cout << "Generated kernel: " << kernel_string << std::endl;
    EXPECT_GT(kernel_string.size(), 0);

    EXPECT_TRUE(kernel_string.starts_with("DeviceGroupedConvFwdMultipleD_Wmma_CShuffle"));

    // Verify specialization is correct
    if(FwdConvSpecialization == ConvFwdSpecialization::DEFAULT)
        EXPECT_TRUE(kernel_string.find("Default") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_1X1_PAD0)
        EXPECT_TRUE(kernel_string.find("Filter1x1Pad0") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_1X1_STRIDE1_PAD0)
        EXPECT_TRUE(kernel_string.find("Filter1x1Stride1Pad0") != std::string::npos);
    else if(FwdConvSpecialization == ConvFwdSpecialization::FILTER_3x3)
        EXPECT_TRUE(kernel_string.find("Filter3x3") != std::string::npos);

    const auto invoker_ptr = instance.MakeInvokerPointer();
    EXPECT_NE(invoker_ptr, nullptr);
}

} // namespace ck_tile::builder::test_utils
