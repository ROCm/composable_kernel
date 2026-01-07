// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"
#include "ck_tile/host/device_prop.hpp"

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;
using enum ck_tile::builder::TensorLayout;

constexpr auto SIGNATURE = ckt::ConvSignature{.spatial_dim = 3,
                                              .direction   = ckb::ConvDirection::BACKWARD_WEIGHT,
                                              .data_type   = ckb::DataType::BF16,
                                              .accumulation_data_type = ckb::DataType::FP32,
                                              .input  = {.config = {.layout = NGCDHW}},
                                              .weight = {.config = {.layout = GKZYXC}},
                                              .output = {.config = {.layout = NGKDHW}}};

constexpr auto ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvBwdWeight_Wmma_CShuffle{}
                               .with_thread_block(cku::ThreadBlock_64_32x32x32)
                               .with_gemm_config(cku::GemmParams_Wmma_16x16_2x1_per_wave)
                               .with_transfer(cku::BwdTransfer_4x8x1_4x16x1_v3)
                               .with_bwd_specialization(ckb::ConvSpecialization::DEFAULT)
                               .with_prefetch_config(1, ckb::PipelineScheduler::DEFAULT)
                               .with_gridwise_gemm_pipeline(ckb::PipelineVersion::V1);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

TEST(BwdWeight_3DBf16_Wmma_CShuffle, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    std::cout << "Expected Transfer Parameters: " << expected_transfer_parameters << std::endl;
    cku::run_test<Builder>({"DeviceGroupedConvBwdWeight_Wmma_CShuffle",
                            expected_transfer_parameters,
                            "Default",
                            "NGCDHW,GKZYXC,NGKDHW",
                            "PassThrough,PassThrough,PassThrough",
                            "v1"});
}
