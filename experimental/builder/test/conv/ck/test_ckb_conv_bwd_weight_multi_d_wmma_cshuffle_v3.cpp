// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"
#include "ck_tile/host/device_prop.hpp"

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;

constexpr auto SIGNATURE =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::GNDHWC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::GNDHWK}}};

constexpr auto ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffle_V3{}
                               .with_thread_block(cku::ThreadBlock_64_32x32x32)
                               .with_gemm_config(cku::GemmParams_Wmma_16x16_2x1_per_wave)
                               .with_transfer(cku::BwdTransfer_4x8x1_4x16x1_v3)
                               .with_bwd_specialization(ckb::ConvSpecialization::DEFAULT)
                               .with_block_gemm(cku::BlockGemmDesc_v1_intrawave);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

TEST(BwdWeight_3DFp16_MultiD_Wmma_ShuffleV3_GNHWC, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    std::cout << "Expected Transfer Parameters: " << expected_transfer_parameters << std::endl;
    cku::run_test<Builder>({"DeviceGroupedConvBwdWeightMultipleD_Wmma_CShuffleV3",
                            expected_transfer_parameters,
                            "Default",
                            "GNDHWC,GKZYXC,GNDHWK",
                            "PassThrough,PassThrough,PassThrough",
                            "fp16,fp16>"}); // check compute types
}
