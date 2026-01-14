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
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::GNHWC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::GNHWK}}};

constexpr auto ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvBwdWeight_TwoStage_Xdl_CShuffle{}
                               .with_thread_block(cku::ThreadBlock_64_32x32x32)
                               .with_gemm_config(cku::BwdGemmParams_Xdl_1x1_per_wave)
                               .with_transfer(cku::BwdTransfer_4x8x1_4x16x1_v3)
                               .with_bwd_specialization(ckb::ConvSpecialization::DEFAULT)
                               .with_block_gemm(cku::BlockGemmDesc_v2_intrawave)
                               .with_num_conv_groups_to_merge(2)
                               .with_transpose_params(2, 4);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

TEST(BwdWeight_2DBf16_TwoStage_CShuffle, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    cku::run_test<Builder>({"DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle",
                            expected_transfer_parameters,
                            "Default",
                            "GNHWC,GKYXC,GNHWK",
                            "PassThrough,PassThrough,PassThrough",
                            "Intrawave,v2",     // pipeline versions
                            "bf16,bf16,2,4>"}); // compute types and transpose params
}
