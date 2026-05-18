// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "utils/ckb_conv_test_configs.hpp"
#include "utils/ckb_conv_test_utils.hpp"
#include "utils/conv_algorithm_type_utils.hpp"

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

constexpr auto ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvBwdWeight_Dl{}
                               .with_thread_block(cku::ThreadBlock_256_128x128x16)
                               .with_bwd_specialization(cku::ConvSpecialization::DEFAULT)
                               .with_dl_thread_config(cku::DlThreadConfig_16x1x4x4x1)
                               .with_dl_thread_cluster(cku::DlThreadCluster_8x2)
                               .with_dl_transfer(cku::DlTransfer5D);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

TEST(BwdWeight_2DBf16_DL, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    std::cout << "Expected Transfer Parameters: " << expected_transfer_parameters << std::endl;
    cku::run_test<Builder>({"DeviceGroupedConvBwdWeight_Dl",
                            expected_transfer_parameters,
                            "Default",
                            "GNHWC,GKYXC,GNHWK",
                            "PassThrough,PassThrough,PassThrough"});
}
