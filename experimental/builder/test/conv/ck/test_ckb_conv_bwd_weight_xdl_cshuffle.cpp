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
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::GNHWC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::GNHWK}}};

constexpr auto ALGORITHM = cku::ConvAlgorithm_DeviceGroupedConvBwdWeight_Xdl_CShuffle{}
                               .with_thread_block(cku::ThreadBlock_256_128x128x8)
                               .with_gemm_config(cku::BwdGemmParams_Xdl_4x4_per_wave)
                               .with_transfer(cku::BwdTransfer_4x64x1)
                               .with_bwd_specialization(ckb::ConvSpecialization::DEFAULT)
                               .with_transpose_params(2, 2);

using Builder  = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using Instance = Builder::Instance;

TEST(BwdWeight_2DFp16_CShuffle_GNHWC, Create)
{
    const auto expected_transfer_parameters = to_string(ALGORITHM);
    cku::run_test<Builder>({"DeviceGroupedConvBwdWeight_Xdl_CShuffle",
                            expected_transfer_parameters,
                            "Default",
                            "GNHWC,GKYXC,GNHWK",
                            "PassThrough,PassThrough,PassThrough",
                            "fp16,fp16,2,2>"}); // check compute types and transpose params
}
