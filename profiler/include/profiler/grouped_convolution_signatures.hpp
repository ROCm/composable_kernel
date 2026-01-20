// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>

#include "../../experimental/builder/test/impl/conv_signature_types.hpp"
#include "ck_tile/builder/testing/conv_fwd_ck_tile.hpp"

namespace ck_tile::builder::profiling {

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;

constexpr auto SIGNATURE_NHWGC_FP32_FWD =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP32,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto SIGNATURE_NHWGC_BF16_FWD =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto SIGNATURE_NHWGC_FP16_FWD =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto SIGNATURE_NDHWGC_FP32_FWD =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP32,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

constexpr auto SIGNATURE_NDHWGC_BF16_FWD =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

constexpr auto SIGNATURE_NDHWGC_FP16_FWD =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

} // namespace ck_tile::builder::profiling
