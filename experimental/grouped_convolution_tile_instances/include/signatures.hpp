// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>

#include "../../builder/test/impl/conv_signature_types.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"

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

/////////////////////////////////////////
// FWD signatures (NGCHW / NGCDHW)
//////////////////////////////////////////

constexpr auto SIGNATURE_NGCHW_FP32_FWD =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP32,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NGCHW}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKCYX}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NGKHW}}};

constexpr auto SIGNATURE_NGCHW_FP16_FWD =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NGCHW}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKCYX}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NGKHW}}};

constexpr auto SIGNATURE_NGCHW_BF16_FWD =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NGCHW}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKCYX}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NGKHW}}};
/////////////////////////////////////////
// BWD WEIGHT signatures
//////////////////////////////////////////

constexpr auto SIGNATURE_NHWGC_BF16_BWD_WEIGHT =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto SIGNATURE_NHWGC_FP16_BWD_WEIGHT =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto SIGNATURE_NHWGC_FP32_BWD_WEIGHT =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
                       .data_type              = ckb::DataType::FP32,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto SIGNATURE_NDHWGC_BF16_BWD_WEIGHT =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

constexpr auto SIGNATURE_NDHWGC_FP16_BWD_WEIGHT =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

constexpr auto SIGNATURE_NDHWGC_FP32_BWD_WEIGHT =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::BACKWARD_WEIGHT,
                       .data_type              = ckb::DataType::FP32,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

/////////////////////////////////////////
// BWD DATA signatures
//////////////////////////////////////////

constexpr auto SIGNATURE_NHWGC_BF16_BWD_DATA =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::BACKWARD_DATA,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto SIGNATURE_NHWGC_FP16_BWD_DATA =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::BACKWARD_DATA,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto SIGNATURE_NHWGC_FP32_BWD_DATA =
    ckt::ConvSignature{.spatial_dim            = 2,
                       .direction              = ckb::ConvDirection::BACKWARD_DATA,
                       .data_type              = ckb::DataType::FP32,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NHWGK}}};

constexpr auto SIGNATURE_NDHWGC_BF16_BWD_DATA =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::BACKWARD_DATA,
                       .data_type              = ckb::DataType::BF16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

constexpr auto SIGNATURE_NDHWGC_FP16_BWD_DATA =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::BACKWARD_DATA,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

constexpr auto SIGNATURE_NDHWGC_FP32_BWD_DATA =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::BACKWARD_DATA,
                       .data_type              = ckb::DataType::FP32,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

} // namespace ck_tile::builder::profiling
