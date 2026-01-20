#include "../../builder/test/utils/ckb_conv_tile_test_configs.hpp"
#include "ck_tile/builder/testing/conv/fwd.hpp"
#include "ck_tile/builder/testing/conv/ck_tile.hpp"

namespace ckb = ck_tile::builder;
namespace ckt = ck_tile::builder::test;
namespace cku = ck_tile::builder::test_utils;

namespace ck_tile::builder::profiling {

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
