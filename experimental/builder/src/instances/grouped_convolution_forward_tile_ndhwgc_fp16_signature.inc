namespace {

constexpr auto SIGNATURE =
    ckt::ConvSignature{.spatial_dim            = 3,
                       .direction              = ckb::ConvDirection::FORWARD,
                       .data_type              = ckb::DataType::FP16,
                       .accumulation_data_type = ckb::DataType::FP32,
                       .input                  = {.config = {.layout = ckb::TensorLayout::NDHWGC}},
                       .weight                 = {.config = {.layout = ckb::TensorLayout::GKZYXC}},
                       .output                 = {.config = {.layout = ckb::TensorLayout::NDHWGK}}};

} // namespace
