// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <type_traits>

#include "ck_tile/builder/factory/helpers/conv_tensor_layout.hpp"
#include "impl/conv_signature_types.hpp"

namespace {

namespace ckb = ::ck_tile::builder;
using ::ck_tile::builder::DataType;
using ::ck_tile::builder::ElementwiseOperation;
using ::ck_tile::builder::TensorLayout;
using ::ck_tile::builder::factory::internal::AuxiliaryTensorLayouts;
using ::ck_tile::builder::factory::internal::ConvTensorLayouts;
using ::ck_tile::builder::factory::internal::LayoutToCK;

using namespace ::ck_tile::builder::test;
using enum ::ck_tile::builder::ConvDirection;

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NWGC_GKXC_NWGK)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 1,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::NWGC}},
                        .weight                 = {.config = {.layout = TensorLayout::GKXC}},
                        .output                 = {.config = {.layout = TensorLayout::NWGK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NWGK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NGCW_GKXC_NGKW)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 1,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::NGCW}},
                        .weight                 = {.config = {.layout = TensorLayout::GKXC}},
                        .output                 = {.config = {.layout = TensorLayout::NGKW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_GNWC_GKXC_GNWK)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 1,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::GNWC}},
                        .weight                 = {.config = {.layout = TensorLayout::GKXC}},
                        .output                 = {.config = {.layout = TensorLayout::GNWK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNWK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor1D_NGCW_GKCX_NGKW)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 1,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::NGCW}},
                        .weight                 = {.config = {.layout = TensorLayout::GKCX}},
                        .output                 = {.config = {.layout = TensorLayout::NGKW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NGCHW_GKYXC_NGKHW)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 2,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::NGCHW}},
                        .weight                 = {.config = {.layout = TensorLayout::GKYXC}},
                        .output                 = {.config = {.layout = TensorLayout::NGKHW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NHWGC_GKYXC_NHWGK)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 2,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::NHWGC}},
                        .weight                 = {.config = {.layout = TensorLayout::GKYXC}},
                        .output                 = {.config = {.layout = TensorLayout::NHWGK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NHWGK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_GNHWC_GKYXC_GNHWK)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 2,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::GNHWC}},
                        .weight                 = {.config = {.layout = TensorLayout::GKYXC}},
                        .output                 = {.config = {.layout = TensorLayout::GNHWK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNHWK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor2D_NGCHW_GKCYX_NGKHW)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 2,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::NGCHW}},
                        .weight                 = {.config = {.layout = TensorLayout::GKCYX}},
                        .output                 = {.config = {.layout = TensorLayout::NGKHW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCYX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_NGCDHW_GKCZYX_NGKDHW)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 3,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::NGCDHW}},
                        .weight                 = {.config = {.layout = TensorLayout::GKCZYX}},
                        .output                 = {.config = {.layout = TensorLayout::NGKDHW}}};

    using TensorLayouts = ConvTensorLayouts<sig, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCDHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKCZYX>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKDHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_NDHWGC_GKZYXC_NDHWGK)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 3,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::NDHWGC}},
                        .weight                 = {.config = {.layout = TensorLayout::GKZYXC}},
                        .output                 = {.config = {.layout = TensorLayout::NDHWGK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NDHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NDHWGK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(ConvTensorLayout, AssignsLayoutsFor3D_GNDHWC_GKZYXC_GNDHWK)
{
    static constexpr auto sig =
        ConvSignature<>{.spatial_dim            = 3,
                        .direction              = FORWARD,
                        .data_type              = DataType::FP16,
                        .accumulation_data_type = DataType::FP32,
                        .input                  = {.config = {.layout = TensorLayout::GNDHWC}},
                        .weight                 = {.config = {.layout = TensorLayout::GKZYXC}},
                        .output                 = {.config = {.layout = TensorLayout::GNDHWK}}};

    using TensorLayouts = ConvTensorLayouts<sig, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNDHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNDHWK>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ck::Tuple<>>));
}

TEST(AuxiliaryTensorLayout, AssignsLayoutForG_K_strided)
{
    using CKLayout = LayoutToCK<TensorLayout::G_K_strided>::type;
    EXPECT_TRUE((std::is_same_v<CKLayout, ck::tensor_layout::convolution::G_K>));
}

TEST(AuxiliaryTensorLayout, AssignsLayoutForGC)
{
    using CKLayout = LayoutToCK<TensorLayout::GC>::type;
    EXPECT_TRUE((std::is_same_v<CKLayout, ck::tensor_layout::convolution::GC>));
}

TEST(AuxiliaryTensorLayout, AssignsLayoutForG_C_strided)
{
    using CKLayout = LayoutToCK<TensorLayout::G_C_strided>::type;
    EXPECT_TRUE((std::is_same_v<CKLayout, ck::tensor_layout::convolution::G_C>));
}

TEST(AuxiliaryTensorLayout, EmptyAuxiliaryTensorLayoutIsEmptyTuple)
{
    using ::ck_tile::builder::factory::internal::EmptyAuxiliaryTensorLayout;
    using EmptyLayout = EmptyAuxiliaryTensorLayout::type;
    EXPECT_TRUE((std::is_same_v<EmptyLayout, ck::Tuple<>>));
}

struct MockAuxiliaryTensorConfig
{
    TensorLayout layout;
};

TEST(AuxiliaryTensorLayoutIntegration, SingleBiasTensorWithG_K_Layout)
{
    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = TensorLayout::G_K_strided}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::G_K>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, SingleBiasTensorWithGC_Layout)
{
    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = TensorLayout::GC}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, SingleBiasTensorWithG_C_Layout)
{
    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = TensorLayout::G_C_strided}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::G_C>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, TwoAuxiliaryTensors)
{
    static constexpr std::array<MockAuxiliaryTensorConfig, 2> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = TensorLayout::G_K_strided},
        MockAuxiliaryTensorConfig{.layout = TensorLayout::GC}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 2);
    using ExpectedType =
        ck::Tuple<ck::tensor_layout::convolution::G_K, ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, ThreeAuxiliaryTensors)
{
    static constexpr std::array<MockAuxiliaryTensorConfig, 3> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = TensorLayout::G_K_strided},
        MockAuxiliaryTensorConfig{.layout = TensorLayout::GC},
        MockAuxiliaryTensorConfig{.layout = TensorLayout::G_C_strided}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 2, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 3);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::G_K,
                                   ck::tensor_layout::convolution::GC,
                                   ck::tensor_layout::convolution::G_C>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, WorksWith1DConvolution)
{
    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = TensorLayout::G_K_strided}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 1, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::G_K>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(AuxiliaryTensorLayoutIntegration, WorksWith3DConvolution)
{
    static constexpr std::array<MockAuxiliaryTensorConfig, 1> aux_configs = {
        MockAuxiliaryTensorConfig{.layout = TensorLayout::GC}};

    using AuxLayouts = AuxiliaryTensorLayouts<aux_configs, 3, FORWARD>;

    EXPECT_EQ(AuxLayouts::Size, 1);
    using ExpectedType = ck::Tuple<ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<AuxLayouts::type, ExpectedType>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv2DWithSingleBiasG_K)
{
    using OutputOp = TensorOperation<TensorConfig{.layout = TensorLayout::G_K_strided}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 2,
            .direction              = FORWARD,
            .data_type              = DataType::FP16,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = TensorLayout::NGCHW}},
            .weight                 = {.config = {.layout = TensorLayout::GKYXC}},
            .output                 = {.config = {.layout = TensorLayout::NGKHW},
                                       .operation =
                                           OutputOp{.elementwise_operation = ElementwiseOperation::SCALE}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NGCHW>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NGKHW>));

    using ExpectedDsLayout = ck::Tuple<ck::tensor_layout::convolution::G_K>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv2DWithSingleBiasGC)
{
    using OutputOp = TensorOperation<TensorConfig{.layout = TensorLayout::GC}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 2,
            .direction              = FORWARD,
            .data_type              = DataType::BF16,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = TensorLayout::NHWGC}},
            .weight                 = {.config = {.layout = TensorLayout::GKYXC}},
            .output                 = {.config = {.layout = TensorLayout::NHWGK},
                                       .operation =
                                           OutputOp{.elementwise_operation = ElementwiseOperation::SCALE}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NHWGK>));

    using ExpectedDsLayout = ck::Tuple<ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv2DWithTwoAuxiliaryTensors)
{
    using OutputOp = TensorOperation<TensorConfig{.layout = TensorLayout::G_K_strided},
                                     TensorConfig{.layout = TensorLayout::GC}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 2,
            .direction              = FORWARD,
            .data_type              = DataType::FP16,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = TensorLayout::GNHWC}},
            .weight                 = {.config = {.layout = TensorLayout::GKYXC}},
            .output                 = {.config    = {.layout = TensorLayout::GNHWK},
                                       .operation = OutputOp{.elementwise_operation =
                                                 ElementwiseOperation::SCALEADD_SCALEADD_RELU}}};

    using TensorLayouts = ConvTensorLayouts<sig, 2, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::GNHWC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::GNHWK>));

    using ExpectedDsLayout =
        ck::Tuple<ck::tensor_layout::convolution::G_K, ck::tensor_layout::convolution::GC>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv1DWithBias)
{
    using OutputOp = TensorOperation<TensorConfig{.layout = TensorLayout::G_K_strided}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 1,
            .direction              = FORWARD,
            .data_type              = DataType::FP32,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = TensorLayout::NWGC}},
            .weight                 = {.config = {.layout = TensorLayout::GKXC}},
            .output                 = {.config = {.layout = TensorLayout::NWGK},
                                       .operation =
                                           OutputOp{.elementwise_operation = ElementwiseOperation::SCALE}}};

    using TensorLayouts = ConvTensorLayouts<sig, 1, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NWGK>));

    using ExpectedDsLayout = ck::Tuple<ck::tensor_layout::convolution::G_K>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

TEST(ConvTensorLayoutsWithAuxiliary, Conv3DWithBias)
{
    using OutputOp = TensorOperation<TensorConfig{.layout = TensorLayout::G_C_strided}>;

    static constexpr auto sig =
        ConvSignature<ConvolutionTensor<>, ConvolutionTensor<>, ConvolutionTensor<OutputOp>>{
            .spatial_dim            = 3,
            .direction              = FORWARD,
            .data_type              = DataType::FP16,
            .accumulation_data_type = DataType::FP32,
            .input                  = {.config = {.layout = TensorLayout::NDHWGC}},
            .weight                 = {.config = {.layout = TensorLayout::GKZYXC}},
            .output                 = {.config    = {.layout = TensorLayout::NDHWGK},
                                       .operation = OutputOp{.elementwise_operation =
                                                 ElementwiseOperation::BIAS_BNORM_CLAMP}}};

    using TensorLayouts = ConvTensorLayouts<sig, 3, FORWARD>;

    EXPECT_TRUE((std::is_same_v<TensorLayouts::ALayout, ck::tensor_layout::convolution::NDHWGC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::BLayout, ck::tensor_layout::convolution::GKZYXC>));
    EXPECT_TRUE((std::is_same_v<TensorLayouts::ELayout, ck::tensor_layout::convolution::NDHWGK>));

    using ExpectedDsLayout = ck::Tuple<ck::tensor_layout::convolution::G_C>;
    EXPECT_TRUE((std::is_same_v<TensorLayouts::DsLayout, ExpectedDsLayout>));
}

} // namespace
