// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale_relu.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale_add.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convinvscale.hpp>
#include <ck/library/tensor_operation_instance/device_operation_instance_factory.hpp>
#include "testing_utils.hpp"

using ck_tile::test::InstanceSet;
using ck_tile::test::InstancesMatch;

namespace {

constexpr static auto NumDimSpatial = 3;
using InLayout                      = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                     = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                     = ck::tensor_layout::convolution::NDHWGK;

using ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD;
using ck::tensor_operation::device::instance::CombConvScale;
using ck::tensor_operation::device::instance::CombConvScaleRelu;
using ck::tensor_operation::element_wise::ConvInvscale;
using ck::tensor_operation::element_wise::ConvScale;
using ck::tensor_operation::element_wise::ConvScaleAdd;
using ck::tensor_operation::element_wise::ConvScaleRelu;
using ck::tensor_operation::element_wise::PassThrough;

template <typename DsLayout,
          typename DsDataType,
          typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename Act,
          typename AComputeType,
          typename BComputeType>
using DeviceOp = DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                 InLayout,
                                                 WeiLayout,
                                                 DsLayout,
                                                 OutLayout,
                                                 InDataType,  // InDataType
                                                 WeiDataType, // WeiDataType
                                                 DsDataType,
                                                 OutDataType, // OutDataType
                                                 PassThrough,
                                                 PassThrough,
                                                 Act,
                                                 AComputeType,
                                                 BComputeType>;

} // namespace

template <typename Case>
struct CkFactoryTestConvFwd : public testing::Test
{
    static auto get_actual_instances()
    {
        return InstanceSet::from_factory<typename Case::DeviceOp>();
    }

    static auto get_expected_instances() { return InstanceSet(Case::expected); }
};

struct F8_ConvScale
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>,
                                ck::Tuple<>,
                                ck::f8_t,
                                ck::f8_t,
                                ck::f8_t,
                                ConvScale,
                                ck::f8_t,
                                ck::f8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_BF8_comb1_ConvScale
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>,
                                ck::Tuple<>,
                                ck::bf8_t,
                                ck::bf8_t,
                                ck::f8_t,
                                ConvScale,
                                ck::bf8_t,
                                ck::bf8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_BF8_comb2_ConvScale
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>,
                                ck::Tuple<>,
                                ck::f8_t,
                                ck::bf8_t,
                                ck::f8_t,
                                ConvScale,
                                ck::f8_t,
                                ck::bf8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_BF8_comb3_ConvScale
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>,
                                ck::Tuple<>,
                                ck::bf8_t,
                                ck::f8_t,
                                ck::f8_t,
                                ConvScale,
                                ck::bf8_t,
                                ck::f8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_float_CombConvScale
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>,
                                ck::Tuple<>,
                                ck::f8_t,
                                ck::f8_t,
                                float,
                                CombConvScale,
                                ck::f8_t,
                                ck::f8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_ConvScaleRelu
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>,
                                ck::Tuple<>,
                                ck::f8_t,
                                ck::f8_t,
                                ck::f8_t,
                                ConvScaleRelu,
                                ck::f8_t,
                                ck::f8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_CombConvScaleRelu
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>,
                                ck::Tuple<>,
                                ck::f8_t,
                                ck::f8_t,
                                float,
                                CombConvScaleRelu,
                                ck::f8_t,
                                ck::f8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_ConvScaleAdd
{
    using DeviceOp = ::DeviceOp<ck::Tuple<OutLayout>,
                                ck::Tuple<float>,
                                ck::f8_t,
                                ck::f8_t,
                                ck::f8_t,
                                ConvScaleAdd,
                                ck::f8_t,
                                ck::f8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_ConvInvscale
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>,
                                ck::Tuple<>,
                                ck::f8_t,
                                ck::f8_t,
                                ck::f8_t,
                                ConvInvscale,
                                ck::f8_t,
                                ck::f8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

using TestTypes = ::testing::Types<F8_ConvScale,
                                   F8_BF8_comb1_ConvScale,
                                   F8_BF8_comb2_ConvScale,
                                   F8_BF8_comb3_ConvScale,
                                   F8_float_CombConvScale,
                                   F8_ConvScaleRelu,
                                   F8_CombConvScaleRelu,
                                   F8_ConvScaleAdd,
                                   F8_ConvInvscale>;

TYPED_TEST_SUITE(CkFactoryTestConvFwd, TestTypes);

TYPED_TEST(CkFactoryTestConvFwd, TestInstances)
{
    auto actual   = TestFixture::get_actual_instances();
    auto expected = TestFixture::get_expected_instances();

    EXPECT_THAT(actual, InstancesMatch(expected));
}
