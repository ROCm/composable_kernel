// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale_relu.hpp>
#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_convscale_add.hpp>

#include "testing_utils.hpp"

using ck_tile::test::InstanceSet;
using ck_tile::test::InstancesMatch;

namespace {

constexpr static auto NumDimSpatial = 3;
using InLayout                      = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                     = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                     = ck::tensor_layout::convolution::NDHWGK;

using ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD;
using ck::tensor_operation::device::instance::CombConvScaleRelu;
using ck::tensor_operation::element_wise::ConvScaleAdd;
using ck::tensor_operation::element_wise::ConvScaleRelu;
using ck::tensor_operation::element_wise::PassThrough;

template <typename DsLayout, typename DsDataType, typename OutDataType, typename Act>
using DeviceOp = DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                 InLayout,
                                                 WeiLayout,
                                                 DsLayout,
                                                 OutLayout,
                                                 ck::f8_t, // InDataType
                                                 ck::f8_t, // WeiDataType
                                                 DsDataType,
                                                 OutDataType, // OutDataType
                                                 PassThrough,
                                                 PassThrough,
                                                 Act>;

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

struct F8_ConvScaleRelu
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>, ck::Tuple<>, ck::f8_t, ConvScaleRelu>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_CombConvScaleRelu
{
    using DeviceOp = ::DeviceOp<ck::Tuple<>, ck::Tuple<>, float, CombConvScaleRelu>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F8_ConvScaleAdd
{
    using DeviceOp = ::DeviceOp<ck::Tuple<OutLayout>, ck::Tuple<float>, ck::f8_t, ConvScaleAdd>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

using TestTypes = ::testing::Types<F8_ConvScaleRelu, F8_CombConvScaleRelu, F8_ConvScaleAdd>;

TYPED_TEST_SUITE(CkFactoryTestConvFwd, TestTypes);

TYPED_TEST(CkFactoryTestConvFwd, TestInstances)
{
    auto actual   = TestFixture::get_actual_instances();
    auto expected = TestFixture::get_expected_instances();

    EXPECT_THAT(actual, InstancesMatch(expected));
}
