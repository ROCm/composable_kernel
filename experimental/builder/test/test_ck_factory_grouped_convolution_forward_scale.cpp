// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp>

#include "testing_utils.hpp"

using ck_tile::test::InstanceSet;
using ck_tile::test::InstancesMatch;

namespace {

constexpr static auto NumDimSpatial = 3;
using InLayout                      = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                     = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                     = ck::tensor_layout::convolution::NDHWGK;

using ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD;
using ck::tensor_operation::element_wise::PassThrough;
using ck::tensor_operation::element_wise::Scale;

template <typename T, typename ComputeType = T>
using DeviceOp = DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                 InLayout,
                                                 WeiLayout,
                                                 ck::Tuple<>, // DsLayout
                                                 OutLayout,
                                                 T,           // InDataType
                                                 T,           // WeiDataType
                                                 ck::Tuple<>, // DsDataType
                                                 T,           // OutDataType
                                                 PassThrough,
                                                 PassThrough,
                                                 Scale,
                                                 ComputeType>;

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

struct F32
{
    using DeviceOp = ::DeviceOp<float>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F32_TF32
{
    using DeviceOp = ::DeviceOp<float, ck::tf32_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F16
{
    using DeviceOp = ::DeviceOp<ck::half_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct BF16
{
    using DeviceOp = ::DeviceOp<ck::bhalf_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct S8
{
    using DeviceOp = ::DeviceOp<int8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

using TestTypes = ::testing::Types<F32, F32_TF32, F16, BF16, S8>;

TYPED_TEST_SUITE(CkFactoryTestConvFwd, TestTypes);

TYPED_TEST(CkFactoryTestConvFwd, TestInstances)
{
    auto actual   = TestFixture::get_actual_instances();
    auto expected = TestFixture::get_expected_instances();

    EXPECT_THAT(actual, InstancesMatch(expected));
}
