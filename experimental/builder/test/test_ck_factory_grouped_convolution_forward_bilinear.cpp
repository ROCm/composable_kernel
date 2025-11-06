// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp>
#include "ck/utility/data_type.hpp"
#include "testing_utils.hpp"

using ck_tile::test::InstanceSet;
using ck_tile::test::InstancesMatch;

namespace {

constexpr static auto NumDimSpatial = 3;
using InLayout                      = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                     = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                     = ck::tensor_layout::convolution::NDHWGK;
using DsLayout                      = ck::Tuple<ck::tensor_layout::convolution::NDHWGK>;

using ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD;
using ck::tensor_operation::element_wise::Bilinear;
using ck::tensor_operation::element_wise::PassThrough;

template <typename type, typename computeType = type>
using DeviceOp = DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                 InLayout,
                                                 WeiLayout,
                                                 DsLayout,
                                                 OutLayout,
                                                 type, // InDataType
                                                 type, // WeiDataType
                                                 ck::Tuple<type>,
                                                 type, // OutDataType
                                                 PassThrough,
                                                 PassThrough,
                                                 Bilinear,
                                                 computeType,
                                                 computeType>;

} // namespace

template <typename Case>
struct CkFactoryTestBilinearFwd : public testing::Test
{
    static auto get_actual_instances()
    {
        return InstanceSet::from_factory<typename Case::DeviceOp>();
    }

    static auto get_expected_instances() { return InstanceSet(Case::expected); }
};

struct Bilinear_F32
{
    using DeviceOp = ::DeviceOp<float>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct Bilinear_F32_TF32
{
    using DeviceOp = ::DeviceOp<float, ck::tf32_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct Bilinear_F16
{
    using DeviceOp = ::DeviceOp<ck::half_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct Bilinear_BF16
{
    using DeviceOp = ::DeviceOp<ck::bhalf_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct Bilinear_INT8
{
    using DeviceOp = ::DeviceOp<int8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

using TestTypes =
    ::testing::Types<Bilinear_F32, Bilinear_F32_TF32, Bilinear_F16, Bilinear_BF16, Bilinear_INT8>;

TYPED_TEST_SUITE(CkFactoryTestBilinearFwd, TestTypes);

TYPED_TEST(CkFactoryTestBilinearFwd, TestInstances)
{
    auto actual   = TestFixture::get_actual_instances();
    auto expected = TestFixture::get_expected_instances();

    EXPECT_THAT(actual, InstancesMatch(expected));
}
