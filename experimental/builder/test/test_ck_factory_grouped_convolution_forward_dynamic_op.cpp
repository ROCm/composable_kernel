// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_dynamic_op.hpp>
#include "ck/utility/data_type.hpp"
#include "testing_utils.hpp"

using ck_tile::test::InstanceSet;
using ck_tile::test::InstancesMatch;

namespace {

using InLayout  = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
using OutLayout = ck::tensor_layout::convolution::NDHWGK;

using ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD;
using ck::tensor_operation::element_wise::DynamicUnaryOp;
using ck::tensor_operation::element_wise::PassThrough;

template <ck::index_t NumDimSpatial, typename T>
struct DeviceOpHelper;

template <typename T>
struct DeviceOpHelper<2, T>
{
    using InLayout  = ck::tensor_layout::convolution::NHWGC;
    using WeiLayout = ck::tensor_layout::convolution::GKYXC;
    using OutLayout = ck::tensor_layout::convolution::NHWGK;

    using Type = DeviceGroupedConvFwdMultipleABD<2,
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
                                                 DynamicUnaryOp>;
};

template <typename T>
struct DeviceOpHelper<3, T>
{
    using InLayout  = ck::tensor_layout::convolution::NDHWGC;
    using WeiLayout = ck::tensor_layout::convolution::GKZYXC;
    using OutLayout = ck::tensor_layout::convolution::NDHWGK;

    using Type = DeviceGroupedConvFwdMultipleABD<3,
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
                                                 DynamicUnaryOp>;
};

template <ck::index_t NumDimSpatial, typename T>
using DeviceOp = DeviceOpHelper<NumDimSpatial, T>::Type;

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

struct DyOp_F32_2
{
    using DeviceOp = ::DeviceOp<2, float>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct DyOp_F32_3
{
    using DeviceOp = ::DeviceOp<3, float>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct DyOp_F16_2
{
    using DeviceOp = ::DeviceOp<2, ck::half_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct DyOp_F16_3
{
    using DeviceOp = ::DeviceOp<3, ck::half_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct DyOp_BF16_2
{
    using DeviceOp = ::DeviceOp<2, ck::bhalf_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct DyOp_BF16_3
{
    using DeviceOp = ::DeviceOp<3, ck::bhalf_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct DyOp_INT8_2
{
    using DeviceOp = ::DeviceOp<2, int8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct DyOp_INT8_3
{
    using DeviceOp = ::DeviceOp<3, int8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

using TestTypes = ::testing::Types<DyOp_F32_2,
                                   DyOp_F32_3,
                                   DyOp_F16_2,
                                   DyOp_F16_3,
                                   DyOp_BF16_2,
                                   DyOp_BF16_3,
                                   DyOp_INT8_2,
                                   DyOp_INT8_3>;

TYPED_TEST_SUITE(CkFactoryTestBilinearFwd, TestTypes);

TYPED_TEST(CkFactoryTestBilinearFwd, TestInstances)
{
    auto actual   = TestFixture::get_actual_instances();
    auto expected = TestFixture::get_expected_instances();

    EXPECT_THAT(actual, InstancesMatch(expected));
}
