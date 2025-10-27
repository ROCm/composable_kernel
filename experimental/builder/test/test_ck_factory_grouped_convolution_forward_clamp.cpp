// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp.hpp>
#include "ck/utility/data_type.hpp"
#include "testing_utils.hpp"

using ck_tile::test::InstanceSet;
using ck_tile::test::InstancesMatch;

namespace {

using InLayout                      = ck::tensor_layout::convolution::NDHWGC;
using WeiLayout                     = ck::tensor_layout::convolution::GKZYXC;
using OutLayout                     = ck::tensor_layout::convolution::NDHWGK;
using DsLayout                      = ck::Tuple<>;

using ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD;
using ck::tensor_operation::element_wise::Clamp;
using ck::tensor_operation::element_wise::PassThrough;


template <int numSpatialDim, typename type, typename computeType=type>
using DeviceOp = DeviceGroupedConvFwdMultipleABD<numSpatialDim,
                                                 InLayout,
                                                 WeiLayout,
                                                 DsLayout,
                                                 OutLayout,
                                                 type,  // InDataType
                                                 type, // WeiDataType
                                                 ck::Tuple<>,
                                                 type, // OutDataType
                                                 PassThrough,
                                                 PassThrough,
                                                 Clamp,
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


struct F32_3D
{
    using DeviceOp = ::DeviceOp<3, float>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};


struct F32_TF32_3D
{
    using DeviceOp = ::DeviceOp<3, float, ck::tf32_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct F16_3D
{
    using DeviceOp = ::DeviceOp<3, ck::half_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct BF16_3D
{
    using DeviceOp = ::DeviceOp<3, ck::bhalf_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};

struct INT8_3D
{
    using DeviceOp = ::DeviceOp<3, int8_t>;

    constexpr static auto expected = {
        // clang-format off
        ""
        // clang-format on
    };
};


using TestTypes = ::testing::Types<F32_3D,
                                   F32_TF32_3D,
                                   F16_3D,
                                   BF16_3D,
                                   INT8_3D>;

TYPED_TEST_SUITE(CkFactoryTestBilinearFwd, TestTypes);

TYPED_TEST(CkFactoryTestBilinearFwd, TestInstances)
{
    auto actual   = TestFixture::get_actual_instances();
    auto expected = TestFixture::get_expected_instances();
    
    EXPECT_THAT(actual, InstancesMatch(expected));
}
