// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp>

#include "testing_utils.hpp"

using ck_tile::test::InstanceMatcher;
using ck_tile::test::InstanceSet;
using ck_tile::test::StringEqWithDiff;

TEST(InstanceSet, FromFactory)
{
    using DeviceOp = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<
        2,                                                   // NDimSpatial
        ck::tensor_operation::device::instance::NHWGC,       // InLayout
        ck::tensor_operation::device::instance::GKYXC,       // WeiLayout
        ck::tensor_operation::device::instance::Empty_Tuple, // DsLayout
        ck::tensor_operation::device::instance::NHWGK,       // OutLayout
        ck::half_t,                                          // ADataType
        ck::half_t,                                          // BDataType
        ck::Tuple<>,                                         // DsDataType
        ck::half_t,                                          // EDataType
        ck::tensor_operation::element_wise::PassThrough,     // AElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough,     // BElementwiseOperation
        ck::tensor_operation::element_wise::PassThrough,     // CDEElementwiseOperation
        ck::half_t,                                          // AComputeType
        ck::half_t>;                                         // BComputeType

    const auto instances = InstanceSet::from_factory<DeviceOp>();

    EXPECT_THAT(instances.instances.size(), testing::Gt(0));

    const auto* el =
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle_V3<2,NHWGC,GKYXC,EmptyTuple,NHWGK,fp16,fp16,"
        "fp32,fp16,EmptyTuple,fp16,PassThrough,PassThrough,PassThrough,OddC,MNKPadding,64,16,16,64,"
        "8,8,16,16,1,1,Seq(8,8,1),Seq(1,0,2),Seq(1,0,2),2,8,8,0,Seq(8,8,1),Seq(1,0,2),Seq(1,0,2),2,"
        "8,8,0,1,1,Seq(1,16,1,4),4,Intrawave,v2,fp16,fp16>";
    EXPECT_THAT(instances.instances, testing::Contains(el));
}

TEST(InstanceMatcher, Basic)
{
    auto a = InstanceSet{
        "python",
        "cobra",
        "boa",
    };

    auto b = InstanceSet{
        "cobra",
        "boa",
        "python",
    };

    auto c = InstanceSet{
        "adder",
        "boa",
        "cobra",
    };

    auto d = InstanceSet{
        "boa",
        "python",
    };

    EXPECT_THAT(a, InstancesMatch(b));
    EXPECT_THAT(c, Not(InstancesMatch(b)));
    EXPECT_THAT(a, Not(InstancesMatch(d)));
    EXPECT_THAT(d, Not(InstancesMatch(b)));
}

TEST(InstanceMatcher, ExplainMatchResult)
{
    auto actual = InstanceSet{
        "python",
        "cobra",
        "boa",
    };

    auto expected = InstanceSet{
        "adder",
        "boa",
        "cobra",
        "rattlesnake",
    };

    testing::StringMatchResultListener listener;
    EXPECT_TRUE(!ExplainMatchResult(InstancesMatch(expected), actual, &listener));

    EXPECT_THAT(listener.str(),
                StringEqWithDiff("\n"
                                 " Missing: 2\n"
                                 "- adder\n"
                                 "- rattlesnake\n"
                                 "Unexpected: 1\n"
                                 "- python\n"));
}
