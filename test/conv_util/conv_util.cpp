// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp"

namespace {

class TestConvUtil : public ::testing::Test
{
    public:
    void SetNDParams(std::size_t ndims, std::size_t s, std::size_t d, std::size_t p)
    {
        conv_params = ck::utils::conv::ConvParam(ndims,
                                                 2,
                                                 128,
                                                 192,
                                                 256,
                                                 std::vector<ck::long_index_t>(ndims, 3),
                                                 std::vector<ck::long_index_t>(ndims, 71),
                                                 std::vector<ck::long_index_t>(ndims, s),
                                                 std::vector<ck::long_index_t>(ndims, d),
                                                 std::vector<ck::long_index_t>(ndims, p),
                                                 std::vector<ck::long_index_t>(ndims, p));
    }

    protected:
    // -------  default 2D -------
    // input GNCHW {2, 128, 192, 71, 71},
    // weights GKCYX {2, 256, 192, 3, 3},
    // stride {s, s},
    // dilations {d, d},
    // padding {{p, p}, {p, p}
    ck::utils::conv::ConvParam conv_params;
};

} // namespace

TEST(TestIsPackedTensor, Packed1D)
{
    using namespace ck::tensor_operation::device;
    EXPECT_TRUE((IsPackedTensor<int, 1>({5}, {1})));
}

TEST(TestIsPackedTensor, Packed2D)
{
    using namespace ck::tensor_operation::device;
    // row-major: lengths [3,4], strides [4,1]
    EXPECT_TRUE((IsPackedTensor<int, 2>({3, 4}, {4, 1})));
    // col-major: lengths [3,4], strides [1,3]
    EXPECT_TRUE((IsPackedTensor<int, 2>({3, 4}, {1, 3})));
}

TEST(TestIsPackedTensor, Packed3D)
{
    using namespace ck::tensor_operation::device;
    EXPECT_TRUE((IsPackedTensor<int, 3>({2, 3, 4}, {12, 4, 1})));
    EXPECT_TRUE((IsPackedTensor<int, 3>({2, 3, 4}, {1, 2, 6})));
    EXPECT_TRUE((IsPackedTensor<int, 3>({2, 1, 4}, {4, 1, 1})));
    EXPECT_TRUE((IsPackedTensor<int, 3>({2, 1, 4}, {4, 4, 1})));
}

TEST(TestIsPackedTensor, NotPacked2D)
{
    using namespace ck::tensor_operation::device;
    // smallest stride is not 1
    EXPECT_FALSE((IsPackedTensor<int, 2>({3, 4}, {8, 2})));
    // gap between dimensions (stride[0] > lengths[1] * stride[1])
    EXPECT_FALSE((IsPackedTensor<int, 2>({3, 4}, {8, 1})));
    // Dim equal to the 1 but not packed
    EXPECT_FALSE((IsPackedTensor<int, 2>({1, 4}, {1, 8})));
}

TEST(TestIsPackedTensor, NotPacked3D)
{
    using namespace ck::tensor_operation::device;
    // gap between dimensions 1 and 2
    EXPECT_FALSE((IsPackedTensor<int, 3>({2, 3, 4}, {12, 5, 1})));
}

TEST(TestIsPackedTensor, UnitDimension2D)
{
    using namespace ck::tensor_operation::device;
    // unit dimension with stride=1 — is packed
    EXPECT_TRUE((IsPackedTensor<int, 2>({4, 1}, {1, 1})));
    // unit dimension with stride=4 — also valid packed (stride is irrelevant when length=1)
    EXPECT_TRUE((IsPackedTensor<int, 2>({4, 1}, {1, 4})));
    // unit dimension first (col-major layout)
    EXPECT_TRUE((IsPackedTensor<int, 2>({1, 4}, {1, 1})));
    EXPECT_TRUE((IsPackedTensor<int, 2>({1, 4}, {4, 1})));
}

TEST(TestIsPackedTensor, UnitDimension3D)
{
    using namespace ck::tensor_operation::device;
    // middle dimension = 1, remaining dimensions packed
    EXPECT_TRUE((IsPackedTensor<int, 3>({4, 1, 3}, {3, 1, 1})));
    EXPECT_TRUE((IsPackedTensor<int, 3>({4, 1, 3}, {3, 99, 1})));
    // two unit dimensions
    EXPECT_TRUE((IsPackedTensor<int, 3>({5, 1, 1}, {1, 1, 1})));
    EXPECT_TRUE((IsPackedTensor<int, 3>({5, 1, 1}, {1, 5, 99})));
}

TEST(TestIsPackedTensor, AllUnitDimensions)
{
    using namespace ck::tensor_operation::device;
    EXPECT_TRUE((IsPackedTensor<int, 3>({1, 1, 1}, {1, 1, 1})));
    EXPECT_TRUE((IsPackedTensor<int, 3>({1, 1, 1}, {1, 42, 7})));
}

TEST(TestIsPackedTensor, NHWGCLayout)
{
    using namespace ck::tensor_operation::device;
    // lengths[0,1,2,3,4] = {2, 3, 5, 4, 6}
    // Sorted by stride: dim1(s=1), dim3(s=3), dim4(s=12), dim0(s=72), dim2(s=144)
    // i.e. sorted dimension order: 1, 3, 4, 0, 2
    EXPECT_TRUE((IsPackedTensor<int, 5>({2, 3, 5, 4, 6}, {72, 1, 144, 3, 12})));

    // Gap after dim3: stride[4] = 13 instead of 12
    EXPECT_FALSE((IsPackedTensor<int, 5>({2, 3, 5, 4, 6}, {72, 1, 144, 3, 13})));
}

TEST(TestIsPackedTensor, LongIndexType)
{
    using namespace ck::tensor_operation::device;
    using ck::long_index_t;
    EXPECT_TRUE((IsPackedTensor<long_index_t, 2>({128, 256}, {256, 1})));
    EXPECT_TRUE((IsPackedTensor<long_index_t, 2>({128, 1}, {1, 128})));
    EXPECT_FALSE((IsPackedTensor<long_index_t, 2>({128, 256}, {512, 1})));
}

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths1D)
{
    // stride 2, dilation 1, pad 1
    SetNDParams(1, 2, 1, 1);
    std::vector<ck::long_index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::long_index_t>{36}, "Error: ConvParams 1D."));

    // stride 1, dilation 1, pad 1
    SetNDParams(1, 1, 1, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::long_index_t>{71}, "Error: ConvParams 1D stride {1}."));

    // stride 2, dilation 1, pad 2
    SetNDParams(1, 2, 1, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{37},
                                     "Error: ConvParams 1D padding left/right {2}."));

    // stride 2, dilation 2, pad 2
    SetNDParams(1, 2, 2, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::long_index_t>{36}, "Error: ConvParams 1D dilation {2}."));

    // stride 3, dilation 2, pad 1
    SetNDParams(1, 3, 2, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(
        ck::utils::check_err(out_spatial_len,
                             std::vector<ck::long_index_t>{23},
                             "Error: ConvParams 1D strides{3}, padding {1}, dilations {2}."));
}

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths2D)
{
    // stride 2, dilation 1, pad 1
    SetNDParams(2, 2, 1, 1);
    std::vector<ck::long_index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{36, 36},
                                     "Error: ConvParams 2D default constructor."));

    // stride 1, dilation 1, pad 1
    SetNDParams(2, 1, 1, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{71, 71},
                                     "Error: ConvParams 2D stride {1,1}."));

    // stride 2, dilation 1, pad 2
    SetNDParams(2, 2, 1, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{37, 37},
                                     "Error: ConvParams 2D padding left/right {2,2}."));

    // stride 2, dilation 2, pad 2
    SetNDParams(2, 2, 2, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{36, 36},
                                     "Error: ConvParams 2D dilation {2,2}."));

    // stride 3, dilation 2, pad 1
    SetNDParams(2, 3, 2, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(
        ck::utils::check_err(out_spatial_len,
                             std::vector<ck::long_index_t>{23, 23},
                             "Error: ConvParams 2D strides{3,3}, padding {1,1}, dilations {2,2}."));
}

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths3D)
{
    // stride 2, dilation 1, pad 1
    SetNDParams(3, 2, 1, 1);
    std::vector<ck::long_index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::long_index_t>{36, 36, 36}, "Error: ConvParams 3D."));

    // stride 1, dilation 1, pad 1
    SetNDParams(3, 1, 1, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{71, 71, 71},
                                     "Error: ConvParams 3D stride {1, 1, 1}."));

    // stride 2, dilation 1, pad 2
    SetNDParams(3, 2, 1, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{37, 37, 37},
                                     "Error: ConvParams 3D padding left/right {2, 2, 2}."));

    // stride 2, dilation 2, pad 2
    SetNDParams(3, 2, 2, 2);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::long_index_t>{36, 36, 36},
                                     "Error: ConvParams 3D dilation {2, 2, 2}."));

    // stride 3, dilation 2, pad 1
    SetNDParams(3, 3, 2, 1);
    out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len,
        std::vector<ck::long_index_t>{23, 23, 23},
        "Error: ConvParams 3D strides{3, 3, 3}, padding {1, 1, 1}, dilations {2, 2, 2}."));
}
