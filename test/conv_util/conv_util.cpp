// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/convolution_parameter.hpp"

namespace {

class TestConvUtil : public ::testing::Test
{
    public:
    void SetNDParams(std::size_t ndims)
    {
        conv_params.num_dim_spatial_        = ndims;
        conv_params.filter_spatial_lengths_ = std::vector<ck::index_t>(ndims, 3);
        conv_params.input_spatial_lengths_  = std::vector<ck::index_t>(ndims, 71);
        conv_params.conv_filter_strides_    = std::vector<ck::index_t>(ndims, 2);
        conv_params.conv_filter_dilations_  = std::vector<ck::index_t>(ndims, 1);
        conv_params.input_left_pads_        = std::vector<ck::index_t>(ndims, 1);
        conv_params.input_right_pads_       = std::vector<ck::index_t>(ndims, 1);
    }

    protected:
    // -------  default 2D -------
    // input NCHW {128,192,71,71},
    // weights KCYX {256,192,3,3},
    // stride {2,2},
    // dilations {1,1},
    // padding {{1,1}, {1,1}}
    ck::utils::conv::ConvParam conv_params;
};

} // namespace

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths1D)
{
    SetNDParams(1);

    std::vector<ck::index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36}, "Error: ConvParams 1D."));

    conv_params.conv_filter_strides_ = std::vector<ck::index_t>{1};
    out_spatial_len                  = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{71}, "Error: ConvParams 1D stride {1}."));

    conv_params.conv_filter_strides_ = std::vector<ck::index_t>{2};
    conv_params.input_left_pads_     = std::vector<ck::index_t>{2};
    conv_params.input_right_pads_    = std::vector<ck::index_t>{2};
    out_spatial_len                  = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{37},
                                     "Error: ConvParams 1D padding left/right {2}."));

    conv_params.conv_filter_dilations_ = std::vector<ck::index_t>{2};
    out_spatial_len                    = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36}, "Error: ConvParams 1D dilation {2}."));

    conv_params.conv_filter_strides_   = std::vector<ck::index_t>{3};
    conv_params.input_left_pads_       = std::vector<ck::index_t>{1};
    conv_params.input_right_pads_      = std::vector<ck::index_t>{1};
    conv_params.conv_filter_dilations_ = std::vector<ck::index_t>{2};
    out_spatial_len                    = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(
        ck::utils::check_err(out_spatial_len,
                             std::vector<ck::index_t>{23},
                             "Error: ConvParams 1D strides{3}, padding {1}, dilations {2}."));
}

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths2D)
{
    ck::utils::conv::ConvParam conv_params;
    std::vector<ck::index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{36, 36},
                                     "Error: ConvParams 2D default constructor."));

    conv_params.conv_filter_strides_ = std::vector<ck::index_t>{1, 1};
    out_spatial_len                  = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{71, 71}, "Error: ConvParams 2D stride {1,1}."));

    conv_params.conv_filter_strides_ = std::vector<ck::index_t>{2, 2};
    conv_params.input_left_pads_     = std::vector<ck::index_t>{2, 2};
    conv_params.input_right_pads_    = std::vector<ck::index_t>{2, 2};
    out_spatial_len                  = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{37, 37},
                                     "Error: ConvParams 2D padding left/right {2,2}."));

    conv_params.conv_filter_dilations_ = std::vector<ck::index_t>{2, 2};
    out_spatial_len                    = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36, 36}, "Error: ConvParams 2D dilation {2,2}."));

    conv_params.conv_filter_strides_   = std::vector<ck::index_t>{3, 3};
    conv_params.input_left_pads_       = std::vector<ck::index_t>{1, 1};
    conv_params.input_right_pads_      = std::vector<ck::index_t>{1, 1};
    conv_params.conv_filter_dilations_ = std::vector<ck::index_t>{2, 2};
    out_spatial_len                    = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(
        ck::utils::check_err(out_spatial_len,
                             std::vector<ck::index_t>{23, 23},
                             "Error: ConvParams 2D strides{3,3}, padding {1,1}, dilations {2,2}."));
}

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths3D)
{
    SetNDParams(3);

    std::vector<ck::index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36, 36, 36}, "Error: ConvParams 3D."));

    conv_params.conv_filter_strides_ = std::vector<ck::index_t>{1, 1, 1};
    out_spatial_len                  = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{71, 71, 71},
                                     "Error: ConvParams 3D stride {1, 1, 1}."));

    conv_params.conv_filter_strides_ = std::vector<ck::index_t>{2, 2, 2};
    conv_params.input_left_pads_     = std::vector<ck::index_t>{2, 2, 2};
    conv_params.input_right_pads_    = std::vector<ck::index_t>{2, 2, 2};
    out_spatial_len                  = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{37, 37, 37},
                                     "Error: ConvParams 3D padding left/right {2, 2, 2}."));

    conv_params.conv_filter_dilations_ = std::vector<ck::index_t>{2, 2, 2};
    out_spatial_len                    = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(out_spatial_len,
                                     std::vector<ck::index_t>{36, 36, 36},
                                     "Error: ConvParams 3D dilation {2, 2, 2}."));

    conv_params.conv_filter_strides_   = std::vector<ck::index_t>{3, 3, 3};
    conv_params.input_left_pads_       = std::vector<ck::index_t>{1, 1, 1};
    conv_params.input_right_pads_      = std::vector<ck::index_t>{1, 1, 1};
    conv_params.conv_filter_dilations_ = std::vector<ck::index_t>{2, 2, 2};
    out_spatial_len                    = conv_params.GetOutputSpatialLengths();
    EXPECT_TRUE(ck::utils::check_err(
        out_spatial_len,
        std::vector<ck::index_t>{23, 23, 23},
        "Error: ConvParams 3D strides{3, 3, 3}, padding {1, 1, 1}, dilations {2, 2, 2}."));
}
