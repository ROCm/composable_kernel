#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "config.hpp"
#include "conv_util.hpp"
#include "tensor_layout.hpp"
#include "check_err.hpp"

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
    ck::utils::conv::ConvParams conv_params;
};

} // namespace

TEST_F(TestConvUtil, ConvParamsGetOutputSpatialLengths2D)
{
    ck::utils::conv::ConvParams conv_params;
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

TEST(ConvUtil, GetHostTensorDescriptor)
{
    namespace tl = ck::tensor_layout::convolution;
    std::vector<std::size_t> dims{2, 3, 4, 5};
    HostTensorDescriptor h = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NHWC{});
    EXPECT_TRUE(ck::utils::check_err(
        h.GetLengths(), {2, 3, 4, 5}, "Error: wrong NHWC dimensions lengths!"));
    EXPECT_TRUE(ck::utils::check_err(
        h.GetStrides(), {3 * 4 * 5, 1, 3 * 5, 3}, "Error: wrong NHWC dimensions strides!"));

    h = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NCHW{});
    EXPECT_TRUE(ck::utils::check_err(
        h.GetLengths(), {2, 3, 4, 5}, "Error: wrong NCHW dimensions lengths!"));
    EXPECT_TRUE(ck::utils::check_err(
        h.GetStrides(), {3 * 4 * 5, 4 * 5, 5, 1}, "Error: wrong NCHW dimensions strides!"));

    dims = std::vector<std::size_t>{2, 3, 4};
    h    = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NWC{});
    EXPECT_TRUE(
        ck::utils::check_err(h.GetLengths(), {2, 3, 4}, "Error: wrong NWC dimensions lengths!"));
    EXPECT_TRUE(ck::utils::check_err(
        h.GetStrides(), {3 * 4, 1, 3}, "Error: wrong NWC dimensions strides!"));

    h = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NCW{});
    EXPECT_TRUE(
        ck::utils::check_err(h.GetLengths(), {2, 3, 4}, "Error: wrong NCW dimensions lengths!"));
    EXPECT_TRUE(ck::utils::check_err(
        h.GetStrides(), {3 * 4, 4, 1}, "Error: wrong NCW dimensions strides!"));

    dims = std::vector<std::size_t>{2, 3, 4, 5, 6};
    h    = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NDHWC{});
    EXPECT_TRUE(
        ck::utils::check_err(h.GetLengths(), dims, "Error: wrong NDHWC dimensions lengths!"));
    EXPECT_TRUE(ck::utils::check_err(h.GetStrides(),
                                     {3 * 4 * 5 * 6, // N
                                      1,             // C
                                      3 * 5 * 6,     // D
                                      3 * 6,         // H
                                      3},            // W
                                     "Error: wrong NDHWC dimensions strides!"));

    h = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NCDHW{});
    EXPECT_TRUE(
        ck::utils::check_err(h.GetLengths(), dims, "Error: wrong NCDHW dimensions lengths!"));
    EXPECT_TRUE(ck::utils::check_err(h.GetStrides(),
                                     {3 * 4 * 5 * 6, // N
                                      4 * 5 * 6,     // C
                                      5 * 6,         // D
                                      6,             // H
                                      1},            // W
                                     "Error: wrong NCDHW dimensions strides!"));
}
