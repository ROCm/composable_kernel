#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "conv_utils.hpp"
#include "tensor_layout.hpp"
#include "check_err.hpp"

namespace {

bool TestConvParams_GetOutputSpatialLengths()
{
    bool res{true};
    // -------------------------- default 2D ------------------------------------
    // input NCHW {128,192,71,71},
    // weights KCYX {256,192,3,3},
    // stride {2,2},
    // dilations {1,1},
    // padding {{1,1}, {1,1}}
    ck::conv_util::ConvParams conv_params;
    std::vector<ck::index_t> out_spatial_len = conv_params.GetOutputSpatialLengths();
    res                                      = ck::utils::check_err(out_spatial_len,
                               std::vector<ck::index_t>{36, 36},
                               "Error: ConvParams 2D default constructor.");

    conv_params.conv_filter_strides = std::vector<ck::index_t>{1, 1};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    res                             = ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{71, 71}, "Error: ConvParams 2D stride {1,1}.");

    conv_params.conv_filter_strides = std::vector<ck::index_t>{2, 2};
    conv_params.input_left_pads     = std::vector<ck::index_t>{2, 2};
    conv_params.input_right_pads    = std::vector<ck::index_t>{2, 2};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    res                             = ck::utils::check_err(out_spatial_len,
                               std::vector<ck::index_t>{37, 37},
                               "Error: ConvParams 2D padding left/right {2,2}.");

    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2, 2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    res                               = ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36, 36}, "Error: ConvParams 2D dilation {2,2}.");

    conv_params.conv_filter_strides   = std::vector<ck::index_t>{3, 3};
    conv_params.input_left_pads       = std::vector<ck::index_t>{1, 1};
    conv_params.input_right_pads      = std::vector<ck::index_t>{1, 1};
    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2, 2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    res =
        ck::utils::check_err(out_spatial_len,
                             std::vector<ck::index_t>{23, 23},
                             "Error: ConvParams 2D strides{3,3}, padding {1,1}, dilations {2,2}.");

    // -------------------------- 1D ------------------------------------
    conv_params.num_dim_spatial        = 1;
    conv_params.filter_spatial_lengths = std::vector<ck::index_t>{3};
    conv_params.input_spatial_lengths  = std::vector<ck::index_t>{71};
    conv_params.conv_filter_strides    = std::vector<ck::index_t>{2};
    conv_params.conv_filter_dilations  = std::vector<ck::index_t>{1};
    conv_params.input_left_pads        = std::vector<ck::index_t>{1};
    conv_params.input_right_pads       = std::vector<ck::index_t>{1};

    out_spatial_len = conv_params.GetOutputSpatialLengths();
    res             = ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36}, "Error: ConvParams 1D.");

    conv_params.conv_filter_strides = std::vector<ck::index_t>{1, 1};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    res                             = ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{71}, "Error: ConvParams 1D stride {1}.");

    conv_params.conv_filter_strides = std::vector<ck::index_t>{2};
    conv_params.input_left_pads     = std::vector<ck::index_t>{2};
    conv_params.input_right_pads    = std::vector<ck::index_t>{2};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    res                             = ck::utils::check_err(out_spatial_len,
                               std::vector<ck::index_t>{37},
                               "Error: ConvParams 1D padding left/right {2}.");

    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    res                               = ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36}, "Error: ConvParams 1D dilation {2}.");

    conv_params.conv_filter_strides   = std::vector<ck::index_t>{3};
    conv_params.input_left_pads       = std::vector<ck::index_t>{1};
    conv_params.input_right_pads      = std::vector<ck::index_t>{1};
    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    res                               = ck::utils::check_err(out_spatial_len,
                               std::vector<ck::index_t>{23},
                               "Error: ConvParams 1D strides{3}, padding {1}, dilations {2}.");

    // -------------------------- 3D ------------------------------------
    conv_params.num_dim_spatial        = 3;
    conv_params.filter_spatial_lengths = std::vector<ck::index_t>{3, 3, 3};
    conv_params.input_spatial_lengths  = std::vector<ck::index_t>{71, 71, 71};
    conv_params.conv_filter_strides    = std::vector<ck::index_t>{2, 2, 2};
    conv_params.conv_filter_dilations  = std::vector<ck::index_t>{1, 1, 1};
    conv_params.input_left_pads        = std::vector<ck::index_t>{1, 1, 1};
    conv_params.input_right_pads       = std::vector<ck::index_t>{1, 1, 1};

    out_spatial_len = conv_params.GetOutputSpatialLengths();
    res             = ck::utils::check_err(
        out_spatial_len, std::vector<ck::index_t>{36, 36, 36}, "Error: ConvParams 3D.");

    conv_params.conv_filter_strides = std::vector<ck::index_t>{1, 1, 1};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    res                             = ck::utils::check_err(out_spatial_len,
                               std::vector<ck::index_t>{71, 71, 71},
                               "Error: ConvParams 3D stride {1, 1, 1}.");

    conv_params.conv_filter_strides = std::vector<ck::index_t>{2, 2, 2};
    conv_params.input_left_pads     = std::vector<ck::index_t>{2, 2, 2};
    conv_params.input_right_pads    = std::vector<ck::index_t>{2, 2, 2};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    res                             = ck::utils::check_err(out_spatial_len,
                               std::vector<ck::index_t>{37, 37, 37},
                               "Error: ConvParams 3D padding left/right {2, 2, 2}.");

    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2, 2, 2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    res                               = ck::utils::check_err(out_spatial_len,
                               std::vector<ck::index_t>{36, 36, 36},
                               "Error: ConvParams 3D dilation {2, 2, 2}.");

    conv_params.conv_filter_strides   = std::vector<ck::index_t>{3, 3, 3};
    conv_params.input_left_pads       = std::vector<ck::index_t>{1, 1, 1};
    conv_params.input_right_pads      = std::vector<ck::index_t>{1, 1, 1};
    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2, 2, 2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    res                               = ck::utils::check_err(
        out_spatial_len,
        std::vector<ck::index_t>{23, 23, 23},
        "Error: ConvParams 3D strides{3, 3, 3}, padding {1, 1, 1}, dilations {2, 2, 2}.");

    return res;
}

bool TestGetHostTensorDescriptor()
{
    bool res{true};
    namespace tl = ck::tensor_layout::convolution;
    std::vector<std::size_t> dims{2, 3, 4, 5};
    HostTensorDescriptor h = ck::conv_util::GetHostTensorDescriptor(dims, tl::NHWC{});
    res =
        ck::utils::check_err(h.GetLengths(), {2, 3, 4, 5}, "Error: wrong NHWC dimensions lengths!");
    res = ck::utils::check_err(
        h.GetStrides(), {3 * 4 * 5, 1, 3 * 5, 3}, "Error: wrong NHWC dimensions strides!");

    h = ck::conv_util::GetHostTensorDescriptor(dims, tl::NCHW{});
    res =
        ck::utils::check_err(h.GetLengths(), {2, 3, 4, 5}, "Error: wrong NCHW dimensions lengths!");
    res = ck::utils::check_err(
        h.GetStrides(), {3 * 4 * 5, 4 * 5, 5, 1}, "Error: wrong NCHW dimensions strides!");

    dims = std::vector<std::size_t>{2, 3, 4};
    h    = ck::conv_util::GetHostTensorDescriptor(dims, tl::NWC{});
    res  = ck::utils::check_err(h.GetLengths(), {2, 3, 4}, "Error: wrong NWC dimensions lengths!");
    res =
        ck::utils::check_err(h.GetStrides(), {3 * 4, 1, 3}, "Error: wrong NWC dimensions strides!");

    h   = ck::conv_util::GetHostTensorDescriptor(dims, tl::NCW{});
    res = ck::utils::check_err(h.GetLengths(), {2, 3, 4}, "Error: wrong NCW dimensions lengths!");
    res =
        ck::utils::check_err(h.GetStrides(), {3 * 4, 4, 1}, "Error: wrong NCW dimensions strides!");

    dims = std::vector<std::size_t>{2, 3, 4, 5, 6};
    h    = ck::conv_util::GetHostTensorDescriptor(dims, tl::NDHWC{});
    res  = ck::utils::check_err(h.GetLengths(), dims, "Error: wrong NDHWC dimensions lengths!");
    res  = ck::utils::check_err(h.GetStrides(),
                               {3 * 4 * 5 * 6, // N
                                1,             // C
                                3 * 5 * 6,     // D
                                3 * 6,         // H
                                3},            // W
                               "Error: wrong NDHWC dimensions strides!");

    h   = ck::conv_util::GetHostTensorDescriptor(dims, tl::NCDHW{});
    res = ck::utils::check_err(h.GetLengths(), dims, "Error: wrong NCDHW dimensions lengths!");
    res = ck::utils::check_err(h.GetStrides(),
                               {3 * 4 * 5 * 6, // N
                                4 * 5 * 6,     // C
                                5 * 6,         // D
                                6,             // H
                                1},            // W
                               "Error: wrong NCDHW dimensions strides!");

    return res;
}

} // namespace

int main(void)
{
    bool res = TestConvParams_GetOutputSpatialLengths();
    std::cout << "TestConvParams_GetOutputSpatialLengths ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    res = TestGetHostTensorDescriptor();
    std::cout << "TestGetHostTensorDescriptor ..... " << (res ? "SUCCESS" : "FAILURE") << std::endl;
    return 0;
}
