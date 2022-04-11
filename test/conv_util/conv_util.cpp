#include <iostream>
#include <string>
#include <vector>

#include "config.hpp"
#include "conv_fwd_util.hpp"
#include "tensor_layout.hpp"
#include "check_err.hpp"

namespace {

bool test_conv_params_get_output_spatial_lengths()
{
    bool res{true};
    // -------------------------- default 2D ------------------------------------
    // input NCHW {128,192,71,71},
    // weights KCYX {256,192,3,3},
    // stride {2,2},
    // dilations {1,1},
    // padding {{1,1}, {1,1}}
    ck::utils::conv::ConvParams conv_params;
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

    conv_params.conv_filter_strides = std::vector<ck::index_t>{1};
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

bool test_get_host_tensor_descriptor()
{
    bool res{true};
    namespace tl = ck::tensor_layout::convolution;
    std::vector<std::size_t> dims{2, 3, 4, 5};
    HostTensorDescriptor h = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NHWC{});
    res =
        ck::utils::check_err(h.GetLengths(), {2, 3, 4, 5}, "Error: wrong NHWC dimensions lengths!");
    res = ck::utils::check_err(
        h.GetStrides(), {3 * 4 * 5, 1, 3 * 5, 3}, "Error: wrong NHWC dimensions strides!");

    h = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NCHW{});
    res =
        ck::utils::check_err(h.GetLengths(), {2, 3, 4, 5}, "Error: wrong NCHW dimensions lengths!");
    res = ck::utils::check_err(
        h.GetStrides(), {3 * 4 * 5, 4 * 5, 5, 1}, "Error: wrong NCHW dimensions strides!");

    dims = std::vector<std::size_t>{2, 3, 4};
    h    = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NWC{});
    res  = ck::utils::check_err(h.GetLengths(), {2, 3, 4}, "Error: wrong NWC dimensions lengths!");
    res =
        ck::utils::check_err(h.GetStrides(), {3 * 4, 1, 3}, "Error: wrong NWC dimensions strides!");

    h   = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NCW{});
    res = ck::utils::check_err(h.GetLengths(), {2, 3, 4}, "Error: wrong NCW dimensions lengths!");
    res =
        ck::utils::check_err(h.GetStrides(), {3 * 4, 4, 1}, "Error: wrong NCW dimensions strides!");

    dims = std::vector<std::size_t>{2, 3, 4, 5, 6};
    h    = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NDHWC{});
    res  = ck::utils::check_err(h.GetLengths(), dims, "Error: wrong NDHWC dimensions lengths!");
    res  = ck::utils::check_err(h.GetStrides(),
                               {3 * 4 * 5 * 6, // N
                                1,             // C
                                3 * 5 * 6,     // D
                                3 * 6,         // H
                                3},            // W
                               "Error: wrong NDHWC dimensions strides!");

    h   = ck::utils::conv::get_host_tensor_descriptor(dims, tl::NCDHW{});
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
    bool res = test_conv_params_get_output_spatial_lengths();
    std::cout << "test_conv_params_get_output_spatial_lengths ..... "
              << (res ? "SUCCESS" : "FAILURE") << std::endl;
    res = test_get_host_tensor_descriptor();
    std::cout << "test_get_host_tensor_descriptor ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    return res ? 0 : 1;
}
