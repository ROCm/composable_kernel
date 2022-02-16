#include <iostream>
#include <vector>

#include "config.hpp"
#include "conv_utils.hpp"

namespace {

bool cmp_vec(const std::vector<ck::index_t>& out, const std::vector<ck::index_t>& ref)
{
    if(out.size() != ref.size())
    {
        std::cout << "out.size() != ref.size(), :" << out.size() << " != " << ref.size()
                  << std::endl;
        return false;
    }

    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        if(out[i] != ref[i])
        {
            std::cout << "out[" << i << "] != ref[" << i << "]: " << out[i] << "!=" << ref[i]
                      << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace

static bool TestConvParams_GetOutputSpatialLengths()
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
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{36, 36}))
    {
        std::cout << "Error: ConvParams 2D default constructor." << std::endl;
        res = false;
    }

    conv_params.conv_filter_strides = std::vector<ck::index_t>{1, 1};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{71, 71}))
    {
        std::cout << "Error: ConvParams 2D stride {1,1}." << std::endl;
        res = false;
    }

    conv_params.conv_filter_strides = std::vector<ck::index_t>{2, 2};
    conv_params.input_left_pads     = std::vector<ck::index_t>{2, 2};
    conv_params.input_right_pads    = std::vector<ck::index_t>{2, 2};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{37, 37}))
    {
        std::cout << "Error: ConvParams 2D padding left/right {2,2}." << std::endl;
        res = false;
    }

    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2, 2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{36, 36}))
    {
        std::cout << "Error: ConvParams 2D dilation {2,2}." << std::endl;
        res = false;
    }

    conv_params.conv_filter_strides   = std::vector<ck::index_t>{3, 3};
    conv_params.input_left_pads       = std::vector<ck::index_t>{1, 1};
    conv_params.input_right_pads      = std::vector<ck::index_t>{1, 1};
    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2, 2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{23, 23}))
    {
        std::cout << "Error: ConvParams 2D strides{3,3}, padding {1,1}, dilations {2,2}."
                  << std::endl;
        res = false;
    }

    // -------------------------- 1D ------------------------------------
    conv_params.spatial_dims           = 1;
    conv_params.filter_spatial_lengths = std::vector<ck::index_t>{3};
    conv_params.input_spatial_lengths  = std::vector<ck::index_t>{71};
    conv_params.conv_filter_strides    = std::vector<ck::index_t>{2};
    conv_params.conv_filter_dilations  = std::vector<ck::index_t>{1};
    conv_params.input_left_pads        = std::vector<ck::index_t>{1};
    conv_params.input_right_pads       = std::vector<ck::index_t>{1};

    out_spatial_len = conv_params.GetOutputSpatialLengths();
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{36}))
    {
        std::cout << "Error: ConvParams 1D default constructor." << std::endl;
        res = false;
    }

    conv_params.conv_filter_strides = std::vector<ck::index_t>{1, 1};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{71}))
    {
        std::cout << "Error: ConvParams 1D stride {1}." << std::endl;
        res = false;
    }

    conv_params.conv_filter_strides = std::vector<ck::index_t>{2};
    conv_params.input_left_pads     = std::vector<ck::index_t>{2};
    conv_params.input_right_pads    = std::vector<ck::index_t>{2};
    out_spatial_len                 = conv_params.GetOutputSpatialLengths();
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{37}))
    {
        std::cout << "Error: ConvParams 1D padding left/right {2}." << std::endl;
        res = false;
    }

    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{36}))
    {
        std::cout << "Error: ConvParams 1D dilation {2}." << std::endl;
        res = false;
    }

    conv_params.conv_filter_strides   = std::vector<ck::index_t>{3};
    conv_params.input_left_pads       = std::vector<ck::index_t>{1};
    conv_params.input_right_pads      = std::vector<ck::index_t>{1};
    conv_params.conv_filter_dilations = std::vector<ck::index_t>{2};
    out_spatial_len                   = conv_params.GetOutputSpatialLengths();
    if(!cmp_vec(out_spatial_len, std::vector<ck::index_t>{23}))
    {
        std::cout << "Error: ConvParams 1D strides{3}, padding {1}, dilations {2}." << std::endl;
        res = false;
    }

    return res;
}

int main(void)
{
    bool res = TestConvParams_GetOutputSpatialLengths();
    std::cout << "TestConvParams_GetOutputSpatialLengths ..... " << (res ? "SUCCESS" : "FAILURE")
              << std::endl;
    return 0;
}
