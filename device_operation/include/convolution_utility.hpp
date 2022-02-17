#ifndef CONVOLUTION_UTILITY_HPP
#define CONVOLUTION_UTILITY_HPP

#include <vector>

namespace ck {
namespace tensor_operation {

struct ConvolutionUtility
{
    static std::vector<ck::index_t>
    ComputeOutputSpatialLengths(std::vector<ck::index_t> input_spatial_lengths,
                                std::vector<ck::index_t> filter_spatial_lengths,
                                std::vector<ck::index_t> conv_strides,
                                std::vector<ck::index_t> conv_dilations,
                                std::vector<ck::index_t> in_left_pads,
                                std::vector<ck::index_t> in_right_pads)
    {
        if(input_spatial_lengths.size() == 2)
        {
            assert(filter_spatial_lengths.size() == 2);
            assert(conv_strides.size() == 2);
            assert(conv_dilations.size() == 2);
            assert(in_left_pads.size() == 2);
            assert(in_right_pads.size() == 2);

            const index_t YEff = (filter_spatial_lengths[0] - 1) * conv_dilations[0] + 1;
            const index_t XEff = (filter_spatial_lengths[1] - 1) * conv_dilations[1] + 1;

            const index_t Hi = input_spatial_lengths[0];
            const index_t Wi = input_spatial_lengths[1];

            const index_t Ho =
                (Hi + in_left_pads[0] + in_right_pads[0] - YEff) / conv_strides[0] + 1;
            const index_t Wo =
                (Wi + in_left_pads[1] + in_right_pads[1] - XEff) / conv_strides[1] + 1;

            return {Ho, Wo};
        }
        else if(input_spatial_lengths.size() == 3)
        {
            assert(filter_spatial_lengths.size() == 3);
            assert(conv_strides.size() == 3);
            assert(conv_dilations.size() == 3);
            assert(in_left_pads.size() == 3);
            assert(in_right_pads.size() == 3);

            const index_t ZEff = (filter_spatial_lengths[0] - 1) * conv_dilations[0] + 1;
            const index_t YEff = (filter_spatial_lengths[1] - 1) * conv_dilations[1] + 1;
            const index_t XEff = (filter_spatial_lengths[2] - 1) * conv_dilations[2] + 1;

            const index_t Di = input_spatial_lengths[0];
            const index_t Hi = input_spatial_lengths[1];
            const index_t Wi = input_spatial_lengths[2];

            const index_t Do =
                (Di + in_left_pads[0] + in_right_pads[0] - ZEff) / conv_strides[0] + 1;
            const index_t Ho =
                (Hi + in_left_pads[1] + in_right_pads[1] - YEff) / conv_strides[1] + 1;
            const index_t Wo =
                (Wi + in_left_pads[2] + in_right_pads[2] - XEff) / conv_strides[2] + 1;
            return {Do, Ho, Wo};
        }
        else
        {
            return {};
        }
    }
};

} // namespace tensor_operation
} // namespace ck
#endif

