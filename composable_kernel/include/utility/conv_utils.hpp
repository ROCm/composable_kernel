#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <vector>

#include "config.hpp"

namespace ck {
namespace conv_util {

/**
 * @brief      Calculate number of FLOPs for Convolution
 *
 * @param[in]  N                       Batch size.
 * @param[in]  C                       Number of input channels.
 * @param[in]  K                       Number of output channels.
 * @param[in]  filter_spatial_lengths  Filter spatial dimensions lengths.
 * @param[in]  output_spatial_lengths  Convolution output spatial dimensions
 *                                     lengths.
 *
 * @return     The number of flops.
 */
std::size_t GetFlops(ck::index_t N,
                     ck::index_t C,
                     ck::index_t K,
                     const std::vector<ck::index_t>& filter_spatial_lengths,
                     const std::vector<ck::index_t>& output_spatial_lengths)
{
    // 2 * N * K * <output spatial lengths product> * C * <filter spatial lengths product>
    return ck::index_t(2) * N * K *
           std::accumulate(std::begin(output_spatial_lengths),
                           std::end(output_spatial_lengths),
                           1,
                           std::multiplies<ck::index_t>()) *
           C *
           std::accumulate(std::begin(filter_spatial_lengths),
                           std::end(filter_spatial_lengths),
                           1,
                           std::multiplies<ck::index_t>());
}

template <typename InDataType  = float,
          typename WeiDataType = InDataType,
          typename OutDataType = InDataType>
ck::index_t GetBtype(ck::index_t N,
                     ck::index_t C,
                     ck::index_t K,
                     const std::vector<ck::index_t>& input_spatial_lengths,
                     const std::vector<ck::index_t>& filter_spatial_lengths,
                     const std::vector<ck::index_t>& output_spatial_lengths)
{
    // sizeof(InDataType) * (N * C * <input spatial lengths product>) +
    // sizeof(WeiDataType) * (K * C * <filter spatial lengths product>) +
    // sizeof(OutDataType) * (N * K * <output spatial lengths product>);
    return sizeof(InDataType) * (N * C *
                                 std::accumulate(std::begin(input_spatial_lengths),
                                                 std::end(input_spatial_lengths),
                                                 1,
                                                 std::multiplies<ck::index_t>())) +
           sizeof(WeiDataType) * (K * C *
                                  std::accumulate(std::begin(filter_spatial_lengths),
                                                  std::end(filter_spatial_lengths),
                                                  1,
                                                  std::multiplies<ck::index_t>())) +
           sizeof(OutDataType) * (N * K *
                                  std::accumulate(std::begin(output_spatial_lengths),
                                                  std::end(output_spatial_lengths),
                                                  1,
                                                  std::multiplies<ck::index_t>()));
}

struct ConvParams
{
    ConvParams()
        : spatial_dims(2),
          N(128),
          K(256),
          C(192),
          filter_spatial_lengths(2, 3),
          input_spatial_lengths(2, 71),
          conv_filter_strides(2, 2),
          conv_filter_dilations(2, 1),
          input_left_pads(2, 1),
          input_right_pads(2, 1)
    {
    }

    ck::index_t spatial_dims;
    ck::index_t N;
    ck::index_t K;
    ck::index_t C;

    std::vector<ck::index_t> filter_spatial_lengths;
    std::vector<ck::index_t> input_spatial_lengths;

    std::vector<ck::index_t> conv_filter_strides;
    std::vector<ck::index_t> conv_filter_dilations;

    std::vector<ck::index_t> input_left_pads;
    std::vector<ck::index_t> input_right_pads;

    std::vector<ck::index_t> GetOutputSpatialLengths() const
    {
        std::vector<ck::index_t> out_spatial_len(spatial_dims, 0);
        for(ck::index_t i = 0; i < spatial_dims; ++i)
        {
            // XEff = (X - 1) * conv_dilation_w + 1;
            // Wo = (Wi + in_left_pad_w + in_right_pad_w - XEff) / conv_stride_w + 1;
            const ck::index_t idx_eff =
                (filter_spatial_lengths[i] - 1) * conv_filter_dilations[i] + 1;
            out_spatial_len[i] =
                (input_spatial_lengths[i] + input_left_pads[i] + input_right_pads[i] - idx_eff) /
                    conv_filter_strides[i] +
                1;
        }
        return out_spatial_len;
    }
};

} // namespace conv_util
} // namespace ck
