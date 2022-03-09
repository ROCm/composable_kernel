#ifndef CONV_UTILS_HPP
#define CONV_UTILS_HPP

#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <vector>

#include "config.hpp"
#include "host_tensor.hpp"
#include "tensor_layout.hpp"

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
    return static_cast<std::size_t>(2) * N * K *
           std::accumulate(std::begin(output_spatial_lengths),
                           std::end(output_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<std::size_t>()) *
           C *
           std::accumulate(std::begin(filter_spatial_lengths),
                           std::end(filter_spatial_lengths),
                           static_cast<std::size_t>(1),
                           std::multiplies<std::size_t>());
}

/**
 * @brief      Calculate number of bytes read/write by convolution algorithm.
 *
 * @param[in]  N                       Batch size.
 * @param[in]  C                       Number of input channels.
 * @param[in]  K                       Number of output channels.
 * @param[in]  input_spatial_lengths   Input spatial dimensions lengths.
 * @param[in]  filter_spatial_lengths  Filter spatial dimensions lengths.
 * @param[in]  output_spatial_lengths  Output spatial dimensions lengths
 *
 * @tparam     InDataType              Input tensor data type.
 * @tparam     WeiDataType             Weights tensor data type.
 * @tparam     OutDataType             Output tensor data type.
 *
 * @return     The number of used bytes.
 */
template <typename InDataType  = float,
          typename WeiDataType = InDataType,
          typename OutDataType = InDataType>
std::size_t GetBtype(ck::index_t N,
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
                                                 static_cast<std::size_t>(1),
                                                 std::multiplies<std::size_t>())) +
           sizeof(WeiDataType) * (K * C *
                                  std::accumulate(std::begin(filter_spatial_lengths),
                                                  std::end(filter_spatial_lengths),
                                                  static_cast<std::size_t>(1),
                                                  std::multiplies<std::size_t>())) +
           sizeof(OutDataType) * (N * K *
                                  std::accumulate(std::begin(output_spatial_lengths),
                                                  std::end(output_spatial_lengths),
                                                  static_cast<std::size_t>(1),
                                                  std::multiplies<std::size_t>()));
}

struct ConvParams
{
    ConvParams()
        : num_dim_spatial(2),
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

    ck::index_t num_dim_spatial;
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
        std::vector<ck::index_t> out_spatial_len(num_dim_spatial, 0);
        for(ck::index_t i = 0; i < num_dim_spatial; ++i)
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

/**
 * @brief      Gets the host tensor descriptor.
 *
 * @param[in]  dims          The tensor dimensions lengths. Always in NCHW format.
 * @param[in]  layout        The tensor data layout.
 *
 * @tparam     TensorLayout  Layout type.
 *
 * @return     The host tensor descriptor object.
 */
template <typename TensorLayout>
HostTensorDescriptor GetHostTensorDescriptor(const std::vector<std::size_t>& dims,
                                             const TensorLayout& layout)
{
    std::size_t C = dims[1];
    // 1D
    if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NCW>::value ||
                 std::is_same<TensorLayout, ck::tensor_layout::convolution::KCX>::value ||
                 std::is_same<TensorLayout, ck::tensor_layout::convolution::NKW>::value)
    {

        return HostTensorDescriptor(dims, std::vector<std::size_t>({C * dims[2], dims[2], 1}));
    }
    else if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NWC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::KXC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::NWK>::value)
    {
        return HostTensorDescriptor(dims, std::vector<std::size_t>({C * dims[2], 1, C}));
    }
    // 2D
    else if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NCHW>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::KCYX>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::NKHW>::value)
    {

        return HostTensorDescriptor(
            dims, std::vector<std::size_t>{C * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1});
    }
    else if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NHWC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::KYXC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::NHWK>::value)
    {
        return HostTensorDescriptor(
            dims, std::vector<std::size_t>{C * dims[2] * dims[3], 1, dims[3] * C, C});
    }

    std::stringstream err_msg;
    err_msg << "Unsupported data layout provided: " << layout << "!";
    throw std::runtime_error(err_msg.str());
}

} // namespace conv_util
} // namespace ck

#endif
