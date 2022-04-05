#ifndef CONV_FWD_UTIL_HPP
#define CONV_FWD_UTIL_HPP

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>
#include <random>
#include <tuple>
#include <type_traits>
#include <vector>

#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "device_conv_fwd.hpp"
#include "device_tensor.hpp"
#include "element_wise_operation.hpp"
#include "host_tensor.hpp"
#include "reference_conv_fwd.hpp"
#include "tensor_layout.hpp"

namespace ck {
namespace utils {
namespace conv {

using DeviceConvFwdNoOpPtr =
    ck::tensor_operation::device::DeviceConvFwdPtr<ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough,
                                                   ck::tensor_operation::element_wise::PassThrough>;

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
std::size_t get_flops(ck::index_t N,
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
std::size_t get_btype(ck::index_t N,
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

    ConvParams(ck::index_t n_dim,
               ck::index_t n_batch,
               ck::index_t n_out_channels,
               ck::index_t n_in_channels,
               const std::vector<ck::index_t>& filters_len,
               const std::vector<ck::index_t>& input_len,
               const std::vector<ck::index_t>& strides,
               const std::vector<ck::index_t>& dilations,
               const std::vector<ck::index_t>& left_pads,
               const std::vector<ck::index_t>& right_pads)
        : num_dim_spatial(n_dim),
          N(n_batch),
          K(n_out_channels),
          C(n_in_channels),
          filter_spatial_lengths(filters_len),
          input_spatial_lengths(input_len),
          conv_filter_strides(strides),
          conv_filter_dilations(dilations),
          input_left_pads(left_pads),
          input_right_pads(right_pads)
    {
        if(filter_spatial_lengths.size() != num_dim_spatial ||
           input_spatial_lengths.size() != num_dim_spatial ||
           conv_filter_strides.size() != num_dim_spatial ||
           conv_filter_dilations.size() != num_dim_spatial ||
           input_left_pads.size() != num_dim_spatial || input_right_pads.size() != num_dim_spatial)
        {
            throw(std::runtime_error(
                "ConvParams::GetOutputSpatialLengths: "
                "parameter size is different from number of declared dimensions!"));
        }
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
        if(filter_spatial_lengths.size() != num_dim_spatial ||
           input_spatial_lengths.size() != num_dim_spatial ||
           conv_filter_strides.size() != num_dim_spatial ||
           conv_filter_dilations.size() != num_dim_spatial ||
           input_left_pads.size() != num_dim_spatial || input_right_pads.size() != num_dim_spatial)
        {
            throw(std::runtime_error(
                "ConvParams::GetOutputSpatialLengths: "
                "parameter size is different from number of declared dimensions!"));
        }

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
HostTensorDescriptor get_host_tensor_descriptor(const std::vector<std::size_t>& dims,
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
    // 3D
    else if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NCDHW>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::KCZYX>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::NKDHW>::value)
    {

        return HostTensorDescriptor(dims,
                                    std::vector<std::size_t>{C * dims[2] * dims[3] * dims[4],
                                                             dims[2] * dims[3] * dims[4],
                                                             dims[3] * dims[4],
                                                             dims[4],
                                                             1});
    }
    else if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NDHWC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::KZYXC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::NDHWK>::value)
    {
        return HostTensorDescriptor(
            dims,
            std::vector<std::size_t>{
                C * dims[2] * dims[3] * dims[4], 1, C * dims[3] * dims[4], C * dims[4], C});
    }

    std::stringstream err_msg;
    err_msg << "Unsupported data layout provided: " << layout << "!";
    throw std::runtime_error(err_msg.str());
}

template <typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float,
          typename InLayout    = ck::tensor_layout::convolution::NHWC,
          typename WeiLayout   = ck::tensor_layout::convolution::KYXC,
          typename OutLayout   = ck::tensor_layout::convolution::NHWK>
auto get_host_tensors(const ConvParams& params, bool init = true)
{
    std::vector<std::size_t> input_dims{static_cast<std::size_t>(params.N),
                                        static_cast<std::size_t>(params.C)};
    input_dims.insert(std::end(input_dims),
                      std::begin(params.input_spatial_lengths),
                      std::end(params.input_spatial_lengths));

    std::vector<std::size_t> filter_dims{static_cast<std::size_t>(params.K),
                                         static_cast<std::size_t>(params.C)};
    filter_dims.insert(std::end(filter_dims),
                       std::begin(params.filter_spatial_lengths),
                       std::end(params.filter_spatial_lengths));

    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();
    std::vector<std::size_t> output_dims{static_cast<std::size_t>(params.N),
                                         static_cast<std::size_t>(params.K)};
    output_dims.insert(std::end(output_dims),
                       std::begin(output_spatial_lengths),
                       std::end(output_spatial_lengths));

    Tensor<InDataType> input(ck::utils::conv::get_host_tensor_descriptor(input_dims, InLayout{}));
    Tensor<WeiDataType> weights(
        ck::utils::conv::get_host_tensor_descriptor(filter_dims, WeiLayout{}));
    Tensor<OutDataType> host_output(
        ck::utils::conv::get_host_tensor_descriptor(output_dims, OutLayout{}));
    Tensor<OutDataType> device_output(
        ck::utils::conv::get_host_tensor_descriptor(output_dims, OutLayout{}));

    if(init)
    {
        std::mt19937 gen(11939);
        if constexpr(std::is_same<InDataType, uint8_t>::value)
        {
            std::uniform_int_distribution<> dis(-5, 5);
            std::generate(
                input.begin(), input.end(), [&dis, &gen]() { return InDataType(dis(gen)); });
            std::generate(
                weights.begin(), weights.end(), [&dis, &gen]() { return WeiDataType(dis(gen)); });
        }
        else
        {
            std::uniform_real_distribution<> dis(0.f, 1.f);
            std::generate(
                input.begin(), input.end(), [&dis, &gen]() { return InDataType(dis(gen)); });
            std::generate(
                weights.begin(), weights.end(), [&dis, &gen]() { return WeiDataType(dis(gen)); });
        }
        std::fill(host_output.begin(), host_output.end(), OutDataType(0.f));
        std::fill(device_output.begin(), device_output.end(), OutDataType(0.f));
    }

    return std::make_tuple(input, weights, host_output, device_output);
}

HostTensorDescriptor get_output_host_tensor_descriptor(const std::vector<std::size_t>& dims,
                                                       int num_dim_spatial = 2)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 3: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NDHWK{});
    }
    case 2: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NHWK{});
    }
    case 1: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NWK{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

HostTensorDescriptor get_filters_host_tensor_descriptor(const std::vector<std::size_t>& dims,
                                                        int num_dim_spatial = 2)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 3: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::KZYXC{});
    }
    case 2: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::KYXC{});
    }
    case 1: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::KXC{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

HostTensorDescriptor get_input_host_tensor_descriptor(const std::vector<std::size_t>& dims,
                                                      int num_dim_spatial = 2)
{
    namespace tl = ck::tensor_layout::convolution;

    switch(num_dim_spatial)
    {
    case 3: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NDHWC{});
    }
    case 2: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NHWC{});
    }
    case 1: {
        return ck::utils::conv::get_host_tensor_descriptor(dims, tl::NWC{});
    }
    default: {
        throw std::runtime_error("Unsupported number of spatial dimensions provided!");
    }
    }
}

template <ck::index_t NDim,
          typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float>
void run_reference_convolution_forward(const ConvParams& params,
                                       const Tensor<InDataType>& input,
                                       const Tensor<WeiDataType>& weights,
                                       Tensor<OutDataType>& output)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;
    auto ref_conv     = ck::tensor_operation::host::ReferenceConvFwd<InDataType,
                                                                 WeiDataType,
                                                                 OutDataType,
                                                                 PassThrough,
                                                                 PassThrough,
                                                                 PassThrough,
                                                                 NDim>();
    auto ref_invoker  = ref_conv.MakeInvoker();
    auto ref_argument = ref_conv.MakeArgument(input,
                                              weights,
                                              output,
                                              params.conv_filter_strides,
                                              params.conv_filter_dilations,
                                              params.input_left_pads,
                                              params.input_right_pads,
                                              PassThrough{},
                                              PassThrough{},
                                              PassThrough{});

    ref_invoker.Run(ref_argument);
}

template <ck::index_t NDim,
          typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float,
          template <ck::index_t, typename, typename, typename>
          class DeviceConvNDFwdInstance>
void run_convolution_forward(const ConvParams& params,
                             const Tensor<InDataType>& input,
                             const Tensor<WeiDataType>& weights,
                             Tensor<OutDataType>& output)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weights.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * output.mDesc.GetElementSpace());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weights.mData.data());
    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();

    auto conv     = DeviceConvNDFwdInstance<NDim, InDataType, WeiDataType, OutDataType>();
    auto invoker  = conv.MakeInvoker();
    auto argument = conv.MakeArgument(static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
                                      static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
                                      static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
                                      params.N,
                                      params.K,
                                      params.C,
                                      params.input_spatial_lengths,
                                      params.filter_spatial_lengths,
                                      output_spatial_lengths,
                                      params.conv_filter_strides,
                                      params.conv_filter_dilations,
                                      params.input_left_pads,
                                      params.input_right_pads,
                                      PassThrough{},
                                      PassThrough{},
                                      PassThrough{});

    if(!conv.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "Error! device_conv with the specified compilation parameters does "
            "not support this Conv problem");
    }

    invoker.Run(argument);
    out_device_buf.FromDevice(output.mData.data());
}

template <ck::index_t NDim,
          typename InDataType  = float,
          typename WeiDataType = float,
          typename OutDataType = float>
bool run_convolution_forward_instances(const ConvParams& params,
                                       const std::vector<DeviceConvFwdNoOpPtr>& conv_ptrs,
                                       const Tensor<InDataType>& input,
                                       const Tensor<WeiDataType>& weights,
                                       Tensor<OutDataType>& output,
                                       const Tensor<OutDataType>& host_output)
{
    using PassThrough = ck::tensor_operation::element_wise::PassThrough;

    DeviceMem in_device_buf(sizeof(InDataType) * input.mDesc.GetElementSpace());
    DeviceMem wei_device_buf(sizeof(WeiDataType) * weights.mDesc.GetElementSpace());
    DeviceMem out_device_buf(sizeof(OutDataType) * output.mDesc.GetElementSpace());

    in_device_buf.ToDevice(input.mData.data());
    wei_device_buf.ToDevice(weights.mData.data());
    const std::vector<ck::index_t>& output_spatial_lengths = params.GetOutputSpatialLengths();

    bool res{true};
    for(auto& conv_ptr : conv_ptrs)
    {
        auto invoker  = conv_ptr->MakeInvokerPointer();
        auto argument = conv_ptr->MakeArgumentPointer(
            static_cast<InDataType*>(in_device_buf.GetDeviceBuffer()),
            static_cast<WeiDataType*>(wei_device_buf.GetDeviceBuffer()),
            static_cast<OutDataType*>(out_device_buf.GetDeviceBuffer()),
            params.N,
            params.K,
            params.C,
            params.input_spatial_lengths,
            params.filter_spatial_lengths,
            output_spatial_lengths,
            params.conv_filter_strides,
            params.conv_filter_dilations,
            params.input_left_pads,
            params.input_right_pads,
            PassThrough{},
            PassThrough{},
            PassThrough{});

        if(conv_ptr->IsSupportedArgument(argument.get()))
        {
            float atol{1e-5f};
            float rtol{1e-4f};
            if constexpr(std::is_same_v<InDataType, ck::half_t>)
            {
                atol = 1e-4f;
                rtol = 2.5e-3f;
            }
            invoker->Run(argument.get());
            out_device_buf.FromDevice(output.mData.data());
            res = res &&
                  ck::utils::check_err(
                      output.mData, host_output.mData, "Error: incorrect results!", atol, rtol);
            hipGetErrorString(
                hipMemset(out_device_buf.GetDeviceBuffer(), 0, out_device_buf.mMemSize));
        }
    }
    return res;
}

} // namespace conv
} // namespace utils
} // namespace ck

#endif
