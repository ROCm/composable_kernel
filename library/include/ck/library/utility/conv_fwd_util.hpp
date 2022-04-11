#pragma once

#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <vector>

#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "device_conv_fwd.hpp"
#include "device_tensor.hpp"
#include "element_wise_operation.hpp"
#include "fill.hpp"
#include "host_tensor.hpp"
#include "op_instance_engine.hpp"
#include "reference_conv_fwd.hpp"
#include "tensor_layout.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

using DeviceConvFwdNoOpPtr = DeviceConvFwdPtr<element_wise::PassThrough,
                                              element_wise::PassThrough,
                                              element_wise::PassThrough>;
namespace device_conv1d_fwd_instance {

void add_device_conv1d_fwd_xdl_nwc_kxc_nwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv1d_fwd_xdl_nwc_kxc_nwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv1d_fwd_xdl_nwc_kxc_nwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv1d_fwd_xdl_nwc_kxc_nwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);

} // namespace device_conv1d_fwd_instance
namespace device_conv2d_fwd_instance {

void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(
    std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);

} // namespace device_conv2d_fwd_instance
namespace device_conv3d_fwd_instance {

void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f16_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f32_instances(std::vector<DeviceConvFwdNoOpPtr>&);
void add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_int8_instances(std::vector<DeviceConvFwdNoOpPtr>&);

} // namespace device_conv3d_fwd_instance

} // namespace device
} // namespace tensor_operation
} // namespace ck

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

ConvParams parse_conv_params(int num_dim_spatial, int arg_idx, char* const argv[])
{
    ck::utils::conv::ConvParams params;

    params.num_dim_spatial = num_dim_spatial;
    params.N               = std::stoi(argv[arg_idx++]);
    params.K               = std::stoi(argv[arg_idx++]);
    params.C               = std::stoi(argv[arg_idx++]);

    params.filter_spatial_lengths.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.filter_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_spatial_lengths.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_spatial_lengths[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_strides.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_strides[i] = std::stoi(argv[arg_idx++]);
    }
    params.conv_filter_dilations.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.conv_filter_dilations[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_left_pads.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_left_pads[i] = std::stoi(argv[arg_idx++]);
    }
    params.input_right_pads.resize(num_dim_spatial);
    for(int i = 0; i < num_dim_spatial; ++i)
    {
        params.input_right_pads[i] = std::stoi(argv[arg_idx++]);
    }

    return params;
}

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

        return HostTensorDescriptor(dims, std::vector<std::size_t>{C * dims[2], dims[2], 1});
    }
    else if constexpr(std::is_same<TensorLayout, ck::tensor_layout::convolution::NWC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::KXC>::value ||
                      std::is_same<TensorLayout, ck::tensor_layout::convolution::NWK>::value)
    {
        return HostTensorDescriptor(dims, std::vector<std::size_t>{C * dims[2], 1, C});
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

template <typename InDataType, typename WeiDataType, typename OutDataType>
struct ConvolutionFwdInstances;

template <>
struct ConvolutionFwdInstances<float, float, float>
{
    template <int NumDimSpatial,
              typename std::enable_if<NumDimSpatial >= 1 && NumDimSpatial <= 3, bool>::type = false>
    static std::vector<DeviceConvFwdNoOpPtr> Get()
    {
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
        if constexpr(NumDimSpatial == 1)
        {
            ck::tensor_operation::device::device_conv1d_fwd_instance::
                add_device_conv1d_fwd_xdl_nwc_kxc_nwk_f32_instances(conv_ptrs);
        }
        else if constexpr(NumDimSpatial == 2)
        {
            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances(conv_ptrs);
        }
        else if constexpr(NumDimSpatial == 3)
        {
            ck::tensor_operation::device::device_conv3d_fwd_instance::
                add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f32_instances(conv_ptrs);
        }
        return conv_ptrs;
    }
};

template <>
struct ConvolutionFwdInstances<half_t, half_t, half_t>
{
    template <int NumDimSpatial,
              typename std::enable_if<NumDimSpatial >= 1 && NumDimSpatial <= 3, bool>::type = false>
    static std::vector<DeviceConvFwdNoOpPtr> Get()
    {
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
        if constexpr(NumDimSpatial == 1)
        {
            ck::tensor_operation::device::device_conv1d_fwd_instance::
                add_device_conv1d_fwd_xdl_nwc_kxc_nwk_f16_instances(conv_ptrs);
            return conv_ptrs;
        }
        else if constexpr(NumDimSpatial == 2)
        {
            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances(conv_ptrs);
        }
        else if constexpr(NumDimSpatial == 3)
        {
            ck::tensor_operation::device::device_conv3d_fwd_instance::
                add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_f16_instances(conv_ptrs);
        }
        return conv_ptrs;
    }
};

template <>
struct ConvolutionFwdInstances<bhalf_t, bhalf_t, bhalf_t>
{
    template <int NumDimSpatial,
              typename std::enable_if<NumDimSpatial >= 1 && NumDimSpatial <= 3, bool>::type = false>
    static std::vector<DeviceConvFwdNoOpPtr> Get()
    {
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
        if constexpr(NumDimSpatial == 1)
        {
            ck::tensor_operation::device::device_conv1d_fwd_instance::
                add_device_conv1d_fwd_xdl_nwc_kxc_nwk_bf16_instances(conv_ptrs);
        }
        else if constexpr(NumDimSpatial == 2)
        {
            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances(conv_ptrs);
        }
        else if constexpr(NumDimSpatial == 3)
        {
            ck::tensor_operation::device::device_conv3d_fwd_instance::
                add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_bf16_instances(conv_ptrs);
        }
        return conv_ptrs;
    }
};

template <>
struct ConvolutionFwdInstances<int8_t, int8_t, int8_t>
{
    template <int NumDimSpatial,
              typename std::enable_if<NumDimSpatial >= 1 && NumDimSpatial <= 3, bool>::type = false>
    static std::vector<DeviceConvFwdNoOpPtr> Get()
    {
        std::vector<DeviceConvFwdNoOpPtr> conv_ptrs;
        if constexpr(NumDimSpatial == 1)
        {
            ck::tensor_operation::device::device_conv1d_fwd_instance::
                add_device_conv1d_fwd_xdl_nwc_kxc_nwk_int8_instances(conv_ptrs);
        }
        else if constexpr(NumDimSpatial == 2)
        {
            ck::tensor_operation::device::device_conv2d_fwd_instance::
                add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances(conv_ptrs);
        }
        else if constexpr(NumDimSpatial == 3)
        {
            ck::tensor_operation::device::device_conv3d_fwd_instance::
                add_device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk_int8_instances(conv_ptrs);
        }
        return conv_ptrs;
    }
};

template <typename InDataType,
          typename WeiDataType,
          typename OutDataType,
          typename InLayout         = ck::tensor_layout::convolution::NHWC,
          typename WeiLayout        = ck::tensor_layout::convolution::KYXC,
          typename OutLayout        = ck::tensor_layout::convolution::NHWK,
          typename InElementwiseOp  = ck::tensor_operation::element_wise::PassThrough,
          typename WeiElementwiseOp = ck::tensor_operation::element_wise::PassThrough,
          typename OutElementwiseOp = ck::tensor_operation::element_wise::PassThrough,
          typename InputInitFun     = FillUniform<InDataType>,
          typename WeightsInitFun   = FillUniform<WeiDataType>>
class ConvFwdOpInstance : public ck::utils::OpInstance<OutDataType, InDataType, WeiDataType>
{
    using DeviceConvFwdOp = tensor_operation::device::
        DeviceConvFwd<InElementwiseOp, WeiElementwiseOp, OutElementwiseOp>;
    using DeviceMemPtr  = std::unique_ptr<DeviceMem>;
    using DeviceBuffers = std::vector<DeviceMemPtr>;
    using BaseType      = ck::utils::OpInstance<OutDataType, InDataType, WeiDataType>;
    template <typename T>
    using TensorPtr      = std::unique_ptr<Tensor<T>>;
    using InTensorsTuple = std::tuple<TensorPtr<InDataType>, TensorPtr<WeiDataType>>;

    public:
    ConvFwdOpInstance()                         = delete;
    ConvFwdOpInstance(const ConvFwdOpInstance&) = default;
    ConvFwdOpInstance& operator=(const ConvFwdOpInstance&) = default;

    ConvFwdOpInstance(const ConvParams& params,
                      bool do_init                         = true,
                      const InputInitFun& input_init_f     = InputInitFun{},
                      const WeightsInitFun& weights_init_f = WeightsInitFun{})
        : BaseType(),
          params_{params},
          output_spatial_lengths_{params.GetOutputSpatialLengths()},
          do_init_{do_init},
          input_init_f_{input_init_f},
          weights_init_f_{weights_init_f}
    {
    }

    virtual ~ConvFwdOpInstance() override{};

    virtual InTensorsTuple GetInputTensors() const override
    {
        std::vector<std::size_t> input_dims{static_cast<std::size_t>(params_.N),
                                            static_cast<std::size_t>(params_.C)};
        input_dims.insert(std::end(input_dims),
                          std::begin(params_.input_spatial_lengths),
                          std::end(params_.input_spatial_lengths));

        std::vector<std::size_t> filter_dims{static_cast<std::size_t>(params_.K),
                                             static_cast<std::size_t>(params_.C)};
        filter_dims.insert(std::end(filter_dims),
                           std::begin(params_.filter_spatial_lengths),
                           std::end(params_.filter_spatial_lengths));

        auto input = std::make_unique<Tensor<InDataType>>(
            get_host_tensor_descriptor(input_dims, InLayout{}));
        auto weights = std::make_unique<Tensor<WeiDataType>>(
            get_host_tensor_descriptor(filter_dims, WeiLayout{}));

        if(do_init_)
        {
            input_init_f_(input->begin(), input->end());
            weights_init_f_(weights->begin(), weights->end());
        }

        return std::make_tuple(std::move(input), std::move(weights));
    }

    virtual TensorPtr<OutDataType> GetOutputTensor() const override
    {
        std::vector<std::size_t> output_dims{static_cast<std::size_t>(params_.N),
                                             static_cast<std::size_t>(params_.K)};
        output_dims.insert(std::end(output_dims),
                           std::begin(output_spatial_lengths_),
                           std::end(output_spatial_lengths_));
        auto output = std::make_unique<Tensor<OutDataType>>(
            get_host_tensor_descriptor(output_dims, OutLayout{}));

        if(do_init_)
        {
            std::fill(output->begin(), output->end(), OutDataType(0.f));
        }
        return output;
    }

    virtual std::unique_ptr<tensor_operation::device::BaseInvoker>
    MakeInvokerPointer(tensor_operation::device::BaseOperator* op_ptr) const override
    {
        static_assert(
            std::is_same_v<InElementwiseOp, ck::tensor_operation::element_wise::PassThrough>);
        static_assert(
            std::is_same_v<OutElementwiseOp, ck::tensor_operation::element_wise::PassThrough>);
        static_assert(
            std::is_same_v<WeiElementwiseOp, ck::tensor_operation::element_wise::PassThrough>);

        auto conv_ptr = dynamic_cast<DeviceConvFwdOp*>(op_ptr);
        if(!conv_ptr)
        {
            throw std::runtime_error(
                "[ConvFwdOpInstance]: couldn't cast op_ptr to DeviceConvFwdNoOpPtr type!");
        }
        return conv_ptr->MakeInvokerPointer();
    }

    virtual std::unique_ptr<tensor_operation::device::BaseArgument>
    MakeArgumentPointer(tensor_operation::device::BaseOperator* op_ptr,
                        const DeviceBuffers& in_device_buffers,
                        const DeviceMemPtr& out_device_buffer) const override
    {
        static_assert(
            std::is_same_v<InElementwiseOp, ck::tensor_operation::element_wise::PassThrough>);
        static_assert(
            std::is_same_v<OutElementwiseOp, ck::tensor_operation::element_wise::PassThrough>);
        static_assert(
            std::is_same_v<WeiElementwiseOp, ck::tensor_operation::element_wise::PassThrough>);

        auto conv_ptr = dynamic_cast<DeviceConvFwdOp*>(op_ptr);
        if(!conv_ptr)
        {
            throw std::runtime_error(
                "[ConvFwdOpInstance]: couldn't cast op_ptr to DeviceConvFwdNoOpPtr type!");
        }

        return conv_ptr->MakeArgumentPointer(
            static_cast<InDataType*>(in_device_buffers[0]->GetDeviceBuffer()),
            static_cast<WeiDataType*>(in_device_buffers[1]->GetDeviceBuffer()),
            static_cast<OutDataType*>(out_device_buffer->GetDeviceBuffer()),
            params_.N,
            params_.K,
            params_.C,
            params_.input_spatial_lengths,
            params_.filter_spatial_lengths,
            output_spatial_lengths_,
            params_.conv_filter_strides,
            params_.conv_filter_dilations,
            params_.input_left_pads,
            params_.input_right_pads,
            InElementwiseOp{},
            WeiElementwiseOp{},
            OutElementwiseOp{});
    }

    virtual std::size_t GetFlops() const override
    {
        return get_flops(params_.N,
                         params_.C,
                         params_.K,
                         params_.filter_spatial_lengths,
                         output_spatial_lengths_);
    }

    virtual std::size_t GetBtype() const override
    {
        return get_btype<InDataType, WeiDataType, OutDataType>(params_.N,
                                                               params_.C,
                                                               params_.K,
                                                               params_.input_spatial_lengths,
                                                               params_.filter_spatial_lengths,
                                                               output_spatial_lengths_);
    }

    private:
    const ConvParams& params_;
    const std::vector<ck::index_t> output_spatial_lengths_;
    const bool do_init_;
    const InputInitFun& input_init_f_;
    const WeightsInitFun& weights_init_f_;
};

} // namespace conv
} // namespace utils
} // namespace ck

std::ostream& operator<<(std::ostream& os, const ck::utils::conv::ConvParams& p)
{
    os << "ConvParams {"
       << "\nnum_dim_spatial: " << p.num_dim_spatial << "\nN: " << p.N << "\nK: " << p.K
       << "\nC: " << p.C << "\nfilter_spatial_lengths: " << p.filter_spatial_lengths
       << "\ninput_spatial_lengths: " << p.input_spatial_lengths
       << "\nconv_filter_strides: " << p.conv_filter_strides
       << "\nconv_filter_dilations: " << p.conv_filter_dilations
       << "\ninput_left_pads: " << p.input_left_pads
       << "\ninput_right_pads: " << p.input_right_pads;
    return os;
}
