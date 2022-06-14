#pragma once

#include "host_interface.hpp"

enum ConvDataType
{
    F32_F32_F32,    // 0
    F16_F16_F16,    // 1
    BF16_BF16_BF16, // 2
    INT8_INT8_INT8, // 3
};

enum ConvInputLayout
{
    NCHW, // 0
    NHWC, // 1
};

enum ConvWeightLayout
{
    KCYX, // 0
    KYXC, // 1
};

enum ConvOutputLayout
{
    NKHW, // 0
    NHWK, // 1
};

void check_hip_error(void)
{
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
        exit(err);
    }
}
std::string getDeviceName(int device)
{
    struct hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, device);
    check_hip_error();
    return std::string(prop.name);
}

int getDriver(void)
{
    int driver;
    hipDriverGetVersion(&driver);
    check_hip_error();
    return driver;
}

namespace ck {
namespace app {
struct DeviceMem
{
    DeviceMem() = delete;
    DeviceMem(std::size_t mem_size);
    void* GetDeviceBuffer();
    void ToDevice(const void* p);
    void FromDevice(void* p);
    ~DeviceMem();

    void* mpDeviceBuf;
    std::size_t mMemSize;
};

DeviceMem::DeviceMem(std::size_t mem_size) : mMemSize(mem_size)
{
    hipGetErrorString(hipMalloc(static_cast<void**>(&mpDeviceBuf), mMemSize));
}

void* DeviceMem::GetDeviceBuffer() { return mpDeviceBuf; }

void DeviceMem::ToDevice(const void* p)
{
    hipGetErrorString(
        hipMemcpy(mpDeviceBuf, const_cast<void*>(p), mMemSize, hipMemcpyHostToDevice));
}

void DeviceMem::FromDevice(void* p)
{
    hipGetErrorString(hipMemcpy(p, mpDeviceBuf, mMemSize, hipMemcpyDeviceToHost));
}

DeviceMem::~DeviceMem() { hipGetErrorString(hipFree(mpDeviceBuf)); }

void profile_conv_fwd_impl(int do_verification,
                           int init_method,
                           bool do_log,
                           bool time_kernel,
                           ConvDataType data_type,
                           ck::index_t N,
                           ck::index_t K,
                           ck::index_t C,
                           std::vector<ck::index_t> input_spatial_lengths,
                           std::vector<ck::index_t> filter_spatial_lengths,
                           std::vector<ck::index_t> output_spatial_lengths,
                           std::vector<ck::index_t> conv_filter_strides,
                           std::vector<ck::index_t> conv_filter_dilations,
                           std::vector<ck::index_t> input_left_pads,
                           std::vector<ck::index_t> input_right_pads)
{
    const ck::index_t Y = filter_spatial_lengths[0];
    const ck::index_t X = filter_spatial_lengths[1];

    const ck::index_t Hi = input_spatial_lengths[0];
    const ck::index_t Wi = input_spatial_lengths[1];

    const ck::index_t Ho = output_spatial_lengths[0];
    const ck::index_t Wo = output_spatial_lengths[1];

    const auto in_sz  = N * C * Hi * Wi;
    const auto wei_sz = K * C * Y * X;
    const auto out_sz = N * K * Ho * Wo;

    using WeiDataType = float;
    using InDataType  = float;
    using OutDataType = float;

    app::DeviceMem in_device_buf(sizeof(InDataType) * in_sz);
    app::DeviceMem wei_device_buf(sizeof(WeiDataType) * wei_sz);
    app::DeviceMem out_device_buf(sizeof(OutDataType) * out_sz);
    // data is already on device!

    // add device Conv instances
    std::vector<DeviceConvFwdPtr_t> conv_ptrs;
    if(data_type == F16_F16_F16)
    {
        add_device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk_f16_instances_t(conv_ptrs);
        add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f16_instances_t(conv_ptrs);
    }
    else if(data_type == BF16_BF16_BF16)
        add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_bf16_instances_t(conv_ptrs);
    else if(data_type == F32_F32_F32)
        add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_f32_instances_t(conv_ptrs);
    else if(data_type == INT8_INT8_INT8)
        add_device_conv2d_fwd_xdl_nhwc_kyxc_nhwk_int8_instances_t(conv_ptrs);
    else
        throw std::runtime_error("wrong! Invalid data type");
    if(conv_ptrs.empty())
    {
        throw std::runtime_error("wrong! no device Conv instance found");
    }

    std::string best_conv_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    int deviceIndex       = 0;
    hipSetDevice(deviceIndex);
    check_hip_error();

    StreamConfig stream_config{nullptr, time_kernel};
    hipStreamCreate(&stream_config.stream_id_);
    check_hip_error();

    // profile device Conv instances
    for(auto& conv_ptr : conv_ptrs)
    {
        auto argument_ptr =
            conv_ptr.MakeArgumentPointer(static_cast<void*>(in_device_buf.GetDeviceBuffer()),
                                         static_cast<void*>(wei_device_buf.GetDeviceBuffer()),
                                         static_cast<void*>(out_device_buf.GetDeviceBuffer()),
                                         N,
                                         K,
                                         C,
                                         input_spatial_lengths,
                                         filter_spatial_lengths,
                                         output_spatial_lengths,
                                         conv_filter_strides,
                                         conv_filter_dilations,
                                         input_left_pads,
                                         input_right_pads);

        auto invoker_ptr = conv_ptr.MakeInvokerPointer();

        if(conv_ptr.IsSupportedArgument(argument_ptr.get()))
        {
            std::string conv_name = conv_ptr.GetTypeString();
            float ave_time        = invoker_ptr->Run(argument_ptr.get(), stream_config);

            std::size_t flop = std::size_t(2) * N * K * Ho * Wo * C * Y * X;

            std::size_t num_btype = sizeof(InDataType) * (N * C * Hi * Wi) +
                                    sizeof(WeiDataType) * (K * C * Y * X) +
                                    sizeof(OutDataType) * (N * K * Ho * Wo);

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << conv_name << std::endl;

            if(tflops > best_tflops)
            {
                best_conv_name  = conv_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_conv_name << std::endl;
}

} // namespace app
} // namespace ck
