#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <stdlib.h>
#include <half.hpp>

#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "device_tensor.hpp"
#include "check_err.hpp"
#include "transpose_vectors.hpp"
#include "common_header.hpp"

using SrcDataType = ck::half_t;
using DstDataType = ck::bhalf_t;

__global__ void gpu_convert_data(SrcDataType* in, DstDataType* out, int size)
{
    using namespace ck;

    ck::index_t num    = blockIdx.x * blockDim.x + threadIdx.x * 2;
    const auto src_buf = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(in, size);
    auto dst_buf       = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(out, size);

    auto src_data = src_buf.template Get<ck::half2_t>(num, true);
    ck::bhalf2_t dst_data;
    convert_half2_to_bhalf2(src_data, dst_data);

    dst_buf.template Set<ck::bhalf2_t>(num, true, dst_data);
}

__global__ void
gpu_transpose_convert_data(SrcDataType* in, DstDataType* out, const int size, const int stride)
{
    using namespace ck;

    ck::index_t num    = blockIdx.x * blockDim.x + threadIdx.x * 2;
    const auto src_buf = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(in, size);
    auto dst_buf       = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(out, size);
    int x              = num % stride;
    int y              = num / stride;
    int num1           = (y + 1) * stride + x;

    auto src_data0 = src_buf.template Get<ck::half2_t>(num, true);
    auto src_data1 = src_buf.template Get<ck::half2_t>(num1, true);
    ck::bhalf2_t dst_data0, dst_data1;
    transpose_half_to_bhalf_2x2(src_data0, src_data1, dst_data0, dst_data1);

    // rewrite
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    const vector_type<bhalf_t, 2> vx0{dst_data0}, vx1{dst_data1};
    vector_type<bhalf_t, 2> vy0, vy1;
    vy0.template AsType<bhalf_t>()(I0) = vx0.template AsType<bhalf_t>()[I0];
    vy0.template AsType<bhalf_t>()(I1) = vx1.template AsType<bhalf_t>()[I0];

    vy1.template AsType<bhalf_t>()(I0) = vx0.template AsType<bhalf_t>()[I1];
    vy1.template AsType<bhalf_t>()(I1) = vx1.template AsType<bhalf_t>()[I1];

    dst_buf.template Set<ck::bhalf2_t>(num, true, vy0.template AsType<ck::bhalf2_t>()[I0]);
    dst_buf.template Set<ck::bhalf2_t>(num1, true, vy1.template AsType<ck::bhalf2_t>()[I0]);
}

void host_convert_data(SrcDataType* in, DstDataType* out, size_t len)
{
    for(size_t i = 0; i < len; i++)
    {
        out[i] = ck::type_convert<ck::bhalf_t, ck::half_t>(in[i]);
    }
}
int main(int, char*[])
{
    bool pass                = true;
    constexpr int N          = 4;
    constexpr int K          = 4;
    constexpr int size       = N * K;
    constexpr int thread_num = size / 2;

    // create tensor
    Tensor<SrcDataType> src_n_k_host(
        HostTensorDescriptor(std::vector<std::size_t>({N, K}), std::vector<std::size_t>({K, 1})));
    Tensor<DstDataType> dst_n_k_host_result(
        HostTensorDescriptor(std::vector<std::size_t>({N, K}), std::vector<std::size_t>({K, 1})));
    Tensor<DstDataType> dst_n_k_device_result(
        HostTensorDescriptor(std::vector<std::size_t>({N, K}), std::vector<std::size_t>({K, 1})));

    // init data
    src_n_k_host.GenerateTensorValue(GeneratorTensor_3<SrcDataType>{-5, 5});
    dst_n_k_host_result.GenerateTensorValue(GeneratorTensor_1<DstDataType>{0});
    dst_n_k_device_result.GenerateTensorValue(GeneratorTensor_1<DstDataType>{0});

    // alloc gpu memory
    DeviceMem in_dev_buf(sizeof(SrcDataType) * src_n_k_host.mDesc.GetElementSpace());
    DeviceMem out_dev_buf(sizeof(DstDataType) * dst_n_k_host_result.mDesc.GetElementSpace());
    // init gpu memory
    in_dev_buf.ToDevice(src_n_k_host.mData.data());
    out_dev_buf.SetZero();

    // run cpu data convert
    host_convert_data(src_n_k_host.mData.data(), dst_n_k_host_result.mData.data(), size);
    // run kernel to convert data
    gpu_convert_data<<<1, thread_num>>>(static_cast<SrcDataType*>(in_dev_buf.GetDeviceBuffer()),
                                        static_cast<DstDataType*>(out_dev_buf.GetDeviceBuffer()),
                                        src_n_k_host.mDesc.GetElementSpace());
    // read from gpu
    out_dev_buf.FromDevice(dst_n_k_device_result.mData.data());

    pass = ck::utils::check_err(dst_n_k_device_result.mData, dst_n_k_host_result.mData);

    // run kernel to tanspos and convert data
    gpu_transpose_convert_data<<<1, thread_num / 2>>>(
        static_cast<SrcDataType*>(in_dev_buf.GetDeviceBuffer()),
        static_cast<DstDataType*>(out_dev_buf.GetDeviceBuffer()),
        src_n_k_host.mDesc.GetElementSpace(),
        K);
    // read from gpu
    out_dev_buf.FromDevice(dst_n_k_device_result.mData.data());
    pass &= ck::utils::check_err(dst_n_k_device_result.mData, dst_n_k_host_result.mData);
#if 1
    LogRangeAsType<float>(std::cout << "in : ", src_n_k_host.mData, ",") << std::endl;
    LogRangeAsType<float>(std::cout << "out device: ", dst_n_k_device_result.mData, ",")
        << std::endl;
    LogRangeAsType<float>(std::cout << "out host: ", dst_n_k_host_result.mData, ",") << std::endl;
#endif

    if(pass)
    {
        std::cout << "fp16 transfer to bf16: Pass" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "fp16 transfer to bf16: Fail" << std::endl;
        return -1;
    }
}
