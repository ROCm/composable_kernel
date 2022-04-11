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
//#include "tensor_descriptor_helper.hpp"
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
    transfer_half2_to_bhalf2(src_data, dst_data);

    dst_buf.template Set<ck::bhalf2_t>(num, true, dst_data);
}

void host_conver_data(SrcDataType* in, DstDataType* out, size_t len)
{
    for(int i = 0; i < len; i++)
    {
        float tmp = static_cast<float>(in[i]);
        out[i]    = ck::type_convert<DstDataType, float>(tmp);
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
    src_n_k_host.GenerateTensorValue(GeneratorTensor_2<SrcDataType>{-5, 5});
    dst_n_k_host_result.GenerateTensorValue(GeneratorTensor_1<DstDataType>{0});
    dst_n_k_device_result.GenerateTensorValue(GeneratorTensor_1<DstDataType>{0});

    // alloc gpu memory
    DeviceMem in_dev_buf(sizeof(SrcDataType) * src_n_k_host.mDesc.GetElementSpace());
    DeviceMem out_dev_buf(sizeof(DstDataType) * dst_n_k_host_result.mDesc.GetElementSpace());
    // init gpu memory
    in_dev_buf.ToDevice(src_n_k_host.mData.data());
    out_dev_buf.SetZero();

    // run kernel to convert data
    gpu_convert_data<<<1, thread_num>>>(static_cast<SrcDataType*>(in_dev_buf.GetDeviceBuffer()),
                                        static_cast<DstDataType*>(out_dev_buf.GetDeviceBuffer()),
                                        src_n_k_host.mDesc.GetElementSpace());
    // read from gpu
    out_dev_buf.FromDevice(dst_n_k_device_result.mData.data());

    // run cpu data convert
    host_conver_data(src_n_k_host.mData.data(), dst_n_k_host_result.mData.data(), size);

#if 1
    LogRangeAsType<float>(std::cout << "in : ", src_n_k_host.mData, ",") << std::endl;
    LogRangeAsType<float>(std::cout << "out device: ", dst_n_k_device_result.mData, ",")
        << std::endl;
    LogRangeAsType<float>(std::cout << "out host: ", dst_n_k_host_result.mData, ",") << std::endl;
#endif

    pass = ck::utils::check_err(dst_n_k_device_result.mData, dst_n_k_host_result.mData);
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
