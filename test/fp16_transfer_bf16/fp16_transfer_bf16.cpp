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

using SrcDataType = ck::half_t;
using DstDataType = ck::bhalf_t;

__global__ void gpu_convert_data(SrcDataType* in, DstDataType* out)
{
    int num = blockIdx.x * blockDim.x + threadIdx.x;
    
    float tmp = static_cast<float>(in[num]);
	out[num] = ck::type_convert<DstDataType>(tmp);
}

void host_conver_data(SrcDataType* in, DstDataType* out, size_t len)
{
     for(int i =0; i < len; i++){
        float tmp = static_cast<float>(in[i]);
	    out[i] = ck::type_convert<DstDataType,float>(tmp);
     }
}
int main(int, char*[])
{
    bool pass = true;
    constexpr int N = 8;
    constexpr int K = 8;
    constexpr int thread_num = N * K;
    

    //create tensor
    Tensor<SrcDataType> src_n_k_host(HostTensorDescriptor(std::vector<std::size_t>({N, K}),
                                            std::vector<std::size_t>({K, 1})));
    Tensor<DstDataType> dst_n_k_host_result(HostTensorDescriptor(std::vector<std::size_t>({N, K}),
                                            std::vector<std::size_t>({K, 1})));
    Tensor<DstDataType> dst_n_k_device_result(HostTensorDescriptor(std::vector<std::size_t>({N, K}),
                                            std::vector<std::size_t>({K, 1})));

    //init data
    src_n_k_host.GenerateTensorValue(GeneratorTensor_3<SrcDataType>{-5, 5});
    dst_n_k_host_result.GenerateTensorValue(GeneratorTensor_1<DstDataType>{0});
    dst_n_k_device_result.GenerateTensorValue(GeneratorTensor_1<DstDataType>{0});
    
    //alloc gpu memory
    DeviceMem in_dev_buf(sizeof(SrcDataType) * src_n_k_host.mDesc.GetElementSpace());
    DeviceMem out_dev_buf(sizeof(DstDataType) * src_n_k_host.mDesc.GetElementSpace());
    //init gpu memory
    in_dev_buf.ToDevice(src_n_k_host.mData.data());
    out_dev_buf.SetZero();
    
    //run kernel to convert data
    gpu_convert_data<<<1, thread_num>>>(
            static_cast<SrcDataType*>(in_dev_buf.GetDeviceBuffer()),
            static_cast<DstDataType*>(out_dev_buf.GetDeviceBuffer()));
    //read from gpu
    out_dev_buf.FromDevice(dst_n_k_device_result.mData.data());

    //run cpu data convert
    host_conver_data(src_n_k_host.mData.data(), dst_n_k_host_result.mData.data(), thread_num);

    
#if 0
    LogRangeAsType<float>(std::cout << "in : ", src_n_k_host.mData, ",")
                        << std::endl;
    LogRangeAsType<float>(std::cout << "out device: ", dst_n_k_device_result.mData, ",")
                        << std::endl;
    LogRangeAsType<float>(std::cout << "out host: ", dst_n_k_host_result.mData, ",")
                        << std::endl;
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
