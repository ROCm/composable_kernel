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
//#include "tensor_descriptor.hpp"
//#include "tensor_descriptor_helper.hpp"
#include "common_header.hpp"

using SrcDataType = ck::half_t;
using DstDataType = ck::bhalf_t;

__global__ void gpu_convert_data(SrcDataType* in, DstDataType* out, int size)
{
    using namespace ck;
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};

    ck::index_t num    = blockIdx.x * blockDim.x + threadIdx.x * 2;
    const auto src_buf = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(in, size);
    auto dst_buf       = ck::make_dynamic_buffer<ck::AddressSpaceEnum::Global>(out, size);

    using src_vector_t = vector_type_maker<SrcDataType, 2>::type;
    // using dst_vector_t = vector_type_maker<DstDataType, 2>::type;

    src_vector_t src_data = src_buf.template Get<src_vector_t>(num, true);

    const vector_type<half_t, 2> vx0{src_data};
    vector_type<DstDataType, 2> vy0;

    float v1 = static_cast<float>(vx0.template AsType<half_t>()[I0]);
    float v2 = static_cast<float>(vx0.template AsType<half_t>()[I1]);

    vy0.template AsType<DstDataType>()(I0) = ck::type_convert<DstDataType>(v1);
    vy0.template AsType<DstDataType>()(I1) = ck::type_convert<DstDataType>(v2);

    // dst_vector_t vy = vy0.template AsType<DstDataType>()[I0];
    // dst_buf.template Set<dst_vector_t>(num, true, vy);
    dst_buf(num)     = vy0.template AsType<DstDataType>()[I0];
    dst_buf(num + 1) = vy0.template AsType<DstDataType>()[I1];
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
    src_n_k_host.GenerateTensorValue(GeneratorTensor_3<SrcDataType>{-5, 5});
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
