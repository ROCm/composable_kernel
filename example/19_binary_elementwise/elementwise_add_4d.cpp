#include <iostream>
#include <cstdlib>
#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"

#include "device_tensor.hpp"
#include "binary_element_wise_operation.hpp"
#include "device_binary_elementwise.hpp"

using F16 = ck::half_t;
using F32 = float;

using ABDataType             = F16;
using CDataType              = F16;
using EltwiseComputeDataType = F32;

using Add = ck::tensor_operation::binary_element_wise::Add;

using DeviceElementwiseAddInstance = ck::tensor_operation::device::
    DeviceBinaryElementwise<ABDataType, ABDataType, CDataType, EltwiseComputeDataType, Add, 4, 8>;

template <typename HostTensorA,
          typename HostTensorB,
          typename HostTensorC,
          typename ComputeDataType,
          typename Functor>
void host_elementwise4D(HostTensorC& C,
                        const HostTensorA& A,
                        const HostTensorB& B,
                        const std::vector<std::size_t>& shape,
                        Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0, 0, 0, 0))>;

    for(std::size_t n = 0; n < shape[0]; ++n)
        for(std::size_t c = 0; c < shape[1]; ++c)
            for(std::size_t h = 0; h < shape[2]; ++h)
                for(std::size_t w = 0; w < shape[3]; ++w)
                {
                    ComputeDataType a_val = static_cast<ComputeDataType>(A(n, c, h, w));
                    ComputeDataType b_val = static_cast<ComputeDataType>(B(n, c, h, w));
                    ComputeDataType c_val = 0;
                    functor(c_val, a_val, b_val);
                    C(n, c, h, w) = static_cast<ctype>(c_val);
                }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    std::vector<std::size_t> nchw = {4, 16, 32, 32};

    Tensor<ABDataType> a_m(nchw);
    Tensor<ABDataType> b_m(nchw);
    Tensor<ABDataType> c_m(nchw);

    a_m.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});
    b_m.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});

    DeviceMem a_m_device_buf(sizeof(ABDataType) * a_m.mDesc.GetElementSpace());
    DeviceMem b_m_device_buf(sizeof(ABDataType) * b_m.mDesc.GetElementSpace());
    DeviceMem c_m_device_buf(sizeof(CDataType) * c_m.mDesc.GetElementSpace());

    a_m_device_buf.ToDevice(a_m.mData.data());
    b_m_device_buf.ToDevice(b_m.mData.data());

    auto broadcastAdd = DeviceElementwiseAddInstance{};
    auto argument     = broadcastAdd.MakeArgumentPointer(
        a_m_device_buf.GetDeviceBuffer(),
        b_m_device_buf.GetDeviceBuffer(),
        c_m_device_buf.GetDeviceBuffer(),
        std::vector<ck::index_t>{nchw.begin(), nchw.end()},
        std::vector<ck::index_t>{a_m.mDesc.GetStrides().begin(), a_m.mDesc.GetStrides().end()},
        std::vector<ck::index_t>{b_m.mDesc.GetStrides().begin(), b_m.mDesc.GetStrides().end()},
        std::vector<ck::index_t>{c_m.mDesc.GetStrides().begin(), c_m.mDesc.GetStrides().end()},
        Add{});

    if(!broadcastAdd.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error("The runtime parameters seems not supported by the "
                                 "DeviceBinaryElementwise_2D instance, exiting!");
    };

    auto broadcastAdd_invoker_ptr = broadcastAdd.MakeInvokerPointer();
    float ave_time =
        broadcastAdd_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_m_device_buf.FromDevice(c_m.mData.data());
        Tensor<CDataType> host_c_m(nchw);

        host_elementwise4D<Tensor<ABDataType>,
                           Tensor<ABDataType>,
                           Tensor<CDataType>,
                           EltwiseComputeDataType,
                           Add>(host_c_m, a_m, b_m, nchw, Add{});

        pass &= ck::utils::check_err(
            c_m.mData, host_c_m.mData, "Error: Incorrect results d1", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
