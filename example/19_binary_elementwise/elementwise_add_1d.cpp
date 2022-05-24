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
    DeviceBinaryElementwise<ABDataType, ABDataType, CDataType, EltwiseComputeDataType, Add, 1, 8>;

template <typename HostTensorA,
          typename HostTensorB,
          typename HostTensorC,
          typename ComputeDataType,
          typename Functor>
void host_elementwise1D(
    HostTensorC& C, const HostTensorA& A, const HostTensorB& B, int M, Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0))>;

    for(int m = 0; m < M; ++m)
    {
        ComputeDataType Am = static_cast<ComputeDataType>(A(m));
        ComputeDataType Bm = static_cast<ComputeDataType>(B(m));
        ComputeDataType Cm = 0;
        functor(Cm, Am, Bm);
        C(m) = static_cast<ctype>(Cm);
    }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    ck::index_t M = 1024;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor(std::vector<std::size_t>({len}),
                                    std::vector<std::size_t>({stride}));
    };

    Tensor<ABDataType> a_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<ABDataType> b_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<CDataType> c_m(f_host_tensor_descriptor1d(M, 1));

    a_m.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});
    b_m.GenerateTensorValue(GeneratorTensor_3<ABDataType>{0.0, 1.0});

    DeviceMem a_m_device_buf(sizeof(ABDataType) * a_m.mDesc.GetElementSpace());
    DeviceMem b_m_device_buf(sizeof(ABDataType) * b_m.mDesc.GetElementSpace());
    DeviceMem c_m_device_buf(sizeof(CDataType) * c_m.mDesc.GetElementSpace());

    a_m_device_buf.ToDevice(a_m.mData.data());
    b_m_device_buf.ToDevice(b_m.mData.data());

    auto broadcastAdd = DeviceElementwiseAddInstance{};
    auto argument     = broadcastAdd.MakeArgumentPointer(a_m_device_buf.GetDeviceBuffer(),
                                                     b_m_device_buf.GetDeviceBuffer(),
                                                     c_m_device_buf.GetDeviceBuffer(),
                                                     {M},
                                                     {1},
                                                     {1},
                                                     {1},
                                                     Add{});

    if(!broadcastAdd.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error("The runtime parameters seems not supported by the "
                                 "DeviceBinaryElementwise instance, exiting!");
    };

    auto broadcastAdd_invoker_ptr = broadcastAdd.MakeInvokerPointer();
    float ave_time =
        broadcastAdd_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_m_device_buf.FromDevice(c_m.mData.data());
        Tensor<CDataType> host_c_m(f_host_tensor_descriptor1d(M, 1));

        host_elementwise1D<Tensor<ABDataType>,
                           Tensor<ABDataType>,
                           Tensor<CDataType>,
                           EltwiseComputeDataType,
                           Add>(host_c_m, a_m, b_m, M, Add{});

        pass &= ck::utils::check_err(
            c_m.mData, host_c_m.mData, "Error: Incorrect results c", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
