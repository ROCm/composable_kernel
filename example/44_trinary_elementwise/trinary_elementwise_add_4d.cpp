#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/trinary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ABCDataType = F16;
using DDataType   = F16;

using TriAdd = ck::tensor_operation::element_wise::TriAdd;

using DeviceElementwiseAddInstance = ck::tensor_operation::device::DeviceElementwise<
    ck::Tuple<ABCDataType, ABCDataType, ABCDataType>,
    ck::Tuple<DDataType>,
    TriAdd,
    4,
    8,
    ck::Sequence<8, 8, 8>,
    ck::Sequence<8>>;

template <typename HostTensorA,
          typename HostTensorB,
          typename HostTensorC,
          typename HostTensorD,
          typename Functor>
void host_elementwise4D(HostTensorD& D,
                        const HostTensorC& C,
                        const HostTensorA& A,
                        const HostTensorB& B,
                        const std::vector<std::size_t>& shape,
                        Functor functor)
{
    using dtype = ck::remove_reference_t<decltype(D(0, 0, 0, 0))>;

    for(std::size_t n = 0; n < shape[0]; ++n)
        for(std::size_t c = 0; c < shape[1]; ++c)
            for(std::size_t h = 0; h < shape[2]; ++h)
                for(std::size_t w = 0; w < shape[3]; ++w)
                {
                    auto a_val  = A(n, c, h, w);
                    auto b_val  = B(n, c, h, w);
                    auto c_val  = C(n, c, h, w);
                    dtype d_val = 0;
                    functor(d_val, c_val, a_val, b_val);
                    D(n, c, h, w) = d_val;
                }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = true;

    std::vector<std::size_t> nchw = {4, 16, 32, 32};

    Tensor<ABCDataType> a(nchw);
    Tensor<ABCDataType> b(nchw);
    Tensor<ABCDataType> c(nchw);
    Tensor<DDataType> d(nchw);

    a.GenerateTensorValue(GeneratorTensor_3<ABCDataType>{0.0, 1.0});
    b.GenerateTensorValue(GeneratorTensor_3<ABCDataType>{0.0, 1.0});
    c.GenerateTensorValue(GeneratorTensor_3<ABCDataType>{0.0, 1.0});

    DeviceMem a_device_buf(sizeof(ABCDataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(ABCDataType) * b.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(ABCDataType) * c.mDesc.GetElementSpaceSize());
    DeviceMem d_device_buf(sizeof(DDataType) * d.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());
    b_device_buf.ToDevice(b.mData.data());
    c_device_buf.ToDevice(c.mData.data());

    std::array<const void*, 3> input = {a_device_buf.GetDeviceBuffer(),
                                        b_device_buf.GetDeviceBuffer(),
                                        c_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {d_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 4> abcd_lengths;
    std::array<ck::index_t, 4> a_strides;
    std::array<ck::index_t, 4> b_strides;
    std::array<ck::index_t, 4> c_strides;
    std::array<ck::index_t, 4> d_strides;

    std::copy(nchw.begin(), nchw.end(), abcd_lengths.begin());
    std::copy(a.mDesc.GetStrides().begin(), a.mDesc.GetStrides().end(), a_strides.begin());
    std::copy(b.mDesc.GetStrides().begin(), b.mDesc.GetStrides().end(), b_strides.begin());
    std::copy(c.mDesc.GetStrides().begin(), c.mDesc.GetStrides().end(), c_strides.begin());
    std::copy(d.mDesc.GetStrides().begin(), d.mDesc.GetStrides().end(), d_strides.begin());

    auto broadcastAdd = DeviceElementwiseAddInstance{};
    auto argument     = broadcastAdd.MakeArgumentPointer(
        abcd_lengths, {a_strides, b_strides, c_strides}, {d_strides}, input, output, TriAdd{});

    if(!broadcastAdd.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    auto broadcastAdd_invoker_ptr = broadcastAdd.MakeInvokerPointer();
    float ave_time =
        broadcastAdd_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        d_device_buf.FromDevice(d.mData.data());
        Tensor<DDataType> host_d(nchw);

        host_elementwise4D<Tensor<ABCDataType>,
                           Tensor<ABCDataType>,
                           Tensor<ABCDataType>,
                           Tensor<DDataType>,
                           TriAdd>(host_d, a, b, c, nchw, TriAdd{});

        pass &=
            ck::utils::check_err(d.mData, host_d.mData, "Error: Incorrect results d", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
