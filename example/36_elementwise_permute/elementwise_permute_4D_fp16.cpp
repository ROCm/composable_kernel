#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/device_elementwise.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using F16 = ck::half_t;
using F32 = float;

using ADataType = F16;
using BDataType = F16;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using DeviceElementwisePermuteInstance =
    ck::tensor_operation::device::DeviceElementwise<ck::Tuple<ADataType>,
                                                    ck::Tuple<BDataType>,
                                                    PassThrough,
                                                    4,
                                                    8,
                                                    ck::Sequence<8>,
                                                    ck::Sequence<1>>;

template <typename HostTensorA, typename HostTensorB, typename Functor>
void host_elementwise4D(HostTensorB& B_nhwc,
                        const HostTensorA& A_nchw,
                        const std::vector<std::size_t>& shape_nchw,
                        Functor functor)
{
    using btype = ck::remove_reference_t<decltype(B(0, 0, 0, 0))>;
    for(std::size_t n = 0; n < shape_nchw[0]; ++n)
        for(std::size_t c = 0; c < shape_nchw[1]; ++c)
            for(std::size_t h = 0; h < shape_nchw[2]; ++h)
                for(std::size_t w = 0; w < shape_nchw[3]; ++w)
                {
                    auto a_val = A(n, c, h, w);
                    functor(B(n, h, w, c), a_val);
                }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    std::vector<std::size_t> nchw = {4, 16, 32, 32};
    Tensor<ADataType> a(nchw);
    Tensor<BDataType> b(nhwc);

    a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 4> ab_lengths;
    std::array<ck::index_t, 4> a_strides;
    std::array<ck::index_t, 4> b_strides;

    std::copy(nchw.begin(), nchw.end(), ab_lengths.begin());
    std::copy(a.mDesc.GetStrides().begin(), a.mDesc.GetStrides().end(), a_strides.begin());
    std::copy(b.mDesc.GetStrides().begin(), b.mDesc.GetStrides().end(), b_strides.begin());

    auto broadcastPermute = DeviceElementwisePermuteInstance{};
    auto argument         = broadcastPermute.MakeArgumentPointer(
        ab_lengths, {a_strides}, {c_strides}, input, output, PassThrough{});

    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };
    auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time =
        broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;

    if(do_verification)
    {
        b_device_buf.FromDevice(b.mData.data());
        Tensor<BDataType> host_b(nchw);
        host_elementwise4D<Tensor<BDataType>, Tensor<ADataType>, PassThrough>(
            host_b, a, nchw, PassThrough{});
        pass &=
            ck::utils::check_err(b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
