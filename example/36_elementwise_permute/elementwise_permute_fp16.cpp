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
void host_elementwise4D(HostTensorB& B,
                        const HostTensorA& A,
                        const std::vector<std::size_t>& shape,
                        Functor functor)
{
    using btype = ck::remove_reference_t<decltype(B(0, 0, 0, 0))>;
    for(std::size_t n = 0; n < shape[0]; ++n)
        for(std::size_t c = 0; c < shape[1]; ++c)
            for(std::size_t h = 0; h < shape[2]; ++h)
                for(std::size_t w = 0; w < shape[3]; ++w)
                {
                    auto a_val  = A(n, c, h, w);
                    btype b_val = 0;
                    functor(b_val, a_val);
                    B(n, h, w, c) = b_val;
                }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = false;

    std::size_t N = 4, C = 16, H = 32, W = 32;
    std::vector<std::size_t> nchw = {N, C, H, W};
    std::vector<std::size_t> nhwc = {N, H, W, C};
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
        ab_lengths, {a_strides}, {b_strides}, input, output, PassThrough{});

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
        Tensor<BDataType> host_b(nhwc);
        host_elementwise4D<Tensor<ADataType>, Tensor<BDataType>, PassThrough>(
            host_b, a, nhwc, PassThrough{});
        pass &=
            ck::utils::check_err(b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
