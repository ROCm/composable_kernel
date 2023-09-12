#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_3d_impl.hpp"

#include "ck/library/utility/algorithm.hpp"
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
    ck::tensor_operation::device::DeviceElementwise3dImpl<ck::Tuple<ADataType>,
                                                          ck::Tuple<BDataType>,
                                                          PassThrough,
                                                          2, // NumDim_m, {N, C}
                                                          2, // NumDim_n, {H, W}
                                                          1, // NumDim_k, {D}
                                                          1,
                                                          1,
                                                          1,
                                                          ck::Sequence<1>,
                                                          ck::Sequence<1>>;

template <typename HostTensorA, typename HostTensorB, typename Functor>
void host_elementwise4D(HostTensorB& B_nchwd, const HostTensorA& A_ncdhw, Functor functor)
{
    for(std::size_t n = 0; n < A_ncdhw.mDesc.GetLengths()[0]; ++n)
        for(std::size_t c = 0; c < A_ncdhw.mDesc.GetLengths()[1]; ++c)
            for(std::size_t d = 0; d < A_ncdhw.mDesc.GetLengths()[2]; ++d)
                for(std::size_t h = 0; h < A_ncdhw.mDesc.GetLengths()[3]; ++h)
                    for(std::size_t w = 0; w < A_ncdhw.mDesc.GetLengths()[4]; ++w)
                    {
                        auto a_val = A_ncdhw(n, c, d, h, w);
                        functor(B_nchwd(n, c, h, w, d), a_val);
                    }
}

int main()
{
    bool do_verification = true;
    bool time_kernel     = true;

    const int N = 1;
    const int C = 2;
    const int H = 3;
    const int W = 4;
    const int D = 16;

    std::vector<std::size_t> ncdhw = {N, C, D, H, W};
    std::vector<std::size_t> nchwd = {N, C, H, W, D};
    Tensor<ADataType> a(ncdhw);
    Tensor<BDataType> b(nchwd);

    a.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 5> ab_lengths{N, C, H, W, D};
    std::array<ck::index_t, 5> a_strides = {C * D * H * W, D * H * W, H, 1, H * W}; // N, C, D, H, W
    std::array<ck::index_t, 5> b_strides = {C * H * W * D, H * W * D, W * D, D, 1}; // N, C, H, W, D

    auto broadcastPermute = DeviceElementwisePermuteInstance{};
    auto argument         = broadcastPermute.MakeArgumentPointer(
        ab_lengths, {a_strides}, {b_strides}, input, output, PassThrough{});

    if(!broadcastPermute.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    std::cout << "A (ncdhw): " << a.mDesc << std::endl;
    std::cout << "B (nchwd): " << b.mDesc << std::endl;

    auto broadcastPermute_invoker_ptr = broadcastPermute.MakeInvokerPointer();
    float ave_time =
        broadcastPermute_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});
    std::size_t flop = std::size_t(2) * ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4];

    std::size_t num_btype =
        sizeof(ADataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]) +
        sizeof(BDataType) * (ncdhw[0] * ncdhw[1] * ncdhw[2] * ncdhw[3] * ncdhw[4]);

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    // LogRangeAsType<float>(std::cout << "A  : ", a.mData, ",") << std::endl;
    // LogRangeAsType<float>(std::cout << "B  : ", b.mData, ",") << std::endl;
    // std::cout << "A: " << a.mData.data() << std::endl;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s"
              << std::endl;

    bool pass = true;

    if(do_verification)
    {
        b_device_buf.FromDevice(b.mData.data());

        // LogRangeAsType<float>(std::cout << "A  : ", a.mData, ",") << std::endl;
        LogRangeAsType<float>(std::cout << "B  : ", b.mData, ",") << std::endl;
        Tensor<BDataType> host_b(nchwd);
        host_elementwise4D(host_b, a, PassThrough{});
        LogRangeAsType<float>(std::cout << "Host B  : ", host_b.mData, ",") << std::endl;

        pass &=
            ck::utils::check_err(b.mData, host_b.mData, "Error: Incorrect results b", 1e-3, 1e-3);
    }

    return pass ? 0 : 1;
}
