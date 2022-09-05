// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

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

bool run_elementwise_permute(const ExecutionConfig& config, const Problem& problem)
{
    std::size_t N = 4, C = 16, H = 32, W = 32;
    std::vector<std::size_t> nchw = {N, C, H, W};
    std::vector<std::size_t> nhwc = {N, H, W, C};
    Tensor<ADataType> a(nchw);
    Tensor<BDataType> b(nhwc);

    std::iota(begin(a.mData), end(a.mData), 0);

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
    float ave_time                    = broadcastPermute_invoker_ptr->Run(argument.get(),
                                                       StreamConfig{nullptr, config.time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    if(config.do_verification)
    {
        b_device_buf.FromDevice(b.mData.data());
        Tensor<BDataType> host_b(nhwc);
        host_elementwise4D<Tensor<ADataType>, Tensor<BDataType>, PassThrough>(
            host_b, a, nhwc, PassThrough{});

        return ck::utils::check_err(
            b.mData, host_b.mData, "Error: incorrect results in tensor B", 1e-10, 1e-10);
    }

    return true;
}

bool run_elementwise_permute_example(int argc, char* argv[])
{
    ExecutionConfig config;
    Problem problem;

    return parse_cmd_args(argc, argv, config, problem) && run_elementwise_permute(config, problem);
}
