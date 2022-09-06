// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

bool run_elementwise_permute(const ExecutionConfig& config, const Problem& problem)
{
    const auto& nchw = problem.shape;
    std::vector<std::size_t> nhwc;
    transpose_shape(problem.shape, problem.axes, std::back_inserter(nhwc));

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

    auto permute = DeviceElementwisePermuteInstance{};
    auto argument =
        permute.MakeArgument(ab_lengths, {a_strides}, {b_strides}, input, output, PassThrough{});

    if(!permute.IsSupportedArgument(argument))
    {
        std::cerr << "The runtime parameters seems not supported by the device instance, exiting!"
                  << std::endl;
        return false;
    };

    auto invoker   = permute.MakeInvoker();
    float ave_time = invoker.Run(argument, StreamConfig{nullptr, config.time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    if(config.do_verification)
    {
        Tensor<BDataType> host_b(nhwc);
        host_elementwise_permute(a, problem.axes, PassThrough{}, host_b);

        b_device_buf.FromDevice(b.mData.data());

        return ck::utils::check_err(
            b.mData, host_b.mData, "Error: incorrect results in output tensor", 1e-10, 1e-10);
    }

    return true;
}

bool run_elementwise_permute_example(int argc, char* argv[])
{
    ExecutionConfig config;
    Problem problem;

    return parse_cmd_args(argc, argv, config, problem) && run_elementwise_permute(config, problem);
}
