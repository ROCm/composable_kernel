// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

bool run_permute(const ExecutionConfig& config, const Problem& problem)
{
    using std::begin, std::end;

    const auto& shape = problem.shape;
    ck::remove_cvref_t<decltype(shape)> transposed_shape;
    transpose_shape(problem.shape, problem.axes, begin(transposed_shape));

    Tensor<ADataType> a(shape);
    Tensor<BDataType> b(transposed_shape);

    std::iota(begin(a.mData), end(a.mData), 0);

    DeviceMem a_device_buf(sizeof(ADataType) * a.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a.mData.data());

    std::array<const void*, 1> input = {a_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {b_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 4> ab_lengths;
    std::array<ck::index_t, 4> a_strides, b_strides;

    std::copy(begin(shape), end(shape), begin(ab_lengths));
    std::copy(begin(a.mDesc.GetStrides()), end(a.mDesc.GetStrides()), begin(a_strides));
    std::copy(begin(b.mDesc.GetStrides()), end(b.mDesc.GetStrides()), begin(b_strides));

    static_assert(std::is_default_constructible_v<DevicePermuteInstance>);
    static_assert(detail::is_device_op_v<DevicePermuteInstance>);

    auto permute = DevicePermuteInstance{};
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
        Tensor<BDataType> host_b(transposed_shape);
        host_permute(a, problem.axes, PassThrough{}, host_b);

        b_device_buf.FromDevice(b.mData.data());

        return ck::utils::check_err(
            b.mData, host_b.mData, "Error: incorrect results in output tensor", 1e-10, 1e-10);
    }

    return true;
}

bool run_permute_example(int argc, char* argv[])
{
    ExecutionConfig config;
    Problem problem;

    return parse_cmd_args(argc, argv, config, problem) && run_permute(config, problem);
}
