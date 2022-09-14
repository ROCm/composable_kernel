// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

bool run_permute_element(const Problem& problem)
{
    using std::begin, std::end;

    const auto& shape = problem.shape;
    ck::remove_cvref_t<decltype(shape)> transposed_shape;
    transpose_shape(problem.shape, problem.axes, begin(transposed_shape));

    Tensor<ADataType> a(shape);
    Tensor<BDataType> b(transposed_shape);

    ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a);

    DeviceMem a_device_buf(a.GetElementSpaceSizeInBytes());
    DeviceMem b_device_buf(b.GetElementSpaceSizeInBytes());

    using std::data;
    a_device_buf.ToDevice(data(a));

    std::array<ck::index_t, Problem::NumDim> a_lengths, b_lengths;
    std::array<ck::index_t, Problem::NumDim> a_strides, b_strides;

    const void* input = a_device_buf.GetDeviceBuffer();
    void* output      = b_device_buf.GetDeviceBuffer();

    std::copy(begin(shape), end(shape), begin(a_lengths));
    std::copy(begin(a.GetStrides()), end(a.GetStrides()), begin(a_strides));
    std::copy(begin(transposed_shape), end(transposed_shape), begin(b_lengths));
    std::copy(begin(b.GetStrides()), end(b.GetStrides()), begin(b_strides));

    static_assert(std::is_default_constructible_v<DevicePermuteInstance>);

    auto permute  = DevicePermuteInstance{};
    auto argument = permute.MakeArgument(
        a_lengths, a_strides, b_lengths, b_strides, input, output, PassThrough{});

    if(!permute.IsSupportedArgument(argument))
    {
        std::cerr << "The runtime parameters seems not supported by the device instance, exiting!"
                  << std::endl;
        return false;
    };

    auto invoker   = permute.MakeInvoker();
    float ave_time = invoker.Run(argument, StreamConfig{nullptr, true});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    b_device_buf.FromDevice(data(b));

    Tensor<BDataType> host_b(transposed_shape);
    if(!host_permute(a, problem.axes, PassThrough{}, host_b))
    {
        return false;
    }

    return ck::utils::check_err(
        b.mData, host_b.mData, "Error: incorrect results in output tensor", 1e-6, 1e-6);
}

bool run_permute_element_example(const Problem::Shape& shape, const Problem::Axes& axes)
{
    return run_permute_element(Problem{shape, axes});
}
