// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

bool run_permute_bundle(const Problem& problem)
{
    constexpr std::size_t NumElemsInBundle = sizeof(BundleType) / sizeof(DataType);

    using std::begin, std::end;

    const auto& shape = problem.shape;
    ck::remove_cvref_t<decltype(shape)> transposed_shape;
    transpose_shape(problem.shape, problem.axes, begin(transposed_shape));

    Tensor<BundleType> a(shape);
    Tensor<BundleType> b(transposed_shape);

    // initialize tensor by assigning DataType values
    using std::data, std::size;
    {
        auto* const elems = reinterpret_cast<DataType*>(data(a));
        ck::utils::FillUniformDistribution<DataType>{-1.f, 1.f}(
            elems, elems + (a.GetElementSpaceSize() * NumElemsInBundle));
    }

    DeviceMem a_device_buf(a.GetElementSpaceSizeInBytes());
    DeviceMem b_device_buf(b.GetElementSpaceSizeInBytes());

    a_device_buf.ToDevice(data(a));

    std::array<ck::index_t, Problem::NumDim> a_lengths, b_lengths;
    std::array<ck::index_t, Problem::NumDim> a_strides, b_strides;

    const void* input = a_device_buf.GetDeviceBuffer();
    void* output      = b_device_buf.GetDeviceBuffer();

    std::copy(begin(shape), end(shape), begin(a_lengths));
    std::copy(begin(a.mDesc.GetStrides()), end(a.mDesc.GetStrides()), begin(a_strides));
    std::copy(begin(transposed_shape), end(transposed_shape), begin(b_lengths));
    std::copy(begin(b.mDesc.GetStrides()), end(b.mDesc.GetStrides()), begin(b_strides));

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

    // extend tensor shape from [N, H, W] to [N, H, W, NumElemsInBundle]
    const auto extended_shape = extend_shape(shape, NumElemsInBundle);
    const auto extended_axes  = extend_axes(problem.axes);

    ck::remove_cvref_t<decltype(extended_shape)> transposed_extended_shape;
    transpose_shape(extended_shape, extended_axes, begin(transposed_extended_shape));

    Tensor<DataType> extended_a(extended_shape);
    std::memcpy(data(extended_a), data(a), a.GetElementSpaceSizeInBytes());

    Tensor<DataType> extended_host_b(transposed_extended_shape);
    if(!host_permute(extended_a, extended_axes, PassThrough{}, extended_host_b))
    {
        return false;
    }

    return ck::utils::check_err(
        ck::span<const DataType>{reinterpret_cast<DataType*>(data(b)),
                                 b.GetElementSpaceSize() * NumElemsInBundle},
        ck::span<const DataType>{extended_host_b.mData},
        "Error: incorrect results in output tensor",
        1e-6,
        1e-6);
}

bool run_permute_bundle_example(const Problem::Shape& shape, const Problem::Axes& axes)
{
    return run_permute_bundle(Problem{shape, axes});
}
