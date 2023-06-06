// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <typename DeviceInstance>
int run_groupnorm_example()
{
    bool time_kernel = false;

    ck::index_t M      = 1024;
    ck::index_t N      = 1024;
    ck::index_t Stride = N;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor({len}, {stride});
    };

    auto f_host_tensor_descriptor2d = [](std::size_t row, std::size_t col, std::size_t stride) {
        using namespace ck::literals;

        return HostTensorDescriptor({row, col}, {stride, 1_uz});
    };

    Tensor<XDataType> x(f_host_tensor_descriptor2d(M, N, Stride));
    Tensor<GammaDataType> gamma(f_host_tensor_descriptor1d(N, 1));
    Tensor<BetaDataType> beta(f_host_tensor_descriptor1d(N, 1));
    Tensor<YDataType> y(f_host_tensor_descriptor2d(M, N, Stride));

    x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
    gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{0.0, 1.0});
    beta.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{0.0, 1.0});

    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
    DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpaceSize());
    DeviceMem y_dev(sizeof(YDataType) * y.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    gamma_dev.ToDevice(gamma.mData.data());
    beta_dev.ToDevice(beta.mData.data());

    auto device_instance = DeviceInstance{};
    auto argument_ptr    = device_instance.MakeArgumentPointer(
        {M, N},
        std::vector<ck::index_t>{x.mDesc.GetStrides().begin(), x.mDesc.GetStrides().end()},
        {0, 1},
        {0, 1},
        std::vector<ck::index_t>{y.mDesc.GetStrides().begin(), y.mDesc.GetStrides().end()},
        {1},
        1e-4,
        x_dev.GetDeviceBuffer(),
        gamma_dev.GetDeviceBuffer(),
        beta_dev.GetDeviceBuffer(),
        y_dev.GetDeviceBuffer(),
        nullptr,
        nullptr,
        PassThrough{});

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout << "The runtime parameters are not supported" << std::endl;
        return 1;
    };

    size_t workspace_sz = device_instance.GetWorkSpaceSize(argument_ptr.get());
    DeviceMem workspace_dev(workspace_sz);
    device_instance.SetWorkSpacePointer(argument_ptr.get(), workspace_dev.GetDeviceBuffer());

    auto invoker_ptr = device_instance.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    bool pass = true;
    {
        Tensor<YDataType> host_y(f_host_tensor_descriptor2d(M, N, Stride));
        using ReferenceInstance = ck::tensor_operation::host::ReferenceLayernorm<XDataType,
                                                                                 GammaDataType,
                                                                                 BetaDataType,
                                                                                 YDataType,
                                                                                 ComputeDataType,
                                                                                 PassThrough,
                                                                                 Rank,
                                                                                 NumReduceDim>;

        ReferenceInstance ref;
        auto ref_argument =
            ref.MakeArgument(x, gamma, beta, host_y, PassThrough{}, {M, N}, {1}, 1e-4);
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);

        y_dev.FromDevice(y.mData.data());
        pass &= ck::utils::check_err(y, host_y, "Error: Incorrect results", 1e-3, 1e-3);
    }

    return (pass ? 0 : 1);
}
