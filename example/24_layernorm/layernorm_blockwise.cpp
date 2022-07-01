// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/device_layernorm.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/host_tensor/device_memory.hpp"
#include "ck/library/host_tensor/host_common_util.hpp"
#include "ck/library/host_tensor/host_tensor.hpp"
#include "ck/library/host_tensor/host_tensor_generator.hpp"

using XDataType     = ck::half_t;
using GammaDataType = ck::half_t;
using BetaDataType  = ck::half_t;
using YDataType     = ck::half_t;
using AccDataType   = float;

constexpr int Rank         = 2;
constexpr int NumReduceDim = 1;

using DeviceInstance = ck::tensor_operation::device::DeviceLayernorm<XDataType,
                                                                     GammaDataType,
                                                                     BetaDataType,
                                                                     AccDataType,
                                                                     YDataType,
                                                                     Rank,
                                                                     NumReduceDim,
                                                                     256, // BlockSize
                                                                     8,   // ClusterM
                                                                     32,  // ClusterK
                                                                     1,   // SliceM
                                                                     8,   // SliceK
                                                                     1,   // SrcVecDim (0=M, 1=K)
                                                                     8,   // SrcScalarPerVector
                                                                     1,   // GammaVecDim (0=M, 1=K)
                                                                     8,   // GammaScalarPerVector
                                                                     1,   // BetaVecDim (0=M, 1=K)
                                                                     8,   // BetaScalarPerVector
                                                                     1>;  // OutScalarPerVector

template <typename XDataType,
          typename GammaDataType,
          typename BetaDataType,
          typename YDataType,
          typename AccDataType>
void host_layernorm2d(const Tensor<XDataType>& x_m_n,
                      const Tensor<GammaDataType>& gamma_n,
                      const Tensor<BetaDataType>& beta_n,
                      Tensor<YDataType>& y_m_n,
                      int M,
                      int N,
                      AccDataType epislon = 1e-4)
{
    Tensor<AccDataType> mean({M});
    Tensor<AccDataType> var({M});

    for(int m = 0; m < M; ++m)
    {
        mean(m) = 0;
        var(m)  = 0;

        for(int n = 0; n < N; ++n)
        {
            auto x_val = ck::type_convert<AccDataType>(x_m_n(m, n));
            mean(m) += x_val;
            var(m) += x_val * x_val;
        }

        mean(m) = mean(m) / N;
        var(m)  = (var(m) / N) - (mean(m) * mean(m));
    }

    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            auto x_val  = ck::type_convert<AccDataType>(x_m_n(m, n));
            auto y_val  = (x_val - mean(m)) / sqrt(var(m) + epislon);
            y_val       = (y_val * gamma_n(n)) + beta_n(n);
            y_m_n(m, n) = ck::type_convert<YDataType>(y_val);
        }
    }
}

int main()
{
    bool time_kernel = false;

    ck::index_t M      = 1024;
    ck::index_t N      = 1024;
    ck::index_t Stride = 1024;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor(std::vector<std::size_t>({len}),
                                    std::vector<std::size_t>({stride}));
    };

    auto f_host_tensor_descriptor2d = [](std::size_t row, std::size_t col, std::size_t stride) {
        return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                    std::vector<std::size_t>({stride, 1}));
    };

    Tensor<XDataType> x(f_host_tensor_descriptor2d(M, N, Stride));
    Tensor<GammaDataType> gamma(f_host_tensor_descriptor1d(N, 1));
    Tensor<BetaDataType> beta(f_host_tensor_descriptor1d(N, 1));
    Tensor<YDataType> y(f_host_tensor_descriptor2d(M, N, Stride));

    x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
    gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{0.0, 1.0});
    beta.GenerateTensorValue(GeneratorTensor_3<BetaDataType>{0.0, 1.0});

    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpace());
    DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpace());
    DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpace());
    DeviceMem y_dev(sizeof(YDataType) * y.mDesc.GetElementSpace());

    x_dev.ToDevice(x.mData.data());
    gamma_dev.ToDevice(gamma.mData.data());
    beta_dev.ToDevice(beta.mData.data());

    auto device_instance = DeviceInstance{};
    auto argument_ptr    = device_instance.MakeArgumentPointer({M, N},
                                                            {Stride, 1},
                                                            {0, 1},
                                                            {0, 1},
                                                            {1},
                                                            1e-4,
                                                            x_dev.GetDeviceBuffer(),
                                                            gamma_dev.GetDeviceBuffer(),
                                                            beta_dev.GetDeviceBuffer(),
                                                            y_dev.GetDeviceBuffer());

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout << "The runtime parameters are not supported" << std::endl;
        return 1;
    };

    auto invoker_ptr = device_instance.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    bool pass = true;
    {
        Tensor<YDataType> host_y(f_host_tensor_descriptor2d(M, N, Stride));
        host_layernorm2d<XDataType, GammaDataType, BetaDataType, YDataType, AccDataType>(
            x, gamma, beta, host_y, M, N);

        y_dev.FromDevice(y.mData.data());
        pass &=
            ck::utils::check_err(y.mData, host_y.mData, "Error: Incorrect results d1", 1e-3, 1e-3);
    }
    return (pass ? 0 : 1);
}
