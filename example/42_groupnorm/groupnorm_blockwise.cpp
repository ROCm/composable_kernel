// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/ck.hpp"
#include "ck/utility/reduction_enums.hpp"
#include "ck/tensor_operation/gpu/device/device_layernorm_impl.hpp"
#include "ck/tensor_operation/gpu/device/reduction_operator_mapping.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_groupnorm.hpp"

using XDataType     = ck::half_t;
using GammaDataType = ck::half_t;
using BetaDataType  = ck::half_t;
using YDataType     = ck::half_t;
using AccDataType   = float;
using PassThrough   = ck::tensor_operation::element_wise::PassThrough;

constexpr int Rank         = 5;
constexpr int NumReduceDim = 3;

using DeviceInstance =
    ck::tensor_operation::device::DeviceLayernormImpl<XDataType,
                                                      GammaDataType,
                                                      BetaDataType,
                                                      AccDataType,
                                                      YDataType,
                                                      PassThrough,
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
                                                      8>;  // OutScalarPerVector


int main()
{
    ck::index_t N = 1;
    ck::index_t H = 16;
    ck::index_t W = 16;
    ck::index_t G = 32;
    ck::index_t C = 40;

    Tensor<XDataType> x({N, H, W, G, C});
    Tensor<YDataType> y({N, H, W, G, C});
    Tensor<GammaDataType> gamma({G, C});
    Tensor<BetaDataType> beta({G, C});

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
        {N, H, W, G, C},
        std::vector<ck::index_t>{x.mDesc.GetStrides().begin(), x.mDesc.GetStrides().end()},
        {0, 0, 0, C, 1},
        {0, 0, 0, C, 1},
        std::vector<ck::index_t>{y.mDesc.GetStrides().begin(), y.mDesc.GetStrides().end()},
        {1, 2, 4}, // [H, W, C]
        1e-6,
        x_dev.GetDeviceBuffer(),
        gamma_dev.GetDeviceBuffer(),
        beta_dev.GetDeviceBuffer(),
        y_dev.GetDeviceBuffer(),
        PassThrough{});

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout << "The runtime parameters are not supported" << std::endl;
        return 1;
    };

    bool time_kernel = false;
    auto invoker_ptr = device_instance.MakeInvokerPointer();
    invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    bool pass = true;
    {
        Tensor<YDataType> host_y({N, H, W, G, C});
        using ReferenceInstance = ck::tensor_operation::host::ReferenceGroupnorm<XDataType,
                                                                                 GammaDataType,
                                                                                 BetaDataType,
                                                                                 YDataType,
                                                                                 AccDataType,
                                                                                 PassThrough>;

        ReferenceInstance ref;
        auto ref_argument =
            ref.MakeArgument(x, gamma, beta, host_y, PassThrough{}, {N, H, W, G, C}, 1e-6);
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);

        y_dev.FromDevice(y.mData.data());
        pass &=
            ck::utils::check_err(y.mData, host_y.mData, "Error: Incorrect results d1", 1e-3, 1e-3);
    }
    return (pass ? 0 : 1);
}
