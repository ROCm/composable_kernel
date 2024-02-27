// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <getopt.h>

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_normalization_bwd_gamma_beta_impl.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_layernorm_bwd.hpp"

using DYDataType         = ck::half_t;
using XDataType          = ck::half_t;
using GammaDataType      = ck::half_t;
using MeanInvStdDataType = float;
using DGammaDataType     = ck::half_t;
using DBetaDataType      = ck::half_t;
using DXDataType         = ck::half_t;
using ComputeDataType    = float;

constexpr int Rank         = 2;
constexpr int NumReduceDim = 1;

// Layernorm:

// Input shape
// dy:        [M, N]
// x:         [M, N]
// mean:      [M, 1]
// inv_std:   [M, 1]

// Output shape
// dgamma: [1, N]
// dbeta:  [1, N]

// dgamma = reduce_sum(dy * (x - mean) * inv_std, axis=0)
// dbeta = reduce_sum(dy, axis=0)

// [CAUSION]
// In DeviceNormalizationBwdGammaBetaImpl, M is invarient dimension, K is reduced dimension
// Hence, M in this example and DeviceNormalizationBwdGammaBetaImpl is different
using GammaBetaDeviceInstance = ck::tensor_operation::device::DeviceNormalizationBwdGammaBetaImpl<
    DYDataType,
    XDataType,
    MeanInvStdDataType,
    ComputeDataType,
    DGammaDataType,
    DBetaDataType,
    Rank,
    NumReduceDim,
    256,   // BlockSize
    8,     // ClusterInvarient
    32,    // ClusterReduce
    8,     // SliceInvarient
    1,     // SliceReduce
    false, // IsDYFastestDimReduced
    8,     // DYSrcVectorSize
    false, // IsXFastestDimReduced
    8,     // XSrcVectorSize
    true,  // IsMeanInvStdFastestDimReduced
    1,     // MeanInvStdSrcVectorSize
    1,     // DGammaDstVectorSize
    1>;    // DBetaDstVectorSize

int main()
{
    bool time_kernel = false;

    ck::index_t M = 1024;
    ck::index_t N = 512;

    Tensor<DYDataType> dy({M, N});
    Tensor<XDataType> x({M, N});
    Tensor<GammaDataType> gamma({N});
    Tensor<MeanInvStdDataType> mean({M});
    Tensor<MeanInvStdDataType> inv_std({M});

    Tensor<DGammaDataType> dgamma({N});
    Tensor<DBetaDataType> dbeta({N});
    Tensor<DXDataType> dx({M, N});

    dy.GenerateTensorValue(GeneratorTensor_3<DYDataType>{0.0, 1.0});
    x.GenerateTensorValue(GeneratorTensor_3<XDataType>{0.0, 1.0});
    gamma.GenerateTensorValue(GeneratorTensor_3<GammaDataType>{0.0, 1.0});
    mean.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{0.0, 1.0});
    inv_std.GenerateTensorValue(GeneratorTensor_3<MeanInvStdDataType>{0.0, 1.0});

    DeviceMem dy_dev(sizeof(DYDataType) * dy.mDesc.GetElementSpaceSize());
    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem mean_dev(sizeof(MeanInvStdDataType) * mean.mDesc.GetElementSpaceSize());
    DeviceMem inv_std_dev(sizeof(MeanInvStdDataType) * inv_std.mDesc.GetElementSpaceSize());
    DeviceMem dgamma_dev(sizeof(DGammaDataType) * dgamma.mDesc.GetElementSpaceSize());
    DeviceMem dbeta_dev(sizeof(DBetaDataType) * dbeta.mDesc.GetElementSpaceSize());

    dy_dev.ToDevice(dy.mData.data());
    x_dev.ToDevice(x.mData.data());
    mean_dev.ToDevice(mean.mData.data());
    inv_std_dev.ToDevice(inv_std.mData.data());

    auto gamma_beta_device_instance = GammaBetaDeviceInstance{};
    auto gamma_beta_argument_ptr =
        gamma_beta_device_instance.MakeArgumentPointer({M, N}, // inLengths
                                                       {N, 1}, // dyStrides
                                                       {N, 1}, // xStrides
                                                       {1, 0}, // meanStrides
                                                       {1, 0}, // invStdStrides
                                                       {N},    // outLengths
                                                       {1},    // dgammaStrides
                                                       {1},    // dbetaStrides
                                                       {0},    // reduceDims
                                                       dy_dev.GetDeviceBuffer(),
                                                       x_dev.GetDeviceBuffer(),
                                                       mean_dev.GetDeviceBuffer(),
                                                       inv_std_dev.GetDeviceBuffer(),
                                                       dgamma_dev.GetDeviceBuffer(),
                                                       dbeta_dev.GetDeviceBuffer());

    if(!gamma_beta_device_instance.IsSupportedArgument(gamma_beta_argument_ptr.get()))
    {
        std::cout << "The runtime parameters are not supported" << std::endl;
        return 1;
    };

    auto gamma_beta_invoker_ptr = gamma_beta_device_instance.MakeInvokerPointer();
    gamma_beta_invoker_ptr->Run(gamma_beta_argument_ptr.get(), StreamConfig{nullptr, time_kernel});

    bool pass = true;
    {
        Tensor<DGammaDataType> host_dgamma({N});
        Tensor<DBetaDataType> host_dbeta({N});
        Tensor<DXDataType> host_dx({M, N});
        using ReferenceInstance =
            ck::tensor_operation::host::ReferenceLayernormBwd<DYDataType,
                                                              XDataType,
                                                              GammaDataType,
                                                              MeanInvStdDataType,
                                                              DGammaDataType,
                                                              DBetaDataType,
                                                              DXDataType,
                                                              ComputeDataType>;

        ReferenceInstance ref;
        auto ref_argument =
            ref.MakeArgument(dy, x, gamma, mean, inv_std, host_dgamma, host_dbeta, host_dx, {M, N});
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);

        dgamma_dev.FromDevice(dgamma.mData.data());
        dbeta_dev.FromDevice(dbeta.mData.data());

        pass &= ck::utils::check_err(dgamma, host_dgamma, "Error: Incorrect dgamma", 1e-3, 1e-3);
        pass &= ck::utils::check_err(dbeta, host_dbeta, "Error: Incorrect dbeta", 1e-3, 1e-3);
    }

    return (pass ? 0 : 1);
}
