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

#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_common_util.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/ranges.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_groupnorm.hpp"

constexpr int Rank         = 5;
constexpr int NumReduceDim = 3;

using XDataType     = ck::half_t;
using GammaDataType = ck::half_t;
using BetaDataType  = ck::half_t;
using YDataType     = ck::half_t;
using AccDataType   = float;

struct YElementOp
{
    template <typename T>
    __host__ __device__ void operator()(T& y, const T& x) const
    {
        static_assert(ck::is_same<T, float>::value || ck::is_same<T, double>::value ||
                          ck::is_same<T, ck::half_t>::value,
                      "Data type is not supported by this operation!");

        T a;

        ck::tensor_operation::element_wise::Sigmoid{}(a, x);

        y = x * a;
    };
};

using DeviceInstance =
    ck::tensor_operation::device::DeviceLayernormImpl<XDataType,
                                                      GammaDataType,
                                                      BetaDataType,
                                                      AccDataType,
                                                      YDataType,
                                                      YElementOp,
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

int main(int argc, char* argv[])
{
    ck::index_t N = 128;
    ck::index_t H = 16;
    ck::index_t W = 16;
    ck::index_t G = 32;
    ck::index_t C = 40;

    if(argc == 1)
    {
        // use default case
    }
    else if(argc == 6)
    {
        N = std::stoi(argv[1]);
        H = std::stoi(argv[2]);
        W = std::stoi(argv[3]);
        G = std::stoi(argv[4]);
        C = std::stoi(argv[5]);
    }
    else
    {
        std::cerr << "arg1 to 5: N, H, W, G, C" << std::endl;

        return 1;
    }

    Tensor<XDataType> x({N, H, W, G, C});
    Tensor<YDataType> y({N, H, W, G, C});
    Tensor<GammaDataType> gamma({G, C});
    Tensor<BetaDataType> beta({G, C});

    ck::utils::FillUniformDistribution<XDataType>{0.f, 1.f}(x.begin(), x.end());
    ck::utils::FillUniformDistribution<GammaDataType>{0.f, 1.f}(gamma.begin(), gamma.end());
    ck::utils::FillUniformDistribution<BetaDataType>{0.f, 1.f}(beta.begin(), beta.end());

    DeviceMem x_dev(sizeof(XDataType) * x.mDesc.GetElementSpaceSize());
    DeviceMem gamma_dev(sizeof(GammaDataType) * gamma.mDesc.GetElementSpaceSize());
    DeviceMem beta_dev(sizeof(BetaDataType) * beta.mDesc.GetElementSpaceSize());
    DeviceMem y_dev(sizeof(YDataType) * y.mDesc.GetElementSpaceSize());

    x_dev.ToDevice(x.mData.data());
    gamma_dev.ToDevice(gamma.mData.data());
    beta_dev.ToDevice(beta.mData.data());

    const auto y_element_op = YElementOp{};

    using ck::ranges::to;

    auto device_instance = DeviceInstance{};
    auto argument_ptr =
        device_instance.MakeArgumentPointer({N, H, W, G, C},
                                            to<std::vector<ck::index_t>>(x.GetStrides()),
                                            {0, 0, 0, C, 1},
                                            {0, 0, 0, C, 1},
                                            to<std::vector<ck::index_t>>(y.GetStrides()),
                                            {1, 2, 4}, // reduction dimension: [H, W, C]
                                            1e-6,
                                            x_dev.GetDeviceBuffer(),
                                            gamma_dev.GetDeviceBuffer(),
                                            beta_dev.GetDeviceBuffer(),
                                            y_dev.GetDeviceBuffer(),
                                            y_element_op);

    if(!device_instance.IsSupportedArgument(argument_ptr.get()))
    {
        std::cout << "The runtime parameters are not supported" << std::endl;
        return 1;
    };

    auto invoker_ptr = device_instance.MakeInvokerPointer();
    float ave_time   = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, true, true});

    std::size_t num_btype = sizeof(XDataType) * N * H * W * G * C +
                            sizeof(YDataType) * N * H * W * G * C + sizeof(GammaDataType) * G * C +
                            sizeof(BetaDataType) * G * C;

    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << gb_per_sec << " GB/s, "
              << device_instance.GetTypeString() << std::endl;

    bool pass = true;
    {
        Tensor<YDataType> host_y({N, H, W, G, C});
        using ReferenceInstance = ck::tensor_operation::host::ReferenceGroupnorm<XDataType,
                                                                                 GammaDataType,
                                                                                 BetaDataType,
                                                                                 YDataType,
                                                                                 AccDataType,
                                                                                 YElementOp>;

        ReferenceInstance ref;
        auto ref_argument =
            ref.MakeArgument(x, gamma, beta, host_y, y_element_op, {N, H, W, G, C}, 1e-6);
        auto ref_invoker = ref.MakeInvoker();
        ref_invoker.Run(ref_argument);

        y_dev.FromDevice(y.mData.data());
        pass &= ck::utils::check_err(y, host_y, "Error: Incorrect results", 1e-3, 1e-3);
    }

    return (pass ? 0 : 1);
}
