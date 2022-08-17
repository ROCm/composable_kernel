// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include <iostream>

#include "ck/ck.hpp"
#include "ck/host_utility/io.hpp"
#include "ck/stream_config.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
using F32  = float;
using F64  = double;
using INT8  = std::int8_t;
using INT32 = std::int32_t;

template <typename ADataType, typename BDataType, typename EDataType, typename R0DataType>
void DumpGemmReduceMaxPerf(float ave_time, int M, int N, int K)
{
    std::size_t flop          = std::size_t(2) * M * N * K;
    std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(EDataType) * M * N + sizeof(R0DataType) * M;

    float tflops          = static_cast<float>(flop) / 1.E9 / ave_time;
    float gemm_gb_per_sec = gemm_num_byte / 1.E6 / ave_time;

    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gemm_gb_per_sec
              << " GB/s, " << std::endl;
}

template <typename ADataType,
          typename BDataType,
          typename EDataType,
          typename R0DataType,
          typename ALayout,
          typename BLayout,
          typename ELayout,
          typename AElementOp,
          typename BElementOp,
          typename CDEElementOp,
          typename QsElementOp,
          typename RsElementOp,
          typename RsThreadReduceOp,
          typename ReduceAccDataType,
          typename DeviceOpInstance,
          typename ReferenceGemmInstance>
auto run_gemm_reduce_max_xdl(ck::index_t M,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t StrideA,
                             ck::index_t StrideB,
                             ck::index_t StrideE,
                             bool do_verification,
                             int init_method,
                             bool time_kernel)
{
    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor(std::vector<std::size_t>({len}),
                                    std::vector<std::size_t>({stride}));
    };

    auto f_host_tensor_descriptor2d =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor2d(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor2d(K, N, StrideB, BLayout{}));
    Tensor<EDataType> e_m_n(f_host_tensor_descriptor2d(M, N, StrideE, ELayout{}));
    Tensor<R0DataType> r0_m(f_host_tensor_descriptor1d(M, 1));

    switch(init_method)
    {
    case 0: break;
    case 1:
        ck::utils::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_m_k.begin(),
                                                                             a_m_k.end());
        ck::utils::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_k_n.begin(),
                                                                             b_k_n.end());
        break;
    default:
        ck::utils::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k.begin(), a_m_k.end());
        ck::utils::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n.begin(), b_k_n.end());
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem e_device_buf(sizeof(EDataType) * e_m_n.mDesc.GetElementSpaceSize());
    DeviceMem r0_device_buf(sizeof(R0DataType) * r0_m.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op   = AElementOp{};
    auto b_element_op   = BElementOp{};
    auto cde_element_op = CDEElementOp{};
    auto qs_element_op  = QsElementOp{};
    auto rs_element_op  = RsElementOp{};

    // Prepare GEMM, max
    auto device_op = DeviceOpInstance{};
    auto invoker   = device_op.MakeInvoker();
    auto argument  = device_op.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                           b_device_buf.GetDeviceBuffer(),
                                           {},
                                           e_device_buf.GetDeviceBuffer(),
                                           {r0_device_buf.GetDeviceBuffer()},
                                           M,
                                           N,
                                           K,
                                           StrideA,
                                           StrideB,
                                           {},
                                           StrideE,
                                           a_element_op,
                                           b_element_op,
                                           cde_element_op,
                                           qs_element_op,
                                           rs_element_op);

    if(!device_op.IsSupportedArgument(argument))
    {
        throw std::runtime_error("wrong! this device_op instance does not support this problem");
    }

    // [CAUTION]: launch_and_time_kernel will not initialize D.
    // If we evaluate kernel multiple time but without initialize D. Verification will fail
    r0_device_buf.SetValue(ck::NumericLimits<R0DataType>::Lowest());

    invoker.Run(argument, StreamConfig{nullptr, false});

    bool pass = true;

    if(do_verification)
    {
        auto I0 = ck::Number<0>{};

        Tensor<EDataType> e_m_n_host(e_m_n.mDesc);
        Tensor<R0DataType> r0_m_host(r0_m.mDesc);

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, e_m_n_host, a_element_op, b_element_op, cde_element_op);

        ref_invoker.Run(ref_argument);

        auto reduce0_op = RsThreadReduceOp{}[I0];

        for(int m = 0; m < M; ++m)
        {
            auto reduce0_acc = reduce0_op.template GetIdentityValue<ReduceAccDataType>();

            for(int n = 0; n < N; ++n)
            {
                auto e_val = ck::type_convert<ReduceAccDataType>(e_m_n_host(m, n));
                reduce0_op(reduce0_acc, e_val);
            };

            r0_m_host(m) = ck::type_convert<R0DataType>(reduce0_acc);
        }

        e_device_buf.FromDevice(e_m_n.mData.data());
        r0_device_buf.FromDevice(r0_m.mData.data());

        pass = ck::utils::check_err(
            e_m_n.mData, e_m_n_host.mData, "Error: Incorrect results c", 1e-2, 1e-2);
        pass &= ck::utils::check_err(
            r0_m.mData, r0_m_host.mData, "Error: Incorrect results d0", 1e-2, 1e-2);

        if(pass)
        {
            std::cout << "Success!" << std::endl;
        }
    }

    if(time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel});
        DumpGemmReduceMaxPerf<ADataType, BDataType, EDataType, R0DataType>(ave_time, M, N, K);
    }

    return pass ? 0 : 1;
}
