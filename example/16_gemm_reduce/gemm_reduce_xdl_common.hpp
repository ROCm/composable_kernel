// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16   = ck::half_t;
using BF16  = ck::bhalf_t;
using F32   = float;
using F64   = double;
using INT4  = ck::int4_t;
using INT8  = std::int8_t;
using INT32 = std::int32_t;

template <typename ADataType, typename BDataType, typename CDataType, typename ReduceDataType>
void DumpGemmReduceMaxPerf(float gemm_reduce_time, int M, int N, int K)
{
    std::size_t gemm_flop     = std::size_t(2) * M * N * K;
    std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(CDataType) * M * N + sizeof(ReduceDataType) * M;

    float tflops          = static_cast<float>(gemm_flop) / 1.E9 / gemm_reduce_time;
    float gemm_gb_per_sec = gemm_num_byte / 1.E6 / gemm_reduce_time;

    std::cout << "gemm + reduceMax Perf: " << gemm_reduce_time << " ms, " << tflops << " TFlops, "
              << gemm_gb_per_sec << " GB/s, " << std::endl;
}

template <typename ADataType, typename BDataType, typename CDataType, typename ReduceDataType>
void DumpGemmReduceMeanSquareMeanPerf(float gemm_reduce_time, int M, int N, int K)
{
    std::size_t gemm_flop     = std::size_t(2) * M * N * K;
    std::size_t gemm_num_byte = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                sizeof(CDataType) * M * N + sizeof(ReduceDataType) * M +
                                sizeof(ReduceDataType) * M;

    float tflops          = static_cast<float>(gemm_flop) / 1.E9 / gemm_reduce_time;
    float gemm_gb_per_sec = gemm_num_byte / 1.E6 / gemm_reduce_time;

    std::cout << "gemm + reduce_mean + reduce_mean_square Perf: " << gemm_reduce_time << " ms, "
              << tflops << " TFlops, " << gemm_gb_per_sec << " GB/s, " << std::endl;
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ReduceDataType,
          typename ReduceAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ReduceOps,
          typename ReduceElementOps,
          typename DeviceGemmReduceInstance,
          typename ReferenceGemmInstance>
auto run_gemm_reduce_max_xdl(ck::index_t M,
                             ck::index_t N,
                             ck::index_t K,
                             ck::index_t StrideA,
                             ck::index_t StrideB,
                             ck::index_t StrideC,
                             bool do_verification,
                             int init_method,
                             bool time_kernel)
{
    auto f_host_tensor_descriptor =
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

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> reduce_m_host_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));

    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> reduce_m_device_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "reduce_m: " << reduce_m_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem reduce_device_buf(sizeof(ReduceDataType) *
                                reduce_m_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op                       = AElementwiseOperation{};
    auto b_element_op                       = BElementwiseOperation{};
    auto c_element_op                       = CElementwiseOperation{};
    auto reduce_element_op                  = ReduceElementOps{}[ck::Number<0>{}];
    std::array<void*, 3> gemm_element_ops   = {&a_element_op, &b_element_op, &c_element_op};
    std::array<void*, 1> reduce_element_ops = {&reduce_element_op};
    std::array<void*, 1> p_reduces          = {reduce_device_buf.GetDeviceBuffer()};

    // do GEMM
    auto gemm     = DeviceGemmReduceInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                      b_device_buf.GetDeviceBuffer(),
                                      nullptr,
                                      {},
                                      c_device_buf.GetDeviceBuffer(),
                                      p_reduces,
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      {},
                                      gemm_element_ops,
                                      {},
                                      reduce_element_ops,
                                      reduce_element_ops);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    // [CAUTION]: launch_and_time_kernel will not initialize D.
    // If we evaluate kernel multiple time but without initialize D. Verification will fail
    reduce_device_buf.SetValue(ck::NumericLimits<ReduceDataType>::Lowest());
    invoker.Run(argument, StreamConfig{nullptr, false});

    bool pass = true;

    if(do_verification)
    {
        c_device_buf.FromDevice(c_m_n_device_result.mData.data());
        reduce_device_buf.FromDevice(reduce_m_device_result.mData.data());

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        auto reduce_op = ReduceOps{}[ck::Number<0>{}];

        for(int m = 0; m < M; ++m)
        {
            ReduceAccDataType reduce_acc = reduce_op.template GetIdentityValue<ReduceAccDataType>();

            for(int n = 0; n < N; ++n)
            {
                ReduceAccDataType curr_val =
                    ck::type_convert<ReduceAccDataType>(c_m_n_host_result(m, n));
                reduce_op(reduce_acc, curr_val);
            };

            reduce_m_host_result(m) = reduce_acc;
        }

        pass = ck::utils::check_err(c_m_n_device_result.mData,
                                    c_m_n_host_result.mData,
                                    "Error: Incorrect results c") &&
               ck::utils::check_err(reduce_m_device_result.mData,
                                    reduce_m_host_result.mData,
                                    "Error: Incorrect results d",
                                    1e-3,
                                    1e-3);
    }

    if(time_kernel)
    {
        float gemm_reduce_ave_time = invoker.Run(argument, StreamConfig{nullptr, true});
        DumpGemmReduceMaxPerf<ADataType, BDataType, CDataType, ReduceDataType>(
            gemm_reduce_ave_time, M, N, K);
    }

    return pass ? 0 : 1;
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ReduceDataType,
          typename ReduceAccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename UnaryIdenticElementOp,
          typename UnarySquareElementOp,
          typename UnaryDivElementOp,
          typename ReduceOp0,
          typename ReduceOp1,
          typename DeviceGemmReduceInstance,
          typename ReferenceGemmInstance>
int run_gemm_reduce_mean_squaremean_xdl(ck::index_t M,
                                        ck::index_t N,
                                        ck::index_t K,
                                        ck::index_t StrideA,
                                        ck::index_t StrideB,
                                        ck::index_t StrideC,
                                        bool do_verification,
                                        int init_method,
                                        bool time_kernel)
{
    auto f_host_tensor_descriptor =
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

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> reduce0_m_host_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));
    Tensor<ReduceDataType> reduce1_m_host_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));

    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<ReduceDataType> reduce0_m_device_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));
    Tensor<ReduceDataType> reduce1_m_device_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "reduce0_m: " << reduce0_m_host_result.mDesc << std::endl;
    std::cout << "reduce1_m: " << reduce1_m_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        break;
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());
    DeviceMem reduce0_device_buf(sizeof(ReduceDataType) *
                                 reduce0_m_device_result.mDesc.GetElementSpaceSize());
    DeviceMem reduce1_device_buf(sizeof(ReduceDataType) *
                                 reduce1_m_device_result.mDesc.GetElementSpaceSize());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

    auto a_element_op                     = AElementwiseOperation{};
    auto b_element_op                     = BElementwiseOperation{};
    auto c_element_op                     = CElementwiseOperation{};
    std::array<void*, 3> gemm_element_ops = {&a_element_op, &b_element_op, &c_element_op};

    auto passthrough                            = UnaryIdenticElementOp{};
    auto square                                 = UnarySquareElementOp{};
    auto div                                    = UnaryDivElementOp{N};
    std::array<void*, 2> reduce_in_element_ops  = {&passthrough, &square};
    std::array<void*, 2> reduce_out_element_ops = {&div, &div};

    std::array<void*, 2> p_reduces = {reduce0_device_buf.GetDeviceBuffer(),
                                      reduce1_device_buf.GetDeviceBuffer()};

    // do GEMM
    auto gemm     = DeviceGemmReduceInstance{};
    auto invoker  = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(a_device_buf.GetDeviceBuffer(),
                                      b_device_buf.GetDeviceBuffer(),
                                      nullptr,
                                      {},
                                      c_device_buf.GetDeviceBuffer(),
                                      p_reduces,
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      {},
                                      gemm_element_ops,
                                      {},
                                      reduce_in_element_ops,
                                      reduce_out_element_ops);

    if(!gemm.IsSupportedArgument(argument))
    {
        throw std::runtime_error(
            "wrong! device_gemm with the specified compilation parameters does "
            "not support this GEMM problem");
    }

    // init reducetion buffer to 0
    reduce0_device_buf.SetZero();
    reduce1_device_buf.SetZero();

    // if time_kernel == true, kernel will run multiple times. This kernel use atomic-add so result
    // will not be correct. need to set time_kernel = false for correctness test
    invoker.Run(argument, StreamConfig{nullptr, false});
    bool pass = true;

    if(do_verification)
    {
        c_device_buf.FromDevice(c_m_n_device_result.mData.data());
        reduce0_device_buf.FromDevice(reduce0_m_device_result.mData.data());
        reduce1_device_buf.FromDevice(reduce1_m_device_result.mData.data());

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        auto reduce0_op = ReduceOp0{};
        auto reduce1_op = ReduceOp1{};

        for(int m = 0; m < M; ++m)
        {
            auto reduce0_acc = reduce0_op.template GetIdentityValue<ReduceAccDataType>();
            auto reduce1_acc = reduce1_op.template GetIdentityValue<ReduceAccDataType>();

            for(int n = 0; n < N; ++n)
            {
                auto c_val = ck::type_convert<ReduceAccDataType>(c_m_n_host_result(m, n));
                ReduceAccDataType square_c_val;
                square(square_c_val, c_val);

                reduce0_op(reduce0_acc, c_val);
                reduce1_op(reduce1_acc, square_c_val);
            }

            div(reduce0_acc, reduce0_acc);
            div(reduce1_acc, reduce1_acc);
            reduce0_m_host_result(m) = ck::type_convert<ReduceDataType>(reduce0_acc);
            reduce1_m_host_result(m) = ck::type_convert<ReduceDataType>(reduce1_acc);
        }

        pass = ck::utils::check_err(c_m_n_device_result.mData,
                                    c_m_n_host_result.mData,
                                    "Error: Incorrect results c") &&
               ck::utils::check_err(reduce0_m_device_result.mData,
                                    reduce0_m_host_result.mData,
                                    "Error: Incorrect results d0",
                                    1e-4,
                                    1e-5) &&
               ck::utils::check_err(reduce1_m_device_result.mData,
                                    reduce1_m_host_result.mData,
                                    "Error: Incorrect results d1",
                                    1e-3,
                                    1e-5);
    }

    if(time_kernel)
    {
        float ave_time = invoker.Run(argument, StreamConfig{nullptr, true});
        DumpGemmReduceMeanSquareMeanPerf<ADataType, BDataType, CDataType, ReduceDataType>(
            ave_time, M, N, K);
    }

    return pass ? 0 : 1;
}
