// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <gtest/gtest.h>

#include "ck/utility/data_type.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/number.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_mx.hpp"
#include "ck/library/tensor_operation_instance/gpu/gemm_mx.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_mx_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

namespace ck {
namespace test {

namespace {
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
} // namespace

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          int ScaleBlockSize>
bool profile_gemm_mx_impl(int do_verification,
                          int init_method,
                          bool do_log,
                          bool time_kernel,
                          int M,
                          int N,
                          int K,
                          int StrideA,
                          int StrideB,
                          int StrideC,
                          int KBatch,
                          int n_warmup,
                          int n_iter,
                          uint64_t rotating = 0)
{
    if(K % ScaleBlockSize != 0)
    {
        throw std::runtime_error("wrong! K must be multiple of ScaleBlockSize.");
    };

    using ScaleDataType = e8m0_bexp_t;
    using AScaleLayout  = Row;
    using BScaleLayout  = Col;

    bool pass = true;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            using namespace ck::literals;

            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };
    auto f_get_default_stride =
        [](ck::index_t row, ck::index_t col, ck::index_t stride, auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<ck::index_t>(col);
                }
                else
                {
                    return static_cast<ck::index_t>(row);
                }
            }
            else
                return static_cast<ck::index_t>(stride);
        };

    auto Scale_Stride_AM = f_get_default_stride(M, K / ScaleBlockSize, -1, AScaleLayout{});
    auto Scale_Stride_BN = f_get_default_stride(K / ScaleBlockSize, N, -1, BScaleLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    Tensor<ScaleDataType> a_m_k_scale(f_host_tensor_descriptor(
        M, K / ScaleBlockSize, Scale_Stride_AM, AScaleLayout{})); // scales for A
    Tensor<ScaleDataType> b_k_n_scale(f_host_tensor_descriptor(
        K / ScaleBlockSize, N, Scale_Stride_BN, BScaleLayout{})); // scales for B

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::size_t total_gemm_needed =
        a_m_k.GetElementSpaceSizeInBytes() + b_k_n.GetElementSpaceSizeInBytes() +
        a_m_k_scale.GetElementSpaceSizeInBytes() + b_k_n_scale.GetElementSpaceSizeInBytes();
    int rotating_count = std::max(
        1,
        std::min(n_iter,
                 static_cast<int>(std::ceil(static_cast<double>(rotating) / total_gemm_needed))));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "a_m_k_scale: " << a_m_k_scale.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "b_k_n_scale: " << b_k_n_scale.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_device_result.mDesc << std::endl;
    std::cout << "rotating count: " << rotating_count << std::endl;

    switch(init_method)
    {
    case 0: // Initializations for development and debugging
        ck::utils::FillConstant<ADataType>{ck::type_convert<ADataType>(1.0f)}(a_m_k);
        ck::utils::FillConstant<ScaleDataType>{ck::type_convert<ScaleDataType>(2.0f)}(a_m_k_scale);
        ck::utils::FillConstant<BDataType>{ck::type_convert<BDataType>(0.5f)}(b_k_n);
        ck::utils::FillConstant<ScaleDataType>{ck::type_convert<ScaleDataType>(1.0f)}(b_k_n_scale);
        if(do_log)
        {
            std::cout << "Init A = {1}" << std::endl;
            std::cout << "Init A scale = {2.0}" << std::endl;
            std::cout << "Init B = {0.5}" << std::endl;
            std::cout << "Init B scale = {1.0}" << std::endl;
            std::cout << "Expect C = {K}" << std::endl;
        }
        break;

    case 1:

        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-4, 5}); // Z[-4,4]
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-4, 5}); // Z[-4,4]

        a_m_k_scale.GenerateTensorValue(
            GeneratorTensor_2<ScaleDataType>{125, 129}); // scales: {0.25, 0.5, 1, 2}
        b_k_n_scale.GenerateTensorValue(
            GeneratorTensor_2<ScaleDataType>{125, 129}); // scales: {0.25, 0.5, 1, 2}

        break;

    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-2.0, 2.0});
        a_m_k_scale.GenerateTensorValue(
            GeneratorTensor_3<ScaleDataType>{powf(2.0f, -125.0f), 1.0f}); // R[2^-125, 1]

        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-2.0, 2.0});
        b_k_n_scale.GenerateTensorValue(
            GeneratorTensor_3<ScaleDataType>{powf(2.0f, -125.0f), 1.0f});
        break;
    }

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    if(do_log > 0)
        std::cout << "Device memory allocation..." << std::endl;

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem a_scale_device_buf(sizeof(ScaleDataType) * a_m_k_scale.mDesc.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem b_scale_device_buf(sizeof(ScaleDataType) * b_k_n_scale.mDesc.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    if(do_log > 0)
        std::cout << "Upload data to device..." << std::endl;
    a_device_buf.ToDevice(a_m_k.mData.data());
    a_scale_device_buf.ToDevice(a_m_k_scale.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    b_scale_device_buf.ToDevice(b_k_n_scale.mData.data());

    if(do_log > 0)
        std::cout << "Done." << std::endl;

    using DeviceOp = ck::tensor_operation::device::DeviceGemmMX<ALayout,
                                                                BLayout,
                                                                CLayout,
                                                                ADataType,
                                                                ScaleDataType,
                                                                BDataType,
                                                                ScaleDataType,
                                                                CDataType,
                                                                ScaleBlockSize,
                                                                AElementOp,
                                                                BElementOp,
                                                                CElementOp>;

    // get device op instances
    const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();

    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;

    // Run reference GEMM
    if(do_verification)
    {
        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceMXGemm<ADataType,
                                                        BDataType,
                                                        CDataType,
                                                        float, // AccDataType
                                                        ScaleDataType,
                                                        AElementOp,
                                                        BElementOp,
                                                        CElementOp,
                                                        float, // ComputeTypeA
                                                        float  // ComputeTypeB
                                                        >;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_m_k,
                                                  a_m_k_scale,
                                                  b_k_n,
                                                  b_k_n_scale,
                                                  c_m_n_host_result,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op);

        ref_invoker.Run(ref_argument);
    }

    std::string best_op_name;
    std::optional<std::string> best_op_object_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;

    // profile device GEMM instances
    for(auto& op_ptr : op_ptrs)
    {
        std::vector<int> kbatch_list = {1, 2, 4, 8, 16, 19, 32, 38}; // use these when KBatch <= 0

        if(KBatch > 0)
        {
            kbatch_list = {KBatch};
        }

        for(std::size_t i = 0; i < kbatch_list.size(); i++)
        {
            auto kbatch_curr = kbatch_list[i];

            auto argument_ptr = op_ptr->MakeArgumentPointer(
                static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                static_cast<ScaleDataType*>(a_scale_device_buf.GetDeviceBuffer()),
                static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                static_cast<ScaleDataType*>(b_scale_device_buf.GetDeviceBuffer()),
                static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                M,
                N,
                K,
                StrideA,
                Scale_Stride_AM,
                StrideB,
                Scale_Stride_BN,
                StrideC,
                kbatch_curr,
                a_element_op,
                b_element_op,
                c_element_op);

            auto invoker_ptr = op_ptr->MakeInvokerPointer();

            if(op_ptr->IsSupportedArgument(argument_ptr.get()))
            {

                // re-init C to zero before profiling next kernel
                c_device_buf.SetZero();

                invoker_ptr->Run(argument_ptr.get(),
                                 StreamConfig{nullptr, false, 0, n_warmup, n_iter});

                if(do_verification)
                {
                    c_device_buf.FromDevice(c_m_n_device_result.mData.data());

                    if(do_log)
                    {

                        if(init_method == 0)
                        {
                            auto expected = static_cast<float>(K);
                            auto computed = type_convert<float>(c_m_n_device_result(0, 12));

                            pass = pass & (std::abs(expected - computed) <= 0.0f);
                            std::cout << "\nExpected vs Computed: " << expected << " vs "
                                      << computed << ((pass) ? " (PASSED!)" : " (FAILED!)")
                                      << std::endl
                                      << std::endl;
                        }
                        else
                        {
                            LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(std::cout << "a_scale : ", a_m_k_scale.mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(std::cout << "b_scale: ", b_k_n_scale.mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(
                                std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                                << std::endl;
                            LogRangeAsType<float>(
                                std::cout << "c_device: ", c_m_n_device_result.mData, ",")
                                << std::endl;
                        }
                    }

                    pass = pass & ck::utils::check_err(c_m_n_device_result, c_m_n_host_result);
                }

                std::string op_name                    = op_ptr->GetTypeString();
                std::optional<std::string> op_obj_name = op_ptr->GetObjectName();

                float ave_time = invoker_ptr->Run(argument_ptr.get(),
                                                  StreamConfig{nullptr,
                                                               time_kernel,
                                                               0,
                                                               n_warmup,
                                                               n_iter,
                                                               rotating_count > 1,
                                                               rotating_count});

                // Output size(M*N) * [dot product(2K) + product of scales(K/ScaleBlockSize) +
                // scaling of partial sums(K/ScaleBlockSize)]
                // FLOPS = 2 * M * N * K + 2 * M * N * K / ScaleBlockSize
                std::size_t flop =
                    std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / ScaleBlockSize;

                std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                        sizeof(CDataType) * M * N +
                                        sizeof(ScaleDataType) * (M * K + K * N) / ScaleBlockSize;

                float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

                float gb_per_sec = num_btype / 1.E6 / ave_time;

                std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops
                          << " TFlops, " << gb_per_sec << " GB/s, " << op_name << ", KBatch "
                          << kbatch_curr << std::endl;

                if(tflops > best_tflops && ave_time > 1e-10)
                {
                    best_op_name        = op_name;
                    best_op_object_name = op_obj_name;
                    best_tflops         = tflops;
                    best_ave_time       = ave_time;
                    best_gb_per_sec     = gb_per_sec;
                    best_kbatch         = kbatch_curr;
                }
            }
            else
            {
                std::cout << op_ptr->GetTypeString() << " does not support this problem"
                          << std::endl;
            }
        }
    }

    if constexpr(is_same<CDataType, float>::value)
    {
        std::cout << "Best Perf for datatype = f32";
    }
    else if constexpr(is_same<CDataType, half_t>::value)
    {
        std::cout << "Best Perf for datatype = f16";
    }
    else if constexpr(is_same<CDataType, bhalf_t>::value)
    {
        std::cout << "Best Perf for datatype = bf16";
    }
    else if constexpr(is_same<CDataType, int8_t>::value)
    {
        std::cout << "Best Perf for datatype = int8";
    }

    if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " ALayout =  RowMajor";
    }
    else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " ALayout =  ColumnMajor";
    }

    if constexpr(is_same<BLayout, tensor_layout::gemm::RowMajor>::value)
    {
        std::cout << " BLayout =  RowMajor";
    }
    else if constexpr(is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value)
    {
        std::cout << " BLayout =  ColumnMajor";
    }

    std::cout << " M = " << M << " N = " << N << " K = " << K << " StrideA = " << StrideA
              << " StrideB = " << StrideB << " StrideC = " << StrideC << " KBatch = " << best_kbatch
              << " : " << best_ave_time << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec
              << " GB/s, " << best_op_name << std::endl;

    if(best_op_object_name)
        std::cout << best_op_object_name.value() << std::endl;

    return pass;
}

template <typename Tuple>
class TestGemmMX : public testing::Test
{
    using Row       = ck::tensor_layout::gemm::RowMajor;
    using F32       = float;
    using ScaleType = e8m0_bexp_t;

    protected:
    using ALayout     = std::tuple_element_t<0, Tuple>;
    using BLayout     = std::tuple_element_t<1, Tuple>;
    using CLayout     = Row;
    using ADataType   = std::tuple_element_t<2, Tuple>;
    using BDataType   = std::tuple_element_t<3, Tuple>;
    using CDataType   = std::tuple_element_t<4, Tuple>;
    using AccDataType = float;

    public:
    static constexpr index_t ScaleBlockSize = std::tuple_element_t<5, Tuple>{};
    static constexpr bool verify_           = true;
    static constexpr int init_method_       = 2; // decimal value initialization
    static constexpr bool log_              = false;
    static constexpr bool bench_            = false; // measure kernel performance
    std::vector<int> k_batches_;

    void SetUp() override { k_batches_ = {1}; }

    void Run(const int M,
             const int N,
             const int K,
             const int StrideA,
             const int StrideB,
             const int StrideC)
    {
        for(auto kb : k_batches_)
        {
            RunSingle(M, N, K, StrideA, StrideB, StrideC, kb);
        }
    }

    void RunSingle(const int M,
                   const int N,
                   const int K,
                   const int StrideA,
                   const int StrideB,
                   const int StrideC,
                   int kbatch   = 1,
                   int n_warmup = 1,
                   int n_iter   = 10)
    {
        bool pass = ck::test::profile_gemm_mx_impl<ADataType,
                                                   BDataType,
                                                   CDataType,
                                                   ALayout,
                                                   BLayout,
                                                   CLayout,
                                                   ScaleBlockSize>(verify_,
                                                                   init_method_,
                                                                   log_,
                                                                   bench_,
                                                                   M,
                                                                   N,
                                                                   K,
                                                                   StrideA,
                                                                   StrideB,
                                                                   StrideC,
                                                                   kbatch,
                                                                   n_warmup,
                                                                   n_iter);
        EXPECT_TRUE(pass);
    }
};

} // namespace test
} // namespace ck
