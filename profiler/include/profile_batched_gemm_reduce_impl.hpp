#pragma once

#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_conv.hpp"
#include "tensor_layout.hpp"
#include "device_tensor.hpp"
#include "element_wise_operation.hpp"
#include "element_wise_reduce_operation.hpp"
#include "device_gemm_reduce.hpp"
#include "reference_batched_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {

using DeviceGemmReduceNoOpPtr = ck::tensor_operation::device::DeviceGemmReducePtr<
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::ReduceSum,
    ck::tensor_operation::element_wise::ReduceSquareSum>;

void add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

} // namespace device_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename DDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
bool profile_batched_gemm_reduce_impl(int do_verification,
                                      int init_method,
                                      bool do_log,
                                      int nrepeat,
                                      int M,
                                      int N,
                                      int K,
                                      int StrideA,
                                      int StrideB,
                                      int StrideC,
                                      int BatchCount)
{
    bool pass = true;

    auto f_host_tensor_descriptor = [](std::size_t batch_count,
                                       std::size_t row,
                                       std::size_t col,
                                       std::size_t stride,
                                       auto layout) {
        if(std::is_same<decltype(layout), ck::tensor_layout::gemm::RowMajor>::value)
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({row * stride, stride, 1}));
        }
        else
        {
            return HostTensorDescriptor(std::vector<std::size_t>({batch_count, row, col}),
                                        std::vector<std::size_t>({col * stride, 1, stride}));
        }
    };

    Tensor<ADataType> a_g_m_k(f_host_tensor_descriptor(BatchCount, M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_g_k_n(f_host_tensor_descriptor(BatchCount, K, N, StrideB, BLayout{}));

    Tensor<CDataType> c_g_m_n_host_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
    Tensor<DDataType> d0_g_m_host_result(HostTensorDescriptor(std::vector<std::size_t>(
        {static_cast<std::size_t>(BatchCount), static_cast<std::size_t>(M)})));
    Tensor<DDataType> d1_g_m_host_result(HostTensorDescriptor(std::vector<std::size_t>(
        {static_cast<std::size_t>(BatchCount), static_cast<std::size_t>(M)})));

    Tensor<CDataType> c_g_m_n_device_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
    Tensor<DDataType> d0_g_m_device_result(HostTensorDescriptor(std::vector<std::size_t>(
        {static_cast<std::size_t>(BatchCount), static_cast<std::size_t>(M)})));
    Tensor<DDataType> d1_g_m_device_result(HostTensorDescriptor(std::vector<std::size_t>(
        {static_cast<std::size_t>(BatchCount), static_cast<std::size_t>(M)})));

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "c_g_m_n: " << c_g_m_n_host_result.mDesc << std::endl;
    std::cout << "d0_g_m: " << d0_g_m_host_result.mDesc << std::endl;
    std::cout << "d1_g_m: " << d1_g_m_host_result.mDesc << std::endl;

    std::size_t num_thread = std::thread::hardware_concurrency();
    switch(init_method)
    {
    case 0: break;
    case 1:
        std::srand(0);
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5}, num_thread);
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5}, num_thread);
        break;
    default:
        std::srand(0);
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}, num_thread);
        b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5}, num_thread);
    }

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;
    using D0ReduceOp = ck::tensor_operation::element_wise::ReduceSum;
    using D1ReduceOp = ck::tensor_operation::element_wise::ReduceSquareSum;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};
    const auto d0_reduce_op = D0ReduceOp{};
    const auto d1_reduce_op = D1ReduceOp{};

    if(do_verification)
    {
        using ReferenceBatchedGemmInstance =
            ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
                                                             BDataType,
                                                             CDataType,
                                                             AElementOp,
                                                             BElementOp,
                                                             CElementOp>;

        auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
        auto ref_invoker      = ref_batched_gemm.MakeInvoker();

        auto ref_argument = ref_batched_gemm.MakeArgument(
            a_g_m_k, b_g_k_n, c_g_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        for(int batch = 0; batch < BatchCount; ++batch)
        {
            for(int m = 0; m < M; ++m)
            {
                float d0_acc = d0_reduce_op.GetReduceZeroValue();
                float d1_acc = d1_reduce_op.GetReduceZeroValue();

                for(int n = 0; n < N; ++n)
                {
                    d0_reduce_op.Reduce(d0_acc, c_g_m_n_host_result(batch, m, n));
                    d1_reduce_op.Reduce(d1_acc, c_g_m_n_host_result(batch, m, n));
                }

                d0_g_m_host_result(batch, m) = d0_acc;
                d1_g_m_host_result(batch, m) = d1_acc;
            }
        }
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(CDataType) * c_g_m_n_device_result.mDesc.GetElementSpace());
    DeviceMem d0_device_buf(sizeof(DDataType) * d0_g_m_device_result.mDesc.GetElementSpace());
    DeviceMem d1_device_buf(sizeof(DDataType) * d1_g_m_device_result.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());

    // add device GEMM instances
    std::vector<ck::tensor_operation::device::device_gemm_instance::DeviceGemmReduceNoOpPtr>
        gemm_ptrs;

    if constexpr(is_same<ADataType, half_t>::value && is_same<BDataType, half_t>::value &&
                 is_same<CDataType, half_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn_instances(
                    gemm_ptrs);
        }
    }

    if(gemm_ptrs.size() <= 0)
    {
        throw std::runtime_error("wrong! no device GEMM instance found");
    }

    std::string best_gemm_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;

    // profile device GEMM instances
    for(auto& gemm_ptr : gemm_ptrs)
    {
        auto argument_ptr =
            gemm_ptr->MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                          static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                          static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                          static_cast<DDataType*>(d0_device_buf.GetDeviceBuffer()),
                                          static_cast<DDataType*>(d1_device_buf.GetDeviceBuffer()),
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          d0_reduce_op,
                                          d1_reduce_op,
                                          BatchCount);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // warm up
            invoker_ptr->Run(argument_ptr.get());

            // timing
            float total_time = 0;

            for(int i = 0; i < nrepeat; ++i)
            {
                // init DO, D1 to 0
                d0_device_buf.SetZero();
                d1_device_buf.SetZero();

                KernelTimer timer;

                timer.Start();

                invoker_ptr->Run(argument_ptr.get());

                timer.End();

                total_time += timer.GetElapsedTime();
            }

            float ave_time = total_time / nrepeat;

            std::string gemm_name = gemm_ptr->GetTypeString();

            std::size_t flop      = std::size_t(2) * BatchCount * M * N * K;
            std::size_t num_btype = sizeof(ADataType) * BatchCount * M * K +
                                    sizeof(BDataType) * BatchCount * K * N +
                                    sizeof(CDataType) * BatchCount * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << gemm_name << std::endl;

            if(tflops > best_tflops)
            {
                best_gemm_name  = gemm_name;
                best_tflops     = tflops;
                best_ave_time   = ave_time;
                best_gb_per_sec = gb_per_sec;
            }

            if(do_verification)
            {
                c_device_buf.FromDevice(c_g_m_n_device_result.mData.data());
                d0_device_buf.FromDevice(d0_g_m_device_result.mData.data());
                d1_device_buf.FromDevice(d1_g_m_device_result.mData.data());

                float c_error  = check_error(c_g_m_n_host_result, c_g_m_n_device_result);
                float d0_error = check_error(d0_g_m_host_result, d0_g_m_device_result);
                float d1_error = check_error(d1_g_m_host_result, d1_g_m_device_result);

                pass = pass && (c_error < 1E-6);
                pass = pass && (d0_error < 1E-6);
                pass = pass && (d1_error < 1E-6);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_g_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_g_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host: ", c_g_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "c_device: ", c_g_m_n_device_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "d0_host: ", d0_g_m_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "d0_device: ", d0_g_m_device_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "d1_host: ", d1_g_m_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "d1_device: ", d1_g_m_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << "does not support this GEMM problem" << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_gemm_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
