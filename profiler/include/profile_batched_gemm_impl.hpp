#pragma once

#include <memory>

#include "check_err.hpp"
#include "config.hpp"
#include "element_wise_operation.hpp"
#include "tensor_layout.hpp"
#include "device.hpp"
#include "host_tensor_generator.hpp"
#include "device_gemm.hpp"
#include "reference_batched_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_batched_gemm_instance {

using DeviceGemmNoOpPtr =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

void add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gkn_gmn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gnk_gmn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gkn_gmn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gnk_gmn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f32_f32_f32_gmk_gnk_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f32_f32_f32_gkm_gkn_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_f32_f32_f32_gkm_gnk_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_int8_int8_int8_gmk_gkn_gmn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_int8_int8_int8_gmk_gnk_gmn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_int8_int8_int8_gkm_gkn_gmn_instances(
    std::vector<DeviceGemmNoOpPtr>&);
void add_device_batched_gemm_xdl_int8_int8_int8_gkm_gnk_gmn_instances(
    std::vector<DeviceGemmNoOpPtr>&);

} // namespace device_batched_gemm_instance
} // namespace device
} // namespace tensor_operation
} // namespace ck

namespace ck {
namespace profiler {

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
bool profile_batched_gemm_impl(int do_verification,
                               int init_method,
                               bool do_log,
                               bool time_kernel,
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
        if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
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
    Tensor<CDataType> c_g_m_n_device_result(
        f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
    std::unique_ptr<Tensor<float>> c_f32_g_m_n_host_result   = nullptr;
    std::unique_ptr<Tensor<float>> c_f32_g_m_n_device_result = nullptr;

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "c_g_m_n: " << c_g_m_n_host_result.mDesc << std::endl;

    std::size_t num_thread = 1;
    switch(init_method)
    {
    case 0: break;
    case 1:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5}, num_thread);
        b_g_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5}, num_thread);
        break;
    default:
        a_g_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}, num_thread);
        b_g_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5}, num_thread);
    }
    // set zero to c_device_buf
    c_g_m_n_device_result.GenerateTensorValue(GeneratorTensor_0<CDataType>{}, num_thread);

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    if(do_verification)
    {
        if constexpr(is_same<ADataType, ck::bhalf_t>::value &&
                     is_same<BDataType, ck::bhalf_t>::value &&
                     is_same<CDataType, ck::bhalf_t>::value)
        {
            Tensor<float> a_f32_g_m_k(
                f_host_tensor_descriptor(BatchCount, M, K, StrideA, ALayout{}));
            Tensor<float> b_f32_g_k_n(
                f_host_tensor_descriptor(BatchCount, K, N, StrideB, BLayout{}));
            c_f32_g_m_n_host_result = std::make_unique<Tensor<float>>(
                f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));
            c_f32_g_m_n_device_result = std::make_unique<Tensor<float>>(
                f_host_tensor_descriptor(BatchCount, M, N, StrideC, CLayout{}));

            bf16_to_f32_(a_g_m_k, a_f32_g_m_k);
            bf16_to_f32_(b_g_k_n, b_f32_g_k_n);

            using ReferenceBatchedGemmInstance = ck::tensor_operation::host::
                ReferenceBatchedGemm<float, float, float, AElementOp, BElementOp, CElementOp>;

            auto ref_batched_gemm = ReferenceBatchedGemmInstance{};
            auto ref_invoker      = ref_batched_gemm.MakeInvoker();

            auto ref_argument = ref_batched_gemm.MakeArgument(a_f32_g_m_k,
                                                              b_f32_g_k_n,
                                                              *c_f32_g_m_n_host_result,
                                                              a_element_op,
                                                              b_element_op,
                                                              c_element_op);

            ref_invoker.Run(ref_argument);
        }
        else
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
        }
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_g_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_g_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(CDataType) * c_g_m_n_device_result.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_g_m_k.mData.data());
    b_device_buf.ToDevice(b_g_k_n.mData.data());
    c_device_buf.ToDevice(c_g_m_n_device_result.mData.data());

    // add device GEMM instances
    std::vector<ck::tensor_operation::device::device_batched_gemm_instance::DeviceGemmNoOpPtr>
        gemm_ptrs;

    if constexpr(is_same<ADataType, half_t>::value && is_same<BDataType, half_t>::value &&
                 is_same<CDataType, half_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f16_f16_f16_gmk_gnk_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f16_f16_f16_gkm_gkn_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f16_f16_f16_gkm_gnk_gmn_instances(gemm_ptrs);
        }
    }
    else if constexpr(is_same<ADataType, bhalf_t>::value && is_same<BDataType, bhalf_t>::value &&
                      is_same<CDataType, bhalf_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gkn_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gmk_gnk_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gkn_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gnk_gmn_instances(gemm_ptrs);
        }
    }
    else if constexpr(is_same<ADataType, float>::value && is_same<BDataType, float>::value &&
                      is_same<CDataType, float>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f32_f32_f32_gmk_gnk_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f32_f32_f32_gkm_gkn_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f32_f32_f32_gkm_gnk_gmn_instances(gemm_ptrs);
        }
    }
    else if constexpr(is_same<ADataType, int8_t>::value && is_same<BDataType, int8_t>::value &&
                      is_same<CDataType, int8_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_int8_int8_int8_gmk_gkn_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_int8_int8_int8_gmk_gnk_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_int8_int8_int8_gkm_gkn_gmn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_int8_int8_int8_gkm_gnk_gmn_instances(gemm_ptrs);
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
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          ck::tensor_operation::element_wise::PassThrough{},
                                          BatchCount);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string gemm_name = gemm_ptr->GetTypeString();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::size_t flop = std::size_t(2) * BatchCount * M * N * K;

            std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                     sizeof(CDataType) * M * N) *
                                    BatchCount;

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

                if constexpr(is_same<ADataType, ck::bhalf_t>::value &&
                             is_same<BDataType, ck::bhalf_t>::value &&
                             is_same<CDataType, ck::bhalf_t>::value)
                {

                    bf16_to_f32_(c_g_m_n_device_result, *c_f32_g_m_n_device_result);
                    float err = check_error(*c_f32_g_m_n_host_result, *c_f32_g_m_n_device_result);
                    pass      = pass && (err < 1E-6);
                }
                else
                {
                    float err = check_error(c_g_m_n_host_result, c_g_m_n_device_result);
                    pass      = pass && (err < 1E-6);
                }

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_g_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_g_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host: ", c_g_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(
                        std::cout << "c_device: ", c_g_m_n_device_result.mData, ",")
                        << std::endl;
                }
            }
        }
        else
        {
            std::cout << "this device GEMM instance does not support this GEMM problem"
                      << std::endl;
        }
    }

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_gemm_name << std::endl;

    return pass;
}

} // namespace profiler
} // namespace ck
