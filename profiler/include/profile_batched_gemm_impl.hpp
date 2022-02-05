#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_batched_gemm_instance {

using DeviceGemmNoOpPtr =
    ck::tensor_operation::device::DeviceGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

void add_device_batched_gemm_xdl_f16_f16_f16_gmk_gkn_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
// void add_device_batched_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
// void add_device_batched_gemm_xdl_f16_f16_f16_km_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
// void add_device_batched_gemm_xdl_f16_f16_f16_km_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);

// void
// add_device_batched_gemm_xdl_f32_f32_f32_gmk_gkn_gmn_instances(std::vector<DeviceGemmNoOpPtr>&);
// void add_device_batched_gemm_xdl_f32_f32_f32_mk_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
// void add_device_batched_gemm_xdl_f32_f32_f32_km_kn_mn_instances(std::vector<DeviceGemmNoOpPtr>&);
// void add_device_batched_gemm_xdl_f32_f32_f32_km_nk_mn_instances(std::vector<DeviceGemmNoOpPtr>&);

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
void profile_batched_gemm_impl(int do_verification,
                               int init_method,
                               bool do_log,
                               int nrepeat,
                               int M,
                               int N,
                               int K,
                               int StrideA,
                               int StrideB,
                               int StrideC,
                               int BatchCount = 1)
{
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

    std::cout << "a_g_m_k: " << a_g_m_k.mDesc << std::endl;
    std::cout << "b_g_k_n: " << b_g_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_g_m_n_host_result.mDesc << std::endl;

    std::size_t num_thread = std::thread::hardware_concurrency();
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

    if(do_verification)
    {
        // host_gemm_gmk_gkn_gmn(a_g_m_k,
        // b_g_k_n,
        // c_g_m_n_host_result,
        // ck::tensor_operation::element_wise::PassThrough{},
        // ck::tensor_operation::element_wise::PassThrough{},
        // ck::tensor_operation::element_wise::PassThrough{});
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
#if 0
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f16_f16_f16_km_kn_mn_instances(gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_batched_gemm_instance::
                add_device_batched_gemm_xdl_f16_f16_f16_km_nk_mn_instances(gemm_ptrs);
        }
#endif
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

            float ave_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

            std::size_t flop = std::size_t(2) * BatchCount * M * N * K;

            std::size_t num_btype = (sizeof(ADataType) * M * K + sizeof(BDataType) * K * M +
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

                check_error(c_g_m_n_host_result, c_g_m_n_device_result);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_g_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_g_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host  : ", c_g_m_n_host_result.mData, ",")
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
}

} // namespace profiler
} // namespace ck
