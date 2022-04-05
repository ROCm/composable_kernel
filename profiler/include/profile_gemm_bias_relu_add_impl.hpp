#pragma once

#include "check_err.hpp"
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_conv.hpp"
#include "tensor_layout.hpp"
#include "device_tensor.hpp"
#include "element_wise_operation.hpp"
#include "device_gemm_bias_activation_add.hpp"
#include "reference_gemm_bias_activation_add.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {

using DeviceGemmBiasReluAddPtr = ck::tensor_operation::device::DeviceGemmBiasActivationAddPtr<
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::AddReluAdd>;

void add_device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_mk_kn_mn_instances(
    std::vector<DeviceGemmBiasReluAddPtr>&);
void add_device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_mk_nk_mn_instances(
    std::vector<DeviceGemmBiasReluAddPtr>&);
void add_device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_km_kn_mn_instances(
    std::vector<DeviceGemmBiasReluAddPtr>&);
void add_device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_km_nk_mn_instances(
    std::vector<DeviceGemmBiasReluAddPtr>&);

} // namespace device_gemm_instance
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
void profile_gemm_bias_relu_add_impl(int do_verification,
                                     int init_method,
                                     bool do_log,
                                     int nrepeat,
                                     int M,
                                     int N,
                                     int K,
                                     int StrideA,
                                     int StrideB,
                                     int StrideC,
                                     int StrideC1,
                                     int KBatch = 1)
{
    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(is_same<decltype(layout), tensor_layout::gemm::RowMajor>::value)
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
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    // c0_n[n]
    Tensor<CDataType> c0_n(HostTensorDescriptor(
        std::vector<std::size_t>({static_cast<std::size_t>(N)}), std::vector<std::size_t>({1})));

    // c1_m_n[m ,n]
    Tensor<BDataType> c1_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "c0_n: " << c0_n.mDesc << std::endl;
    std::cout << "c1_m_n: " << c1_m_n.mDesc << std::endl;

    switch(init_method)
    {
    case 0: break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5});
        c0_n.GenerateTensorValue(GeneratorTensor_2<CDataType>{-5, 5});
        c1_m_n.GenerateTensorValue(GeneratorTensor_2<CDataType>{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5});
        c0_n.GenerateTensorValue(GeneratorTensor_3<CDataType>{0.0, 1.0});
        c1_m_n.GenerateTensorValue(GeneratorTensor_3<CDataType>{0.0, 1.0});
    }

    // set zero to c_device_buf
    c_m_n_device_result.GenerateTensorValue(GeneratorTensor_0<CDataType>{});

    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::AddReluAdd;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    if(do_verification)
    {
        using ReferenceGemmInstance =
            ck::tensor_operation::host::ReferenceGemmBiasActivationAdd<ADataType,
                                                                       BDataType,
                                                                       CDataType,
                                                                       AElementOp,
                                                                       BElementOp,
                                                                       CElementOp>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_m_k,
                                                  b_k_n,
                                                  c_m_n_host_result,
                                                  c0_n,
                                                  c1_m_n,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op);

        ref_invoker.Run(ref_argument);
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpace());
    DeviceMem c0_n_device_buf(sizeof(CDataType) * c0_n.mDesc.GetElementSpace());
    DeviceMem c1_m_n_device_buf(sizeof(CDataType) * c1_m_n.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    c_device_buf.ToDevice(c_m_n_device_result.mData.data());
    c0_n_device_buf.ToDevice(c0_n.mData.data());
    c1_m_n_device_buf.ToDevice(c1_m_n.mData.data());

    // add device GEMM instances
    std::vector<ck::tensor_operation::device::device_gemm_instance::DeviceGemmBiasReluAddPtr>
        gemm_ptrs;

    if constexpr(is_same<ADataType, half_t>::value && is_same<BDataType, half_t>::value &&
                 is_same<CDataType, half_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_mk_kn_mn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_mk_nk_mn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_km_kn_mn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_xdl_c_shuffle_bias_relu_add_f16_f16_f16_km_nk_mn_instances(
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
        auto argument_ptr = gemm_ptr->MakeArgumentPointer(
            static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c0_n_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c1_m_n_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC,
            StrideC1,
            a_element_op,
            b_element_op,
            c_element_op,
            KBatch);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string gemm_name = gemm_ptr->GetTypeString();

            float ave_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * M +
                                    sizeof(CDataType) * M * N + sizeof(CDataType) * N +
                                    sizeof(CDataType) * M * N;

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
                c_device_buf.FromDevice(c_m_n_device_result.mData.data());

                ck::utils::check_err(c_m_n_device_result.mData, c_m_n_host_result.mData);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a: ", a_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c0: ", c0_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c1: ", c1_m_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host: ", c_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "c_device: ", c_m_n_device_result.mData, ",")
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
}

} // namespace profiler
} // namespace ck
