#pragma once
#include <iomanip>
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_conv.hpp"
#include "tensor_layout.hpp"
#include "device_tensor.hpp"
#include "element_wise_operation.hpp"
#include "device_gemm.hpp"
#include "reference_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_grouped_gemm_instance {

using DeviceGroupedGemmNoOpPtr =
    ck::tensor_operation::device::DeviceGroupedGemmPtr<ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

void add_device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(std::vector<DeviceGroupedGemmNoOpPtr>&);
//void add_device_grouped_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(std::vector<DeviceGroupedGemmNoOpPtr>&);
//void add_device_grouped_gemm_xdl_f16_f16_f16_km_kn_mn_instances(std::vector<DeviceGroupedGemmNoOpPtr>&);
//void add_device_grouped_gemm_xdl_f16_f16_f16_km_nk_mn_instances(std::vector<DeviceGroupedGemmNoOpPtr>&);

} // namespace device_grouped_gemm_instance
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
void profile_grouped_gemm_impl(int do_verification,
                       int init_method,
                       bool do_log,
                       int nrepeat,
                       std::vector<int> Ms,
                       std::vector<int> Ns,
                       std::vector<int> Ks,
                       std::vector<int> StrideAs,
                       std::vector<int> StrideBs,
                       std::vector<int> StrideCs)
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


    std::vector<Tensor<ADataType>> a_m_k;
    std::vector<Tensor<BDataType>> b_k_n;
    std::vector<Tensor<CDataType>> c_m_n;

    for(int i = 0; i < Ms.size(); i++)
    {
        a_m_k.push_back(Tensor<ADataType>(f_host_tensor_descriptor(
            Ms[i], Ks[i], StrideAs[i], ALayout{})));
        b_k_n.push_back(Tensor<BDataType>(f_host_tensor_descriptor(
            Ks[i], Ns[i], StrideBs[i], BLayout{})));
        c_m_n.push_back(Tensor<CDataType>(f_host_tensor_descriptor(
            Ms[i], Ns[i], StrideCs[i], CLayout{})));

        std::cout << "a_m_k[" << i << "]:" << a_m_k[i].mDesc << std::endl;
        std::cout << "b_k_n[" << i << "]:" << b_k_n[i].mDesc << std::endl;
        std::cout << "c_m_n[" << i << "]:" << c_m_n[i].mDesc << std::endl;

        std::size_t num_thread = std::thread::hardware_concurrency();
        switch(init_method)
        {
            case 0: break;
            case 1:
                    a_m_k[i].GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5}, num_thread);
                    b_k_n[i].GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5}, num_thread);
                    break;
            default:
                    a_m_k[i].GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}, num_thread);
                    b_k_n[i].GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5}, num_thread);
        }

        // set zero to c_device_buf
        c_m_n[i].GenerateTensorValue(GeneratorTensor_0<CDataType>{}, num_thread);
    }


    using AElementOp = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp = ck::tensor_operation::element_wise::PassThrough;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};

    // if(do_verification)
    // {

    // }

    std::vector<DeviceMem> a_device_buf, b_device_buf, c_device_buf;
    //DeviceMem a_device_buf(sizeof(ADataType) * a_m_k[i].mDesc.GetElementSpace());
    //DeviceMem b_device_buf(sizeof(BDataType) * b_k_n[i].mDesc.GetElementSpace());
    //DeviceMem c_device_buf(sizeof(CDataType) * c_m_n[i].mDesc.GetElementSpace());

    for(int i = 0; i < Ms.size(); i++)
    {
        a_device_buf.push_back(DeviceMem(sizeof(ADataType) * a_m_k[i].mDesc.GetElementSpace()));
        b_device_buf.push_back(DeviceMem(sizeof(BDataType) * b_k_n[i].mDesc.GetElementSpace()));
        c_device_buf.push_back(DeviceMem(sizeof(CDataType) * c_m_n[i].mDesc.GetElementSpace()));

        a_device_buf[i].ToDevice(a_m_k[i].mData.data());
        b_device_buf[i].ToDevice(b_k_n[i].mData.data());
        c_device_buf[i].ToDevice(c_m_n[i].mData.data());
    }


    // add device GEMM instances
    std::vector<ck::tensor_operation::device::device_grouped_gemm_instance::DeviceGroupedGemmNoOpPtr> gemm_ptrs;

    if constexpr(is_same<ADataType, half_t>::value && is_same<BDataType, half_t>::value &&
                      is_same<CDataType, half_t>::value)
    {
        if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                     is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_grouped_gemm_instance::
                add_device_grouped_gemm_xdl_f16_f16_f16_mk_kn_mn_instances(gemm_ptrs);

        }
#if 0
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            if(KBatch > 1)
            {
                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_splitk_f16_f16_f16_mk_nk_mn_instances(gemm_ptrs);
            }
            else
            {
                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_f16_f16_f16_mk_nk_mn_instances(gemm_ptrs);

                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instances(gemm_ptrs);

                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instances(gemm_ptrs);
            }
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            if(KBatch > 1)
            {
                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_splitk_f16_f16_f16_km_kn_mn_instances(gemm_ptrs);
            }
            else
            {
                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_f16_f16_f16_km_kn_mn_instances(gemm_ptrs);

                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instances(gemm_ptrs);
            }
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            if(KBatch > 1)
            {
                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_splitk_f16_f16_f16_km_nk_mn_instances(gemm_ptrs);
            }
            else
            {
                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_f16_f16_f16_km_nk_mn_instances(gemm_ptrs);

                ck::tensor_operation::device::device_grouped_gemm_instance::
                    add_device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instances(gemm_ptrs);
            }
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

#if 0
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
                                          KBatch);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            std::string gemm_name = gemm_ptr->GetTypeString();

            float ave_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * M + sizeof(CDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
                      << gb_per_sec << " GB/s, " << gemm_name << std::endl;

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

                if constexpr(is_same<ADataType, ck::bhalf_t>::value &&
                             is_same<BDataType, ck::bhalf_t>::value &&
                             is_same<CDataType, ck::bhalf_t>::value)
                {
                    Tensor<float> a_f32_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
                    Tensor<float> b_f32_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
                    Tensor<float> c_m_n_host_result(
                        f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
                    Tensor<float> c_m_n_device_f32_result(
                        f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

                    bf16_to_f32_(a_m_k, a_f32_m_k);
                    bf16_to_f32_(b_k_n, b_f32_k_n);
                    bf16_to_f32_(c_m_n_device_result, c_m_n_device_f32_result);

                    using ReferenceGemmInstance = ck::tensor_operation::host::
                        ReferenceGemm<float, float, float, AElementOp, BElementOp, CElementOp>;

                    auto ref_gemm    = ReferenceGemmInstance{};
                    auto ref_invoker = ref_gemm.MakeInvoker();

                    auto ref_argument = ref_gemm.MakeArgument(a_f32_m_k,
                                                              b_f32_k_n,
                                                              c_m_n_host_result,
                                                              a_element_op,
                                                              b_element_op,
                                                              c_element_op);

                    ref_invoker.Run(ref_argument);

                    check_error(c_m_n_host_result, c_m_n_device_f32_result);

                    if(do_log)
                    {
                        LogRangeAsType<float>(
                            std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                            << std::endl;
                    }
                }
                else
                {
                    Tensor<CDataType> c_m_n_host_result(
                        f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

                    using ReferenceGemmInstance =
                        ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                  BDataType,
                                                                  CDataType,
                                                                  AElementOp,
                                                                  BElementOp,
                                                                  CElementOp>;

                    auto ref_gemm    = ReferenceGemmInstance{};
                    auto ref_invoker = ref_gemm.MakeInvoker();

                    auto ref_argument = ref_gemm.MakeArgument(
                        a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

                    ref_invoker.Run(ref_argument);
                    check_error(c_m_n_host_result, c_m_n_device_result);

                    if(do_log)
                    {
                        LogRangeAsType<float>(
                            std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                            << std::endl;
                    }
                }

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
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
#endif

    std::cout << "Best Perf: " << best_ave_time << " ms, " << best_tflops << " TFlops, "
              << best_gb_per_sec << " GB/s, " << best_gemm_name << std::endl;
}

} // namespace profiler
} // namespace ck
