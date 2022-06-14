#pragma once
#include "config.hpp"
#include "device.hpp"
#include "host_tensor.hpp"
#include "host_tensor_generator.hpp"
#include "host_conv.hpp"
#include "tensor_layout.hpp"
#include "device_tensor.hpp"
#include "element_wise_operation.hpp"
#include "reduction_operator.hpp"
#include "device_gemm_reduce.hpp"
#include "reference_gemm.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {

using F32            = float;
using F16            = ck::half_t;
using DPtrsGlobal    = ck::Tuple<F32*, F32*>;
using Div            = ck::tensor_operation::element_wise::UnaryIdentic<F32, F32, true>;
using Identity       = ck::tensor_operation::element_wise::UnaryIdentic<F32, F32, false>;
using Square         = ck::tensor_operation::element_wise::UnarySquare<F32, F32, false>;
using DInElementOps  = ck::Tuple<Identity, Square>;
using DOutElementOps = ck::Tuple<Div, Div>;

using DeviceGemmReduceNoOpPtr = ck::tensor_operation::device::DeviceGemmReducePtr<
    DPtrsGlobal,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    ck::tensor_operation::element_wise::PassThrough,
    DInElementOps,
    DOutElementOps>;

void add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instances(
    std::vector<DeviceGemmReduceNoOpPtr>&);

void add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instances(
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
bool profile_gemm_reduce_impl(int do_verification,
                              int init_method,
                              bool do_log,
                              bool time_kernel,
                              int M,
                              int N,
                              int K,
                              int StrideA,
                              int StrideB,
                              int StrideC)
{
    bool pass = true;

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
    Tensor<DDataType> d0_m_host_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));
    Tensor<DDataType> d1_m_host_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));

    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<DDataType> d0_m_device_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));
    Tensor<DDataType> d1_m_device_result(
        HostTensorDescriptor(std::vector<std::size_t>({static_cast<std::size_t>(M)})));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;
    std::cout << "d0_m: " << d0_m_host_result.mDesc << std::endl;
    std::cout << "d1_m: " << d1_m_host_result.mDesc << std::endl;

    std::size_t num_thread = 1;
    switch(init_method)
    {
    case 0: break;
    case 1:
        std::srand(0);
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-5, 5}, num_thread);
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-5, 5}, num_thread);
        break;
    default:
        std::srand(0);
        a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{0.0, 1.0}, num_thread);
        b_k_n.GenerateTensorValue(GeneratorTensor_3<BDataType>{-0.5, 0.5}, num_thread);
    }

    using AElementOp        = ck::tensor_operation::element_wise::PassThrough;
    using BElementOp        = ck::tensor_operation::element_wise::PassThrough;
    using CElementOp        = ck::tensor_operation::element_wise::PassThrough;
    using D0ReduceOp        = ck::reduce::Add<float>;
    using D1ReduceOp        = ck::reduce::Add<float>;
    using UnaryDivElementOp = ck::tensor_operation::element_wise::UnaryIdentic<float, float, true>;
    using UnaryIdenticElementOp =
        ck::tensor_operation::element_wise::UnaryIdentic<float, float, false>;
    using UnarySquareElementOp =
        ck::tensor_operation::element_wise::UnarySquare<float, float, false>;
    using DxsInElementOps  = ck::Tuple<UnaryIdenticElementOp, UnarySquareElementOp>;
    using DxsOutElementOps = ck::Tuple<UnaryDivElementOp, UnaryDivElementOp>;

    const auto a_element_op = AElementOp{};
    const auto b_element_op = BElementOp{};
    const auto c_element_op = CElementOp{};
    const auto d0_reduce_op = D0ReduceOp{};
    const auto d1_reduce_op = D1ReduceOp{};

    auto dxs_in_element_op  = DxsInElementOps{};
    auto dxs_out_element_op = DxsOutElementOps{M, M};

    if(do_verification)
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
                                                                                BDataType,
                                                                                CDataType,
                                                                                DDataType,
                                                                                AElementOp,
                                                                                BElementOp,
                                                                                CElementOp>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(
            a_m_k, b_k_n, c_m_n_host_result, a_element_op, b_element_op, c_element_op);

        ref_invoker.Run(ref_argument);

        for(int m = 0; m < M; ++m)
        {
            float d0_acc = d0_reduce_op.GetIdentityValue();
            float d1_acc = d1_reduce_op.GetIdentityValue();

            for(int n = 0; n < N; ++n)
            {
                float c_val  = ck::type_convert<float>(c_m_n_host_result(m, n));
                float d0_val = 0;
                float d1_val = 0;

                dxs_in_element_op(ck::Number<0>{})(d0_val, c_val);
                dxs_in_element_op(ck::Number<1>{})(d1_val, c_val);
                d0_reduce_op(d0_acc, d0_val);
                d1_reduce_op(d1_acc, d1_val);
            }

            dxs_out_element_op(ck::Number<0>{})(d0_acc, d0_acc);
            dxs_out_element_op(ck::Number<1>{})(d1_acc, d1_acc);
            d0_m_host_result(m) = ck::type_convert<DDataType>(d0_acc);
            d1_m_host_result(m) = ck::type_convert<DDataType>(d1_acc);
        }
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpace());
    DeviceMem d0_device_buf(sizeof(DDataType) * d0_m_device_result.mDesc.GetElementSpace());
    DeviceMem d1_device_buf(sizeof(DDataType) * d1_m_device_result.mDesc.GetElementSpace());

    auto dxs_global = ck::make_tuple(static_cast<DDataType*>(d0_device_buf.GetDeviceBuffer()),
                                     static_cast<DDataType*>(d1_device_buf.GetDeviceBuffer()));

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());

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
                add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::RowMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instances(
                    gemm_ptrs);
        }
        else if constexpr(is_same<ALayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<BLayout, tensor_layout::gemm::ColumnMajor>::value &&
                          is_same<CLayout, tensor_layout::gemm::RowMajor>::value)
        {
            ck::tensor_operation::device::device_gemm_instance::
                add_device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instances(
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
                                          dxs_global,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideC,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          dxs_in_element_op,
                                          dxs_out_element_op);

        auto invoker_ptr = gemm_ptr->MakeInvokerPointer();

        if(gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            // init DO, D1 to 0
            d0_device_buf.SetZero();
            d1_device_buf.SetZero();

            float ave_time =
                invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, time_kernel});

            std::string gemm_name = gemm_ptr->GetTypeString();

            std::size_t flop = std::size_t(2) * M * N * K;

            std::size_t num_btype = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                                    sizeof(CDataType) * M * N + sizeof(CDataType) * N;

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
                d0_device_buf.FromDevice(d0_m_device_result.mData.data());
                d1_device_buf.FromDevice(d1_m_device_result.mData.data());

                float c_error  = check_error(c_m_n_host_result, c_m_n_device_result);
                float d0_error = check_error(d0_m_host_result, d0_m_device_result);
                float d1_error = check_error(d1_m_host_result, d1_m_device_result);

                pass = pass && (c_error < 1E-6);
                pass = pass && (d0_error < 1E-6);
                pass = pass && (d1_error < 1E-6);

                if(do_log)
                {
                    LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
                    LogRangeAsType<float>(std::cout << "c_host: ", c_m_n_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "c_device: ", c_m_n_device_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "d0_host: ", d0_m_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "d0_device: ", d0_m_device_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "d1_host: ", d1_m_host_result.mData, ",")
                        << std::endl;
                    LogRangeAsType<float>(std::cout << "d1_device: ", d1_m_device_result.mData, ",")
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
