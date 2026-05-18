// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

template <bool BPreShuffle, typename GemmInstance, typename ProblemType>
bool run_gemm(GemmInstance& op_ptrs, const ProblemType& problem_size, const ExecutionConfig& config)
{
    using namespace ck::literals;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;
    auto KBatch  = problem_size.KBatch;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
            {
                return HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride =
        [](std::size_t row, std::size_t col, ck::index_t stride, auto layout) {
            if(stride == -1 || stride == 0)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                {
                    return static_cast<std::size_t>(col);
                }
                else
                {
                    return static_cast<std::size_t>(row);
                }
            }
            else
                return static_cast<std::size_t>(stride);
        };

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));

    switch(config.init_method)
    {
    case 0:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    case 1:
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1<ADataType>{1});
        b_k_n.GenerateTensorValue(GeneratorTensor_2<BDataType>{-2, 2});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2<ADataType>{-2, 2});
        b_k_n.GenerateTensorValue(GeneratorTensor_1<BDataType>{1});
        break;
    }

    Tensor<float> a_m_k_f32({M, K});
    Tensor<float> b_k_n_f32({K, N});
    Tensor<float> a_m_k_gfx9_f32({M, K});
    Tensor<float> b_k_n_gfx9_f32({K, N});

    if constexpr(ck::is_same_v<ADataType, ck::pk_i4_t>)
    {
        permute_a_pk_i4(a_m_k, M, K, a_m_k_f32, a_m_k_gfx9_f32);
    }

    if constexpr(ck::is_same_v<BDataType, ck::pk_i4_t>)
    {
        permute_b_pk_i4(b_k_n, N, K, b_k_n_f32, b_k_n_gfx9_f32);
    }

    Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_gfx9_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    Tensor<BDataType> b_preshuffled(
        f_host_tensor_descriptor(K, N, StrideB, BLayout{})); // use laout only for size
    if constexpr(BPreShuffle)
    {
        static_assert(std::is_same_v<BLayout, ck::tensor_layout::gemm::RowMajor> == false);
        constexpr int NPerXdl = GemmConfig::N_Warp_Tile;
        preShuffleBuffer(b_k_n.mData.data(), b_preshuffled.mData.data(), N, K, NPerXdl, KPack);
    }

    DeviceMem a_m_k_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpaceSize());
    DeviceMem b_k_n_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpaceSize());
    DeviceMem c_m_n_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpaceSize());

    a_m_k_device_buf.ToDevice(a_m_k.mData.data());
    if constexpr(BPreShuffle)
    {
        b_k_n_device_buf.ToDevice(b_preshuffled.mData.data());
    }
    else
    {
        b_k_n_device_buf.ToDevice(b_k_n.mData.data());
    }
    DeviceMem workspace;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    bool pass = true;
    if((config.do_verification == 1) || (config.do_verification == 3))
    {
        if constexpr(ck::is_same_v<ADataType, ck::pk_i4_t> && ck::is_same_v<BDataType, ck::pk_i4_t>)
        {
            using PkInt4ReferenceGemmInstance =
                ck::tensor_operation::host::ReferenceGemm<float,
                                                          float,
                                                          CDataType,
                                                          AccDataType,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough>;
            auto ref_gemm    = PkInt4ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(a_m_k_f32,
                                                      b_k_n_f32,
                                                      c_m_n_host_result,
                                                      PassThrough{},
                                                      PassThrough{},
                                                      PassThrough{});

            ref_invoker.Run(ref_argument);

            auto ref_gfx9_argument = ref_gemm.MakeArgument(a_m_k_gfx9_f32,
                                                           b_k_n_gfx9_f32,
                                                           c_m_n_gfx9_host_result,
                                                           PassThrough{},
                                                           PassThrough{},
                                                           PassThrough{});

            ref_invoker.Run(ref_gfx9_argument);
        }
        else
        {
            auto ref_gemm    = ReferenceGemmInstance{};
            auto ref_invoker = ref_gemm.MakeInvoker();

            auto ref_argument = ref_gemm.MakeArgument(
                a_m_k, b_k_n, c_m_n_host_result, PassThrough{}, PassThrough{}, PassThrough{});

            ref_invoker.Run(ref_argument);
        }
    }
    std::cout << "found " << op_ptrs.size() << " instances" << std::endl;
    // do GEMM
    std::string best_op_name;
    std::optional<std::string> best_op_object_name;
    float best_ave_time   = 0;
    float best_tflops     = 0;
    float best_gb_per_sec = 0;
    float best_kbatch     = 0;
    size_t best_index     = 0;
    for(size_t i = 0; i < op_ptrs.size(); i++)
    {
        auto& gemm       = op_ptrs[i];
        auto invoker_ptr = gemm->MakeInvokerPointer();
        float ave_time   = 0;

        if(config.instance_index != -1 && config.instance_index != static_cast<int>(i))
        {
            continue;
        }

        auto argument_ptr =
            gemm->MakeArgumentPointer(static_cast<ADataType*>(a_m_k_device_buf.GetDeviceBuffer()),
                                      static_cast<BDataType*>(b_k_n_device_buf.GetDeviceBuffer()),
                                      static_cast<CDataType*>(c_m_n_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideC,
                                      KBatch,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op);

        if(!gemm->IsSupportedArgument(argument_ptr.get()))
        {
            std::cerr << gemm->GetTypeString() << " does not support this problem" << std::endl;
            continue;
        }

        if(config.do_verification)
        {
            ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false, 1});

            c_m_n_device_buf.FromDevice(c_m_n_device_result.mData.data());

            bool useGfx9Result =
                (ck::is_same_v<ADataType, ck::pk_i4_t> && ck::is_same_v<BDataType, ck::pk_i4_t> &&
                 (gemm->GetKPerBlock() & DisableGfx9I4ToF32) == 0);

            bool ret =
                ck::utils::check_err(c_m_n_device_result,
                                     useGfx9Result ? c_m_n_gfx9_host_result : c_m_n_host_result,
                                     "Error: Incorrect results!",
                                     get_rtol<CDataType>(),
                                     get_atol<CDataType>());
            pass &= ret;
            if(!ret)
            {
                std::cout << "Error: [" << i << "]: " << gemm->GetTypeString()
                          << " results incorrect!" << std::endl;
            }
        }

        if(config.time_kernel)
        {
            ave_time = invoker_ptr->Run(argument_ptr.get(),
                                        StreamConfig{nullptr,
                                                     config.time_kernel,
                                                     0,
                                                     config.cold_niters,
                                                     config.nrepeat,
                                                     true,
                                                     config.rotating_count});

            std::size_t flop = 2_uz * M * N * K;
            std::size_t num_btype =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s, " << " [" << i << "] " << gemm->GetTypeString() << std::endl;
            if(tflops > best_tflops && ave_time > 1e-10)
            {
                std::string op_name                    = gemm->GetTypeString();
                std::optional<std::string> op_obj_name = gemm->GetObjectName();

                best_op_name        = op_name;
                best_op_object_name = op_obj_name;
                best_tflops         = tflops;
                best_ave_time       = ave_time;
                best_gb_per_sec     = gb_per_sec;
                best_kbatch         = KBatch;
                best_index          = i;
            }
        }
    }

    if(config.time_kernel)
    {
        std::cout << "Best Perf for M = " << M << " N = " << N << " K = " << K
                  << " StrideA = " << StrideA << " StrideB = " << StrideB
                  << " StrideC = " << StrideC << " KBatch = " << best_kbatch << " : "
                  << best_ave_time << " ms, " << best_tflops << " TFlops, " << best_gb_per_sec
                  << " GB/s, " << " [" << best_index << "] " << best_op_name << std::endl;

        if(best_op_object_name)
            std::cout << best_op_object_name.value() << std::endl;
    }

    return pass;
}

template <bool BPreshuffle, typename GemmInstance>
bool run_gemm_splitk_example(GemmInstance& op_ptrs, int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) ||
           run_gemm<BPreshuffle>(op_ptrs, problem_size, config);
}
