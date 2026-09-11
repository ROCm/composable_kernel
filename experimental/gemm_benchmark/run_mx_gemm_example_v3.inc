// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <bool BPreShuffle, typename GemmInstance, typename ProblemType>
bool run_mx_gemm(GemmInstance& op_ptrs,
                 const ProblemType& problem_size,
                 const ExecutionConfig& config)
{
    using BRefLayout = ck::conditional_t<BPreShuffle, Col, BLayout>;

    auto M       = problem_size.M;
    auto N       = problem_size.N;
    auto K       = problem_size.K;
    auto StrideA = problem_size.StrideA;
    auto StrideB = problem_size.StrideB;
    auto StrideC = problem_size.StrideC;
    auto KBatch  = problem_size.KBatch;

    auto f_host_tensor_descriptor =
        [](ck::index_t row, ck::index_t col, ck::index_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                return HostTensorDescriptor({row, col}, {stride, 1});
            else
                return HostTensorDescriptor({row, col}, {1, stride});
        };
    auto f_get_default_stride =
        [](ck::index_t row, ck::index_t col, ck::index_t stride, auto layout) {
            if(stride == -1)
            {
                // give a chance if stride is -1, return a default packed stride
                if constexpr(std::is_same_v<decltype(layout), ck::tensor_layout::gemm::RowMajor>)
                    return static_cast<ck::index_t>(col);
                else
                    return static_cast<ck::index_t>(row);
            }
            else
                return static_cast<ck::index_t>(stride);
        };

    StrideA = f_get_default_stride(M, K, StrideA, ALayout{});
    StrideB = f_get_default_stride(K, N, StrideB, BLayout{});
    StrideC = f_get_default_stride(M, N, StrideC, CLayout{});

    if(K % ScaleBlockSize != 0)
    {
        throw std::runtime_error("wrong! K must be multiple of ScaleBlockSize.");
    };

    if(K % ck::packed_size_v<ADataType> != 0 || K % ck::packed_size_v<BDataType> != 0)
    {
        throw std::runtime_error("wrong! K must be multiple of packed size.");
    };

    // Hardcode scale layouts as per pipeline assumptions
    // TODO: Allow user to specify scale layouts
    using AScaleLayout = Row;
    using BScaleLayout = Col;

    auto Scale_Padded_M = ck::math::integer_least_multiple(M, ScaleBlockSize);
    auto Scale_Stride_AM =
        f_get_default_stride(Scale_Padded_M, K / ScaleBlockSize, -1, AScaleLayout{});
    auto Scale_Stride_BN = f_get_default_stride(K / ScaleBlockSize, N, -1, BScaleLayout{});

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    auto b_k_n =
        std::make_shared<Tensor<BDataType>>(f_host_tensor_descriptor(K, N, StrideB, BRefLayout{}));
    auto b_input = b_k_n;
    if constexpr(BPreShuffle)
        b_input = std::make_shared<Tensor<BDataType>>(
            f_host_tensor_descriptor(K, N, StrideB, BRefLayout{})); // use layout only for size

    // scales for A and B
    Tensor<XDataType> a_m_k_scale(f_host_tensor_descriptor(
        Scale_Padded_M, K / ScaleBlockSize, Scale_Stride_AM, AScaleLayout{}));
    Tensor<XDataType> b_k_n_scale(
        f_host_tensor_descriptor(K / ScaleBlockSize, N, Scale_Stride_BN, BScaleLayout{}));

    // shuffled scales for A and B
    Tensor<XDataType> a_shuffled_scale(f_host_tensor_descriptor(
        Scale_Padded_M, K / ScaleBlockSize, Scale_Stride_AM, AScaleLayout{}));
    Tensor<XDataType> b_shuffled_scale(
        f_host_tensor_descriptor(K / ScaleBlockSize, N, Scale_Stride_BN, BScaleLayout{}));

    Tensor<CDataType> c_m_n_host_result(
        f_host_tensor_descriptor(M, N, StrideC, CLayout{})); // host verification
    Tensor<CDataType> c_m_n_device_result(
        f_host_tensor_descriptor(M, N, StrideC, CLayout{})); // device result downloaded to host

    if(config.verbosity >= 0)
    {
        std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
        std::cout << "a_m_k_scale: " << a_m_k_scale.mDesc << std::endl;
        std::cout << "b_k_n: " << b_k_n->mDesc << std::endl;
        std::cout << "b_k_n_scale: " << b_k_n_scale.mDesc << std::endl;
        std::cout << "c_m_n_device_result: " << c_m_n_device_result.mDesc << std::endl;
    }

    auto a_data_element = [](float x) { return ck::type_convert<ADataType>(x); };
    auto b_data_element = [](float x) { return ck::type_convert<BDataType>(x); };

    using int_distr   = std::uniform_int_distribution<int>;
    using float_distr = std::uniform_real_distribution<float>;
    switch(config.init_method)
    {
    case 0: // Initializations for development and debugging

        ck::utils::FillConstant<ADataType>{a_data_element(0.5f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(2.0f)}(a_m_k_scale);

        ck::utils::FillConstant<BDataType>{b_data_element(2.0f)}(*b_k_n);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(0.5f)}(b_k_n_scale);

        if(config.verbosity > 0)
        {
            std::cout << "Init A = {0.5}" << std::endl;
            std::cout << "Init A scale = {2.0}" << std::endl;
            std::cout << "Init B = {2.0}" << std::endl;
            std::cout << "Init B scale = {0.5}" << std::endl;
            std::cout << "Expect C = {K}" << std::endl;
        }
        break;

    case 1:
        a_m_k.GenerateTensorDistr(
            int_distr{-5, 5}, ck::identity{}, std::minstd_rand(time(nullptr))); // Z[-5,5]
        b_k_n->GenerateTensorDistr(int_distr{-5, 5});                           // Z[-5,5]
        static_assert(ck::is_same_v<XDataType, ck::e8m0_bexp_t>);
        a_m_k_scale.GenerateTensorDistr(int_distr{125, 128}); // scales: {0.25, 0.5, 1, 2}
        b_k_n_scale.GenerateTensorDistr(int_distr{125, 128}); // scales: {0.25, 0.5, 1, 2}
        break;

    case 2:
        a_m_k.GenerateTensorDistr(
            float_distr{-2.0, 2.0}, ck::identity{}, std::minstd_rand(time(nullptr))); // R[-2,2]
        a_m_k_scale.GenerateTensorDistr(float_distr{powf(2.0f, -125.0f), 1.0f});

        b_k_n->GenerateTensorDistr(float_distr{-2.0, 2.0});
        b_k_n_scale.GenerateTensorDistr(float_distr{powf(2.0f, -125.0f), 1.0f});
        break;
    case 3:
        a_m_k.GenerateTensorDistr(float_distr{-2.0, 2.0}); // R[-2,2]
        a_m_k_scale.GenerateTensorDistr(float_distr{powf(2.0f, -125.0f), 1.0f});

        ck::utils::FillConstant<BDataType>{b_data_element(2.0f)}(*b_k_n);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(0.5f)}(b_k_n_scale);
        break;
    case 4:
        ck::utils::FillConstant<ADataType>{a_data_element(0.5f)}(a_m_k);
        ck::utils::FillConstant<XDataType>{ck::type_convert<XDataType>(2.0f)}(a_m_k_scale);

        b_k_n->GenerateTensorDistr(float_distr{-2.0, 2.0});
        b_k_n_scale.GenerateTensorDistr(float_distr{powf(2.0f, -125.0f), 1.0f});
        break;
    default:
        if(config.verbosity > 0)
        {
            std::cout << "NOTE: No input data initialization." << std::endl;
        }
    }

    if(ck::get_warp_size() == 64)
    {
        preShuffleScaleBuffer_gfx950<ck::is_same_v<ALayout, Row>>(a_m_k_scale.mData.data(),
                                                                  a_shuffled_scale.mData.data(),
                                                                  Scale_Padded_M,
                                                                  K / ScaleBlockSize);

        preShuffleScaleBuffer_gfx950<ck::is_same_v<BRefLayout, Col>>(
            b_k_n_scale.mData.data(), b_shuffled_scale.mData.data(), N, K / ScaleBlockSize);
    }
    else if(ck::get_warp_size() == 32)
    {
        preShuffleScaleBuffer_gfx1250<ck::e8m0_bexp_t, ScaleBlockSize, ck::is_same_v<ALayout, Row>>(
            a_m_k_scale.mData.data(),
            a_shuffled_scale.mData.data(),
            Scale_Padded_M,
            K / ScaleBlockSize);

        preShuffleScaleBuffer_gfx1250<ck::e8m0_bexp_t,
                                      ScaleBlockSize,
                                      ck::is_same_v<BRefLayout, Col>>(
            b_k_n_scale.mData.data(), b_shuffled_scale.mData.data(), N, K / ScaleBlockSize);
    }
    else
    {
        throw std::runtime_error("wrong! unsupported warp size");
    }

    if constexpr(BPreShuffle)
    {
        int NPerXdl = 16; // Fixed 16
        preShuffleBuffer(b_k_n->mData.data(), b_input->mData.data(), N, K, NPerXdl, KPack);
    }

    if(config.verbosity > 0)
        std::cout << "Device memory allocation..." << std::endl;
    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.GetElementSpaceSize());
    DeviceMem a_scale_device_buf(sizeof(XDataType) * a_m_k_scale.GetElementSpaceSize());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n->GetElementSpaceSize());
    DeviceMem b_scale_device_buf(sizeof(XDataType) * b_k_n_scale.GetElementSpaceSize());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.GetElementSpaceSize());

    if(config.verbosity > 0)
        std::cout << "Upload data to device..." << std::endl;
    a_device_buf.ToDevice(a_m_k.mData.data());
    a_scale_device_buf.ToDevice(a_shuffled_scale.mData.data());
    b_device_buf.ToDevice(b_input->mData.data());
    b_scale_device_buf.ToDevice(b_shuffled_scale.mData.data());

    if(config.verbosity > 0)
        std::cout << "Done." << std::endl;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto c_element_op = CElementOp{};

    bool pass = true;
    if((config.do_verification == 1) || (config.do_verification == 3))
    {
        using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceMXGemm<ADataType,
                                                                                  BDataType,
                                                                                  CDataType,
                                                                                  AccDataType,
                                                                                  XDataType,
                                                                                  PassThrough,
                                                                                  PassThrough,
                                                                                  PassThrough,
                                                                                  float,
                                                                                  float>;

        auto ref_gemm    = ReferenceGemmInstance{};
        auto ref_invoker = ref_gemm.MakeInvoker();

        auto ref_argument = ref_gemm.MakeArgument(a_m_k,
                                                  a_m_k_scale,
                                                  *b_k_n,
                                                  b_k_n_scale,
                                                  c_m_n_host_result,
                                                  PassThrough{},
                                                  PassThrough{},
                                                  PassThrough{});

        ref_invoker.Run(ref_argument);
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

        auto argument_ptr =
            gemm->MakeArgumentPointer(static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
                                      a_scale_device_buf.GetDeviceBuffer(),
                                      static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
                                      b_scale_device_buf.GetDeviceBuffer(),
                                      static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      Scale_Stride_AM,
                                      StrideB,
                                      Scale_Stride_BN,
                                      StrideC,
                                      KBatch,
                                      a_element_op,
                                      b_element_op,
                                      c_element_op);

        if(config.instance_index != -1 && config.instance_index != static_cast<int>(i))
        {
            continue;
        }

        if(!gemm->IsSupportedArgument(argument_ptr.get()))
        {
            std::cerr << gemm->GetTypeString() << " does not support this problem" << std::endl;
            continue;
        }

        if(config.do_verification)
        {
            ave_time = invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false, 1});

            c_device_buf.FromDevice(c_m_n_device_result.mData.data());

            bool ret = ck::utils::check_err(c_m_n_device_result,
                                            c_m_n_host_result,
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

            // Output size(M*N) * [dot product(2K) + product of scales(K/ScaleBlockSize) + scaling
            // of partial sums(K/ScaleBlockSize)] FLOPS = 2 * M * N * K + 2 * M * N * K /
            // ScaleBlockSize
            std::size_t flop =
                std::size_t(2) * M * N * K + std::size_t(2) * M * N * K / ScaleBlockSize;
            std::size_t num_btype = sizeof(ADataType) * M * K / ck::packed_size_v<ADataType> +
                                    sizeof(BDataType) * K * N / ck::packed_size_v<BDataType> +
                                    sizeof(CDataType) * M * N +
                                    sizeof(XDataType) * M * K / ScaleBlockSize +
                                    sizeof(XDataType) * N * K / ScaleBlockSize;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = static_cast<float>(num_btype) / 1e6f / ave_time;

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
bool run_mx_gemm_splitk_example(GemmInstance& op_ptrs, int argc, char* argv[])
{
    ProblemSizeSplitK problem_size;
    ExecutionConfig config;

    return !parse_cmd_args(argc, argv, problem_size, config) ||
           run_mx_gemm<BPreshuffle>(op_ptrs, problem_size, config);
}
