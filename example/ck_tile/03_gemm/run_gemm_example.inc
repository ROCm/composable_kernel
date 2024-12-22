// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

template <typename ALayout, typename BLayout, typename CLayout>
float invoke_gemm(ck_tile::DeviceMem& a_m_k_dev_buf,
                  ck_tile::DeviceMem& b_k_n_dev_buf,
                  ck_tile::DeviceMem& c_m_n_dev_buf,
                  ck_tile::index_t M,
                  ck_tile::index_t N,
                  ck_tile::index_t K,
                  ck_tile::index_t stride_A,
                  ck_tile::index_t stride_B,
                  ck_tile::index_t stride_C,
                  ck_tile::index_t kbatch,
                  int n_warmup,
                  int n_repeat)
{
    ck_tile::GemmHostArgs args;
    args.a_ptr    = a_m_k_dev_buf.GetDeviceBuffer();
    args.b_ptr    = b_k_n_dev_buf.GetDeviceBuffer();
    args.c_ptr    = c_m_n_dev_buf.GetDeviceBuffer();
    args.k_batch  = kbatch;
    args.M        = M;
    args.N        = N;
    args.K        = K;
    args.stride_A = stride_A;
    args.stride_B = stride_B;
    args.stride_C = stride_C;

    float ave_time = gemm_calc<ALayout, BLayout, CLayout>(
        args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::size_t flop = std::size_t(2) * M * N * K;
    std::size_t num_byte =
        sizeof(ADataType) * M * K + sizeof(BDataType) * N * K + sizeof(CDataType) * M * N;
    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << "Run Gemm kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideB =" << stride_B << " StrideC =" << stride_C
              << " : " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << std::endl;

    return ave_time;
}

template <typename ALayout, typename BLayout, typename CLayout>
int run_gemm_example_with_layouts(int argc,
                                  char* argv[],
                                  const ALayout a_layout                  = ALayout{},
                                  const BLayout b_layout                  = BLayout{},
                                  [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    ck_tile::index_t kbatch = arg_parser.get_int("split_k");
    int n_warmup            = arg_parser.get_int("warmup");
    int n_repeat            = arg_parser.get_int("repeat");

    using namespace ck_tile::literals;

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return ck_tile::HostTensorDescriptor({row, col}, {stride, 1_uz});
            }
            else
            {
                return ck_tile::HostTensorDescriptor({row, col}, {1_uz, stride});
            }
        };

    auto f_get_default_stride = [](std::size_t row,
                                   std::size_t col,
                                   std::size_t stride,
                                   auto layout) {
        if(stride == 0)
        {
            // give a chance if stride is zero, return a default packed stride
            if constexpr(std::is_same_v<decltype(layout), ck_tile::tensor_layout::gemm::RowMajor>)
            {
                return col;
            }
            else
            {
                return row;
            }
        }
        else
            return stride;
    };

    stride_A = f_get_default_stride(M, K, stride_A, a_layout);
    stride_B = f_get_default_stride(K, N, stride_B, b_layout);
    stride_C = f_get_default_stride(M, N, stride_C, CLayout{});

    ck_tile::HostTensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, stride_A, a_layout));
    ck_tile::HostTensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, stride_B, b_layout));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        f_host_tensor_descriptor(M, N, stride_C, CLayout{}));

    // TODO: add different init types
    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n);

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    invoke_gemm<ALayout, BLayout, CLayout>(a_m_k_dev_buf,
                                           b_k_n_dev_buf,
                                           c_m_n_dev_buf,
                                           M,
                                           N,
                                           K,
                                           stride_A,
                                           stride_B,
                                           stride_C,
                                           kbatch,
                                           n_warmup,
                                           n_repeat);

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
    bool pass = true;

    if(arg_parser.get_int("v") == 1)
    {
        ck_tile::HostTensor<CDataType> c_m_n_host_ref(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));
        c_m_n_host_ref.SetZero();

        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
            a_m_k, b_k_n, c_m_n_host_ref);

        pass = ck_tile::check_err(c_m_n_dev_result, c_m_n_host_ref);

        std::cout << "The CPU veification result is:" << (pass ? "correct" : "fail") << std::endl;
    }
    else if(arg_parser.get_int("v") == 2)
    {
        ck_tile::HostTensor<CDataType> c_m_n_gpu_ref(
            f_host_tensor_descriptor(M, N, stride_C, CLayout{}));
        ck_tile::DeviceMem c_m_n_gpu_buf_ref(c_m_n_gpu_ref.get_element_space_size_in_bytes());
        c_m_n_gpu_ref.SetZero();
        c_m_n_gpu_buf_ref.SetZero();

        ADataType* d_A;
        BDataType* d_B;
        CDataType* d_C;

        ck_tile::hip_check_error(hipMalloc(&d_A, M * K * sizeof(ADataType)));
        ck_tile::hip_check_error(hipMalloc(&d_B, N * K * sizeof(BDataType)));
        ck_tile::hip_check_error(hipMalloc(&d_C, M * N * sizeof(CDataType)));

        ck_tile::hip_check_error(hipMemcpy(d_A,
                                           a_m_k_dev_buf.GetDeviceBuffer(),
                                           M * K * sizeof(ADataType),
                                           hipMemcpyHostToDevice));
        ck_tile::hip_check_error(hipMemcpy(d_B,
                                           b_k_n_dev_buf.GetDeviceBuffer(),
                                           N * K * sizeof(BDataType),
                                           hipMemcpyHostToDevice));

        ck_tile::reference_gemm_gpu<ADataType,
                                    BDataType,
                                    AccDataType,
                                    CDataType,
                                    ALayout,
                                    BLayout,
                                    CLayout>(d_A, d_B, d_C, M, N, K, stride_A, stride_B, stride_C);

        ck_tile::hip_check_error(hipMemcpy(c_m_n_gpu_buf_ref.GetDeviceBuffer(),
                                           d_C,
                                           M * N * sizeof(CDataType),
                                           hipMemcpyDeviceToHost));

        ck_tile::hip_check_error(hipFree(d_A));
        ck_tile::hip_check_error(hipFree(d_B));
        ck_tile::hip_check_error(hipFree(d_C));

        c_m_n_gpu_buf_ref.FromDevice(c_m_n_gpu_ref.data());
        pass = ck_tile::check_err(c_m_n_dev_result, c_m_n_gpu_ref);

        std::cout << "The GPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

int run_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    std::string a_layout = arg_parser.get_str("a_layout");
    std::string b_layout = arg_parser.get_str("b_layout");

    if(a_layout == "R" && b_layout == "R")
    {
        return run_gemm_example_with_layouts(argc, argv, Row{}, Row{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "C")
    {
        return run_gemm_example_with_layouts(argc, argv, Row{}, Col{}, Row{});
    }
    // TODO: Fixme: with latest changes to GemmPipelineAGmemBGmemCRegV1DefaultPolicy below do not
    // work.
    // else if(a_layout == "C" && b_layout == "C")
    // {
    //     return run_gemm_example_with_layouts(argc, argv, Col{}, Col{}, Row{});
    // }
    // else if(a_layout == "C" && b_layout == "R")
    // {
    //     return run_gemm_example_with_layouts(argc, argv, Col{}, Row{}, Row{});
    // }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}
