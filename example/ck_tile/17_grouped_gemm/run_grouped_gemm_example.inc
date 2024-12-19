// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

template <typename ALayout, typename BLayout, typename CLayout>
float invoke_gemm(int n_warmup,
                  int n_repeat,
                  int group_count,
                  const std::vector<grouped_gemm_kargs>& args)
{

    ck_tile::DeviceMem gemm_workspace;
    gemm_workspace.Realloc(GetWorkspaceSize(args));

    float ave_time = grouped_gemm<ALayout, BLayout, CLayout>(
        args,
        ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat},
        gemm_workspace.GetDeviceBuffer());

    std::string op_name{"Grouped Gemm"};

    std::size_t flop = 0, num_btype = 0;
    for(int j = 0; j < group_count; ++j)
    {
        flop += std::size_t(2) * args[j].M * args[j].N * args[j].K;

        num_btype += sizeof(ADataType) * args[j].M * args[j].K +
                     sizeof(BDataType) * args[j].K * args[j].N +
                     sizeof(CDataType) * args[j].M * args[j].N;
    }

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, "
              << gb_per_sec << " GB/s, " << op_name << std::endl;

    return ave_time;
}

template <typename ALayout, typename BLayout, typename CLayout>
int run_grouped_gemm_example_with_layouts(int argc,
                                          char* argv[],
                                          const ALayout a_layout                  = ALayout{},
                                          const BLayout b_layout                  = BLayout{},
                                          [[maybe_unused]] const CLayout c_layout = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);

    if(!result)
    {
        return -1;
    };

    auto valid_input_data = [&](int group_count, const auto&... args) {
        return !(args.empty() || ...) && group_count == (args.size() == ...);
    };

    const int group_count = arg_parser.get_int("group_count");
    const int repeat      = arg_parser.get_int("repeat");
    const int warmup      = arg_parser.get_int("warmup");

    std::vector<ck_tile::index_t> Ms        = arg_parser.get_int_vec("Ms");
    std::vector<ck_tile::index_t> Ns        = arg_parser.get_int_vec("Ns");
    std::vector<ck_tile::index_t> Ks        = arg_parser.get_int_vec("Ks");
    std::vector<ck_tile::index_t> stride_As = arg_parser.get_int_vec("stride_As");
    std::vector<ck_tile::index_t> stride_Bs = arg_parser.get_int_vec("stride_Bs");
    std::vector<ck_tile::index_t> stride_Cs = arg_parser.get_int_vec("stride_Cs");

    if(!valid_input_data(group_count, Ms, Ns, Ks, stride_As, stride_Bs, stride_Cs))
    {
        std::cout << "Please check the input data. Default values will be used." << std::endl;
        for(int i = 0; i < group_count; i++)
        {
            Ms.push_back(256 + 256 * i);
            Ns.push_back(128 + 128 * i);
            Ks.push_back(128 + 64 * i);

            stride_As.push_back(Ks[i]);
            stride_Bs.push_back(Ks[i]);
            stride_Cs.push_back(Ns[i]);
        }
    }

    std::vector<ck_tile::HostTensor<ADataType>> a_m_k_tensors;
    std::vector<ck_tile::HostTensor<BDataType>> b_k_n_tensors;
    std::vector<ck_tile::HostTensor<CDataType>> c_m_n_tensors;

    a_m_k_tensors.reserve(group_count);
    b_k_n_tensors.reserve(group_count);
    c_m_n_tensors.reserve(group_count);

    std::vector<std::unique_ptr<ck_tile::DeviceMem>> a_m_k_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> b_k_n_dev_buf;
    std::vector<std::unique_ptr<ck_tile::DeviceMem>> c_m_n_dev_buf;

    a_m_k_dev_buf.reserve(group_count);
    b_k_n_dev_buf.reserve(group_count);
    c_m_n_dev_buf.reserve(group_count);

    std::vector<grouped_gemm_kargs> gemm_descs;
    gemm_descs.reserve(group_count);

    for(int i = 0; i < group_count; ++i)
    {
        const ck_tile::index_t M = Ms[i];
        const ck_tile::index_t N = Ns[i];
        const ck_tile::index_t K = Ks[i];

        stride_As[i] = f_get_default_stride(M, N, stride_As[i], a_layout);
        stride_Bs[i] = f_get_default_stride(K, N, stride_Bs[i], b_layout);
        stride_Cs[i] = f_get_default_stride(M, N, stride_Cs[i], CLayout{});

        a_m_k_tensors.push_back(
            ck_tile::HostTensor<ADataType>(f_host_tensor_descriptor(M, K, stride_As[i], a_layout)));
        b_k_n_tensors.push_back(
            ck_tile::HostTensor<BDataType>(f_host_tensor_descriptor(K, N, stride_Bs[i], b_layout)));
        c_m_n_tensors.push_back(ck_tile::HostTensor<CDataType>(
            f_host_tensor_descriptor(M, N, stride_Cs[i], CLayout{})));

        std::cout << "gemm[" << i << "]"
                  << " a_m_k: " << a_m_k_tensors[i].mDesc << " b_k_n: " << b_k_n_tensors[i].mDesc
                  << " c_m_n: " << c_m_n_tensors[i].mDesc << std::endl;

        ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k_tensors[i]);
        ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n_tensors[i]);

        a_m_k_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            a_m_k_tensors[i].get_element_space_size_in_bytes()));
        b_k_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            b_k_n_tensors[i].get_element_space_size_in_bytes()));
        c_m_n_dev_buf.push_back(std::make_unique<ck_tile::DeviceMem>(
            c_m_n_tensors[i].get_element_space_size_in_bytes()));

        a_m_k_dev_buf[i]->ToDevice(a_m_k_tensors[i].data());
        b_k_n_dev_buf[i]->ToDevice(b_k_n_tensors[i].data());
        c_m_n_dev_buf[i]->SetZero();
        c_m_n_tensors[i].SetZero();

        const void* p_a = a_m_k_dev_buf[i]->GetDeviceBuffer();
        const void* p_b = b_k_n_dev_buf[i]->GetDeviceBuffer();
        void* p_c       = c_m_n_dev_buf[i]->GetDeviceBuffer();

        gemm_descs.push_back({p_a, p_b, p_c, M, N, K, stride_As[i], stride_Bs[i], stride_Cs[i]});
    }

    invoke_gemm<ALayout, BLayout, CLayout>(warmup, repeat, group_count, gemm_descs);

    for(int i = 0; i < group_count; i++)
    {
        c_m_n_dev_buf[i]->FromDevice(c_m_n_tensors[i].data());
    }

    bool pass{true};
    if(arg_parser.get_int("validate"))
    {
        for(int i = 0; i < group_count; ++i)
        {
            ck_tile::HostTensor<CDataType> c_m_n_host_ref(
                f_host_tensor_descriptor(Ms[i], Ns[i], stride_Cs[i], CLayout{}));
            c_m_n_host_ref.SetZero();
            ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(
                a_m_k_tensors[i], b_k_n_tensors[i], c_m_n_host_ref);
            pass &= ck_tile::check_err(c_m_n_tensors[i], c_m_n_host_ref);
        }
        std::cout << "The CPU veification result is:" << (pass ? "correct" : "fail") << std::endl;
    }

    return pass;
}

int run_grouped_gemm_example(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }

    const std::string a_layout = arg_parser.get_str("a_layout");
    const std::string b_layout = arg_parser.get_str("b_layout");

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        return run_grouped_gemm_example_with_layouts(argc, argv, Row{}, Col{}, Row{});
    }
    else if(a_layout == "R" && b_layout == "R")
    {
        return run_grouped_gemm_example_with_layouts(argc, argv, Row{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}
