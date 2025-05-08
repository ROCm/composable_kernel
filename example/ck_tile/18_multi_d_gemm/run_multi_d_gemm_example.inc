// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <cstddef>

template <typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename CDEElementWise = ck_tile::element_wise::PassThrough>
float invoke_multi_d_gemm(const void* a_m_k_dev_buf,
                          const void* b_k_n_dev_buf,
                          const std::array<const void*, DsDataType::size()>& ds_m_n_dev_buf,
                          void* c_m_n_dev_buf,
                          ck_tile::index_t M,
                          ck_tile::index_t N,
                          ck_tile::index_t K,
                          ck_tile::index_t StrideA,
                          ck_tile::index_t StrideB,
                          const std::array<ck_tile::index_t, DsDataType::size()>& StrideDs,
                          ck_tile::index_t StrideC,
                          int n_warmup,
                          int n_repeat)
{
    multiple_d_gemm_kargs gemm_descs({a_m_k_dev_buf,
                                      b_k_n_dev_buf,
                                      ds_m_n_dev_buf,
                                      c_m_n_dev_buf,
                                      /*kbatch */ 1,
                                      M,
                                      N,
                                      K,
                                      StrideA,
                                      StrideB,
                                      StrideDs,
                                      StrideC});

    float ave_time = multiple_d_gemm<ADataType,
                                     BDataType,
                                     DsDataType,
                                     AccDataType,
                                     CDataType,
                                     ALayout,
                                     BLayout,
                                     DsLayout,
                                     CLayout,
                                     CDEElementWise>(
        gemm_descs, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    std::string op_name{"Multiple-D Gemm"};
    static constexpr ck_tile::index_t NumDTensor = DsDataType::size();

    std::size_t flop = 0, num_btype = 0;

    flop += std::size_t(2) * M * N * K;

    ck_tile::static_for<0, NumDTensor, 1>{}([&](auto i) {
        num_btype += sizeof(ck_tile::remove_cvref_t<std::tuple_element_t<i, DsDataType>>) * M * N;
        flop += sizeof(ck_tile::remove_cvref_t<std::tuple_element_t<i, DsDataType>>) * M * N;
    });

    num_btype += sizeof(ADataType) * M * K + sizeof(BDataType) * K * N + sizeof(CDataType) * M * N;

    float tflops     = static_cast<float>(flop) / 1.E9 / ave_time;
    float gb_per_sec = num_btype / 1.E6 / ave_time;

    std::cout << "Run Multiple-D Gemm kernel with:\n";
    std::cout << "M =" << M << " N =" << N << " K =" << K << "\n";
    std::cout << "StrideA = " << StrideA << " StrideB = " << StrideB << " StrideC = " << StrideC
              << "\n";
    std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec << " GB/s, "
              << "\n";

    return ave_time;
}

template <typename ALayout,
          typename BLayout,
          typename D0Layout,
          typename D1Layout,
          typename CLayout>
int run_multiple_d_gemm_example_with_layouts(int argc,
                                             char* argv[],
                                             const ALayout a_layout   = ALayout{},
                                             const BLayout b_layout   = BLayout{},
                                             const D0Layout d0_layout = D0Layout{},
                                             const D1Layout d1_layout = D1Layout{},
                                             const CLayout c_layout   = CLayout{})
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
    {
        return -1;
    }
    using CDElementWiseFn = ck_tile::element_wise::ElementWiseAdd;
    using DsLayout        = ck_tile::tuple<D0Layout, D1Layout>;

    ck_tile::index_t M = arg_parser.get_int("m");
    ck_tile::index_t N = arg_parser.get_int("n");
    ck_tile::index_t K = arg_parser.get_int("k");

    ck_tile::index_t StrideA = arg_parser.get_int("stride_a");
    ck_tile::index_t StrideB = arg_parser.get_int("stride_b");
    ck_tile::index_t StrideD = arg_parser.get_int("stride_d");
    ck_tile::index_t StrideC = arg_parser.get_int("stride_c");

    ck_tile::index_t StrideD0 = StrideD;
    ck_tile::index_t StrideD1 = StrideD;

    const int n_warmup = arg_parser.get_int("warmup");
    const int n_repeat = arg_parser.get_int("repeat");

    StrideA  = f_get_default_stride(M, N, StrideA, a_layout);
    StrideB  = f_get_default_stride(K, N, StrideB, b_layout);
    StrideD0 = f_get_default_stride(M, N, StrideD, d0_layout);
    StrideD1 = f_get_default_stride(M, N, StrideD, d1_layout);
    StrideC  = f_get_default_stride(M, N, StrideC, c_layout);

    ck_tile::HostTensor<ADataType> a_m_k_tesnor(f_host_tensor_descriptor(M, K, StrideA, a_layout));
    ck_tile::HostTensor<BDataType> b_k_n_tensors(f_host_tensor_descriptor(K, N, StrideB, b_layout));
    ck_tile::HostTensor<D0DataType> d0_m_n_tensors(
        f_host_tensor_descriptor(M, N, StrideD0, d0_layout));
    ck_tile::HostTensor<D1DataType> d1_m_n_tensors(
        f_host_tensor_descriptor(M, N, StrideD1, d1_layout));
    ck_tile::HostTensor<CDataType> c_m_n_device_result(
        f_host_tensor_descriptor(M, N, StrideC, c_layout));

    ck_tile::FillUniformDistribution<ADataType>{-5.f, 5.f}(a_m_k_tesnor);
    ck_tile::FillUniformDistribution<BDataType>{-5.f, 5.f}(b_k_n_tensors);
    ck_tile::FillUniformDistribution<D0DataType>{-1.f, 1.f}(d0_m_n_tensors);
    ck_tile::FillUniformDistribution<D1DataType>{-1.f, 1.f}(d1_m_n_tensors);

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k_tesnor.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n_tensors.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d0_m_n_dev_buf(d0_m_n_tensors.get_element_space_size_in_bytes());
    ck_tile::DeviceMem d1_m_n_dev_buf(d1_m_n_tensors.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_device_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k_tesnor.mData.data());
    b_k_n_dev_buf.ToDevice(b_k_n_tensors.mData.data());
    d0_m_n_dev_buf.ToDevice(d0_m_n_tensors.mData.data());
    d1_m_n_dev_buf.ToDevice(d1_m_n_tensors.mData.data());

    c_m_n_dev_buf.SetZero();
    c_m_n_device_result.SetZero();

    std::array<const void*, DsDataType::size()> ds_ptr_buf = {d0_m_n_dev_buf.GetDeviceBuffer(),
                                                              d1_m_n_dev_buf.GetDeviceBuffer()};

    std::array<ck_tile::index_t, DsDataType::size()> stridesDs = {StrideD0, StrideD1};

    invoke_multi_d_gemm<ADataType,
                        BDataType,
                        DsDataType,
                        AccDataType,
                        CDataType,
                        ALayout,
                        BLayout,
                        DsLayout,
                        CLayout,
                        CDElementWiseFn>(a_m_k_dev_buf.GetDeviceBuffer(),
                                         b_k_n_dev_buf.GetDeviceBuffer(),
                                         ds_ptr_buf,
                                         c_m_n_dev_buf.GetDeviceBuffer(),
                                         M,
                                         N,
                                         K,
                                         StrideA,
                                         StrideB,
                                         stridesDs,
                                         StrideC,
                                         n_warmup,
                                         n_repeat);

    c_m_n_dev_buf.FromDevice(c_m_n_device_result.data());

    ck_tile::HostTensor<CDataType> c_m_n_host_ref(
        f_host_tensor_descriptor(M, N, StrideC, c_layout));
    c_m_n_host_ref.SetZero();

    ck_tile::reference_gemm_multiple_d<
        ADataType,
        BDataType,
        std::tuple<ck_tile::HostTensor<D0DataType>, ck_tile::HostTensor<D1DataType>>,
        AccDataType,
        CDataType,
        CDElementWiseFn>(
        a_m_k_tesnor, b_k_n_tensors, std::tie(d0_m_n_tensors, d1_m_n_tensors), c_m_n_host_ref);

    bool pass{true};
    if(arg_parser.get_int("v"))
    {
        const float max_accumulated_value =
            *std::max_element(c_m_n_host_ref.mData.begin(), c_m_n_host_ref.mData.end());
        (void)(max_accumulated_value);
        const auto rtol_atol = calculate_rtol_atol(K, 1, max_accumulated_value);

        pass &= ck_tile::check_err(c_m_n_device_result,
                                   c_m_n_host_ref,
                                   "Error: Incorrect results!",
                                   rtol_atol.at(ck_tile::number<0>{}),
                                   rtol_atol.at(ck_tile::number<1>{}));

        std::cout << "Relative error threshold: " << rtol_atol.at(ck_tile::number<0>{})
                  << std::endl;
        std::cout << "Absolute error threshold: " << rtol_atol.at(ck_tile::number<1>{})
                  << std::endl;
        std::cout << "The CPU veification result is: " << (pass ? "correct" : "fail") << std::endl;
    }
    return pass;
}

int run_multiple_d_gemm_example(int argc, char* argv[])
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

    using Row = ck_tile::tensor_layout::gemm::RowMajor;
    using Col = ck_tile::tensor_layout::gemm::ColumnMajor;

    if(a_layout == "R" && b_layout == "C")
    {
        return run_multiple_d_gemm_example_with_layouts(
            argc, argv, Row{}, Col{}, Row{}, Row{}, Row{});
    }
    else
    {
        throw std::runtime_error("Unsupported data layout configuration for A,B and C tensors!");
    }
}
