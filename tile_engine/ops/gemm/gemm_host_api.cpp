// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "gemm_common.hpp"
#include "gemm_dispatcher.hpp"
#include "gemm_host_api.hpp"

void gemm_kernel_launch(ck_tile::DeviceMem& c_m_n_dev_buf,
                        ck_tile::HostTensor<CDataType>& c_m_n_host_result,
                        ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                        int verify,
                        bool structured_sparsity,
                        KernelTraits& trait,
                        ck_tile::GemmHostArgs& args,
                        const ck_tile::stream_config& stream)
{
    return GemmDispatcher::dispatch(c_m_n_dev_buf,
                                    c_m_n_host_result,
                                    c_m_n_dev_result,
                                    verify,
                                    structured_sparsity,
                                    trait,
                                    args,
                                    stream);
}

template <typename ADataType,
          typename BDataType,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
void run(const ck_tile::ArgParser& arg_parser)
{
    const ALayout a_layout = ALayout{};
    const BLayout b_layout = BLayout{};

    ck_tile::index_t kbatch = arg_parser.get_int("split_k");
    ck_tile::index_t M      = arg_parser.get_int("m");
    ck_tile::index_t N      = arg_parser.get_int("n");
    ck_tile::index_t K      = arg_parser.get_int("k");

    ck_tile::index_t stride_A = arg_parser.get_int("stride_a");
    ck_tile::index_t stride_B = arg_parser.get_int("stride_b");
    ck_tile::index_t stride_C = arg_parser.get_int("stride_c");

    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    int verify                   = arg_parser.get_int("v");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    bool structured_sparsity     = arg_parser.get_bool("structured_sparsity");

    stride_A = ck_tile::get_default_stride(M, K, stride_A, is_row_major(a_layout));
    stride_B = ck_tile::get_default_stride(K, N, stride_B, is_row_major(b_layout));
    stride_C = ck_tile::get_default_stride(M, N, stride_C, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    if(init_method == 0)
    {
        ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
        ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
    }
    else if(init_method == 1)
    {
        ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
        ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
    }
    else if(init_method == 2)
    {
        ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_m_k);
        ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_k_n);
    }
    else
    {
        a_m_k.SetZero();
        b_k_n.SetZero();
    }

    if(structured_sparsity)
    {
        ck_tile::AdjustToStructuredSparsity<ADataType>{}(a_m_k);
    }

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    if constexpr(std::is_same_v<BDataType, ck_tile::pk_int4_t>)
    {
        // Permute vector pk_i4x4 data for device implementation
        ck_tile::HostTensor<BDataType> b_k_n_dev = b_k_n;
        // permute_tensor_b<decltype(b_k_n_dev)>(b_k_n_dev);
        permute_vectors_i4x4_b(b_k_n_dev);
        b_k_n_dev_buf.ToDevice(b_k_n_dev.data());
    }
    else
    {
        b_k_n_dev_buf.ToDevice(b_k_n.data());
    }

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    ck_tile::GemmHostArgs gemm_args;
    gemm_args.a_ptr    = a_m_k_dev_buf.GetDeviceBuffer();
    gemm_args.b_ptr    = b_k_n_dev_buf.GetDeviceBuffer();
    gemm_args.c_ptr    = c_m_n_dev_buf.GetDeviceBuffer();
    gemm_args.k_batch  = kbatch;
    gemm_args.M        = M;
    gemm_args.N        = N;
    gemm_args.K        = K;
    gemm_args.stride_A = stride_A;
    gemm_args.stride_B = stride_B;
    gemm_args.stride_C = stride_C;

    KernelTraits trait;
    trait.pipeline  = arg_parser.get_str("pipeline");
    trait.scheduler = arg_parser.get_str("scheduler");
    trait.epilogue  = arg_parser.get_str("epilogue");
    trait.kPadM     = arg_parser.get_bool("pad_m");
    trait.kPadN     = arg_parser.get_bool("pad_n");
    trait.kPadK     = arg_parser.get_bool("pad_k");

    std::cout << "Run Gemm kernel with M =" << M << " N =" << N << " K =" << K
              << " StrideA =" << stride_A << " StrideB =" << stride_B << " StrideC =" << stride_C
              << " A_Layout =" << ALayout::name << " B_Layout =" << BLayout::name
              << " C_Layout =" << CLayout::name << " A Type = " << DataTypeTraits<ADataType>::name
              << " B Type = " << DataTypeTraits<BDataType>::name
              << " C Type = " << DataTypeTraits<CDataType>::name << std::endl;

    ck_tile::HostTensor<CDataType> c_m_n_host_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    if(verify)
    {
        gemm_host_reference<ADataType,
                            BDataType,
                            AccDataType,
                            CDataType,
                            ALayout,
                            BLayout,
                            CLayout>(verify,
                                     a_m_k,
                                     b_k_n,
                                     c_m_n_host_result,
                                     a_m_k_dev_buf,
                                     b_k_n_dev_buf,
                                     M,
                                     N,
                                     K,
                                     stride_A,
                                     stride_B,
                                     stride_C);
    }

    gemm_kernel_launch(c_m_n_dev_buf,
                       c_m_n_host_result,
                       c_m_n_dev_result,
                       verify,
                       structured_sparsity,
                       trait,
                       gemm_args,
                       ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});

    return;
}

int main(int argc, char* argv[])
{
    try
    {
        auto [result, parser] = create_args(argc, argv);
        if(!result)
            return EXIT_FAILURE;
        run<ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout>(parser);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return EXIT_FAILURE;
    }
}
