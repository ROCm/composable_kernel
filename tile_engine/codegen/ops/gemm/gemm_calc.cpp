// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include "gemm_host.hpp"
#include "tensor_configuration.hpp"

template <typename ADataType_, typename BDataType_, typename AccDataType_, typename CDataType_>
bool run(const ck_tile::ArgParser& arg_parser)
{
    ck_tile::index_t kbatch      = arg_parser.get_int("split_k");
    int n_warmup                 = arg_parser.get_int("warmup");
    int n_repeat                 = arg_parser.get_int("repeat");
    ck_tile::index_t init_method = arg_parser.get_int("init");
    ck_tile::HostTensor<ADataType> a_m_k(
        ck_tile::host_tensor_descriptor(M, K, stride_A, is_row_major(a_layout)));
    ck_tile::HostTensor<BDataType> b_k_n(
        ck_tile::host_tensor_descriptor(K, N, stride_B, is_row_major(b_layout)));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(
        ck_tile::host_tensor_descriptor(M, N, stride_C, is_row_major(CLayout{})));

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());
    init_host_tensor(
        arg_parser, a_m_k, b_k_n, c_m_n_dev_result, a_m_k_dev_buf, b_k_n_dev_buf, c_m_n_dev_buf);
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;
    run<ADataType, BDataType, CDatatType>(const ck_tile::ArgParser& arg_parser);
    return 0;
}