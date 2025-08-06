// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <string>
#include <iostream>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

int main()
{
    // TODO: GemmTypeConfig
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using CDataType   = ck_tile::half_t;
    using AccDataType = ck_tile::half_t;

    // ArgParser
    ck_tile::index_t M            = 1024;
    ck_tile::index_t N            = 1024;
    ck_tile::index_t K            = 1024;
    ck_tile::index_t verification = 0;

    ck_tile::index_t stride_a = K;
    ck_tile::index_t stride_b = K;
    ck_tile::index_t stride_c = N;

    auto a_lengths = std::array<ck_tile::index_t, 2>{M, K};
    auto b_lengths = std::array<ck_tile::index_t, 2>{K, N};
    auto c_lengths = std::array<ck_tile::index_t, 2>{M, N};

    auto a_strides = std::array<ck_tile::index_t, 2>{stride_a, 1};
    auto b_strides = std::array<ck_tile::index_t, 2>{stride_b, 1};
    auto c_strides = std::array<ck_tile::index_t, 2>{stride_c, 1};

    // tensors on host (cpu)
    ck_tile::HostTensor<ADataType> a_host(a_lengths, a_strides);
    ck_tile::HostTensor<BDataType> b_host(b_lengths, b_strides);
    ck_tile::HostTensor<CDataType> c_host(c_lengths, c_strides);

    // initialize tensors
    ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5.f, 5.f}(a_host);
    ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5.f, 5.f}(b_host);

    // Print the tensors using the new print_first_n member function
    std::cout << "Tensor A (first 5 elements): ";
    a_host.print_first_n(5);
    std::cout << std::endl;

    std::cout << "Tensor B (first 5 elements): ";
    b_host.print_first_n(5);
    std::cout << std::endl;

    std::cout << "Tensor C (first 5 elements): ";
    c_host.print_first_n(5);
    std::cout << std::endl;

    // Create device tensors of same size as host tensors and copy data
    ck_tile::DeviceMem a_device(a_host);
    ck_tile::DeviceMem b_device(b_host);
    ck_tile::DeviceMem c_device(c_host);

    (void)verification;
    (void)AccDataType{}; // Fake usage to suppress unused warning

    return 0;
}
