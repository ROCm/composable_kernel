// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <cstring>
#include <iostream>
#include <ostream>
#include <string>
#include <tuple>

#include "ck_tile/host.hpp"
#include "kernel.hpp"

template <typename DataType>
float load_store_tile(const ck_tile::AsyncLSKernelArgs& args, const ck_tile::stream_config& s)

{
    constexpr int kBlockPerCu = 1;

    constexpr ck_tile::index_t M_Tile = 32;
    constexpr ck_tile::index_t N_Tile = 256;

    constexpr ck_tile::index_t M_Warp = 2;
    constexpr ck_tile::index_t N_Warp = 1;

    using Shape = ck_tile::TileShape<M_Tile, N_Tile, M_Warp, N_Warp>;

    using Policy = ck_tile::AsyncLSPolicy<DataType, Shape>;

    using Kernel = ck_tile::AsyncLSKernel<Policy>;

    const dim3 grids  = Kernel::GridSize(args.M, args.N);
    const dim3 blocks = Kernel::BlockSize();

    float ave_time = ck_tile::launch_kernel(
        s, ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, args));

    return ave_time;
}

template <typename DataType>
float invoke_load_store_tile(ck_tile::DeviceMem& a_dev_buf,
                             ck_tile::DeviceMem& b_dev_buf,
                             ck_tile::index_t M,
                             ck_tile::index_t N)
{
    auto args = ck_tile::AsyncLSKernelArgs{
        a_dev_buf.GetDeviceBuffer(), b_dev_buf.GetDeviceBuffer(), M, N, N, N};
    auto sc        = ck_tile::stream_config{nullptr, true, 1, 0, 1, true, true, 1};
    float ave_time = load_store_tile<DataType>(args, sc);

    std::cout << "Run Load_Store_Tile kernel with M=" << M << " N=" << N << " : " << ave_time
              << " ms, \n";

    return ave_time;
}

template <typename DataType>
bool run_load_store_tile()
{
    constexpr size_t m = 32;
    constexpr size_t n = 256;
    constexpr size_t s = 1;

    ck_tile::HostTensor<DataType> a_m_n({m, n}, {n, s});
    ck_tile::HostTensor<DataType> b_m_n({m, n}, {n, s});

    ck_tile::FillMonotonicSeq<DataType>{}(a_m_n);
    b_m_n.SetZero();

    ck_tile::DeviceMem a_m_n_dev_buf(a_m_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_m_n_dev_buf(b_m_n.get_element_space_size_in_bytes());
    a_m_n_dev_buf.ToDevice(a_m_n.data());

    invoke_load_store_tile<DataType>(a_m_n_dev_buf, b_m_n_dev_buf, m, n);

    b_m_n_dev_buf.FromDevice(b_m_n.data());

    bool pass = ck_tile::check_err(b_m_n, a_m_n);

    return pass;
}
