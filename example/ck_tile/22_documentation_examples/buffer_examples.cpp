// SPDX-License-Identifier: MIT
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck_tile/host.hpp"
#include <cstring>
#include "ck_tile/ops/documentation_examples/kernel/example_buffer_kernel.hpp"

template <typename DataType>
bool run()
{
    using XDataType       = DataType;

    auto problem_shape = ck_tile::make_tuple(10);

    ck_tile::HostTensor<XDataType> x_host({10}, {1});

    for(int i=0; i<10; i++)
    {
        x_host.data()[i] = static_cast<XDataType>(i);
    }

    ck_tile::DeviceMem x_buf(x_host.get_element_space_size_in_bytes());

    x_buf.ToDevice(x_host.data());

    constexpr ck_tile::index_t kBlockSize  = 1;
    constexpr ck_tile::index_t kBlockPerCu = 1;    
    constexpr ck_tile::index_t kGridSize = 1;

    using Kernel = ck_tile::Docs<XDataType>;

    launch_kernel(ck_tile::stream_config{nullptr, true, 0, 0, 1},
                                   ck_tile::make_kernel<kBlockSize, kBlockPerCu>(
                                       Kernel{},
                                       1,
                                       kBlockSize,
                                       0,
                                       static_cast<XDataType*>(x_buf.GetDeviceBuffer()),
                                       problem_shape));

    return true;
}

int main()
{
    return run<int>() ? 0 : -2;
}
